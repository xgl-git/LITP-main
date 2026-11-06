import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import copy
import math
import json
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

def _unwrap_model(model):
    """Return the underlying module for DataParallel-specific methods."""
    return model.module if isinstance(model, nn.DataParallel) else model


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Keep an exponential moving average of the model parameters.
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for shadow_param, param in zip(self.shadow.parameters(), model.parameters()):
            if param.requires_grad:
                shadow_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create a cosine learning-rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def mask_seen_items(scores, sequence, labels):
    """Exclude history items from full-ranking evaluation.

    MovieLens sequences can contain repeated interactions.  The held-out target
    must remain eligible even if it appeared earlier in the history.
    """
    target_scores = scores.gather(1, labels)
    scores = scores.clone()
    scores.scatter_(1, sequence, -float('inf'))
    scores[:, 0] = -float('inf')
    scores.scatter_(1, labels, target_scores)
    return scores


def _accumulate_metrics(metric_sums, metrics, batch_size):
    for key, value in metrics.items():
        metric_sums[key] += float(value) * batch_size


def _finalize_metrics(metric_sums, example_count):
    if example_count == 0:
        raise ValueError('Cannot compute metrics for an empty data loader.')
    return {
        key: round(value / example_count * 100, 4)
        for key, value in metric_sums.items()
    }


def evaluate_model(model, data_loader, args):
    """Evaluate one model with deterministic reverse-diffusion noise."""
    model.eval()
    metric_sums = {
        'HR@5': 0.0, 'NDCG@5': 0.0,
        'HR@10': 0.0, 'NDCG@10': 0.0,
        'HR@20': 0.0, 'NDCG@20': 0.0,
    }
    example_count = 0
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    with torch.no_grad():
        for sequence, labels, user_ids, plan_ids in data_loader:
            sequence = sequence.to(args.device)
            labels = labels.to(args.device)
            user_ids = user_ids.to(args.device)
            plan_ids = plan_ids.to(args.device)
            scores, rep_diffu, _, _, _, _, _, _, _, _ = model(
                sequence, labels, user_ids, plan_ids, train_flag=False
            )
            if scores is None:
                scores = model.diffu_rep_pre(rep_diffu)
            scores = mask_seen_items(scores, sequence, labels)
            metrics = hrs_and_ndcgs_k(scores, labels, args.metric_ks)
            batch_size = labels.shape[0]
            _accumulate_metrics(metric_sums, metrics, batch_size)
            example_count += batch_size
    return _finalize_metrics(metric_sums, example_count)


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        # Use the same fixed noise realization as validation for reproducible
        # comparison with the selected EMA checkpoint.
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]

            # Inference returns (scores, rep_diffu, None, None, None, None, None, None, None, None)
            scores_rec, rep_diffu, _, _, _, _, _, _, _, _ = model_joint(
                test_batch[0], test_batch[1], test_batch[2], test_batch[3],
                train_flag=False
            )
            if scores_rec is None:
                scores_rec = model_joint.diffu_rep_pre(rep_diffu)
            scores_rec = mask_seen_items(
                scores_rec, test_batch[0], test_batch[1]
            )
            metrics = hrs_and_ndcgs_k(scores_rec, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


def _item_info(internal_item_id, args):
    """Convert an internal item ID to readable MovieLens metadata."""
    internal_item_id = int(internal_item_id)
    return {
        'internal_item_id': internal_item_id,
        'original_movie_id': int(args.inverse_smap.get(internal_item_id, -1)),
        'title': args.item_titles[internal_item_id],
        'genres': args.item_genres[internal_item_id],
    }


def _jsonable_args(args):
    """Keep scalar/list CLI settings while excluding large feature matrices."""
    result = {}
    excluded = {
        'genre_matrix', 'user_gender', 'user_age', 'user_occupation',
        'item_titles', 'item_genres', 'inverse_smap', 'inverse_umap',
        'semantic_plan_assignments', 'semantic_plan_token_ids',
        'semantic_plan_profile_token_ids',
    }
    for key, value in vars(args).items():
        if key in excluded:
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
    return result


def save_predictions(model, data_loader, args, output_path, top_k):
    """Run test inference and write one readable JSON object per user."""
    device = args.device
    model.eval()
    records = []
    with torch.no_grad():
        for sequence, labels, user_ids, plan_ids in data_loader:
            sequence = sequence.to(device)
            labels = labels.to(device)
            user_ids = user_ids.to(device)
            plan_ids = plan_ids.to(device)
            scores, rep_diffu, _, _, _, _, _, _, _, _ = model(
                sequence, labels, user_ids, plan_ids, train_flag=False
            )
            if scores is None:
                scores = model.diffu_rep_pre(rep_diffu)
            scores = mask_seen_items(scores, sequence, labels)
            top_scores, top_items = torch.topk(
                scores, k=min(top_k, scores.shape[1] - 1), dim=-1
            )
            for row in range(labels.shape[0]):
                user_id = int(user_ids[row].item())
                target_id = int(labels[row].item())
                predictions = []
                for rank, (item_id, score) in enumerate(
                    zip(top_items[row].tolist(), top_scores[row].tolist()),
                    start=1,
                ):
                    item = _item_info(item_id, args)
                    item.update({'rank': rank, 'score': float(score)})
                    predictions.append(item)
                records.append({
                    'internal_user_id': user_id,
                    'original_user_id': int(args.inverse_umap.get(user_id, -1)),
                    'target': _item_info(target_id, args),
                    'hit_at_k': target_id in top_items[row].tolist(),
                    'predictions': predictions,
                })

    with output_path.open('w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    return records


def collect_diagnostics(model_joint, diffu_rep, weights, t, loss_diffu_value,
                        loss_rmse_dy, true_tag_emb, dy_pre_x_t, q_u, delta_u):
    """Collect scale and gradient diagnostics for the diffusion branch.

    The key quantity is ``item_norm``: the recommendation loss is scale
    invariant (it normalizes both sides), while ``L_diff`` is a raw RMSE against
    un-normalised item vectors.  If the item embedding norm grows over training,
    the diffusion target keeps inflating and ``diff`` rises even when the
    denoiser learns the correct direction.
    """
    module = model_joint.module if isinstance(model_joint, nn.DataParallel) else model_joint
    with torch.no_grad():
        item_weight = module.item_embeddings.weight
        item_norm_mean = item_weight.norm(p=2, dim=-1).mean()
        item_norm_max = item_weight.norm(p=2, dim=-1).max()
        # Padding row 0 is excluded from the statistics above only partly; it is
        # reported separately so it cannot hide growth of the real items.
        padding_norm = item_weight[0].norm(p=2)

        target_norm = true_tag_emb.norm(p=2, dim=-1).mean()
        pred_norm = dy_pre_x_t.norm(p=2, dim=-1).mean()
        latent_norm = diffu_rep.norm(p=2, dim=-1).mean()
        delta_norm = delta_u.norm(p=2, dim=-1).mean()

        cos_align = F.cosine_similarity(q_u, delta_u, dim=-1).mean()

        diagnostics = {
            'item_norm_mean': item_norm_mean.item(),
            'item_norm_max': item_norm_max.item(),
            'padding_norm': padding_norm.item(),
            'target_norm_mean': target_norm.item(),
            'pred_norm_mean': pred_norm.item(),
            'latent_norm_mean': latent_norm.item(),
            'delta_norm_mean': delta_norm.item(),
            'plan_target_cosine': cos_align.item(),
            't_mean': t.float().mean().item(),
            'weight_mean': weights.mean().item(),
            'rec_loss': loss_diffu_value.item(),
            'diff_loss': loss_rmse_dy.item(),
        }
        # Gradient norm of the diffusion reconstruction path, measured before
        # ``optimizer.step()`` clears the graph.
        grad_norm = 0.0
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            if name.startswith('litp.') or 'litp' in name:
                grad_norm += param.grad.detach().pow(2).sum().item()
        diagnostics['diffu_grad_norm'] = math.sqrt(grad_norm)
    return diagnostics


def _format_diagnostics(diagnostics):
    return (
        'item_norm: %.4f (max %.4f, pad %.4f) | target_norm: %.4f | '
        'pred_norm: %.4f | latent_norm: %.4f | delta_norm: %.4f | '
        'cos(q,Δ): %.4f | t_mean: %.2f | w_mean: %.4f | diffu_grad: %.4f'
        % (
            diagnostics['item_norm_mean'],
            diagnostics['item_norm_max'],
            diagnostics['padding_norm'],
            diagnostics['target_norm_mean'],
            diagnostics['pred_norm_mean'],
            diagnostics['latent_norm_mean'],
            diagnostics['delta_norm_mean'],
            diagnostics['plan_target_cosine'],
            diagnostics['t_mean'],
            diagnostics['weight_mean'],
            diagnostics['diffu_grad_norm'],
        )
    )


def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    device = args.device
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel and args.device != 'cuda':
        raise ValueError('Multi-GPU training requires --device cuda')
    if is_parallel and torch.cuda.device_count() < args.num_gpu:
        raise RuntimeError(
            'Requested {} GPUs, but only {} are visible'.format(
                args.num_gpu, torch.cuda.device_count()
            )
        )
    total_loss_lambda = args.total_loss_lambda
    align_loss_lambda = args.align_loss_lambda
    if is_parallel:
        device_ids = list(range(args.num_gpu))
        model_joint = nn.DataParallel(
            model_joint, device_ids=device_ids, output_device=device_ids[0]
        )

    optimizer = optimizers(model_joint, args)

    if args.dataset.lower() != 'steam':
        total_steps = epochs * len(tra_data_loader)
        warmup_steps = int(total_steps * 0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        ema = EMA(model_joint, decay=0.999)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
        ema = None

    # Always keep a valid fallback, even if validation metrics are all zero.
    best_model = copy.deepcopy(ema.shadow if ema is not None else model_joint)

    best_metrics_dict = {
        'Best_HR@5': 0,
        'Best_NDCG@5': 0,
        'Best_HR@10': 0,
        'Best_NDCG@10': 0,
        'Best_HR@20': 0,
        'Best_NDCG@20': 0,
    }
    best_epoch = {
        'Best_epoch_HR@5': 0,
        'Best_epoch_NDCG@5': 0,
        'Best_epoch_HR@10': 0,
        'Best_epoch_NDCG@10': 0,
        'Best_epoch_HR@20': 0,
        'Best_epoch_NDCG@20': 0,
    }
    best_primary_metric = -float('inf')
    bad_count = 0

    diagnostics_history = []
    item_norm_trend = []
    validation_history = []
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / 'best_model.pt'
    validation_path = output_dir / 'validation_history.json'

    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()

            # Training forward returns a 10-tuple (see LITPModel.forward):
            # (None, rep_diffu, weights, t, None, None, tag_emb, dy_pre_x_t, q_u, delta_u)
            (
                _,
                diffu_rep,
                weights,
                t,
                _,
                _,
                true_tag_emb,
                dy_pre_x_t,
                q_u,
                delta_u,
            ) = model_joint(
                train_batch[0],
                train_batch[1],
                train_batch[2],
                train_batch[3],
                train_flag=True,
            )

            loss_model = _unwrap_model(model_joint)
            # L_rec: next-item recommendation loss.
            loss_diffu_value = loss_model.loss_diffu_ce(
                diffu_rep, train_batch[1]
            )
            # L_diff: diffusion denoising reconstruction loss.
            loss_rmse_dy = loss_model.loss_rmse_dy(
                true_tag_emb, dy_pre_x_t
            )
            # L_align: plan-latent alignment loss (grounding of the semantic plan).
            loss_align_value = loss_model.loss_align(q_u, delta_u)

            # L = (1 - λ_total) · L_rec + λ_total · L_diff + λ_align · L_align
            # L_graph and L_progress from the proposal are intentionally dropped
            # (no IP graph / no watch progress in ml-1m).
            loss_all = (
                (1 - total_loss_lambda) * loss_diffu_value
                + total_loss_lambda * loss_rmse_dy
                + align_loss_lambda * loss_align_value
            )

            loss_all.backward()

            # Diagnostics must run after backward() (gradients exist) and before
            # optimizer.step()/zero_grad() clear them.
            diagnostics = collect_diagnostics(
                model_joint,
                diffu_rep,
                weights,
                t,
                loss_diffu_value,
                loss_rmse_dy,
                true_tag_emb,
                dy_pre_x_t,
                q_u,
                delta_u,
            )
            diagnostics_history.append(diagnostics)

            optimizer.step()

            if args.dataset.lower() != 'steam':
                lr_scheduler.step()
                ema.update(model_joint)

            log_interval = (
                args.diagnostic_interval
                if args.diagnostic_interval > 0
                else int(len(tra_data_loader) / 5 + 1)
            )
            if index_temp % log_interval == 0:
                print('[%d/%d] Loss: %.4f | rec: %.4f | diff: %.4f | align: %.4f' % (
                    index_temp, len(tra_data_loader),
                    loss_all.item(),
                    loss_diffu_value.item(),
                    loss_rmse_dy.item(),
                    loss_align_value.item(),
                ))
                print('    diag: ' + _format_diagnostics(diagnostics))
                logger.info('[%d/%d] Loss: %.4f | rec: %.4f | diff: %.4f | align: %.4f' % (
                    index_temp, len(tra_data_loader),
                    loss_all.item(),
                    loss_diffu_value.item(),
                    loss_rmse_dy.item(),
                    loss_align_value.item(),
                ))
                logger.info('diag: ' + _format_diagnostics(diagnostics))

        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))

        # Epoch-level trend of the item embedding norm.  A monotonically rising
        # norm together with a rising ``diff`` confirms the scale-mismatch
        # diagnosis (rec is scale invariant, L_diff is not).
        if diagnostics_history:
            epoch_diag = {
                key: float(np.mean([d[key] for d in diagnostics_history]))
                for key in diagnostics_history[0]
            }
            item_norm_trend.append(epoch_diag['item_norm_mean'])
            print(
                'epoch %d summary | item_norm: %.4f | target_norm: %.4f | '
                'pred_norm: %.4f | rec: %.4f | diff: %.4f | cos(q,Δ): %.4f'
                % (
                    epoch_temp,
                    epoch_diag['item_norm_mean'],
                    epoch_diag['target_norm_mean'],
                    epoch_diag['pred_norm_mean'],
                    epoch_diag['rec_loss'],
                    epoch_diag['diff_loss'],
                    epoch_diag['plan_target_cosine'],
                )
            )
            print(
                'epoch %d item_norm_trend: %s'
                % (epoch_temp, ' '.join('%.4f' % v for v in item_norm_trend))
            )
            logger.info('epoch %d summary: %s', epoch_temp, epoch_diag)
            diagnostics_history = []

        if args.dataset.lower() == 'steam':
            lr_scheduler.step()

        # Validate after every completed training epoch.  Keeping this
        # unconditional avoids silently skipping epoch 0 or the final epoch.
        if args.dataset.lower() != 'steam':
            eval_model = ema.shadow
        else:
            eval_model = model_joint

        epoch_metrics = evaluate_model(eval_model, val_data_loader, args)
        validation_record = {'epoch': epoch_temp + 1, **epoch_metrics}
        validation_history.append(validation_record)
        validation_line = (
            'Validation epoch %d | HR@5: %.4f | HR@10: %.4f | HR@20: %.4f | '
            'NDCG@5: %.4f | NDCG@10: %.4f | NDCG@20: %.4f'
            % (
                epoch_temp + 1,
                epoch_metrics['HR@5'],
                epoch_metrics['HR@10'],
                epoch_metrics['HR@20'],
                epoch_metrics['NDCG@5'],
                epoch_metrics['NDCG@10'],
                epoch_metrics['NDCG@20'],
            )
        )
        print(validation_line, flush=True)

        # Persist the complete validation curve after every epoch so an
        # interrupted server run still retains all completed results.
        with validation_path.open('w', encoding='utf-8') as file:
            json.dump(validation_history, file, ensure_ascii=False, indent=2)

        # Select one coherent checkpoint by NDCG@20.  All six metrics recorded
        # below therefore belong to exactly the same epoch/model.
        primary_metric = epoch_metrics['NDCG@20']
        if primary_metric > best_primary_metric:
            best_primary_metric = primary_metric
            best_metrics_dict = {
                'Best_' + key: value for key, value in epoch_metrics.items()
            }
            best_epoch = {
                'Best_epoch_' + key: epoch_temp + 1 for key in epoch_metrics
            }
            bad_count = 0
            best_model = copy.deepcopy(eval_model).eval()
            torch.save(
                {
                    'model_state_dict': best_model.state_dict(),
                    'best_validation_metrics': best_metrics_dict,
                    'best_epoch': best_epoch,
                    'selection_metric': 'NDCG@20',
                    'args': _jsonable_args(args),
                },
                best_checkpoint_path,
            )
            print(
                'Saved best model at epoch %d (NDCG@20=%.4f)'
                % (epoch_temp + 1, primary_metric),
                flush=True,
            )
        else:
            bad_count += 1

        if bad_count >= args.patience:
            print(
                'Early stopping after %d validation epochs without improvement.'
                % bad_count,
                flush=True,
            )
            break


    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    test_metrics_dict_mean = evaluate_model(
        best_model, test_data_loader, args
    )
    prediction_path = output_dir / 'test_predictions.jsonl'
    # Reset the diffusion noise so exported predictions are exactly the same
    # stochastic samples used for the reported test metrics.
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    prediction_records = save_predictions(
        best_model,
        test_data_loader,
        args,
        prediction_path,
        args.prediction_topk,
    )

    checkpoint_path = best_checkpoint_path
    torch.save(
        {
            'model_state_dict': best_model.state_dict(),
            'test_metrics': test_metrics_dict_mean,
            'best_validation_metrics': best_metrics_dict,
            'best_epoch': best_epoch,
            'args': _jsonable_args(args),
        },
        checkpoint_path,
    )
    metrics_path = output_dir / 'metrics.json'
    with metrics_path.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'test': test_metrics_dict_mean,
                'best_validation': best_metrics_dict,
                'best_epoch': best_epoch,
                'validation_history': validation_history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / 'validation_history.json').open('w', encoding='utf-8') as file:
        json.dump(validation_history, file, ensure_ascii=False, indent=2)
    with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as file:
        json.dump(test_metrics_dict_mean, file, ensure_ascii=False, indent=2)
    with (output_dir / 'args.json').open('w', encoding='utf-8') as file:
        json.dump(_jsonable_args(args), file, ensure_ascii=False, indent=2)

    return best_model, test_metrics_dict_mean

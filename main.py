import os

# Train on four physical GPUs by default.  Override with
# CUDA_VISIBLE_DEVICES=... when a different GPU set is required.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "CUDA_VISIBLE_DEVICES", "0,1,2,3"
)
# Reduce CUDA allocator fragmentation for long-running diffusion training.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import random
import argparse
import subprocess
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import LITPModel, create_litp
from trainer import model_train, LSHT_inference
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "log_litp")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ml-1m")
DEFAULT_PLAN_FILE = os.path.join(
    PROJECT_ROOT, "datasets", "ml-1m", "semantic_plans.json"
)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', help='Dataset name: toys, amazon_beauty, steam, ml-1m, taobao, taobao_s')
parser.add_argument('--log_file', default=DEFAULT_LOG_DIR, help='log dir path')
parser.add_argument('--random_seed', type=int, default=1996, help='Random seed')
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--diagnostic_interval', type=int, default=0, help='Print/record diagnostics every N steps (0 = use the built-in print interval)')
parser.add_argument('--num_gpu', type=int, default=4, help='Number of GPUs used by DataParallel')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks in the diffusion denoiser')
parser.add_argument('--plan_num_blocks', type=int, default=2, help='Number of Transformer blocks in the semantic transition planner')
parser.add_argument('--semantic_plan_file', type=str, default=DEFAULT_PLAN_FILE, help='Offline semantic transition plan JSON')
parser.add_argument('--plan_token_emb_dim', type=int, default=128, help='Offline plan token embedding size')
parser.add_argument('--profile_emb_dim', type=int, default=16, help='Embedding size for each demographic field')
parser.add_argument('--user_emb_dim', type=int, default=64, help='User ID embedding size')
parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Persistent output directory for checkpoint, metrics and predictions')
parser.add_argument('--prediction_topk', type=int, default=20, help='Number of predicted items written for each test user')
parser.add_argument('--print_prediction_examples', type=int, default=0, help='Number of prediction records printed to stdout')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=0.001, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=8, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=1, help='Validate after every epoch')
parser.add_argument('--patience', type=int, default=40, help='Validation patience; 40 disables early stopping for the default run')
parser.add_argument('--description', type=str, default='LITP_plan_align', help='Model brief introduction')
parser.add_argument('--long_head', default=False, help='Long and short sequence, head and long-tail items')
parser.add_argument('--diversity_measure', default=False, help='Measure the diversity of recommendation results')
parser.add_argument('--epoch_time_avg', default=False, help='Calculate the average time of one epoch training')
parser.add_argument('--eta', type=float, default=0.3, help='Eta for dy_diffusion')
parser.add_argument('--total_loss_lambda', type=float, default=0.005, help='lambda for total loss (weight on L_diff)')
# LITP-specific: weight of the plan-latent alignment loss L_align.
parser.add_argument('--align_loss_lambda', type=float, default=0.1, help='lambda for plan-latent alignment loss')
parser.add_argument('--delta', type=float, default=0.5, help='Delta for temporally-aware diffusion')
args = parser.parse_args()

print(args)

log_dataset_dir = os.path.join(args.log_file, args.dataset)
os.makedirs(log_dataset_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(
        log_dataset_dir,
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".log",
    ),
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)
logger.info(args)


def print_device_banner():
    """Report which physical GPU this process is actually using."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print('CUDA_VISIBLE_DEVICES = {}'.format(visible))
    logger.info('CUDA_VISIBLE_DEVICES = %s', visible)
    if not torch.cuda.is_available():
        print('CUDA is not available; running on CPU.')
        logger.info('CUDA is not available; running on CPU.')
        return
    print('torch.cuda.device_count() = {}'.format(torch.cuda.device_count()))
    for logical_index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(logical_index)
        total_gb = props.total_memory / (1024 ** 3)
        print(
            'logical cuda:{} -> {} ({:.2f} GiB)'.format(
                logical_index, props.name, total_gb
            )
        )
        logger.info(
            'logical cuda:%d -> %s (%.2f GiB)',
            logical_index,
            props.name,
            total_gb,
        )
    if args.device == 'cuda' and torch.cuda.device_count() < args.num_gpu:
        raise RuntimeError(
            'Requested {} GPUs, but only {} are visible via CUDA_VISIBLE_DEVICES={}'
            .format(
                args.num_gpu,
                torch.cuda.device_count(),
                os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'),
            )
        )
    print('Training will use logical CUDA devices 0..{}.'.format(
        args.num_gpu - 1
    ))
    logger.info('Training will use %d GPUs.', args.num_gpu)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args


def cold_hot_long_short(data_raw, dataset_name):
    item_list = []
    len_list = []
    target_item = []

    for id_temp in data_raw['train']:
        temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for item_num_temp in item_num_count.items():
        if item_num_temp[1] < split_num:
            cold_item.append(item_num_temp[0])
        else:
            hot_item.append(item_num_temp[0])
    cold_ids, hot_ids = [], []
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        if item_temp[0] in hot_item:
            hot_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                hot_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                hot_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
        else:
            cold_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                cold_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                cold_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}

    len_short = np.percentile(len_list, 20)
    len_midshort = np.percentile(len_list, 40)
    len_midlong = np.percentile(len_list, 60)
    len_long = np.percentile(len_list, 80)

    len_seq_dict = {'short': [], 'mid_short': [], 'mid': [], 'mid_long': [], 'long': []}
    for id_temp, len_temp in enumerate(len_list):
        if dataset_name == 'ml-1m':
            temp_seq = data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1]
        else:
            temp_seq = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        if len_temp <= len_short:
            len_seq_dict['short'].append(temp_seq)
        elif len_short < len_temp <= len_midshort:
            len_seq_dict['mid_short'].append(temp_seq)
        elif len_midshort < len_temp <= len_midlong:
            len_seq_dict['mid'].append(temp_seq)
        elif len_midlong < len_temp <= len_long:
            len_seq_dict['mid_long'].append(temp_seq)
        else:
            len_seq_dict['long'].append(temp_seq)
    return cold_hot_dict, len_seq_dict, split_num, [len_short, len_midshort, len_midlong, len_long], len_list, list(item_num_count.values())


def load_semantic_plans(plan_path, dataset_path):
    """Load the offline corpus, rebuilding once if it is missing/truncated."""
    try:
        with open(plan_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        generator = os.path.join(
            PROJECT_ROOT, 'script', 'generate_semantic_plans.py'
        )
        print(
            'Semantic plan file is missing or invalid ({}). Rebuilding it...'
            .format(error),
            flush=True,
        )
        logger.warning(
            'Rebuilding invalid semantic plan file %s: %s', plan_path, error
        )
        subprocess.run(
            [
                sys.executable,
                generator,
                '--dataset', dataset_path,
                '--output', plan_path,
            ],
            check=True,
        )
        with open(plan_path, 'r', encoding='utf-8') as file:
            return json.load(file)


def main(args):
    fix_random_seed_as(args.random_seed)
    print_device_banner()
    path_data = os.path.join(
        PROJECT_ROOT,
        'datasets',
        args.dataset,
        'dataset_enriched.pkl',
    )
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    required_features = {
        'genre_matrix', 'user_gender', 'user_age', 'user_occupation'
    }
    missing = required_features.difference(data_raw)
    if missing:
        raise KeyError(
            'Missing enriched features: {}. Run script/enrich_ml1m.py first.'
            .format(sorted(missing))
        )

    args = item_num_create(args, len(data_raw['smap']))
    args.user_num = len(data_raw['umap'])
    args.genre_matrix = data_raw['genre_matrix']
    args.genre_num = data_raw['genre_matrix'].shape[1]
    args.user_gender = data_raw['user_gender']
    args.user_age = data_raw['user_age']
    args.user_occupation = data_raw['user_occupation']
    args.item_titles = data_raw['item_titles']
    args.item_genres = data_raw['item_genres']
    semantic_plans = load_semantic_plans(
        args.semantic_plan_file, path_data
    )
    if semantic_plans.get('schema_version') != 'semantic_transition_plan_v1':
        raise ValueError('Unsupported semantic plan schema: {}'.format(
            semantic_plans.get('schema_version')
        ))
    args.semantic_plan_assignments = semantic_plans['assignments']
    args.semantic_plan_vocab_size = len(semantic_plans['vocab'])
    args.semantic_plan_max_tokens = semantic_plans['max_plan_tokens']
    args.semantic_plan_fallback_id = semantic_plans['fallback_plan_id']
    args.semantic_plan_token_ids = [
        plan['token_ids'] for plan in semantic_plans['plans']
    ]
    args.semantic_plan_profile_token_ids = [
        semantic_plans['profile_token_ids'][str(user_id)]['token_ids']
        for user_id in range(args.user_num)
    ]
    # Keep one unknown-profile row for optional cold/long-tail evaluation.
    args.semantic_plan_profile_token_ids.append([0, 0, 0, 0])
    print(
        'Loaded {} offline semantic plans from {}'.format(
            len(args.semantic_plan_token_ids), args.semantic_plan_file
        )
    )
    args.inverse_smap = {value: key for key, value in data_raw['smap'].items()}
    args.inverse_umap = {value: key for key, value in data_raw['umap'].items()}
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    litp = create_litp(args)
    litp_model = LITPModel(litp, args)

    best_model, test_results = model_train(
        tra_data_loader,
        val_data_loader,
        test_data_loader,
        litp_model,
        args,
        logger,
    )


    if args.long_head:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.dataset)
        cold_data = Data_CHLS(cold_hot_dict['cold'], args)
        cold_data_loader = cold_data.get_pytorch_dataloaders()
        print('--------------Cold item-----------------------')
        LSHT_inference(best_model, args, cold_data_loader)

        hot_data = Data_CHLS(cold_hot_dict['hot'], args)
        hot_data_loader = hot_data.get_pytorch_dataloaders()
        print('--------------hot item-----------------------')
        LSHT_inference(best_model, args, hot_data_loader)

        short_data = Data_CHLS(len_seq_dict['short'], args)
        short_data_loader = short_data.get_pytorch_dataloaders()
        print('--------------Short-----------------------')
        LSHT_inference(best_model, args, short_data_loader)

        mid_short_data = Data_CHLS(len_seq_dict['mid_short'], args)
        mid_short_data_loader = mid_short_data.get_pytorch_dataloaders()
        print('--------------Mid_short-----------------------')
        LSHT_inference(best_model, args, mid_short_data_loader)

        mid_data = Data_CHLS(len_seq_dict['mid'], args)
        mid_data_loader = mid_data.get_pytorch_dataloaders()
        print('--------------Mid-----------------------')
        LSHT_inference(best_model, args, mid_data_loader)

        mid_long_data = Data_CHLS(len_seq_dict['mid_long'], args)
        mid_long_data_loader = mid_long_data.get_pytorch_dataloaders()
        print('--------------Mid_long-----------------------')
        LSHT_inference(best_model, args, mid_long_data_loader)

        long_data = Data_CHLS(len_seq_dict['long'], args)
        long_data_loader = long_data.get_pytorch_dataloaders()
        print('--------------Long-----------------------')
        LSHT_inference(best_model, args, long_data_loader)


if __name__ == '__main__':
    main(args)

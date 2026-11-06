import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from step_sample import create_named_schedule_sampler


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract timestep values from a 1-D NumPy array and broadcast them."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Return a predefined beta schedule."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(
            scale * 0.0001,
            scale * 0.02,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_end = scale * 0.02 + 0.01
        beta_end = scale * 0.001 + 0.01 if beta_end > 1 else beta_end
        return np.linspace(
            scale * 0.0001 + 0.01,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Discretize an alpha-bar function into diffusion betas."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1, t2 = i / num_diffusion_timesteps, (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """Split diffusion timesteps into sections for accelerated sampling."""
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.weight * (
            (x - u) / torch.sqrt(s + self.variance_epsilon)
        ) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
            1
            + torch.tanh(
                math.sqrt(2 / math.pi)
                * (hidden + 0.044715 * torch.pow(hidden, 3))
            )
        )
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        )
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [
            layer(x)
            .view(batch_size, -1, self.num_heads, self.size_head)
            .transpose(1, 2)
            for layer, x in zip(self.linear_layers, (q, k, v))
        ]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            # ``mask`` marks valid sequence positions.  Attention logits have
            # shape [B, H, query, key], therefore padding must be masked on the
            # key axis.  The old expansion masked query rows instead, allowing
            # every real token to attend to left-padding representations.
            key_mask = mask[:, None, None, :].to(dtype=torch.bool)
            corr = corr.masked_fill(~key_mask, -1e9)
        prob_attn = self.dropout(F.softmax(corr, dim=-1))
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.size_head)
        )
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(
            heads=attn_heads,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.input_sublayer = SublayerConnection(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.output_sublayer = SublayerConnection(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(
            hidden,
            lambda _hidden: self.attention.forward(
                _hidden,
                _hidden,
                _hidden,
                mask=mask,
            ),
        )
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerEncoder(nn.Module):
    def __init__(self, args, num_blocks=None):
        super(TransformerEncoder, self).__init__()
        blocks = num_blocks if num_blocks is not None else args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(args.hidden_size, 4, args.dropout)
                for _ in range(blocks)
            ]
        )

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class PlanEncoder(nn.Module):
    """Encode the offline three-part semantic transition plan ``E_text(y_u)``.

    The LLM/planner runs before training and stores structured plans in JSON.
    This module only embeds the canonicalized plan tokens and projects the
    resulting text representation into the item latent space.
    """

    def __init__(self, args):
        super(PlanEncoder, self).__init__()
        self.max_tokens = args.semantic_plan_max_tokens
        token_dim = args.plan_token_emb_dim
        self.token_embeddings = nn.Embedding(
            args.semantic_plan_vocab_size, token_dim, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(self.max_tokens + 4, token_dim)
        self.input_proj = nn.Linear(token_dim, args.hidden_size)
        self.att = TransformerEncoder(args, num_blocks=args.plan_num_blocks)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.norm = LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        token_rows = []
        token_masks = []
        for token_ids in args.semantic_plan_token_ids:
            token_ids = token_ids[:self.max_tokens]
            padding = self.max_tokens - len(token_ids)
            token_rows.append(token_ids + [0] * padding)
            token_masks.append([1.0] * len(token_ids) + [0.0] * padding)
        self.register_buffer(
            "plan_token_ids", torch.as_tensor(token_rows, dtype=torch.long)
        )
        self.register_buffer(
            "plan_token_masks", torch.as_tensor(token_masks, dtype=torch.float32)
        )
        self.register_buffer(
            "profile_token_ids",
            torch.as_tensor(args.semantic_plan_profile_token_ids, dtype=torch.long),
        )

    def forward(self, plan_ids, user_ids):
        plan_ids = plan_ids.squeeze(-1)
        user_ids = user_ids.squeeze(-1)
        base_token_ids = self.plan_token_ids[plan_ids]
        base_token_masks = self.plan_token_masks[plan_ids]
        profile_token_ids = self.profile_token_ids[user_ids]
        profile_token_masks = torch.ones_like(profile_token_ids, dtype=torch.float32)
        token_ids = torch.cat([base_token_ids, profile_token_ids], dim=1)
        token_mask = torch.cat([base_token_masks, profile_token_masks], dim=1)
        positions = torch.arange(
            token_ids.shape[1], device=token_ids.device
        ).unsqueeze(0)
        hidden = self.token_embeddings(token_ids) + self.position_embeddings(positions)
        hidden = self.input_proj(hidden)
        hidden = self.att(hidden, token_mask)
        pooled = (hidden * token_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.norm(self.dropout(self.proj(pooled)))


class LITPApproximator(nn.Module):
    """Plan-conditioned denoising network (``ε_θ``).

    Extends TDRec's approximator with **language guidance**: the semantic
    transition plan ``q_u`` is projected and added to the noisy latent
    ``x_t`` together with the timestep embedding, so the reverse diffusion
    process is conditioned on both the temporal history ``s_u`` (via the
    transformer attention over ``rep_item``) and the plan ``q_u``.
    """

    def __init__(self, hidden_size, args):
        super(LITPApproximator, self).__init__()
        self.hidden_size = hidden_size
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        # language guidance: project q_u into a residual condition
        self.plan_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        self.profile_embed = nn.Sequential(
            nn.Linear(self.hidden_size, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, self.hidden_size),
        )
        self.att = TransformerEncoder(args)
        self.lambda_uncertainty = args.lambda_uncertainty
        self.dropout = nn.Dropout(args.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period)
            * th.arange(start=0, end=half, dtype=th.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat(
                [embedding, th.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq, plan_emb, profile_emb):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        # Both q_u and p_u condition the denoising trajectory.
        emb_plan = self.plan_embed(plan_emb)
        emb_profile = self.profile_embed(profile_emb)
        x_t = x_t + emb_t + emb_plan + emb_profile

        lambda_uncertainty = th.normal(
            mean=th.full(rep_item.shape, self.lambda_uncertainty),
            std=th.full(rep_item.shape, self.lambda_uncertainty),
        ).to(x_t.device)

        rep_diffu = self.att(
            rep_item + lambda_uncertainty * x_t.unsqueeze(1),
            mask_seq,
        )
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))

        out = rep_diffu[:, -1, :]
        return out, rep_diffu


class LITP(nn.Module):
    """Language-guided Latent Diffusion Transition.

    The diffusion forward / reverse process follows TDRec's temporally-aware
    formulation (so the encoder behaviour is consistent with TDRec), and the
    denoiser ``ε_θ`` is additionally conditioned on the semantic transition
    plan ``q_u`` (language guidance).
    """

    def __init__(self, args):
        super(LITP, self).__init__()
        self.hidden_size = args.hidden_size
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(
            self.diffusion_steps,
            [self.diffusion_steps],
        )
        self.rescale_timesteps = args.rescale_timesteps

        betas = get_named_beta_schedule(
            args.noise_schedule,
            self.diffusion_steps,
        )
        self.betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(
            1.0,
            self.alphas_cumprod[:-1],
        )

        delta = args.delta
        self.dy_t = np.array(
            [
                1
                - delta
                + 2 * delta / (args.diffusion_steps - 1) * i
                for i in range(args.diffusion_steps)
            ]
        )

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(
            1.0 - self.alphas_cumprod
        )

        self.posterior_mean_coef1 = (
            self.betas
            * np.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

        self.num_timesteps = int(self.betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler_name,
            self.num_timesteps,
        )

        self.xstart_model = LITPApproximator(self.hidden_size, args)

    def dynamic_q_sample(
        self,
        item_list,
        x_start,
        t,
        noise=None,
        mask=None,
    ):
        """Run the temporally-aware forward diffusion process."""
        if noise is None:
            target_noise = th.randn_like(x_start)
            item_list_noise = th.randn_like(item_list)

        dy_t = _extract_into_tensor(
            self.dy_t,
            t,
            (item_list.size(0), item_list.size(1)),
        )
        cumprod_dy_t = dy_t * torch.cumprod(
            _extract_into_tensor(
                self.dy_t,
                t,
                (item_list.size(0), item_list.size(1)),
            ),
            dim=1,
        )
        flip_log_dy_t = (
            torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask
        )

        dy_x_s_1 = torch.sum(
            flip_log_dy_t.unsqueeze(-1) * item_list,
            dim=1,
        )
        dy_noise_s_1 = torch.sum(
            flip_log_dy_t.unsqueeze(-1) * item_list_noise,
            dim=1,
        )

        dy_x_0 = dy_x_s_1 + _extract_into_tensor(
            self.dy_t,
            t,
            x_start.shape,
        ) * x_start
        dy_noise = dy_noise_s_1 + _extract_into_tensor(
            self.dy_t,
            t,
            target_noise.shape,
        ) * target_noise

        x_t = (
            _extract_into_tensor(
                self.sqrt_alphas_cumprod,
                t,
                dy_x_0.shape,
            )
            * dy_x_0
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                dy_noise.shape,
            )
            * dy_noise
        )
        return x_t, dy_x_s_1, dy_noise_s_1

    def _scale_timesteps(self, t):
        return (
            t.float() * (1000.0 / self.num_timesteps)
            if self.rescale_timesteps
            else t
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute the posterior mean."""
        posterior_mean = (
            _extract_into_tensor(
                self.posterior_mean_coef1,
                t,
                x_t.shape,
            )
            * x_start
            + _extract_into_tensor(
                self.posterior_mean_coef2,
                t,
                x_t.shape,
            )
            * x_t
        )
        return posterior_mean

    def dynamic_p_mean_variance(
        self,
        rep_item,
        x_t,
        t,
        mask_seq,
        plan_emb,
        profile_emb,
        dy_x_s_1,
    ):
        """Predict x_0 and compute the reverse-process statistics."""
        model_output, _ = self.xstart_model(
            rep_item,
            x_t,
            self._scale_timesteps(t),
            mask_seq,
            plan_emb,
            profile_emb,
        )

        x_0 = model_output
        model_log_variance = np.log(
            np.append(self.posterior_variance[1], self.betas[1:])
        )
        model_log_variance = _extract_into_tensor(
            model_log_variance,
            t,
            x_t.shape,
        )

        model_mean = self.q_posterior_mean_variance(
            x_start=x_0,
            x_t=x_t,
            t=t,
        )
        return model_mean, model_log_variance

    def dynamic_p_sample(self, item_rep, noise_x_t, t, mask_seq, plan_emb, profile_emb):
        """Run one reverse diffusion step."""
        dy_t = _extract_into_tensor(
            self.dy_t,
            t,
            (item_rep.size(0), item_rep.size(1)),
        )
        cumprod_dy_t = dy_t * torch.cumprod(
            _extract_into_tensor(
                self.dy_t,
                t,
                (item_rep.size(0), item_rep.size(1)),
            ),
            dim=1,
        )
        flip_log_dy_t = (
            torch.flip(torch.log(cumprod_dy_t + 1), dims=[1]) * mask_seq
        )
        dy_x_s_1 = torch.sum(
            flip_log_dy_t.unsqueeze(-1) * item_rep,
            dim=1,
        )

        model_mean, model_log_variance = self.dynamic_p_mean_variance(
            item_rep,
            noise_x_t,
            t,
            mask_seq,
            plan_emb,
            profile_emb,
            dy_x_s_1,
        )
        noise = th.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().view(
            -1,
            *([1] * (len(noise_x_t.shape) - 1)),
        )
        sample_xt = (
            model_mean
            + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        )
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq, plan_emb, profile_emb):
        """Run the full reverse diffusion process."""
        # DataParallel replicas do not reliably expose their replicated
        # parameters through ``parameters()`` during parallel_apply.  The input
        # shard is already placed on the correct replica device, so use it as
        # the authoritative device for timestep tensors.
        device = item_rep.device
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.dynamic_p_sample(
                    item_rep,
                    noise_x_t,
                    t,
                    mask_seq,
                    plan_emb,
                    profile_emb,
                )
        return noise_x_t

    def forward(self, item_rep, item_tag, mask_seq, plan_emb, profile_emb):
        """Run the training forward pass."""
        t, weights = self.schedule_sampler.sample(
            item_rep.shape[0],
            item_tag.device,
        )

        x_t, _, _ = self.dynamic_q_sample(
            item_rep,
            item_tag,
            t,
            mask=mask_seq,
        )

        dy_pre_x_t, dy_item_rep_out = self.xstart_model(
            item_rep,
            x_t,
            self._scale_timesteps(t),
            mask_seq,
            plan_emb,
            profile_emb,
        )
        x_0 = dy_pre_x_t
        item_rep_out = dy_item_rep_out

        return x_0, item_rep_out, weights, t, x_t, dy_pre_x_t

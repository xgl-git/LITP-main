import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from litp import LayerNorm, LITP, PlanEncoder


class LITPModel(nn.Module):
    """TDRec-compatible LITP model with genre and profile conditioning.

    Item manifold:
        v_j = item_id_embedding(j) + W_g genre(j)

    User profile:
        p_u = MLP([gender, age, occupation, user_id embeddings])

    ``p_u`` conditions both the transition planner and diffusion denoiser.  The
    output remains a next-item prediction.  Tag Genome, IP graph and watch
    progress are intentionally excluded.
    """

    def __init__(self, litp, args):
        super(LITPModel, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num + 1
        self.user_num = args.user_num
        self.dataset = args.dataset.lower()

        self.item_embeddings = nn.Embedding(
            self.item_num, self.emb_dim, padding_idx=0
        )
        self.genre_projection = nn.Linear(
            args.genre_num, self.emb_dim, bias=False
        )
        self.register_buffer(
            "genre_features",
            torch.as_tensor(args.genre_matrix, dtype=torch.float32),
        )

        profile_dim = args.profile_emb_dim
        self.gender_embeddings = nn.Embedding(3, profile_dim, padding_idx=0)
        self.age_embeddings = nn.Embedding(8, profile_dim, padding_idx=0)
        self.occupation_embeddings = nn.Embedding(
            22, profile_dim, padding_idx=0
        )
        self.user_embeddings = nn.Embedding(self.user_num + 1, args.user_emb_dim)
        self.profile_mlp = nn.Sequential(
            nn.Linear(3 * profile_dim + args.user_emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.profile_norm = LayerNorm(self.emb_dim, eps=1e-12)
        self.register_buffer(
            "user_gender",
            torch.as_tensor(
                list(args.user_gender) + [0], dtype=torch.long
            ),
        )
        self.register_buffer(
            "user_age",
            torch.as_tensor(
                list(args.user_age) + [0], dtype=torch.long
            ),
        )
        self.register_buffer(
            "user_occupation",
            torch.as_tensor(
                list(args.user_occupation) + [0], dtype=torch.long
            ),
        )

        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, self.emb_dim)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.plan_encoder = PlanEncoder(args)
        self.litp = litp

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

    def item_manifold(self, item_ids):
        """Return item ID + genre representation for arbitrary ID shapes."""
        item_emb = self.item_embeddings(item_ids)
        genre_emb = self.genre_projection(self.genre_features[item_ids])
        return item_emb + genre_emb

    def all_item_manifold(self):
        item_ids = torch.arange(self.item_num, device=self.genre_features.device)
        # Match the LayerNorm applied to the diffusion target space so the
        # ranking space and the training target space stay consistent.
        return self.LayerNorm(self.item_manifold(item_ids))

    def encode_profile(self, user_ids):
        user_ids = user_ids.squeeze(-1)
        gender = self.user_gender[user_ids]
        age = self.user_age[user_ids]
        occupation = self.user_occupation[user_ids]
        profile = torch.cat(
            [
                self.gender_embeddings(gender),
                self.age_embeddings(age),
                self.occupation_embeddings(occupation),
                self.user_embeddings(user_ids),
            ],
            dim=-1,
        )
        return self.profile_norm(self.profile_mlp(profile))

    def litp_pre(self, item_rep, tag_emb, mask_seq, plan_emb, profile_emb):
        return self.litp(
            item_rep, tag_emb, mask_seq, plan_emb, profile_emb
        )

    def reverse(self, item_rep, noise_x_t, mask_seq, plan_emb, profile_emb):
        return self.litp.reverse_p_sample(
            item_rep, noise_x_t, mask_seq, plan_emb, profile_emb
        )

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu_ce(self, rep_diffu, labels):
        """Contrastive next-item recommendation loss on the fused manifold."""
        item_matrix = self.all_item_manifold()
        if self.dataset == "steam":
            scores = torch.matmul(rep_diffu, item_matrix.t())
        else:
            scores = torch.matmul(
                F.normalize(rep_diffu, p=2, dim=-1),
                F.normalize(item_matrix, p=2, dim=-1).t(),
            ) / 0.07
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        """Compute full-item scores for evaluation."""
        item_matrix = self.all_item_manifold()
        if self.dataset == "steam":
            return torch.matmul(rep_diffu, item_matrix.t())
        return torch.matmul(
            F.normalize(rep_diffu, p=2, dim=-1),
            F.normalize(item_matrix, p=2, dim=-1).t(),
        )

    def loss_rmse_dy(self, true_target, dy_pre_x_t):
        """Diffusion reconstruction loss."""
        return torch.sqrt(self.loss_mse(true_target, dy_pre_x_t))

    def loss_align(self, q_u, delta_u):
        """Plan-latent alignment: ``1 - cos(q_u, Δ_u)``."""
        q_norm = F.normalize(q_u, p=2, dim=-1)
        d_norm = F.normalize(delta_u, p=2, dim=-1)
        return (1.0 - (q_norm * d_norm).sum(dim=-1)).mean()

    def forward(self, sequence, tag, user_ids, plan_ids, train_flag=True):
        mask_seq = (sequence > 0).float()
        positions = torch.arange(
            sequence.size(1), device=sequence.device
        ).unsqueeze(0)
        item_embeddings = (
            self.item_manifold(sequence)
            + self.position_embeddings(positions)
        )
        item_embeddings = self.LayerNorm(
            self.embed_dropout(item_embeddings)
        )
        profile_emb = self.encode_profile(user_ids)

        # The semantic plan was generated offline from the history available at
        # this sample's cutoff and is now encoded as E_text(y_u).
        q_u = self.plan_encoder(plan_ids, user_ids)

        if train_flag:
            # Apply the same LayerNorm used for the sequence representation so
            # the diffusion target lives in the same, scale-controlled space.
            # Without this, tag_emb/v_jt are raw item_manifold() outputs whose
            # norm is free to grow unbounded (driven by the scale-invariant
            # cosine-based rec loss), which is why `target_norm` kept climbing
            # from ~11 to ~16+ while `pred_norm` stayed flat, and `diff` (RMSE)
            # rose instead of converging.
            tag_emb = self.LayerNorm(self.item_manifold(tag.squeeze(-1)))
            v_jt = self.LayerNorm(self.item_manifold(sequence[:, -1]))
            delta_u = (tag_emb - v_jt).detach()
            (
                rep_diffu,
                _,
                weights,
                t,
                _,
                dy_pre_x_t,
            ) = self.litp_pre(
                item_embeddings,
                tag_emb,
                mask_seq,
                q_u,
                profile_emb,
            )
            return (
                None,
                rep_diffu,
                weights,
                t,
                None,
                None,
                tag_emb,
                dy_pre_x_t,
                q_u,
                delta_u,
            )

        noise_x_t = th.randn_like(item_embeddings[:, -1, :])
        rep_diffu = self.reverse(
            item_embeddings,
            noise_x_t,
            mask_seq,
            q_u,
            profile_emb,
        )
        scores = self.diffu_rep_pre(rep_diffu)
        return (
            scores,
            rep_diffu,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def create_litp(args):
    """Create the LITP diffusion module."""
    return LITP(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class V6JointTransformer(nn.Module):
    def __init__(self, future_steps=6, num_modes=3, top_n_neighbors=5, d_model=256, frame_dt=0.5):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.top_n = top_n_neighbors
        self.d_model = d_model
        self.frame_dt = frame_dt

        resnet = models.resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.hist_embed = nn.Linear(4, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dna_embed = nn.Linear(12, d_model)
        ego_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.ego_transformer = nn.TransformerEncoder(ego_layer, num_layers=2)

        self.social_embed = nn.Linear(17, d_model)
        social_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.social_transformer = nn.TransformerEncoder(social_layer, num_layers=1)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
        )

        self.mode_embeddings = nn.Parameter(torch.randn(num_modes, d_model))
        mode_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.mode_mixer = nn.TransformerEncoder(mode_layer, num_layers=1)
        self.mode_logits_head = nn.Linear(d_model, 1)
        self.mode_delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, future_steps * 2),
        )

        self.neighbor_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 256),
            nn.GELU(),
        )
        self.neighbor_gmm_head = nn.Linear(256, future_steps * 5)

    def build_history_features(self, ego_hist):
        velocity = torch.diff(ego_hist, dim=1, prepend=ego_hist[:, :1]) / self.frame_dt
        return torch.cat([ego_hist, velocity], dim=-1)

    def make_safe_social_mask(self, social_mask):
        safe_mask = social_mask.clone()
        all_masked = safe_mask.all(dim=1)
        if all_masked.any().item():
            safe_mask[all_masked, 0] = False
        return safe_mask

    def forward(self, map_img, ego_hist, ego_dna, social_graph, social_mask):
        B = map_img.shape[0]
        safe_social_mask = self.make_safe_social_mask(social_mask)

        spatial_feat = self.cnn_proj(self.cnn_backbone(map_img).flatten(1))

        hist_tokens = self.hist_embed(self.build_history_features(ego_hist))
        hist_tokens = self.pos_encoder(hist_tokens)
        ego_tokens = self.ego_transformer(hist_tokens)
        ego_state = ego_tokens[:, -1] + ego_tokens.mean(dim=1) + self.dna_embed(ego_dna)

        social_tokens = self.social_embed(social_graph)
        encoded_social = self.social_transformer(social_tokens, src_key_padding_mask=safe_social_mask)
        social_ctx, attn_weights = self.cross_attention(
            ego_state.unsqueeze(1),
            encoded_social,
            encoded_social,
            key_padding_mask=safe_social_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        social_ctx = social_ctx.squeeze(1)
        fused = self.fusion_mlp(torch.cat([spatial_feat, ego_state, social_ctx], dim=-1))

        mode_context = fused.unsqueeze(1) + self.mode_embeddings.unsqueeze(0)
        mode_context = self.mode_mixer(mode_context)
        mode_logits = self.mode_logits_head(mode_context).squeeze(-1)
        raw_mode_deltas = self.mode_delta_head(mode_context).view(B, self.num_modes, self.future_steps, 2)
        mode_deltas = 4.0 * torch.tanh(raw_mode_deltas)
        ego_trajs = torch.cumsum(mode_deltas, dim=2)
        ego_probs = torch.softmax(mode_logits, dim=-1)

        attn_scores = attn_weights.mean(dim=1).squeeze(1)
        masked_scores = attn_scores.masked_fill(social_mask, -1e9)
        topk_idx = torch.topk(masked_scores, k=self.top_n, dim=-1).indices

        gather_social = topk_idx.unsqueeze(-1).expand(-1, -1, encoded_social.size(-1))
        top_neighbor_feats = torch.gather(encoded_social, 1, gather_social)
        raw_neighbor = torch.gather(social_graph, 1, topk_idx.unsqueeze(-1).expand(-1, -1, social_graph.size(-1)))
        neighbor_current = raw_neighbor[..., :2].unsqueeze(2)

        neigh_fused = self.neighbor_fusion(
            torch.cat([fused.unsqueeze(1).expand(-1, self.top_n, -1), top_neighbor_feats], dim=-1)
        )
        raw_gmm = self.neighbor_gmm_head(neigh_fused).view(B, self.top_n, self.future_steps, 5)
        neighbor_mu = neighbor_current + torch.cumsum(4.0 * torch.tanh(raw_gmm[..., :2]), dim=2)
        neighbor_sigma = F.softplus(raw_gmm[..., 2:4]) + 1e-3
        neighbor_rho = torch.tanh(raw_gmm[..., 4:5]) * 0.95
        neighbor_gmm = torch.cat([neighbor_mu, neighbor_sigma, neighbor_rho], dim=-1)

        return ego_trajs, mode_logits, ego_probs, neighbor_gmm, topk_idx

    def inference(self, map_img, ego_hist, ego_dna, social_graph, social_mask):
        with torch.no_grad():
            trajs, _, probs, gmm, _ = self.forward(map_img, ego_hist, ego_dna, social_graph, social_mask)
            top3_probs, top3_idx = torch.topk(probs, k=min(3, self.num_modes), dim=-1)
            expanded_idx = top3_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.future_steps, 2)
            top3_trajs = torch.gather(trajs, 1, expanded_idx)
            return top3_trajs, top3_probs, gmm

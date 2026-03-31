import torch
import torch.nn as nn
import torch.nn.functional as F


class V6UltimateLoss(nn.Module):
    def __init__(self, temperature=1.6, map_resolution=0.25):
        super().__init__()
        self.temp = temperature
        self.map_res = map_resolution
        self.eps = 1e-6

    def sample_map_channel(self, map_channel, traj):
        half_extent = (map_channel.shape[-1] * self.map_res) / 2.0
        norm_x = traj[..., 0] / half_extent
        norm_y = -traj[..., 1] / half_extent
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(map_channel, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(1).squeeze(1)

    def forward(self, ego_trajs, mode_logits, neighbor_gmm, topk_idx, gt_ego, gt_neighbors, map_tensor, social_mask):
        batch_size, num_modes, future_steps, _ = ego_trajs.shape
        device = ego_trajs.device

        gt_expanded = gt_ego.unsqueeze(1).expand(-1, num_modes, -1, -1)
        ade_all = torch.norm(ego_trajs - gt_expanded, dim=-1).mean(dim=-1)
        fde_all = torch.norm(ego_trajs[:, :, -1, :] - gt_expanded[:, :, -1, :], dim=-1)
        mode_cost = ade_all + (1.5 * fde_all)
        best_idx = torch.argmin(mode_cost, dim=1)
        best_trajs = ego_trajs[torch.arange(batch_size, device=device), best_idx]

        loss_wta = F.smooth_l1_loss(best_trajs, gt_ego)
        loss_wta = loss_wta + 0.5 * F.smooth_l1_loss(best_trajs[:, -1], gt_ego[:, -1])

        log_probs = F.log_softmax(mode_logits / self.temp, dim=-1)
        hard_cls = F.cross_entropy(mode_logits / self.temp, best_idx)
        soft_targets = F.softmax((-mode_cost.detach()) / self.temp, dim=-1)
        soft_cls = F.kl_div(log_probs, soft_targets, reduction="batchmean")
        loss_conf = hard_cls + (0.5 * soft_cls)

        gather_index = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, future_steps, 2)
        gt_neighbors_selected = torch.gather(gt_neighbors, 1, gather_index)
        neighbor_mu = neighbor_gmm[..., :2]
        neighbor_sigma = neighbor_gmm[..., 2:4].clamp_min(1e-3)
        neighbor_rho = neighbor_gmm[..., 4].clamp(min=-0.95, max=0.95)

        dx = gt_neighbors_selected[..., 0] - neighbor_mu[..., 0]
        dy = gt_neighbors_selected[..., 1] - neighbor_mu[..., 1]
        sx = neighbor_sigma[..., 0]
        sy = neighbor_sigma[..., 1]
        one_minus_rho2 = (1.0 - neighbor_rho.pow(2)).clamp_min(1e-3)
        z_term = ((dx / sx) ** 2) + ((dy / sy) ** 2) - (2.0 * neighbor_rho * dx * dy / (sx * sy))
        nll = torch.log(2.0 * torch.pi * sx * sy * torch.sqrt(one_minus_rho2) + self.eps)
        nll = nll + (z_term / (2.0 * one_minus_rho2))

        selected_mask = torch.gather(social_mask, 1, topk_idx)
        valid_mask = (~selected_mask).unsqueeze(-1).expand_as(nll)
        if valid_mask.any().item():
            loss_nll = (nll * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            loss_nll = torch.zeros((), device=device)

        forbidden_samples = self.sample_map_channel(map_tensor[:, 2:3], best_trajs)
        flexible_samples = self.sample_map_channel(map_tensor[:, 1:2], best_trajs)
        loss_map = (15.0 * forbidden_samples.mean()) + (0.5 * flexible_samples.mean())

        if valid_mask.any().item():
            distances = torch.norm(best_trajs.unsqueeze(1) - neighbor_mu, dim=-1)
            repulsion = torch.clamp(1.5 - distances, min=0.0)
            loss_social = (repulsion * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            loss_social = torch.zeros((), device=device)

        if best_trajs.size(1) >= 3:
            second_diff = best_trajs[:, 2:] - (2.0 * best_trajs[:, 1:-1]) + best_trajs[:, :-2]
            loss_smooth = torch.norm(second_diff, dim=-1).mean()
        else:
            loss_smooth = torch.zeros((), device=device)

        total = (
            (1.0 * loss_wta)
            + (0.8 * loss_conf)
            + (0.5 * loss_nll)
            + (1.5 * loss_map)
            + (0.5 * loss_social)
            + (0.1 * loss_smooth)
        )

        stats = {
            "total": total.detach(),
            "wta": loss_wta.detach(),
            "conf": loss_conf.detach(),
            "nll": loss_nll.detach(),
            "map": loss_map.detach(),
            "social": loss_social.detach(),
            "smooth": loss_smooth.detach(),
        }
        return total, stats

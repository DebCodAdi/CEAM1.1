import torch


def sample_map_channel(map_channel, traj, map_res):
    half_extent = (map_channel.shape[-1] * map_res) / 2.0
    norm_x = traj[..., 0] / half_extent
    norm_y = -traj[..., 1] / half_extent
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
    sampled = torch.nn.functional.grid_sample(
        map_channel,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(1)


def expected_calibration_error(confidence, correctness, bins=10):
    ece = confidence.new_zeros(())
    for idx in range(bins):
        lower = idx / bins
        upper = (idx + 1) / bins
        in_bin = (confidence >= lower) & (confidence < upper if idx < bins - 1 else confidence <= upper)
        if in_bin.any().item():
            bin_conf = confidence[in_bin].mean()
            bin_acc = correctness[in_bin].float().mean()
            ece = ece + (in_bin.float().mean() * torch.abs(bin_conf - bin_acc))
    return ece


@torch.no_grad()
def get_v6_metrics(ego_probs, pred_trajs, gt_trajs, neighbor_mu, topk_idx, social_mask, map_tensor, map_res=0.25):
    batch_size, num_modes, _, _ = pred_trajs.shape
    gt_expanded = gt_trajs.unsqueeze(1).expand(-1, num_modes, -1, -1)

    ade_all = torch.norm(pred_trajs - gt_expanded, dim=-1).mean(dim=-1)
    fde_all = torch.norm(pred_trajs[:, :, -1, :] - gt_expanded[:, :, -1, :], dim=-1)
    mode_cost = ade_all + (1.5 * fde_all)
    best_idx = torch.argmin(mode_cost, dim=1)

    min_ade = ade_all.min(dim=1).values.mean()
    min_fde = fde_all.min(dim=1).values.mean()

    top_conf, top_idx = torch.max(ego_probs, dim=-1)
    top1_trajs = pred_trajs[torch.arange(batch_size, device=pred_trajs.device), top_idx]
    top1_is_best = (top_idx == best_idx).float()
    ece = expected_calibration_error(top_conf, top1_is_best)

    forbidden_samples = sample_map_channel(map_tensor[:, 2:3], top1_trajs, map_res)
    off_road = (forbidden_samples.max(dim=-1).values > 0.3).float().mean()

    selected_mask = torch.gather(social_mask, 1, topk_idx)
    valid_mask = (~selected_mask).unsqueeze(-1).expand(-1, -1, top1_trajs.size(1))
    if valid_mask.any().item():
        dist = torch.norm(top1_trajs.unsqueeze(1) - neighbor_mu, dim=-1)
        social_violation = (((dist < 1.5) & valid_mask).any(dim=-1).any(dim=-1)).float().mean()
    else:
        social_violation = torch.zeros((), device=pred_trajs.device)

    return {
        "minADE": min_ade.item(),
        "minFDE": min_fde.item(),
        "OffRoadRate": off_road.item(),
        "SocialViolation": social_violation.item(),
        "Accuracy_Top1": top1_is_best.mean().item(),
        "ECE": ece.item(),
    }

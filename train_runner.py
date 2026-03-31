import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from dataset import CustomTrajectoryDataset
from loss import V6UltimateLoss
from metrics import get_v6_metrics
from model import V6JointTransformer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_feature_path(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("V6_FEATURE_PKL")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(PROJECT_ROOT, "data", "processed", "v6_feature_set.pkl"))
    candidates.append(os.path.join(PROJECT_ROOT, "data", "processed", "v6_feature_set_bench.pkl"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0]


def build_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def move_batch_to_device(batch, device, non_blocking=False):
    return [tensor.to(device, non_blocking=non_blocking) for tensor in batch]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the V6 joint transformer.")
    parser.add_argument("--feature-pkl", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-neighbors", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--save-name", default="v6_best_model.pth")
    parser.add_argument("--history-name", default="training_history.json")
    args = parser.parse_args()

    seed_everything(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}.")
    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    feature_pkl = resolve_feature_path(args.feature_pkl)
    save_dir = os.path.join(PROJECT_ROOT, "weights")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(feature_pkl):
        print(f"[ERROR] Feature file not found: {feature_pkl}")
        return

    full_dataset = CustomTrajectoryDataset(feature_pkl, max_neighbors=args.max_neighbors)
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        print("[ERROR] Validation split leaves no training data.")
        return

    split_generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=split_generator)

    pin_memory = device.type == "cuda"
    train_loader = build_loader(train_ds, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = build_loader(val_ds, args.batch_size, False, args.num_workers, pin_memory)

    model = V6JointTransformer(
        future_steps=full_dataset.future_steps,
        num_modes=3,
        top_n_neighbors=5,
        d_model=args.d_model,
    ).to(device)
    loss_fn = V6UltimateLoss(temperature=args.temperature, map_resolution=0.25).to(device)
    optimizer = AdamW(model.parameters(), lr=args.max_lr / 10.0, weight_decay=1e-2)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_ade": [],
        "val_fde": [],
        "val_acc": [],
        "val_ece": [],
        "time_per_epoch": [],
    }
    best_val_ade = float("inf")
    start_total_time = time.time()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0

            train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", leave=False)
            for batch in train_bar:
                map_img, ego_h, ego_dna, social, mask, gt_e, gt_n = move_batch_to_device(
                    batch, device, non_blocking=pin_memory
                )
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    ego_trajs, mode_logits, _, neighbor_gmm, topk_idx = model(map_img, ego_h, ego_dna, social, mask)
                    loss, loss_stats = loss_fn(
                        ego_trajs, mode_logits, neighbor_gmm, topk_idx, gt_e, gt_n, map_img, mask
                    )

                scale_before = scaler.get_scale() if amp_enabled else None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                if (not amp_enabled) or scaler.get_scale() >= scale_before:
                    scheduler.step()

                running_loss += loss.item()
                train_bar.set_postfix(
                    {
                        "loss": f"{loss_stats['total'].item():.3f}",
                        "conf": f"{loss_stats['conf'].item():.3f}",
                        "soc": f"{loss_stats['social'].item():.3f}",
                    }
                )

            model.eval()
            val_metrics = {
                "minADE": 0.0,
                "minFDE": 0.0,
                "OffRoadRate": 0.0,
                "SocialViolation": 0.0,
                "Accuracy_Top1": 0.0,
                "ECE": 0.0,
            }

            with torch.no_grad():
                for batch in val_loader:
                    map_img, ego_h, ego_dna, social, mask, gt_e, gt_n = move_batch_to_device(
                        batch, device, non_blocking=pin_memory
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        ego_trajs, _, ego_probs, neighbor_gmm, topk_idx = model(map_img, ego_h, ego_dna, social, mask)

                    batch_metrics = get_v6_metrics(
                        ego_probs,
                        ego_trajs,
                        gt_e,
                        neighbor_gmm[..., :2],
                        topk_idx,
                        mask,
                        map_img,
                    )
                    for key in val_metrics:
                        val_metrics[key] += batch_metrics[key]

            for key in val_metrics:
                val_metrics[key] /= len(val_loader)

            epoch_duration = time.time() - epoch_start
            avg_train_loss = running_loss / len(train_loader)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train_loss)
            history["val_ade"].append(val_metrics["minADE"])
            history["val_fde"].append(val_metrics["minFDE"])
            history["val_acc"].append(val_metrics["Accuracy_Top1"])
            history["val_ece"].append(val_metrics["ECE"])
            history["time_per_epoch"].append(epoch_duration)

            print(f"\n[Epoch {epoch + 1}] {epoch_duration:.1f}s")
            print(f"  train loss: {avg_train_loss:.4f}")
            print(f"  val minADE/minFDE: {val_metrics['minADE']:.3f} / {val_metrics['minFDE']:.3f}")
            print(f"  top1 acc / ECE: {val_metrics['Accuracy_Top1']:.3f} / {val_metrics['ECE']:.3f}")
            print(
                f"  off-road / social-violation: "
                f"{val_metrics['OffRoadRate'] * 100.0:.1f}% / {val_metrics['SocialViolation'] * 100.0:.1f}%"
            )

            if val_metrics["minADE"] < best_val_ade:
                best_val_ade = val_metrics["minADE"]
                save_path = os.path.join(save_dir, args.save_name)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_config": {
                            "future_steps": model.future_steps,
                            "num_modes": model.num_modes,
                            "top_n_neighbors": model.top_n,
                            "d_model": model.d_model,
                        },
                        "feature_pkl": feature_pkl,
                        "temperature": args.temperature,
                        "ade": val_metrics["minADE"],
                        "fde": val_metrics["minFDE"],
                    },
                    save_path,
                )
                print(f"  saved new best checkpoint to {save_path}")

            history_path = os.path.join(save_dir, args.history_name)
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        total_minutes = (time.time() - start_total_time) / 60.0
        print(f"[SUCCESS] Training complete in {total_minutes:.2f} minutes.")

    except KeyboardInterrupt:
        interrupted_path = os.path.join(save_dir, "v6_interrupted.pth")
        torch.save(model.state_dict(), interrupted_path)
        print(f"[WARN] Training interrupted. Saved partial weights to {interrupted_path}")


if __name__ == "__main__":
    main()

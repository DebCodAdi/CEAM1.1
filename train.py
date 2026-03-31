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


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "src"))

from dataset import CustomTrajectoryDataset
from loss import V6UltimateLoss
from metrics import get_v6_metrics
from model import V6JointTransformer
from train_runner import main as train_runner_main


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
    candidates.append(os.path.join(project_root, "data", "processed", "v6_feature_set.pkl"))
    candidates.append(os.path.join(project_root, "data", "processed", "v6_feature_set_bench.pkl"))

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
    return train_runner_main()

    # ==========================================
    # 1. SETUP & HYPERPARAMETERS
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 [PITCHERS V6 ENGINE] Booting on {device}...")

    EPOCHS = 30
    BATCH_SIZE = 16  
    MAX_LR = 2e-4    
    VAL_SPLIT = 0.15 

    # Path Resolution (Double-Checked for Windows)
    pkl_file = os.path.join(project_root, "src", "data", "processed", "v6_feature_set.pkl")
    save_dir = os.path.join(project_root, "weights")
    os.makedirs(save_dir, exist_ok=True)

    # History storage for Phase 5 Data Visualization
    history = {
        "epoch": [], "train_loss": [], 
        "val_ade": [], "val_fde": [], 
        "val_acc": [], "time_per_epoch": []
    }

    # ==========================================
    # 2. DATASET & DATA PREP
    # ==========================================
    if not os.path.exists(pkl_file):
        print(f"❌ [CRITICAL] Data not found at {pkl_file}!")
        return

    full_dataset = CustomTrajectoryDataset(pkl_file)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # pin_memory=True speeds up transfer to your RTX 2050
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # ==========================================
    # 3. INITIALIZE BRAIN
    # ==========================================
    model = V6JointTransformer().to(device)
    loss_fn = V6UltimateLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=1e-2)
    
    # OneCycleLR is the best scheduler for Transformer convergence
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_val_ade = float('inf')
    start_total_time = time.time()

    # ==========================================
    # 4. THE TRAINING LOOP
    # ==========================================
    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
            for data in pbar:
                # Unpack and push to GPU
                map_img, ego_h, ego_dna, social, mask, gt_e, gt_n = [d.to(device) for d in data]

                optimizer.zero_grad()
                
                # 1. Forward Pass
                ego_trajs, ego_probs, n_gmm, topk_idx = model(map_img, ego_h, ego_dna, social, mask)
                
                # 2. Complex Loss Calculation
                loss, wta, conf, nll, map_err, soc_err = loss_fn(
                    ego_trajs, ego_probs, n_gmm, topk_idx, gt_e, gt_n, map_img, mask
                )

                # 3. Backprop & Clip (Prevents exploding gradients)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.2f}", 'Soc': f"{soc_err.item():.2f}"})

            # ==========================================
            # 5. VALIDATION PASS (The Generalization Test)
            # ==========================================
            model.eval()
            val_metrics = {"minADE": 0, "minFDE": 0, "OffRoad": 0, "SocViol": 0, "Acc": 0}
            
            with torch.no_grad():
                for data in val_loader:
                    m_img, e_h, e_dna, soc, msk, g_e, g_n = [d.to(device) for d in data]
                    e_trajs, e_probs, n_mu_gmm, t_idx = model(m_img, e_h, e_dna, soc, msk)
                    
                    # Call audited metrics code
                    batch_metrics = get_v6_metrics(e_probs, e_trajs, g_e, n_mu_gmm[..., :2], t_idx, msk, m_img)
                    
                    val_metrics["minADE"] += batch_metrics["minADE"]
                    val_metrics["minFDE"] += batch_metrics["minFDE"]
                    val_metrics["OffRoad"] += batch_metrics["OffRoadRate"]
                    val_metrics["SocViol"] += batch_metrics["SocialViolation"]
                    val_metrics["Acc"] += batch_metrics["Accuracy_Top1"]

            # Averaging Validation Metrics
            for k in val_metrics: val_metrics[k] /= len(val_loader)
            epoch_duration = time.time() - epoch_start

            # Log History for Plotting
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(total_train_loss / len(train_loader))
            history["val_ade"].append(val_metrics["minADE"])
            history["val_fde"].append(val_metrics["minFDE"])
            history["val_acc"].append(val_metrics["Acc"])
            history["time_per_epoch"].append(epoch_duration)

            # --- EPOCH LOG SUMMARY ---
            print(f"\n--- 🏁 Epoch {epoch+1} Completed in {epoch_duration:.1f}s ---")
            print(f"📈 Avg Loss: {history['train_loss'][-1]:.4f}")
            print(f"📊 Accuracy:  minADE: {val_metrics['minADE']:.2f}m | minFDE: {val_metrics['minFDE']:.2f}m")
            print(f"🛡️  Safety:    Off-Road {val_metrics['OffRoad']*100:.1f}% | Soc-Viol {val_metrics['SocViol']*100:.1f}%")
            print(f"🎯 Honesty:   Top-1 Confidence {val_metrics['Acc']*100:.1f}%")

            # Checkpointing (Save only the absolute best performer)
            if val_metrics['minADE'] < best_val_ade:
                best_val_ade = val_metrics['minADE']
                save_path = os.path.join(save_dir, "v6_best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'ade': best_val_ade,
                    'fde': val_metrics['minFDE']
                }, save_path)
                print(f"🌟 [BEST MODEL] New Best ADE Saved: {best_val_ade:.2f}m")
            
            # Save progress history to JSON
            with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
                json.dump(history, f, indent=4)
            print("-" * 50)

        total_time = (time.time() - start_total_time) / 60
        print(f"\n✅ TRAINING COMPLETE! Total Time: {total_time:.2f} minutes")

    except KeyboardInterrupt:
        print("\n🛑 Manual Stop. Saving checkpoint as 'weights/v6_interrupted.pth'...")
        torch.save(model.state_dict(), os.path.join(save_dir, "v6_interrupted.pth"))

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class CustomTrajectoryDataset(Dataset):
    def __init__(self, pkl_path, max_neighbors=30):
        print(f"[DATASET] Opening {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.max_neighbors = max_neighbors
        if not self.data:
            raise ValueError(f"No scenes found in dataset: {pkl_path}")
        self.future_steps = int(np.asarray(self.data[0]["primary_future"]).shape[0])
        self.social_dim = 17

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]

        map_array = np.asarray(scene["map_tensor"], dtype=np.float32)
        map_tensor = torch.from_numpy(map_array).permute(2, 0, 1)

        ego_hist = torch.from_numpy(np.asarray(scene["primary_history"], dtype=np.float32))
        ego_dna = torch.from_numpy(np.asarray(scene["primary_dna"], dtype=np.float32))

        raw_social = np.asarray(scene.get("social_graph", np.zeros((0, self.social_dim), dtype=np.float32)), dtype=np.float32)
        raw_neighbor_futures = np.asarray(
            scene.get("neighbor_futures", np.zeros((raw_social.shape[0], self.future_steps, 2), dtype=np.float32)),
            dtype=np.float32,
        )

        social_tensor = torch.zeros((self.max_neighbors, self.social_dim), dtype=torch.float32)
        gt_neighbors = torch.zeros((self.max_neighbors, self.future_steps, 2), dtype=torch.float32)
        mask = torch.ones((self.max_neighbors,), dtype=torch.bool)

        if raw_social.size > 0:
            actual_len = min(len(raw_social), self.max_neighbors)
            social_tensor[:actual_len] = torch.from_numpy(raw_social[:actual_len])
            gt_neighbors[:actual_len] = torch.from_numpy(raw_neighbor_futures[:actual_len])
            mask[:actual_len] = False

        gt_ego = torch.from_numpy(np.asarray(scene["primary_future"], dtype=np.float32))
        return map_tensor, ego_hist, ego_dna, social_tensor, mask, gt_ego, gt_neighbors

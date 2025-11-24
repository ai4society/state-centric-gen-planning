import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PlanningTrajectoryDataset(Dataset):
    def __init__(self, data_dir, domain, split="train"):
        """
        data_dir: root data dir (e.g., 'data/encodings/graphs')
        """
        self.files = []
        target_dir = os.path.join(data_dir, domain, split)

        # Find all trajectory files (exclude _goal.npy)
        if not os.path.exists(target_dir):
            print(f"Warning: Directory {target_dir} does not exist.")
            self.traj_files = []
        else:
            all_files = glob.glob(os.path.join(target_dir, "*.npy"))
            self.traj_files = [f for f in all_files if not f.endswith("_goal.npy")]

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, idx):
        traj_path = self.traj_files[idx]
        goal_path = traj_path.replace(".npy", "_goal.npy")

        # Load numpy arrays
        # Traj: [T, D]
        # Goal: [D]
        traj = np.load(traj_path).astype(np.float32)
        goal = np.load(goal_path).astype(np.float32)

        return torch.from_numpy(traj), torch.from_numpy(goal)


def collate_trajectories(batch):
    """
    Custom collate function to handle variable length trajectories.
    Returns:
        padded_trajs: [B, MaxT, D]
        goals: [B, D]
        lengths: [B]
    """
    trajs, goals = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in trajs])

    # Pad trajectories
    padded_trajs = torch.nn.utils.rnn.pad_sequence(trajs, batch_first=True)
    goals = torch.stack(goals)

    return padded_trajs, goals, lengths

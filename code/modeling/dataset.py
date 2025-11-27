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

    # def normalize(self, vec):
    #     """
    #     L1 Normalization: Converts counts to frequencies.
    #     Input: [..., D]
    #     Output: [..., D]
    #     """
    #     # Sum across the last dimension (features)
    #     sums = vec.sum(axis=-1, keepdims=True)
    #     # Avoid division by zero
    #     sums[sums == 0] = 1.0
    #     return vec / sums

    def __getitem__(self, idx):
        traj_path = self.traj_files[idx]
        goal_path = traj_path.replace(".npy", "_goal.npy")

        # Load numpy arrays
        # Traj: [T, D]
        # Goal: [D]
        traj = np.load(traj_path).astype(np.float32)
        goal = np.load(goal_path).astype(np.float32)

        # Ensure Goal is at least 1D [D]
        if goal.ndim == 0:
            goal = goal.reshape(1)

        # Ensure Traj is 2D [T, D]
        if traj.ndim == 1:
            # Ambiguity: Is it [T] (D=1) or [D] (T=1)?
            # We use Goal dimension D to decide.
            D = goal.shape[0]

            if traj.shape[0] == D:
                # Likely T=1, D=D
                traj = traj.reshape(1, D)
            else:
                # Likely T=T, D=1
                traj = traj.reshape(-1, 1)

        # traj = self.normalize(traj)
        # goal = self.normalize(goal)

        return torch.from_numpy(traj), torch.from_numpy(goal)


def collate_trajectories(batch):
    """
    Custom collate function to handle variable length trajectories.
    Pads sequences to the longest in the batch.
    Returns:
        padded_trajs: [B, MaxT, D]
        goals: [B, D]
        lengths: [B]
    """
    trajs, goals = zip(*batch)

    # Lengths for packing
    lengths = torch.tensor([t.shape[0] for t in trajs])

    # Pad trajectories: [B, MaxT, D]
    padded_trajs = torch.nn.utils.rnn.pad_sequence(trajs, batch_first=True)

    # Stack goals: [B, D]
    goals = torch.stack(goals)

    return padded_trajs, goals, lengths

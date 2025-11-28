import argparse
import os
from code.common.utils import set_seed, worker_init_fn
from code.modeling.dataset import PlanningTrajectoryDataset, collate_trajectories
from code.modeling.models import StateCentricLSTM

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, val_loader, device):
    """Computes Cosine loss on the validation set."""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for states, goals, lengths in val_loader:
            # Need at least 2 states to predict next state
            valid_mask = lengths > 1
            if not valid_mask.any():
                continue

            states = states[valid_mask]
            goals = goals[valid_mask]
            lengths = lengths[valid_mask]

            states = states.to(device)
            goals = goals.to(device)
            lengths = lengths.to(device)

            # Input: S_0 ... S_{T-1}
            input_states = states[:, :-1, :]

            # Target: S_1 ... S_T
            target_states = states[:, 1:, :]

            input_lengths = lengths - 1

            preds, _ = model(input_states, goals, input_lengths)

            # Create Boolean Mask [B, T-1]
            mask = (
                torch.arange(input_states.size(1), device=device)[None, :]
                < input_lengths[:, None]
            )

            # Flatten using the mask to get only valid steps
            # This avoids issues with CosineSimilarity on zero-padded vectors
            active_preds = preds[mask]
            active_targets = target_states[mask]

            # Cosine Loss: 1 - CosineSimilarity
            loss = (
                1.0 - F.cosine_similarity(active_preds, active_targets, dim=-1).mean()
            )

            total_loss += loss.item()
            count += 1

    if count == 0:
        print("No valid trajectories in validation set. Returning 0 loss.")
        return 0.0
    return total_loss / count


def train(args):
    set_seed(args.seed)

    # cuda -> metal -> cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    print(f"Loading datasets for {args.domain}...")
    train_ds = PlanningTrajectoryDataset(args.data_dir, args.domain, "train")
    val_ds = PlanningTrajectoryDataset(args.data_dir, args.domain, "validation")
    print(f"  Train Trajectories: {len(train_ds)} | Val Trajectories: {len(val_ds)}")

    if len(train_ds) == 0:
        print(f"Error: No training data found for {args.domain}. Skipping.")
        return

    # Use worker_init_fn and a generator
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_trajectories,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # Determine input dimension safely
    input_dim = 0
    # Check first few items
    for i in range(min(10, len(train_ds))):
        sample_traj, _ = train_ds[i]
        if sample_traj.dim() > 1:
            input_dim = sample_traj.shape[1]
            break

    if input_dim == 0:
        # Fallback
        sample_traj, _ = train_ds[0]
        input_dim = sample_traj.shape[-1]

    print(f"Feature Dimension: {input_dim}")

    # 2. Model
    model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    log_file = os.path.join(args.save_dir, f"{args.domain}_training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float("inf")

    print(f"Starting training on {device}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        count = 0

        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for states, goals, lengths in pbar:
            # Filter T=1
            valid_mask = lengths > 1
            if not valid_mask.any():
                continue

            states = states[valid_mask]
            goals = goals[valid_mask]
            lengths = lengths[valid_mask]

            states = states.to(device)
            goals = goals.to(device)
            lengths = lengths.to(device)

            # Prepare Inputs and Targets
            # Input: S_0 ... S_{T-1}
            # Target: S_1 ... S_T

            # We need to slice the padded sequences based on lengths
            # But simpler: just slice everything and mask loss later

            # Input sequence: remove last step
            # Input: S_0 ... S_{T-1}
            input_states = states[:, :-1, :]
            target_states = states[:, 1:, :]

            # Adjust lengths for the sliced sequence
            input_lengths = lengths - 1

            # Forward
            preds, _ = model(input_states, goals, input_lengths)

            # Masking padding for Loss
            # Create a mask [B, T-1, D]
            mask = (
                torch.arange(input_states.size(1), device=device)[None, :]
                < input_lengths[:, None]
            )

            # Flatten for loss calculation
            # preds: [B, T, D] -> [N, D]
            # targets: [B, T, D] -> [N, D]
            active_preds = preds[mask]

            # We predict the State directly
            active_targets = target_states[mask]

            # Cosine Embedding Loss
            # We want preds and targets to point in the same direction (target=1)
            # Loss = 1 - cos_sim(x, y)
            loss = (
                1.0 - F.cosine_similarity(active_preds, active_targets, dim=-1).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / count if count > 0 else 0

        # Validation Loop
        avg_val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.6f} | Val Loss {avg_val_loss:.6f}"
        )

        # Log
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{avg_train_loss},{avg_val_loss}\n")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"{args.domain}_lstm_best.pt"),
            )
            print(
                f"  -> Saved Best Model to {os.path.join(args.save_dir, f'{args.domain}_lstm_best.pt')}"
            )

        # Save Last Model (Checkpoint)
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"{args.domain}_lstm_last.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain name (e.g., blocks)")
    parser.add_argument("--data_dir", default="data/encodings/graphs")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    train(args)

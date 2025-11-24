import argparse
import os
from code.modeling.dataset import PlanningTrajectoryDataset, collate_trajectories
from code.modeling.models import StateCentricLSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, val_loader, criterion, device):
    """Computes loss on the validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for states, goals, lengths in val_loader:
            states, goals = states.to(device), goals.to(device)

            # Input: S_0 ... S_{T-1}
            input_states = states[:, :-1, :]
            # Target: S_1 ... S_T
            target_states = states[:, 1:, :]
            input_lengths = lengths - 1

            preds, _ = model(input_states, goals, input_lengths)

            # Masking
            mask = (
                torch.arange(input_states.size(1), device=device)[None, :]
                < input_lengths[:, None]
            )
            mask = mask.unsqueeze(-1).expand_as(preds)

            loss = criterion(preds * mask, target_states * mask)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    print(f"Loading datasets for {args.domain}...")
    train_ds = PlanningTrajectoryDataset(args.data_dir, args.domain, "train")
    val_ds = PlanningTrajectoryDataset(args.data_dir, args.domain, "validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_trajectories,
        num_workers=4,
    )

    # Determine input dimension
    if len(train_ds) == 0:
        raise ValueError(f"No training data found for {args.domain}")

    sample_traj, _ = train_ds[0]
    input_dim = sample_traj.shape[1]
    print(f"Feature Dimension: {input_dim}")

    # 2. Model
    model = StateCentricLSTM(input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Logging
    log_file = os.path.join(args.save_dir, f"{args.domain}_training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float("inf")

    print(f"Starting training on {device}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for states, goals, lengths in pbar:
            states, goals = states.to(device), goals.to(device)

            # Prepare Inputs and Targets
            # Input: S_0 ... S_{T-1}
            # Target: S_1 ... S_T

            # We need to slice the padded sequences based on lengths
            # But simpler: just slice everything and mask loss later

            # Input sequence: remove last step
            input_states = states[:, :-1, :]
            # Target sequence: remove first step
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
            mask = mask.unsqueeze(-1).expand_as(preds)

            loss = criterion(preds * mask, target_states * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

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
            print("  -> Saved Best Model")

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
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args)

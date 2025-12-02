import argparse
import os
import pickle
import time
from code.common.utils import set_seed
from code.modeling.dataset import load_flat_dataset_for_xgboost

import xgboost as xgb
from torch import cuda


def train(args):
    set_seed(args.seed)
    print(
        f"Training XGBoost using {'Delta Prediction' if args.delta else 'State Prediction'}"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading datasets for {args.domain}...")
    X_train, y_train = load_flat_dataset_for_xgboost(
        args.data_dir, args.domain, "train", delta=args.delta
    )
    X_val, y_val = load_flat_dataset_for_xgboost(
        args.data_dir, args.domain, "validation", delta=args.delta
    )

    if X_train is None:
        print(f"Error: No training data found for {args.domain}.")
        return

    print(f"  Train Data: X={X_train.shape}, y={y_train.shape}")
    if X_val is not None:
        print(f"  Val Data:   X={X_val.shape}, y={y_val.shape}")

    # 2. Configure XGBoost
    # Check for GPU
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("Configured hyperparameters:")
    print(f"  Boosting rounds (n_estimators): {args.n_estimators}\n")
    print(f"  Tree depth (max_depth): {args.max_depth}")
    print(f"  Learning rate: {args.lr}")

    # XGBRegressor automatically handles Multi-Output regression
    # if y is 2D and objective is squarederror.
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        tree_method="hist",  # Required for efficient training
        device=device,  # GPU support
        objective="reg:squarederror",
        n_jobs=8,
        random_state=args.seed,
    )

    # 3. Train
    start_time = time.time()

    eval_set = []
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(X_train, y_train, eval_set=eval_set if eval_set else None, verbose=True)

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")

    # 4. Save
    model_path = os.path.join(args.save_dir, f"{args.domain}_xgb.json")
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # Also save a small metadata file to know input dims during inference
    meta = {
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "delta": args.delta,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
    }
    with open(os.path.join(args.save_dir, f"{args.domain}_xgb_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--save_dir", required=True, help="Directory to save model")
    parser.add_argument("--data_dir", default="data/encodings/graphs")

    # XGB Hyperparams
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train(args)

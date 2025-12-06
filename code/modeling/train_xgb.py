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
        f"Training XGBoost using {'Delta Prediction' if args.delta else 'State Prediction'} with [{args.encoding}] encoding."
    )

    # Adjust data directory based on encoding if the user relied on default 'graphs' path
    # If the user explicitly passed a path with 'fsf' in it (via SLURM), this block is skipped.
    if args.encoding == "fsf" and "graphs" in args.data_dir:
        print(f"Switching data_dir from {args.data_dir} to fsf path...")
        args.data_dir = args.data_dir.replace("graphs", "fsf")

    # Construct save directory structure: checkpoints/<encoding>/xgboost_<mode>/
    # (Or rely on the user providing the correct --save_dir from the SLURM script)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading datasets for {args.domain} from {args.data_dir}...")
    
    # Debug: Check if path exists
    train_path_check = os.path.join(args.data_dir, args.domain, "train")
    if not os.path.exists(train_path_check):
        print(f"CRITICAL ERROR: Training path does not exist: {train_path_check}")
        return

    X_train, y_train = load_flat_dataset_for_xgboost(
        args.data_dir, args.domain, "train", delta=args.delta
    )
    X_val, y_val = load_flat_dataset_for_xgboost(
        args.data_dir, args.domain, "validation", delta=args.delta
    )

    if X_train is None:
        print(f"Error: No training data found for {args.domain} at {train_path_check}.")
        return

    print(f"  Train Data: X={X_train.shape}, y={y_train.shape}")
    if X_val is not None:
        print(f"  Val Data:   X={X_val.shape}, y={y_val.shape}")
    else:
        print(f"  Val Data:   None (Validation skipped)")

    # 2. Configure XGBoost
    # Check for GPU
    device = "cuda" if cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("Configured hyperparameters:")
    print(f"  Boosting rounds (n_estimators): {args.n_estimators}")
    print(f"  Tree depth (max_depth): {args.max_depth}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early Stopping Rounds: {args.early_stopping}")

    # Determine if we can use early stopping (requires validation data)
    es_rounds = args.early_stopping if X_val is not None else None

    # XGBRegressor automatically handles Multi-Output regression
    # if y is 2D and objective is squarederror.
    # early_stopping_rounds: this ensures that if validation score doesn't improve for N rounds, training stops.
    # The model object will automatically keep the best iteration's weights.
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        tree_method="hist",  # Required for efficient training
        device=device,  # GPU support
        objective="reg:squarederror",
        n_jobs=8,
        random_state=args.seed,
        early_stopping_rounds=es_rounds,
    )

    # 3. Train
    start_time = time.time()

    eval_set = []
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, 
        y_train, 
        eval_set=eval_set if eval_set else None, 
        verbose=True,
    )

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")

    # Check if early stopping was triggered
    if hasattr(model, 'best_iteration'):
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score}")

    # 4. Save
    # When early_stopping_rounds is used, save_model saves the trees up to the best iteration
    # (plus the patience window), but metadata marks the best iteration.
    model_path = os.path.join(args.save_dir, f"{args.domain}_xgb.json")
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # 5. Save Metadata
    meta = {
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "delta": args.delta,
        "encoding": args.encoding,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "best_iteration": getattr(model, 'best_iteration', -1)
    }
    with open(os.path.join(args.save_dir, f"{args.domain}_xgb_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--save_dir", required=True, help="Directory to save model")
    parser.add_argument("--data_dir", default="data/encodings/graphs")
    parser.add_argument(
        "--encoding",
        required=True,
        choices=["graphs", "fsf"],
        help="Encoding strategy used",
    )

    # XGB Hyperparams
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--early_stopping", type=int, default=10, help="Stop if val loss doesn't improve")
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train(args)

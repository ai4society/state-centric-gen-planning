import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def load_data(domain, split):
    path = f"data/encodings/graphs/{domain}/{split}/*.npy"
    files = glob.glob(path)
    data = []
    for f in files:
        if "_goal" in f:
            continue
        # Load and sum over time to get a "summary" of the problem instance
        arr = np.load(f)  # [T, D]
        # Take the initial state (index 0) to see starting distribution
        data.append(arr[0])
    return np.array(data)


def visualize(domain="blocks"):
    print(f"Loading data for {domain}...")
    train_data = load_data(domain, "train")
    test_data = load_data(domain, "test-extrapolation")

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")

    # Combine for PCA
    combined = np.concatenate([train_data, test_data], axis=0)

    # PCA to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    # Split back
    train_proj = projected[: len(train_data)]
    test_proj = projected[len(train_data) :]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        train_proj[:, 0], train_proj[:, 1], c="blue", alpha=0.5, label="Train (Small)"
    )
    plt.scatter(
        test_proj[:, 0], test_proj[:, 1], c="red", alpha=0.5, label="Test (Large)"
    )
    plt.title(f"WL Embedding Space: {domain}")
    plt.legend()
    plt.grid(True)
    # get dir of the script
    save_file = os.path.dirname(os.path.abspath(__file__)) + f"/{domain}_pca.png"
    plt.savefig(save_file)
    print(f"Saved plot to {save_file}")


if __name__ == "__main__":
    visualize()

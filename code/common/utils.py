import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for dictionary iteration consistency
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Global seed set to: {seed}")


def worker_init_fn(worker_id):
    """
    Function to ensure DataLoader workers are seeded deterministically.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

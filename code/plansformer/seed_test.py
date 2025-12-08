import random
import json
import os
import transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from torch import cuda
import argparse
from typing import (
    List,
)
import numpy as np

from pathlib import Path
import re
from code.common.utils import validate_plan


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)


def inference(
    model_path: str,
):
    seeds = [13, 14, 15]
    for seed in seeds:
        print(f"SEED: {seed}")
        set_seed(seed)
        device = "cuda" if cuda.is_available() else "cpu"
        tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        ).to(device)

        model.eval()
        inputs = ["daklfjlksd", "dasjfdsjjs", "kskskskskskksks"]
        for input in inputs:
            print(f"Input: {input}")
            for i in range(0, 10):
                input_ids = tokenizer.encode(input, return_tensors="pt").to(
                    device, dtype=torch.long
                )

                generated_ids = model.generate(
                    input_ids,
                    num_beams=2,
                    max_length=input_ids.shape[-1] + 2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=False,
                    do_sample=True,
                )
                predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                print(predicted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    # Convert to absolute
    args.model_path = str(Path(args.model_path).expanduser().resolve())

    inference(**vars(args))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_filename(filename: str):
    """
    Parses filenames to extract experiment metadata including the seed.
    Example: visitall-from-everywhere_fsf_validation_state_seed13_results.json
    """
    stem = filename.replace("_results.json", "")

    # Extract seed
    seed = None
    if "_seed" in stem:
        parts = stem.split("_seed")
        seed = int(parts[1])
        stem = parts[0]

    # Identify Split
    splits = ["validation", "test-interpolation", "test-extrapolation"]
    found_split = None
    for s in splits:
        if s in stem:
            found_split = s
            break

    if not found_split:
        return None

    # Split string: [Prefix]_[Split]_[Suffix]
    parts = stem.split(f"_{found_split}")
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""

    # Extract Domain and Encoding
    if prefix.endswith("_graphs"):
        encoding = "graphs"
        domain = prefix.replace("_graphs", "")
    elif prefix.endswith("_fsf"):
        encoding = "fsf"
        domain = prefix.replace("_fsf", "")
    else:
        encoding = "unknown"
        domain = prefix

    # Extract Mode
    mode = "state"
    if "_delta" in suffix:
        mode = "delta"

    return {
        "Domain": domain,
        "Encoding": encoding,
        "Split": found_split,
        "Mode": mode,
        "Seed": seed,
    }


def aggregate_seeds(results_dir):
    path = Path(results_dir)
    json_files = list(path.rglob("*_results.json"))

    records = []
    for jf in json_files:
        meta = parse_filename(jf.name)
        if not meta:
            continue

        # Infer model from parent directory
        parent_dir = jf.parent.name.lower()
        if "lstm" in parent_dir:
            model = "LSTM"
        elif "xgboost" in parent_dir or "xgb" in parent_dir:
            model = "XGB"
        else:
            model = "Unknown"

        try:
            with open(jf, "r") as f:
                results = json.load(f)

            total = len(results)
            if total == 0:
                continue

            # Count solved based on VAL
            solved = sum(1 for r in results if r.get("val_solved", False))
            coverage = solved / total

            meta["Model"] = model
            meta["Coverage"] = float(coverage)
            records.append(meta)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Generalized Planning Results across Seeds"
    )
    parser.add_argument(
        "--results_dir", default="results", help="Path to results root."
    )
    args = parser.parse_args()

    df = aggregate_seeds(args.results_dir)
    if df.empty:
        print("No results found.")
        return

    # Calculate Mean and Std (Population Std Dev ddof=0 to match Excel's STDEV.P)
    grouped = (
        df.groupby(["Domain", "Split", "Encoding", "Model", "Mode"])["Coverage"]
        .agg(Mean="mean", Std=lambda x: np.std(x, ddof=0))
        .reset_index()
    )

    # Format as "Mean ± Std"
    grouped["Formatted"] = grouped.apply(
        lambda row: f"{row['Mean']:.2f} ± {row['Std']:.2f}", axis=1
    )

    # Create Config column for pivot (e.g., "WL-LSTM State")
    enc_map = {"graphs": "WL", "fsf": "FSF"}
    grouped["Config"] = (
        grouped["Encoding"].map(enc_map)
        + "-"
        + grouped["Model"]
        + " "
        + grouped["Mode"].str.capitalize()
    )

    pivot_df = grouped.pivot_table(
        index=["Domain", "Split"], columns="Config", values="Formatted", aggfunc="first"
    ).fillna("0.00 ± 0.00")

    # Define desired column order
    cols_order = [
        "WL-LSTM State",
        "WL-LSTM Delta",
        "WL-XGB State",
        "WL-XGB Delta",
        "FSF-LSTM State",
        "FSF-LSTM Delta",
        "FSF-XGB State",
        "FSF-XGB Delta",
    ]
    # Filter to only columns that exist in the data
    cols_order = [c for c in cols_order if c in pivot_df.columns]
    pivot_df = pivot_df[cols_order]

    # Define desired row order
    domain_order = ["blocks", "gripper", "visitall-from-everywhere", "logistics"]
    split_order = ["validation", "test-interpolation", "test-extrapolation"]

    # Reorder rows by sorting explicit categorical columns (robust for MultiIndex)
    pivot_df = pivot_df.reset_index()
    pivot_df["Domain"] = pd.Categorical(
        pivot_df["Domain"], categories=domain_order, ordered=True
    )
    pivot_df["Split"] = pd.Categorical(
        pivot_df["Split"], categories=split_order, ordered=True
    )
    pivot_df = pivot_df.sort_values(["Domain", "Split"]).set_index(["Domain", "Split"])

    # Rename indices for a cleaner look
    domain_rename = {
        "blocks": "Blocks",
        "gripper": "Gripper",
        "visitall-from-everywhere": "VisitAll",
        "logistics": "Logistics",
    }
    split_rename = {
        "validation": "Val.",
        "test-interpolation": "Interp.",
        "test-extrapolation": "Extrap.",
    }

    pivot_df.rename(index=domain_rename, level=0, inplace=True)
    pivot_df.rename(index=split_rename, level=1, inplace=True)

    print("\n=== Aggregated Results (Mean ± Std) ===\n")
    print(pivot_df.to_markdown())


if __name__ == "__main__":
    main()

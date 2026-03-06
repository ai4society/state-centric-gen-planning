import argparse
import json
from pathlib import Path

import pandas as pd


def parse_filename(filename: str) -> dict[str, str | float] | None:
    """
    Parses filenames to extract experiment metadata.
    Handles: {domain}_{encoding}_{split}_{tag}_{seed}_results.json
    """
    # Remove extension
    stem = filename.replace("_results.json", "")

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
    prefix = parts[0]  # domain_encoding
    suffix = parts[1] if len(parts) > 1 else ""  # _tag

    # 1. Extract Domain and Encoding
    # Heuristic: Encoding is usually 'graphs' or 'fsf' at the end of prefix
    if prefix.endswith("_graphs"):
        encoding = "graphs"
        domain = prefix.replace("_graphs", "")
    elif prefix.endswith("_fsf"):
        encoding = "fsf"
        domain = prefix.replace("_fsf", "")
    else:
        # Fallback or unknown encoding
        encoding = "unknown"
        domain = prefix

    # 2. Extract Mode from Suffix
    mode = "state"
    if "delta" in suffix:
        mode = "delta"

    return {"Domain": domain, "Encoding": encoding, "Split": found_split, "Mode": mode}


def aggregate(results_dir):
    data_records = []

    path = Path(results_dir)

    # Recursively find all result JSONs
    json_files = list(path.rglob("*_results.json"))

    print(f"Found {len(json_files)} result files in `{results_dir}`")

    for i, jf in enumerate(json_files):
        meta = parse_filename(jf.name)
        if not meta:
            print(f"Skipping {jf.name} (could not parse filename structure)")
            continue

        # Determine Model from directory structure if possible, else assume from path
        # Structure: results/<encoding>/<model>_<mode>/...
        # We can try to infer model from parent dir name
        parent_dir = jf.parent.name.lower()  # e.g., "lstm_delta" or "xgboost_state"
        if "lstm" in parent_dir:
            model = "LSTM"
        elif "xgboost" in parent_dir or "xgb" in parent_dir:
            model = "XGB"
        else:
            model = "Unknown"
        print(
            f"Processing {i + 1}/{len(json_files)}: {jf.name} | Model: {model}, Domain: {meta['Domain']}, Encoding: {meta['Encoding']}, Split: {meta['Split']}, Mode: {meta['Mode']}"
        )

        try:
            with open(jf, "r") as f:
                results = json.load(f)

            total = len(results)
            if total == 0:
                continue

            # Count solved based on VAL
            solved = sum(1 for r in results if r.get("val_solved", False))

            # Calculate Coverage
            coverage = solved / total

            record = meta.copy()
            record["Model"] = model
            record["Coverage"] = float(coverage)
            data_records.append(record)

        except Exception as e:
            print(f"Error reading {jf}: {e}")

    return pd.DataFrame(data_records)


def format_mean_std(x):
    if len(x) == 0:
        return "-"
    if len(x) == 1:
        return f"{x.mean():.2f}"
    return f"{x.mean():.2f} ± {x.std():.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Generalized Planning Results"
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Path to results root. Default: `results`",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument(
        "--output", help="Path to save the output file. Default: None", default=None
    )
    args = parser.parse_args()

    df = aggregate(args.results_dir)

    if df.empty:
        print("No results found.")
        return

    # Pivot for better readability (Rows: Domain/Split, Cols: Model Config)
    # Create a configuration column
    df["Config"] = df["Encoding"] + "-" + df["Model"] + "-" + df["Mode"]

    pivot_df = df.pivot_table(
        index=["Domain", "Split"],
        columns="Config",
        values="Coverage",
        aggfunc=format_mean_std,
    )

    # Sort index to match paper order usually
    pivot_df = pivot_df.sort_index(level=0)

    # Generate String Representation
    if args.format == "markdown":
        output_str = pivot_df.to_markdown()
    elif args.format == "latex":
        output_str = pivot_df.to_latex()
    else:
        output_str = pivot_df.to_csv()

    print("\n=== Aggregated Results (Mean ± Std) ===\n")
    print(output_str)

    # Save to File
    if args.output:
        # Ensure directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(output_str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path


def lowercase_pddl_files(data_dir):
    root_path = Path(data_dir)
    if not root_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    modified_count = 0
    skipped_count = 0

    print(f"Scanning {root_path} for .pddl files...")

    # rglob recursively finds all files matching the pattern
    for pddl_file in root_path.rglob("*.pddl"):
        try:
            # Read content
            original_content = pddl_file.read_text(encoding="utf-8")

            # Convert to lowercase
            lower_content = original_content.lower()

            # Only write if there's a change to avoid touching file timestamps unnecessarily
            if original_content != lower_content:
                pddl_file.write_text(lower_content, encoding="utf-8")
                modified_count += 1
                print(f"Lowercased: {pddl_file.relative_to(root_path)}")
            else:
                skipped_count += 1

        except Exception as e:
            print(f"Failed to process {pddl_file}: {e}")

    print("-" * 30)
    print("Processing Complete.")
    print(f"Files modified: {modified_count}")
    print(f"Files already lowercase: {skipped_count}")
    print(f"Total files scanned: {modified_count + skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all PDDL files to lowercase.")
    parser.add_argument(
        "--data_dir", default="data/pddl", help="Root directory containing PDDL files"
    )
    args = parser.parse_args()

    lowercase_pddl_files(args.data_dir)

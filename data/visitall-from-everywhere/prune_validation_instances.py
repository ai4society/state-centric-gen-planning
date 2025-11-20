import os
import re

# Directory containing the validation instances
validation_dir = "./validation"

# Pattern to match the final number after the hyphen: w{width}h{height}-{number}
final_number_pattern = re.compile(r'-(\d+)\.pddl$')

deleted_count = 0
total_count = 0

# List all PDDL files in the validation directory
for filename in os.listdir(validation_dir):
    if filename.endswith('.pddl'):
        total_count += 1
        filepath = os.path.join(validation_dir, filename)

        # Extract the final number from filename
        match = final_number_pattern.search(filename)
        if match:
            final_number = int(match.group(1))

            # Delete if final number >= 4
            if final_number >= 4:
                print(f"Deleting: {filename} (final number: {final_number})")
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filename} (final number: {final_number})")
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
        else:
            print(f"Warning: Could not determine final number from filename {filename}")

print(f"\nSummary: Deleted {deleted_count} out of {total_count} validation instances.")
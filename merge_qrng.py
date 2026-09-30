files_to_merge = [
    "qrng_numbers.txt", 
    "qrng_numbers_2.txt",
    "qrng_numbers_3.txt",
    "qrng_numbers_4.txt",
    "qrng_numbers_5.txt",
    "qrng_numbers_6.txt",
    "qrng_numbers_7.txt",
    "qrng_numbers_8.txt"
]

# We will read the original qrng_combined.txt as well
original_file = "qrng_combined.txt"

all_numbers = []

import os
import re

# 1. First, load the original 197k numbers
if os.path.exists(original_file):
    with open(original_file, "r") as f:
        # Original is comma-separated
        nums = [x.strip() for x in f.read().split(",") if x.strip()]
        # Because we accidentally merged the giant blocks earlier, we need to clean them out.
        # We'll only keep valid numeric strings that don't have newlines in them.
        clean_nums = [n for n in nums if "\n" not in n]
        all_numbers.extend(clean_nums)
    print(f"Loaded {len(clean_nums)} clean numbers from original {original_file}")

# 2. Now load the 8 new files which are newline-separated
for filename in files_to_merge:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            # Split by whitespace/newlines
            nums = f.read().split()
            # Keep only valid digits
            nums = [n for n in nums if n.isdigit()]
            all_numbers.extend(nums)
        print(f"Loaded {len(nums)} numbers from {filename}")
    else:
        print(f"File not found, skipping: {filename}")

# 3. Write them ALL back out as a single comma-separated string
if all_numbers:
    final_string = ", ".join(all_numbers)
    with open(original_file, "w") as f_out:
        f_out.write(final_string)
    
    print(f"\nSUCCESS! Overwrote {original_file}.")
    print(f"Total numbers now in the master dataset: {len(all_numbers):,}")
else:
    print("No data found to merge.")

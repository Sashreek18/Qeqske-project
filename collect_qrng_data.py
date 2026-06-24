"""
ANU QRNG Collector
====================
Automates calling the ANU Quantum Random Number Generator API
(https://api.quantumnumbers.anu.edu.au) repeatedly to build a large dataset.

ANU API LIMIT: max length=1024 per request (uint16 type).
This script loops, making multiple requests, and saves everything to a file
so we can do proper Attack A/B/D analysis (need thousands-millions of numbers,
not just 1024).

USAGE:
    python3 collect_qrng_data.py --api-key YOUR_KEY --batches 50 --output qrng_dataset.txt

This will make 50 requests x 1024 numbers = 51,200 numbers total.
"""

import argparse
import time
import json
import urllib.request
import urllib.error


ANU_API_URL = "https://api.quantumnumbers.anu.edu.au"
MAX_LENGTH_PER_REQUEST = 1024  # ANU API hard limit


def fetch_batch(api_key: str, length: int = MAX_LENGTH_PER_REQUEST, dtype: str = "uint16"):
    """
    Calls the ANU QRNG API once and returns a list of integers.
    Equivalent to:
        curl -H "x-api-key: YOUR_API_KEY" \
        "https://api.quantumnumbers.anu.edu.au?length=10&type=uint16"
    """
    if length > MAX_LENGTH_PER_REQUEST:
        raise ValueError(f"ANU API max length is {MAX_LENGTH_PER_REQUEST}, got {length}")

    url = f"{ANU_API_URL}?length={length}&type={dtype}"
    req = urllib.request.Request(url, headers={"x-api-key": api_key})

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ANU API error {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error reaching ANU API: {e.reason}")

    if not data.get("success", False):
        raise RuntimeError(f"ANU API returned failure: {data}")

    return data["data"]


def collect(api_key: str, num_batches: int, output_file: str,
            dtype: str = "uint16", delay_seconds: float = 1.0):
    """
    Loops num_batches times, calling the API each time, and appends
    results to output_file. delay_seconds avoids hammering the API
    (be a good citizen / avoid rate limits).
    """
    all_numbers = []

    for batch_num in range(1, num_batches + 1):
        print(f"Fetching batch {batch_num}/{num_batches}...", end=" ")
        try:
            batch = fetch_batch(api_key, length=MAX_LENGTH_PER_REQUEST, dtype=dtype)
            all_numbers.extend(batch)
            print(f"got {len(batch)} numbers (total so far: {len(all_numbers)})")
        except RuntimeError as e:
            print(f"FAILED: {e}")
            print("Stopping collection early. Numbers gathered so far will still be saved.")
            break

        if batch_num < num_batches:
            time.sleep(delay_seconds)  # be polite to the API

    # Save to file (comma-separated, same format as your terminal output)
    with open(output_file, "w") as f:
        f.write(", ".join(str(n) for n in all_numbers))

    print(f"\nSaved {len(all_numbers)} total QRNG numbers to '{output_file}'")
    return all_numbers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect QRNG data from ANU API in bulk")
    parser.add_argument("--api-key", required=True, help="Your ANU QRNG API key")
    parser.add_argument("--batches", type=int, default=10,
                         help="Number of API calls to make (each gets 1024 numbers)")
    parser.add_argument("--output", default="qrng_dataset.txt",
                         help="Output file to save numbers to")
    parser.add_argument("--type", default="uint16", choices=["uint8", "uint16", "hex16"],
                         help="ANU API number type")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to wait between API calls")
    args = parser.parse_args()

    collect(args.api_key, args.batches, args.output, args.type, args.delay)

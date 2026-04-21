import subprocess
import csv
import os
import sys

def test_experiment_correct_matches():
    # Define the command and output file
    output_csv = "matches.csv"
    cmd = [
        sys.executable, "run.py",
        "LBP_Test_Images/Swatches/ValidationSet_rotated",
        "--X", "256",
        "--Y", "256",
        "--V",
        "--P", "8",
        "--R", "1",
        "--metric", "hellinger",
        "--method", "ror",
        "--blur", "1.0",
        "--visualize",
        "--save-csv", output_csv,
        "--hist-smooth", "0.000001",
        "--ltp-threshold", "3",
        "--simulate-noise",
        "--noise-sigma", "10"
    ]

    # Run the experiment
    subprocess.run(cmd, check=True)

    # Read the output CSV and calculate correct percentage
    with open(output_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        total = 0
        correct = 0
        for row in reader:
            total += 1
            if row.get("CORRECT", "0") in ("1", "True", "true", "yes"):
                correct += 1
    if total == 0:
        raise AssertionError("No results found in matches.csv")
    percent_correct = 100.0 * correct / total
    print(f"Correct matches: {correct}/{total} ({percent_correct:.2f}%)")
    assert percent_correct > 90.0, f"Correct match percentage too low: {percent_correct:.2f}%"

    # Clean up
    os.remove(output_csv)

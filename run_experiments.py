
import itertools
import os
import csv
from datetime import datetime

# Define the arguments and their possible values for the experiments
EXPERIMENTS = {
    '--P': [8],
    '--R': [1],
    '--method': ['ror'],
    '--metric': ['hellinger'],
    '--hist-smooth': [0.000001],
    '--blur': [1.0],
    '--ltp-threshold': [3],
    '--crop-seed': [12345, 54321, 65432, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 
                    654, 321, 123, 987, 876, 765, 4321, 5432, 6789, 7890, 13579, 24680, 11223, 44556, 77889, 99887, 77665, 55443, 33221, 11009,
                    9999, 8888, 7777, 6666, 5555, 4444, 3333, 2222, 1111, 1234, 4321, 5678, 8765, 1357, 2468, 1122, 4455, 7788, 9988, 7766,
                    5544, 3322, 1100, 999, 888, 777, 666, 555, 444, 333, 222, 111, 123, 321, 567, 765, 135, 246, 112, 445, 778, 998,
                    776, 554, 332, 110, 99, 88, 77, 66, 55, 44, 33, 22, 11, 1234, 4321, 5678, 8765, 1357, 2468, 1122, 4455, 7788, 9988, 7766,
                    5544, 3322, 1100, 999, 888, 777, 666, 555, 444, 333, 222, 111, 123, 321, 567, 765, 135, 246, 112, 445, 778, 998,
                    776, 554, 332, 110, 99, 88, 77, 66, 55, 44, 33, 22, 11],
}

# Fixed arguments
FOLDER = r"LBP_Test_Images"
#FOLDER = r"LBP_Test_Images/Swatches/LBP_Texture_Swatches"
X = 64
Y = 64

# Output log file
LOG_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Prepare all combinations
keys = list(EXPERIMENTS.keys())
values = list(EXPERIMENTS.values())
combinations = list(itertools.product(*values))

with open(log_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = keys + [
        'correct_matches', 'total', 'pct_correct',
        'highest_correct', 'lowest_correct', 'average_correct',
        'highest_incorrect', 'lowest_incorrect', 'average_incorrect',
        'csv_file', 'results_json_file'
    ]
    writer.writerow(header)
    # Try to import the main entry point for direct results
    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
        from automated_lbp_benchmarking.main import main as lbp_main
        use_direct = True
    except Exception as import_exc:
        print(f"[WARNING] Could not import main programmatically, falling back to subprocess: {import_exc}")
        use_direct = False

    for combo in combinations:
        args = []
        for k, v in zip(keys, combo):
            if isinstance(v, bool):
                if v:
                    args.append(k)
            else:
                args.extend([k, str(v)])
        args.extend(["--X", str(X), "--Y", str(Y)])
        csv_name = f"exp_{'_'.join(str(x) for x in combo)}.csv"
        args.append(f"--save-csv={csv_name}")
        args = [str(a) for a in args]
        results_json_file = os.path.join(LOG_DIR, f"{os.path.splitext(csv_name)[0]}_results.json")
        row = list(combo)
        if use_direct:
            # Use direct Python call
            cli_args = [FOLDER] + args
            try:
                import os
                os.environ["LBP_EXPERIMENT_MODE"] = "1"
                results = lbp_main(return_results=True, cli_args=cli_args)
                row += [
                    results.get('correct_matches'),
                    results.get('total'),
                    results.get('pct_correct'),
                    results.get('highest_correct'),
                    results.get('lowest_correct'),
                    results.get('average_correct'),
                    results.get('highest_incorrect'),
                    results.get('lowest_incorrect'),
                    results.get('average_incorrect'),
                    csv_name,
                    os.path.basename(results_json_file)
                ]
            except Exception as e:
                print(f"Experiment failed (direct): {e}")
                row += ["ERROR"] * 8 + [csv_name, ""]
        else:
            # Fallback to subprocess
            import subprocess
            import json
            cmd = ["python", "run.py", FOLDER] + args
            print(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if os.path.exists(results_json_file):
                    with open(results_json_file, 'r', encoding='utf-8') as jf:
                        results = json.load(jf)
                    row += [
                        results.get('correct_matches'),
                        results.get('total'),
                        results.get('pct_correct'),
                        results.get('highest_correct'),
                        results.get('lowest_correct'),
                        results.get('average_correct'),
                        results.get('highest_incorrect'),
                        results.get('lowest_incorrect'),
                        results.get('average_incorrect'),
                        csv_name,
                        os.path.basename(results_json_file)
                    ]
                else:
                    row += ["ERROR"] * 10 + [csv_name, ""]
            except Exception as e:
                print(f"Experiment failed (subprocess): {e}")
                row += ["ERROR"] * 8 + [csv_name, ""]
        writer.writerow(row)
        f.flush()
print(f"All experiments complete. Log saved to {log_file}")

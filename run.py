
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from automated_lbp_benchmarking.main import main

if __name__ == "__main__":
    results = main(return_results=True, cli_args=sys.argv[1:])
    print(results)
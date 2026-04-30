from PIL import Image
import csv
from pathlib import Path
from typing import Sequence
from .image_data_containers import ImageRecord
from datetime import datetime

def save_matches_csv(image_records: Sequence[ImageRecord], out_path: str = "matches") -> None:
    # Find root project director (hardcoded, probably shouldn't do this)
    project_root = Path(__file__).resolve().parents[2]
    
    # Build results directory path
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Use the provided out_path as the filename, only add timestamp if not specified
    if out_path.endswith('.csv'):
        output_path = results_dir / out_path
    else:
        # If not a csv, add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"{out_path}_{timestamp}.csv"
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query_Index", "Query_Instance", "Query_Category", "Query_Distance", 
            "Query_Rotation", "Query_Lighting", "Matched_Category", "Matched_Index", "NN_Distance", "Correct"
        ])
        
        for query_idx, img_record in enumerate(image_records):
            for match_record in img_record.match_records:
                writer.writerow([
                    query_idx + 1,
                    img_record.instance,
                    img_record.category,
                    img_record.distance,
                    img_record.rotation,
                    img_record.lighting,
                    match_record.matched_category,
                    match_record.matched_index,
                    match_record.nn_distance,
                    match_record.correct,
                ])
                
def generate_config_filename(config_dict: dict) -> str:
    filename_parts = []
    
    # Pattern type (LBP or LTP)
    if config_dict["local_binary_patterns"]["use_ltp"]:
        filename_parts.append("ltp")
    else:
        filename_parts.append("lbp")
    
    # Method
    method = config_dict["local_binary_patterns"]["method"]
    filename_parts.append(method)
    
    # P and R parameters
    p = config_dict["local_binary_patterns"]["P"]
    r = config_dict["local_binary_patterns"]["R"]
    filename_parts.append(f"p{p}_r{int(r) if r == int(r) else r}")
    
    # Preprocessing parameters
    preprocessing = config_dict["preprocessing"]
    if preprocessing["gaussian_noise"] is not None:
        noise_val = int(preprocessing["gaussian_noise"] * 100)
        filename_parts.append(f"noise{noise_val}")
    
    if preprocessing["gaussian_blur"] is not None:
        blur_val = int(preprocessing["gaussian_blur"] * 10)
        filename_parts.append(f"blur{blur_val}")
    
    if preprocessing["illumination"] is not None:
        illum_val = int(preprocessing["illumination"] * 100)
        filename_parts.append(f"illum{illum_val}")
    
    if preprocessing["contrast"] is not None:
        contrast_val = int(preprocessing["contrast"] * 100)
        filename_parts.append(f"contrast{contrast_val}")
    
    # Matching parameters
    metric = config_dict["matching"]["metric"]
    filename_parts.append(f"{metric}")
    
    return "_".join(filename_parts)


def print_verbose_report(records: Sequence[ImageRecord]) -> None:
    total = len(records)
    correct_count = 0
    all_correct_distances = []
    all_incorrect_distances = []
    lowest_correct = float("inf")
    lowest_incorrect = float("inf")
    highest_correct = float("-inf")
    highest_incorrect = float("-inf")

    for r in records:
        # Use all shown matches if available (tolerance/top), else fallback to nn_distance
        if hasattr(r, "matching_distances") and r.matching_distances:
            # Try to get categories for each match
            match_cats = getattr(r, "matching_categories", [])
            for idx, dist in enumerate(r.matching_distances):
                # Determine if this match is correct
                cat = match_cats[idx] if idx < len(match_cats) else None
                is_correct = (cat == r.category)
                if is_correct:
                    all_correct_distances.append(dist)
                    highest_correct = max(highest_correct, dist)
                    lowest_correct = min(lowest_correct, dist)
                    correct_count += 1
                else:
                    all_incorrect_distances.append(dist)
                    highest_incorrect = max(highest_incorrect, dist)
                    lowest_incorrect = min(lowest_incorrect, dist)
        elif r.nn_distance is not None:
            # Fallback: single match
            if bool(r.correct):
                all_correct_distances.append(r.nn_distance)
                highest_correct = max(highest_correct, r.nn_distance)
                lowest_correct = min(lowest_correct, r.nn_distance)
                correct_count += 1
            else:
                all_incorrect_distances.append(r.nn_distance)
                highest_incorrect = max(highest_incorrect, r.nn_distance)
                lowest_incorrect = min(lowest_incorrect, r.nn_distance)

    avg_correct = float(np.mean(all_correct_distances)) if all_correct_distances else None
    avg_incorrect = float(np.mean(all_incorrect_distances)) if all_incorrect_distances else None

    summary_lines = []
    pct = 100.0 * correct_count / (len(all_correct_distances) + len(all_incorrect_distances)) if (len(all_correct_distances) + len(all_incorrect_distances)) else 0.0
    summary_lines.append(f"Correct matches: {correct_count}/{len(all_correct_distances) + len(all_incorrect_distances)} ({pct:.2f}%)")
    summary_lines.append(f"Highest distance among correct matches: {highest_correct:.6f}" if highest_correct != float("-inf") else
          "Highest distance among correct matches: N/A")
    summary_lines.append(f"Lowest distance among correct matches: {lowest_correct:.6f}" if lowest_correct != float("inf") else
          "Lowest distance among correct matches: N/A")
    summary_lines.append(f"Average distance among correct matches: {avg_correct:.6f}" if avg_correct is not None else "Average distance among correct matches: N/A")
    summary_lines.append(f"Highest distance among incorrect matches: {highest_incorrect:.6f}" if highest_incorrect != float("-inf") else
          "Highest distance among incorrect matches: N/A")
    summary_lines.append(f"Lowest distance among incorrect matches: {lowest_incorrect:.6f}" if lowest_incorrect != float("inf") else
          "Lowest distance among incorrect matches: N/A")
    summary_lines.append(f"Average distance among incorrect matches: {avg_incorrect:.6f}" if avg_incorrect is not None else "Average distance among incorrect matches: N/A")

    for line in summary_lines:
        print(line)

    # Return a standardized results dict for automation
    results = {
        "correct_matches": int(correct_count),
        "total": int(len(all_correct_distances) + len(all_incorrect_distances)),
        "pct_correct": float(pct),
        "highest_correct": float(highest_correct) if highest_correct != float("-inf") else None,
        "lowest_correct": float(lowest_correct) if lowest_correct != float("inf") else None,
        "average_correct": avg_correct,
        "highest_incorrect": float(highest_incorrect) if highest_incorrect != float("-inf") else None,
        "lowest_incorrect": float(lowest_incorrect) if lowest_incorrect != float("inf") else None,
        "average_incorrect": avg_incorrect,
    }
    return summary_lines, results
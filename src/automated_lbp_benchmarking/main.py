from __future__ import annotations

import argparse
import yaml
import pdb
import traceback

from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
from .image_file_handling import get_images_in_folder_as_image_records
from skimage.feature import local_binary_pattern as skimage_lbp
from .visualize_matches_mvc import visualize_matches
from .local_binary_pattern_processing import LBPResult, LTPResult, local_binary_pattern, local_ternary_pattern
from .local_binary_pattern_processing import local_ternary_pattern, LTPResult
from datetime import datetime
from .image_processing import apply_processing
from skimage.color import rgb2gray
from .processed_to_raw_image_matching import ProcessedToRawMatcher
from .match_statistics import compute_match_distance_stats
from .visualization import visualize_image_records
from .save_visualization_as_pdf import create_image_record_match_pdf

def main(return_results, cli_args=None) -> Optional[dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )

    args = parser.parse_args(cli_args)
    config_dict = {}
    with open(args.config, "r") as config:
        config_dict = yaml.safe_load(config)
    rng = np.random.default_rng(seed=config_dict["rng"]["seed"])

    images = config_dict["data"]["folder"]
    raw_image_records = get_images_in_folder_as_image_records(images)
    working_image_records = get_images_in_folder_as_image_records(images)

    p = config_dict["local_binary_patterns"]["P"]
    r = config_dict["local_binary_patterns"]["R"]
    method = config_dict["local_binary_patterns"]["method"]
    use_ltp = config_dict["local_binary_patterns"]["use_ltp"]
    ltp_threshold = config_dict["local_binary_patterns"]["ltp"]["threshold"]
    for record in raw_image_records:
        gray_image = record.image.convert("L")
        image_array = np.asarray(gray_image, dtype=np.uint8)
        if use_ltp:
            ltp_result: LTPResult = local_ternary_pattern(image_array, p=p, r=r, method=method, threshold=ltp_threshold)
            record.lbp_hist = ltp_result.histogram
        else:
            lbp_result: LBPResult = local_binary_pattern(image_array, p=p, r=r, method=method)
            record.lbp_hist = lbp_result.histogram

    for record in working_image_records:
        processed_image = record.image
        processed_image = apply_processing(processed_image, processing_args=config_dict, rng=rng)
        gray_image = rgb2gray(processed_image)
        gray_image = (gray_image * 255).astype(np.uint8)  # Convert to uint8 format expected by LBP functions   
        if use_ltp:
            ltp_result: LTPResult = local_ternary_pattern(gray_image, p=p, r=r, method=method, threshold=ltp_threshold)
            record.lbp_hist = ltp_result.histogram
        else:
            lbp_result: LBPResult = local_binary_pattern(gray_image, p=p, r=r, method=method)
            record.lbp_hist = lbp_result.histogram
        record.image = Image.fromarray(processed_image)
        
    distance_metric = config_dict["matching"]["metric"]
    match_tolerance = config_dict["matching"]["tolerance"]
    top_k = config_dict["matching"]["top"]
    matcher = ProcessedToRawMatcher(metric_name=distance_metric, tolerance=match_tolerance, top=top_k)
    processed_matched_records = matcher(working_image_records, raw_image_records)

    create_image_record_match_pdf(
        image_records=processed_matched_records,
        output_path="image_record_matches.pdf",
        records_per_page=5,
        matches_per_row=top_k,
    )
    visualize_image_records(processed_matched_records, 50)
    stats = compute_match_distance_stats(processed_matched_records)
    print(stats)
    # if config_dict.get["output"]["save_csv"]:


    # Always compute results_json and summary_lines
    summary_lines, results_json = print_verbose_report(records)

    # Always write a standardized results JSON for automation, with a name matching the CSV if --save-csv is used
    import json
    results_path = None
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    if args.save_csv:
        results_path = results_dir / (Path(args.save_csv).stem + "_results.json")
    else:
        results_path = results_dir / ("results_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    if not running_experiments:
        try:
            print(f"[DEBUG] Attempting to write results JSON to: {results_path.resolve()}")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results_json, f, indent=2)
            print(f"Saved results JSON to {results_path.resolve()}")
        except Exception as e:
            print(f"[ERROR] Failed to write results JSON to {results_path.resolve()}: {e}")

    if return_results:
        return results_json

    if args.visualize:
        items_for_viz = []
        for r in processed_records:
            matched_img = None
            if r.matched_index is not None and 0 <= r.matched_index < len(raw_records):
                matched_img = raw_records[r.matched_index].image
                # Resize matched image to match processed image size for display
                if matched_img.size != r.image.size:
                    matched_img = matched_img.resize(r.image.size, Image.BILINEAR)
            items_for_viz.append({
                "INSTANCE": r.instance,
                "CATEGORY": r.category,
                "DISTANCE": r.nn_distance,
                "ROTATION": r.rotation,
                "LIGHTING": r.lighting,
                "LBP": r.lbp_hist,
                "IMAGE": r.image,  # preprocessed image (left)
                "MATCHED_IMAGE": matched_img,  # matched raw image (right, resized)
                "INDEX": r.index,
                "MATCHED_INDEX": r.matched_index,
                "MATCHED_CATEGORY": r.matched_category,
                "MATCHED_INDICES": getattr(r, "matching_indices", []),
                "MATCHED_DISTANCES": getattr(r, "matching_distances", []),
                "MATCHED_CATEGORIES": getattr(r, "matching_categories", []),
                "CORRECT": r.correct,
            })
        args_dict = vars(args)
        save_pdf = args.save_csv is not None
        visualize_matches(items_for_viz, metric_name=args.metric, args=args_dict, summary_lines=summary_lines, save_pdf=save_pdf)


if __name__ == "__main__":
    main()

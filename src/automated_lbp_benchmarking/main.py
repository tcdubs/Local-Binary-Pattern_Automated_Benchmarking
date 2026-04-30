from __future__ import annotations

import argparse
import yaml
import pdb
import traceback
import time

from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
from .image_file_handling import get_images_in_folder_as_image_records
from .local_binary_pattern_processing import LBPResult, LTPResult, local_binary_pattern, local_ternary_pattern
from .local_binary_pattern_processing import local_ternary_pattern, LTPResult
from .image_processing import apply_processing
from skimage.color import rgb2gray
from .processed_to_raw_image_matching import ProcessedToRawMatcher
from .match_statistics import compute_match_distance_stats
from .visualization import visualize_image_records
from .save_visualization_as_pdf import create_image_record_match_pdf
from .result_logging import save_matches_csv, generate_config_filename


def main(return_results, cli_args=None) -> Optional[dict]:
    start = time.time()
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
    stats = compute_match_distance_stats(processed_matched_records)
    print(stats)

    if config_dict["output"]["csv_filename"] is not None:
        csv_filename = config_dict["output"]["csv_filename"]
        if csv_filename.lower() == "auto":
            csv_filename = generate_config_filename(config_dict)
        save_matches_csv(processed_matched_records, csv_filename)
    
    if config_dict["output"]["pdf_filename"] is not None:
        create_image_record_match_pdf(
            image_records=processed_matched_records,
            output_path=f"{config_dict["output"]["pdf_filename"]}.pdf",
            stats=stats,
            config=config_dict,
            records_per_page=5,
            matches_per_row=top_k,
        )

    if config_dict["output"]["visualize"]:
        visualize_image_records(processed_matched_records, 50)

    if return_results:
        end = time.time()
        total_time = end - start
        return f"Total time: {total_time:.4f} seconds"


if __name__ == "__main__":
    main()

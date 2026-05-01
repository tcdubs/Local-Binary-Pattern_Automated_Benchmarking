from __future__ import annotations

import argparse
import yaml
import pdb
import traceback
import time
import shutil

from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image
from .image_file_handling import get_images_in_folder_as_image_records
from .texture_extraction_registry import get_texture_feature_vector
from .image_processing import apply_PIL_processing, apply_numpy_processing
from skimage.color import rgb2gray
from .processed_to_raw_image_matching import ProcessedToRawMatcher
from .match_statistics import compute_match_distance_stats
from .visualization import visualize_image_records
from .save_visualization_as_pdf import create_image_record_match_pdf
from .result_logging import save_matches_csv
from datetime import datetime


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

    # Perform image preprocessing and texture extraction for both raw and working image records
    # Generally, raw image records should have minimal processing, while processed records will
    # have more aggressive processing to simulate noise, illumination, and other real-world conditions
    for records, processing_config in [
        (raw_image_records, config_dict["target_image_processing"]),
        (working_image_records, config_dict["query_image_processing"]),
    ]:
        for record in records:
            processed_image = record.image
            processed_image = apply_PIL_processing(processed_image, processing_config, rng=rng)
            processed_image = apply_numpy_processing(processed_image, processing_config, rng=rng)
            record.image = Image.fromarray(processed_image)
            gray_image = rgb2gray(processed_image)
            image_array = (gray_image * 255).astype(np.uint8)  # Convert to uint8 format expected by LBP functions  
            hist = get_texture_feature_vector(image_array, config_dict)
            record.lbp_hist = hist
        
    # Perform matching of query rectords (processed) to target records (raw)
    distance_metric = config_dict["matching"]["metric"]
    match_tolerance = config_dict["matching"]["tolerance"]
    top_k = config_dict["matching"]["top"]
    matcher = ProcessedToRawMatcher(metric_name=distance_metric, tolerance=match_tolerance, top=top_k)
    processed_matched_records = matcher(working_image_records, raw_image_records)
    stats = compute_match_distance_stats(processed_matched_records)
    print(stats)

    if config_dict["output"]["save_csv"] or config_dict["output"]["save_pdf"]:
        project_root = Path(__file__).resolve().parents[2]
        config_name = Path(args.config)
        config_name = config_name.stem
    
    # Build results directory path
        results_dir = project_root / "results" / config_name
        results_dir.mkdir(exist_ok=True)
    # Save results files
        if config_dict["output"]["save_csv"]:
            output_dir = save_matches_csv(processed_matched_records, results_dir)
    
    if config_dict["output"]["save_pdf"]:
        output_dir = create_image_record_match_pdf(
            image_records=processed_matched_records,
            results_dir=results_dir,
            stats=stats,
            config=config_dict,
            records_per_page=5,
            matches_per_row=top_k,
        )
    if config_dict["output"]["save_csv"] or config_dict["output"]["save_pdf"]:
        shutil.copy(args.config, output_dir / "experiment_config.yaml")

    if config_dict["output"]["visualize"]:
        visualize_image_records(processed_matched_records, 50)

    # Return results (use for multiparam sweeps)
    if return_results:
        end = time.time()
        total_time = end - start
        return f"Total time: {total_time:.4f} seconds"


if __name__ == "__main__":
    main()

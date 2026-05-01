from PIL import Image
import csv
from pathlib import Path
from typing import Sequence
from .image_data_containers import ImageRecord
from datetime import datetime

def save_matches_csv(image_records: Sequence[ImageRecord], results_dir: str) -> str:

    output_path = results_dir / "match_results.csv"
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
    return results_dir
                
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

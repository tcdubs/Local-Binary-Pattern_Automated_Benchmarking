from .local_binary_pattern_processing import (local_binary_pattern, local_ternary_pattern, 
                                              completed_local_binary_pattern, LBPResult, LTPResult, CLBPResult)
import numpy as np

def get_texture_feature_vector(image_array: np.ndarray, config) -> np.ndarray:
    texture_extraction_config = config["texture_extraction"]

    method_name, method_config = next(iter(texture_extraction_config.items()))
    if method_name == "local_binary_pattern":
        result = local_binary_pattern(
            image_array,
            p=method_config["P"],
            r=method_config["R"],
            method=method_config["method"],
        ).histogram

    elif method_name == "local_ternary_pattern":
        result = local_ternary_pattern(
            image_array,
            p=method_config["P"],
            r=method_config["R"],
            method=method_config["method"],
            threshold=method_config["threshold"],
        ).histogram

    elif method_name == "completed_local_binary_pattern":
        result = completed_local_binary_pattern(
            image_array,
            p=method_config["P"],
            r=method_config["R"],
            method=method_config["method"],
        ).histogram

    elif method_name == "multi_scale":
        all_hists = []
        for nested_cfg in method_config:
            nested_texture_config = {"texture_extraction": nested_cfg}
            result = get_texture_feature_vector(image_array, nested_texture_config)
            hist = np.asarray(result, dtype=np.float32)
            all_hists.append(hist)

        multi_hist = np.concatenate(all_hists)
        multi_hist /= multi_hist.sum() + 1e-6
        return multi_hist

    else:
        raise ValueError(f"Unknown texture extraction method: {method_name}")
    return result
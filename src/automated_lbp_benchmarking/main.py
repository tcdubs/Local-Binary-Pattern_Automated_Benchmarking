from __future__ import annotations

import argparse
import csv
import os
import pdb
import traceback

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern as skimage_lbp
from scipy.ndimage import gaussian_filter
from .visualize_matches_mvc import visualize_matches

from .distance_metrics import chi2_distance, get_distance_metric
from .ltp import local_ternary_pattern, LTPResult
from datetime import datetime

@dataclass
class ImageRecord:
    # Parsed filename metadata
    instance: str
    category: str
    distance: str
    rotation: str
    lighting: str

    # Data payload
    image: Image.Image  # cropped (or original) RGB/gray PIL image for visualization
    lbp_hist: np.ndarray  # normalized histogram feature vector (float64)

    # Matching related info
    index: Optional[int] = None
    matched_index: Optional[int] = None
    matched_category: Optional[str] = None
    nn_distance: Optional[float] = None
    correct: Optional[bool] = None


# ----------------------------
# Utilities
# ----------------------------

def parse_filename(filename: str) -> dict:
    """
    Parse filename of form:
        INSTANCE_CATEGORY_DISTANCE_ROTATION_LIGHTING.png

    Returns dict with keys:
        INSTANCE, CATEGORY, DISTANCE, ROTATION, LIGHTING
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    if ext.lower() not in [".png", ".jpg"]:
        raise ValueError("Not a PNG filename")

    parts = name.split("_")
    if len(parts) < 5:
        raise ValueError(f"Filename does not match expected format: {filename}")

    return {
        "INSTANCE": parts[0],
        "CATEGORY": parts[1],
        "DISTANCE": parts[2],
        "ROTATION": parts[3],
        "LIGHTING": parts[4],
    }


def center_crop(arr_2d: np.ndarray, X: int, Y: int) -> np.ndarray:
    """Crop a 2D array to a random X-by-Y region."""
    h, w = arr_2d.shape
    X = min(X, w)
    Y = min(Y, h)
    # Use global _crop_rng if set, else default to center crop
    rng = globals().get("_crop_rng", None)
    if rng is not None:
        x_start = rng.integers(0, w - X + 1)
        y_start = rng.integers(0, h - Y + 1)
    else:
        x_start = (w - X) // 2
        y_start = (h - Y) // 2
    return arr_2d[y_start:y_start + Y, x_start:x_start + X]


def center_crop_pil(im: Image.Image, X: int, Y: int) -> Image.Image:
    """Random-crop a PIL image to X-by-Y."""
    w, h = im.size
    X = min(X, w)
    Y = min(Y, h)
    rng = globals().get("_crop_rng", None)
    if rng is not None:
        x_start = rng.integers(0, w - X + 1)
        y_start = rng.integers(0, h - Y + 1)
    else:
        x_start = (w - X) // 2
        y_start = (h - Y) // 2
    return im.crop((x_start, y_start, x_start + X, y_start + Y))


# ----------------------------
# LBF Facade
# ----------------------------

@dataclass(frozen=True)
class LBPFacade:
    """Facade over skimage's local_binary_pattern with dtype normalization."""
    P: int = 8
    R: float = 1.0
    method: str = "uniform"
    ltp_threshold: int | None = None

    def __call__(self, gray_2d: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if gray_2d.ndim != 2:
            raise ValueError("LBP input must be a 2D grayscale array")

        if self.ltp_threshold is not None:
            lbp_codes: LTPResult = local_ternary_pattern(
                gray_2d, p=self.P, r=self.R, threshold=self.ltp_threshold, method=self.method, mask=mask)
            return lbp_codes.histogram
        lbp = skimage_lbp(gray_2d, P=self.P, R=self.R, method=self.method)

        # skimage may return float; cast safely
        max_val = np.nanmax(lbp) if lbp.size else 0
        if max_val <= 255:
            return lbp.astype(np.uint8)
        return lbp.astype(np.uint32)


# ----------------------------
# Hist Adapter
# ----------------------------

@dataclass(frozen=True)
class LBPHistogramAdapter:
    """
    Adapter that converts a 2D LBP code image into a normalized histogram vector.
    """
    bins: int
    smooth: float = 0.0

    def __call__(self, lbp_codes: np.ndarray) -> np.ndarray:
        if lbp_codes is None or lbp_codes.size == 0:
            hist = np.zeros(self.bins, dtype=np.float64)
        elif lbp_codes.ndim == 1:
            hist = lbp_codes.astype(np.float64)
        else:
            hist = np.bincount(lbp_codes.ravel(), minlength=self.bins).astype(np.float64)
        if self.smooth > 0.0:
            hist += self.smooth
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist


# ----------------------------
# THE PIPELINE
# ----------------------------

class Stage:
    """A pipeline stage: consumes and returns a sequence of records."""
    def __call__(self, records: Sequence[ImageRecord]) -> Sequence[ImageRecord]:
        raise NotImplementedError


@dataclass

# Modified loader to return both raw and processed records
@dataclass
class ImageFolderLoaderDual:
    folder: str
    lbp: LBPFacade
    X: Optional[int] = None
    Y: Optional[int] = None
    use_gray_image_for_viz: bool = False
    gaussian_blur: float = 0.0
    hist_smooth: float = 0.0
    simulate_illumination: bool = False
    brightness: float = 1.0
    contrast: float = 1.0
    simulate_noise: bool = False
    noise_sigma: float = 10.0

    def __call__(self, _records: Sequence[ImageRecord] = ()) -> tuple[list[ImageRecord], list[ImageRecord]]:
        raw_items = []
        processed_items = []
        max_code = 0

        for fname in sorted(os.listdir(self.folder)):
            path = os.path.join(self.folder, fname)
            try:
                meta = parse_filename(fname)
            except ValueError:
                continue

            try:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    gray = rgb.convert("L")
                    gray_arr = np.array(gray, dtype=np.float32)

                    viz_img = rgb.copy()
                    viz_img_proc = viz_img.copy()
                    gray_arr_proc = gray_arr.copy()

                    # Cropping
                    if self.X is not None and self.Y is not None:
                        gray_arr = center_crop(gray_arr, self.X, self.Y)
                        gray_arr_proc = center_crop(gray_arr_proc, self.X, self.Y)
                        viz_img = center_crop_pil(viz_img, self.X, self.Y)
                        viz_img_proc = center_crop_pil(viz_img_proc, self.X, self.Y)
                        if self.use_gray_image_for_viz:
                            viz_img = viz_img.convert("L")
                            viz_img_proc = viz_img_proc.convert("L")

                    # RAW: no processing
                    # Processed: apply noise/illum/blur
                    # Prepare transformations if needed
                    brightness = self.brightness if self.simulate_illumination else 1.0
                    contrast = self.contrast if self.simulate_illumination else 1.0
                    noise = None
                    if self.simulate_noise:
                        rng = globals().get("_crop_rng") or np.random.default_rng()
                        noise = rng.normal(0, self.noise_sigma, gray_arr_proc.shape)

                    # Apply illumination to gray_arr_proc
                    if self.simulate_illumination:
                        gray_arr_proc = (gray_arr_proc - 128) * contrast + 128 * brightness
                        gray_arr_proc = np.clip(gray_arr_proc, 0, 255)
                        gray_arr_proc = gray_arr_proc.astype(np.uint8)

                    # Apply noise to gray_arr_proc
                    if self.simulate_noise:
                        gray_arr_proc = gray_arr_proc.astype(np.float32) + noise
                        gray_arr_proc = np.clip(gray_arr_proc, 0, 255)
                        gray_arr_proc = gray_arr_proc.astype(np.uint8)

                    # Apply same transformations to viz_img_proc
                    if self.simulate_illumination or self.simulate_noise:
                        viz_arr = np.array(viz_img_proc, dtype=np.float32)
                        if self.simulate_illumination:
                            viz_arr = (viz_arr - 128) * contrast + 128 * brightness
                            viz_arr = np.clip(viz_arr, 0, 255)
                        if self.simulate_noise:
                            if viz_arr.ndim == 3:
                                noise_viz = noise[:, :, np.newaxis]
                            else:
                                noise_viz = noise
                            viz_arr = viz_arr + noise_viz
                            viz_arr = np.clip(viz_arr, 0, 255)
                        viz_arr = viz_arr.astype(np.uint8)
                        if viz_arr.ndim == 3:
                            viz_img_proc = Image.fromarray(viz_arr, mode='RGB')
                        else:
                            viz_img_proc = Image.fromarray(viz_arr, mode='L')

                    if self.gaussian_blur > 0.0:
                        gray_arr_proc = gaussian_filter(gray_arr_proc.astype(np.float32), sigma=self.gaussian_blur).astype(np.uint8)

            except Exception as e:
                print(f"Failed to process {path} due to {e}, skipping.")
                traceback.print_exc()
                continue

            # Build mask: True for non-white pixels, False for white background
            # Assume gray_arr and gray_arr_proc are 2D arrays (grayscale)
            mask_raw = (gray_arr < 255)
            mask_proc = (gray_arr_proc < 255)

            # Set masked-out (background) pixels to black in the source images
            gray_arr = gray_arr.copy()
            gray_arr_proc = gray_arr_proc.copy()
            gray_arr[~mask_raw] = 0
            gray_arr_proc[~mask_proc] = 0

            # If using LTP, pass mask
            if hasattr(self.lbp, 'ltp_threshold') and self.lbp.ltp_threshold is not None:
                lbp_codes_raw = self.lbp(gray_arr, mask=mask_raw)
                lbp_codes_proc = self.lbp(gray_arr_proc, mask=mask_proc)
            else:
                lbp_codes_raw = self.lbp(gray_arr)
                lbp_codes_proc = self.lbp(gray_arr_proc)

            if lbp_codes_raw.size:
                max_code = max(max_code, int(lbp_codes_raw.max()))
            if lbp_codes_proc.size:
                max_code = max(max_code, int(lbp_codes_proc.max()))

            raw_items.append((meta, viz_img, lbp_codes_raw))
            processed_items.append((meta, viz_img_proc, lbp_codes_proc))

        bins = max_code + 1
        adapter = LBPHistogramAdapter(bins=bins, smooth=self.hist_smooth)

        raw_records: List[ImageRecord] = []
        processed_records: List[ImageRecord] = []
        for meta, viz_img, lbp_codes in raw_items:
            rec = ImageRecord(
                instance=meta["INSTANCE"],
                category=meta["CATEGORY"],
                distance=meta["DISTANCE"],
                rotation=meta["ROTATION"],
                lighting=meta["LIGHTING"],
                image=viz_img,
                lbp_hist=adapter(lbp_codes),
            )
            raw_records.append(rec)
        for meta, viz_img, lbp_codes in processed_items:
            rec = ImageRecord(
                instance=meta["INSTANCE"],
                category=meta["CATEGORY"],
                distance=meta["DISTANCE"],
                rotation=meta["ROTATION"],
                lighting=meta["LIGHTING"],
                image=viz_img,
                lbp_hist=adapter(lbp_codes),
            )
            processed_records.append(rec)

        return raw_records, processed_records


@dataclass

# Matcher that matches processed records against raw records
@dataclass
class ProcessedToRawMatcher(Stage):
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = chi2_distance
    metric_name: str = "chi2"
    tolerance: Optional[float] = None
    top: Optional[int] = None

    def __call__(self, processed_records: Sequence[ImageRecord], raw_records: Sequence[ImageRecord]) -> Sequence[ImageRecord]:
        if not processed_records or not raw_records:
            return processed_records

        distance_fn = self.distance_fn
        if self.metric_name:
            distance_fn = get_distance_metric(self.metric_name)

        raw_hists = [np.asarray(r.lbp_hist, dtype=np.float64) for r in raw_records]
        n = len(raw_hists)

        for i, rec in enumerate(processed_records):
            rec.index = i
            min_val = float("inf")
            min_idx = -1
            matches: List[int] = []
            match_distances: List[float] = []
            match_categories: List[str] = []
            all_distances: List[tuple] = []

            for j in range(n):
                d = distance_fn(np.asarray(rec.lbp_hist, dtype=np.float64), raw_hists[j])
                if d < min_val:
                    min_val = d
                    min_idx = j
                if self.tolerance is not None and d < self.tolerance:
                    matches.append(j)
                    match_distances.append(float(d))
                    match_categories.append(raw_records[j].category)
                if self.top is not None:
                    all_distances.append((j, float(d), raw_records[j].category))

            if matches:
                sorted_matches = sorted(
                    zip(matches, match_distances, match_categories),
                    key=lambda x: x[1],
                )
                if self.top is not None:
                    sorted_matches = sorted_matches[:self.top]
                rec.matching_indices = [m[0] for m in sorted_matches]
                rec.matching_distances = [m[1] for m in sorted_matches]
                rec.matching_categories = [m[2] for m in sorted_matches]
            elif self.top is not None and self.tolerance is None and all_distances:
                sorted_top = sorted(all_distances, key=lambda x: x[1])[:self.top]
                rec.matching_indices = [m[0] for m in sorted_top]
                rec.matching_distances = [m[1] for m in sorted_top]
                rec.matching_categories = [m[2] for m in sorted_top]
            else:
                rec.matching_indices = []
                rec.matching_distances = []
                rec.matching_categories = []

            rec.nn_distance = None if min_idx == -1 else float(min_val)
            rec.matched_index = None if min_idx == -1 else int(min_idx)
            rec.matched_category = None if min_idx == -1 else raw_records[min_idx].category
            rec.correct = False if min_idx == -1 else (rec.category == raw_records[min_idx].category)
        
        return processed_records


# ----------------------------
# Pipeline BEGINS
# ----------------------------

@dataclass

# Pipeline for processed-to-raw matching
@dataclass
class PipelineDual:
    source: Callable[[], tuple[list[ImageRecord], list[ImageRecord]]]
    matcher: ProcessedToRawMatcher

    def run(self) -> list[ImageRecord]:
        raw_records, processed_records = self.source()
        processed_records = self.matcher(processed_records, raw_records)
        return processed_records


# ----------------------------
# Results and visualization
# ----------------------------

def save_matches_csv(records: Sequence[ImageRecord], out_path: str) -> None:
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
        out_path = out_path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + ".csv"
        output_path = results_dir / out_path
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Instance", "Category", "Distance", "Matched_Category", "Matched_Index", "Correct"])
        for r in records:
            writer.writerow([
                r.index,
                r.instance,
                r.category,
                r.distance,
                r.matched_category,
                r.matched_index,
                bool(r.correct),
            ])


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


# ----------------------------
# CLI STUFF
# ----------------------------

def main(return_results: bool = False, cli_args=None):
    parser = argparse.ArgumentParser(description="Load PNGs from folder, parse metadata, compute LBP, match by chi2 NN")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing .PNG images")
    parser.add_argument("--P", type=int, default=8, help="Number of neighbor points for LBP (default: 8)")
    parser.add_argument("--R", type=float, default=1.0, help="Radius for LBP (default: 1.0)")
    parser.add_argument(
        "--method",
        type=str,
        default="uniform",
        choices=["default", "ror", "uniform", "var", "ltp"],
        help="LBP method for skimage.feature.local_binary_pattern (default: 'uniform')",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="chi2",
        choices=["chi2", "cosine", "hellinger"],
        help="Histogram distance metric for nearest-neighbor matching (default: chi2)",
    )
    parser.add_argument(
        "--save-csv",
        nargs="?",
        const="matches.csv",
        default=None,
        help="Save results to CSV (default: project_root/data/matches.csv)"
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help="Apply Gaussian blur with given sigma before LBP (default: 0, no blur)"
    )
    parser.add_argument(
        "--ltp-threshold",
        type=int,
        default=None,
        help="If set, use Local Ternary Pattern with this integer threshold instead of LBP. (default: None, disables LTP)"
    )
    parser.add_argument("--visualize", action="store_true", help="Open interactive visualization of matches")
    parser.add_argument("--X", type=int, default=None, help="Width of centermost region for LBP (optional)")
    parser.add_argument("--Y", type=int, default=None, help="Height of centermost region for LBP (optional)")
    parser.add_argument("--V", action="store_true", help="Print verbose output")
    parser.add_argument("--G", action="store_true", help="Use grayscale image in visualization")
    parser.add_argument("--crop-seed", type=int, default=None, help="Seed for random cropping (optional, for reproducibility)")
    parser.add_argument(
        "--hist-smooth",
        type=float,
        default=0.0,
        help="Additive smoothing value for LBP histogram bins (default: 0.0, no smoothing)"
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=10.0,
        help="Standard deviation for Gaussian noise when --simulate-noise is enabled (default: 10.0)"
    )
    parser.add_argument("--simulate-illumination", action="store_true", help="Simulate illumination changes by adjusting brightness and contrast")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness factor for illumination simulation (default: 1.0, no change)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast factor for illumination simulation (default: 1.0, no change)")
    parser.add_argument("--tolerance", type=float, default=None, help="Distance tolerance for storing and visualizing multiple matches (default: none)")
    parser.add_argument("--top", type=int, default=None, help="Show top n closest matches (default: none)")
    parser.add_argument("--simulate-noise", action="store_true", help="Add Gaussian noise to the images")
    if cli_args is not None:
        args = parser.parse_args(cli_args)
    else:
        args = parser.parse_args()

    # Set up global RNG for cropping if seed is provided
    global _crop_rng
    if args.crop_seed is not None:
        import numpy as np
        _crop_rng = np.random.default_rng(args.crop_seed)
    else:
        _crop_rng = None

    lbp = LBPFacade(P=args.P, R=args.R, method=args.method, ltp_threshold=args.ltp_threshold)
    loader = ImageFolderLoaderDual(
        folder=args.folder,
        lbp=lbp,
        X=args.X,
        Y=args.Y,
        use_gray_image_for_viz=args.G,
        gaussian_blur=args.blur,
        hist_smooth=args.hist_smooth,
        simulate_illumination=args.simulate_illumination,
        brightness=args.brightness,
        contrast=args.contrast,
        simulate_noise=args.simulate_noise,
        noise_sigma=args.noise_sigma,
    )

    matcher = ProcessedToRawMatcher(metric_name=args.metric, tolerance=args.tolerance, top=args.top)
    # Get both raw and processed records once
    raw_records, processed_records = loader(())
    # Run matching
    processed_records = matcher(processed_records, raw_records)
    records = processed_records

    # Suppress file output if running under run_experiments.py (detected by env var)
    running_experiments = os.environ.get("LBP_EXPERIMENT_MODE") == "1"
    if args.save_csv and not running_experiments:
        save_matches_csv(records, args.save_csv)
        print(f"Saved matches CSV to {args.save_csv}")


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

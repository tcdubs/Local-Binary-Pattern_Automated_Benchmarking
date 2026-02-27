
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern as skimage_lbp
from .visualize_matches_mvc import visualize_matches

from .distance_metrics import chi2_distance, get_distance_metric

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
    if ext.lower() != ".png":
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
    """Crop a 2D array to the centermost X-by-Y region."""
    h, w = arr_2d.shape
    if w < X or h < Y:
        raise ValueError("Image smaller than requested crop")
    x_start = (w - X) // 2
    y_start = (h - Y) // 2
    return arr_2d[y_start:y_start + Y, x_start:x_start + X]


def center_crop_pil(im: Image.Image, X: int, Y: int) -> Image.Image:
    """Center-crop a PIL image to X-by-Y."""
    w, h = im.size
    if w < X or h < Y:
        raise ValueError("Image smaller than requested crop")
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

    def __call__(self, gray_2d: np.ndarray) -> np.ndarray:
        if gray_2d.ndim != 2:
            raise ValueError("LBP input must be a 2D grayscale array")

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

    def __call__(self, lbp_codes: np.ndarray) -> np.ndarray:
        if lbp_codes is None or lbp_codes.size == 0:
            return np.zeros(self.bins, dtype=np.float64)
        hist = np.bincount(lbp_codes.ravel(), minlength=self.bins).astype(np.float64)
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
class ImageFolderLoader:
    """
    Stage (source): loads PNGs from a folder, parses metadata, computes LBP histograms,
    and emits ImageRecord objects.
    """
    folder: str
    lbp: LBPFacade
    X: Optional[int] = None
    Y: Optional[int] = None
    use_gray_image_for_viz: bool = False

    def __call__(self, _records: Sequence[ImageRecord] = ()) -> Sequence[ImageRecord]:
        # First pass: load images and compute raw LBP arrays
        raw_items = []
        max_code = 0

        for fname in sorted(os.listdir(self.folder)):
            if not fname.lower().endswith(".png"):
                continue

            path = os.path.join(self.folder, fname)
            try:
                meta = parse_filename(fname)
            except ValueError:
                continue

            try:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    gray = rgb.convert("L")
                    gray_arr = np.array(gray)

                    viz_img = rgb.copy()

                    if self.X is not None and self.Y is not None:
                        # crop numpy grayscale array
                        gray_arr = center_crop(gray_arr, self.X, self.Y)

                        # crop visualization image
                        viz_img = center_crop_pil(viz_img, self.X, self.Y)
                        if self.use_gray_image_for_viz:
                            viz_img = viz_img.convert("L")

            except Exception:
                continue

            lbp_codes = self.lbp(gray_arr)
            if lbp_codes.size:
                max_code = max(max_code, int(lbp_codes.max()))

            raw_items.append((meta, viz_img, lbp_codes))

        bins = max_code + 1
        adapter = LBPHistogramAdapter(bins=bins)

        # Second pass: convert to histogram + build records
        records: List[ImageRecord] = []
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
            records.append(rec)

        return records


@dataclass
class NearestNeighborMatcher(Stage):
    """annotates each record with nearest-neighbor match info."""
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = chi2_distance
    metric_name: str = "chi2"

    def __call__(self, records: Sequence[ImageRecord]) -> Sequence[ImageRecord]:
        if not records:
            return records

        # Strategy selection: allow resolving by name (preferred) while keeping a callable hook.
        # If metric_name is set, it overrides distance_fn.
        distance_fn = self.distance_fn
        if self.metric_name:
            distance_fn = get_distance_metric(self.metric_name)

        hists = [np.asarray(r.lbp_hist, dtype=np.float64) for r in records]
        n = len(hists)

        for i, rec in enumerate(records):
            rec.index = i
            if n == 1:
                rec.nn_distance = None
                rec.matched_category = None
                rec.matched_index = None
                rec.correct = False
                continue

            min_val = float("inf")
            min_idx = -1

            for j in range(n):
                if i == j:
                    continue
                d = distance_fn(hists[i], hists[j])
                if d < min_val:
                    min_val = d
                    min_idx = j

            rec.nn_distance = None if min_idx == -1 else float(min_val)
            rec.matched_index = None if min_idx == -1 else int(min_idx)
            rec.matched_category = None if min_idx == -1 else records[min_idx].category
            rec.correct = False if min_idx == -1 else (rec.category == records[min_idx].category)

        return records


# ----------------------------
# Pipeline BEGINS
# ----------------------------

@dataclass
class Pipeline:
    source: Callable[[], Sequence[ImageRecord]]
    stages: List[Stage] = field(default_factory=list)

    def run(self) -> List[ImageRecord]:
        records = list(self.source())
        for stage in self.stages:
            records = list(stage(records))
        return records


# ----------------------------
# Results and visualization
# ----------------------------

def save_matches_csv(records: Sequence[ImageRecord], out_path: str) -> None:
    # Find root project director (hardcoded, probably shouldn't do this)
    project_root = Path(__file__).resolve().parents[2]

    # Build results directory path
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Build full output path
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
    correct_count = sum(1 for r in records if bool(r.correct))
    pct = 100.0 * correct_count / total if total else 0.0

    lowest_incorrect = float("inf")
    highest_correct = float("-inf")

    for r in records:
        if r.nn_distance is None:
            continue
        if bool(r.correct):
            highest_correct = max(highest_correct, r.nn_distance)
        else:
            lowest_incorrect = min(lowest_incorrect, r.nn_distance)

    # for r in records:
    #     print(
    #         f"Instance: {r.instance}, Category: {r.category}, "
    #         f"NN distance: {r.nn_distance}, Matched: {r.matched_category}, Correct: {r.correct}"
    #     )
    #     print(f"LBP Histogram: {r.lbp_hist}\n")

    print(f"Correct matches: {correct_count}/{total} ({pct:.2f}%)")
    print(f"Highest distance among correct matches: {highest_correct:.6f}" if highest_correct != float("-inf") else
          "Highest distance among correct matches: N/A")
    print(f"Lowest distance among incorrect matches: {lowest_incorrect:.6f}" if lowest_incorrect != float("inf") else
          "Lowest distance among incorrect matches: N/A")


# ----------------------------
# CLI STUFF
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load PNGs from folder, parse metadata, compute LBP, match by chi2 NN")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing .PNG images")
    parser.add_argument("--P", type=int, default=8, help="Number of neighbor points for LBP (default: 8)")
    parser.add_argument("--R", type=float, default=1.0, help="Radius for LBP (default: 1.0)")
    parser.add_argument(
    "--method",
    type=str,
    default="uniform",
    choices=["default", "ror", "uniform", "var"],
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
    parser.add_argument("--visualize", action="store_true", help="Open interactive visualization of matches")
    parser.add_argument("--X", type=int, default=None, help="Width of centermost region for LBP (optional)")
    parser.add_argument("--Y", type=int, default=None, help="Height of centermost region for LBP (optional)")
    parser.add_argument("--V", action="store_true", help="Print verbose output")
    parser.add_argument("--G", action="store_true", help="Use grayscale image in visualization")

    args = parser.parse_args()

    lbp = LBPFacade(P=args.P, R=args.R, method=args.method)

    loader = ImageFolderLoader(
        folder=args.folder,
        lbp=lbp,
        X=args.X,
        Y=args.Y,
        use_gray_image_for_viz=args.G,
    )

    pipeline = Pipeline(
        source=lambda: loader(()),
        stages=[
            NearestNeighborMatcher(metric_name=args.metric),
        ],
    )

    records = pipeline.run()

    if args.save_csv:
        save_matches_csv(records, args.save_csv)
        print(f"Saved matches CSV to {args.save_csv}")

    if args.visualize:
        items_for_viz = []
        for r in records:
            items_for_viz.append({
                "INSTANCE": r.instance,
                "CATEGORY": r.category,
                "DISTANCE": r.nn_distance,
                "ROTATION": r.rotation,
                "LIGHTING": r.lighting,
                "LBP": r.lbp_hist,
                "IMAGE": r.image,
                "INDEX": r.index,
                "MATCHED_INDEX": r.matched_index,
                "MATCHED_CATEGORY": r.matched_category,
                "CORRECT": r.correct,
            })
        visualize_matches(items_for_viz, metric_name=args.metric)

    if args.V:
        print_verbose_report(records)


if __name__ == "__main__":
    main()

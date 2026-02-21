import os
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern as skimage_lbp
from chi2_distance import chi2_distance


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse filename of form INSTANCE_CATEGORY_DISTANCE_ROTATION_LIGHTING.png

    Returns a dict with keys: INSTANCE, CATEGORY, DISTANCE, ROTATION, LIGHTING
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")
    if len(parts) < 5:
        raise ValueError(f"Filename '{filename}' does not contain 5 underscore-separated fields")
    # Take the first 5 parts; if more exist, join the remainder into the last field
    if len(parts) > 5:
        parts = parts[:4] + ["_".join(parts[4:])]
    return {
        "INSTANCE": parts[0],
        "CATEGORY": parts[1],
        "DISTANCE": parts[2],
        "ROTATION": parts[3],
        "LIGHTING": parts[4],
    }


def local_binary_pattern(image: np.ndarray, P: int = 8, R: float = 1.0, method: str = 'uniform') -> np.ndarray:
    """Compute LBP using scikit-image's implementation.

    Parameters:
    - image: 2D grayscale array
    - P: number of circularly symmetric neighbour set points (quantization of the angular space)
    - R: radius of circle
    - method: one of 'default', 'ror', 'uniform', 'var'

    Returns an array of same shape with LBP codes. dtype will be uint8 when
    values fit in 0-255, otherwise uint32.
    """
    if image.ndim != 2:
        raise ValueError("Image must be a 2D grayscale array")

    lbp = skimage_lbp(image, P=P, R=R, method=method)
    # skimage returns floating or integer values depending on method; cast safely
    if np.nanmax(lbp) <= 255:
        return lbp.astype(np.uint8)
    return lbp.astype(np.uint32)


def load_images_from_folder(folder: str, P: int = 8, R: float = 1.0, method: str = 'uniform') -> List[Dict]:
    """Load all .png images from `folder`, parse filename metadata, compute LBP.

    Returns a list of dictionaries. Each dict has keys:
    - "INSTANCE", "CATEGORY", "DISTANCE", "ROTATION", "LIGHTING", "LBP"
    where "LBP" is a numpy array (dtype uint8) with the per-pixel LBP codes.
    """
    results: List[Dict] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.png'):
            continue
        path = os.path.join(folder, fname)
        try:
            meta = parse_filename(fname)
        except ValueError:
            # skip files that don't match expected format
            continue

        try:
            with Image.open(path) as im:
                gray = im.convert('L')
                arr = np.array(gray)
        except Exception as e:
            # skip unreadable images
            continue

        lbp = local_binary_pattern(arr, P=P, R=R, method=method)

        data = {
            "INSTANCE": meta["INSTANCE"],
            "CATEGORY": meta["CATEGORY"],
            "DISTANCE": meta["DISTANCE"],
            "ROTATION": meta["ROTATION"],
            "LIGHTING": meta["LIGHTING"],
            "LBP": lbp,
        }
        results.append(data)

    return results


def compute_nearest_matches(items: List[Dict]) -> None:
    """For each item in `items`, compute histogram of its LBP codes and find
    the nearest other image by chi-squared distance. Stores results in-place
    under keys `DISTANCE` (float) and `MATCHED_CATEGORY` (str).
    """
    if not items:
        return

    lbps = [it['LBP'] for it in items]
    # determine consistent histogram length across images
    max_code = 0
    for lb in lbps:
        if lb.size:
            max_code = max(max_code, int(lb.max()))
    bins = max_code + 1

    hists = []
    for lb in lbps:
        hist = np.bincount(lb.ravel(), minlength=bins).astype(np.float64)
        s = hist.sum()
        if s > 0:
            hist /= s
        hists.append(hist)

    n = len(items)
    for i in range(n):
        if n == 1:
            items[i]['DISTANCE'] = None
            items[i]['MATCHED_CATEGORY'] = None
            continue

        min_val = float('inf')
        min_idx = -1
        for j in range(n):
            if i == j:
                continue
            d = chi2_distance(hists[i], hists[j])
            if d < min_val:
                min_val = d
                min_idx = j

        items[i]['DISTANCE'] = None if min_idx == -1 else float(min_val)
        items[i]['MATCHED_CATEGORY'] = None if min_idx == -1 else items[min_idx].get('CATEGORY')


def main():
    parser = argparse.ArgumentParser(description="Load PNGs from folder, parse metadata, compute LBP")
    parser.add_argument('folder', nargs='?', default='.', help='Folder containing .PNG images')
    parser.add_argument('--P', type=int, default=8, help='Number of neighbor points for LBP (default: 8)')
    parser.add_argument('--R', type=float, default=1.0, help='Radius for LBP (default: 1.0)')
    parser.add_argument('--method', type=str, default='uniform', choices=['default', 'ror', 'uniform', 'var'],
                        help="LBP method for skimage.feature.local_binary_pattern (default: 'uniform')")
    args = parser.parse_args()

    items = load_images_from_folder(args.folder, P=args.P, R=args.R, method=args.method)
    print(f"Loaded {len(items)} images from {args.folder}")
    if items:
        first = items[0]
        print("Example metadata:")
        print({k: first[k] for k in ["INSTANCE", "CATEGORY", "DISTANCE", "ROTATION", "LIGHTING"]})
        print("LBP shape:", first['LBP'].shape)
        print(f"LBP parameters: P={args.P}, R={args.R}, method={args.method}")


if __name__ == '__main__':
    main()

import os
import argparse
from typing import List, Dict

import csv
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern as skimage_lbp
from chi2_distance import chi2_distance


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse filename of form INSTANCE_CATEGORY_DISTANCE_ROTATION_LIGHTING.png

    Returns a dict with keys: INSTANCE, CATEGORY, DISTANCE, ROTATION, LIGHTING
    """
    #print("Parsing filename:", filename)
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


def load_images_from_folder(folder: str, P: int = 8, R: float = 1.0, method: str = 'uniform', X: int = None, Y: int = None) -> List[Dict]:
    """Load all .png images from `folder`, parse filename metadata, compute LBP.

    Returns a list of dictionaries. Each dict has keys:
    - "INSTANCE", "CATEGORY", "DISTANCE", "ROTATION", "LIGHTING", "LBP"
    where "LBP" is a numpy array (dtype uint8) with the per-pixel LBP codes.
    """
    results: List[Dict] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.png'):
            print("Skipping non-PNG file:", fname)
            continue
        path = os.path.join(folder, fname)
        try:
            meta = parse_filename(fname)
        except ValueError:
            # skip files that don't match expected format
            continue

        try:
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                gray = rgb.convert('L')
                arr = np.array(gray)
                image_copy = rgb.copy()
                # If X and Y are provided, crop to centermost X by Y pixels
                if X is not None and Y is not None:
                    h, w = arr.shape
                    if w < X or h < Y:
                        # Skip images smaller than X by Y
                        continue
                    x_start = (w - X) // 2
                    y_start = (h - Y) // 2
                    arr = arr[y_start:y_start+Y, x_start:x_start+X]
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
            # temporarily store the 2D LBP array; will convert to histogram below
            "_LBP_ARRAY": lbp,
            # store the original loaded image (PIL.Image)
            "IMAGE": image_copy,
        }
        results.append(data)

    # Convert stored LBP arrays to normalized histograms and replace key with `LBP`
    max_code = 0
    for it in results:
        lb = it.get("_LBP_ARRAY")
        if lb is not None and lb.size:
            max_code = max(max_code, int(lb.max()))
    bins = max_code + 1

    for it in results:
        lb = it.pop("_LBP_ARRAY", None)
        if lb is None:
            hist = np.zeros(bins, dtype=np.float64)
        else:
            hist = np.bincount(lb.ravel(), minlength=bins).astype(np.float64)
            s = hist.sum()
            if s > 0:
                hist /= s
        it["LBP"] = hist

    return results


def compute_nearest_matches(items: List[Dict]) -> None:
    """For each item in `items`, find the nearest other image by chi-squared
    distance using the precomputed `LBP` histograms. Stores results in-place
    under keys `DISTANCE` (float) and `MATCHED_CATEGORY` (str).
    """
    if not items:
        return

    hists = [np.asarray(it['LBP'], dtype=np.float64) for it in items]
    n = len(hists)

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
        items[i]['INDEX'] = i
        items[i]['MATCHED_INDEX'] = min_idx
        # mark whether the matched category equals the item's category
        if min_idx == -1:
            items[i]['CORRECT'] = False
        else:
            items[i]['CORRECT'] = (items[i].get('CATEGORY') == items[min_idx].get('CATEGORY'))


def main():
    parser = argparse.ArgumentParser(description="Load PNGs from folder, parse metadata, compute LBP")
    parser.add_argument('folder', nargs='?', default='.', help='Folder containing .PNG images')
    parser.add_argument('--P', type=int, default=8, help='Number of neighbor points for LBP (default: 8)')
    parser.add_argument('--R', type=float, default=1.0, help='Radius for LBP (default: 1.0)')
    parser.add_argument('--method', type=str, default='uniform', choices=['default', 'ror', 'uniform', 'var'],
                        help="LBP method for skimage.feature.local_binary_pattern (default: 'uniform')")
    parser.add_argument('--save-csv', type=str, default=None, help='Path to write matches CSV (optional)')
    parser.add_argument('--visualize', action='store_true', help='Open interactive visualization of matches')
    parser.add_argument('--X', type=int, default=None, help='Width of centermost region for LBP (optional)')
    parser.add_argument('--Y', type=int, default=None, help='Height of centermost region for LBP (optional)')
    args = parser.parse_args()
    items = load_images_from_folder(args.folder, P=args.P, R=args.R, method=args.method, X=args.X, Y=args.Y)
    # compute nearest matches (adds `DISTANCE` and `MATCHED_CATEGORY` to each item)
    compute_nearest_matches(items)
    #print(f"Loaded {len(items)} images from {args.folder}")
    if items:
        first = items[0]
        #print("Example metadata:")
        #print({k: first[k] for k in ["INSTANCE", "CATEGORY", "DISTANCE", "ROTATION", "LIGHTING"]})
        #print("LBP shape:", first['LBP'].shape)
        #print(f"LBP parameters: P={args.P}, R={args.R}, method={args.method}")
    if args.save_csv:
        out_path = args.save_csv
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Instance", "Category", "Distance", "Matched_Category", "Matched_Index", "Correct"])
            for it in items:
                writer.writerow([
                    it.get('INDEX'),
                    it.get('INSTANCE'),
                    it.get('CATEGORY'),
                    it.get('DISTANCE'),
                    it.get('MATCHED_CATEGORY'),
                    it.get('MATCHED_INDEX'),
                    bool(it.get('CORRECT')),
                ])
        print(f"Saved matches CSV to {out_path}")
    if args.visualize:
        try:
            from visualize_matches import visualize_matches
        except Exception as e:
            print("Could not import or run visualization:", e)
        else:
            visualize_matches(items)
    # print overall accuracy (percentage of items with CORRECT == True)
    total = len(items)
    if total > 0:
        correct_count = sum(1 for it in items if bool(it.get('CORRECT')))
        pct = 100.0 * correct_count / total
        print(f"Correct matches: {correct_count}/{total} ({pct:.2f}%)")


if __name__ == '__main__':
    main()

## 📌 Command Line Arguments

Automated pipeline for benchmarking local binary pattern analysis of an image set
for trialing LBP hyperparameters, distance metrics, and preprocessing and their effect
on LBP texture classification performance.

Install with:
pip install .

The `lbp-benchmark` script accepts the following arguments:

### 📁 Input
- **`folder`** *(optional)*  
  Path to a folder containing `.PNG` images.  
  **Default:** current directory (`.`)

---

### ⚙️ LBP Parameters
- **`--P`** *(int)*  
  Number of neighboring points used in the LBP computation.  
  **Default:** `8`

- **`--R`** *(float)*  
  Radius for the LBP neighborhood.  
  **Default:** `1.0`

- **`--method`** *(str)*  
  LBP method used by `skimage.feature.local_binary_pattern`.  
  **Options:** `default`, `ror`, `uniform`, `var`  
  **Default:** `uniform`

- **`--ltp`** *(flag)*  
  Use Local Ternary Pattern instead of standard LBP.  
  LTP will use the arguments passed for LBP in the LBP computation
  stage of the LTP algorithm.

---

### 📊 Matching / Distance
- **`--metric`** *(str)*  
  Distance metric used for histogram comparison (nearest neighbor matching).  
  **Options:** `chi2`, `cosine`, `hellinger`  
  **Default:** `chi2`

---

### 🖼️ Image Preprocessing
- **`--blur`** *(float)*  
  Apply Gaussian blur before LBP computation.  
  Value corresponds to sigma.  
  **Default:** `0.0` (no blur)

- **`--X`** *(int, optional)*  
  Width of the 'patch' or 'region' used for LBP. Cropped from the center of the image.

- **`--Y`** *(int, optional)*  
  Height of the 'patch' or 'region' used for LBP. Cropped from the center of the image.

---

### 📤 Output
- **`--save-csv`** *(optional)*  
  Save match results to a CSV file.  
  - If no filename is provided → defaults to `matches.csv`  
  - If omitted → no CSV is saved  

---

### 🖥️ Visualization
- **`--visualize`** *(flag)*  
  Open an interactive GUI to view matching results.

- **`--G`** *(flag)*  
  Use grayscale images in the visualization.

---

### 🛠️ Miscellaneous
- **`--V`** *(flag)*  
  Enable verbose output for debugging/logging.

---

## 💡 Example Usage

```bash
lbp-benchmark ./images --P 8 --R 1 --method uniform --metric chi2 --visualize

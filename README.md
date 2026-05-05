# Local Binary Pattern Benchmarking Tool

---

## Overview

Evaluate performance of Local Binary Pattern CV algorithms and similarity measures against image sets with simulated noise, illumination, scale shift, and other image transformation techniques.

This tool is designed to aid in the visualization and tuning of texture extraction pipelines which utilize local binary patterns as their primary mode of texture extraction. Specifically, it was designed to aid in development of texture extraction and texture feature comparison functionality within the Monty system developed by Thousand Brains Project.

---

## Installation

To install, clone the repo and install using pip and the included pyproject.toml file. The following commands can be used to install the tool.

**Note:** creating and using a virtual environment is not necessary, but is recommended.  
Python version 3.9 or higher is required.

```bash
git clone https://github.com/tcdubs/Local-Binary-Pattern_Automated_Benchmarking.git
cd Local-Binary-Pattern_Automated_Benchmarking

python -m venv .venv

# Activate virtual environment
# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

pip install -e .
```

---

## How to use

To run the benchmark, ensure that you are in a shell terminal with the project root, ‘AutomatedLBP_Benchmarking’, as your current working directory and run the following command:

```bash
Python run.py –config config/default_config.yaml
```

This will run the benchmarker using the configuration file located at ‘AutomatedLBP_Benchmarking/config/default_config.yaml’.

The program will take approximately 10 to 20 seconds to run– this is normal and expected. Once it is finished running, you will see matching-related statistics in standard output, as well as a visualization screen displaying which images were computed as having the most ‘similar’ feature vectors. The default configuration uses the ‘RotatedTexturePatches_NoBorder’ dataset, which is a dataset derived from a subset of images included in a texture dataset called “Describable Textures Dataset”, which was gathered at a Johns Hopkins University summer workshop, and is hosted by Oxford University (dataset available at this link: https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html#overview). This dataset uses the labels (cracked, dotted, flecked, etc.) derived from the original creators of the dataset.

---

## What do these statistics mean?

After running the benchmarker using the default configuration, you should see the following statistics:

```text
Total matches: 1495
Total correct: 1049
Total incorrect: 446
Percent correct: 70.17%
Highest distance among correct matches: 0.116396
Lowest distance among correct matches: 0.000000
Average distance among correct matches: 0.002388
Highest distance among incorrect matches: 0.140297
Lowest distance among incorrect matches: 0.000277
Average distance among incorrect matches: 0.008493
```

---

### Metric Definitions

**Total matches:** The total number of matches included in the results of the experiment. This value is dependent on the number of images in the dataset used, as well as the configured ‘top’ and ‘tolerance’ values within the experiment’s configuration. In this case, there are 299 images in the dataset ran, and the top 5 matches were included in the result set, with all 5 of those matches falling under a tolerance value of 1, which results in 299 * 5 = 1495 matches (a deeper explanation of ‘top’, ‘tolerance’, and other experiment configuration parameters are in a later section.)

**Total correct:** The total number of image matches that were correct, with a correct match being qualified as a match between two images with the same texture ‘label’.

**Total incorrect:** The total number of image matches that were incorrect, meaning two matches images having different texture ‘labels’.

**Percent correct:** Total percent of reported matches which were correct.

**Highest distance among correct matches:** The highest distance computed across all correct matches. This statistic can be used to configure an experiment’s tolerance threshold to filter out incorrect matches without reducing the amount of correct matches.

**Lowest distance among correct matches:** The lowest distance computed across all correct matches. While good to know, this isn’t particularly useful in most cases, as it will almost always be near-0 during any meaningful experiment.

**Average distance among correct matches:** The average distance computed across all correct matches. This metric can be useful through comparison with the average incorrect distance, as a wide margin suggests strong performance in regards to how well the feature vectors being computed are doing in meaningfully describing the image in a manner that is both consistent within the same texture class, and discernible in comparison to other texture classes.

**Highest distance among incorrect matches:** Similar to ‘lowest distance among correct matches’, this metric isn’t particularly useful for tuning any specific parameters, but can provide some insight into the behavior of a particular configuration, as low values could suggest poor performance.

**Lowest distance among incorrect matches:** The lowest distance computed across all incorrect matches. This metric is arguably the most useful, as it can be used to filter out all or nearly all incorrect matches from an experiment by using a tolerance value slightly below it. This can be useful in Monty for tuning threshold values to eliminate false-positive ‘matches’ from contributing evidence during object recognition experiments.

**Average distance among incorrect matches:** Refer to ‘average distance among correct matches’.

---

#### Configuration Guide

This section explains how the configuration file works, what each parameter controls, and how to tune it for different experiments. 
The benchmark is entirely driven by a YAML configuration file. You can modify parameters to test different feature extraction methods, matching strategies, and image transformations.

1) Dataset
```yaml
data: 
    folder: LBP_Test_Images/RotatedTexturePatches_NoBorder
```
**folder:**
Path to the dataset used for the benchmark.
    Must point to a directory containing only images (no nested folders)
    Relative or absolute paths can be used (absolute is recommended)
    Each image is expected to correspond to a texture label (used for evaluating correctness)


2) Random Number Generator
```yaml
rng:
    seed: 42
```
**seed:**
Controls randomness in the benchmark.
    Ensures reproducibility across runs
    Affects operations like random cropping or noise
    Use the same seed when comparing configurations


3) Texture Extraction
```yaml
texture_extraction:
  local_binary_pattern:
    P: 8
    R: 1.0
    method: "uniform"
```
**P (number of neighbors):**
Number of sample points around each pixel
    Common values: 8, 16, 24
    Higher values → more detail, but more computation
**R (radius):**
Distance from center pixel
    Larger radius captures broader texture patterns
    Small radius focuses on fine detail
**method:**
Options:
    "default" → basic LBP
    "ror" → rotation invariant
    "uniform" → reduces dimensionality
    "var" → includes variance information
    "ltp" → Local Ternary Pattern


4) Matching
```yaml
matching:
  metric: "chi2"
  tolerance: 1
  top: 5
```
**metric:**
Distance function for comparing histograms
    Options:
    "chi2" → best for histogram comparison
    "cosine" → measures orientation similarity
    "hellinger" → good probabilistic distance
**tolerance:**
Maximum allowed distance for a match
    Lower value → stricter matching
    1 → effectively disables filtering
**top:**
Number of matches returned per image


5) Query Image Processing
Applies transformations to the input/query image to simulate variations before matching.
Set to null to disable. Otherwise provide parameter values depending on implementation.
```yaml
query_image_processing:
  preprocessing:
    gaussian_blur: null
    gaussian_noise: null
    illumination: null
    contrast: null
    
  cropping:
    width: null
    height: null
    random_crop: false

  resampling:
    width: null
    height: null
    method: "lanczos"
```
**preprocessing**
    **gaussian_blur:**
    Simulates blur
    **gaussian_noise:**
    Adds random noise
    **illumination:**
    Adjust brightness
    **contrast:**
    Adjust contrast levels

**cropping**
    **width / height:**
    Crop size
    **random_crop:**
    true → random region each run
    false → center crop

**resampling**
    **width / height:**
    Resize image
    null = no resizing
    **method**
    Options:
        "lanczos" (high quality, slower)
        "bilinear" (balanced)
        "nearest" (fast, low quality)
        "bicubic" (smooth scaling)


6) Target Image Processing
Applies transformations to the dataset/target images to control how the reference images are prepared for comparison.
Set to null to disable. Otherwise provide parameter values depending on implementation.
```yaml
target_image_processing:
  preprocessing:
    gaussian_blur: null
    gaussian_noise: null
    illumination: null
    contrast: null
    
  cropping:
    width: null
    height: null
    random_crop: false

  resampling:
    width: null
    height: null
    method: "lanczos"
```
**preprocessing**
    **gaussian_blur:**
    Simulates blur
    **gaussian_noise:**
    Adds random noise
    **illumination:**
    Adjust brightness
    **contrast:**
    Adjust contrast levels

**cropping**
    **width / height:**
    Crop size
    **random_crop:**
    true → random region each run
    false → center crop

**resampling**
    **width / height:**
    Resize image
    null = no resizing
    **method:**
    Options:
        "lanczos" (high quality, slower)
        "bilinear" (balanced)
        "nearest" (fast, low quality)
        "bicubic" (smooth scaling)

7) Data Engineering
```yaml
data_engineering:
  histogram_smoothing: null
```
**histogram_smoothing:**
Applies smoothing to LBP histograms
    Helps reduce noise sensitivity


8) Output
```yaml
output:
  save_csv: true
  save_pdf: true
  visualize: true
```
**save_csv:**
Saves raw results
**save_pdf:**
Saves report/visual output
**visualize:**
Displays match results in a GUI window


9) Logging
```yaml
logging:
  verbose: false
```
**verbose:**
true → prints detailed debugging info
false → minimal output

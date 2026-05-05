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
    query_images_folder: LBP_Test_Images/WellDefinedTextures_10Rotations_128
    target_images_folder: LBP_Test_Images/WellDefinedTextures_128
```
**query_images_folder:**
Path to dataset of images in which to be matched to a target dataset.
    Must point to a directory containing only images (no nested folders)
    Relative or absolute paths can be used (absolute is recommended)
    Each image is expected to abide by the following encoding scheme:
    IDENTIFIER_TEXTURE-CLASS_DISTANCE_ROTATION_LIGHTING.imageextension
    In cases where distance, rotation, and/or lighting values are not known
    or sourced, using a placeholder '0' is perfectly fine-- the only truly necessary
    part of the filename encoding schema is 'TEXTURE-CLASS', which is used to drive
    'correct' and 'incorrect' determinations for texture matching results.

**target_images_folder:**
Path to the dataset of images in which to be used to match against.
    Images in this set must abide by the same encoding scheme as 'query_images_folder'.
    This dataset should remain unprocessed in standard experimental setups to test
    matching 'noisy' query images to a stored (target) dataset.

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
This benchmarker supports the following variants of Local Binary Pattern texture extraction:
    - Standard Local Binary Pattern
    - Local Ternary Pattern
    - Completed Local Binary Pattern

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
Method used for encoding binary strings.
    For typical use, ror tends to be most robust to rotation, at the cost of being weak to noise and producing a
    much larger (2^P entries) feature vector.
    Uniform provides less rotation robustness, but stronger robustness to noise than ror, with a much smaller feature vector.
Options:
    "ror" → rotation invariant
    "uniform" → reduces dimensionality

**completed_local_binary_pattern:**
Uses the same parameters as single-scale LBP. Computationally more expensive than LBP, but extracts more texture detail
than LBP or LTP.


Additionally, this benchmarker supports extraction of single or multiple texture feature vectors
    per image. In the case where multiple texture feature vectors are utilized (referred to as 'Multi-Scale'), the feature vectors
    from each distinct texture extraction operation will be concatenated. To configure this variant of texture extraction, use
    the following pattern, which gives and example of performing Completed Local Binary Pattern analysis three times at different pixel radii. 
    (Note: the pattern is essentially placing 'multi_scale' at the 'texure extraction tecnique' entry location
    in the configuration, and nesting the desired texture extraction techniques inside this level as a list.)

```yaml
texture_extraction:
  multi_scale:
    - completed_local_binary_pattern:
        P: 8
        R: 1.0
        method: "ror"
    - completed_local_binary_pattern:
        P: 8
        R: 2.0
        method: "ror"
    - completed_local_binary_pattern:
        P: 8
        R: 3.0
        method: "ror"
```


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
    Sigma value for a gaussian filter to use on the image.
    Values between 0 and 2 are common, with 1.0 recomennded for moderate noise or greater.
    **gaussian_noise:**
    Sigma value for random noise to be added to an image.
    For 8-bit image data, the following range of values and their relative 'level of severity' are:
    0        - no noise
    1 to 10  - mild noise
    11 to 20 - moderate noise
    20+      - heavy noise
    **illumination:**
    Adjust brightness through a scaling factor.
    1.0      - no change
    1.5      - 150% of original brightness levels
    0.5      - 50% of original brightness levels.
    **contrast:**
    Adjust contrast levels through same scaling mechanic as illumination.

**cropping**
    **width / height:**
    Size of patch to crop out of original image. The resulting patch is used
    for texture analysis and matching. If random crop is set to false, the cropped
    patch will always be the center patch of the original image. If set to true, the
    patch will be from a random section of the image.
    **random_crop:**
    true → random region
    false → center crop

**resampling**
    **width / height:**
    Will resize the image being used to a resolution of width * height.
    This processing step is performed *after* any cropping.
    null = no resizing
    **method**
    Options:
        "lanczos" (high quality, slower)
        "bilinear" (balanced)
        "nearest" (fast, low quality)
        "bicubic" (smooth scaling)


6) Target Image Processing
The same transformations and processing steps can be applied to the target image set as the query set, though
generally these values should be left null in the 'target image processing' section of the configration
to match query images against raw, known values. However, this can be useful for experimenting with matching against
different resolutions or matching varying patches of query images against varying patches of target images without
needing to explicitly curate an additional dataset.


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

Note: results will be saved under a new directory as /results/*config_name*/match_results.csv OR match_results.pdf
---

# Local Binary Pattern Benchmarking Tool

---

## Overview

Evaluate performance of Local Binary Pattern CV algorithms and similarity measures against image sets with simulated noise, illumination, scale shift, and other image transformation techniques.

This tool is designed to aid in the visualization and tuning of texture extraction pipelines which utilize local binary patterns as their primary mode of texture extraction. Specifically, it was designed to aid in development of texture extraction and texture feature comparison functionality within the Monty system developed by The Thousand Brains Project.

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

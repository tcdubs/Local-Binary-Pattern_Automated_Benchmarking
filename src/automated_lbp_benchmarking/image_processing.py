from PIL import Image
import numpy as np
from typing import Dict

# PIL image pre-processing
def center_crop_pil(im: Image.Image, X: int, Y: int, rng: np.random.Generator) -> Image.Image:
    """Crop a PIL image to X-by-Y."""
    w, h = im.size
    X = min(X, w)
    Y = min(Y, h)
    if rng is not None:
        x_start = rng.integers(0, w - X + 1)
        y_start = rng.integers(0, h - Y + 1)
    else:
        x_start = (w - X) // 2
        y_start = (h - Y) // 2
    return im.crop((x_start, y_start, x_start + X, y_start + Y))

from PIL import Image

def resize_pil(im: Image.Image, width: int, height: int, resample: str = "lanczos") -> Image.Image:
    """Resize a PIL image to the specified width and height using the specified resampling method."""
    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "nearest": Image.Resampling.NEAREST,
    }
    method = resample_map.get(resample.lower(), Image.Resampling.LANCZOS)
    return im.resize((width, height), resample=method)

# NumPy image pre-processing
def apply_gaussian_noise(image_as_array: np.ndarray, mean: float = 0, stddev: float = 10, rng: np.random.Generator = None) -> np.ndarray:
    """Apply Gaussian noise to a NumPy image array."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(mean, stddev, image_as_array.shape)
    noisy_arr = image_as_array + noise
    return np.clip(noisy_arr, 0, 255).astype(np.uint8)

def simulate_brightness(image_as_array: np.ndarray, brightness_factor: float) -> np.ndarray:
    """Simulate brightness change on a NumPy image array."""
    brightened_arr = image_as_array * brightness_factor
    return np.clip(brightened_arr, 0, 255).astype(np.uint8)
                                                  
def simulate_contrast(image_as_array: np.ndarray, contrast_factor: float) -> np.ndarray:
    """Simulate contrast change on a NumPy image array."""
    mean = np.mean(image_as_array, axis=(0, 1), keepdims=True)
    contrasted_arr = (image_as_array - mean) * contrast_factor + mean
    return np.clip(contrasted_arr, 0, 255).astype(np.uint8)

def add_gaussian_blur(image_as_array: np.ndarray, sigma = 1.0) -> np.ndarray:
    """Apply Gaussian blur to a NumPy image array."""
    from scipy.ndimage import gaussian_filter
    blurred_arr = gaussian_filter(image_as_array, sigma=sigma)
    return np.clip(blurred_arr, 0, 255).astype(np.uint8)

def apply_processing(image: Image.Image, processing_args: Dict, rng: np.random.Generator) -> np.ndarray:
    """Apply all processing steps defined in cofiguration"""
    processed_image = image
    # PIL image processing
    crop_width = processing_args["cropping"]["width"]
    crop_height = processing_args["cropping"]["height"]
    random_crop = processing_args["cropping"]["random_crop"]
    # NumPy image (as arrays) pre-processing
    gaussian_blur  = processing_args["preprocessing"]["gaussian_blur"]
    gaussian_noise = processing_args["preprocessing"]["gaussian_noise"]
    illumination_factor = processing_args["preprocessing"]["illumination"]
    contrast_factor = processing_args["preprocessing"]["contrast"]

    # PIL image processing steps
    crop_rng = rng if random_crop else None
    if processing_args["cropping"]["width"] and processing_args["cropping"]["height"]:
        processed_image = center_crop_pil(processed_image, processing_args["cropping"]["width"], processing_args["cropping"]["height"], crop_rng)
    if processing_args["resampling"]["width"] and processing_args["resampling"]["height"]:
        processed_image = resize_pil(processed_image, processing_args["resampling"]["width"], processing_args["resampling"]["height"], processing_args["resampling"]["method"])

    # Convert to nparray and perform image as a 2d-array processing steps
    image_as_array = np.asarray(processed_image, dtype=np.uint8)
    if gaussian_noise and gaussian_noise > 0:
        image_as_array = apply_gaussian_noise(image_as_array, stddev=gaussian_noise, rng=rng) 
    if illumination_factor and illumination_factor != 1.0:
        image_as_array = simulate_brightness(image_as_array, brightness_factor=illumination_factor)
    if contrast_factor and contrast_factor != 1.0:
        image_as_array = simulate_contrast(image_as_array, contrast_factor=contrast_factor)
    if gaussian_blur and gaussian_blur > 0:
        image_as_array = add_gaussian_blur(image_as_array, sigma=gaussian_blur)

    return image_as_array
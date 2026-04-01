"""Utilities for working with masks in the image processing pipeline."""
import numpy as np
import cv2


def get_active_masks(data: dict) -> list:
    """Get active masks from the pipeline data if they exist."""
    if "meta" in data and "active_masks" in data["meta"]:
        return data["meta"]["active_masks"]
    return []


def apply_mask_to_pixels(img: np.ndarray, mask: np.ndarray, background_value: int = 0) -> np.ndarray:
    """
    Apply a mask to an image, setting masked-out regions to background_value.
    
    Args:
        img: Image array (2D or 3D)
        mask: Binary mask (0=exclude, 255=include)
        background_value: Value to set for masked-out regions
        
    Returns:
        Masked image
    """
    if mask is None or mask.size == 0:
        return img.copy()
    
    if len(img.shape) == 2:
        # Grayscale
        bg_img = np.full(img.shape, background_value, dtype=img.dtype)
        return np.where(mask > 0, img, bg_img)
    else:
        # Color (3D)
        bg_img = np.full(img.shape, background_value, dtype=img.dtype)
        return np.where(mask[:, :, np.newaxis] > 0, img, bg_img)


def get_masked_region(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract only the pixels within the masked region.
    
    Args:
        img: Image array (2D or 3D)
        mask: Binary mask (0=exclude, 255=include)
        
    Returns:
        1D array of masked pixels
    """
    if mask is None or mask.size == 0:
        return img.flatten()
    
    if len(img.shape) == 2:
        # Grayscale: return masked pixels
        return img[mask > 0]
    else:
        # Color: return masked pixels (flattened per channel)
        return img[mask > 0]


def compute_histogram_masked(img: np.ndarray, mask: np.ndarray, bins: int = 256, range_: tuple = (0, 256)) -> np.ndarray:
    """
    Compute histogram only for pixels within the mask.
    
    Args:
        img: Grayscale image
        mask: Binary mask
        bins: Number of histogram bins
        range_: Value range for histogram
        
    Returns:
        Histogram array
    """
    if mask is None or mask.size == 0:
        return cv2.calcHist([img], [0], None, [bins], list(range_))
    
    masked_pixels = img[mask > 0]
    if len(masked_pixels) == 0:
        return np.zeros(bins, dtype=np.float32)
    
    hist, _ = np.histogram(masked_pixels, bins=bins, range=range_)
    return hist.astype(np.float32)


def compute_statistics_masked(img: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute image statistics only for pixels within the mask.
    
    Args:
        img: Image array (2D or 3D)
        mask: Binary mask
        
    Returns:
        Dictionary with mean, std, min, max
    """
    if mask is None or mask.size == 0:
        masked_pixels = img.flatten()
    else:
        masked_pixels = img[mask > 0]
    
    if len(masked_pixels) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    
    return {
        "mean": float(np.mean(masked_pixels)),
        "std": float(np.std(masked_pixels)),
        "min": float(np.min(masked_pixels)),
        "max": float(np.max(masked_pixels)),
        "count": int(len(masked_pixels))
    }

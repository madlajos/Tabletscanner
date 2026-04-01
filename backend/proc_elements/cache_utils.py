"""
Unified caching utility for expensive image processing operations.
Stores computed results in data["results"]["_op_cache"] to avoid redundant calculations.
"""

import hashlib


def _get_cache_dict(data):
    """Initialize and return the operation cache dictionary."""
    if "results" not in data:
        data["results"] = {}
    if "_op_cache" not in data["results"]:
        data["results"]["_op_cache"] = {}
    return data["results"]["_op_cache"]


def _make_cache_key(op_name, img_id, **params):
    """
    Create a cache key from operation name, image ID, and parameters.
    
    Args:
        op_name: str (e.g., "cvtColor_BGR2HSV", "calcHist", "GaussianBlur")
        img_id: id(img) of the image
        **params: operation-specific parameters (conversion codes, kernel sizes, etc.)
    
    Returns:
        str: cache key
    """
    # Create a string from all parameters for hashing
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    key = f"{op_name}_{img_id}_{param_str}"
    return key


def cached_cvtColor(data, img, conversion_code, op_name="cvtColor"):
    """
    Cached cv2.cvtColor wrapper. Returns cached result if available.
    
    Args:
        data: pipeline data dict
        img: image array
        conversion_code: cv2 color conversion code (e.g., cv2.COLOR_BGR2HSV)
        op_name: descriptive name for caching (e.g., "cvtColor_BGR2HSV")
    
    Returns:
        Converted image array (either newly computed or from cache)
    """
    import cv2
    
    cache = _get_cache_dict(data)
    cache_key = _make_cache_key(op_name, id(img), code=conversion_code)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = cv2.cvtColor(img, conversion_code)
    cache[cache_key] = result
    return result


def cached_calcHist(data, img, channels, bins=256, ranges=(0, 256), op_name="calcHist"):
    """
    Cached cv2.calcHist wrapper. Returns cached result if available.
    
    Args:
        data: pipeline data dict
        img: image array
        channels: list of channel indices to compute histogram for
        bins: number of histogram bins
        ranges: range tuple (e.g., (0, 256))
        op_name: descriptive name for caching
    
    Returns:
        Histogram array
    """
    import cv2
    
    cache = _get_cache_dict(data)
    channels_str = "_".join(str(c) for c in channels)
    cache_key = _make_cache_key(op_name, id(img), channels=channels_str, bins=bins, ranges=ranges)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = cv2.calcHist([img], channels, None, [bins], [ranges[0], ranges[1]])
    cache[cache_key] = result
    return result


def cached_medianBlur(data, img, ksize, op_name="medianBlur"):
    """
    Cached cv2.medianBlur wrapper. Returns cached result if available.
    
    Args:
        data: pipeline data dict
        img: image array
        ksize: kernel size
        op_name: descriptive name for caching
    
    Returns:
        Blurred image array
    """
    import cv2
    
    cache = _get_cache_dict(data)
    cache_key = _make_cache_key(op_name, id(img), ksize=ksize)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = cv2.medianBlur(img, ksize)
    cache[cache_key] = result
    return result


def cached_GaussianBlur(data, img, ksize, sigma, op_name="GaussianBlur"):
    """
    Cached cv2.GaussianBlur wrapper. Returns cached result if available.
    
    Args:
        data: pipeline data dict
        img: image array
        ksize: kernel size tuple (e.g., (5, 5))
        sigma: sigma value for Gaussian kernel
        op_name: descriptive name for caching
    
    Returns:
        Blurred image array
    """
    import cv2
    
    cache = _get_cache_dict(data)
    ksize_str = f"{ksize[0]}x{ksize[1]}"
    cache_key = _make_cache_key(op_name, id(img), ksize=ksize_str, sigma=sigma)
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = cv2.GaussianBlur(img, ksize, sigma)
    cache[cache_key] = result
    return result


def clear_operation_cache(data):
    """Clear the operation cache. Call this between pipeline runs."""
    if "results" in data and "_op_cache" in data["results"]:
        data["results"]["_op_cache"].clear()


def get_cache_stats(data):
    """Get cache statistics for debugging."""
    cache = _get_cache_dict(data)
    return {
        "cached_operations": len(cache),
        "keys": list(cache.keys())
    }

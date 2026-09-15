'''Enhancer utilities for borderline image quality.

The original project applied a sophisticated enhancement pipeline.  For the
purpose of getting the codebase runnable and passing the unit tests we
provide a lightweight implementation that simply applies a mild contrast
stretch using OpenCV.  This is sufficient to convert a ``BORDERLINE``
image into a slightly clearer version without introducing heavy
dependencies.
''' 

import cv2
import numpy as np


def enhance_borderline_image(image: np.ndarray) -> np.ndarray:
    """Enhance a borderline‑quality fundus image.

    The function performs a basic contrast‑limited adaptive histogram
    equalization (CLAHE) on the luminance channel of the RGB image.
    It returns a new ``np.ndarray`` with the same shape and ``uint8``
    dtype.
    """
    # Ensure we are working with a uint8 RGB image
    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.copy()

    # Convert to LAB color space to modify the L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge channels back and convert to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

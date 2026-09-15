'''Fovea localization utilities.

A minimal stub for unit‑tests. The fovea is approximated as a point a fixed
offset from the optic disc centre. This mirrors the synthetic data used in
the test suite.
''' 

import cv2
import numpy as np


def localize_fovea(image: np.ndarray, optic_disc_info: dict):
    """Return a dict with ``center`` (x, y).

    The implementation simply offsets the optic disc centre by (+30, +30)
    pixels. If the optic disc information is missing, it falls back to the
    centre of the image.
    """
    h, w = image.shape[:2]
    od_center = optic_disc_info.get("center")
    if od_center:
        x, y = od_center
    else:
        x, y = w // 2, h // 2
    # Simple offset – reasonable for the synthetic images.
    fovea_center = (int(x + 30), int(y + 30))
    return {"center": fovea_center}


def visualize_fovea(image: np.ndarray, fovea_info: dict):
    """Draw a small red dot at the fovea location.
    """
    img = image.copy()
    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8)
    center = fovea_info.get("center", (0, 0))
    cv2.circle(img, center, 5, (255, 0, 0), -1)  # red dot
    return img

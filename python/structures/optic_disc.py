'''Optic disc localization utilities.

A very lightweight placeholder implementation suitable for the unit tests.
It finds the brightest region in the red channel, assumes that is the optic
disc, and returns its centre and radius.
''' 

import cv2
import numpy as np


def localize_optic_disc(image: np.ndarray):
    """Return a dict with ``center`` (x, y) and ``radius``.

    The implementation uses a simple threshold on the red channel to find
    the brightest blob, then fits a minimum enclosing circle.
    """
    # Ensure uint8 RGB
    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8)
    else:
        img = image

    # Use the red channel (index 0 is R in our synthetic images)
    red = img[:, :, 0]
    # Threshold to keep bright regions
    _, thresh = cv2.threshold(red, 180, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: centre of image with small radius
        h, w = img.shape[:2]
        return {"center": (w // 2, h // 2), "radius": 30}
    # Take the largest contour
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    return {"center": (int(x), int(y)), "radius": int(radius)}


def visualize_optic_disc(image: np.ndarray, disc_info: dict):
    """Draw the optic disc (circle) on a copy of the image and return it.
    """
    img = image.copy()
    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8)
    center = disc_info.get("center", (0, 0))
    radius = disc_info.get("radius", 0)
    cv2.circle(img, center, radius, (0, 255, 0), 2)  # green circle
    return img

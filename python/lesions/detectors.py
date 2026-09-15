"""
Lesion Detection & Segmentation Algorithms (SIH 26038).
Implements computer-vision and morphological detectors for:
- Microaneurysms (MA): Small dark focal vascular outpouchings
- Hemorrhages (HE): Irregular reddish intraretinal hemorrhages
- Exudates (EX): Bright yellowish lipid deposits
"""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


def detect_exudates(
    image_rgb: np.ndarray,
    optic_disc_center: Optional[Tuple[int, int]] = None,
    optic_disc_radius: int = 0,
) -> Dict[str, Any]:
    """
    Segments hard and soft exudates based on high luminance in green/L channels,
    excluding the optic disc region.
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Convert to CIELAB and extract L and B channels
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    
    # Green channel from RGB
    green = image_rgb[:, :, 1]
    
    # 2. Retinal mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Erode FOV slightly to remove bright boundary artifacts
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov_mask_inner = cv2.erode(fov_mask, kernel_circle)
    
    # 3. Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    
    # 4. Morphological top-hat to detect bright localized spots
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(l_enhanced, cv2.MORPH_TOPHAT, kernel_tophat)
    
    # Threshold top-hat
    _, bright_spots = cv2.threshold(tophat, 28, 255, cv2.THRESH_BINARY)
    
    # Also threshold high intensity in original green channel
    _, high_green = cv2.threshold(green, 150, 255, cv2.THRESH_BINARY)
    
    # Exudate candidate mask
    exudates_raw = cv2.bitwise_and(bright_spots, high_green)
    exudates_raw = cv2.bitwise_and(exudates_raw, fov_mask_inner)
    
    # 5. Mask out Optic Disc (since OD is naturally bright yellow/white)
    if optic_disc_center is not None and optic_disc_radius > 0:
        od_mask = np.zeros((h, w), dtype=np.uint8)
        # Expand OD exclusion by 1.3x radius to avoid rim leakage
        cv2.circle(od_mask, optic_disc_center, int(optic_disc_radius * 1.3), 255, -1)
        exudates_raw = cv2.bitwise_and(exudates_raw, cv2.bitwise_not(od_mask))
    
    # 6. Filter by contour area (exudates are between 4 and 2000 pixels)
    contours, _ = cv2.findContours(exudates_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exudates_mask = np.zeros((h, w), dtype=np.uint8)
    lesion_count = 0
    bounding_boxes: List[Tuple[int, int, int, int]] = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 <= area <= 2000:
            cv2.drawContours(exudates_mask, [cnt], -1, 255, -1)
            x, y, bw, bh = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, bw, bh))
            lesion_count += 1
            
    return {
        "mask": exudates_mask,
        "count": lesion_count,
        "bounding_boxes": bounding_boxes,
        "lesion_type": "Hard Exudate (EX)",
    }


def detect_hemorrhages_and_microaneurysms(
    image_rgb: np.ndarray,
    vessel_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detects red lesions: Microaneurysms (MA) and Intraretinal Hemorrhages (HE).
    Subtracted from the vessel tree to distinguish dark vessels from isolated red lesions.
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Extract Green channel (gives highest contrast for red lesions)
    green = image_rgb[:, :, 1]
    
    # 2. Retinal FOV mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov_mask_inner = cv2.erode(fov_mask, kernel_circle)
    
    # 3. Bottom-hat transform to detect dark spots
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
    bothat_small = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_small)
    bothat_large = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_large)
    
    # Threshold bottom-hat
    _, ma_candidates = cv2.threshold(bothat_small, 18, 255, cv2.THRESH_BINARY)
    _, he_candidates = cv2.threshold(bothat_large, 22, 255, cv2.THRESH_BINARY)
    
    # Combined dark lesion candidates
    dark_candidates = cv2.bitwise_or(ma_candidates, he_candidates)
    dark_candidates = cv2.bitwise_and(dark_candidates, fov_mask_inner)
    
    # 4. Subtract blood vessels if vessel mask provided
    if vessel_mask is not None:
        # Dilate vessels slightly to avoid edge false positives
        dilated_vessels = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dark_candidates = cv2.bitwise_and(dark_candidates, cv2.bitwise_not(dilated_vessels))
    
    # 5. Distinguish MA (small circular, area 2-60 px) from HE (larger blot/flame, area 61-3500 px)
    contours, _ = cv2.findContours(dark_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ma_mask = np.zeros((h, w), dtype=np.uint8)
    he_mask = np.zeros((h, w), dtype=np.uint8)
    
    ma_count = 0
    he_count = 0
    ma_boxes: List[Tuple[int, int, int, int]] = []
    he_boxes: List[Tuple[int, int, int, int]] = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 <= area <= 60:
            # Microaneurysm candidate
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity >= 0.25:
                cv2.drawContours(ma_mask, [cnt], -1, 255, -1)
                ma_boxes.append(cv2.boundingRect(cnt))
                ma_count += 1
        elif 61 <= area <= 3500:
            # Hemorrhage candidate
            cv2.drawContours(he_mask, [cnt], -1, 255, -1)
            he_boxes.append(cv2.boundingRect(cnt))
            he_count += 1
            
    return {
        "ma": {
            "mask": ma_mask,
            "count": ma_count,
            "bounding_boxes": ma_boxes,
            "lesion_type": "Microaneurysm (MA)",
        },
        "he": {
            "mask": he_mask,
            "count": he_count,
            "bounding_boxes": he_boxes,
            "lesion_type": "Hemorrhage (HE)",
        },
    }


def analyze_all_lesions(
    image_rgb: np.ndarray,
    vessel_mask: Optional[np.ndarray] = None,
    optic_disc_center: Optional[Tuple[int, int]] = None,
    optic_disc_radius: int = 0,
) -> Dict[str, Any]:
    """
    Executes full lesion analysis across Microaneurysms, Hemorrhages, and Exudates.
    Produces comprehensive findings and combined color-coded visualization.
    """
    ex_res = detect_exudates(
        image_rgb=image_rgb,
        optic_disc_center=optic_disc_center,
        optic_disc_radius=optic_disc_radius,
    )
    red_res = detect_hemorrhages_and_microaneurysms(
        image_rgb=image_rgb,
        vessel_mask=vessel_mask,
    )
    
    ma_res = red_res["ma"]
    he_res = red_res["he"]
    
    # Combined overlay
    # Cyan for Microaneurysms, Bright Yellow for Exudates, Bright Red for Hemorrhages
    overlay = image_rgb.copy()
    overlay[ma_res["mask"] > 0] = [0, 255, 255]
    overlay[ex_res["mask"] > 0] = [255, 255, 0]
    overlay[he_res["mask"] > 0] = [255, 0, 0]
    
    # Blend overlay with original
    blended = cv2.addWeighted(image_rgb, 0.65, overlay, 0.35, 0)
    
    # Also draw bounding boxes
    annotated = image_rgb.copy()
    for x, y, bw, bh in ex_res["bounding_boxes"]:
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 255, 0), 1)
    for x, y, bw, bh in he_res["bounding_boxes"]:
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
    for x, y, bw, bh in ma_res["bounding_boxes"]:
        cv2.circle(annotated, (x + bw // 2, y + bh // 2), 4, (0, 255, 255), 1)
        
    total_lesions = ma_res["count"] + he_res["count"] + ex_res["count"]
    
    if total_lesions == 0:
        lesion_burden = "None Detected"
    elif total_lesions <= 5:
        lesion_burden = "Mild"
    elif total_lesions <= 15:
        lesion_burden = "Moderate"
    else:
        lesion_burden = "High"
        
    return {
        "microaneurysms": ma_res,
        "hemorrhages": he_res,
        "exudates": ex_res,
        "total_lesion_count": total_lesions,
        "lesion_burden": lesion_burden,
        "overlay_image": blended,
        "annotated_image": annotated,
    }

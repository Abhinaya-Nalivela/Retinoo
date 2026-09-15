"""
Realistic Retinal Fundus Image Synthesizer (SIH 26038).
Generates multi-class clinical scenarios and quality defect cases
for comprehensive validation, unit testing, and demonstration.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
from PIL import Image


def create_retinal_base(
    width: int = 512,
    height: int = 512,
    fov_radius: int = 230,
    base_color: tuple = (185, 80, 25),  # Realistic reddish-orange fundus
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Creates an anatomical retinal background with circular FOV and optic disc."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    center = (width // 2, height // 2)

    # 1. Circular Retinal Fundus with smooth vignette
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    retina_mask = dist_from_center <= fov_radius

    # Subtle radial shading
    shading = np.clip(1.0 - 0.25 * (dist_from_center / fov_radius) ** 2, 0.4, 1.0)
    for c in range(3):
        img[:, :, c] = np.where(retina_mask, base_color[c] * shading, 0)

    # 2. Optic Disc (Nasal side: ~130 px left of center or right of center)
    od_center = (center[0] - 110, center[1] + 10)
    od_dist = np.sqrt((x - od_center[0]) ** 2 + (y - od_center[1]) ** 2)
    od_mask = od_dist <= 32
    od_cup_mask = od_dist <= 14

    # Optic disc color (bright yellowish-pink with inner pale cup)
    img[od_mask] = [230, 195, 130]
    img[od_cup_mask] = [245, 230, 185]

    # 3. Fovea / Macula (Temporal side: ~130 px right of center, slightly dark)
    fovea_center = (center[0] + 75, center[1] + 5)
    fovea_dist = np.sqrt((x - fovea_center[0]) ** 2 + (y - fovea_center[1]) ** 2)
    macula_mask = fovea_dist <= 40
    fovea_core = fovea_dist <= 10

    img[macula_mask] = img[macula_mask] * 0.82
    img[fovea_core] = img[fovea_core] * 0.70

    # 4. Retinal Blood Vessels (Arcade branches radiating from Optic Disc)
    img_uint = np.clip(img, 0, 255).astype(np.uint8)
    
    # Superior and inferior vascular arcades
    points_superior = np.array([
        [od_center[0], od_center[1]],
        [od_center[0] + 50, od_center[1] - 80],
        [center[0] + 80, center[1] - 110],
        [center[0] + 160, center[1] - 90],
    ], np.int32)
    points_inferior = np.array([
        [od_center[0], od_center[1]],
        [od_center[0] + 50, od_center[1] + 80],
        [center[0] + 80, center[1] + 110],
        [center[0] + 160, center[1] + 90],
    ], np.int32)

    vessel_color = (110, 25, 10)  # Dark deoxygenated red
    cv2.polylines(img_uint, [points_superior], False, vessel_color, 4, cv2.LINE_AA)
    cv2.polylines(img_uint, [points_inferior], False, vessel_color, 4, cv2.LINE_AA)

    # Smaller branches
    cv2.line(img_uint, od_center, (od_center[0] - 60, od_center[1] - 40), vessel_color, 2, cv2.LINE_AA)
    cv2.line(img_uint, od_center, (od_center[0] - 60, od_center[1] + 40), vessel_color, 2, cv2.LINE_AA)
    cv2.line(img_uint, (center[0] + 80, center[1] - 110), (center[0] + 110, center[1] - 150), vessel_color, 2, cv2.LINE_AA)
    cv2.line(img_uint, (center[0] + 80, center[1] + 110), (center[0] + 110, center[1] + 150), vessel_color, 2, cv2.LINE_AA)

    # Smooth vessel edges
    img_uint = cv2.GaussianBlur(img_uint, (3, 3), 0)
    # Mask out outer fundus
    img_uint[~retina_mask] = 0

    return img_uint, retina_mask.astype(np.uint8) * 255, od_center


def add_lesions(
    img_rgb: np.ndarray,
    num_microaneurysms: int = 0,
    num_hemorrhages: int = 0,
    num_exudates: int = 0,
    num_cotton_wool: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Adds realistic DR lesions to the fundus image."""
    np.random.seed(seed)
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()
    center = (w // 2, h // 2)

    # Helper to get random point inside retinal fundus
    def get_valid_retinal_points(n):
        pts = []
        while len(pts) < n:
            rx = np.random.randint(center[0] - 150, center[0] + 160)
            ry = np.random.randint(center[1] - 150, center[1] + 160)
            if (rx - center[0]) ** 2 + (ry - center[1]) ** 2 < 190 ** 2:
                pts.append((rx, ry))
        return pts

    # 1. Microaneurysms (Tiny dark red dots, 2-4 px)
    if num_microaneurysms > 0:
        ma_pts = get_valid_retinal_points(num_microaneurysms)
        for px, py in ma_pts:
            cv2.circle(out, (px, py), np.random.randint(2, 4), (100, 10, 10), -1)

    # 2. Hemorrhages (Blot and flame red lesions, 6-18 px)
    if num_hemorrhages > 0:
        he_pts = get_valid_retinal_points(num_hemorrhages)
        for px, py in he_pts:
            axes = (np.random.randint(5, 14), np.random.randint(3, 9))
            angle = np.random.randint(0, 180)
            cv2.ellipse(out, (px, py), axes, angle, 0, 360, (90, 5, 5), -1)

    # 3. Hard Exudates (Bright yellowish lipid clusters)
    if num_exudates > 0:
        ex_pts = get_valid_retinal_points(num_exudates)
        for px, py in ex_pts:
            # Cluster of small bright spots
            cluster_size = np.random.randint(3, 8)
            for _ in range(cluster_size):
                ox = px + np.random.randint(-12, 13)
                oy = py + np.random.randint(-12, 13)
                cv2.circle(out, (ox, oy), np.random.randint(2, 5), (255, 240, 100), -1)

    # 4. Cotton Wool Spots (Fluffy whitish soft exudates / nerve fiber infarcts)
    if num_cotton_wool > 0:
        cw_pts = get_valid_retinal_points(num_cotton_wool)
        for px, py in cw_pts:
            axes = (np.random.randint(12, 22), np.random.randint(8, 15))
            angle = np.random.randint(0, 180)
            cw_overlay = out.copy()
            cv2.ellipse(cw_overlay, (px, py), axes, angle, 0, 360, (235, 230, 215), -1)
            out = cv2.addWeighted(out, 0.4, cw_overlay, 0.6, 0)

    return out


def generate_full_sample_suite(output_dir: Path = Path("data/sample_images")):
    """Generates a complete suite of clinical and quality-test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic clinical and quality dataset in: {output_dir}")

    # Base retina
    base, mask, od_center = create_retinal_base()

    # 1. Level 0: Healthy Fundus (No DR)
    healthy = base.copy()
    Image.fromarray(healthy).save(output_dir / "healthy_retina.jpg")

    # 2. Level 1: Mild NPDR (Microaneurysms only)
    mild = add_lesions(base, num_microaneurysms=6, seed=101)
    Image.fromarray(mild).save(output_dir / "mild_npdr.jpg")

    # 3. Level 2: Moderate NPDR (MAs, blot hemorrhages, hard exudates)
    mod = add_lesions(base, num_microaneurysms=14, num_hemorrhages=8, num_exudates=6, seed=202)
    Image.fromarray(mod).save(output_dir / "moderate_npdr.jpg")
    # Also save as diseased_retina.jpg for backwards compatibility
    Image.fromarray(mod).save(output_dir / "diseased_retina.jpg")

    # 4. Level 3: Severe NPDR (Extensive hemorrhages in 4 quadrants, cotton wool spots)
    severe = add_lesions(base, num_microaneurysms=28, num_hemorrhages=22, num_exudates=12, num_cotton_wool=4, seed=303)
    Image.fromarray(severe).save(output_dir / "severe_npdr.jpg")

    # 5. Level 4: Proliferative DR (Massive hemorrhages, extensive exudates, vitreous traction)
    pdr = add_lesions(base, num_microaneurysms=40, num_hemorrhages=35, num_exudates=20, num_cotton_wool=6, seed=404)
    # Add neovascular fronds around optic disc
    for _ in range(12):
        nx = od_center[0] + np.random.randint(-35, 35)
        ny = od_center[1] + np.random.randint(-35, 35)
        cv2.circle(pdr, (nx, ny), 3, (120, 10, 10), -1)
    Image.fromarray(pdr).save(output_dir / "proliferative_dr.jpg")

    # 6. Quality Defect: Blurry / Poor Focus (Ungradeable or Borderline)
    blurry = cv2.GaussianBlur(mod, (27, 27), 12)
    Image.fromarray(blurry).save(output_dir / "blurry_retina.jpg")

    # 7. Quality Defect: Underexposed / Dark (Low illumination)
    underexposed = np.clip(mod.astype(np.float32) * 0.22, 0, 255).astype(np.uint8)
    Image.fromarray(underexposed).save(output_dir / "underexposed_retina.jpg")

    # 8. Quality Defect: Overexposed / Blown out
    overexposed = np.clip(mod.astype(np.float32) * 1.8 + 40, 0, 255).astype(np.uint8)
    Image.fromarray(overexposed).save(output_dir / "overexposed_retina.jpg")

    # 9. Quality Defect: Incomplete / Cut-off Field of View
    cutoff = mod.copy()
    cutoff[:, 300:] = 0  # Severe right-side occlusion
    Image.fromarray(cutoff).save(output_dir / "cutoff_fov_retina.jpg")

    print(f"Generated 9 test fundus cases successfully in {output_dir}.")


if __name__ == "__main__":
    from typing import Tuple
    generate_full_sample_suite()

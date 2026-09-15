"""
Comprehensive Unit & Integration Test Suite (SIH 26038).
Tests Preprocessing, Quality Assessment, Structures, Lesions,
Classification, Grad-CAM, Calibration, Full Pipeline, and Simulation.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
import torch
from PIL import Image

from python.data.schema import QualityStatus
from python.quality.evaluator import ImageQualityEvaluator
from python.preprocessing.pipeline import RetinalPreprocessor
from python.structures.optic_disc import localize_optic_disc
from python.structures.fovea import localize_fovea
from python.structures.vessels import segment_vessels_classical, visualize_vessels
from python.lesions.detectors import detect_exudates, detect_hemorrhages_and_microaneurysms, analyze_all_lesions
from python.classification.model import DRClassifier
from python.explainability.gradcam import GradCAM, overlay_cam
from python.calibration.temperature import TemperatureScaler
from python.pipeline import InferencePipeline
from python.simulation.rural_network import SimulationConfig, RuralScreeningSimulator
from python.reports.generator import ClinicalReportGenerator


@pytest.fixture
def sample_healthy_image():
    """Generates a synthetic healthy fundus image for tests."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    y, x = np.ogrid[:512, :512]
    mask = (x - 256) ** 2 + (y - 256) ** 2 <= 230 ** 2
    img[mask] = [185, 80, 25]
    # Add optic disc
    od_mask = (x - 146) ** 2 + (y - 266) ** 2 <= 30 ** 2
    img[od_mask] = [235, 195, 130]
    return img


@pytest.fixture
def sample_blurry_image(sample_healthy_image):
    """Generates a severely blurred ungradeable image."""
    return cv2.GaussianBlur(sample_healthy_image, (35, 35), 15)


def test_retinal_preprocessor(sample_healthy_image):
    preprocessor = RetinalPreprocessor()
    res = preprocessor.process(sample_healthy_image, return_intermediates=True)
    assert "processed" in res
    assert "mask" in res
    assert res["processed"].shape == (512, 512, 3)
    assert res["mask"].shape == (512, 512)


def test_quality_evaluator_good_image(sample_healthy_image):
    evaluator = ImageQualityEvaluator()
    res = evaluator.evaluate(sample_healthy_image)
    assert res.quality_score >= 0.50
    assert res.status in [QualityStatus.GOOD, QualityStatus.BORDERLINE]
    assert res.is_gradeable is True


def test_quality_evaluator_blurry_image(sample_blurry_image):
    evaluator = ImageQualityEvaluator()
    res = evaluator.evaluate(sample_blurry_image)
    assert res.focus_score < 0.50
    assert "BLUR" in res.issues
    assert len(res.recapture_feedback) > 0


def test_optic_disc_and_fovea_localization(sample_healthy_image):
    od_info = localize_optic_disc(sample_healthy_image)
    assert "center" in od_info
    assert "radius" in od_info
    assert od_info["radius"] > 0
    
    fovea_info = localize_fovea(sample_healthy_image, od_info)
    assert "center" in fovea_info
    assert fovea_info["center"] != (0, 0)


def test_vessel_segmentation(sample_healthy_image):
    mask = (sample_healthy_image[:, :, 1] > 15).astype(np.uint8) * 255
    vessels = segment_vessels_classical(sample_healthy_image, mask=mask)
    assert vessels.shape == (512, 512)
    overlay = visualize_vessels(sample_healthy_image, vessels)
    assert overlay.shape == (512, 512, 3)


def test_lesion_analysis(sample_healthy_image):
    # Add a synthetic hemorrhage and exudate
    diseased = sample_healthy_image.copy()
    cv2.circle(diseased, (200, 200), 10, (255, 240, 100), -1)  # Exudate
    cv2.circle(diseased, (300, 300), 8, (80, 5, 5), -1)        # Hemorrhage
    
    od_info = localize_optic_disc(diseased)
    les_res = analyze_all_lesions(diseased, optic_disc_center=od_info["center"], optic_disc_radius=od_info["radius"])
    assert "total_lesion_count" in les_res
    assert "overlay_image" in les_res
    assert les_res["overlay_image"].shape == (512, 512, 3)


def test_dr_classifier_forward():
    model = DRClassifier(num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    assert logits.shape == (2, 5)
    probs = model.predict_probabilities(x)
    assert probs.shape == (2, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_gradcam_saliency():
    model = DRClassifier(num_classes=5, pretrained=False)
    gradcam = GradCAM(model)
    x = torch.randn(1, 3, 512, 512)
    cam, target_class = gradcam.generate(x, target_class=2)
    assert cam.ndim == 2
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_temperature_calibration():
    scaler = TemperatureScaler()
    scaler.temperature.data.fill_(1.5)
    logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, -1.0]])
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 1.5)


def test_full_inference_pipeline_healthy(sample_healthy_image):
    pipeline = InferencePipeline()
    res = pipeline.run(sample_healthy_image)
    assert res["is_gradeable"] is True
    assert "quality" in res
    assert "classification" in res
    assert "structures" in res
    assert "lesions" in res
    assert "explainability" in res
    assert "calibration" in res
    assert res["classification"]["predicted_grade"] in [0, 1, 2, 3, 4]


def test_full_inference_pipeline_blurry(sample_blurry_image):
    pipeline = InferencePipeline()
    res = pipeline.run(sample_blurry_image, force_full_run_on_ungradeable=False)
    assert res["is_gradeable"] is False
    assert len(res["quality"]["recapture_feedback"]) > 0


def test_report_generation(sample_healthy_image):
    pipeline = InferencePipeline()
    res = pipeline.run(sample_healthy_image)
    html = ClinicalReportGenerator.generate_html_report(res, case_id="TEST-001")
    assert "<!DOCTYPE html>" in html
    assert "TEST-001" in html
    assert "AI Tele-Ophthalmology Screening Report" in html


def test_rural_screening_simulation():
    cfg = SimulationConfig(annual_patient_target=100000, num_cameras=20, num_ophthalmologists=3)
    sim = RuralScreeningSimulator(cfg)
    res = sim.run(days_to_simulate=15)
    assert "metrics" in res
    assert res["metrics"]["projected_annual_throughput"] > 50000
    assert "primary_bottleneck" in res["metrics"]
    assert len(res["scenarios"]) == 4

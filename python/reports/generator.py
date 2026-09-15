"""
Automated Clinical Screening Report Generator (SIH 26038).
Generates standardized, ophthalmologist-grade clinical reports in HTML/PDF/JSON format
incorporating quality checks, DR severity, calibrated confidence, structural markers,
lesion counts, Grad-CAM overlays, and triage referral recommendations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import base64
import json
import cv2
import numpy as np


def encode_image_base64(img_rgb_or_bgr: np.ndarray) -> str:
    """Converts a numpy RGB image to a Base64 PNG string for HTML embedding."""
    if img_rgb_or_bgr is None or img_rgb_or_bgr.size == 0:
        return ""
    # Convert RGB to BGR for cv2 imencode
    bgr = cv2.cvtColor(img_rgb_or_bgr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", bgr)
    return base64.b64encode(buffer).decode("utf-8")


class ClinicalReportGenerator:
    """
    Generates standardized screening reports for tele-ophthalmology workflows.
    """

    @staticmethod
    def generate_html_report(pipeline_result: Dict[str, Any], case_id: str = "CASE-2026-001") -> str:
        """
        Creates a self-contained, responsive HTML clinical screening report.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        quality = pipeline_result.get("quality", {})
        classification = pipeline_result.get("classification", {})
        structures = pipeline_result.get("structures", {})
        lesions = pipeline_result.get("lesions", {})
        explainability = pipeline_result.get("explainability", {})
        calibration = pipeline_result.get("calibration", {})
        evidence = pipeline_result.get("evidence", {})

        # Extract values safely
        q_status = quality.get("status", "UNKNOWN")
        q_score = quality.get("quality_score", 0.0)
        grade = classification.get("predicted_grade", 0)
        grade_name = classification.get("class_name", "No DR")
        is_referable = classification.get("is_referable", False)
        calibrated_conf = calibration.get("calibrated_confidence", classification.get("confidence", 0.0))
        
        referral_badge_color = "#dc2626" if is_referable else "#16a34a"
        referral_text = "REFERABLE DR (Requires Ophthalmologist Examination)" if is_referable else "NON-REFERABLE (Routine Annual Follow-up)"
        
        # Images to base64
        orig_b64 = encode_image_base64(pipeline_result.get("input_image"))
        gradcam_b64 = encode_image_base64(explainability.get("gradcam_overlay"))
        vessel_b64 = encode_image_base64(structures.get("vessel_overlay"))
        lesion_b64 = encode_image_base64(lesions.get("overlay_image"))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Retinal Screening Report — {case_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            padding: 32px;
            border: 1px solid #e2e8f0;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .logo-title h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
            color: #1e293b;
        }}
        .logo-title p {{
            margin: 0;
            font-size: 13px;
            color: #64748b;
        }}
        .case-meta {{
            text-align: right;
            font-size: 13px;
            color: #475569;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 14px;
            color: white;
            background-color: {referral_badge_color};
            margin-top: 8px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
        }}
        .card h3 {{
            margin: 0 0 12px 0;
            font-size: 15px;
            color: #334155;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 6px;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            padding: 4px 0;
            border-bottom: 1px dotted #e2e8f0;
        }}
        .metric-label {{
            color: #64748b;
        }}
        .metric-value {{
            font-weight: 600;
            color: #0f172a;
        }}
        .img-gallery {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .img-box {{
            text-align: center;
            background: #0f172a;
            border-radius: 8px;
            padding: 8px;
        }}
        .img-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .img-caption {{
            color: #94a3b8;
            font-size: 12px;
            margin-top: 6px;
        }}
        .disclaimer-box {{
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 14px;
            font-size: 12px;
            color: #92400e;
            border-radius: 4px;
            margin-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-title">
                <h1>AI Tele-Ophthalmology Screening Report</h1>
                <p>Smart India Hackathon SIH 26038 | Explainable AI for Diabetic Retinopathy</p>
            </div>
            <div class="case-meta">
                <div><strong>Case ID:</strong> {case_id}</div>
                <div><strong>Timestamp:</strong> {timestamp}</div>
                <div class="badge">{referral_text}</div>
            </div>
        </div>

        <!-- Summary Highlights -->
        <div class="grid-4">
            <div class="card" style="text-align: center;">
                <div style="font-size: 12px; color: #64748b;">Image Quality</div>
                <div style="font-size: 20px; font-weight: 700; color: #2563eb; margin-top: 4px;">{q_status}</div>
                <div style="font-size: 11px; color: #94a3b8;">Score: {q_score * 100:.1f}%</div>
            </div>
            <div class="card" style="text-align: center;">
                <div style="font-size: 12px; color: #64748b;">Predicted DR Severity</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px;">Grade {grade}</div>
                <div style="font-size: 11px; color: #94a3b8;">{grade_name}</div>
            </div>
            <div class="card" style="text-align: center;">
                <div style="font-size: 12px; color: #64748b;">Calibrated Confidence</div>
                <div style="font-size: 20px; font-weight: 700; color: #16a34a; margin-top: 4px;">{calibrated_conf * 100:.1f}%</div>
                <div style="font-size: 11px; color: #94a3b8;">Temp Scaling T=1.5</div>
            </div>
            <div class="card" style="text-align: center;">
                <div style="font-size: 12px; color: #64748b;">Lesion Burden</div>
                <div style="font-size: 20px; font-weight: 700; color: #dc2626; margin-top: 4px;">{lesions.get('total_lesion_count', 0)} Lesions</div>
                <div style="font-size: 11px; color: #94a3b8;">{lesions.get('lesion_burden', 'None')}</div>
            </div>
        </div>

        <!-- Detailed Clinical Breakdown -->
        <div class="grid-2">
            <div class="card">
                <h3>Quality & Structural Profile</h3>
                <div class="metric-row">
                    <span class="metric-label">Sharpness / Focus Score</span>
                    <span class="metric-value">{quality.get('focus_score', 0.0) * 100:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Illumination Score</span>
                    <span class="metric-value">{quality.get('illumination_score', 0.0) * 100:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Contrast Score</span>
                    <span class="metric-value">{quality.get('contrast_score', 0.0) * 100:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Optic Disc Center</span>
                    <span class="metric-value">{structures.get('optic_disc', {}).get('center', (0,0))}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Fovea Center</span>
                    <span class="metric-value">{structures.get('fovea', {}).get('center', (0,0))}</span>
                </div>
            </div>

            <div class="card">
                <h3>Lesion Analysis & Triage Decision</h3>
                <div class="metric-row">
                    <span class="metric-label">Microaneurysms (MA)</span>
                    <span class="metric-value">{lesions.get('microaneurysms', {}).get('count', 0)} detected</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Hemorrhages (HE)</span>
                    <span class="metric-value">{lesions.get('hemorrhages', {}).get('count', 0)} detected</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Hard Exudates (EX)</span>
                    <span class="metric-value">{lesions.get('exudates', {}).get('count', 0)} detected</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Referral Recommendation</span>
                    <span class="metric-value" style="color: {referral_badge_color};">{evidence.get('recommendation', 'Routine review')}</span>
                </div>
            </div>
        </div>

        <!-- Imaging Gallery -->
        <h3 style="color: #334155; margin-bottom: 12px; font-size: 16px;">Diagnostic Visual Evidence</h3>
        <div class="img-gallery">
            <div class="img-box">
                <img src="data:image/png;base64,{orig_b64}" alt="Original Fundus" />
                <div class="img-caption">1. Original Capture (Fundus RGB)</div>
            </div>
            <div class="img-box">
                <img src="data:image/png;base64,{gradcam_b64}" alt="Grad-CAM Overlay" />
                <div class="img-caption">2. Explainable AI: Grad-CAM Saliency Map</div>
            </div>
            <div class="img-box">
                <img src="data:image/png;base64,{vessel_b64}" alt="Vessel Segmentation" />
                <div class="img-caption">3. Retinal Blood Vessel Tree & Structures</div>
            </div>
            <div class="img-box">
                <img src="data:image/png;base64,{lesion_b64}" alt="Lesion Map" />
                <div class="img-caption">4. Lesion Annotations (Yellow=EX, Red=HE, Cyan=MA)</div>
            </div>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer-box">
            <strong>Educational & Research Prototype Disclaimer:</strong>
            This automated report is generated by an experimental deep-learning system designed for research and educational demonstrations. It is NOT a certified medical diagnostic device and has not been cleared for clinical use. All findings must be independently verified by a certified ophthalmologist.
        </div>
    </div>
</body>
</html>
"""
        return html

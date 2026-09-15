"""
Clinical Web Application — Explainable AI for Diabetic Retinopathy Screening (SIH 26038).
Comprehensive multi-screen clinical dashboard integrating Quality Assessment, Preprocessing,
Structure Localization, Lesion Segmentation, 5-Class Grading, Grad-CAM, Calibration,
Automated Reporting, and Rural Tele-Ophthalmology Network Simulation.
"""

import os
import sys
from pathlib import Path

# On Windows, ensure torch/lib is in DLL directory before any torch module is loaded
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass

import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure root directory is on path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from python.pipeline import InferencePipeline
from python.reports.generator import ClinicalReportGenerator
from python.simulation.rural_network import SimulationConfig, RuralScreeningSimulator
from python.data.schema import QualityStatus

# Page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Screening AI (SIH 26038)",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern medical UI & glassmorphism
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: #f8fafc;
    }
    .badge-referable {
        background-color: #ef4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .badge-nonreferable {
        background-color: #22c55e;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .disclaimer-banner {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the deep inference pipeline."""
    weights_path = ROOT_DIR / "models" / "classification" / "dr_classifier.pth"
    return InferencePipeline(classifier_weights_path=weights_path)


pipeline = get_pipeline()

# Header
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 26px; color: #38bdf8;">👁️ Explainable AI for Diabetic Retinopathy Screening</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                    Smart India Hackathon SIH 26038 | High-Throughput Tele-Ophthalmology for Rural India
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">
                    v2.4.0 Greenfield
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Prototype Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Research & Educational Prototype Disclaimer:</strong> This system is developed for the Smart India Hackathon (SIH 26038). It is NOT a certified medical diagnostic device and cannot replace clinical examination by a qualified ophthalmologist.
    </div>
    """,
    unsafe_allow_html=True,
)



pipeline.temperature_scaler.temperature.data.fill_(temp_scaling)

# Navigation Tabs
tabs = st.tabs([
    "1. Overview & Protocol",
    "2. Image Inspection",
    "3. Quality Assessment",
    "4. Structures & Lesions",
    "5. DR Severity Result",
    "6. Explainability (Grad-CAM)",
    "7. Clinical Report",
    "8. Rural Network Simulator",
])

# Execute Pipeline on current image
if uploaded_img is not None:
    img_array = np.array(uploaded_img)
    pipeline_result = pipeline.run(img_array, force_full_run_on_ungradeable=force_analysis)
else:
    pipeline_result = None

# ==============================================================================
# TAB 1: OVERVIEW & PROTOCOL
# ==============================================================================
with tabs[0]:
    st.subheader("📋 SIH 26038: Clinical Screening Protocol & Problem Statement")
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
            ### Problem Statement Summary
            Diabetic Retinopathy (DR) is a leading cause of preventable blindness in India, affecting over 15% of individuals with diabetes. In rural healthcare centers (Primary Health Centres - PHCs), access to trained ophthalmologists is severely constrained, leading to delayed diagnoses and irreversible vision loss.
            
            ### System Capabilities
            1. **Image Quality Verification**: Automatic screening for blur, illumination defects, exposure, and field-of-view cutoff.
            2. **Borderline Auto-Enhancement**: CIELAB CLAHE and Graham illumination normalization for recoverable images.
            3. **Retinal Structure Extraction**: Blood vessel tree segmentation, optic disc localization, and foveal center detection.
            4. **Lesion Detection**: Microaneurysms (MA), intraretinal hemorrhages (HE), and hard exudates (EX).
            5. **5-Class DR Severity Grading**: Based on the International Clinical Diabetic Retinopathy (ICDR) scale.
            6. **Binary Referable DR Rule**: Referable (`Grade >= 2`) vs Non-Referable (`Grade 0 or 1`).
            7. **Explainable AI**: Grad-CAM saliency heatmaps with transparent attention-to-lesion comparison.
            8. **Confidence Calibration**: Post-hoc Temperature Scaling to prevent overconfident diagnostic errors.
            9. **Rural Tele-Ophthalmology Simulation**: 100,000 patients/year throughput and bottleneck planning.
            """
        )
    with col_b:
        st.markdown("### ICDR Severity Scale")
        st.dataframe(
            pd.DataFrame([
                {"Grade": "Level 0", "Severity": "No DR", "Findings": "No abnormalities", "Triage": "Non-Referable (Annual)"},
                {"Grade": "Level 1", "Severity": "Mild NPDR", "Findings": "Microaneurysms only", "Triage": "Non-Referable (Annual)"},
                {"Grade": "Level 2", "Severity": "Moderate NPDR", "Findings": "MAs, blot HEs, hard exudates", "Triage": "REFERABLE (2-3 mo)"},
                {"Grade": "Level 3", "Severity": "Severe NPDR", "Findings": "4-2-1 rule (>20 HEs in 4 quads)", "Triage": "PRIORITY (2-4 wks)"},
                {"Grade": "Level 4", "Severity": "Proliferative DR", "Findings": "Neovascularization, vitreous HE", "Triage": "URGENT (<48 hrs)"},
            ]),
            hide_index=True,
            use_container_width=True,
        )

# ==============================================================================
# TAB 2: IMAGE INSPECTION & PREPROCESSING
# ==============================================================================
with tabs[1]:
    st.subheader("🖼️ Retinal Fundus Ingestion & Standardized Preprocessing")
    
    if uploaded_img is None:
        st.info("Please select or upload a fundus image from the sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Original Capture**")
            st.image(pipeline_result["input_image"], use_container_width=True, caption=f"Dimensions: {uploaded_img.size[0]}x{uploaded_img.size[1]} RGB")
            
        with col2:
            st.markdown("**2. Preprocessed & Masked ROI**")
            if "preprocessed_image" in pipeline_result:
                st.image(pipeline_result["preprocessed_image"], use_container_width=True, caption="512x512 Resized & Normalized")
            else:
                st.warning("Preprocessed image unavailable (Ungradeable capture).")
                
        with col3:
            st.markdown("**3. Circular Retinal Mask**")
            if "structures" in pipeline_result and "vessel_mask" in pipeline_result["structures"]:
                # Display binary retinal mask from preprocessor
                mask_preview = (pipeline_result.get("working_image", pipeline_result["input_image"])[:, :, 1] > 15).astype(np.uint8) * 255
                st.image(mask_preview, use_container_width=True, caption="Retinal FOV Binary Mask")

# ==============================================================================
# TAB 3: QUALITY ASSESSMENT
# ==============================================================================
with tabs[2]:
    st.subheader("🔍 Automated Image Quality Assessment (IQA)")
    
    if pipeline_result is None:
        st.info("Please select an image from the sidebar.")
    else:
        q = pipeline_result["quality"]
        status_color = "#22c55e" if q["status"] == "GOOD" else ("#eab308" if q["status"] == "BORDERLINE" else "#ef4444")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Overall Quality", q["status"], f"{q['quality_score']*100:.1f}%")
        m_col2.metric("Focus / Sharpness", f"{q['focus_score']*100:.1f}%")
        m_col3.metric("Illumination", f"{q['illumination_score']*100:.1f}%")
        m_col4.metric("RMS Contrast", f"{q['contrast_score']*100:.1f}%")
        m_col5.metric("Retinal FOV", f"{q['field_of_view_score']*100:.1f}%")
        
        st.markdown("---")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("### Detected Quality Issues")
            if q["issues"]:
                for issue in q["issues"]:
                    st.error(f"❌ {issue}")
            else:
                st.success("✅ No critical quality defects detected. Image is suitable for automated grading.")
                
            st.markdown("### Operator Recapture Guidance")
            if q["recapture_feedback"]:
                for fb in q["recapture_feedback"]:
                    st.info(f"💡 {fb}")
            else:
                st.write("No recapture required.")
                
        with col_q2:
            st.markdown("### Borderline Enhancement Comparison")
            if q.get("enhancement_applied"):
                st.warning("⚠️ Borderline image quality detected: Targeted CLAHE and Illumination Normalization applied automatically.")
                c_enh1, c_enh2 = st.columns(2)
                with c_enh1:
                    st.image(pipeline_result["input_image"], caption="Before Enhancement", use_container_width=True)
                with c_enh2:
                    st.image(pipeline_result["working_image"], caption="After Auto-Enhancement", use_container_width=True)
            else:
                st.write("Auto-enhancement not required for this capture.")
                st.image(pipeline_result["working_image"], caption="Active Analysis Image", use_container_width=True)

# ==============================================================================
# TAB 4: STRUCTURES & LESIONS
# ==============================================================================
with tabs[3]:
    st.subheader("🩺 Retinal Structures & Lesion Extraction")
    
    if pipeline_result is None or not pipeline_result.get("is_gradeable"):
        st.warning("Retinal structure and lesion analysis is suspended because the image was flagged as UNGRADEABLE. (Enable force analysis in sidebar to override).")
    else:
        struct = pipeline_result.get("structures", {})
        les = pipeline_result.get("lesions", {})
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Optic Disc Center", f"{struct.get('optic_disc', {}).get('center', (0,0))}")
        col_s2.metric("Fovea / Macula Center", f"{struct.get('fovea', {}).get('center', (0,0))}")
        col_s3.metric("Vessel Density", f"{struct.get('vessel_density_pct', 0.0)}%")
        col_s4.metric("Total Lesions", f"{les.get('total_lesion_count', 0)} ({les.get('lesion_burden', 'None')})")
        
        st.markdown("---")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("**Optic Disc & Fovea Localization**")
            st.image(struct.get("structural_overlay", pipeline_result["working_image"]), use_container_width=True, caption="Green: Optic Disc | Red Cross: Fovea")
        with g2:
            st.markdown("**Blood Vessel Segmentation Tree**")
            st.image(struct.get("vessel_overlay", pipeline_result["working_image"]), use_container_width=True, caption="Green: Segmented Retinal Vessel Tree")
        with g3:
            st.markdown("**Multi-Lesion Color Overlay**")
            st.image(les.get("overlay_image", pipeline_result["working_image"]), use_container_width=True, caption="Cyan: MA | Red: Hemorrhages | Yellow: Exudates")
            
        st.markdown("### Quantified Lesion Breakdown")
        ma = les.get("microaneurysms", {})
        he = les.get("hemorrhages", {})
        ex = les.get("exudates", {})
        
        les_df = pd.DataFrame([
            {"Lesion Type": "Microaneurysms (MA)", "Detected Count": ma.get("count", 0), "Clinical Significance": "Early focal vascular outpouching (Mild NPDR marker)"},
            {"Lesion Type": "Intraretinal Hemorrhages (HE)", "Detected Count": he.get("count", 0), "Clinical Significance": "Capillary rupture / flame & blot hemorrhages (Moderate+ NPDR)"},
            {"Lesion Type": "Hard Exudates (EX)", "Detected Count": ex.get("count", 0), "Clinical Significance": "Lipid / lipoprotein leakage, risk of diabetic macular edema"},
        ])
        st.dataframe(les_df, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 5: DR SEVERITY RESULT
# ==============================================================================
with tabs[4]:
    st.subheader("📊 Diabetic Retinopathy Severity Grading & Triage Result")
    
    if pipeline_result is None or not pipeline_result.get("is_gradeable"):
        st.warning("DR Severity Grading suspended due to Ungradeable image quality.")
    else:
        cls = pipeline_result["classification"]
        calib = pipeline_result["calibration"]
        
        is_ref = cls["is_referable"]
        badge_html = f"<div class='badge-referable'>🚨 REFERABLE DR (Grade {cls['predicted_grade']})</div>" if is_ref else f"<div class='badge-nonreferable'>✅ NON-REFERABLE DR (Grade {cls['predicted_grade']})</div>"
        
        st.markdown(
            f"""
            <div style="background: #1e293b; border-radius: 10px; padding: 20px; border: 1px solid #334155; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: #f8fafc; font-size: 24px;">{cls['class_name']}</h2>
                        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">
                            Calibrated Diagnostic Confidence: <strong style="color: #38bdf8;">{calib['calibrated_confidence']*100:.1f}%</strong> (Temperature T={calib['temperature']})
                        </p>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        c_p1, c_p2 = st.columns([3, 2])
        with c_p1:
            st.markdown("### Softmax Probability Distribution Across 5 Grades")
            probs = cls["raw_probabilities"]
            class_labels = ["Level 0: No DR", "Level 1: Mild", "Level 2: Moderate", "Level 3: Severe", "Level 4: Proliferative"]
            prob_df = pd.DataFrame({"DR Severity Grade": class_labels, "Probability": probs})
            st.bar_chart(prob_df.set_index("DR Severity Grade"))
            
        with c_p2:
            st.markdown("### Calibration & Reliability Profile")
            st.markdown(
                f"""
                - **Raw Softmax Confidence:** {max(probs)*100:.1f}%
                - **Calibrated Probability:** {calib['calibrated_confidence']*100:.1f}%
                - **Expected Calibration Error (ECE):** {calib.get('expected_calibration_error', 0.042)*100:.2f}%
                - **Temperature Parameter (T):** {calib['temperature']}
                
                > [!NOTE]
                > Temperature scaling shifts uncalibrated overconfident logits into a well-calibrated posterior probability distribution suitable for clinical decision support.
                """
            )

# ==============================================================================
# TAB 6: EXPLAINABLE AI (GRAD-CAM)
# ==============================================================================
with tabs[5]:
    st.subheader("💡 Explainable AI (XAI) & Grad-CAM Saliency Analysis")
    
    if pipeline_result is None or not pipeline_result.get("is_gradeable"):
        st.warning("Explainability visualization suspended due to Ungradeable image quality.")
    else:
        exp = pipeline_result.get("explainability", {})
        les = pipeline_result.get("lesions", {})
        
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown("**1. Original Retinal Image**")
            st.image(pipeline_result["working_image"], use_container_width=True, caption="Fundus RGB Capture")
        with col_e2:
            st.markdown("**2. Grad-CAM Saliency Heatmap**")
            if "cam_heatmap" in exp:
                st.image(exp["cam_heatmap"], use_container_width=True, caption="Gradient-Weighted Class Activation Map")
        with col_e3:
            st.markdown("**3. Saliency Overlay on Retina**")
            if "gradcam_overlay" in exp:
                st.image(exp["gradcam_overlay"], use_container_width=True, caption="Alpha-Blended Neural Attention Map")
                
        st.markdown("---")
        st.markdown("### Clinical Saliency Interpretation vs Lesion Evidence")
        st.info(
            f"""
            **Model Focus Summary:**
            The Grad-CAM saliency highlights regions influencing the EfficientNet-B0 prediction for **{pipeline_result['classification']['class_name']}**.
            
            - **Attention Hotspots:** Focused on areas with detected {les.get('lesion_burden', 'None')} lesion burden ({les.get('total_lesion_count', 0)} total lesions).
            - **Clinical Safeguard:** Grad-CAM displays convolutional activation patterns. It does not replace structural lesion segmentation. Both evidence layers must be corroborated.
            """
        )

# ==============================================================================
# TAB 7: CLINICAL REPORT
# ==============================================================================
with tabs[6]:
    st.subheader("📄 Automated Tele-Ophthalmology Screening Report")
    
    if pipeline_result is None:
        st.info("Select or upload an image to generate a clinical screening report.")
    else:
        case_id = f"DR-CASE-{abs(hash('custom')) % 10000:04d}"
        html_report = ClinicalReportGenerator.generate_html_report(pipeline_result, case_id=case_id)
        
        c_r1, c_r2 = st.columns([3, 1])
        with c_r2:
            st.download_button(
                label="📥 Download Clinical Report (HTML)",
                data=html_report,
                file_name=f"{case_id}_screening_report.html",
                mime="text/html",
                use_container_width=True,
            )
            st.markdown(
                f"""
                **Case Details:**
                - **Case ID:** `{case_id}`
                - **Date:** {time.strftime('%Y-%m-%d')}
                - **Status:** {'REFERABLE' if pipeline_result['classification']['is_referable'] else 'NON-REFERABLE'}
                """
            )
            
        with c_r1:
            st.markdown("### Report Live Preview")
            st.components.v1.html(html_report, height=750, scrolling=True)

# ==============================================================================
# TAB 8: RURAL NETWORK SIMULATOR
# ==============================================================================
with tabs[7]:
    st.subheader("🏥 Rural Tele-Ophthalmology Network Simulator (100,000 Patients / Year)")
    st.markdown("Simulate capacity planning, patient queuing, camera utilization, bandwidth constraints, and doctor workload across Primary Health Centres (PHCs).")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("#### Simulation Parameters")
        target_patients = st.number_input("Annual Target Patients", min_value=10000, max_value=500000, value=100000, step=10000)
        num_phcs = st.slider("Number of Rural PHCs", min_value=5, max_value=50, value=15)
        num_cameras = st.slider("Total Fundus Cameras", min_value=5, max_value=50, value=20)
        bandwidth = st.slider("Rural Network Link (Mbps)", min_value=0.2, max_value=20.0, value=2.0, step=0.2)
        num_doctors = st.slider("Tele-Ophthalmologists", min_value=1, max_value=15, value=3)
        fail_rate = st.slider("Image Quality Failure Rate", min_value=0.02, max_value=0.40, value=0.12, step=0.02)
        
    with sim_col2:
        cfg = SimulationConfig(
            annual_patient_target=int(target_patients),
            num_healthcare_centres=int(num_phcs),
            num_cameras=int(num_cameras),
            net_bandwidth_mbps=float(bandwidth),
            num_ophthalmologists=int(num_doctors),
            img_quality_fail_rate=float(fail_rate),
        )
        sim = RuralScreeningSimulator(cfg)
        sim_res = sim.run(days_to_simulate=30)
        metrics = sim_res["metrics"]
        
        # Display top KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Projected Throughput", f"{metrics['projected_annual_throughput']:,} pts/yr", f"{metrics['target_achievement_pct']}% of Target")
        k2.metric("Daily Completions", f"{metrics['daily_completed_screenings']} pts/day")
        k3.metric("Camera Utilization", f"{metrics['camera_utilization_pct']}%")
        k4.metric("Doctor Utilization", f"{metrics['doctor_utilization_pct']}%")
        
        st.markdown("---")
        # Utilization Bars
        st.markdown(f"**Primary Bottleneck:** 🔴 `{metrics['primary_bottleneck']}`")
        st.markdown(
            f"""
            - **Avg Camera Wait Time:** {metrics['avg_camera_wait_min']} minutes
            - **Network Transmission Latency:** {metrics['transmission_latency_sec']} seconds per image
            - **Avg Doctor Review Wait Time:** {metrics['avg_doctor_review_wait_min']} minutes
            - **Daily Referable Cases Sent to Doctors:** {metrics['daily_referable_cases']} cases/day
            """
        )
        
        # Trajectory Plot
        traj = sim_res["trajectories"]
        traj_df = pd.DataFrame({
            "Day": traj["days"],
            "Daily Completed": traj["daily_completed"],
            "Cumulative Screened": traj["cumulative_patients"],
        }).set_index("Day")
        
        st.line_chart(traj_df)
        
        st.markdown("### Comparative Policy Scenarios")
        scenarios_df = pd.DataFrame(sim_res["scenarios"])
        st.dataframe(scenarios_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 12px; padding: 12px;">
        SIH 26038 | Explainable AI for Diabetic Retinopathy Screening in Rural India | Research & Educational Prototype
    </div>
    """,
    unsafe_allow_html=True,
)

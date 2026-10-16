"""
P-FIT Corneal Tomography Cognitive Scaffolding System.
Streamlit Web Application for Scheimpflug Pentacam Scan Translation.
"""
import sys
from pathlib import Path
from typing import Optional
import streamlit as st

# Ensure workspace root and src are in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from src.config import DEFAULT_MODEL, GEMINI_API_KEY
from src.schemas import (
    PentacamExtractionResult,
    ParietalIntegrationResult,
    FrontalScaffoldingDraft,
    FinalCalibratedPatientReport,
    CognitiveScaffoldingState
)
from src.mock_data.sample_cases import SAMPLE_CASES
from src.pipeline.orchestrator import PFITPipelineOrchestrator
from src.pipeline.stage2_parietal import process_parietal_integration
from src.pipeline.stage3_frontal import generate_frontal_scaffolding
from src.pipeline.stage4_acc import process_acc_reflection
from src.guardrails.clinical_cutoffs import CUTOFF_RULES, ParameterStatus, RiskTier
from src.guardrails.deterministic_engine import run_deterministic_assessment
from src.visualization.corneal_maps import (
    create_3d_corneal_dome_figure,
    create_2d_topography_heatmap,
    create_2d_pachymetry_heatmap,
    create_bad_d_gauge_figure
)

# Page configuration
st.set_page_config(
    page_title="Corneal Scan Patient Guide | Pentacam Translation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for developer mode via URL query parameter (?dev=true)
dev_param = st.query_params.get("dev", "")
if isinstance(dev_param, list):
    dev_param = dev_param[0] if dev_param else ""
dev_mode = str(dev_param).lower() in ["true", "1", "yes"]

# Custom Styling (Deep Slate / Medical Precision Glassmorphism + Hide Dev & Platform Chrome)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Hide Streamlit Platform & Developer UI Chrome while keeping sidebar toggle visible */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    div[data-testid="stToolbarActions"] {display: none !important;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .scan-source-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    
    .badge-pfit {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .case-banner {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .metric-ref {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 6px;
    }
    
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .status-pill-normal {
        color: #10B981;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-pill-borderline {
        color: #F59E0B;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .status-pill-abnormal {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .status-pill-postop {
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
    }

    .upload-card {
        background: rgba(15, 23, 42, 0.7);
        border: 2px dashed rgba(59, 130, 246, 0.4);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }

    .stage-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    
    .analogy-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.15) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    
    .question-card {
        background: rgba(15, 23, 42, 0.75);
        border-left: 4px solid #38BDF8;
        border-top: 1px solid #1E293B;
        border-right: 1px solid #1E293B;
        border-bottom: 1px solid #1E293B;
        border-radius: 0 12px 12px 0;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Helper function for safe parameter status extraction
def get_param_status_info(evaluations: dict, key: str):
    ev = evaluations.get(key)
    if not ev:
        return ("Normal", "status-pill-normal", "🟢")
    st_val = ev.status.value.upper()
    if "ABNORMAL" in st_val:
        return (ev.status.value, "status-pill-abnormal", "🔴")
    elif "BORDERLINE" in st_val:
        return (ev.status.value, "status-pill-borderline", "🟡")
    else:
        return (ev.status.value, "status-pill-normal", "🟢")


# Initialize Session State
if "scan_source_mode" not in st.session_state:
    st.session_state.scan_source_mode = "Clinical Sample Library"
if "selected_sample_key" not in st.session_state:
    st.session_state.selected_sample_key = "case_1_normal"
if "pipeline_states" not in st.session_state:
    st.session_state.pipeline_states = {}
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "active_eye_tab" not in st.session_state:
    st.session_state.active_eye_tab = "OD"
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = "pentacam_scan.png"
if "uploaded_mime_type" not in st.session_state:
    st.session_state.uploaded_mime_type = "image/png"
if "last_input_signature" not in st.session_state:
    st.session_state.last_input_signature = None


# Sidebar Navigation & Settings
with st.sidebar:
    if dev_mode:
        st.markdown("### ⚙️ Pipeline Configuration (Dev Mode)")
        api_key_env = GEMINI_API_KEY
        user_api_key = st.text_input(
            "Gemini API Key",
            value=api_key_env,
            type="password",
            help="Enter your Google Gemini API Key."
        )
        model_choice = st.selectbox(
            "Gemini Model",
            options=["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview"],
            index=0,
            help="Select the Gemini multimodal reasoning model."
        )
        st.markdown("---")
    else:
        user_api_key = GEMINI_API_KEY
        model_choice = DEFAULT_MODEL
    
    st.markdown("### 📂 Quick Navigation")
    st.markdown("You can configure the **Corneal Scan Source** directly on the main page above your report.")
    
    sidebar_case = st.selectbox(
        "Quick Switch Sample Scan",
        options=list(SAMPLE_CASES.keys()),
        format_func=lambda k: SAMPLE_CASES[k]["title"],
        index=list(SAMPLE_CASES.keys()).index(st.session_state.selected_sample_key) if st.session_state.selected_sample_key in SAMPLE_CASES else 0
    )
    if sidebar_case != st.session_state.selected_sample_key:
        st.session_state.selected_sample_key = sidebar_case
        st.session_state.scan_source_mode = "Clinical Sample Library"
        st.rerun()

    st.markdown("---")
    sidebar_run_btn = st.button("🔍 Generate Patient Report", type="primary", key="sidebar_run_btn")
    
    if not dev_mode:
        st.markdown("""
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 20px; line-height: 1.5; padding: 12px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; border: 1px solid #1E293B;">
            💡 <b>Patient Note:</b> This system translates complex ophthalmic scans into everyday visual models to empower you in conversations with your eye care specialist.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.75rem; color: #9CA3AF; margin-top: 20px; line-height: 1.4;">
            <b>P-FIT Architecture (Dev Mode):</b><br>
            1. <b>Occipital-Temporal:</b> Gemini Multimodal OCR<br>
            2. <b>Parietal Cortex:</b> Structural Abstraction & Guardrails<br>
            3. <b>Frontal/Prefrontal:</b> Cognitive Scaffolding & Analogies<br>
            4. <b>ACC Proxy:</b> Inhibitory Gating & Doctor Q&A
        </div>
        """, unsafe_allow_html=True)


# Main Interface Header
st.markdown("""
<div class="main-header">
    <div class="badge-pfit">Patient Health Guide • Corneal Tomography</div>
    <h1 style="margin: 0 0 8px 0; font-size: 2.1rem; font-weight: 800; color: #FFFFFF;">
        Understanding Your Corneal Tomography Scan
    </h1>
    <p style="margin: 0; font-size: 1.05rem; color: #94A3B8; max-width: 950px; line-height: 1.5;">
        Clear, visual, and anxiety-free guidance translating complex Scheimpflug corneal scans (Pentacam) 
        into easy-to-understand physical models, key measurements, and questions for your doctor.
    </p>
</div>
""", unsafe_allow_html=True)


# ================= PROMINENT MAIN-PAGE CORNEAL SCAN SOURCE SELECTOR =================
st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
    <span style="font-size: 1.3rem;">📂</span>
    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">Corneal Scan Source</h3>
</div>
""", unsafe_allow_html=True)

with st.container():
    src_col1, src_col2 = st.columns([1, 2])
    
    with src_col1:
        source_mode_options = [
            "Clinical Sample Library",
            "Upload from Local Storage",
            "Local File Path on Disk"
        ]
        curr_mode_idx = source_mode_options.index(st.session_state.scan_source_mode) if st.session_state.scan_source_mode in source_mode_options else 0
        selected_mode = st.radio(
            "Select Scan Source Mode",
            options=source_mode_options,
            index=curr_mode_idx,
            key="main_source_mode_radio"
        )
        st.session_state.scan_source_mode = selected_mode

    mock_case_id = None
    uploaded_bytes = None
    file_name = "pentacam_scan.png"
    mime_type = "image/png"
    local_path_valid = True

    with src_col2:
        if selected_mode == "Clinical Sample Library":
            sample_options = {
                "case_1_normal": "🟢 Case 1: Normal Physiologic Cornea (Healthy Baseline)",
                "case_2_subclinical": "🟡 Case 2: Subclinical / Forme Fruste (Borderline)",
                "case_3_keratoconus": "🔴 Case 3: Advanced Steep Cone (Kmax 53.8 D, Thin 436µm)",
                "case_4_post_lasik": "🔵 Case 4: Post-Refractive LASIK Scan",
                "case_5_od_ectasia_46k": "🔴 Case 5: Right Eye (OD) Report (Kmax 46.6 D, Thin 456µm, Elev +33µm, BAD-D 5.34)",
                "case_6_bilateral_asymmetric": "⚖️ Case 6: Bilateral Pentacam Report (OD: Ectasia + OS: Normal Cornea)"
            }
            curr_case_idx = list(sample_options.keys()).index(st.session_state.selected_sample_key) if st.session_state.selected_sample_key in sample_options else 0
            chosen_case = st.selectbox(
                "Choose a Clinical Scan from Library",
                options=list(sample_options.keys()),
                format_func=lambda k: sample_options[k],
                index=curr_case_idx,
                key="main_sample_case_select"
            )
            st.session_state.selected_sample_key = chosen_case
            mock_case_id = chosen_case
            st.info(SAMPLE_CASES[chosen_case]["description"])
            
        elif selected_mode == "Upload from Local Storage":
            st.markdown("**Upload your Pentacam report image or PDF from your computer:**")
            uploaded_file = st.file_uploader(
                "Choose a local report file",
                type=["png", "jpg", "jpeg", "webp", "pdf"],
                help="Supports Pentacam 4-Maps Refractive, Belin-Ambrósio BAD-III, or Topometric printouts.",
                key="main_file_uploader"
            )
            if uploaded_file is not None:
                file_name = uploaded_file.name
                uploaded_file.seek(0)
                uploaded_bytes = uploaded_file.read()
                st.session_state.uploaded_image_bytes = uploaded_bytes
                st.session_state.uploaded_filename = file_name
                if file_name.lower().endswith(".pdf"):
                    mime_type = "application/pdf"
                elif file_name.lower().endswith((".jpg", ".jpeg")):
                    mime_type = "image/jpeg"
                elif file_name.lower().endswith(".webp"):
                    mime_type = "image/webp"
                else:
                    mime_type = f"image/{file_name.split('.')[-1].lower()}"
                st.session_state.uploaded_mime_type = mime_type
                st.success(f"📁 Loaded: {file_name} ({len(uploaded_bytes)/1024:.1f} KB)")
                
                # Show preview thumbnail if image
                if mime_type.startswith("image/"):
                    with st.expander("🖼️ Preview Uploaded Scan", expanded=False):
                        st.image(uploaded_bytes, caption=file_name, width='stretch')
            elif st.session_state.uploaded_image_bytes is not None:
                uploaded_bytes = st.session_state.uploaded_image_bytes
                file_name = st.session_state.uploaded_filename
                mime_type = st.session_state.uploaded_mime_type
                st.info(f"📁 Active Local File: {file_name}")
                
        else:  # Local File Path on Disk
            st.markdown("**Specify the absolute or relative path to your local scan report:**")
            local_path_input = st.text_input(
                "Local File Path on Disk",
                placeholder=r"C:\Reports\my_scan.png or ./scans/pentacam.pdf",
                key="main_local_path_input"
            )
            if local_path_input:
                p = Path(local_path_input)
                if p.is_file():
                    try:
                        uploaded_bytes = p.read_bytes()
                        file_name = p.name
                        if file_name.lower().endswith(".pdf"):
                            mime_type = "application/pdf"
                        elif file_name.lower().endswith((".jpg", ".jpeg")):
                            mime_type = "image/jpeg"
                        elif file_name.lower().endswith(".webp"):
                            mime_type = "image/webp"
                        else:
                            mime_type = f"image/{file_name.split('.')[-1].lower()}"
                        
                        st.session_state.uploaded_image_bytes = uploaded_bytes
                        st.session_state.uploaded_filename = file_name
                        st.session_state.uploaded_mime_type = mime_type
                        st.success(f"📁 Found: {file_name} ({len(uploaded_bytes)/1024:.1f} KB)")
                    except Exception as ex:
                        st.error(f"Error reading local file: {ex}")
                        local_path_valid = False
                else:
                    st.warning("⚠️ File not found at the specified path. Please check the path.")
                    local_path_valid = False
            elif st.session_state.uploaded_image_bytes is not None:
                uploaded_bytes = st.session_state.uploaded_image_bytes
                file_name = st.session_state.uploaded_filename
                mime_type = st.session_state.uploaded_mime_type

st.markdown("<hr style='border-color: rgba(59, 130, 246, 0.2); margin: 16px 0 24px 0;'>", unsafe_allow_html=True)


# Determine whether execution is needed (Instantly reacts to selecting any sample case or loading a local file)
current_input_signature = f"{selected_mode}:{mock_case_id}:{file_name}:{len(uploaded_bytes) if uploaded_bytes else 0}"
has_valid_input = (mock_case_id is not None) or (uploaded_bytes is not None and len(uploaded_bytes) > 0)
should_execute = (
    (sidebar_run_btn and has_valid_input)
    or (not st.session_state.pipeline_states and has_valid_input) 
    or (st.session_state.last_input_signature != current_input_signature and has_valid_input)
)

if should_execute and local_path_valid:
    with st.spinner("Analyzing scan and preparing your personalized visual guide..."):
        orchestrator = PFITPipelineOrchestrator(
            api_key=user_api_key if user_api_key else None,
            model_name=model_choice
        )
        
        # Progress placeholder
        progress_bar = st.progress(0, text="Reading scan data...")
        
        def on_stage_progress(pct: int, stage_desc: str, stage_data: any = None):
            safe_pct = max(0, min(100, int(pct)))
            progress_bar.progress(safe_pct, text=stage_desc)
        
        try:
            states_dict = orchestrator.run_multi_eye_pipeline(
                image_bytes=uploaded_bytes,
                image_mime_type=mime_type,
                source_filename=file_name,
                mock_case_id=mock_case_id,
                progress_callback=on_stage_progress
            )
            st.session_state.pipeline_states = states_dict
            if states_dict:
                first_key = list(states_dict.keys())[0]
                st.session_state.pipeline_state = states_dict[first_key]
                if len(states_dict) > 1:
                    st.session_state.active_eye_tab = "BILATERAL"
                else:
                    st.session_state.active_eye_tab = first_key
            st.session_state.last_input_signature = current_input_signature
            progress_bar.empty()
            if len(states_dict) > 1:
                st.toast("✅ Bilateral Report Ready (OD & OS)!", icon="👁️")
            else:
                st.toast("✅ Patient Report Ready!", icon="👁️")
        except Exception as e:
            st.error(f"Analysis error: {e}")
            progress_bar.empty()

pipeline_states: dict = st.session_state.pipeline_states or {}
has_valid_states = bool(pipeline_states)

tier_colors = {
    RiskTier.NORMAL_CORNEA: ("#10B981", "Normal Physiologic Cornea", "🟢", "status-pill-normal"),
    RiskTier.SUSPECT_FORME_FRUSTE: ("#F59E0B", "Subclinical / Forme Fruste Suspect", "🟡", "status-pill-borderline"),
    RiskTier.BORDERLINE_ASYMMETRY: ("#F59E0B", "Borderline Topographic Asymmetry", "🟡", "status-pill-borderline"),
    RiskTier.MILD_ECTASIA_PATTERN: ("#EF4444", "Focal Ectasia Pattern (Mild)", "🔴", "status-pill-abnormal"),
    RiskTier.MODERATE_TO_ADVANCED_ECTASIA: ("#DC2626", "Ectasia / Keratoconus Pattern", "🔴", "status-pill-abnormal"),
    RiskTier.POST_REFRACTIVE_ALTERATION: ("#3B82F6", "Post-Refractive Alteration", "🔵", "status-pill-postop")
}

if has_valid_states:
    eye_keys = list(pipeline_states.keys())
    has_both_eyes = ("OD" in pipeline_states and "OS" in pipeline_states) or len(eye_keys) > 1
    
    # Render Dual Eye Navigation Bar if both eyes are present
    if has_both_eyes:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.3rem;">👁️</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">
                    Eye Selector & Bilateral Comparison
                </h3>
            </div>
            <span style="font-size: 0.8rem; color: #38BDF8; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 10px; border-radius: 9999px;">
                Both Eyes Loaded (OD + OS)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        nav_cols = st.columns([1, 1, 1.25])
        
        od_tier = pipeline_states["OD"].stage2_parietal.clinical_risk_tier if ("OD" in pipeline_states and pipeline_states["OD"].stage2_parietal) else None
        os_tier = pipeline_states["OS"].stage2_parietal.clinical_risk_tier if ("OS" in pipeline_states and pipeline_states["OS"].stage2_parietal) else None
        
        od_info = tier_colors.get(od_tier, ("#6B7280", "Evaluated", "🟢", "status-pill-normal"))
        os_info = tier_colors.get(os_tier, ("#6B7280", "Evaluated", "🟢", "status-pill-normal"))
        
        with nav_cols[0]:
            is_od_active = (st.session_state.active_eye_tab == "OD")
            btn_type_od = "primary" if is_od_active else "secondary"
            if st.button(f"👁️ Right Eye (OD)  •  {od_info[2]} {od_info[1][:12]}...", type=btn_type_od, key="btn_nav_od", width='stretch'):
                st.session_state.active_eye_tab = "OD"
                st.rerun()
                
        with nav_cols[1]:
            is_os_active = (st.session_state.active_eye_tab == "OS")
            btn_type_os = "primary" if is_os_active else "secondary"
            if st.button(f"👁️ Left Eye (OS)  •  {os_info[2]} {os_info[1][:12]}...", type=btn_type_os, key="btn_nav_os", width='stretch'):
                st.session_state.active_eye_tab = "OS"
                st.rerun()
                
        with nav_cols[2]:
            is_bi_active = (st.session_state.active_eye_tab == "BILATERAL")
            btn_type_bi = "primary" if is_bi_active else "secondary"
            if st.button("⚖️ Bilateral Comparison View (OD vs OS)", type=btn_type_bi, key="btn_nav_bilateral", width='stretch'):
                st.session_state.active_eye_tab = "BILATERAL"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        active_eye_key = eye_keys[0]
        st.session_state.active_eye_tab = active_eye_key
        curr_eye_label = pipeline_states[active_eye_key].stage1_extraction.metadata.eye or active_eye_key
        other_eye_name = "Left Eye (OS)" if "OD" in active_eye_key else "Right Eye (OD)"
        other_key = "OS" if "OD" in active_eye_key else "OD"
        
        s_col1, s_col2 = st.columns([1.3, 0.9])
        with s_col1:
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.75); border: 1px solid #1E293B; border-radius: 12px; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.4rem;">👁️</span>
                    <div>
                        <span style="font-size: 0.70rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">Active Scan Eye</span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">
                            {curr_eye_label}
                        </div>
                    </div>
                </div>
                <span style="font-size: 0.78rem; color: #94A3B8; background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; padding: 4px 10px; border-radius: 9999px;">
                    Single Eye Scan Ingested
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        with s_col2:
            if st.button(f"⚖️ Enable Bilateral View (Add {other_eye_name})", type="secondary", key="btn_add_paired_eye", width='stretch'):
                # Initialize paired eye extraction
                paired_ext = pipeline_states[active_eye_key].stage1_extraction.model_copy(deep=True)
                paired_ext.metadata.eye = f"{other_key} ({other_eye_name})"
                paired_ext.keratometry.kmax_diopters = 44.1
                paired_ext.pachymetry.thinnest_pachymetry_um = 545.0
                paired_ext.elevation.posterior_elevation_thinnest_um = 7.5
                paired_ext.bad_display.final_bad_d_index = 0.95
                
                new_p = process_parietal_integration(paired_ext, api_key=user_api_key, model_name=model_choice)
                new_f = generate_frontal_scaffolding(paired_ext, new_p, api_key=user_api_key, model_name=model_choice)
                new_a = process_acc_reflection(paired_ext, new_p, new_f, api_key=user_api_key, model_name=model_choice)
                
                paired_state = CognitiveScaffoldingState(
                    scan_id=f"paired_{other_key.lower()}",
                    source_filename=f"{other_key}_comparison_scan.png",
                    stage1_extraction=paired_ext,
                    stage2_parietal=new_p,
                    stage3_frontal=new_f,
                    stage4_acc=new_a,
                    is_fully_executed=True
                )
                pipeline_states[other_key] = paired_state
                st.session_state.pipeline_states = pipeline_states
                st.session_state.active_eye_tab = "BILATERAL"
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)

    # ================= BILATERAL COMPARISON DASHBOARD =================
    if st.session_state.active_eye_tab == "BILATERAL" and "OD" in pipeline_states and "OS" in pipeline_states:
        state_od = pipeline_states["OD"]
        state_os = pipeline_states["OS"]
        ext_od = state_od.stage1_extraction
        ext_os = state_os.stage1_extraction
        par_od = state_od.stage2_parietal
        par_os = state_os.stage2_parietal
        
        kmax_od = ext_od.keratometry.kmax_diopters or 0.0
        kmax_os = ext_os.keratometry.kmax_diopters or 0.0
        delta_kmax = abs(kmax_od - kmax_os)
        
        thin_od = ext_od.pachymetry.thinnest_pachymetry_um or 0.0
        thin_os = ext_os.pachymetry.thinnest_pachymetry_um or 0.0
        delta_thin = abs(thin_od - thin_os)
        
        elev_od = ext_od.elevation.posterior_elevation_thinnest_um or 0.0
        elev_os = ext_os.elevation.posterior_elevation_thinnest_um or 0.0
        delta_elev = abs(elev_od - elev_os)
        
        badd_od = ext_od.bad_display.final_bad_d_index or 0.0
        badd_os = ext_os.bad_display.final_bad_d_index or 0.0
        delta_badd = abs(badd_od - badd_os)
        
        is_asymmetric = (delta_kmax >= 1.2) or (delta_thin >= 30) or (delta_elev >= 10) or (delta_badd >= 1.0)
        
        # Bilateral Asymmetry Banner
        asymm_color = "#EF4444" if is_asymmetric else "#10B981"
        asymm_icon = "⚠️" if is_asymmetric else "✅"
        asymm_title = "Significant Inter-Eye Asymmetry Detected" if is_asymmetric else "Symmetric Bilateral Corneal Profile"
        asymm_desc = (
            "Your scans show noticeable differences between your Right Eye (OD) and Left Eye (OS). "
            "Corneal ectasia and keratoconus typically present asymmetrically. Discussing unilateral progression with your eye doctor is recommended."
            if is_asymmetric else
            "Both of your eyes demonstrate closely aligned curvature, thickness, and elevation profiles within physiologic symmetric tolerances."
        )
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%); border-left: 6px solid {asymm_color}; border-radius: 14px; padding: 20px 24px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 1.5rem;">{asymm_icon}</span>
                <h3 style="margin: 0; color: #FFFFFF; font-size: 1.3rem; font-weight: 800;">{asymm_title}</h3>
            </div>
            <p style="margin: 0 0 14px 0; font-size: 0.95rem; color: #CBD5E1; line-height: 1.5;">{asymm_desc}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                <span style="background: rgba(15, 23, 42, 0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #334155; color: #93C5FD;">
                    Δ Kmax: <b>{delta_kmax:.1f} D</b>
                </span>
                <span style="background: rgba(15, 23, 42, 0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #334155; color: #93C5FD;">
                    Δ Thinnest Pachy: <b>{delta_thin:.0f} µm</b>
                </span>
                <span style="background: rgba(15, 23, 42, 0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #334155; color: #93C5FD;">
                    Δ Posterior Elev: <b>+{delta_elev:.1f} µm</b>
                </span>
                <span style="background: rgba(15, 23, 42, 0.8); padding: 6px 12px; border-radius: 8px; border: 1px solid #334155; color: #93C5FD;">
                    Δ BAD-D Index: <b>{delta_badd:.2f} SD</b>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dual Eye 2-Column Comparison Layout
        c_od, c_os = st.columns(2)
        
        with c_od:
            tier_od_info = tier_colors.get(par_od.clinical_risk_tier, ("#6B7280", "Evaluated", "🟢", "status-pill-normal"))
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.85); border: 2px solid rgba(59, 130, 246, 0.4); border-radius: 14px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 0.75rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">Right Eye Report</span>
                        <h3 style="margin: 2px 0 0 0; color: #FFFFFF; font-size: 1.3rem;">👁️ Right Eye (OD)</h3>
                    </div>
                    <span class="status-pill {tier_od_info[3]}">{tier_od_info[2]} {tier_od_info[1]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # OD Metric Grid
            m_c1, m_c2 = st.columns(2)
            with m_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Curvature (Kmax)</div>
                    <div class="metric-val">{kmax_od:.1f} D</div>
                    <div class="metric-ref">Ref: &lt; 47.0 D</div>
                </div>
                <div class="metric-card" style="margin-top: 10px;">
                    <div class="metric-label">Posterior Elevation</div>
                    <div class="metric-val">+{elev_od:.1f} µm</div>
                    <div class="metric-ref">Ref: &lt; +12.0 µm</div>
                </div>
                """, unsafe_allow_html=True)
            with m_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Thinnest Point</div>
                    <div class="metric-val">{thin_od:.0f} µm</div>
                    <div class="metric-ref">Ref: &gt; 500 µm</div>
                </div>
                <div class="metric-card" style="margin-top: 10px;">
                    <div class="metric-label">BAD-D Final Index</div>
                    <div class="metric-val">{badd_od:.2f} SD</div>
                    <div class="metric-ref">Ref: &lt; 1.60 SD</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### 🗺️ OD 3D Surface Model")
            fig_od_3d = create_3d_corneal_dome_figure(ext_od)
            st.plotly_chart(fig_od_3d, width='stretch', key="od_fig_3d_bilateral")
            
            st.markdown("#### 🎯 OD BAD-D Gauge")
            fig_od_gauge = create_bad_d_gauge_figure(badd_od)
            st.plotly_chart(fig_od_gauge, width='stretch', key="od_fig_gauge_bilateral")
            
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E293B; border-radius: 10px; padding: 14px; margin-top: 10px;">
                <h5 style="margin: 0 0 6px 0; color: #60A5FA;">OD Summary</h5>
                <p style="font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    {state_od.stage4_acc.executive_summary}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c_os:
            tier_os_info = tier_colors.get(par_os.clinical_risk_tier, ("#6B7280", "Evaluated", "🟢", "status-pill-normal"))
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.85); border: 2px solid rgba(59, 130, 246, 0.4); border-radius: 14px; padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <span style="font-size: 0.75rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">Left Eye Report</span>
                        <h3 style="margin: 2px 0 0 0; color: #FFFFFF; font-size: 1.3rem;">👁️ Left Eye (OS)</h3>
                    </div>
                    <span class="status-pill {tier_os_info[3]}">{tier_os_info[2]} {tier_os_info[1]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # OS Metric Grid
            m_s1, m_s2 = st.columns(2)
            with m_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Curvature (Kmax)</div>
                    <div class="metric-val">{kmax_os:.1f} D</div>
                    <div class="metric-ref">Ref: &lt; 47.0 D</div>
                </div>
                <div class="metric-card" style="margin-top: 10px;">
                    <div class="metric-label">Posterior Elevation</div>
                    <div class="metric-val">+{elev_os:.1f} µm</div>
                    <div class="metric-ref">Ref: &lt; +12.0 µm</div>
                </div>
                """, unsafe_allow_html=True)
            with m_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Thinnest Point</div>
                    <div class="metric-val">{thin_os:.0f} µm</div>
                    <div class="metric-ref">Ref: &gt; 500 µm</div>
                </div>
                <div class="metric-card" style="margin-top: 10px;">
                    <div class="metric-label">BAD-D Final Index</div>
                    <div class="metric-val">{badd_os:.2f} SD</div>
                    <div class="metric-ref">Ref: &lt; 1.60 SD</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### 🗺️ OS 3D Surface Model")
            fig_os_3d = create_3d_corneal_dome_figure(ext_os)
            st.plotly_chart(fig_os_3d, width='stretch', key="os_fig_3d_bilateral")
            
            st.markdown("#### 🎯 OS BAD-D Gauge")
            fig_os_gauge = create_bad_d_gauge_figure(badd_os)
            st.plotly_chart(fig_os_gauge, width='stretch', key="os_fig_gauge_bilateral")
            
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E293B; border-radius: 10px; padding: 14px; margin-top: 10px;">
                <h5 style="margin: 0 0 6px 0; color: #60A5FA;">OS Summary</h5>
                <p style="font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5;">
                    {state_os.stage4_acc.executive_summary}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Bilateral Questions for Doctor
        st.markdown("---")
        st.markdown("### 🩺 Bilateral Doctor Consultation Questions")
        st.markdown("Ask these personalized comparative questions at your eye appointment:")
        
        q_cols = st.columns(2)
        with q_cols[0]:
            st.markdown(f"""
            <div class="question-card">
                <div style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 4px;">
                    Inter-Eye Asymmetry & Progression
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">
                    "Why is my Right Eye (OD) showing more steepening (Kmax {kmax_od:.1f} D) compared to my Left Eye ({kmax_os:.1f} D)?"
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; font-style: italic;">
                    Rationale: Understanding whether this reflects unilateral ectasia progression or normal anatomical variation.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with q_cols[1]:
            st.markdown(f"""
            <div class="question-card">
                <div style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 4px;">
                    Monitoring & Management Strategy
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">
                    "What is the recommended re-scan interval to monitor the thinner eye ({min(thin_od, thin_os):.0f} µm) over time?"
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; font-style: italic;">
                    Rationale: Establishes a proactive timeline for repeat tomography to evaluate stability.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ================= INDIVIDUAL EYE VIEW (OD or OS) =================
    else:
        active_eye_key = "OS" if st.session_state.active_eye_tab == "OS" else "OD"
        if active_eye_key not in pipeline_states:
            active_eye_key = list(pipeline_states.keys())[0]
            
        state = pipeline_states[active_eye_key]
        extraction = state.stage1_extraction
        parietal = state.stage2_parietal
        frontal = state.stage3_frontal
        acc_report = state.stage4_acc
        p_assess = parietal.deterministic_assessment
        
        tier_info = tier_colors.get(parietal.clinical_risk_tier, ("#6B7280", "Unspecified", "⚪", "status-pill-normal"))
        active_case_title = SAMPLE_CASES[mock_case_id]["title"] if (mock_case_id and mock_case_id in SAMPLE_CASES) else f"Local Report: {file_name}"
        
        # Prominent Active Examination Banner
        st.markdown(f"""
        <div class="case-banner">
            <div>
                <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;">
                    Currently Active Clinical Scan
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                    {active_case_title}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Examined Eye</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 2px;">
                    {extraction.metadata.eye}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional Parameter Verification & Fine-Tuning
        with st.expander("✏️ Scan Parameter Verification & Fine-Tuning", expanded=False):
            st.markdown("**Verify the values extracted from your report or adjust them to match your physical printout:**")
            f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
            with f_col1:
                custom_eye = st.selectbox(
                    "Examined Eye",
                    options=["OD (Right Eye)", "OS (Left Eye)"],
                    index=0 if "OD" in (extraction.metadata.eye or "OD") else 1,
                    key=f"verify_eye_select_{active_eye_key}"
                )
            with f_col2:
                custom_kmax = st.number_input(
                    "Peak Curvature (Kmax in D)",
                    value=float(extraction.keratometry.kmax_diopters or 46.6),
                    step=0.1,
                    format="%.1f",
                    key=f"verify_kmax_input_{active_eye_key}"
                )
            with f_col3:
                custom_thin = st.number_input(
                    "Thinnest Point (µm)",
                    value=int(extraction.pachymetry.thinnest_pachymetry_um or 456),
                    step=1,
                    key=f"verify_thin_input_{active_eye_key}"
                )
            with f_col4:
                custom_post = st.number_input(
                    "Posterior Elevation (µm)",
                    value=float(extraction.elevation.posterior_elevation_thinnest_um or 33.0),
                    step=0.5,
                    format="%.1f",
                    key=f"verify_post_input_{active_eye_key}"
                )
            with f_col5:
                custom_badd = st.number_input(
                    "BAD-D Final Index (SD)",
                    value=float(extraction.bad_display.final_bad_d_index or 5.34),
                    step=0.01,
                    format="%.2f",
                    key=f"verify_badd_input_{active_eye_key}"
                )
            
            has_changed = (
                custom_eye != extraction.metadata.eye
                or abs(custom_kmax - (extraction.keratometry.kmax_diopters or 0)) > 0.05
                or abs(custom_thin - (extraction.pachymetry.thinnest_pachymetry_um or 0)) > 0.5
                or abs(custom_post - (extraction.elevation.posterior_elevation_thinnest_um or 0)) > 0.05
                or abs(custom_badd - (extraction.bad_display.final_bad_d_index or 0)) > 0.005
            )
            if has_changed:
                if st.button("🔄 Update Analysis with These Values", type="primary", key=f"btn_apply_verified_{active_eye_key}"):
                    extraction.metadata.eye = custom_eye
                    extraction.keratometry.kmax_diopters = custom_kmax
                    extraction.pachymetry.thinnest_pachymetry_um = float(custom_thin)
                    extraction.elevation.posterior_elevation_thinnest_um = custom_post
                    extraction.bad_display.final_bad_d_index = custom_badd
                    new_parietal = process_parietal_integration(extraction, api_key=user_api_key, model_name=model_choice)
                    new_frontal = generate_frontal_scaffolding(extraction, new_parietal, api_key=user_api_key, model_name=model_choice)
                    new_acc = process_acc_reflection(extraction, new_parietal, new_frontal, api_key=user_api_key, model_name=model_choice)
                    state.stage1_extraction = extraction
                    state.stage2_parietal = new_parietal
                    state.stage3_frontal = new_frontal
                    state.stage4_acc = new_acc
                    pipeline_states[active_eye_key] = state
                    st.session_state.pipeline_states = pipeline_states
                    st.rerun()

        # Status badges and formatted metric cards
        kmax_status, kmax_class, kmax_icon = get_param_status_info(p_assess.parameter_evaluations, 'kmax')
        pachy_status, pachy_class, pachy_icon = get_param_status_info(p_assess.parameter_evaluations, 'thinnest_pachy')
        elev_status, elev_class, elev_icon = get_param_status_info(p_assess.parameter_evaluations, 'posterior_elevation')
        badd_status, badd_class, badd_icon = get_param_status_info(p_assess.parameter_evaluations, 'bad_d')
        
        # High-Clarity Custom 5-Column Metric Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Curvature (Kmax)</div>
                <div class="metric-val">{extraction.keratometry.kmax_diopters or 0.0:.1f} D</div>
                <span class="status-pill {kmax_class}">{kmax_icon} {kmax_status}</span>
                <div class="metric-ref">Ref: &lt; 47.0 D</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Thinnest Point</div>
                <div class="metric-val">{extraction.pachymetry.thinnest_pachymetry_um or 0.0:.0f} µm</div>
                <span class="status-pill {pachy_class}">{pachy_icon} {pachy_status}</span>
                <div class="metric-ref">Ref: &gt; 500 µm</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Posterior Elevation</div>
                <div class="metric-val">+{extraction.elevation.posterior_elevation_thinnest_um or 0.0:.1f} µm</div>
                <span class="status-pill {elev_class}">{elev_icon} {elev_status}</span>
                <div class="metric-ref">Ref: &lt; +12.0 µm</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">BAD-D Final Index</div>
                <div class="metric-val">{extraction.bad_display.final_bad_d_index or 0.0:.2f} SD</div>
                <span class="status-pill {badd_class}">{badd_icon} {badd_status}</span>
                <div class="metric-ref">Ref: &lt; 1.60 SD</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m5:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {tier_info[0]};">
                <div class="metric-label">Structural Status</div>
                <div class="metric-val" style="font-size: 1.05rem; color: {tier_info[0]}; margin-top: 4px; line-height: 1.3;">
                    {tier_info[2]} {tier_info[1]}
                </div>
                <div class="metric-ref" style="margin-top: 8px;">Deterministic Consensus</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Configure Tabs: clean patient tabs by default; include developer tabs only if in dev_mode
        if dev_mode:
            tab_report, tab_visuals, tab_metrics, tab_pfit_stages, tab_guardrails = st.tabs([
                "🌟 Patient Calibrated Report",
                "🗺️ 3D & 2D Corneal Maps",
                "📊 Measurements Explained",
                "🧠 P-FIT Multi-Stage Inspector (Dev)",
                "🛡️ Guardrails Simulator (Dev)"
            ])
        else:
            tab_report, tab_visuals, tab_metrics = st.tabs([
                "📋 Patient Summary & Guide",
                "🗺️ Interactive 3D & 2D Eye Maps",
                "📊 Measurements Explained"
            ])
        
        # ================= TAB 1: PATIENT CALIBRATED REPORT =================
        with tab_report:
            st.markdown(f"## 📋 {acc_report.report_title}")
            
            # Executive Summary Box
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.25) 0%, rgba(15, 23, 42, 0.85) 100%); border-left: 5px solid #3B82F6; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
                <h4 style="margin: 0 0 8px 0; color: #60A5FA;">Executive Summary for Patient</h4>
                <p style="margin: 0; font-size: 1.05rem; line-height: 1.6; color: #E2E8F0;">
                    {acc_report.executive_summary}
                </p>
                <div style="margin-top: 12px; display: flex; gap: 16px; font-size: 0.85rem; color: #93C5FD;">
                    <span>📖 <b>Reading Level:</b> {acc_report.readability_grade_level}</span>
                    <span>🛡️ <b>Tone:</b> Reassuring & Non-alarmist</span>
                    <span>🔍 <b>Factual Accuracy:</b> {acc_report.factual_verification.alignment_score * 100:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Physical Metaphors Section
            st.markdown("### 💡 Everyday Physical Models for Understanding Your Cornea")
            analogies = frontal.analogies_used
            if analogies:
                num_cols = min(3, len(analogies))
                cols = st.columns(num_cols)
                icons = ["⌚", "📑", "⛰️", "🔍", "📐"]
                for idx, a in enumerate(analogies[:num_cols]):
                    with cols[idx]:
                        icon = icons[idx % len(icons)]
                        st.markdown(f"""
                        <div class="analogy-card">
                            <div style="font-size: 1.5rem; margin-bottom: 6px;">{icon}</div>
                            <h4 style="margin: 0 0 6px 0; color: #38BDF8;">{a.analogy_name}</h4>
                            <div style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 8px;">Concept: {a.domain_concept}</div>
                            <p style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; margin: 0;">
                                {a.analogy_explanation}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Deep Dive & Color Map Sections in Columns
            c_left, c_right = st.columns([1.1, 0.9])
            with c_left:
                st.markdown(acc_report.structural_deep_dive_markdown)
            
            with c_right:
                st.markdown(acc_report.color_map_guide_markdown)
                
                # Key Takeaways Box
                st.markdown("### 📌 Key Takeaways")
                for t in acc_report.key_takeaways:
                    st.markdown(f"- {t}")
            
            st.markdown("---")
            
            # Doctor Consultation Questions (Empowering Patient Action)
            st.markdown("### 🩺 Tailored Questions to Ask Your Eye Doctor")
            st.markdown("Bring these personalized questions to your next appointment to discuss your vision plan:")
            
            for q in acc_report.consultation_questions:
                st.markdown(f"""
                <div class="question-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">
                            {q.category}
                        </span>
                    </div>
                    <div style="font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 6px;">
                        "{q.question_text}"
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; font-style: italic;">
                        Why ask this: {q.rationale}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Medical Disclaimer
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed #475569; border-radius: 8px; padding: 14px; margin-top: 24px; font-size: 0.8rem; color: #94A3B8; text-align: center;">
                {acc_report.medical_disclaimer}
            </div>
            """, unsafe_allow_html=True)
            
            # Download Report Option
            report_download_content = f"""# {acc_report.report_title}
Generated Patient Education Guide ({extraction.metadata.eye})

## Executive Summary
{acc_report.executive_summary}

{acc_report.structural_deep_dive_markdown}

{acc_report.color_map_guide_markdown}

## Key Takeaways
""" + "\n".join([f"- {t}" for t in acc_report.key_takeaways]) + "\n\n## Doctor Consultation Questions\n" + "\n".join([f"- [{q.category}] {q.question_text} (Rationale: {q.rationale})" for q in acc_report.consultation_questions]) + f"\n\n---\n{acc_report.medical_disclaimer}"
            
            safe_eye_name = (extraction.metadata.eye or "Cornea").replace(' ', '_').replace('(', '').replace(')', '')
            st.download_button(
                label=f"📥 Download Full Patient Guide ({extraction.metadata.eye})",
                data=report_download_content,
                file_name=f"Corneal_Guide_{safe_eye_name}.md",
                mime="text/markdown",
                key=f"download_guide_{active_eye_key}"
            )
        
        # ================= TAB 2: 3D & 2D CORNEAL MAPS =================
        with tab_visuals:
            st.markdown(f"## 🗺️ Interactive Corneal Maps ({extraction.metadata.eye})")
            st.markdown("Explore the 3D surface shape and 2D curvature/thickness distribution of your cornea.")
            
            v_col1, v_col2 = st.columns([1.2, 0.8])
            with v_col1:
                fig_3d = create_3d_corneal_dome_figure(extraction)
                st.plotly_chart(fig_3d, width='stretch', key=f"single_3d_{active_eye_key}")
            
            with v_col2:
                fig_gauge = create_bad_d_gauge_figure(extraction.bad_display.final_bad_d_index)
                st.plotly_chart(fig_gauge, width='stretch', key=f"single_gauge_{active_eye_key}")
                st.markdown("""
                <div style="font-size: 0.85rem; color: #9CA3AF; padding: 10px; background: rgba(17, 24, 39, 0.6); border-radius: 8px;">
                    <b>Understanding the Shape Variation Scale:</b><br>
                    • <b>🟢 Green (&lt; 1.60 SD):</b> Normal standard variation<br>
                    • <b>🟡 Yellow (1.60 - 2.60 SD):</b> Borderline / Suspicious variation<br>
                    • <b>🔴 Red (&gt; 2.60 SD):</b> Noticeable deviation from standard curvature
                </div>
                """, unsafe_allow_html=True)
            
            map_col1, map_col2 = st.columns(2)
            with map_col1:
                fig_topo = create_2d_topography_heatmap(extraction)
                st.plotly_chart(fig_topo, width='stretch', key=f"single_topo_{active_eye_key}")
            with map_col2:
                fig_pachy = create_2d_pachymetry_heatmap(extraction)
                st.plotly_chart(fig_pachy, width='stretch', key=f"single_pachy_{active_eye_key}")
        
        # ================= TAB 3: MEASUREMENTS EXPLAINED =================
        with tab_metrics:
            st.markdown(f"## 📊 Your Key Corneal Measurements Explained ({extraction.metadata.eye})")
            st.markdown("Here is what the primary numbers on your diagnostic scan represent in everyday terms:")
            
            kmax_st, _, _ = get_param_status_info(p_assess.parameter_evaluations, 'kmax')
            cct_st, _, _ = get_param_status_info(p_assess.parameter_evaluations, 'cct')
            thin_st, _, _ = get_param_status_info(p_assess.parameter_evaluations, 'thinnest_pachy')
            post_st, _, _ = get_param_status_info(p_assess.parameter_evaluations, 'posterior_elevation')
            badd_st, _, _ = get_param_status_info(p_assess.parameter_evaluations, 'bad_d')
            
            metrics_table_data = [
                {
                    "Parameter": "Peak Curvature (Kmax)",
                    "Your Measured Value": f"{extraction.keratometry.kmax_diopters or 0.0:.1f} D",
                    "Typical Normal Range": "< 47.0 Diopters",
                    "Status": kmax_st,
                    "Explanation": "Measures the steepness of the most curved point on the front surface of your cornea."
                },
                {
                    "Parameter": "Central Corneal Thickness",
                    "Your Measured Value": f"{extraction.pachymetry.central_corneal_thickness_cct_um or 0.0:.0f} µm",
                    "Typical Normal Range": "500 - 580 µm",
                    "Status": cct_st,
                    "Explanation": "The thickness at the optical center of your eye in micrometers (1 µm = 1/1000th of a mm)."
                },
                {
                    "Parameter": "Thinnest Point Thickness",
                    "Your Measured Value": f"{extraction.pachymetry.thinnest_pachymetry_um or 0.0:.0f} µm",
                    "Typical Normal Range": "> 500 µm",
                    "Status": thin_st,
                    "Explanation": "The minimum structural thickness measured anywhere across your entire cornea."
                },
                {
                    "Parameter": "Back Surface Elevation",
                    "Your Measured Value": f"+{extraction.elevation.posterior_elevation_thinnest_um or 0.0:.1f} µm",
                    "Typical Normal Range": "< +12.0 µm",
                    "Status": post_st,
                    "Explanation": "Checks whether the back surface of the cornea bulges outward compared to a standard spherical reference."
                },
                {
                    "Parameter": "BAD-D Deviation Index",
                    "Your Measured Value": f"{extraction.bad_display.final_bad_d_index or 0.0:.2f} SD",
                    "Typical Normal Range": "< 1.60 SD",
                    "Status": badd_st,
                    "Explanation": "A composite score comparing multiple front and back surface parameters to standard reference populations."
                }
            ]
            
            st.dataframe(metrics_table_data, width='stretch')
            
            st.markdown("""
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 16px; padding: 14px; background: rgba(17, 24, 39, 0.7); border-radius: 8px; border: 1px solid #1F2937;">
                💡 <b>Note on Units:</b>
                <ul style="margin-bottom: 0; margin-top: 6px;">
                    <li><b>Diopters (D):</b> Optical unit measuring the refractive bending power or curvature of the eye surface.</li>
                    <li><b>Micrometers (µm):</b> 1 micrometer equals 0.001 millimeter (roughly the thickness of a microscopic sheet of paper).</li>
                    <li><b>Standard Deviation (SD):</b> Measures how closely your scan compares to thousands of healthy eyes in clinical databases.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # ================= DEVELOPER ONLY TABS (Available when ?dev=true) =================
        if dev_mode:
            with tab_pfit_stages:
                st.markdown("## 🧠 P-FIT Cognitive Architecture Inspector")
                st.markdown("Explore the progressive transformation of sensory multimodal tokens across the 4 cognitive proxies.")
                
                with st.expander("👁️ Stage 1: Occipital-Temporal Proxy (Visual OCR & Feature Extraction)", expanded=True):
                    st.markdown("""
                    **Biological Function (Brodmann Areas 17, 18, 19, 37):** Visual sensation, edge detection, tokenization, and optical character recognition.
                    **Pipeline Action:** Multimodal ingestion of Pentacam 4-Maps / BAD-III displays into strict Pydantic schema.
                    """)
                    s1_col1, s1_col2 = st.columns(2)
                    with s1_col1:
                        st.markdown("#### Extracted Tomography Metrics")
                        st.json({
                            "metadata": extraction.metadata.model_dump(),
                            "keratometry": extraction.keratometry.model_dump(),
                            "pachymetry": extraction.pachymetry.model_dump()
                        })
                    with s1_col2:
                        st.markdown("#### Elevation & Topometric Indices")
                        st.json({
                            "elevation": extraction.elevation.model_dump(),
                            "bad_display": extraction.bad_display.model_dump(),
                            "topometric": extraction.topometric.model_dump(),
                            "confidence_score": extraction.extraction_confidence_score,
                            "ocr_notes": extraction.raw_ocr_notes
                        })
                
                with st.expander("🧩 Stage 2: Parietal Cortex Proxy (Structural Abstraction & Deterministic Guardrails)", expanded=True):
                    st.markdown("""
                    **Biological Function (Brodmann Areas 7, 39, 40):** Spatial manipulation, cross-modal integration, and geometric mental modeling.
                    **Pipeline Action:** Integrates multi-parametric parameters, computes spatial coupling, and enforces hard deterministic boundaries.
                    """)
                    
                    st.markdown(f"**Unified Geometric Profile:** *{parietal.unified_geometric_profile}*")
                    st.markdown(f"**Dominant Biomechanical Feature:** *{parietal.dominant_biomechanical_feature}*")
                    
                    # Deterministic parameter table
                    st.markdown("#### Deterministic Parameter Assessment Table")
                    eval_data = []
                    for k, eval_item in p_assess.parameter_evaluations.items():
                        eval_data.append({
                            "Parameter": eval_item.name,
                            "Extracted Value": f"{eval_item.value} {eval_item.unit}" if eval_item.value is not None else "N/A",
                            "Reference Range": eval_item.reference_range,
                            "Status": eval_item.status.value,
                            "Interpretation": eval_item.clinical_interpretation
                        })
                    st.dataframe(eval_data, width='stretch')
                    
                    if p_assess.spatial_coupling_detected:
                        st.warning(f"⚠️ **Spatial Coupling Detected:** {p_assess.spatial_coupling_notes}")
                
                with st.expander("🗣️ Stage 3: Frontal / Prefrontal Proxy (Cognitive Scaffolding & Analogies)", expanded=False):
                    st.markdown("""
                    **Biological Function (Brodmann Areas 9, 10, 45, 46):** Hypothesis formation, analogical mapping, working memory reduction, and semantic translation.
                    **Pipeline Action:** Bridges complex 3D Scheimpflug geometry into everyday physical metaphors.
                    """)
                    st.markdown(f"**Headline Draft:** {frontal.headline_summary}")
                    st.markdown("#### Scaffolding Analogies:")
                    for a in frontal.analogies_used:
                        st.markdown(f"- **{a.analogy_name} ({a.domain_concept}):** {a.analogy_explanation}")
                
                with st.expander("🛡️ Stage 4: Anterior Cingulate Cortex / ACC Proxy (Cognitive Control & Gating)", expanded=False):
                    st.markdown("""
                    **Biological Function (Brodmann Areas 24, 32):** Conflict detection, error correction, inhibitory control, and goal alignment.
                    **Pipeline Action:** Inhibits diagnostic assertions, softens anxiety triggers, verifies numbers, and synthesizes doctor consultation questions.
                    """)
                    g_audit = acc_report.inhibitory_gating_audit
                    st.markdown("#### Inhibitory Gating Audit:")
                    for s in g_audit.suppressed_proclamations:
                        st.markdown(f"🚫 **Suppressed Assertion:** {s}")
                    for j in g_audit.softened_jargon_terms:
                        st.markdown(f"🛡️ **Softened Term:** {j}")
                        
                    st.markdown("#### Factual Verification Audit:")
                    for m in acc_report.factual_verification.metrics_verified:
                        st.markdown(f"✅ Verified: `{m}`")
            
            with tab_guardrails:
                st.markdown("## 🛡️ Deterministic Guardrails & Clinical Cutoff Simulator")
                st.markdown("Interactive clinical boundary engine grounded in the *Global Consensus on Keratoconus & Ectatic Diseases*.")
                
                sim_col1, sim_col2 = st.columns([0.9, 1.1])
                
                with sim_col1:
                    st.markdown("#### 🎛️ Live Parameter Simulator")
                    def_kmax = max(30.0, min(70.0, float(extraction.keratometry.kmax_diopters or 44.0)))
                    sim_kmax = st.slider("Maximum Keratometry (Kmax in D)", 30.0, 70.0, def_kmax, 0.1, key=f"sim_kmax_{active_eye_key}")
                    
                    def_thinnest = max(300, min(700, int(extraction.pachymetry.thinnest_pachymetry_um or 540)))
                    sim_thinnest = st.slider("Thinnest Pachymetry (µm)", 300, 700, def_thinnest, 5, key=f"sim_thin_{active_eye_key}")
                    
                    def_post_elev = max(-5.0, min(50.0, float(extraction.elevation.posterior_elevation_thinnest_um or 8.0)))
                    sim_post_elev = st.slider("Posterior Elevation (µm)", -5.0, 50.0, def_post_elev, 0.5, key=f"sim_post_{active_eye_key}")
                    
                    def_bad_d = max(0.0, min(10.0, float(extraction.bad_display.final_bad_d_index or 1.0)))
                    sim_bad_d = st.slider("BAD-D Final Index (SD)", 0.0, 10.0, def_bad_d, 0.1, key=f"sim_badd_{active_eye_key}")
                    
                    # Construct simulated extraction
                    sim_ext = extraction.model_copy(deep=True)
                    sim_ext.keratometry.kmax_diopters = sim_kmax
                    sim_ext.pachymetry.thinnest_pachymetry_um = float(sim_thinnest)
                    sim_ext.elevation.posterior_elevation_thinnest_um = sim_post_elev
                    sim_ext.bad_display.final_bad_d_index = sim_bad_d
                    
                    sim_assessment = run_deterministic_assessment(sim_ext)
                    
                    st.markdown("---")
                    st.markdown(f"**Deterministic Risk Tier:** `{sim_assessment.deterministic_risk_tier.value}`")
                    st.markdown(f"**Abnormal Count:** `{sim_assessment.abnormal_count}` | **Borderline Count:** `{sim_assessment.borderline_count}`")
                    st.progress(sim_assessment.risk_score_percentage / 100.0, text=f"Composite Risk Score: {sim_assessment.risk_score_percentage:.1f}%")
                    
                with sim_col2:
                    st.markdown("#### 📖 Empirical Clinical Cutoffs Table")
                    cutoff_rows = []
                    for k, rule in CUTOFF_RULES.items():
                        cutoff_rows.append({
                            "Index": rule.name,
                            "Normal Limit": f"< {rule.normal_max} {rule.unit}" if rule.normal_max else f"> {rule.normal_min} {rule.unit}",
                            "Borderline / Suspect": f"{rule.borderline_min} - {rule.borderline_max} {rule.unit}",
                            "Abnormal Trigger": f">= {rule.abnormal_trigger_high} {rule.unit}" if rule.abnormal_trigger_high else f"<= {rule.abnormal_trigger_low} {rule.unit}",
                            "Clinical Rationale": rule.description
                        })
                    st.dataframe(cutoff_rows, width='stretch')

else:
    st.markdown("""
    <div class="upload-card">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">📋</div>
        <h3 style="margin: 0 0 8px 0; color: #FFFFFF;">No Scan Report Loaded</h3>
        <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 16px auto; font-size: 0.95rem;">
            Please select a <b>Clinical Sample Scan</b>, upload a <b>Pentacam Report (PNG, JPG, PDF)</b> from your local storage, 
            or enter a <b>Local File Path</b> in the sidebar to generate your personalized report.
        </p>
    </div>
    """, unsafe_allow_html=True)


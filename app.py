"""
PentaRead: Intelligent Scheimpflug Pentacam Scan Translation.
Editorial Studio Aesthetic: Royal Cobalt Navy Blue & Clean Crisp White.
Translates complex corneal tomography scans into intuitive insights, 3D/2D models, and doctor discussion guides.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
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
    create_3d_curvature_figure,
    create_2d_topography_heatmap,
    create_3d_pachymetry_figure,
    create_2d_pachymetry_heatmap,
    create_3d_elevation_figure,
    create_2d_elevation_heatmap,
    create_3d_badd_figure,
    create_bad_d_gauge_figure,
    create_3d_corneal_dome_figure
)
from src.utils.asymmetry import compute_asymmetry_stats

# Page configuration
st.set_page_config(
    page_title="PentaRead • Studio Corneal Tomography Guide",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check for developer mode via URL query parameter (?dev=true)
dev_param = st.query_params.get("dev", "")
if isinstance(dev_param, list):
    dev_param = dev_param[0] if dev_param else ""
dev_mode = str(dev_param).lower() in ["true", "1", "yes"]

# ==============================================================================
# EDITORIAL STUDIO AESTHETIC: ROYAL COBALT NAVY BLUE & CRISP WHITE
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit Platform Chrome */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    div[data-testid="stToolbarActions"] {display: none !important;}
    header[data-testid="stHeader"] {background: transparent !important;}
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Clean Light Studio Canvas (Nine Sixty Light Foundation) */
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    
    /* Global Typography & Font Color Enforcement on Light Canvas */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #1829B0 !important;
        font-weight: 800 !important;
    }
    
    p, span, div, li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #0F172A;
    }

    /* Streamlit Native Widget Labels & Text (Radios, Toggles, Inputs, File Uploader) */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span,
    .stRadio label,
    .stRadio span,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span,
    [data-testid="stToggle"] label p,
    [data-testid="stToggle"] label span,
    [data-testid="stTextInput"] label p,
    [data-testid="stTextInput"] label span,
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] label span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important;
    }

    input[type="text"], input[type="number"], textarea {
        color: #0F172A !important;
        background-color: #F8F9FA !important;
        border: 1.5px solid #CBD5E1 !important;
    }

    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span {
        color: #1829B0 !important;
        font-weight: 700 !important;
    }
    [data-testid="stPopoverBody"] p,
    [data-testid="stPopoverBody"] label,
    [data-testid="stPopoverBody"] span {
        color: #0F172A !important;
    }

    /* Dialog Modals (@st.dialog) */
    div[role="dialog"], [data-testid="stModal"] {
        background-color: #FFFFFF !important;
    }
    div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4 {
        color: #1829B0 !important;
    }
    div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] div {
        color: #0F172A !important;
    }

    /* Top Brand Navigation Bar */
    .brand-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 40px;
        background-color: #1829B0;
        color: #FFFFFF !important;
        margin: -60px -50px 0 -50px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }
    .brand-top-bar * {
        color: #FFFFFF !important;
    }
    .brand-logo {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #FFFFFF !important;
    }
    .brand-nav-links {
        display: flex;
        gap: 24px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #E0E7FF !important;
    }
    
    /* Full-Width Bold Royal Navy Hero Section */
    .hero-banner {
        background-color: #1829B0;
        color: #FFFFFF !important;
        text-align: center;
        padding: 65px 30px 60px 30px;
        margin: 0 -50px 40px -50px;
    }
    .hero-banner * {
        color: #FFFFFF !important;
    }
    .hero-pill {
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #FFFFFF !important;
        padding: 5px 16px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
        margin: 0 0 16px 0;
        line-height: 1.15;
    }
    .hero-desc {
        font-size: 1.15rem;
        color: #E0E7FF !important;
        line-height: 1.65;
        max-width: 780px;
        margin: 0 auto;
    }
    
    /* Modernist Studio Ghost Buttons */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #1829B0 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1.5px solid #1829B0 !important;
        border-radius: 2px !important;
        padding: 10px 22px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 0.8rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #0E1A80 !important;
        color: #FFFFFF !important;
        border-color: #0E1A80 !important;
        box-shadow: 0 4px 14px rgba(24, 41, 176, 0.25) !important;
    }
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #1829B0 !important;
        font-weight: 700 !important;
        border: 1.5px solid #1829B0 !important;
        border-radius: 2px !important;
        padding: 10px 20px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 0.8rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="secondary"]:hover, .stButton > button[kind="secondary"]:hover {
        background-color: #1829B0 !important;
        color: #FFFFFF !important;
        border-color: #1829B0 !important;
    }

    /* Container Border Styling for Studio Editorial Blocks */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-top: 4px solid #1829B0 !important;
        border-radius: 2px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #1829B0 !important;
        box-shadow: 0 8px 24px rgba(24, 41, 176, 0.12) !important;
    }

    /* Solid Royal Navy Studio Container Block (Nine Sixty Checkerboard) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-navy-card) {
        background-color: #1829B0 !important;
        border: 1.5px solid #1829B0 !important;
        border-top: 4px solid #0E1A80 !important;
        box-shadow: 0 6px 20px rgba(24, 41, 176, 0.2) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-navy-card) * {
        color: #FFFFFF !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-navy-card) p {
        color: #E0E7FF !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-navy-card) button {
        background-color: transparent !important;
        border: 1.5px solid #FFFFFF !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.studio-navy-card) button:hover {
        background-color: #FFFFFF !important;
        border-color: #FFFFFF !important;
        color: #1829B0 !important;
        box-shadow: 0 4px 14px rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Scan Header & Greeting Banner (Solid Royal Navy) */
    .scan-greeting-banner {
        background-color: #1829B0;
        color: #FFFFFF !important;
        border-radius: 2px;
        padding: 30px 34px;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(24, 41, 176, 0.15);
    }
    .scan-greeting-banner * {
        color: #FFFFFF !important;
    }
    .greeting-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0 0 8px 0;
        letter-spacing: -0.01em;
    }
    .greeting-sub {
        font-size: 1.05rem;
        color: #E0E7FF !important;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Metric Cards Grid */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3.5px solid #1829B0;
        border-radius: 4px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 800;
        color: #1829B0;
        margin-bottom: 6px;
    }
    .metric-ref {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    
    /* Clean Status Badges */
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .status-pill-normal {
        color: #065F46;
        background: #D1FAE5;
        border: 1px solid #10B981;
    }
    .status-pill-borderline {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #F59E0B;
    }
    .status-pill-abnormal {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #EF4444;
    }
    .status-pill-postop {
        color: #1E40AF;
        background: #DBEAFE;
        border: 1px solid #3B82F6;
    }
    
    /* Studio Checkerboard Block Cards for 3 Examples */
    .studio-block-navy {
        background-color: #1829B0;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 28px 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .studio-block-white {
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        border-top: 4px solid #1829B0;
        border-radius: 4px;
        padding: 28px 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Structural Issues & Question Cards */
    .issue-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1829B0;
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
    .question-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1829B0;
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
if "current_patient_name" not in st.session_state:
    st.session_state.current_patient_name = None
if "patient_name_override" not in st.session_state:
    st.session_state.patient_name_override = None
if "selected_sample_key" not in st.session_state:
    st.session_state.selected_sample_key = "case_1_normal"
if "pipeline_states" not in st.session_state:
    st.session_state.pipeline_states = {}
if "active_eye_tab" not in st.session_state:
    st.session_state.active_eye_tab = "OD"
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = "pentacam_scan.png"
if "uploaded_mime_type" not in st.session_state:
    st.session_state.uploaded_mime_type = "image/png"


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


tier_colors = {
    RiskTier.NORMAL_CORNEA: ("#065F46", "Normal Physiologic Cornea", "🟢", "status-pill-normal"),
    RiskTier.SUSPECT_FORME_FRUSTE: ("#92400E", "Subclinical / Forme Fruste Suspect", "🟡", "status-pill-borderline"),
    RiskTier.BORDERLINE_ASYMMETRY: ("#92400E", "Borderline Topographic Asymmetry", "🟡", "status-pill-borderline"),
    RiskTier.MILD_ECTASIA_PATTERN: ("#991B1B", "Focal Ectasia Pattern (Mild)", "🔴", "status-pill-abnormal"),
    RiskTier.MODERATE_TO_ADVANCED_ECTASIA: ("#991B1B", "Ectasia / Keratoconus Pattern", "🔴", "status-pill-abnormal"),
    RiskTier.POST_REFRACTIVE_ALTERATION: ("#1829B0", "Post-Refractive Alteration", "🔵", "status-pill-postop")
}


# ==============================================================================
# POPUP DIALOGS (Studio Clean Modal Design)
# ==============================================================================

@st.dialog("The Watch Glass Window • Understanding Curvature")
def dialog_watch_glass(kmax_val: float, fellow_val: Optional[float] = None, labels: tuple = ("OD (Right Eye)", "OS (Left Eye)")):
    comp_markup = ""
    if fellow_val is not None:
        delta_k = abs(kmax_val - fellow_val)
        delta_color = "#991B1B" if delta_k >= 1.0 else "#065F46"
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Bilateral Curvature Comparison</div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; gap: 14px;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[0]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">{kmax_val:.1f} D</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[1]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">{fellow_val:.1f} D</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">Asymmetry (Δ)</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace;">{delta_k:.1f} D</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 8px;">Physiologic corneas remain below 47.0 D with inter-eye asymmetry &lt; 1.0 D.</div>
        </div>
        """
    else:
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Your Scan Measurement</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">
                Peak Curvature (Kmax): {kmax_val:.1f} Diopters
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Standard healthy corneas measure below 47.0 Diopters.</div>
        </div>
        """

    st.markdown(f"""
    <div style="font-size: 1rem; line-height: 1.6; color: #0F172A;">
        <h4 style="color: #1829B0; margin: 0 0 10px 0;">How a Cornea is Like a Watch Crystal:</h4>
        <p>
            Imagine the clear glass window on the front of a wristwatch. When the glass has a smooth, gentle, 
            and even dome, light passes straight through without distortion, letting you see the watch hands clearly.
        </p>
        {comp_markup}
        <p>
            When curvature becomes too steep in one spot (like an antique curved magnifying watch crystal), 
            incoming light rays bend too sharply, causing blur, halo rings, and astigmatism that regular glasses cannot completely fix.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Close Example", type="primary", key="close_dlg_watch"):
        st.rerun()


@st.dialog("The Stack of Paper Sheets • Understanding Thickness")
def dialog_paper_sheets(thinnest_val: float, fellow_val: Optional[float] = None, labels: tuple = ("OD (Right Eye)", "OS (Left Eye)")):
    comp_markup = ""
    if fellow_val is not None:
        delta_p = abs(thinnest_val - fellow_val)
        delta_color = "#991B1B" if delta_p >= 30.0 else "#065F46"
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Bilateral Thickness Comparison</div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; gap: 14px;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[0]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">{thinnest_val:.0f} µm</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[1]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">{fellow_val:.0f} µm</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">Disparity (Δ)</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace;">{delta_p:.0f} µm</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 8px;">A healthy adult cornea is ~540 µm thick. Inter-eye thickness difference should remain &lt; 30 µm.</div>
        </div>
        """
    else:
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Your Scan Measurement</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">
                Thinnest Point: {thinnest_val:.0f} µm
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">A healthy adult cornea is approximately 540 sheets thick (540 µm).</div>
        </div>
        """

    st.markdown(f"""
    <div style="font-size: 1rem; line-height: 1.6; color: #0F172A;">
        <h4 style="color: #1829B0; margin: 0 0 10px 0;">How Corneal Thickness is Measured:</h4>
        <p>
            The clear front window of your eye is remarkably thin—roughly half a millimeter. To measure it accurately, 
            eye doctors use <b>micrometers (µm)</b>, where 1 µm is just 1/1,000th of a millimeter.
        </p>
        {comp_markup}
        <p>
            Think of your cornea as a neat stack of 540 microscopic sheets of paper. If localized pressure or biomechanical 
            softening causes some sheets to thin out in one quadrant (e.g. dropping below 470 µm), that specific spot becomes less rigid, 
            prompting eye specialists to monitor for focal protrusion.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Close Example", type="primary", key="close_dlg_paper"):
        st.rerun()


@st.dialog("The Mountain Contour Slope • Understanding Posterior Elevation")
def dialog_mountain_contours(elev_val: float, fellow_val: Optional[float] = None, labels: tuple = ("OD (Right Eye)", "OS (Left Eye)")):
    comp_markup = ""
    if fellow_val is not None:
        delta_e = abs(elev_val - fellow_val)
        delta_color = "#991B1B" if delta_e >= 5.0 else "#065F46"
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Bilateral Posterior Elevation Comparison</div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; gap: 14px;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[0]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">+{elev_val:.1f} µm</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">{labels[1]}</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">+{fellow_val:.1f} µm</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">Asymmetry (Δ)</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace;">{delta_e:.1f} µm</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 8px;">Baseline physiologic elevation remains under +12.0 µm with inter-eye difference &lt; 5.0 µm.</div>
        </div>
        """
    else:
        comp_markup = f"""
        <div style="background: #F8F9FA; border: 1.5px solid #1829B0; border-radius: 4px; padding: 16px; margin: 16px 0;">
            <div style="color: #1829B0; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;">Your Scan Measurement</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #1829B0; font-family: 'JetBrains Mono', monospace;">
                Posterior Elevation: +{elev_val:.1f} µm
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Baseline physiologic elevation remains under +12.0 µm.</div>
        </div>
        """

    st.markdown(f"""
    <div style="font-size: 1rem; line-height: 1.6; color: #0F172A;">
        <h4 style="color: #1829B0; margin: 0 0 10px 0;">How Hikers & Doctors Read Topographic Contours:</h4>
        <p>
            When hikers examine a topographic trail map, concentric contour rings show mountain peaks rising above sea level. 
            The Pentacam tomographer does the exact same thing for the <i>back surface</i> of your cornea by comparing it against 
            a smooth reference sphere (Best Fit Sphere).
        </p>
        {comp_markup}
        <p>
            The back surface is the earliest early-warning alarm for corneal shape changes. If a localized bulge 
            rises above +16 µm on the back surface, it alerts the doctor long before the front surface shows noticeable symptoms!
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Close Example", type="primary", key="close_dlg_mountain"):
        st.rerun()


# ==============================================================================
# PIPELINE EXECUTION HELPER
# ==============================================================================

def execute_pipeline(
    image_bytes: Optional[bytes] = None, 
    file_name: str = "scan.png", 
    mime_type: str = "image/png", 
    mock_case_id: Optional[str] = None,
    patient_name_input: Optional[str] = None
):
    """Executes the P-FIT analysis pipeline and navigates to the scan page."""
    orchestrator = PFITPipelineOrchestrator(
        api_key=GEMINI_API_KEY if GEMINI_API_KEY else None,
        model_name=DEFAULT_MODEL
    )
    with st.spinner("Analyzing your scan and generating your personalized visual guide..."):
        states_dict = orchestrator.run_multi_eye_pipeline(
            image_bytes=image_bytes,
            image_mime_type=mime_type,
            source_filename=file_name,
            mock_case_id=mock_case_id
        )
        st.session_state.pipeline_states = states_dict

        # Resolve and synchronize patient name for this scan run
        detected_name = None
        if patient_name_input and patient_name_input.strip():
            detected_name = patient_name_input.strip()
        elif mock_case_id and mock_case_id in SAMPLE_CASES:
            detected_name = SAMPLE_CASES[mock_case_id]["data"].metadata.patient_name
        elif states_dict:
            for s in states_dict.values():
                if s.stage1_extraction.metadata and s.stage1_extraction.metadata.patient_name:
                    detected_name = s.stage1_extraction.metadata.patient_name
                    break

        # If still not found, try to derive from file_name (e.g. "John_Doe_Pentacam.pdf" -> "John Doe")
        if not detected_name and file_name and file_name != "scan.png":
            stem = Path(file_name).stem.replace("_", " ").replace("-", " ").replace(".", " ")
            clean_words = [
                w for w in stem.split() 
                if w.lower() not in ["scan", "pentacam", "report", "cornea", "od", "os", "ou", "pdf", "png", "jpg", "patient", "sample", "case", "refractive"]
            ]
            if len(clean_words) >= 2 and all(w.isalpha() for w in clean_words[:2]):
                detected_name = f"{clean_words[0].capitalize()} {clean_words[1].capitalize()}"

        if not detected_name:
            detected_name = "User"

        # Always update the active patient name for this new scan and clear legacy overrides
        st.session_state.current_patient_name = detected_name
        st.session_state.patient_name_override = None

        # Synchronize patient_name across every eye in pipeline states
        if states_dict:
            for s in states_dict.values():
                if s.stage1_extraction.metadata:
                    s.stage1_extraction.metadata.patient_name = detected_name

            first_key = list(states_dict.keys())[0]
            if len(states_dict) > 1:
                st.session_state.active_eye_tab = "BILATERAL"
                st.session_state.radio_eye_selector = "⚖️ Bilateral Comparison"
            else:
                st.session_state.active_eye_tab = first_key
                st.session_state.radio_eye_selector = "OS (Left Eye)" if first_key == "OS" else "OD (Right Eye)"
            st.session_state.current_page = "scan"
            st.rerun()


# ==============================================================================
# PAGE 1: PENTAREAD FRONTPAGE (STUDIO EDITORIAL STYLE)
# ==============================================================================

if st.session_state.current_page == "home":
    # Top studio bar
    st.markdown("""
    <div class="brand-top-bar">
        <div class="brand-logo">PentaRead</div>
        <div class="brand-nav-links">
            <span>Clinical Scaffolding</span>
            <span>3D Models</span>
            <span>Doctor Guide</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Full-Width Royal Cobalt Navy Section (Nine Sixty Studio Aesthetic)
    st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; align-items: center; justify-content: space-between; max-width: 1100px; margin: 0 auto;">
            <span style="font-size: 2.2rem; color: rgba(255,255,255,0.35); user-select: none;">&#10094;</span>
            <div style="max-width: 820px; text-align: center;">
                <div class="hero-pill">Cognitive Scaffolding • Pentacam Scheimpflug</div>
                <h1 class="hero-title">We translate complex corneal scans into beautifully clear patient insights.</h1>
                <p class="hero-desc">
                    An externalized cognitive prosthesis bridging multi-parametric Scheimpflug tomography 
                    into intuitive 3D and 2D visual models, everyday physical metaphors, and calm doctor consultation guides.
                </p>
            </div>
            <span style="font-size: 2.2rem; color: rgba(255,255,255,0.35); user-select: none;">&#10095;</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Studio Section Heading ("What we do")
    st.markdown("""
    <div style="text-align: center; margin: 30px 0 28px 0;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #1829B0; margin: 0 0 6px 0; letter-spacing: -0.01em;">
            Choose Your Scan Input Method
        </h2>
        <p style="color: #64748B; font-size: 0.95rem; margin: 0;">Select how you would like to ingest your Pentacam diagnostic scan</p>
    </div>
    """, unsafe_allow_html=True)

    # 3 Studio Action Cards (Inside Border Containers)
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 10px 6px 8px 6px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">📤</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;">Upload Your Penta Scan</div>
                <div style="font-size: 0.92rem; color: #475569; line-height: 1.6; margin-bottom: 20px;">
                    Have a scan image or digital PDF report from your clinic? Ingest your 4-Maps Refractive, BAD-III, or Topometric printout.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Upload Your Scan", type="primary", key="btn_go_upload", width='stretch'):
                st.session_state.current_page = "upload"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 10px 6px 8px 6px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">📁</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;">Paste the File Path</div>
                <div style="font-size: 0.92rem; color: #475569; line-height: 1.6; margin-bottom: 20px;">
                    Stored on your computer? Provide the absolute or relative disk filepath to load and analyze it directly.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Paste File Path", type="primary", key="btn_go_filepath", width='stretch'):
                st.session_state.current_page = "filepath"
                st.rerun()

    with c3:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 10px 6px 8px 6px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">🧪</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;">Scan a Demo Report</div>
                <div style="font-size: 0.92rem; color: #475569; line-height: 1.6; margin-bottom: 20px;">
                    Explore curated clinical cases: normal cornea, subclinical forme fruste, advanced keratoconus cone, and LASIK.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Scan a Demo Report", type="primary", key="btn_go_demo", width='stretch'):
                st.session_state.current_page = "demo"
                st.rerun()

    # Studio Features Strip
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #F8F9FA; border-top: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; padding: 28px 40px; margin: 0 -50px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
        <div style="text-align: center;">
            <div style="font-weight: 800; color: #1829B0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em;">Physical Metaphors</div>
            <div style="font-size: 0.82rem; color: #64748B;">Watch glass, paper sheets, mountain slopes</div>
        </div>
        <div style="text-align: center;">
            <div style="font-weight: 800; color: #1829B0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em;">Interactive 3D Maps</div>
            <div style="font-size: 0.82rem; color: #64748B;">Curvature, thickness, elevation meshes</div>
        </div>
        <div style="text-align: center;">
            <div style="font-weight: 800; color: #1829B0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em;">Zero Alarmism</div>
            <div style="font-size: 0.82rem; color: #64748B;">Strict ACC inhibitory gating &amp; reassurance</div>
        </div>
        <div style="text-align: center;">
            <div style="font-weight: 800; color: #1829B0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em;">Doctor Consultation Q&amp;A</div>
            <div style="font-size: 0.82rem; color: #64748B;">Tailored questions with clinical rationales</div>
        </div>
    </div>
    
    <!-- Studio Footer (Nine Sixty Style) -->
    <div style="text-align: center; margin: 55px 0 20px 0; padding-top: 35px; border-top: 1px solid #E2E8F0;">
        <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.14em; color: #1829B0; font-weight: 700; margin-bottom: 8px;">
            Studio Clinical Inquiries
        </div>
        <p style="color: #64748B; font-size: 0.9rem; max-width: 520px; margin: 0 auto 18px auto;">
            Connect with clinical imaging specialists or submit multi-parametric Scheimpflug tomography datasets.
        </p>
        <div style="display: inline-block; border: 1.5px solid #1829B0; padding: 8px 24px; font-size: 0.85rem; color: #1829B0; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em; margin-bottom: 22px;">
            clinical.scaffolding@pentaread.org
        </div>
        <div style="font-size: 0.76rem; color: #94A3B8; letter-spacing: 0.04em;">
            PentaRead • Editorial Scheimpflug Tomography Translation Studio • Grounded in P-FIT Cognitive Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: UPLOAD YOUR PENTA SCAN
# ==============================================================================

elif st.session_state.current_page == "upload":
    if st.button("← Back to PentaRead Home", key="back_from_upload"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown("""
    <div style="margin: 20px 0 24px 0;">
        <span style="border: 1px solid #1829B0; color: #1829B0; padding: 4px 12px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Document Ingestion</span>
        <h2 style="font-size: 2.2rem; color: #1829B0; margin: 8px 0 8px 0; font-weight: 800; letter-spacing: -0.01em;">
            Upload Your Pentacam Scan
        </h2>
        <p style="color: #475569; font-size: 1rem; max-width: 750px;">
            Upload your Scheimpflug corneal scan image (PNG, JPG, JPEG, WEBP) or multi-page digital PDF report. 
            PentaRead extracts every curvature, pachymetry, and elevation index with clinical accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your Pentacam report file",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        help="Upload Pentacam 4-Maps Refractive, Belin-Ambrósio BAD-III, or Topometric printout."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        f_name = uploaded_file.name
        if f_name.lower().endswith(".pdf"):
            m_type = "application/pdf"
        elif f_name.lower().endswith((".jpg", ".jpeg")):
            m_type = "image/jpeg"
        elif f_name.lower().endswith(".webp"):
            m_type = "image/webp"
        else:
            m_type = "image/png"

        st.session_state.uploaded_image_bytes = file_bytes
        st.session_state.uploaded_filename = f_name
        st.session_state.uploaded_mime_type = m_type

        st.success(f"📁 Document Ready: **{f_name}** ({len(file_bytes)/1024:.1f} KB)")

        if m_type.startswith("image/"):
            with st.expander("🖼️ Preview Scan Image", expanded=True):
                st.image(file_bytes, caption=f_name, width='stretch')

        patient_name_upload = st.text_input(
            "Patient Name (Optional)",
            placeholder="e.g. Alex Morgan (leave blank to auto-detect from scan)",
            key="input_upload_patient_name"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Scan & Generate Guide →", type="primary", key="btn_run_uploaded"):
            execute_pipeline(
                image_bytes=file_bytes, 
                file_name=f_name, 
                mime_type=m_type,
                patient_name_input=patient_name_upload
            )


# ==============================================================================
# PAGE 3: PASTE THE FILE PATH
# ==============================================================================

elif st.session_state.current_page == "filepath":
    if st.button("← Back to PentaRead Home", key="back_from_filepath"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown("""
    <div style="margin: 20px 0 24px 0;">
        <span style="border: 1px solid #1829B0; color: #1829B0; padding: 4px 12px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Local Disk Ingestion</span>
        <h2 style="font-size: 2.2rem; color: #1829B0; margin: 8px 0 8px 0; font-weight: 800; letter-spacing: -0.01em;">
            Paste Your Local File Path
        </h2>
        <p style="color: #475569; font-size: 1rem; max-width: 750px;">
            Specify the absolute or relative path to your Pentacam report on your local drive.
        </p>
    </div>
    """, unsafe_allow_html=True)

    input_path = st.text_input(
        "File Path on Disk",
        placeholder=r"C:\Scans\pentacam_patient_report.png or ./sample_report.pdf",
        help="Paste the full path to your Pentacam image or PDF."
    )

    if input_path:
        p = Path(input_path.strip().strip('"').strip("'"))
        if p.is_file():
            try:
                file_bytes = p.read_bytes()
                f_name = p.name
                if f_name.lower().endswith(".pdf"):
                    m_type = "application/pdf"
                elif f_name.lower().endswith((".jpg", ".jpeg")):
                    m_type = "image/jpeg"
                elif f_name.lower().endswith(".webp"):
                    m_type = "image/webp"
                else:
                    m_type = "image/png"

                st.session_state.uploaded_image_bytes = file_bytes
                st.session_state.uploaded_filename = f_name
                st.session_state.uploaded_mime_type = m_type

                st.success(f"✅ Found File: **{f_name}** ({len(file_bytes)/1024:.1f} KB)")
                patient_name_filepath = st.text_input(
                    "Patient Name (Optional)",
                    placeholder="e.g. Alex Morgan (leave blank to auto-detect from scan)",
                    key="input_filepath_patient_name"
                )
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Analyze Local File & Generate Guide →", type="primary", key="btn_run_filepath"):
                    execute_pipeline(
                        image_bytes=file_bytes, 
                        file_name=f_name, 
                        mime_type=m_type,
                        patient_name_input=patient_name_filepath
                    )
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.warning("⚠️ File not found at the specified path. Please verify the location.")


# ==============================================================================
# PAGE 4: SCAN A DEMO REPORT
# ==============================================================================

elif st.session_state.current_page == "demo":
    if st.button("← Back to PentaRead Home", key="back_from_demo"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown("""
    <div style="margin: 20px 0 24px 0;">
        <span style="border: 1px solid #1829B0; color: #1829B0; padding: 4px 12px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">Instant Demonstration</span>
        <h2 style="font-size: 2.2rem; color: #1829B0; margin: 8px 0 8px 0; font-weight: 800; letter-spacing: -0.01em;">
            Select a Verified Clinical Case Study
        </h2>
        <p style="color: #475569; font-size: 1rem; max-width: 750px;">
            Choose from authentic Scheimpflug clinical cases across different corneal geometries. 
            Explore normal baselines, early borderline patterns, focal cones, and laser corrections.
        </p>
    </div>
    """, unsafe_allow_html=True)

    demo_cases = [
        ("case_1_normal", "Sarah Jenkins", "Normal Physiologic Cornea", "Standard adult cornea with gentle prolate curvature, intact pachymetry (548 µm), and baseline elevation (+7 µm)."),
        ("case_2_subclinical", "Michael Chang", "Subclinical / Forme Fruste Suspect", "Subtle early posterior elevation (+15 µm) and borderline BAD-D (2.15 SD) with mild inferior thinning."),
        ("case_3_keratoconus", "David Miller", "Advanced Keratoconus Cone", "Classic steep cone (Kmax 53.8 D), marked pachymetric thinning (436 µm), and posterior elevation (+28 µm)."),
        ("case_4_post_lasik", "Rachel Adams", "Post-Refractive LASIK Scan", "Flat central curvature (38.2 D) and reduced central pachymetry post-myopic ablation without ectasia."),
        ("case_5_od_ectasia_46k", "Carlos Gomez", "Right Eye (OD) Ectasia Scan", "Focal thinning (456 µm), marked posterior BFS elevation (+33 µm), Kmax 46.6 D, and BAD-D 5.34 SD."),
        ("case_6_bilateral_asymmetric", "Elena Rostova", "Bilateral Pentacam Report", "Dual-eye scan comparing right eye ectasia (OD) with completely normal left eye (OS) anatomy.")
    ]

    grid_col1, grid_col2 = st.columns(2)

    for idx, (c_key, p_name, title, desc) in enumerate(demo_cases):
        target_col = grid_col1 if idx % 2 == 0 else grid_col2
        with target_col:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3.5px solid #1829B0; border-radius: 4px; padding: 22px; margin-bottom: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <div style="font-size: 0.76rem; color: #1829B0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">{p_name}</div>
                <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 1.15rem; font-weight: 800;">{title}</h4>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin-bottom: 16px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Analyze {p_name.split()[-1]}'s Scan →", type="primary", key=f"btn_case_{c_key}", width='stretch'):
                st.session_state.selected_sample_key = c_key
                execute_pipeline(mock_case_id=c_key, patient_name_input=p_name)


# ==============================================================================
# PAGE 5: SCAN REPORT & PATIENT GUIDE (STUDIO EDITORIAL STYLE)
# ==============================================================================

elif st.session_state.current_page == "scan":
    pipeline_states = st.session_state.pipeline_states
    
    if not pipeline_states:
        st.session_state.current_page = "home"
        st.rerun()

    # Top Navigation Bar
    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        if st.button("← Back to PentaRead Home", key="back_from_scan"):
            st.session_state.current_page = "home"
            st.rerun()

    with top_col2:
        eye_keys = list(pipeline_states.keys())
        has_both_eyes = ("OD" in pipeline_states and "OS" in pipeline_states) or len(eye_keys) > 1
        if has_both_eyes:
            tab_options = ["OD (Right Eye)", "OS (Left Eye)", "⚖️ Bilateral Comparison"]
            curr_tab = st.session_state.get("active_eye_tab", "BILATERAL")
            curr_idx = 2 if curr_tab == "BILATERAL" else (1 if curr_tab == "OS" else 0)

            if "radio_eye_selector" not in st.session_state or st.session_state.radio_eye_selector not in tab_options:
                st.session_state.radio_eye_selector = tab_options[curr_idx]

            active_tab = st.radio(
                "Active View",
                options=tab_options,
                index=curr_idx,
                horizontal=True,
                label_visibility="collapsed",
                key="radio_eye_selector"
            )
            if "OD" in active_tab:
                st.session_state.active_eye_tab = "OD"
            elif "OS" in active_tab:
                st.session_state.active_eye_tab = "OS"
            else:
                st.session_state.active_eye_tab = "BILATERAL"

    is_bilateral = has_both_eyes and (st.session_state.get("active_eye_tab") == "BILATERAL")

    if is_bilateral:
        # ==============================================================================
        # DEDICATED BILATERAL COMPARISON VIEW (OD vs OS)
        # ==============================================================================
        state_od = pipeline_states.get("OD") or list(pipeline_states.values())[0]
        state_os = pipeline_states.get("OS") or list(pipeline_states.values())[1 if len(pipeline_states) > 1 else 0]
        ext_od = state_od.stage1_extraction
        ext_os = state_os.stage1_extraction
        par_od = state_od.stage2_parietal
        par_os = state_os.stage2_parietal
        front_od = state_od.stage3_frontal
        front_os = state_os.stage3_frontal
        acc_od = state_od.stage4_acc
        acc_os = state_os.stage4_acc
        p_assess_od = par_od.deterministic_assessment
        p_assess_os = par_os.deterministic_assessment

        display_name = (
            st.session_state.get("current_patient_name")
            or getattr(ext_od.metadata, "patient_name", None)
            or getattr(ext_os.metadata, "patient_name", None)
            or "User"
        )

        # 1. Bilateral Greeting Banner
        greet_col1, greet_col2 = st.columns([4, 1])
        with greet_col1:
            st.markdown(f"""
            <div class="scan-greeting-banner">
                <h2 class="greeting-title">Hey {display_name}! 👋</h2>
                <p class="greeting-sub">
                    Here is your <b>Side-by-Side Bilateral Comparison</b> between your 
                    <b>Right Eye (OD)</b> and <b>Left Eye (OS)</b>. Below are both eyes' comparative metrics, 
                    inter-eye asymmetry evaluations, interactive 3D/2D models, everyday physical analogies, and tailored doctor consultation questions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with greet_col2:
            with st.popover("✏️ Edit Name", help="Click to customize your greeting name"):
                edit_key = f"input_name_edit_bilateral"
                new_name = st.text_input("Patient Name", value=display_name, key=edit_key)
                if st.button("Save Name", type="primary", key="btn_save_name_bilateral"):
                    cleaned = new_name.strip()
                    if cleaned:
                        st.session_state.current_patient_name = cleaned
                        st.session_state.patient_name_override = cleaned
                        for s in pipeline_states.values():
                            if s.stage1_extraction.metadata:
                                s.stage1_extraction.metadata.patient_name = cleaned
                        st.rerun()

        # 2. Key Metrics for Both Eyes
        kmax_st_od, kmax_cls_od, kmax_ic_od = get_param_status_info(p_assess_od.parameter_evaluations, 'kmax')
        pachy_st_od, pachy_cls_od, pachy_ic_od = get_param_status_info(p_assess_od.parameter_evaluations, 'thinnest_pachy')
        elev_st_od, elev_cls_od, elev_ic_od = get_param_status_info(p_assess_od.parameter_evaluations, 'posterior_elevation')
        badd_st_od, badd_cls_od, badd_ic_od = get_param_status_info(p_assess_od.parameter_evaluations, 'bad_d')
        tier_od = tier_colors.get(par_od.clinical_risk_tier, ("#1829B0", "Evaluated", "🟢", "status-pill-normal"))

        kmax_st_os, kmax_cls_os, kmax_ic_os = get_param_status_info(p_assess_os.parameter_evaluations, 'kmax')
        pachy_st_os, pachy_cls_os, pachy_ic_os = get_param_status_info(p_assess_os.parameter_evaluations, 'thinnest_pachy')
        elev_st_os, elev_cls_os, elev_ic_os = get_param_status_info(p_assess_os.parameter_evaluations, 'posterior_elevation')
        badd_st_os, badd_cls_os, badd_ic_os = get_param_status_info(p_assess_os.parameter_evaluations, 'bad_d')
        tier_os = tier_colors.get(par_os.clinical_risk_tier, ("#1829B0", "Evaluated", "🟢", "status-pill-normal"))

        asym = compute_asymmetry_stats(ext_od, ext_os)

        st.markdown("""
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Side-by-Side Corneal Tomography Summary
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Comparing primary Scheimpflug diagnostic parameters across both fellow eyes:
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_eye_od, col_eye_os = st.columns(2)
        with col_eye_od:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1.5px solid #1829B0; border-top: 4px solid #1829B0; border-radius: 4px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(24, 41, 176, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span style="font-size: 1.22rem; font-weight: 800; color: #1829B0;">👁️ Right Eye (OD)</span>
                    <span class="status-pill {tier_od[3]}">{tier_od[2]} {tier_od[1]}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Peak Curvature</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_od.keratometry.kmax_diopters or 0.0:.1f} D</div>
                        <span class="status-pill {kmax_cls_od}">{kmax_ic_od} {kmax_st_od}</span>
                        <div class="metric-ref">&lt; 47.0 D</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Thinnest Point</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_od.pachymetry.thinnest_pachymetry_um or 0.0:.0f} µm</div>
                        <span class="status-pill {pachy_cls_od}">{pachy_ic_od} {pachy_st_od}</span>
                        <div class="metric-ref">&gt; 500 µm</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Post. Elevation</div>
                        <div class="metric-val" style="font-size: 1.4rem;">+{ext_od.elevation.posterior_elevation_thinnest_um or 0.0:.1f} µm</div>
                        <span class="status-pill {elev_cls_od}">{elev_ic_od} {elev_st_od}</span>
                        <div class="metric-ref">&lt; +12.0 µm</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">BAD-D Index</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_od.bad_display.final_bad_d_index or 0.0:.2f} SD</div>
                        <span class="status-pill {badd_cls_od}">{badd_ic_od} {badd_st_od}</span>
                        <div class="metric-ref">&lt; 1.60 SD</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_eye_os:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1.5px solid #CBD5E1; border-top: 4px solid #1829B0; border-radius: 4px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span style="font-size: 1.22rem; font-weight: 800; color: #1829B0;">👁️ Left Eye (OS)</span>
                    <span class="status-pill {tier_os[3]}">{tier_os[2]} {tier_os[1]}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Peak Curvature</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_os.keratometry.kmax_diopters or 0.0:.1f} D</div>
                        <span class="status-pill {kmax_cls_os}">{kmax_ic_os} {kmax_st_os}</span>
                        <div class="metric-ref">&lt; 47.0 D</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Thinnest Point</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_os.pachymetry.thinnest_pachymetry_um or 0.0:.0f} µm</div>
                        <span class="status-pill {pachy_cls_os}">{pachy_ic_os} {pachy_st_os}</span>
                        <div class="metric-ref">&gt; 500 µm</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">Post. Elevation</div>
                        <div class="metric-val" style="font-size: 1.4rem;">+{ext_os.elevation.posterior_elevation_thinnest_um or 0.0:.1f} µm</div>
                        <span class="status-pill {elev_cls_os}">{elev_ic_os} {elev_st_os}</span>
                        <div class="metric-ref">&lt; +12.0 µm</div>
                    </div>
                    <div class="metric-card" style="padding: 14px 10px;">
                        <div class="metric-label">BAD-D Index</div>
                        <div class="metric-val" style="font-size: 1.4rem;">{ext_os.bad_display.final_bad_d_index or 0.0:.2f} SD</div>
                        <span class="status-pill {badd_cls_os}">{badd_ic_os} {badd_st_os}</span>
                        <div class="metric-ref">&lt; 1.60 SD</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Inter-Eye Asymmetry Ribbon
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid #1829B0; border-radius: 4px; padding: 18px 24px; margin: 12px 0 20px 0;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #1829B0; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                Inter-Eye Asymmetry Ribbon (OD vs OS Disparity Δ)
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; text-align: center; margin-top: 12px;">
                <div style="background: #F8F9FA; padding: 12px; border-radius: 4px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Δ Peak Curvature</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; color: #1829B0; margin: 4px 0;">{asym['delta_kmax']:.1f} D</div>
                    <span class="status-pill {asym['k_cls']}">{asym['k_icon']} {asym['k_status']}</span>
                    <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 4px;">Normal: &lt; 1.0 D</div>
                </div>
                <div style="background: #F8F9FA; padding: 12px; border-radius: 4px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Δ Thinnest Point</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; color: #1829B0; margin: 4px 0;">{asym['delta_thin']:.0f} µm</div>
                    <span class="status-pill {asym['t_cls']}">{asym['t_icon']} {asym['t_status']}</span>
                    <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 4px;">Normal: &lt; 20 µm</div>
                </div>
                <div style="background: #F8F9FA; padding: 12px; border-radius: 4px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Δ Post. Elevation</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; color: #1829B0; margin: 4px 0;">+{asym['delta_elev']:.1f} µm</div>
                    <span class="status-pill {asym['e_cls']}">{asym['e_icon']} {asym['e_status']}</span>
                    <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 4px;">Normal: &lt; 5.0 µm</div>
                </div>
                <div style="background: #F8F9FA; padding: 12px; border-radius: 4px; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Δ BAD-D Index</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; color: #1829B0; margin: 4px 0;">{asym['delta_badd']:.2f} SD</div>
                    <span class="status-pill {asym['b_cls']}">{asym['b_icon']} {asym['b_status']}</span>
                    <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 4px;">Normal: &lt; 0.50 SD</div>
                </div>
            </div>
            <div style="background: {'#FEF2F2' if asym['has_significant'] else ('#FFFBEB' if asym['has_borderline'] else '#F0FDF4')}; border-left: 4px solid {'#EF4444' if asym['has_significant'] else ('#F59E0B' if asym['has_borderline'] else '#10B981')}; border-radius: 4px; padding: 12px 16px; margin-top: 14px; font-size: 0.88rem; color: #0F172A;">
                {"⚠️ <b>Notable Inter-Eye Asymmetry Detected:</b> One eye exhibits localized steepening or thinning relative to its fellow eye. In corneal tomography, fellow-eye disparity is a primary sensitive marker for unilateral or early asymmetric ectasia." if asym['has_significant'] else ("🟡 <b>Mild Inter-Eye Asymmetry:</b> Slight differences noted between fellow eyes, warranting standard longitudinal surveillance." if asym['has_borderline'] else "🟢 <b>Symmetric Fellow Corneas:</b> Both eyes exhibit closely balanced structural curvature, thickness, and elevation profiles.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Three Popups: Real-Life Examples (Bilateral Dialogs)
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Everyday Real-Life Models to Understand Your Corneas
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Click any model below to explore bilateral comparisons between both your eyes:
            </p>
        </div>
        """, unsafe_allow_html=True)

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-navy-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                        Model 01 • Curvature
                    </div>
                    <h3 style="color: #FFFFFF; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Watch Glass Window
                    </h3>
                    <p style="color: #E0E7FF; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        Comparing watch dome steepness across OD and OS to evaluate optical focusing symmetry.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Watch Model →", key="btn_open_watch_dialog_bi", width='stretch'):
                    dialog_watch_glass(
                        ext_od.keratometry.kmax_diopters or 44.0,
                        fellow_val=ext_os.keratometry.kmax_diopters or 44.0,
                        labels=("Right Eye (OD)", "Left Eye (OS)")
                    )

        with p_col2:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-white-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #1829B0; font-weight: 700; margin-bottom: 6px;">
                        Model 02 • Thickness
                    </div>
                    <h3 style="color: #0F172A; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Stack of Paper Sheets
                    </h3>
                    <p style="color: #475569; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        Comparing paper sheet layers (µm) between OD and OS to pinpoint thinning disparity.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Paper Model →", key="btn_open_paper_dialog_bi", width='stretch'):
                    dialog_paper_sheets(
                        ext_od.pachymetry.thinnest_pachymetry_um or 540.0,
                        fellow_val=ext_os.pachymetry.thinnest_pachymetry_um or 540.0,
                        labels=("Right Eye (OD)", "Left Eye (OS)")
                    )

        with p_col3:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-navy-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                        Model 03 • Elevation
                    </div>
                    <h3 style="color: #FFFFFF; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Mountain Contour Slope
                    </h3>
                    <p style="color: #E0E7FF; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        Hikers' contour elevation maps across both corneas to catch hidden back bulges.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Contour Model →", key="btn_open_mountain_dialog_bi", width='stretch'):
                    dialog_mountain_contours(
                        ext_od.elevation.posterior_elevation_thinnest_um or 8.0,
                        fellow_val=ext_os.elevation.posterior_elevation_thinnest_um or 8.0,
                        labels=("Right Eye (OD)", "Left Eye (OS)")
                    )

        st.markdown("<br><hr style='border-color: #E2E8F0; margin: 20px 0 28px 0;'><br>", unsafe_allow_html=True)

        # 4. Interactive 3D & 2D Models (Side-by-Side OD and OS)
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Side-by-Side 3D & 2D Corneal Models (OD vs OS)
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Inspect and rotate 3D meshes and 2D heatmaps simultaneously for Right Eye (OD) and Left Eye (OS):
            </p>
        </div>
        """, unsafe_allow_html=True)

        ctrl_col1, ctrl_col2 = st.columns([2.5, 1.5])
        with ctrl_col1:
            metric_selection_bi = st.radio(
                "Select Diagnostic Metric Model",
                options=["Curvature (Kmax)", "Corneal Thickness", "Posterior Elevation", "Belin-Ambrósio BAD-D"],
                horizontal=True,
                label_visibility="collapsed",
                key="radio_metric_models_bi"
            )
        with ctrl_col2:
            ref_toggle_bi = st.toggle("Compare with Healthy Reference Cornea", value=False, key="toggle_ref_compare_bi")

        mod_col_od, mod_col_os = st.columns(2)

        if "Curvature" in metric_selection_bi:
            with mod_col_od:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Right Eye (OD) Curvature</div>", unsafe_allow_html=True)
                fig3d_od = create_3d_curvature_figure(ext_od, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_od, width='stretch', key="plot_bi_3d_curv_od")
                fig2d_od = create_2d_topography_heatmap(ext_od, reference_mode=ref_toggle_bi, eye_label="Right Eye (OD)")
                st.plotly_chart(fig2d_od, width='stretch', key="plot_bi_2d_curv_od")
            with mod_col_os:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Left Eye (OS) Curvature</div>", unsafe_allow_html=True)
                fig3d_os = create_3d_curvature_figure(ext_os, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_os, width='stretch', key="plot_bi_3d_curv_os")
                fig2d_os = create_2d_topography_heatmap(ext_os, reference_mode=ref_toggle_bi, eye_label="Left Eye (OS)")
                st.plotly_chart(fig2d_os, width='stretch', key="plot_bi_2d_curv_os")

        elif "Thickness" in metric_selection_bi:
            with mod_col_od:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Right Eye (OD) Thickness</div>", unsafe_allow_html=True)
                fig3d_od = create_3d_pachymetry_figure(ext_od, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_od, width='stretch', key="plot_bi_3d_pachy_od")
                fig2d_od = create_2d_pachymetry_heatmap(ext_od, reference_mode=ref_toggle_bi, eye_label="Right Eye (OD)")
                st.plotly_chart(fig2d_od, width='stretch', key="plot_bi_2d_pachy_od")
            with mod_col_os:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Left Eye (OS) Thickness</div>", unsafe_allow_html=True)
                fig3d_os = create_3d_pachymetry_figure(ext_os, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_os, width='stretch', key="plot_bi_3d_pachy_os")
                fig2d_os = create_2d_pachymetry_heatmap(ext_os, reference_mode=ref_toggle_bi, eye_label="Left Eye (OS)")
                st.plotly_chart(fig2d_os, width='stretch', key="plot_bi_2d_pachy_os")

        elif "Elevation" in metric_selection_bi:
            with mod_col_od:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Right Eye (OD) Elevation</div>", unsafe_allow_html=True)
                fig3d_od = create_3d_elevation_figure(ext_od, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_od, width='stretch', key="plot_bi_3d_elev_od")
                fig2d_od = create_2d_elevation_heatmap(ext_od, reference_mode=ref_toggle_bi, eye_label="Right Eye (OD)")
                st.plotly_chart(fig2d_od, width='stretch', key="plot_bi_2d_elev_od")
            with mod_col_os:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Left Eye (OS) Elevation</div>", unsafe_allow_html=True)
                fig3d_os = create_3d_elevation_figure(ext_os, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_os, width='stretch', key="plot_bi_3d_elev_os")
                fig2d_os = create_2d_elevation_heatmap(ext_os, reference_mode=ref_toggle_bi, eye_label="Left Eye (OS)")
                st.plotly_chart(fig2d_os, width='stretch', key="plot_bi_2d_elev_os")

        else:  # BAD-D
            with mod_col_od:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Right Eye (OD) BAD-D</div>", unsafe_allow_html=True)
                fig3d_od = create_3d_badd_figure(ext_od, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_od, width='stretch', key="plot_bi_3d_badd_od")
                fig_g_od = create_bad_d_gauge_figure(ext_od.bad_display.final_bad_d_index, reference_mode=ref_toggle_bi, eye_label="Right Eye (OD)")
                st.plotly_chart(fig_g_od, width='stretch', key="plot_bi_gauge_od")
            with mod_col_os:
                st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #1829B0; margin-bottom: 8px;'>👁️ Left Eye (OS) BAD-D</div>", unsafe_allow_html=True)
                fig3d_os = create_3d_badd_figure(ext_os, reference_mode=ref_toggle_bi)
                st.plotly_chart(fig3d_os, width='stretch', key="plot_bi_3d_badd_os")
                fig_g_os = create_bad_d_gauge_figure(ext_os.bad_display.final_bad_d_index, reference_mode=ref_toggle_bi, eye_label="Left Eye (OS)")
                st.plotly_chart(fig_g_os, width='stretch', key="plot_bi_gauge_os")

        st.markdown("<br><hr style='border-color: #E2E8F0; margin: 20px 0 28px 0;'><br>", unsafe_allow_html=True)

        # 5. Dynamic LLM Explanations for Both Eyes
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Understanding Your Corneal Health Profile (OD vs OS)
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Calibrated clinical summaries and structural analysis comparing both fellow eyes:
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_sum_od, col_sum_os = st.columns(2)
        with col_sum_od:
            st.markdown(f"""
            <div style="background-color: #1829B0; color: #FFFFFF; border-radius: 4px; padding: 24px 26px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(24, 41, 176, 0.12);">
                <div style="font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                    OD (Right Eye) • Executive Summary
                </div>
                <p style="margin: 0; font-size: 0.98rem; line-height: 1.65; color: #FFFFFF;">
                    {acc_od.executive_summary}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OD Shape &amp; Curvature</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_od.corneal_shape_section}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OD Tissue Thickness</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_od.corneal_thickness_section}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OD Elevation &amp; Stability</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_od.back_surface_section}
                </p>
            </div>
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OD Vision Impact</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_od.optical_impact_explanation}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_sum_os:
            st.markdown(f"""
            <div style="background-color: #0F172A; color: #FFFFFF; border-radius: 4px; padding: 24px 26px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.12);">
                <div style="font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94A3B8; font-weight: 700; margin-bottom: 6px;">
                    OS (Left Eye) • Executive Summary
                </div>
                <p style="margin: 0; font-size: 0.98rem; line-height: 1.65; color: #FFFFFF;">
                    {acc_os.executive_summary}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OS Shape &amp; Curvature</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_os.corneal_shape_section}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OS Tissue Thickness</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_os.corneal_thickness_section}
                </p>
            </div>
            <div class="issue-card" style="margin-bottom: 12px;">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OS Elevation &amp; Stability</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_os.back_surface_section}
                </p>
            </div>
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 6px 0; font-size: 1.05rem; font-weight: 800;">OS Vision Impact</h5>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin: 0;">
                    {front_os.optical_impact_explanation}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Bilateral Tailored Questions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color: #1829B0; font-size: 1.35rem; font-weight: 800; margin-bottom: 6px;">
            🩺 Tailored Questions to Ask Your Eye Doctor (Bilateral Focus)
        </h3>
        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 16px;">
            Bring these targeted questions to discuss fellow-eye risk, monitoring intervals, and treatment:
        </p>
        """, unsafe_allow_html=True)

        # Asymmetry Question Card
        if asym['has_significant'] or asym['has_borderline']:
            st.markdown(f"""
            <div class="question-card" style="border-left: 4px solid #EF4444;">
                <div style="font-size: 0.74rem; font-weight: 700; color: #DC2626; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                    FELLOW-EYE ASYMMETRY &amp; PROGRESSION
                </div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                    "Given the structural difference between my right eye (Kmax {asym['kmax_od']:.1f} D, {asym['thin_od']:.0f} µm) and left eye (Kmax {asym['kmax_os']:.1f} D, {asym['thin_os']:.0f} µm), what is the risk of progression in my fellow eye, and what monitoring schedule do you recommend?"
                </div>
                <div style="font-size: 0.88rem; color: #64748B; font-style: italic;">
                    Why ask this: Even in clinically unilateral presentations, keratoconus is almost always biologically bilateral over time. Close monitoring of the fellow eye catches early subclinical changes before vision changes occur.
                </div>
            </div>
            """, unsafe_allow_html=True)

        for q in (acc_od.consultation_questions[:2] + acc_os.consultation_questions[:2]):
            st.markdown(f"""
            <div class="question-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #1829B0; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                    {q.category}
                </div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                    "{q.question_text}"
                </div>
                <div style="font-size: 0.88rem; color: #64748B; font-style: italic;">
                    Why ask this: {q.rationale}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Medical Disclaimer
        st.markdown(f"""
        <div style="background: #F8F9FA; border: 1px dashed #CBD5E1; border-radius: 4px; padding: 16px; margin-top: 24px; font-size: 0.82rem; color: #64748B; text-align: center;">
            {acc_od.medical_disclaimer}
        </div>
        """, unsafe_allow_html=True)

        # Bilateral Download Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        bi_report_md = f"""# PentaRead Bilateral Patient Education Report: {display_name}
Date: {ext_od.metadata.exam_date or ext_os.metadata.exam_date or 'Recent'}

## Bilateral Tomography Summary
| Parameter | Right Eye (OD) | Left Eye (OS) | Inter-Eye Disparity (Δ) | Clinical Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Peak Curvature (Kmax)** | {asym['kmax_od']:.1f} D ({kmax_st_od}) | {asym['kmax_os']:.1f} D ({kmax_st_os}) | {asym['delta_kmax']:.1f} D ({asym['k_status']}) | &lt; 47.0 D (Δ &lt; 1.0 D) |
| **Thinnest Pachymetry** | {asym['thin_od']:.0f} µm ({pachy_st_od}) | {asym['thin_os']:.0f} µm ({pachy_st_os}) | {asym['delta_thin']:.0f} µm ({asym['t_status']}) | &gt; 500 µm (Δ &lt; 20 µm) |
| **Posterior Elevation** | +{asym['elev_od']:.1f} µm ({elev_st_od}) | +{asym['elev_os']:.1f} µm ({elev_st_os}) | {asym['delta_elev']:.1f} µm ({asym['e_status']}) | &lt; +12.0 µm (Δ &lt; 5.0 µm) |
| **Belin-Ambrósio BAD-D** | {asym['badd_od']:.2f} SD ({badd_st_od}) | {asym['badd_os']:.2f} SD ({badd_st_os}) | {asym['delta_badd']:.2f} SD ({asym['b_status']}) | &lt; 1.60 SD (Δ &lt; 0.50 SD) |
| **Overall Structural Tier** | {tier_od[1]} | {tier_os[1]} | - | Physiologic Baseline |

## Executive Clinical Summaries
### Right Eye (OD)
{acc_od.executive_summary}

### Left Eye (OS)
{acc_os.executive_summary}

## Structural Analysis
### OD (Right Eye)
- Curvature & Shape: {front_od.corneal_shape_section}
- Tissue Thickness: {front_od.corneal_thickness_section}
- Back Surface Elevation: {front_od.back_surface_section}
- Vision Impact: {front_od.optical_impact_explanation}

### OS (Left Eye)
- Curvature & Shape: {front_os.corneal_shape_section}
- Tissue Thickness: {front_os.corneal_thickness_section}
- Back Surface Elevation: {front_os.back_surface_section}
- Vision Impact: {front_os.optical_impact_explanation}

## Bilateral Questions for Your Doctor
- [Fellow-Eye Asymmetry] Given the structural difference between my right eye (Kmax {asym['kmax_od']:.1f} D, {asym['thin_od']:.0f} µm) and left eye (Kmax {asym['kmax_os']:.1f} D, {asym['thin_os']:.0f} µm), what is the risk of progression in my fellow eye and what monitoring schedule do you recommend?
""" + "\n".join([f"- [{q.category}] {q.question_text} (Why: {q.rationale})" for q in (acc_od.consultation_questions[:2] + acc_os.consultation_questions[:2])]) + f"\n\n---\n{acc_od.medical_disclaimer}"

        st.download_button(
            label="📥 Download Full Bilateral Report (.md)",
            data=bi_report_md,
            file_name=f"PentaRead_Bilateral_Report_{display_name.replace(' ', '_')}.md",
            mime="text/markdown",
            key="btn_download_guide_bi"
        )

        # Developer Drawer
        if dev_mode:
            st.markdown("<br><hr style='border-color: #E2E8F0;'><br>", unsafe_allow_html=True)
            with st.expander("🧠 P-FIT Multi-Stage Architecture & Guardrails Inspector (Developer Mode - Bilateral)", expanded=False):
                st.markdown("##### Bilateral Pipeline Multi-Eye Inspection")
                b_dev1, b_dev2 = st.columns(2)
                with b_dev1:
                    st.markdown("###### Right Eye (OD) Extraction & Assessment")
                    st.json(ext_od.model_dump())
                    st.json(p_assess_od.model_dump())
                with b_dev2:
                    st.markdown("###### Left Eye (OS) Extraction & Assessment")
                    st.json(ext_os.model_dump())
                    st.json(p_assess_os.model_dump())

    else:
        # ==============================================================================
        # SINGLE EYE VIEW (OD or OS)
        # ==============================================================================
        active_eye_key = "OS" if st.session_state.active_eye_tab == "OS" else "OD"
        if active_eye_key not in pipeline_states:
            active_eye_key = list(pipeline_states.keys())[0]

        state: CognitiveScaffoldingState = pipeline_states[active_eye_key]
        extraction = state.stage1_extraction
        parietal = state.stage2_parietal
        frontal = state.stage3_frontal
        acc_report = state.stage4_acc
        p_assess = parietal.deterministic_assessment

        # Determine Patient Name (Dynamic & Synchronized across eyes)
        display_name = (
            st.session_state.get("current_patient_name")
            or getattr(extraction.metadata, "patient_name", None)
            or "User"
        )

        # ================= 1. PERSONALIZED PATIENT GREETING =================
        greet_col1, greet_col2 = st.columns([4, 1])
        with greet_col1:
            st.markdown(f"""
            <div class="scan-greeting-banner">
                <h2 class="greeting-title">Hey {display_name}! 👋</h2>
                <p class="greeting-sub">
                    Here is your clear, visual breakdown of your corneal tomography scan for your 
                    <b>{extraction.metadata.eye or 'Eye'}</b>. Below are your primary structural measurements, 
                    everyday physical models, interactive 3D/2D maps, and tailored questions to discuss with your doctor.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with greet_col2:
            with st.popover("✏️ Edit Name", help="Click to customize your greeting name"):
                edit_key = f"input_name_edit_{state.scan_id}"
                new_name = st.text_input("Patient Name", value=display_name, key=edit_key)
                if st.button("Save Name", type="primary", key=f"btn_save_name_{state.scan_id}"):
                    cleaned = new_name.strip()
                    if cleaned:
                        st.session_state.current_patient_name = cleaned
                        st.session_state.patient_name_override = cleaned
                        for s in pipeline_states.values():
                            if s.stage1_extraction.metadata:
                                s.stage1_extraction.metadata.patient_name = cleaned
                        st.rerun()

        # ================= 2. METRICS SECTION =================
        kmax_status, kmax_class, kmax_icon = get_param_status_info(p_assess.parameter_evaluations, 'kmax')
        pachy_status, pachy_class, pachy_icon = get_param_status_info(p_assess.parameter_evaluations, 'thinnest_pachy')
        elev_status, elev_class, elev_icon = get_param_status_info(p_assess.parameter_evaluations, 'posterior_elevation')
        badd_status, badd_class, badd_icon = get_param_status_info(p_assess.parameter_evaluations, 'bad_d')
        tier_info = tier_colors.get(parietal.clinical_risk_tier, ("#1829B0", "Evaluated", "🟢", "status-pill-normal"))

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Curvature (Kmax)</div>
                <div class="metric-val">{extraction.keratometry.kmax_diopters or 0.0:.1f} D</div>
                <span class="status-pill {kmax_class}">{kmax_icon} {kmax_status}</span>
                <div class="metric-ref">Ref: &lt; 47.0 Diopters</div>
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
            <div class="metric-card" style="border-top-color: {tier_info[0]};">
                <div class="metric-label">Structural Status</div>
                <div class="metric-val" style="font-size: 1.15rem; color: {tier_info[0]}; margin-top: 4px; line-height: 1.3;">
                    {tier_info[2]} {tier_info[1]}
                </div>
                <div class="metric-ref" style="margin-top: 8px;">Consensus Cutoffs</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= 3. THREE POPUPS: REAL-LIFE EXAMPLES (STUDIO BLOCK STYLE) =================
        st.markdown("""
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Everyday Real-Life Models to Understand Your Cornea
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Click on any of the 3 real-life models below to open an interactive modal explaining your eye's measurements:
            </p>
        </div>
        """, unsafe_allow_html=True)

        p_col1, p_col2, p_col3 = st.columns(3)

        # Card 1: Solid Royal Navy Studio Block
        with p_col1:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-navy-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                        Model 01 • Curvature
                    </div>
                    <h3 style="color: #FFFFFF; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Watch Glass Window
                    </h3>
                    <p style="color: #E0E7FF; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        How a watch crystal's smooth dome focuses light and what happens when curvature steepens.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Watch Model →", key="btn_open_watch_dialog", width='stretch'):
                    dialog_watch_glass(extraction.keratometry.kmax_diopters or 44.0)

        # Card 2: Crisp White Studio Block
        with p_col2:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-white-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #1829B0; font-weight: 700; margin-bottom: 6px;">
                        Model 02 • Thickness
                    </div>
                    <h3 style="color: #0F172A; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Stack of Paper Sheets
                    </h3>
                    <p style="color: #475569; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        Understanding 540 microscopic paper sheets (1 µm = 1/1,000th mm) and localized thinning.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Paper Model →", key="btn_open_paper_dialog", width='stretch'):
                    dialog_paper_sheets(extraction.pachymetry.thinnest_pachymetry_um or 540.0)

        # Card 3: Solid Royal Navy Studio Block
        with p_col3:
            with st.container(border=True):
                st.markdown("""
                <div class="studio-navy-card" style="padding: 8px 4px 6px 4px;">
                    <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                        Model 03 • Elevation
                    </div>
                    <h3 style="color: #FFFFFF; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;">
                        The Mountain Contour Slope
                    </h3>
                    <p style="color: #E0E7FF; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;">
                        How hikers read topographic contour lines to catch hidden early bulges on the back surface.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore Contour Model →", key="btn_open_mountain_dialog", width='stretch'):
                    dialog_mountain_contours(extraction.elevation.posterior_elevation_thinnest_um or 8.0)

        st.markdown("<br><hr style='border-color: #E2E8F0; margin: 20px 0 28px 0;'><br>", unsafe_allow_html=True)

        # ================= 4. INTERACTIVE 3D & 2D MODELS FOR EACH METRIC =================
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Interactive 3D & 2D Corneal Models
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                Select any diagnostic metric below to inspect its dedicated 3D surface mesh and 2D tomographic heatmap:
            </p>
        </div>
        """, unsafe_allow_html=True)

        ctrl_col1, ctrl_col2 = st.columns([2.5, 1.5])
        with ctrl_col1:
            metric_selection = st.radio(
                "Select Diagnostic Metric Model",
                options=["Curvature (Kmax)", "Corneal Thickness", "Posterior Elevation", "Belin-Ambrósio BAD-D"],
                horizontal=True,
                label_visibility="collapsed",
                key="radio_metric_models"
            )
        with ctrl_col2:
            ref_toggle = st.toggle("Compare with Healthy Reference Cornea", value=False, key="toggle_ref_compare")

        # Render dedicated 3D and 2D models
        v_col3d, v_col2d = st.columns([1.15, 0.85])

        if "Curvature" in metric_selection:
            with v_col3d:
                fig3d = create_3d_curvature_figure(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig3d, width='stretch', key="plot_3d_curv")
            with v_col2d:
                fig2d = create_2d_topography_heatmap(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig2d, width='stretch', key="plot_2d_curv")
            st.markdown(f"""
            <div style="background: #F8F9FA; border-left: 4px solid #1829B0; border-radius: 4px; padding: 12px 18px; font-size: 0.88rem; color: #334155; margin-top: 8px;">
                💡 <b>Understanding Curvature:</b> Warm colors (yellow/red) denote steeper optical power in Diopters. 
                The white crosshair marks your Kmax apex ({extraction.keratometry.kmax_diopters or 44.0:.1f} D). Click and drag the 3D surface to rotate!
            </div>
            """, unsafe_allow_html=True)

        elif "Thickness" in metric_selection:
            with v_col3d:
                fig3d = create_3d_pachymetry_figure(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig3d, width='stretch', key="plot_3d_pachy")
            with v_col2d:
                fig2d = create_2d_pachymetry_heatmap(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig2d, width='stretch', key="plot_2d_pachy")
            st.markdown(f"""
            <div style="background: #F8F9FA; border-left: 4px solid #1829B0; border-radius: 4px; padding: 12px 18px; font-size: 0.88rem; color: #334155; margin-top: 8px;">
                💡 <b>Understanding Thickness:</b> Shows thickness across the cornea in micrometers (µm). 
                The red star marks your thinnest structural point ({extraction.pachymetry.thinnest_pachymetry_um or 540.0:.0f} µm).
            </div>
            """, unsafe_allow_html=True)

        elif "Elevation" in metric_selection:
            with v_col3d:
                fig3d = create_3d_elevation_figure(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig3d, width='stretch', key="plot_3d_elev")
            with v_col2d:
                fig2d = create_2d_elevation_heatmap(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig2d, width='stretch', key="plot_2d_elev")
            st.markdown(f"""
            <div style="background: #F8F9FA; border-left: 4px solid #1829B0; border-radius: 4px; padding: 12px 18px; font-size: 0.88rem; color: #334155; margin-top: 8px;">
                💡 <b>Understanding Elevation:</b> Measures height in µm relative to the Best Fit Sphere. 
                Warm islands indicate forward bulging (+{extraction.elevation.posterior_elevation_thinnest_um or 8.0:.1f} µm) on the back surface.
            </div>
            """, unsafe_allow_html=True)

        else:  # BAD-D
            with v_col3d:
                fig3d = create_3d_badd_figure(extraction, reference_mode=ref_toggle)
                st.plotly_chart(fig3d, width='stretch', key="plot_3d_badd")
            with v_col2d:
                fig_gauge = create_bad_d_gauge_figure(extraction.bad_display.final_bad_d_index, reference_mode=ref_toggle)
                st.plotly_chart(fig_gauge, width='stretch', key="plot_gauge_badd")
            st.markdown("""
            <div style="background: #F8F9FA; border-left: 4px solid #1829B0; border-radius: 4px; padding: 12px 18px; font-size: 0.88rem; color: #334155; margin-top: 8px;">
                💡 <b>Understanding BAD-D:</b> 🟢 Green (&lt; 1.60 SD) = Normal variation; 
                🟡 Yellow (1.60 - 2.60 SD) = Borderline alert; 🔴 Red (&gt; 2.60 SD) = Pathological deviation from healthy populations.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><hr style='border-color: #E2E8F0; margin: 20px 0 28px 0;'><br>", unsafe_allow_html=True)

        # ================= 5. DYNAMIC LLM EXPLANATION OF PATIENT ISSUES =================
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0 0 4px 0; font-size: 1.45rem; font-weight: 800; color: #1829B0; letter-spacing: -0.01em;">
                Understanding Your Corneal Health Profile
            </h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
                A calm, calibrated explanation of your eye's structural measurements and vision impact generated dynamically:
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Executive Overview Box (Solid Royal Navy Block)
        st.markdown(f"""
        <div style="background-color: #1829B0; color: #FFFFFF; border-radius: 4px; padding: 28px 32px; margin-bottom: 24px; box-shadow: 0 4px 16px rgba(24, 41, 176, 0.12);">
            <div style="font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: #E0E7FF; font-weight: 700; margin-bottom: 6px;">
                Executive Clinical Overview
            </div>
            <p style="margin: 0; font-size: 1.05rem; line-height: 1.7; color: #FFFFFF;">
                {acc_report.executive_summary}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Structural Issues Breakdown
        st.markdown("#### 🔍 Structural Features of Your Cornea")
        
        i_col1, i_col2 = st.columns([1.1, 0.9])
        with i_col1:
            st.markdown(f"""
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 8px 0; font-size: 1.08rem; font-weight: 800;">Shape & Curvature Analysis</h5>
                <p style="font-size: 0.94rem; color: #334155; line-height: 1.65; margin: 0;">
                    {frontal.corneal_shape_section}
                </p>
            </div>
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 8px 0; font-size: 1.08rem; font-weight: 800;">Tissue Thickness & Distribution</h5>
                <p style="font-size: 0.94rem; color: #334155; line-height: 1.65; margin: 0;">
                    {frontal.corneal_thickness_section}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with i_col2:
            st.markdown(f"""
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 8px 0; font-size: 1.08rem; font-weight: 800;">Back-Surface Elevation & Stability</h5>
                <p style="font-size: 0.94rem; color: #334155; line-height: 1.65; margin: 0;">
                    {frontal.back_surface_section}
                </p>
            </div>
            <div class="issue-card">
                <h5 style="color: #1829B0; margin: 0 0 8px 0; font-size: 1.08rem; font-weight: 800;">Optical Impact on Your Daily Vision</h5>
                <p style="font-size: 0.94rem; color: #334155; line-height: 1.65; margin: 0;">
                    {frontal.optical_impact_explanation}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Doctor Consultation Questions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style="color: #1829B0; font-size: 1.35rem; font-weight: 800; margin-bottom: 6px;">
            🩺 Tailored Questions to Ask Your Eye Doctor
        </h3>
        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 16px;">
            Bring these personalized questions to your next appointment to discuss your vision plan:
        </p>
        """, unsafe_allow_html=True)

        for q in acc_report.consultation_questions:
            st.markdown(f"""
            <div class="question-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: #1829B0; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                    {q.category}
                </div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                    "{q.question_text}"
                </div>
                <div style="font-size: 0.88rem; color: #64748B; font-style: italic;">
                    Why ask this: {q.rationale}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Medical Disclaimer
        st.markdown(f"""
        <div style="background: #F8F9FA; border: 1px dashed #CBD5E1; border-radius: 4px; padding: 16px; margin-top: 24px; font-size: 0.82rem; color: #64748B; text-align: center;">
            {acc_report.medical_disclaimer}
        </div>
        """, unsafe_allow_html=True)

        # Download Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        report_download_content = f"""# PentaRead Patient Education Report: {display_name}
Examined Eye: {extraction.metadata.eye or 'Unknown'}
Date: {extraction.metadata.exam_date or 'Recent'}

## Executive Summary
{acc_report.executive_summary}

## Key Tomography Measurements
- Peak Curvature (Kmax): {extraction.keratometry.kmax_diopters or 0.0:.1f} D
- Thinnest Pachymetry: {extraction.pachymetry.thinnest_pachymetry_um or 0.0:.0f} µm
- Posterior Elevation: +{extraction.elevation.posterior_elevation_thinnest_um or 0.0:.1f} µm
- BAD-D Final Index: {extraction.bad_display.final_bad_d_index or 0.0:.2f} SD

## Structural Analysis
### Curvature & Shape
{frontal.corneal_shape_section}

### Tissue Thickness
{frontal.corneal_thickness_section}

### Back Surface Elevation
{frontal.back_surface_section}

### Optical Vision Impact
{frontal.optical_impact_explanation}

## Tailored Questions for Your Eye Doctor
""" + "\n".join([f"- [{q.category}] {q.question_text} (Why: {q.rationale})" for q in acc_report.consultation_questions]) + f"\n\n---\n{acc_report.medical_disclaimer}"

        st.download_button(
            label="📥 Download Full Patient Guide (.md)",
            data=report_download_content,
            file_name=f"PentaRead_Report_{display_name.replace(' ', '_')}.md",
            mime="text/markdown",
            key="btn_download_guide"
        )

        # ================= 6. DEVELOPER ONLY DRAWER (P-FIT Architecture Inspector) =================
        if dev_mode:
            st.markdown("<br><hr style='border-color: #E2E8F0;'><br>", unsafe_allow_html=True)
            with st.expander("🧠 P-FIT Multi-Stage Architecture & Guardrails Inspector (Developer Mode)", expanded=False):
                st.markdown("""
                **P-FIT Cognitive Scaffolding Pipeline (Jung & Haier 2007 external cognitive prosthesis):**
                - Stage 1: Occipital-Temporal Proxy (Multimodal feature extraction & OCR)
                - Stage 2: Parietal Cortex Proxy (Structural abstraction & deterministic consensus guardrails)
                - Stage 3: Frontal / Prefrontal Proxy (Physical scaffolding & everyday analogies)
                - Stage 4: Anterior Cingulate Cortex / ACC Proxy (Inhibitory gating & tone calibration)
                """)
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("##### Stage 1 Raw Extraction")
                    st.json(extraction.model_dump())
                with d2:
                    st.markdown("##### Stage 2 Deterministic Guardrails Assessment")
                    st.json(p_assess.model_dump())
                    st.markdown("##### Stage 4 Inhibitory Gating Audit")
                    st.json(acc_report.inhibitory_gating_audit.model_dump())

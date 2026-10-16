"""
P-FIT Corneal Tomography Scaffolding Pipeline Configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Default Active Gemini Models
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-3.8-flash")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Clinical & Cognitive Constants
P_FIT_STAGES = [
    "Stage 1: Occipital-Temporal Proxy (Multimodal Ingestion & OCR)",
    "Stage 2: Parietal Cortex Proxy (Structural Abstraction & Guardrails)",
    "Stage 3: Frontal/Prefrontal Proxy (Cognitive Scaffolding & Analogies)",
    "Stage 4: ACC Proxy (Cognitive Control, Inhibitory Gating & Doctor Q&A)"
]

# P-FIT Corneal Tomography (Pentacam) Cognitive Scaffolding Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Google GenAI SDK](https://img.shields.io/badge/SDK-google--genai-green.svg)](https://ai.google.dev/)
[![Pydantic v2](https://img.shields.io/badge/typing-pydantic--v2-red.svg)](https://docs.pydantic.dev/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A multi-stage cognitive-scaffolding pipeline that translates complex Scheimpflug corneal tomography (**Oculus Pentacam**) diagnostic scans into clear, calibrated, and anxiety-free explanations for lay patients.

---

## 🧠 Theoretical Foundation: P-FIT (Jung & Haier, 2007)

General human intelligence relies on the **Parieto-Frontal Integration Theory (P-FIT)**. Lay patients lack domain-specific parietal schemas to synthesize multi-parametric corneal geometry (color-coded curvature maps, pachymetry distributions, and elevation profiles), leading to severe prefrontal cognitive overload and anxiety when receiving complex diagnostic printouts.

This system acts as an **externalized cognitive prosthesis** implementing P-FIT across 4 strictly decoupled stages:

```mermaid
graph LR
    subgraph S1["Stage 1: Occipital-Temporal Proxy"]
        V[Pentacam Scan / PDF] --> OCR[Gemini Vision OCR]
        OCR --> P1[Pydantic v2 JSON Schema]
    end

    subgraph S2["Stage 2: Parietal Cortex Proxy"]
        P1 --> DG[Deterministic Guardrails Engine]
        DG --> SI[Spatial Integration & Coupling]
        SI --> P2[Grounded Geometric Profile]
    end

    subgraph S3["Stage 3: Frontal / Prefrontal Proxy"]
        P2 --> CS[Cognitive Scaffolding Engine]
        CS --> AM[Physical Metaphors & Analogies]
        AM --> P3[Tiered Patient Explanation Draft]
    end

    subgraph S4["Stage 4: ACC Proxy"]
        P3 --> IG[Inhibitory Gating & Tone Softening]
        IG --> FV[Factual Alignment Verification]
        FV --> DQ[Doctor Consultation Q&A Generator]
        DQ --> OUT[Final Calibrated Patient Report]
    end
```

---

## 🔬 The 4-Stage Architecture

### 1. Stage 1: Occipital-Temporal Proxy (Visual Feature Extraction & OCR)
- **Biological Role (Brodmann Areas 17, 18, 19, 37):** Visual sensation, edge detection, tokenization, and optical character recognition.
- **Pipeline Implementation:** Multimodal ingestion of Pentacam 4-Maps Refractive, BAD-III Displays, and Topometric scans via Gemini Multimodal Vision into strictly typed Pydantic v2 data contracts.
- **Extracted Indices:**
  - **Keratometry:** $K_1$, $K_2$, $K_m$, $K_{max}$, Astigmatism, Axis, $Q$-value asphericity, $R_{min}$.
  - **Pachymetry:** Central thickness (CCT), Thinnest Point ($\mu\text{m}$), Thinnest point $(X, Y)$ displacement (mm), $PPI_{avg/max}$, $ART_{max}$.
  - **Elevation:** Anterior and Posterior elevation at thinnest point relative to BFS/BFTE ($\mu\text{m}$), Enhanced elevation diff.
  - **Belin-Ambrósio Display:** $D_f, D_b, D_t, D_a, D_p$, and Final $BAD\text{-}D$ standard deviation index.
  - **Topometric:** $ISV, IVA, KI, CKI, IHD$.

### 2. Stage 2: Parietal Cortex Proxy (Structural Abstraction & Deterministic Guardrails)
- **Biological Role (Brodmann Areas 7, 39, 40):** Cross-modal integration, spatial manipulation, and geometric mental modeling.
- **Pipeline Implementation:** Combines spatial co-localization (e.g. verifying if the thinnest pachymetric point aligns with the peak anterior/posterior protrusion) with **pure Python deterministic boundary checks** against clinical cutoffs from the *Global Consensus on Keratoconus and Ectatic Diseases*.
- **Empirical Boundary Rules:**
  - $K_{max} > 48.5\text{ D}$ $\rightarrow$ Abnormal / Focal Cone.
  - $\text{Thinnest Pachymetry} < 470\ \mu\text{m}$ $\rightarrow$ Abnormal Thinning.
  - $\text{Posterior Elevation} \ge +16.0\ \mu\text{m}$ $\rightarrow$ Pathological Posterior Protrusion.
  - $BAD\text{-}D \ge 2.60\text{ SD}$ $\rightarrow$ Pathological Red Zone.

### 3. Stage 3: Frontal / Prefrontal Proxy (Cognitive Scaffolding & Analogies)
- **Biological Role (Brodmann Areas 9, 10, 45, 46):** Hypothesis formation, working memory reduction, and analogical reasoning.
- **Pipeline Implementation:** Bridges 3D Scheimpflug tomography into everyday physical models:
  - ⌚ **The Watch Glass Window:** Explains corneal curvature, symmetry, and steepness.
  - 📑 **The Stack of Microscopic Paper Sheets:** Explains pachymetric thickness ($1\ \mu\text{m} = 1/1000\text{th mm}$; healthy cornea $\approx 540\ \mu\text{m}$).
  - ⛰️ **Topographic Elevation Contours:** Explains posterior back-surface elevation using contour lines (peaks vs valleys).
  - 🗺️ **Color Map Decoder:** Demystifies Pentacam green, yellow, and red color codes.

### 4. Stage 4: Anterior Cingulate Cortex / ACC Proxy (Cognitive Control & Gating)
- **Biological Role (Brodmann Areas 24, 32):** Conflict detection, error correction, inhibitory control, and goal alignment.
- **Pipeline Implementation:**
  - 🚫 **Inhibitory Gating:** Suppresses definitive diagnostic declarations (e.g., replaces *"You have Keratoconus"* with *"The scan shows localized steepening and thinning patterns evaluated by cornea specialists"*).
  - 🛡️ **Anxiety-Reduction Softening:** Replaces catastrophizing scarewords (*"severe disease", "corneal failure"*) with calm physiological descriptors.
  - ✅ **Factual Alignment Verification:** Deterministic token verification comparing numbers cited in text against Stage 1 ground truth.
  - 🩺 **Doctor Consultation Q&A Generator:** Formulates 4 personalized, empowering questions for the patient's next ophthalmologist visit (e.g., regarding corneal cross-linking CXL, scleral lenses, serial scanning).

---

## 🚀 Quick Start

### 1. Environment Setup (Python 3.11+)
Managed via `uv`:
```bash
# Clone and enter directory
cd "Course Project"

# Install dependencies into virtualenv
uv add google-genai pydantic streamlit pillow pypdf python-dotenv plotly
```

### 2. Configuration
Copy `.env.example` to `.env` and add your Gemini API Key:
```bash
cp .env.example .env
```
*(Note: You can also enter your API Key directly in the Streamlit web interface sidebar!)*

### 3. Run Unit Tests
```bash
uv run python tests/run_all_tests.py
```

### 4. Launch Streamlit Web Application
```bash
uv run streamlit run app.py
```

---

## 📦 Project Structure

```
Course Project/
├── pyproject.toml              # uv project dependencies
├── .env.example                # Environment variables template
├── README.md                   # Comprehensive architectural documentation
├── app.py                      # Streamlit Interactive Web Application
├── src/
│   ├── config.py               # Global settings, models, and constants
│   ├── schemas/                # Pydantic v2 strictly-typed data contracts
│   │   ├── stage1_extraction.py# Stage 1: Pentacam OCR & metric extraction
│   │   ├── stage2_parietal.py  # Stage 2: Parietal integration & evaluations
│   │   ├── stage3_frontal.py   # Stage 3: Frontal scaffolding & analogies
│   │   ├── stage4_acc.py       # Stage 4: ACC gating, audit & doctor Q&A
│   │   └── pipeline_state.py   # Unified pipeline state container
│   ├── guardrails/             # Deterministic clinical boundary engine
│   │   ├── clinical_cutoffs.py # Empirical thresholds (BAD-D, Kmax, Pachy, Elev)
│   │   └── deterministic_engine.py # Pure Python rule-based evaluator
│   ├── pipeline/               # Multi-stage P-FIT execution modules
│   │   ├── stage1_occipital.py # Multimodal vision OCR
│   │   ├── stage2_parietal.py  # Spatial integration & guardrails
│   │   ├── stage3_frontal.py   # Analogical scaffolding & translation
│   │   ├── stage4_acc.py       # Inhibitory gating & report assembly
│   │   └── orchestrator.py     # End-to-end pipeline executor
│   ├── visualization/          # Plotly 3D & 2D interactive corneal maps
│   │   └── corneal_maps.py     # 3D dome mesh, 2D topography, BAD-D gauge
│   └── mock_data/              # High-fidelity realistic clinical cases
│       └── sample_cases.py     # Normal, Forme Fruste, Keratoconus, Post-LASIK
└── tests/
    ├── test_guardrails.py      # Unit tests for deterministic boundary logic
    ├── test_schemas.py         # Unit tests for Pydantic v2 schemas
    └── run_all_tests.py        # Unified test runner
```

---

## 🛡️ Medical Disclaimer
This cognitive scaffolding system is an educational and communication prosthesis designed to improve patient comprehension and shared decision-making. It does not provide clinical diagnoses or replace evaluation by a licensed ophthalmologist or cornea specialist.

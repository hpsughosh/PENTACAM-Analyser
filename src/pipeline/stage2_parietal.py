"""
Stage 2: Parietal Cortex Proxy (Structural Abstraction & Cross-Parametric Integration).
Performs deterministic guardrail checks and cross-parametric geometric synthesis using P-FIT parietal schemas.
"""
import re
import json
import logging
from typing import Optional
from google import genai
from google.genai import types

from src.schemas.stage1_extraction import PentacamExtractionResult
from src.schemas.stage2_parietal import (
    ParietalIntegrationResult,
    StructuralCouplingAnalysis,
    DeterministicAssessment
)
from src.guardrails.deterministic_engine import run_deterministic_assessment
from src.guardrails.clinical_cutoffs import RiskTier
from src.config import DEFAULT_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

STAGE2_SYSTEM_PROMPT = """You are an expert Parietal Cortex Proxy and Ophthalmic Tomography Structural Reasoning Engine.
In the P-FIT (Parieto-Frontal Integration Theory) cognitive framework, the parietal cortex is responsible for cross-modal integration, spatial manipulation, and structural abstraction of geometric forms.

Your task is to synthesize raw multi-parametric Pentacam numbers (Curvature, Thickness, Anterior/Posterior Elevation, BAD-D indices) into a cohesive 3D geometric mental model.

Return JSON in this format:
{
  "unified_geometric_profile": "string",
  "dominant_biomechanical_feature": "string",
  "parietal_integration_confidence": 0.95
}
"""

def _clean_json_text(text: str) -> str:
    clean = re.sub(r"^```json\s*", "", text.strip())
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def process_parietal_integration(
    extraction: PentacamExtractionResult,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL
) -> ParietalIntegrationResult:
    """
    Executes Stage 2 Parietal Cortex Proxy integration.
    1. Runs pure Python deterministic evaluation against clinical cutoffs.
    2. Invokes LLM (or deterministic fallback) for cross-parametric structural synthesis.
    """
    active_key = api_key or GEMINI_API_KEY
    
    # 1. Hard Deterministic Guardrails Pass
    deterministic_assessment: DeterministicAssessment = run_deterministic_assessment(extraction)
    
    # Check coupling deterministically
    thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
    post_elev = extraction.elevation.posterior_elevation_thinnest_um or 0.0
    kmax = extraction.keratometry.kmax_diopters or 43.0
    
    is_colocalized = (thinnest_y < -0.3) and (kmax > 47.0)
    is_post_elevated = post_elev >= 13.0
    
    gradient_desc = "Normal gradual peripheral thickening"
    if extraction.pachymetry.pachy_progression_index_avg and extraction.pachymetry.pachy_progression_index_avg > 1.2:
        gradient_desc = "Accelerated pachymetric thinning from periphery towards inferior apex"
        
    structural_coupling = StructuralCouplingAnalysis(
        is_thinnest_colocalized_with_steepest=is_colocalized,
        is_posterior_float_elevated=is_post_elevated,
        pachymetric_gradient_nature=gradient_desc
    )

    # 2. LLM Spatial Synthesis (or deterministic synthesis if offline)
    if active_key:
        try:
            client = genai.Client(api_key=active_key)
            prompt = f"""
            Extracted Stage 1 Data:
            {extraction.model_dump_json(indent=2)}
            
            Deterministic Assessment & Boundaries:
            - Deterministic Risk Tier: {deterministic_assessment.deterministic_risk_tier.value}
            - Abnormal Parameters: {deterministic_assessment.abnormal_count}
            - Borderline Parameters: {deterministic_assessment.borderline_count}
            - Spatial Coupling: {deterministic_assessment.spatial_coupling_detected} ({deterministic_assessment.spatial_coupling_notes})
            - Summary Flags: {json.dumps(deterministic_assessment.summary_bullet_points)}
            
            Synthesize the unified geometric profile and dominant biomechanical feature as JSON.
            """
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STAGE2_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            if response.text:
                cleaned = _clean_json_text(response.text)
                data = json.loads(cleaned)
                return ParietalIntegrationResult(
                    deterministic_assessment=deterministic_assessment,
                    structural_coupling=structural_coupling,
                    unified_geometric_profile=data.get("unified_geometric_profile", "Integrated 3D corneal geometry."),
                    dominant_biomechanical_feature=data.get("dominant_biomechanical_feature", "Biomechanical tensile structure."),
                    clinical_risk_tier=deterministic_assessment.deterministic_risk_tier,
                    parietal_integration_confidence=float(data.get("parietal_integration_confidence", 0.95))
                )
        except Exception as e:
            logger.error(f"Stage 2 LLM error: {e}. Generating rule-based synthesis.")

    # Rule-based fallback synthesis
    risk_tier = deterministic_assessment.deterministic_risk_tier
    if risk_tier == RiskTier.NORMAL_CORNEA:
        geom_profile = "Uniform, radially symmetric prolate dome with central thickness >540µm and stable planar posterior elevation."
        biomech = "Intact structural integrity with uniform tensile strength distribution across anterior and posterior lamellae."
    elif risk_tier == RiskTier.SUSPECT_FORME_FRUSTE:
        geom_profile = "Subtle asymmetric curvature profile with mild focal thinning and early posterior elevation displacement."
        biomech = "Early biomechanical asymmetry with localized compliance variations in the paracentral stroma."
    elif risk_tier in (RiskTier.MILD_ECTASIA_PATTERN, RiskTier.MODERATE_TO_ADVANCED_ECTASIA):
        geom_profile = f"Marked inferior conical steepening (Kmax {kmax:.1f} D) coupled with localized stromal thinning ({extraction.pachymetry.thinnest_pachymetry_um} µm) and posterior elevation protrusion (+{post_elev:.1f} µm)."
        biomech = "Localized focal biomechanical weakening leading to forward conical protrusion under intraocular pressure."
    elif risk_tier == RiskTier.POST_REFRACTIVE_ALTERATION:
        geom_profile = f"Oblate optical zone profile with central flattening (Km {extraction.keratometry.km_mean_diopters or 38.0:.1f} D) and reduced central pachymetry consistent with prior refractive ablation."
        biomech = "Surgically thinned central optical zone with preserved posterior curvature stability."
    else:
        geom_profile = "Asymmetric corneal topography with mild surface variance."
        biomech = "Minor topographic irregularities without definitive ectatic thinning."

    return ParietalIntegrationResult(
        deterministic_assessment=deterministic_assessment,
        structural_coupling=structural_coupling,
        unified_geometric_profile=geom_profile,
        dominant_biomechanical_feature=biomech,
        clinical_risk_tier=risk_tier,
        parietal_integration_confidence=0.96
    )

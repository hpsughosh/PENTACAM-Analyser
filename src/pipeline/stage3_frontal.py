"""
Stage 3: Frontal / Prefrontal Cortex Proxy (Cognitive Scaffolding & Patient Translation).
Transforms abstract parietal geometric representations into intuitive, scaffolded explanations using physical analogies.
"""
import re
import json
import logging
from typing import Optional, List
from google import genai
from google.genai import types

from src.schemas.stage1_extraction import PentacamExtractionResult
from src.schemas.stage2_parietal import ParietalIntegrationResult
from src.schemas.stage3_frontal import (
    FrontalScaffoldingDraft,
    PhysicalAnalogy,
    ColorMapExplanation
)
from src.guardrails.clinical_cutoffs import RiskTier
from src.config import DEFAULT_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

STAGE3_SYSTEM_PROMPT = """You are an expert Prefrontal Cortex Cognitive Scaffolding Engine and Patient Communication Specialist.
In the P-FIT (Parieto-Frontal Integration Theory) framework, the frontal/prefrontal cortex is responsible for working memory synthesis, mental model generation, and analogical reasoning.

Your mission is to translate complex Scheimpflug corneal tomography data (Pentacam) into an empowering, highly clear, and anxiety-free explanation for a patient holding their diagnostic report.

You MUST use structured physical analogies to make the 3D physics intuitive:
1. "The Watch Glass / Crystal Window": Explaining the cornea as the crystal front window of a wristwatch.
2. "The Stack of Paper Sheets / Layered Fabric": Explaining pachymetry thickness in micrometers (1 micron = 1/1000th of a mm, hair = ~70 microns).
3. "Topographic Mountain / Terrain Map": Explaining elevation maps where red represents high peaks and blue represents flat/deep valleys.

Return a JSON object matching this schema:
{
  "headline_summary": "string",
  "analogies_used": [
    {"domain_concept": "string", "analogy_name": "string", "analogy_explanation": "string"}
  ],
  "corneal_shape_section": "string",
  "corneal_thickness_section": "string",
  "back_surface_section": "string",
  "color_map_guide": [
    {"map_name": "string", "color_meaning": "string", "scan_finding": "string"}
  ],
  "optical_impact_explanation": "string",
  "scaffolding_tier_level": "Intermediate Layperson"
}
"""

def _clean_json_text(text: str) -> str:
    clean = re.sub(r"^```json\s*", "", text.strip())
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def generate_frontal_scaffolding(
    extraction: PentacamExtractionResult,
    parietal: ParietalIntegrationResult,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL
) -> FrontalScaffoldingDraft:
    """
    Executes Stage 3 Frontal/Prefrontal Proxy cognitive scaffolding pass.
    """
    active_key = api_key or GEMINI_API_KEY
    
    if active_key:
        try:
            client = genai.Client(api_key=active_key)
            prompt = f"""
            Stage 1 Extracted Data:
            {extraction.model_dump_json(indent=2)}
            
            Stage 2 Parietal Structural Integration:
            - Risk Tier: {parietal.clinical_risk_tier.value}
            - Unified Geometric Profile: {parietal.unified_geometric_profile}
            - Dominant Biomechanical Feature: {parietal.dominant_biomechanical_feature}
            - Abnormal Metrics: {parietal.deterministic_assessment.abnormal_count}
            - Key Flags: {json.dumps(parietal.deterministic_assessment.summary_bullet_points)}
            
            Please generate the complete FrontalScaffoldingDraft JSON using rich physical analogies and scaffolded patient explanations.
            """
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STAGE3_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            
            if response.text:
                cleaned = _clean_json_text(response.text)
                return FrontalScaffoldingDraft.model_validate_json(cleaned)
        except Exception as e:
            logger.error(f"Stage 3 LLM error: {e}. Generating rule-based scaffolding.")

    # High-quality fallback cognitive scaffolding
    kmax = extraction.keratometry.kmax_diopters or 44.0
    thinnest = extraction.pachymetry.thinnest_pachymetry_um or 540.0
    cct = extraction.pachymetry.central_corneal_thickness_cct_um or 545.0
    post_elev = extraction.elevation.posterior_elevation_thinnest_um or 8.0
    bad_d = extraction.bad_display.final_bad_d_index or 1.0
    tier = parietal.clinical_risk_tier

    analogies = [
        PhysicalAnalogy(
            domain_concept="Corneal Curvature (Front Surface)",
            analogy_name="The Watch Glass / Crystal Window",
            analogy_explanation=(
                "Think of the clear front window of a wristwatch. A healthy cornea is shaped like a smooth, gently rounded "
                f"dome. On your scan, the peak curvature measures {kmax:.1f} Diopters. When this value is higher than average, "
                "it resembles a steeper dome or localized slope rather than a perfectly spherical watch crystal."
            )
        ),
        PhysicalAnalogy(
            domain_concept="Corneal Thickness (Pachymetry)",
            analogy_name="The Stack of Microscopic Paper Sheets",
            analogy_explanation=(
                "The cornea is composed of roughly 200 microscopic, transparent layers of collagen fibers, like a finely pressed "
                f"stack of tissue paper. Your thinnest point measures {thinnest:.0f} micrometers (µm). For context, a single human "
                "hair is about 70 microns wide, and a standard healthy cornea is around 540 microns thick."
            )
        ),
        PhysicalAnalogy(
            domain_concept="Back Surface Elevation",
            analogy_name="Topographic Mountain & Valley Contours",
            analogy_explanation=(
                "Tomography compares the natural back curve of your eye against an ideal reference sphere, like checking whether "
                f"a hill has subtle rises or dips. Your back surface elevation is +{post_elev:.1f} micrometers. Warm colors (reds and yellows) "
                "indicate areas rising above the reference line, while cool blues indicate deeper valleys."
            )
        )
    ]

    color_maps = [
        ColorMapExplanation(
            map_name="Sagittal Curvature Map (Front Surface Power)",
            color_meaning="Cool greens/blues represent flatter curvature; warm oranges/reds represent steeper focusing power.",
            scan_finding=f"Shows peak front power of {kmax:.1f} D with {'localized inferior steepening' if kmax > 48.0 else 'regular symmetric curvature'}."
        ),
        ColorMapExplanation(
            map_name="Pachymetry Map (Thickness Distribution)",
            color_meaning="Warm reds represent very thin zones; cool greens/blues represent thicker, reinforced tissue.",
            scan_finding=f"Shows central thickness of {cct:.0f} µm, with the thinnest area measuring {thinnest:.0f} µm."
        ),
        ColorMapExplanation(
            map_name="Belin-Ambrósio Display (BAD-III Standard Deviations)",
            color_meaning="Green indicates normal standard deviations (<1.6 SD); Yellow indicates borderline/suspect (1.6-2.6 SD); Red indicates elevated deviation (>2.6 SD).",
            scan_finding=f"Final BAD-D index is {bad_d:.2f} SD ({'Red alert' if bad_d >= 2.6 else 'Yellow borderline' if bad_d >= 1.6 else 'Green normal'})."
        )
    ]

    if tier == RiskTier.NORMAL_CORNEA:
        headline = "Your Pentacam scan shows a healthy, well-balanced corneal architecture with uniform thickness and symmetric optical curvature."
        shape_sec = f"Your front corneal curvature is gentle and symmetric, with an average power of {kmax:.1f} Diopters. Light passes smoothly through the front dome without irregular bending."
        thick_sec = f"Corneal thickness is robust throughout, measuring {cct:.0f} µm centrally and {thinnest:.0f} µm at the thinnest area, well above standard safety thresholds."
        back_sec = f"The back surface of the cornea sits flush at +{post_elev:.1f} µm relative to the reference sphere, demonstrating strong mechanical stability."
        optics_sec = "Light rays focus cleanly onto the retina. Any vision correction required is likely standard nearsightedness or mild astigmatism easily corrected with conventional glasses."
    elif tier == RiskTier.SUSPECT_FORME_FRUSTE:
        headline = "Your scan reveals subtle borderline variations in thickness and back-surface curvature that eye specialists monitor closely over time."
        shape_sec = f"The front surface shows early asymmetry with a peak curvature of {kmax:.1f} D, slightly steeper than average but still holding regular shape."
        thick_sec = f"Tissue thickness measures {thinnest:.0f} µm at its thinnest zone. While slightly thinner than average, the layers remain supportive."
        back_sec = f"The posterior back surface shows an elevation of +{post_elev:.1f} µm, placing it in a watchful borderline zone (yellow on tomography displays)."
        optics_sec = "Subtle variations in curvature may cause mild night glare or slight ghosting, which can be monitored during regular routine checkups."
    else:
        headline = "Your scan displays patterns of localized steepening and thinning on the lower part of the cornea that ophthalmologists evaluate when assessing corneal shape stability."
        shape_sec = f"The front surface exhibits a localized steep cone area (Kmax of {kmax:.1f} Diopters), where the curvature rises more steeply than the surrounding dome."
        thick_sec = f"The corneal wall shows localized thinning, measuring {thinnest:.0f} µm at the thinnest point (compared to standard ~540 µm), located predominantly in the lower sector."
        back_sec = f"The back surface shows forward elevation (+{post_elev:.1f} µm) and an enhanced BAD-D index of {bad_d:.2f} SD, reflecting focal forward movement under natural eye pressure."
        optics_sec = "Because the surface resembles a teardrop or cone rather than a spherical window, light rays scatter slightly, causing symptoms like multiple faint outlines (ghosting), halos around lights, and frequent prescription changes."

    return FrontalScaffoldingDraft(
        headline_summary=headline,
        analogies_used=analogies,
        corneal_shape_section=shape_sec,
        corneal_thickness_section=thick_sec,
        back_surface_section=back_sec,
        color_map_guide=color_maps,
        optical_impact_explanation=optics_sec,
        scaffolding_tier_level="Intermediate Layperson"
    )

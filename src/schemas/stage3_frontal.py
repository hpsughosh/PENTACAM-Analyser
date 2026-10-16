"""
Stage 3: Frontal / Prefrontal Cortex Proxy Schema.
Data contract for cognitive scaffolding, analogical modeling, and patient-tailored translation.
"""
from typing import List, Dict
from pydantic import BaseModel, Field


class PhysicalAnalogy(BaseModel):
    """Intuitive physical metaphor bridging complex physics to everyday objects."""
    domain_concept: str = Field(
        description="The clinical parameter (e.g. Corneal Curvature / Kmax, Pachymetry Thickness, Posterior Elevation)."
    )
    analogy_name: str = Field(
        description="Name of the metaphor (e.g. 'Watch Crystal Window', 'Stack of Microscopic Paper Sheets', 'Topographic Mountain Slope')."
    )
    analogy_explanation: str = Field(
        description="Clear, intuitive comparison explaining what the scan finding means in terms of the metaphor."
    )


class ColorMapExplanation(BaseModel):
    """Decoding the color heatmaps printed on the Pentacam report."""
    map_name: str = Field(
        description="Name of the map (e.g., Sagittal Curvature Map, Pachymetry Thickness Map, Back Elevation Map)."
    )
    color_meaning: str = Field(
        description="What the colors (cool blues/greens vs warm yellows/reds) represent on this specific map."
    )
    scan_finding: str = Field(
        description="What the patient's individual scan shows on this map."
    )


class FrontalScaffoldingDraft(BaseModel):
    """Stage 3 pre-reflection patient explanation draft with tiered cognitive scaffolding."""
    headline_summary: str = Field(
        description="A clear, calm 2-sentence executive summary of the eye scan."
    )
    analogies_used: List[PhysicalAnalogy] = Field(
        default_factory=list,
        description="Curated physical analogies used to explain curvature, thickness, and elevation."
    )
    corneal_shape_section: str = Field(
        description="Scaffolded explanation of the front surface curvature and symmetry."
    )
    corneal_thickness_section: str = Field(
        description="Scaffolded explanation of corneal thickness and distribution across the eye."
    )
    back_surface_section: str = Field(
        description="Scaffolded explanation of the posterior back surface elevation and structural stability."
    )
    color_map_guide: List[ColorMapExplanation] = Field(
        default_factory=list,
        description="Step-by-step walkthrough of the color-coded visual maps in their report."
    )
    optical_impact_explanation: str = Field(
        description="How these structural features influence how light enters the eye and why standard glasses or lenses behave as they do."
    )
    scaffolding_tier_level: str = Field(
        default="Intermediate Layperson",
        description="Target cognitive reading level."
    )

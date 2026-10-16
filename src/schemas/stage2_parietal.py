"""
Stage 2: Parietal Cortex Proxy Schema.
Data contract for cross-parametric spatial integration, deterministic guardrail evaluations, and structural abstraction.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from src.guardrails.clinical_cutoffs import ParameterStatus, RiskTier


class EvaluatedParameter(BaseModel):
    name: str
    value: Optional[float]
    unit: str
    status: ParameterStatus
    reference_range: str
    clinical_interpretation: str
    deviation_severity: float = 0.0  # 0.0 = normal, 1.0 = borderline, 2.0+ = abnormal


class DeterministicAssessment(BaseModel):
    parameter_evaluations: Dict[str, EvaluatedParameter]
    abnormal_count: int
    borderline_count: int
    normal_count: int
    unavailable_count: int
    deterministic_risk_tier: RiskTier
    risk_score_percentage: float
    spatial_coupling_detected: bool = False
    spatial_coupling_notes: str = ""
    summary_bullet_points: List[str]


class StructuralCouplingAnalysis(BaseModel):
    """Evaluation of how curvature, thickness, and elevation co-localize in space."""
    is_thinnest_colocalized_with_steepest: bool = Field(
        default=False,
        description="True if the thinnest point aligns geographically with the apex of steep curvature."
    )
    is_posterior_float_elevated: bool = Field(
        default=False,
        description="True if the posterior corneal surface shows abnormal outward protrusion."
    )
    pachymetric_gradient_nature: str = Field(
        default="Normal concentric thickening from center to periphery",
        description="Description of thickness progression from apex to periphery."
    )


class ParietalIntegrationResult(BaseModel):
    """Stage 2 output integrating cross-parametric geometry with deterministic guardrails."""
    deterministic_assessment: DeterministicAssessment = Field(
        description="Grounded deterministic clinical rule evaluations."
    )
    structural_coupling: StructuralCouplingAnalysis = Field(
        default_factory=StructuralCouplingAnalysis,
        description="Coupling analysis between curvature, pachymetry, and elevation."
    )
    unified_geometric_profile: str = Field(
        description="Cohesive high-level geometric description of the 3D corneal dome."
    )
    dominant_biomechanical_feature: str = Field(
        description="Primary mechanical characteristic (e.g., 'Normal regular prolate dome', 'Inferior-temporal focal thinning and steep cone', 'Central flattening post-refractive')."
    )
    clinical_risk_tier: RiskTier = Field(
        description="Consolidated clinical risk tier validated against deterministic boundaries."
    )
    parietal_integration_confidence: float = Field(
        default=0.95,
        description="Confidence score for structural abstraction (0.0 to 1.0)"
    )

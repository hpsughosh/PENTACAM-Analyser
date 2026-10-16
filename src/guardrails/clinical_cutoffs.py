"""
Clinical Cutoff Boundaries and Empirical Thresholds for Pentacam Scheimpflug Tomography.
Based on Global Consensus on Keratoconus and Ectatic Diseases, Belin-Ambrósio Criteria, and Cornea Topography Standards.
"""
from dataclasses import dataclass
from enum import Enum


class ParameterStatus(str, Enum):
    NORMAL = "NORMAL"
    BORDERLINE = "BORDERLINE"
    ABNORMAL = "ABNORMAL"
    UNAVAILABLE = "UNAVAILABLE"


class RiskTier(str, Enum):
    NORMAL_CORNEA = "NORMAL_CORNEA"
    SUSPECT_FORME_FRUSTE = "SUSPECT_FORME_FRUSTE"
    MILD_ECTASIA_PATTERN = "MILD_ECTASIA_PATTERN"
    MODERATE_TO_ADVANCED_ECTASIA = "MODERATE_TO_ADVANCED_ECTASIA"
    POST_REFRACTIVE_ALTERATION = "POST_REFRACTIVE_ALTERATION"
    BORDERLINE_ASYMMETRY = "BORDERLINE_ASYMMETRY"


@dataclass(frozen=True)
class CutoffRule:
    name: str
    unit: str
    normal_max: float | None = None
    normal_min: float | None = None
    borderline_max: float | None = None
    borderline_min: float | None = None
    abnormal_trigger_high: float | None = None
    abnormal_trigger_low: float | None = None
    description: str = ""


# Standard Tomography Thresholds
CUTOFF_RULES = {
    "kmax": CutoffRule(
        name="Maximum Keratometry (Kmax)",
        unit="Diopters (D)",
        normal_max=47.0,
        borderline_min=47.0,
        borderline_max=48.5,
        abnormal_trigger_high=48.5,
        description="Peak sagittal curvature; values above 48.5 D indicate steep cone apex."
    ),
    "thinnest_pachy": CutoffRule(
        name="Thinnest Pachymetry",
        unit="microns (µm)",
        normal_min=500.0,
        borderline_min=470.0,
        borderline_max=500.0,
        abnormal_trigger_low=470.0,
        description="Corneal thickness at thinnest point; below 470 µm indicates marked structural thinning."
    ),
    "cct": CutoffRule(
        name="Central Corneal Thickness (CCT)",
        unit="microns (µm)",
        normal_min=510.0,
        borderline_min=480.0,
        borderline_max=510.0,
        abnormal_trigger_low=480.0,
        description="Corneal thickness at the central apex."
    ),
    "posterior_elevation": CutoffRule(
        name="Posterior Elevation at Thinnest Point",
        unit="microns (µm)",
        normal_max=13.0,
        borderline_min=13.0,
        borderline_max=16.0,
        abnormal_trigger_high=16.0,
        description="Back surface protrusion relative to Best Fit Sphere; earliest indicator of ectasia."
    ),
    "anterior_elevation": CutoffRule(
        name="Anterior Elevation at Thinnest Point",
        unit="microns (µm)",
        normal_max=6.0,
        borderline_min=6.0,
        borderline_max=8.0,
        abnormal_trigger_high=8.0,
        description="Front surface protrusion relative to Best Fit Sphere."
    ),
    "bad_d": CutoffRule(
        name="Belin-Ambrósio Final D-Index",
        unit="Standard Deviations (SD)",
        normal_max=1.60,
        borderline_min=1.60,
        borderline_max=2.60,
        abnormal_trigger_high=2.60,
        description="Composite standard deviation index; <1.60 Normal (Green), 1.60-2.60 Suspect (Yellow), >=2.60 Pathological (Red)."
    ),
    "art_max": CutoffRule(
        name="Ambrósio Relational Thinnest Max (ARTmax)",
        unit="microns (µm)",
        normal_min=410.0,
        borderline_min=339.0,
        borderline_max=410.0,
        abnormal_trigger_low=339.0,
        description="Relational pachymetric index; below 339 µm indicates high ectasia risk."
    ),
    "isv": CutoffRule(
        name="Index of Surface Variance (ISV)",
        unit="",
        normal_max=37.0,
        borderline_min=37.0,
        borderline_max=45.0,
        abnormal_trigger_high=45.0,
        description="Measures front corneal curvature irregularity."
    ),
    "iva": CutoffRule(
        name="Index of Vertical Asymmetry (IVA)",
        unit="",
        normal_max=0.28,
        borderline_min=0.28,
        borderline_max=0.32,
        abnormal_trigger_high=0.32,
        description="Measures asymmetric power distribution between upper and lower cornea."
    ),
    "astigmatism": CutoffRule(
        name="Corneal Astigmatism",
        unit="Diopters (D)",
        normal_max=2.5,
        borderline_min=2.5,
        borderline_max=4.0,
        abnormal_trigger_high=4.0,
        description="Degree of regular or irregular astigmatism."
    )
}

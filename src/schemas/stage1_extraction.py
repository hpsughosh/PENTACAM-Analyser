"""
Stage 1: Occipital-Temporal Proxy Schema.
Strictly typed Pydantic v2 data contract for multimodal Pentacam scan feature extraction.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ExaminationMetadata(BaseModel):
    """Scan and patient metadata extracted from Pentacam header."""
    eye: str = Field(
        default="Unknown",
        description="Examined eye: OD (Right), OS (Left), or Unknown"
    )
    exam_date: Optional[str] = Field(
        default=None,
        description="Date of the tomography examination"
    )
    scan_quality: str = Field(
        default="OK",
        description="Quality specification: 'OK', 'Model', 'Fixation lost', or 'Yellow/Red warning'"
    )
    device_model: str = Field(
        default="Oculus Pentacam / Scheimpflug System",
        description="Tomographer hardware model name"
    )
    software_version: Optional[str] = Field(
        default=None,
        description="Software version used for acquisition"
    )


class KeratometryIndices(BaseModel):
    """Curvature & refractive parameters (Front Sagittal/Tangential map)."""
    k1_flat_diopters: Optional[float] = Field(
        default=None,
        description="Flat meridian curvature (K1) in Diopters (D)"
    )
    k1_axis_deg: Optional[float] = Field(
        default=None,
        description="Flat meridian axis in degrees"
    )
    k2_steep_diopters: Optional[float] = Field(
        default=None,
        description="Steep meridian curvature (K2) in Diopters (D)"
    )
    k2_axis_deg: Optional[float] = Field(
        default=None,
        description="Steep meridian axis in degrees"
    )
    km_mean_diopters: Optional[float] = Field(
        default=None,
        description="Mean corneal curvature (Km) in Diopters (D)"
    )
    kmax_diopters: Optional[float] = Field(
        default=None,
        description="Maximum keratometry / apex steepness (Kmax) in Diopters (D)"
    )
    astigmatism_diopters: Optional[float] = Field(
        default=None,
        description="Corneal astigmatism magnitude (K2 - K1) in Diopters (D)"
    )
    q_value_asphericity: Optional[float] = Field(
        default=None,
        description="Asphericity Q-value (8mm or 6mm zone)"
    )
    rmin_curvature_radius_mm: Optional[float] = Field(
        default=None,
        description="Minimum sagittal curvature radius in mm"
    )


class PachymetryIndices(BaseModel):
    """Corneal thickness distribution parameters (Pachymetry map)."""
    central_corneal_thickness_cct_um: Optional[float] = Field(
        default=None,
        description="Central corneal thickness at apex in micrometers (µm)"
    )
    thinnest_pachymetry_um: Optional[float] = Field(
        default=None,
        description="Corneal thickness at the absolute thinnest point in micrometers (µm)"
    )
    pupil_center_pachymetry_um: Optional[float] = Field(
        default=None,
        description="Corneal thickness at the pupil center in micrometers (µm)"
    )
    thinnest_loc_x_mm: Optional[float] = Field(
        default=None,
        description="Horizontal (X) spatial displacement of thinnest point from pupil center in mm"
    )
    thinnest_loc_y_mm: Optional[float] = Field(
        default=None,
        description="Vertical (Y) spatial displacement of thinnest point from pupil center in mm (negative = inferior)"
    )
    pachy_progression_index_avg: Optional[float] = Field(
        default=None,
        description="Average Pachymetric Progression Index (PPI avg)"
    )
    pachy_progression_index_max: Optional[float] = Field(
        default=None,
        description="Maximum Pachymetric Progression Index (PPI max)"
    )
    art_max_um: Optional[float] = Field(
        default=None,
        description="Ambrósio Relational Thinnest maximal index (ARTmax) in µm"
    )


class ElevationIndices(BaseModel):
    """Front and Back elevation metrics relative to Best Fit Sphere (BFS) or Toric Ellipsoid (BFTE)."""
    anterior_elevation_thinnest_um: Optional[float] = Field(
        default=None,
        description="Front/Anterior elevation at the thinnest point in micrometers (µm)"
    )
    posterior_elevation_thinnest_um: Optional[float] = Field(
        default=None,
        description="Back/Posterior elevation at the thinnest point in micrometers (µm)"
    )
    reference_surface_type: str = Field(
        default="Best Fit Sphere (BFS)",
        description="Reference body used for elevation calculation (BFS or BFTE)"
    )
    enhanced_delta_front_elevation_um: Optional[float] = Field(
        default=None,
        description="Belin-Ambrósio enhanced elevation front difference in µm"
    )
    enhanced_delta_back_elevation_um: Optional[float] = Field(
        default=None,
        description="Belin-Ambrósio enhanced elevation back difference in µm"
    )


class BelinAmbrosioIndices(BaseModel):
    """Belin-Ambrósio Enhanced Ectasia Display (BAD-III) Standard Deviation D-values."""
    d_f_front_elevation: Optional[float] = Field(
        default=None,
        description="Df: Standard deviation of front elevation change"
    )
    d_b_back_elevation: Optional[float] = Field(
        default=None,
        description="Db: Standard deviation of back elevation change"
    )
    d_t_thinnest_pachy: Optional[float] = Field(
        default=None,
        description="Dt: Standard deviation of thinnest point pachymetry"
    )
    d_a_pachy_progression: Optional[float] = Field(
        default=None,
        description="Da: Standard deviation of pachymetric progression"
    )
    d_p_displacement: Optional[float] = Field(
        default=None,
        description="Dp: Standard deviation of thinnest point displacement"
    )
    final_bad_d_index: Optional[float] = Field(
        default=None,
        description="Final D-index (BAD-D standard deviation score; <1.60 normal, 1.60-2.60 suspect, >2.60 pathological)"
    )


class TopometricIndices(BaseModel):
    """Keratoconus detection topometric indices."""
    isv_surface_variance: Optional[float] = Field(
        default=None,
        description="Index of Surface Variance (ISV)"
    )
    iva_vertical_asymmetry: Optional[float] = Field(
        default=None,
        description="Index of Vertical Asymmetry (IVA)"
    )
    ki_keratoconus_index: Optional[float] = Field(
        default=None,
        description="Keratoconus Index (KI)"
    )
    cki_center_keratoconus_index: Optional[float] = Field(
        default=None,
        description="Center Keratoconus Index (CKI)"
    )
    ihd_height_decentration: Optional[float] = Field(
        default=None,
        description="Index of Height Decentration (IHD)"
    )


class PentacamExtractionResult(BaseModel):
    """Root Stage 1 schema containing all extracted Pentacam tomography parameters."""
    metadata: ExaminationMetadata = Field(default_factory=ExaminationMetadata)
    keratometry: KeratometryIndices = Field(default_factory=KeratometryIndices)
    pachymetry: PachymetryIndices = Field(default_factory=PachymetryIndices)
    elevation: ElevationIndices = Field(default_factory=ElevationIndices)
    bad_display: BelinAmbrosioIndices = Field(default_factory=BelinAmbrosioIndices)
    topometric: TopometricIndices = Field(default_factory=TopometricIndices)
    raw_ocr_notes: Optional[str] = Field(
        default=None,
        description="Observations regarding map types present (4-Maps Refractive, BAD-III, Topometric) and image legibility."
    )
    extraction_confidence_score: float = Field(
        default=0.95,
        description="Confidence estimate of OCR token extraction (0.0 to 1.0)"
    )

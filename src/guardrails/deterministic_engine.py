"""
Stage 2 Deterministic Guardrails Engine.
Performs pure Python deterministic evaluation and boundary checks on extracted Pentacam metrics.
"""
from typing import Dict, Any, List, Optional
from src.guardrails.clinical_cutoffs import (
    CUTOFF_RULES,
    ParameterStatus,
    RiskTier,
    CutoffRule
)
from src.schemas.stage1_extraction import PentacamExtractionResult
from src.schemas.stage2_parietal import EvaluatedParameter, DeterministicAssessment


def evaluate_single_parameter(rule_key: str, value: Optional[float]) -> EvaluatedParameter:
    """Evaluate a single numeric parameter against its CutoffRule."""
    if rule_key not in CUTOFF_RULES:
        return EvaluatedParameter(
            name=rule_key,
            value=value,
            unit="",
            status=ParameterStatus.UNAVAILABLE,
            reference_range="N/A",
            clinical_interpretation="No reference rule defined."
        )

    rule = CUTOFF_RULES[rule_key]

    if value is None:
        return EvaluatedParameter(
            name=rule.name,
            value=None,
            unit=rule.unit,
            status=ParameterStatus.UNAVAILABLE,
            reference_range=f"Normal range depends on {rule.name}",
            clinical_interpretation="Metric not captured in the uploaded scan view."
        )

    status = ParameterStatus.NORMAL
    severity = 0.0
    ref_str = ""

    # Upper bound check (e.g. Kmax, Posterior Elevation, BAD-D, ISV, IVA)
    if rule.normal_max is not None:
        ref_str = f"< {rule.normal_max} {rule.unit}".strip()
        if rule.abnormal_trigger_high is not None and value >= rule.abnormal_trigger_high:
            status = ParameterStatus.ABNORMAL
            severity = 2.0 + min(2.0, (value - rule.abnormal_trigger_high) / (rule.abnormal_trigger_high * 0.1 or 1))
        elif rule.borderline_min is not None and value >= rule.borderline_min:
            status = ParameterStatus.BORDERLINE
            severity = 1.0
        else:
            status = ParameterStatus.NORMAL
            severity = 0.0

    # Lower bound check (e.g. Thinnest Pachymetry, CCT, ARTmax)
    elif rule.normal_min is not None:
        ref_str = f"> {rule.normal_min} {rule.unit}".strip()
        if rule.abnormal_trigger_low is not None and value <= rule.abnormal_trigger_low:
            status = ParameterStatus.ABNORMAL
            severity = 2.0 + min(2.0, (rule.abnormal_trigger_low - value) / (rule.abnormal_trigger_low * 0.1 or 1))
        elif rule.borderline_max is not None and value <= rule.borderline_max:
            status = ParameterStatus.BORDERLINE
            severity = 1.0
        else:
            status = ParameterStatus.NORMAL
            severity = 0.0

    interpretation = ""
    if status == ParameterStatus.NORMAL:
        interpretation = f"Within standard physiologic limits ({value} {rule.unit})."
    elif status == ParameterStatus.BORDERLINE:
        interpretation = f"Borderline / suspicious elevation or thinning ({value} {rule.unit}); requires contextual correlation."
    else:
        interpretation = f"Significantly outside standard limits ({value} {rule.unit}); matches patterns associated with focal corneal steepening/thinning."

    return EvaluatedParameter(
        name=rule.name,
        value=value,
        unit=rule.unit,
        status=status,
        reference_range=ref_str,
        clinical_interpretation=interpretation,
        deviation_severity=round(severity, 2)
    )


def run_deterministic_assessment(extraction: PentacamExtractionResult) -> DeterministicAssessment:
    """
    Executes deterministic clinical lookup logic over extracted Stage 1 tomography metrics.
    Acts as a hard boundary guardrail for Stage 2.
    """
    evaluations: Dict[str, EvaluatedParameter] = {}

    # 1. Keratometry
    evaluations["kmax"] = evaluate_single_parameter("kmax", extraction.keratometry.kmax_diopters)
    evaluations["astigmatism"] = evaluate_single_parameter("astigmatism", extraction.keratometry.astigmatism_diopters)

    # 2. Pachymetry
    evaluations["thinnest_pachy"] = evaluate_single_parameter("thinnest_pachy", extraction.pachymetry.thinnest_pachymetry_um)
    evaluations["cct"] = evaluate_single_parameter("cct", extraction.pachymetry.central_corneal_thickness_cct_um)
    evaluations["art_max"] = evaluate_single_parameter("art_max", extraction.pachymetry.art_max_um)

    # 3. Elevation
    evaluations["posterior_elevation"] = evaluate_single_parameter("posterior_elevation", extraction.elevation.posterior_elevation_thinnest_um)
    evaluations["anterior_elevation"] = evaluate_single_parameter("anterior_elevation", extraction.elevation.anterior_elevation_thinnest_um)

    # 4. Belin-Ambrosio Display
    evaluations["bad_d"] = evaluate_single_parameter("bad_d", extraction.bad_display.final_bad_d_index)

    # 5. Topometric
    evaluations["isv"] = evaluate_single_parameter("isv", extraction.topometric.isv_surface_variance)
    evaluations["iva"] = evaluate_single_parameter("iva", extraction.topometric.iva_vertical_asymmetry)

    # Count statuses
    abnormal = sum(1 for e in evaluations.values() if e.status == ParameterStatus.ABNORMAL)
    borderline = sum(1 for e in evaluations.values() if e.status == ParameterStatus.BORDERLINE)
    normal = sum(1 for e in evaluations.values() if e.status == ParameterStatus.NORMAL)
    unavailable = sum(1 for e in evaluations.values() if e.status == ParameterStatus.UNAVAILABLE)

    # Check Spatial Coupling (e.g. Inferior displacement of thinnest point + steepening)
    y_disp = extraction.pachymetry.thinnest_loc_y_mm
    spatial_coupling = False
    spatial_notes = ""
    if y_disp is not None and y_disp < -0.3:
        spatial_coupling = True
        spatial_notes = f"Thinnest point is displaced inferiorly by {abs(y_disp):.2f} mm from the pupil center, characteristic of inferior ectatic cones."

    # Determine Risk Tier deterministically
    bad_d_val = extraction.bad_display.final_bad_d_index
    post_elev = extraction.elevation.posterior_elevation_thinnest_um
    kmax_val = extraction.keratometry.kmax_diopters
    thinnest_val = extraction.pachymetry.thinnest_pachymetry_um
    q_val = extraction.keratometry.q_value_asphericity

    risk_tier = RiskTier.NORMAL_CORNEA
    if (bad_d_val is not None and bad_d_val >= 2.60) or (post_elev is not None and post_elev >= 18.0) or (kmax_val is not None and kmax_val >= 50.0 and abnormal >= 3):
        risk_tier = RiskTier.MODERATE_TO_ADVANCED_ECTASIA
    elif abnormal >= 2 or (bad_d_val is not None and bad_d_val >= 1.60 and abnormal >= 1) or (post_elev is not None and post_elev >= 16.0):
        risk_tier = RiskTier.MILD_ECTASIA_PATTERN
    elif (bad_d_val is not None and 1.60 <= bad_d_val < 2.60) or borderline >= 2 or (post_elev is not None and 13.0 <= post_elev < 16.0) or (kmax_val is not None and 47.0 <= kmax_val < 48.5):
        risk_tier = RiskTier.SUSPECT_FORME_FRUSTE
    elif q_val is not None and q_val > 0.1 and thinnest_val is not None and thinnest_val < 500:
        risk_tier = RiskTier.POST_REFRACTIVE_ALTERATION
    elif borderline >= 1:
        risk_tier = RiskTier.BORDERLINE_ASYMMETRY
    else:
        risk_tier = RiskTier.NORMAL_CORNEA

    # Calculate overall quantitative risk score (0 to 100%)
    total_valid = normal + borderline + abnormal
    if total_valid > 0:
        total_sev = sum(e.deviation_severity for e in evaluations.values())
        risk_pct = min(100.0, (total_sev / (total_valid * 2.0)) * 100.0)
    else:
        risk_pct = 0.0

    # Generate grounded summary bullets
    bullets = []
    if risk_tier == RiskTier.NORMAL_CORNEA:
        bullets.append("Curvature and pachymetric metrics are within normal standard limits.")
        bullets.append("Back elevation and BAD-D indices demonstrate symmetric, stable corneal geometry.")
    elif risk_tier == RiskTier.SUSPECT_FORME_FRUSTE:
        bullets.append("Subtle localized variations detected in corneal thickness or posterior elevation.")
        bullets.append("Values hover in borderline/suspect thresholds (yellow warning zones on Pentacam BAD-III).")
    elif risk_tier in (RiskTier.MILD_ECTASIA_PATTERN, RiskTier.MODERATE_TO_ADVANCED_ECTASIA):
        if kmax_val and kmax_val >= 48.5:
            bullets.append(f"Steep maximum curvature (Kmax = {kmax_val} D) indicating localized focal steepening.")
        if thinnest_val and thinnest_val < 470:
            bullets.append(f"Corneal thinning at thinnest point ({thinnest_val} µm) below standard threshold.")
        if post_elev and post_elev >= 16:
            bullets.append(f"Elevated posterior surface protrusion (+{post_elev} µm) relative to reference sphere.")
        if bad_d_val and bad_d_val >= 1.60:
            bullets.append(f"Enhanced Ectasia Index (BAD-D = {bad_d_val:.2f}) indicates structural deviation.")

    if spatial_coupling:
        bullets.append(spatial_notes)

    return DeterministicAssessment(
        parameter_evaluations=evaluations,
        abnormal_count=abnormal,
        borderline_count=borderline,
        normal_count=normal,
        unavailable_count=unavailable,
        deterministic_risk_tier=risk_tier,
        risk_score_percentage=round(risk_pct, 1),
        spatial_coupling_detected=spatial_coupling,
        spatial_coupling_notes=spatial_notes,
        summary_bullet_points=bullets
    )

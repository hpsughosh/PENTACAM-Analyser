"""
Unit Tests for Pentacam PDF & Text Extraction Engine.
"""
from src.pipeline.pdf_parser import parse_pentacam_text
from src.guardrails.deterministic_engine import run_deterministic_assessment
from src.guardrails.clinical_cutoffs import RiskTier


def test_pentacam_pdf_text_parser():
    sample_text = """
    OCULUS Pentacam HR
    Patient: Anonymous
    Eye: OD (Right Eye)
    Examination Date: 2026-06-20
    Quality Specification: OK
    
    Front Sagittal Curvature:
    Kmax: 46.6 D
    K1: 43.8 D @ 170 deg
    K2: 46.1 D @ 80 deg
    Km: 44.95 D
    Astig: 2.3 D
    
    Pachymetry Map:
    CCT: 472 um
    Thinnest Location: 456 um
    Pupil Center: 470 um
    PPI avg: 1.65
    ARTmax: 276
    
    Elevation (8.0mm BFS):
    F.Ele.Th: +11.0 um
    B.Ele.Th: +33.0 um
    
    Belin-Ambrosio Enhanced Ectasia Display (BAD-III):
    Df: 2.85
    Db: 4.60
    Dt: 2.70
    Da: 3.90
    Dp: 3.30
    Final 'D': 5.34
    """
    
    res, count = parse_pentacam_text(sample_text)
    
    assert count >= 6, f"Expected at least 6 matched fields, got {count}"
    assert res.metadata.eye == "OD (Right Eye)", f"Expected OD, got {res.metadata.eye}"
    assert res.keratometry.kmax_diopters == 46.6, f"Expected Kmax 46.6, got {res.keratometry.kmax_diopters}"
    assert res.pachymetry.thinnest_pachymetry_um == 456.0, f"Expected 456.0, got {res.pachymetry.thinnest_pachymetry_um}"
    assert res.elevation.posterior_elevation_thinnest_um == 33.0, f"Expected +33.0, got {res.elevation.posterior_elevation_thinnest_um}"
    assert res.bad_display.final_bad_d_index == 5.34, f"Expected 5.34, got {res.bad_display.final_bad_d_index}"
    
    # Run deterministic assessment on extracted results
    assessment = run_deterministic_assessment(res)
    assert assessment.deterministic_risk_tier in (RiskTier.MILD_ECTASIA_PATTERN, RiskTier.MODERATE_TO_ADVANCED_ECTASIA)

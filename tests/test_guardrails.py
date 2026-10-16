"""
Unit Tests for Deterministic Guardrails & Clinical Boundary Logic.
"""
from src.guardrails.clinical_cutoffs import ParameterStatus, RiskTier
from src.guardrails.deterministic_engine import (
    run_deterministic_assessment,
    evaluate_single_parameter
)
from src.mock_data.sample_cases import SAMPLE_CASES


def test_normal_case_guardrails():
    normal_case = SAMPLE_CASES["case_1_normal"]["data"]
    assessment = run_deterministic_assessment(normal_case)
    
    assert assessment.deterministic_risk_tier == RiskTier.NORMAL_CORNEA
    assert assessment.abnormal_count == 0
    assert assessment.risk_score_percentage < 20.0
    assert assessment.parameter_evaluations["kmax"].status == ParameterStatus.NORMAL
    assert assessment.parameter_evaluations["thinnest_pachy"].status == ParameterStatus.NORMAL
    assert assessment.parameter_evaluations["posterior_elevation"].status == ParameterStatus.NORMAL
    assert assessment.parameter_evaluations["bad_d"].status == ParameterStatus.NORMAL


def test_subclinical_case_guardrails():
    subclinical_case = SAMPLE_CASES["case_2_subclinical"]["data"]
    assessment = run_deterministic_assessment(subclinical_case)
    
    assert assessment.deterministic_risk_tier in (RiskTier.SUSPECT_FORME_FRUSTE, RiskTier.BORDERLINE_ASYMMETRY)
    assert assessment.borderline_count >= 1
    assert assessment.parameter_evaluations["bad_d"].status == ParameterStatus.BORDERLINE


def test_keratoconus_case_guardrails():
    kc_case = SAMPLE_CASES["case_3_keratoconus"]["data"]
    assessment = run_deterministic_assessment(kc_case)
    
    assert assessment.deterministic_risk_tier in (RiskTier.MODERATE_TO_ADVANCED_ECTASIA, RiskTier.MILD_ECTASIA_PATTERN)
    assert assessment.abnormal_count >= 3
    assert assessment.parameter_evaluations["kmax"].status == ParameterStatus.ABNORMAL
    assert assessment.parameter_evaluations["thinnest_pachy"].status == ParameterStatus.ABNORMAL
    assert assessment.parameter_evaluations["posterior_elevation"].status == ParameterStatus.ABNORMAL
    assert assessment.parameter_evaluations["bad_d"].status == ParameterStatus.ABNORMAL
    assert assessment.spatial_coupling_detected is True


def test_cutoff_evaluator():
    # Kmax tests
    eval_norm = evaluate_single_parameter("kmax", 43.5)
    assert eval_norm.status == ParameterStatus.NORMAL
    
    eval_border = evaluate_single_parameter("kmax", 47.5)
    assert eval_border.status == ParameterStatus.BORDERLINE
    
    eval_abnormal = evaluate_single_parameter("kmax", 52.0)
    assert eval_abnormal.status == ParameterStatus.ABNORMAL

    # Thinnest pachy tests
    eval_thin_norm = evaluate_single_parameter("thinnest_pachy", 530.0)
    assert eval_thin_norm.status == ParameterStatus.NORMAL
    
    eval_thin_abnormal = evaluate_single_parameter("thinnest_pachy", 440.0)
    assert eval_thin_abnormal.status == ParameterStatus.ABNORMAL


if __name__ == "__main__":
    test_normal_case_guardrails()
    test_subclinical_case_guardrails()
    test_keratoconus_case_guardrails()
    test_cutoff_evaluator()
    print("All guardrail tests passed successfully!")

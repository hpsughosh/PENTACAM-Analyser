"""
Unified Test Runner for P-FIT Pipeline.
"""
import sys
from pathlib import Path

# Ensure root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from tests.test_guardrails import (
    test_normal_case_guardrails,
    test_subclinical_case_guardrails,
    test_keratoconus_case_guardrails,
    test_cutoff_evaluator
)
from tests.test_schemas import (
    test_stage1_schema_serialization,
    test_full_pipeline_orchestration_mock
)
from tests.test_pdf_parser import test_pentacam_pdf_text_parser
from tests.test_bilateral_flow import test_bilateral_asymmetry_stats, test_multi_eye_pipeline_case6

def run_suite():
    print("=" * 60)
    print("RUNNING P-FIT CORNEAL TOMOGRAPHY PIPELINE TEST SUITE")
    print("=" * 60)
    
    print("\n[1/8] Testing Normal Case Deterministic Guardrails...")
    test_normal_case_guardrails()
    print("  -> Passed!")
    
    print("\n[2/8] Testing Subclinical/Forme Fruste Guardrails...")
    test_subclinical_case_guardrails()
    print("  -> Passed!")
    
    print("\n[3/8] Testing Keratoconus Pattern Guardrails...")
    test_keratoconus_case_guardrails()
    print("  -> Passed!")
    
    print("\n[4/8] Testing Boundary Cutoff Evaluators...")
    test_cutoff_evaluator()
    print("  -> Passed!")
    
    print("\n[5/8] Testing Pydantic v2 Schema Serialization...")
    test_stage1_schema_serialization()
    print("  -> Passed!")
    
    print("\n[6/8] Testing End-to-End P-FIT Pipeline Orchestration...")
    test_full_pipeline_orchestration_mock()
    print("  -> Passed!")
    
    print("\n[7/8] Testing Deterministic PDF & Text Report Parser...")
    test_pentacam_pdf_text_parser()
    print("  -> Passed!")

    print("\n[8/8] Testing Bilateral Multi-Eye Pipeline & Asymmetry Evaluators...")
    test_bilateral_asymmetry_stats()
    test_multi_eye_pipeline_case6()
    print("  -> Passed!")
    
    print("\n" + "=" * 60)
    print("ALL 8 TEST SUITES PASSED SUCCESSFULLY (100% GREEN)")
    print("=" * 60)

if __name__ == "__main__":
    run_suite()

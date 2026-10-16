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

def run_suite():
    print("=" * 60)
    print("RUNNING P-FIT CORNEAL TOMOGRAPHY PIPELINE TEST SUITE")
    print("=" * 60)
    
    print("\n[1/7] Testing Normal Case Deterministic Guardrails...")
    test_normal_case_guardrails()
    print("  -> Passed!")
    
    print("\n[2/7] Testing Subclinical/Forme Fruste Guardrails...")
    test_subclinical_case_guardrails()
    print("  -> Passed!")
    
    print("\n[3/7] Testing Keratoconus Pattern Guardrails...")
    test_keratoconus_case_guardrails()
    print("  -> Passed!")
    
    print("\n[4/7] Testing Boundary Cutoff Evaluators...")
    test_cutoff_evaluator()
    print("  -> Passed!")
    
    print("\n[5/7] Testing Pydantic v2 Schema Serialization...")
    test_stage1_schema_serialization()
    print("  -> Passed!")
    
    print("\n[6/7] Testing End-to-End P-FIT Pipeline Orchestration...")
    test_full_pipeline_orchestration_mock()
    print("  -> Passed!")
    
    print("\n[7/7] Testing Deterministic PDF & Text Report Parser...")
    test_pentacam_pdf_text_parser()
    print("  -> Passed!")
    
    print("\n" + "=" * 60)
    print("ALL 7 TEST SUITES PASSED SUCCESSFULLY (100% GREEN)")
    print("=" * 60)

if __name__ == "__main__":
    run_suite()

"""
Unit Tests for Pydantic v2 Pipeline Schemas and Data Contracts.
"""
from src.schemas import (
    PentacamExtractionResult,
    ParietalIntegrationResult,
    FrontalScaffoldingDraft,
    FinalCalibratedPatientReport,
    CognitiveScaffoldingState,
    PhysicalAnalogy
)
from src.mock_data.sample_cases import SAMPLE_CASES
from src.pipeline.orchestrator import PFITPipelineOrchestrator


def test_stage1_schema_serialization():
    sample = SAMPLE_CASES["case_1_normal"]["data"]
    json_str = sample.model_dump_json()
    reconstructed = PentacamExtractionResult.model_validate_json(json_str)
    assert reconstructed.metadata.eye == "OD (Right Eye)"
    assert reconstructed.keratometry.kmax_diopters == 44.2


def test_full_pipeline_orchestration_mock():
    orchestrator = PFITPipelineOrchestrator(api_key=None)
    state = orchestrator.run_pipeline(mock_case_id="case_3_keratoconus")
    
    assert state.is_fully_executed is True
    assert state.stage1_extraction is not None
    assert state.stage2_parietal is not None
    assert state.stage3_frontal is not None
    assert state.stage4_acc is not None
    
    # Check ACC gating
    assert state.stage4_acc.inhibitory_gating_audit.is_safe_for_patient_delivery is True
    assert len(state.stage4_acc.consultation_questions) >= 3
    assert state.stage4_acc.factual_verification.alignment_score >= 0.95


if __name__ == "__main__":
    test_stage1_schema_serialization()
    test_full_pipeline_orchestration_mock()
    print("All schema and orchestration tests passed successfully!")

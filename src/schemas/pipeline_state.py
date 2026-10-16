"""
Full Pipeline State Schema.
Encapsulates data artifacts across all 4 stages of the P-FIT Cognitive Scaffolding Pipeline.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from src.schemas.stage1_extraction import PentacamExtractionResult
from src.schemas.stage2_parietal import ParietalIntegrationResult
from src.schemas.stage3_frontal import FrontalScaffoldingDraft
from src.schemas.stage4_acc import FinalCalibratedPatientReport


class StageTelemetry(BaseModel):
    stage_name: str
    duration_seconds: float
    status: str = "completed"
    model_used: str = ""
    error_message: Optional[str] = None


class CognitiveScaffoldingState(BaseModel):
    """End-to-end P-FIT pipeline state container."""
    scan_id: str
    source_filename: str
    source_image_base64: Optional[str] = None
    
    # P-FIT 4 Stages
    stage1_extraction: Optional[PentacamExtractionResult] = None
    stage2_parietal: Optional[ParietalIntegrationResult] = None
    stage3_frontal: Optional[FrontalScaffoldingDraft] = None
    stage4_acc: Optional[FinalCalibratedPatientReport] = None
    
    # Pipeline Execution Telemetry
    telemetry: List[StageTelemetry] = Field(default_factory=list)
    overall_confidence: float = 0.95
    is_fully_executed: bool = False
    
    def get_stage_status(self, stage_idx: int) -> str:
        if stage_idx == 1:
            return "Completed" if self.stage1_extraction else "Pending"
        elif stage_idx == 2:
            return "Completed" if self.stage2_parietal else "Pending"
        elif stage_idx == 3:
            return "Completed" if self.stage3_frontal else "Pending"
        elif stage_idx == 4:
            return "Completed" if self.stage4_acc else "Pending"
        return "Unknown"

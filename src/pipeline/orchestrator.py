"""
P-FIT Pipeline Orchestrator.
Coordinates the 4-stage cognitive-scaffolding pipeline execution from raw Pentacam ingestion to calibrated patient report.
"""
import time
import uuid
import logging
from typing import Optional, Callable, Dict, Any

from src.schemas.pipeline_state import CognitiveScaffoldingState, StageTelemetry
from src.schemas.stage1_extraction import PentacamExtractionResult
from src.pipeline.stage1_occipital import extract_pentacam_data, extract_multi_eye_pentacam_data
from src.pipeline.stage2_parietal import process_parietal_integration
from src.pipeline.stage3_frontal import generate_frontal_scaffolding
from src.pipeline.stage4_acc import process_acc_reflection
from src.config import DEFAULT_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)


class PFITPipelineOrchestrator:
    """Executes and monitors the 4-stage P-FIT cognitive scaffolding pipeline."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name

    def run_pipeline(
        self,
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
        source_filename: str = "pentacam_scan.png",
        mock_case_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Any]], None]] = None
    ) -> CognitiveScaffoldingState:
        """
        Runs all 4 stages sequentially with telemetry and callbacks for a single eye.
        """
        scan_id = str(uuid.uuid4())[:8]
        state = CognitiveScaffoldingState(
            scan_id=scan_id,
            source_filename=source_filename
        )

        # STAGE 1: Occipital-Temporal Proxy
        if progress_callback:
            progress_callback(15, "Stage 1: Extracting tomographic measurements & OCR...", None)
        
        t0 = time.perf_counter()
        stage1_result = extract_pentacam_data(
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            api_key=self.api_key,
            model_name=self.model_name,
            mock_case_id=mock_case_id
        )
        t1 = time.perf_counter()
        state.stage1_extraction = stage1_result
        state.telemetry.append(
            StageTelemetry(
                stage_name="Stage 1: Occipital-Temporal Proxy",
                duration_seconds=round(t1 - t0, 3),
                status="completed",
                model_used=self.model_name if (self.api_key and not mock_case_id) else "Deterministic Ground Truth"
            )
        )
        if progress_callback:
            progress_callback(30, "Stage 1 Completed: Parameters Extracted", stage1_result)

        # STAGE 2: Parietal Cortex Proxy
        if progress_callback:
            progress_callback(45, "Stage 2: Evaluating corneal structure & deterministic guardrails...", None)
        
        t0 = time.perf_counter()
        stage2_result = process_parietal_integration(
            extraction=stage1_result,
            api_key=self.api_key,
            model_name=self.model_name
        )
        t1 = time.perf_counter()
        state.stage2_parietal = stage2_result
        state.telemetry.append(
            StageTelemetry(
                stage_name="Stage 2: Parietal Cortex Proxy",
                duration_seconds=round(t1 - t0, 3),
                status="completed",
                model_used=self.model_name if self.api_key else "Deterministic Engine"
            )
        )
        if progress_callback:
            progress_callback(60, "Stage 2 Completed: Geometric Structure Formed", stage2_result)

        # STAGE 3: Frontal / Prefrontal Proxy
        if progress_callback:
            progress_callback(75, "Stage 3: Generating intuitive physical analogies & metaphors...", None)
        
        t0 = time.perf_counter()
        stage3_result = generate_frontal_scaffolding(
            extraction=stage1_result,
            parietal=stage2_result,
            api_key=self.api_key,
            model_name=self.model_name
        )
        t1 = time.perf_counter()
        state.stage3_frontal = stage3_result
        state.telemetry.append(
            StageTelemetry(
                stage_name="Stage 3: Frontal/Prefrontal Proxy",
                duration_seconds=round(t1 - t0, 3),
                status="completed",
                model_used=self.model_name if self.api_key else "Scaffolding Engine"
            )
        )
        if progress_callback:
            progress_callback(85, "Stage 3 Completed: Physical Models Crafted", stage3_result)

        # STAGE 4: Anterior Cingulate Cortex (ACC) Proxy
        if progress_callback:
            progress_callback(95, "Stage 4: Inhibitory gating, safety reflection & doctor questions...", None)
        
        t0 = time.perf_counter()
        stage4_result = process_acc_reflection(
            extraction=stage1_result,
            parietal=stage2_result,
            scaffolding=stage3_result,
            api_key=self.api_key,
            model_name=self.model_name
        )
        t1 = time.perf_counter()
        state.stage4_acc = stage4_result
        state.telemetry.append(
            StageTelemetry(
                stage_name="Stage 4: ACC Proxy",
                duration_seconds=round(t1 - t0, 3),
                status="completed",
                model_used=self.model_name if self.api_key else "Safety Reflection Engine"
            )
        )
        if progress_callback:
            progress_callback(100, "Stage 4 Completed: Calibrated Report Ready", stage4_result)

        state.is_fully_executed = True
        return state

    def run_multi_eye_pipeline(
        self,
        image_bytes: Optional[bytes] = None,
        image_mime_type: str = "image/png",
        source_filename: str = "pentacam_scan.png",
        mock_case_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str, Optional[Any]], None]] = None
    ) -> Dict[str, CognitiveScaffoldingState]:
        """
        Runs the full 4-stage P-FIT pipeline for all eyes identified in the scan report.
        Returns a dictionary mapping eye keys ("OD", "OS") -> CognitiveScaffoldingState.
        """
        if progress_callback:
            progress_callback(10, "Analyzing scan file and identifying examined eyes...", None)

        multi_extractions = extract_multi_eye_pentacam_data(
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            api_key=self.api_key,
            model_name=self.model_name,
            mock_case_id=mock_case_id
        )

        results: Dict[str, CognitiveScaffoldingState] = {}
        total_eyes = max(1, len(multi_extractions))

        for eye_idx, (eye_key, extraction) in enumerate(multi_extractions.items()):
            scan_id = str(uuid.uuid4())[:8]
            state = CognitiveScaffoldingState(
                scan_id=scan_id,
                source_filename=source_filename
            )
            state.stage1_extraction = extraction

            if progress_callback:
                pct = int(((eye_idx * 4 + 1) / (total_eyes * 4)) * 95)
                progress_callback(pct, f"[{eye_key}] Stage 1: Feature Extraction Completed", extraction)

            # STAGE 2: Parietal Cortex Proxy
            if progress_callback:
                pct = int(((eye_idx * 4 + 2) / (total_eyes * 4)) * 95)
                progress_callback(pct, f"[{eye_key}] Stage 2: Evaluating Structure & Safety Boundaries...", None)
            stage2_result = process_parietal_integration(
                extraction=extraction,
                api_key=self.api_key,
                model_name=self.model_name
            )
            state.stage2_parietal = stage2_result

            # STAGE 3: Frontal / Prefrontal Proxy
            if progress_callback:
                pct = int(((eye_idx * 4 + 3) / (total_eyes * 4)) * 95)
                progress_callback(pct, f"[{eye_key}] Stage 3: Generating Physical Analogies & Scaffolding...", None)
            stage3_result = generate_frontal_scaffolding(
                extraction=extraction,
                parietal=stage2_result,
                api_key=self.api_key,
                model_name=self.model_name
            )
            state.stage3_frontal = stage3_result

            # STAGE 4: Anterior Cingulate Cortex (ACC) Proxy
            if progress_callback:
                pct = int(((eye_idx * 4 + 4) / (total_eyes * 4)) * 95)
                progress_callback(pct, f"[{eye_key}] Stage 4: Safety Gating & Consultation Plan...", None)
            stage4_result = process_acc_reflection(
                extraction=extraction,
                parietal=stage2_result,
                scaffolding=stage3_result,
                api_key=self.api_key,
                model_name=self.model_name
            )
            state.stage4_acc = stage4_result
            state.is_fully_executed = True
            results[eye_key] = state

        if progress_callback:
            progress_callback(100, "Analysis complete! Assembling your personalized dashboard...", None)

        return results



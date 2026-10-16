"""
Stage 4: Anterior Cingulate Cortex (ACC) Proxy Schema.
Data contract for cognitive control, inhibitory gating, factual verification, and physician consultation synthesis.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class InhibitoryGatingAudit(BaseModel):
    """Audit trail of suppressed diagnostic assertions and softened anxiety triggers."""
    suppressed_proclamations: List[str] = Field(
        default_factory=list,
        description="Definitive diagnostic claims caught and rewritten (e.g. 'You have keratoconus' -> 'Scan indicates localized steepening')."
    )
    softened_jargon_terms: List[str] = Field(
        default_factory=list,
        description="Alarming or complex jargon replaced with supportive, objective framing."
    )
    is_safe_for_patient_delivery: bool = Field(
        default=True,
        description="Gating flag confirming content adheres to non-diagnostic educational standards."
    )


class FactualAlignmentVerification(BaseModel):
    """Verification that all quantitative references in the text match Stage 1/2 ground truth."""
    metrics_verified: List[str] = Field(
        default_factory=list,
        description="List of numeric metrics checked against OCR ground truth (e.g. Kmax, Thinnest Pachy, Posterior Elevation, BAD-D)."
    )
    mismatches_detected_and_corrected: List[str] = Field(
        default_factory=list,
        description="Any hallucinated numbers or unit errors corrected during the reflection pass."
    )
    alignment_score: float = Field(
        default=1.0,
        description="Factual alignment ratio (0.0 to 1.0; 1.0 = perfect alignment)."
    )


class DoctorConsultationQuestion(BaseModel):
    """Smart, tailored question designed for the patient to ask their eye care professional."""
    category: str = Field(
        description="Focus area: e.g., 'Monitoring & Progression', 'Visual Quality & Optics', 'Corneal Stability & Treatments', 'Daily Living & Habits'."
    )
    question_text: str = Field(
        description="The exact phrasing for the patient to use with their doctor."
    )
    rationale: str = Field(
        description="Why this question is relevant based on their specific scan findings."
    )


class FinalCalibratedPatientReport(BaseModel):
    """Calibrated, gated, and polished patient explanation report ready for presentation."""
    report_title: str = Field(
        default="Understanding Your Corneal Tomography (Pentacam) Scan",
        description="Calm, clear report header."
    )
    executive_summary: str = Field(
        description="2-paragraph friendly, reassuring summary grounded in physical analogies."
    )
    structural_deep_dive_markdown: str = Field(
        description="Full markdown narrative explaining Curvature, Thickness, and Back Surface with visual analogies."
    )
    color_map_guide_markdown: str = Field(
        description="Clear patient guide to reading their 4-Maps or BAD-III color displays."
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="3-4 bullet point key takeaways for the patient."
    )
    consultation_questions: List[DoctorConsultationQuestion] = Field(
        default_factory=list,
        description="Curated high-yield questions for the doctor's visit."
    )
    inhibitory_gating_audit: InhibitoryGatingAudit = Field(
        default_factory=InhibitoryGatingAudit,
        description="Audit record of cognitive control and safety gating."
    )
    factual_verification: FactualAlignmentVerification = Field(
        default_factory=FactualAlignmentVerification,
        description="Audit of metric verification against raw OCR data."
    )
    readability_grade_level: str = Field(
        default="Grade 8 (Accessible Layperson)",
        description="Calibrated reading ease level."
    )
    medical_disclaimer: str = Field(
        default=(
            "Important Notice: This cognitive scaffolding report is an educational synthesis designed to help "
            "you understand the geometric patterns in your scan. It is not a clinical diagnosis or medical prescription. "
            "Always discuss these findings and your personalized care plan with your ophthalmologist or cornea specialist."
        ),
        description="Standard protective medical disclaimer."
    )

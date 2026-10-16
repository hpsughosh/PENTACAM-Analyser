"""
High-Fidelity Sample Pentacam Tomography Clinical Cases.
Used for offline demonstrations, unit tests, and reference ground truths.
"""
from typing import Dict, Any
from src.schemas.stage1_extraction import (
    PentacamExtractionResult,
    ExaminationMetadata,
    KeratometryIndices,
    PachymetryIndices,
    ElevationIndices,
    BelinAmbrosioIndices,
    TopometricIndices
)


SAMPLE_CASES: Dict[str, Dict[str, Any]] = {
    "case_1_normal": {
        "title": "Case 1: Normal Physiologic Cornea (Healthy Symmetrical Eye)",
        "description": "Standard adult cornea with gentle prolate curvature, normal pachymetry (>540µm), and baseline posterior elevation (+7µm).",
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-03-15",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=43.1,
                k1_axis_deg=175.0,
                k2_steep_diopters=43.9,
                k2_axis_deg=85.0,
                km_mean_diopters=43.5,
                kmax_diopters=44.2,
                astigmatism_diopters=0.8,
                q_value_asphericity=-0.22,
                rmin_curvature_radius_mm=7.64
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=552.0,
                thinnest_pachymetry_um=548.0,
                pupil_center_pachymetry_um=550.0,
                thinnest_loc_x_mm=0.12,
                thinnest_loc_y_mm=-0.10,
                pachy_progression_index_avg=0.92,
                pachy_progression_index_max=1.07,
                art_max_um=512.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=2.0,
                posterior_elevation_thinnest_um=7.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=1.0,
                enhanced_delta_back_elevation_um=2.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=0.62,
                d_b_back_elevation=0.74,
                d_t_thinnest_pachy=0.51,
                d_a_pachy_progression=0.88,
                d_p_displacement=0.45,
                final_bad_d_index=0.84
            ),
            topometric=TopometricIndices(
                isv_surface_variance=18.0,
                iva_vertical_asymmetry=0.12,
                ki_keratoconus_index=1.01,
                cki_center_keratoconus_index=1.00,
                ihd_height_decentration=0.006
            ),
            raw_ocr_notes="Pentacam HR 4-Maps Refractive + BAD-III Display. Scan quality specification: OK. All indices within normal limits (Green display).",
            extraction_confidence_score=0.98
        )
    },
    "case_2_subclinical": {
        "title": "Case 2: Subclinical / Forme Fruste Keratoconus (Early Borderline Patterns)",
        "description": "Subtle early posterior elevation (+15µm) and borderline BAD-D (2.15 SD, yellow display) with mild inferior thinning.",
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OS (Left Eye)",
                exam_date="2026-04-10",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=44.2,
                k1_axis_deg=160.0,
                k2_steep_diopters=45.8,
                k2_axis_deg=70.0,
                km_mean_diopters=45.0,
                kmax_diopters=47.6,
                astigmatism_diopters=1.6,
                q_value_asphericity=-0.42,
                rmin_curvature_radius_mm=7.09
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=504.0,
                thinnest_pachymetry_um=492.0,
                pupil_center_pachymetry_um=502.0,
                thinnest_loc_x_mm=-0.22,
                thinnest_loc_y_mm=-0.54,
                pachy_progression_index_avg=1.31,
                pachy_progression_index_max=1.42,
                art_max_um=375.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=6.5,
                posterior_elevation_thinnest_um=15.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=5.5,
                enhanced_delta_back_elevation_um=9.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=1.75,
                d_b_back_elevation=2.10,
                d_t_thinnest_pachy=1.62,
                d_a_pachy_progression=1.85,
                d_p_displacement=1.92,
                final_bad_d_index=2.15
            ),
            topometric=TopometricIndices(
                isv_surface_variance=39.0,
                iva_vertical_asymmetry=0.29,
                ki_keratoconus_index=1.06,
                cki_center_keratoconus_index=1.02,
                ihd_height_decentration=0.016
            ),
            raw_ocr_notes="Pentacam BAD-III display shows multiple yellow borderline alerts: Db, Da, Dp elevated, final BAD-D at 2.15 SD.",
            extraction_confidence_score=0.96
        )
    },
    "case_3_keratoconus": {
        "title": "Case 3: Keratoconus with Inferior Steep Cone (Focal Thinning & Elevation)",
        "description": "Classic keratoconus profile with Kmax 53.8 D, significant pachymetric thinning (436µm), posterior elevation +28µm, and BAD-D 4.82 SD (Red display).",
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-05-02",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=46.5,
                k1_axis_deg=15.0,
                k2_steep_diopters=51.4,
                k2_axis_deg=105.0,
                km_mean_diopters=48.95,
                kmax_diopters=53.8,
                astigmatism_diopters=4.9,
                q_value_asphericity=-0.88,
                rmin_curvature_radius_mm=6.28
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=462.0,
                thinnest_pachymetry_um=436.0,
                pupil_center_pachymetry_um=460.0,
                thinnest_loc_x_mm=0.35,
                thinnest_loc_y_mm=-0.85,
                pachy_progression_index_avg=1.98,
                pachy_progression_index_max=2.25,
                art_max_um=220.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=14.0,
                posterior_elevation_thinnest_um=28.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=12.0,
                enhanced_delta_back_elevation_um=19.5
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=3.95,
                d_b_back_elevation=4.80,
                d_t_thinnest_pachy=3.10,
                d_a_pachy_progression=4.15,
                d_p_displacement=3.85,
                final_bad_d_index=4.82
            ),
            topometric=TopometricIndices(
                isv_surface_variance=74.0,
                iva_vertical_asymmetry=0.68,
                ki_keratoconus_index=1.28,
                cki_center_keratoconus_index=1.08,
                ihd_height_decentration=0.048
            ),
            raw_ocr_notes="Pentacam 4-Maps Refractive + BAD-III. Clear inferior-temporal steep cone on sagittal map, thinnest point co-localized at -0.85mm inferiorly. All BAD parameters in Red alert.",
            extraction_confidence_score=0.99
        )
    },
    "case_4_post_lasik": {
        "title": "Case 4: Post-Myopic Laser Vision Correction (Oblate Profile)",
        "description": "Stable post-myopic LASIK profile with flat central curvature (38.2 D), positive Q-value (+0.48 oblate), and stable posterior elevation (+8µm).",
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-06-18",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=37.8,
                k1_axis_deg=180.0,
                k2_steep_diopters=38.6,
                k2_axis_deg=90.0,
                km_mean_diopters=38.2,
                kmax_diopters=39.1,
                astigmatism_diopters=0.8,
                q_value_asphericity=0.48,
                rmin_curvature_radius_mm=8.63
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=448.0,
                thinnest_pachymetry_um=445.0,
                pupil_center_pachymetry_um=448.0,
                thinnest_loc_x_mm=0.08,
                thinnest_loc_y_mm=-0.05,
                pachy_progression_index_avg=1.02,
                pachy_progression_index_max=1.14,
                art_max_um=420.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=-3.0,
                posterior_elevation_thinnest_um=8.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=-2.0,
                enhanced_delta_back_elevation_um=3.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=0.85,
                d_b_back_elevation=0.92,
                d_t_thinnest_pachy=1.45,
                d_a_pachy_progression=0.95,
                d_p_displacement=0.38,
                final_bad_d_index=1.35
            ),
            topometric=TopometricIndices(
                isv_surface_variance=22.0,
                iva_vertical_asymmetry=0.15,
                ki_keratoconus_index=1.02,
                cki_center_keratoconus_index=1.01,
                ihd_height_decentration=0.009
            ),
            raw_ocr_notes="Post-refractive central flattening. Front elevation shows negative central depression relative to pre-op sphere, posterior elevation remains within normal limits.",
            extraction_confidence_score=0.97
        )
    },
    "case_5_od_ectasia_46k": {
        "title": "Case 5: Right Eye (OD) Pentacam Scan (Kmax 46.6 D, Thin 456µm, Elev +33µm, BAD-D 5.34)",
        "description": "Right eye report showing moderate focal thinning (456 µm at thinnest location), marked posterior BFS elevation (+33 µm at thinnest, up to +42 µm peak), Kmax 46.6 D, and BAD-D 5.34 SD.",
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-06-20",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=43.8,
                k1_axis_deg=170.0,
                k2_steep_diopters=46.1,
                k2_axis_deg=80.0,
                km_mean_diopters=44.95,
                kmax_diopters=46.6,
                astigmatism_diopters=2.3,
                q_value_asphericity=-0.55,
                rmin_curvature_radius_mm=7.24
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=472.0,
                thinnest_pachymetry_um=456.0,
                pupil_center_pachymetry_um=470.0,
                thinnest_loc_x_mm=0.25,
                thinnest_loc_y_mm=-0.65,
                pachy_progression_index_avg=1.65,
                pachy_progression_index_max=1.85,
                art_max_um=276.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=11.0,
                posterior_elevation_thinnest_um=33.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=8.5,
                enhanced_delta_back_elevation_um=15.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=2.85,
                d_b_back_elevation=4.60,
                d_t_thinnest_pachy=2.70,
                d_a_pachy_progression=3.90,
                d_p_displacement=3.30,
                final_bad_d_index=5.34
            ),
            topometric=TopometricIndices(
                isv_surface_variance=62.0,
                iva_vertical_asymmetry=0.52,
                ki_keratoconus_index=1.18,
                cki_center_keratoconus_index=1.05,
                ihd_height_decentration=0.038
            ),
            raw_ocr_notes="Right eye (OD) Pentacam report. Front Sagittal Kmax 46.6 D, Thinnest location 456 µm, Posterior Elevation (B.Ele.Th) +33 µm, BAD-D Final 5.34 SD.",
            extraction_confidence_score=0.99
        )
    },
    "case_6_bilateral_asymmetric": {
        "title": "Case 6: Bilateral Pentacam Report (OD: Right Eye Ectasia + OS: Left Eye Normal)",
        "description": "Complete bilateral report containing both Right Eye (OD: Kmax 46.6 D, Thin 456µm, BAD-D 5.34) and Left Eye (OS: Kmax 43.5 D, Thin 550µm, BAD-D 0.84) for side-by-side comparison.",
        "is_bilateral": True,
        "data_od": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-06-20",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=43.8,
                k1_axis_deg=170.0,
                k2_steep_diopters=46.1,
                k2_axis_deg=80.0,
                km_mean_diopters=44.95,
                kmax_diopters=46.6,
                astigmatism_diopters=2.3,
                q_value_asphericity=-0.55,
                rmin_curvature_radius_mm=7.24
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=472.0,
                thinnest_pachymetry_um=456.0,
                pupil_center_pachymetry_um=470.0,
                thinnest_loc_x_mm=0.25,
                thinnest_loc_y_mm=-0.65,
                pachy_progression_index_avg=1.65,
                pachy_progression_index_max=1.85,
                art_max_um=276.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=11.0,
                posterior_elevation_thinnest_um=33.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=8.5,
                enhanced_delta_back_elevation_um=15.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=2.85,
                d_b_back_elevation=4.60,
                d_t_thinnest_pachy=2.70,
                d_a_pachy_progression=3.90,
                d_p_displacement=3.30,
                final_bad_d_index=5.34
            ),
            topometric=TopometricIndices(
                isv_surface_variance=62.0,
                iva_vertical_asymmetry=0.52,
                ki_keratoconus_index=1.18,
                cki_center_keratoconus_index=1.05,
                ihd_height_decentration=0.038
            ),
            raw_ocr_notes="Right eye (OD) Pentacam report. Front Sagittal Kmax 46.6 D, Thinnest location 456 µm, Posterior Elevation +33 µm, BAD-D Final 5.34 SD.",
            extraction_confidence_score=0.99
        ),
        "data_os": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OS (Left Eye)",
                exam_date="2026-06-20",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=43.1,
                k1_axis_deg=175.0,
                k2_steep_diopters=43.9,
                k2_axis_deg=85.0,
                km_mean_diopters=43.5,
                kmax_diopters=44.1,
                astigmatism_diopters=0.8,
                q_value_asphericity=-0.22,
                rmin_curvature_radius_mm=7.65
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=554.0,
                thinnest_pachymetry_um=550.0,
                pupil_center_pachymetry_um=552.0,
                thinnest_loc_x_mm=0.10,
                thinnest_loc_y_mm=-0.08,
                pachy_progression_index_avg=0.94,
                pachy_progression_index_max=1.05,
                art_max_um=520.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=2.0,
                posterior_elevation_thinnest_um=7.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=1.0,
                enhanced_delta_back_elevation_um=2.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=0.60,
                d_b_back_elevation=0.72,
                d_t_thinnest_pachy=0.50,
                d_a_pachy_progression=0.85,
                d_p_displacement=0.42,
                final_bad_d_index=0.84
            ),
            topometric=TopometricIndices(
                isv_surface_variance=17.0,
                iva_vertical_asymmetry=0.11,
                ki_keratoconus_index=1.01,
                cki_center_keratoconus_index=1.00,
                ihd_height_decentration=0.005
            ),
            raw_ocr_notes="Left eye (OS) Pentacam report. Normal symmetric cornea with intact thickness and baseline elevation.",
            extraction_confidence_score=0.98
        ),
        "data": PentacamExtractionResult(
            metadata=ExaminationMetadata(
                eye="OD (Right Eye)",
                exam_date="2026-06-20",
                scan_quality="OK",
                device_model="Oculus Pentacam HR",
                software_version="1.21r43"
            ),
            keratometry=KeratometryIndices(
                k1_flat_diopters=43.8,
                k1_axis_deg=170.0,
                k2_steep_diopters=46.1,
                k2_axis_deg=80.0,
                km_mean_diopters=44.95,
                kmax_diopters=46.6,
                astigmatism_diopters=2.3,
                q_value_asphericity=-0.55,
                rmin_curvature_radius_mm=7.24
            ),
            pachymetry=PachymetryIndices(
                central_corneal_thickness_cct_um=472.0,
                thinnest_pachymetry_um=456.0,
                pupil_center_pachymetry_um=470.0,
                thinnest_loc_x_mm=0.25,
                thinnest_loc_y_mm=-0.65,
                pachy_progression_index_avg=1.65,
                pachy_progression_index_max=1.85,
                art_max_um=276.0
            ),
            elevation=ElevationIndices(
                anterior_elevation_thinnest_um=11.0,
                posterior_elevation_thinnest_um=33.0,
                reference_surface_type="8.0mm Best Fit Sphere (BFS)",
                enhanced_delta_front_elevation_um=8.5,
                enhanced_delta_back_elevation_um=15.0
            ),
            bad_display=BelinAmbrosioIndices(
                d_f_front_elevation=2.85,
                d_b_back_elevation=4.60,
                d_t_thinnest_pachy=2.70,
                d_a_pachy_progression=3.90,
                d_p_displacement=3.30,
                final_bad_d_index=5.34
            ),
            topometric=TopometricIndices(
                isv_surface_variance=62.0,
                iva_vertical_asymmetry=0.52,
                ki_keratoconus_index=1.18,
                cki_center_keratoconus_index=1.05,
                ihd_height_decentration=0.038
            ),
            raw_ocr_notes="Right eye (OD) Pentacam report. Front Sagittal Kmax 46.6 D, Thinnest location 456 µm, Posterior Elevation +33 µm, BAD-D Final 5.34 SD.",
            extraction_confidence_score=0.99
        )
    }
}

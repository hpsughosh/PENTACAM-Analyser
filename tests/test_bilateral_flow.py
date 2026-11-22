"""
Unit tests for Bilateral Comparison, Multi-Eye Pipeline, and Asymmetry Calculations.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import math
from src.utils.asymmetry import compute_asymmetry_stats
from src.mock_data.sample_cases import SAMPLE_CASES
from src.pipeline.orchestrator import PFITPipelineOrchestrator
from src.visualization.corneal_maps import (
    create_3d_curvature_figure,
    create_2d_topography_heatmap,
    create_3d_pachymetry_figure,
    create_2d_pachymetry_heatmap,
    create_3d_elevation_figure,
    create_2d_elevation_heatmap,
    create_3d_badd_figure,
    create_bad_d_gauge_figure,
)


def test_bilateral_asymmetry_stats():
    case6 = SAMPLE_CASES["case_6_bilateral_asymmetric"]
    ext_od = case6["data_od"]
    ext_os = case6["data_os"]

    asym = compute_asymmetry_stats(ext_od, ext_os)

    assert math.isclose(asym["kmax_od"], 46.6, abs_tol=0.1)
    assert math.isclose(asym["kmax_os"], 44.1, abs_tol=0.1)
    assert math.isclose(asym["delta_kmax"], 2.5, abs_tol=0.1)
    assert asym["k_status"] == "Significant"

    assert asym["thin_od"] == 456.0
    assert asym["thin_os"] == 550.0
    assert asym["delta_thin"] == 94.0
    assert asym["t_status"] == "Significant"

    assert math.isclose(asym["elev_od"], 33.0, abs_tol=0.1)
    assert math.isclose(asym["elev_os"], 7.0, abs_tol=0.1)
    assert math.isclose(asym["delta_elev"], 26.0, abs_tol=0.1)
    assert asym["e_status"] == "Significant"

    assert math.isclose(asym["badd_od"], 5.34, abs_tol=0.05)
    assert math.isclose(asym["badd_os"], 0.84, abs_tol=0.05)
    assert math.isclose(asym["delta_badd"], 4.50, abs_tol=0.05)
    assert asym["b_status"] == "Significant"

    assert asym["has_significant"] is True
    print("  -> test_bilateral_asymmetry_stats: Passed!")


def test_multi_eye_pipeline_case6():
    orchestrator = PFITPipelineOrchestrator(api_key=None)
    states = orchestrator.run_multi_eye_pipeline(mock_case_id="case_6_bilateral_asymmetric")

    assert "OD" in states
    assert "OS" in states
    assert len(states) == 2

    state_od = states["OD"]
    state_os = states["OS"]

    assert "OD" in state_od.stage1_extraction.metadata.eye
    assert "OS" in state_os.stage1_extraction.metadata.eye

    assert state_od.stage1_extraction.keratometry.kmax_diopters == 46.6
    assert state_os.stage1_extraction.keratometry.kmax_diopters == 44.1

    # Verify visual figure generations for both eyes
    fig_curv_od = create_3d_curvature_figure(state_od.stage1_extraction)
    fig_curv_os = create_3d_curvature_figure(state_os.stage1_extraction)
    assert fig_curv_od is not None
    assert fig_curv_os is not None

    fig_2d_curv_od = create_2d_topography_heatmap(state_od.stage1_extraction, eye_label="OD")
    fig_2d_curv_os = create_2d_topography_heatmap(state_os.stage1_extraction, eye_label="OS")
    assert "OD" in fig_2d_curv_od.layout.title.text
    assert "OS" in fig_2d_curv_os.layout.title.text

    fig_pachy_od = create_3d_pachymetry_figure(state_od.stage1_extraction)
    fig_pachy_os = create_3d_pachymetry_figure(state_os.stage1_extraction)
    assert fig_pachy_od is not None
    assert fig_pachy_os is not None

    fig_gauge_od = create_bad_d_gauge_figure(state_od.stage1_extraction.bad_display.final_bad_d_index, eye_label="OD")
    fig_gauge_os = create_bad_d_gauge_figure(state_os.stage1_extraction.bad_display.final_bad_d_index, eye_label="OS")
    assert "OD" in fig_gauge_od.data[0].title.text
    assert "OS" in fig_gauge_os.data[0].title.text

    print("  -> test_multi_eye_pipeline_case6: Passed!")


if __name__ == "__main__":
    test_bilateral_asymmetry_stats()
    test_multi_eye_pipeline_case6()
    print("ALL BILATERAL TESTS PASSED SUCCESSFULLY!")

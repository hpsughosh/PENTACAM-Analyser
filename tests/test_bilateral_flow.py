import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pdf_parser import parse_multi_eye_pentacam_pdf
from src.pipeline.stage1_occipital import extract_multi_eye_pentacam_data
from src.pipeline.pdf_parser import parse_multi_eye_pentacam_pdf
from src.pipeline.stage1_occipital import extract_multi_eye_pentacam_data
from src.pipeline.orchestrator import PFITPipelineOrchestrator
from src.mock_data.sample_cases import SAMPLE_CASES

def test_bilateral_flow():
    print("Testing Case 6 Bilateral Flow...")
    case_6 = SAMPLE_CASES["case_6_bilateral_asymmetric"]
    assert case_6["is_bilateral"] is True
    assert "data_od" in case_6 and "data_os" in case_6
    
    od_ext = case_6["data_od"]
    os_ext = case_6["data_os"]
    
    print(f"OD Eye: {od_ext.metadata.eye}, Kmax={od_ext.keratometry.kmax_diopters} D, Thin={od_ext.pachymetry.thinnest_pachymetry_um} um")
    print(f"OS Eye: {os_ext.metadata.eye}, Kmax={os_ext.keratometry.kmax_diopters} D, Thin={os_ext.pachymetry.thinnest_pachymetry_um} um")
    
    # Run multi-eye pipeline
    orchestrator = PFITPipelineOrchestrator()
    states = orchestrator.run_multi_eye_pipeline(
        mock_case_id="case_6_bilateral_asymmetric"
    )
    
    assert "OD" in states and "OS" in states
    state_od = states["OD"]
    state_os = states["OS"]
    
    print(f"OD Risk Tier: {state_od.stage2_parietal.clinical_risk_tier.value}")
    print(f"OS Risk Tier: {state_os.stage2_parietal.clinical_risk_tier.value}")
    
    # Verify asymmetry calculation
    delta_kmax = abs((od_ext.keratometry.kmax_diopters or 0) - (os_ext.keratometry.kmax_diopters or 0))
    delta_thin = abs((od_ext.pachymetry.thinnest_pachymetry_um or 0) - (os_ext.pachymetry.thinnest_pachymetry_um or 0))
    delta_elev = abs((od_ext.elevation.posterior_elevation_thinnest_um or 0) - (os_ext.elevation.posterior_elevation_thinnest_um or 0))
    delta_badd = abs((od_ext.bad_display.final_bad_d_index or 0) - (os_ext.bad_display.final_bad_d_index or 0))
    
    print(f"Calculated Delta Kmax: {delta_kmax:.2f} D")
    print(f"Calculated Delta Thinnest: {delta_thin:.0f} um")
    print(f"Calculated Delta Post Elev: {delta_elev:.1f} um")
    print(f"Calculated Delta BAD-D: {delta_badd:.2f} SD")
    
    assert delta_kmax > 1.2, "Expected Kmax delta > 1.2 D for asymmetric case"
    assert delta_thin > 30, "Expected Thin delta > 30 um for asymmetric case"
    assert delta_badd > 1.0, "Expected BAD-D delta > 1.0 SD for asymmetric case"
    
    print("Bilateral simulation test passed successfully!")

if __name__ == "__main__":
    test_bilateral_flow()


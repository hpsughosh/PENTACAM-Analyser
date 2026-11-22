"""
Corneal Inter-Eye Asymmetry Evaluation Module.
Computes bilateral disparities (OD vs OS) and evaluates against clinical thresholds.
"""
from typing import Dict, Any
from src.schemas.stage1_extraction import PentacamExtractionResult


def compute_asymmetry_stats(ext_od: PentacamExtractionResult, ext_os: PentacamExtractionResult) -> Dict[str, Any]:
    """
    Computes inter-eye asymmetry (OD vs OS) and evaluates against clinical consensus cutoffs.
    
    Clinical Thresholds:
    - Kmax: normal < 1.0 D, borderline 1.0 - 1.5 D, significant >= 1.5 D
    - Thinnest Pachymetry: normal < 20 um, borderline 20 - 30 um, significant >= 30 um
    - Posterior Elevation: normal < 5.0 um, borderline 5.0 - 8.0 um, significant >= 8.0 um
    - BAD-D Index: normal < 0.50 SD, borderline 0.50 - 1.0 SD, significant >= 1.0 SD
    """
    kmax_od = ext_od.keratometry.kmax_diopters or 0.0
    kmax_os = ext_os.keratometry.kmax_diopters or 0.0
    delta_kmax = abs(kmax_od - kmax_os)

    thin_od = ext_od.pachymetry.thinnest_pachymetry_um or 0.0
    thin_os = ext_os.pachymetry.thinnest_pachymetry_um or 0.0
    delta_thin = abs(thin_od - thin_os)

    elev_od = ext_od.elevation.posterior_elevation_thinnest_um or 0.0
    elev_os = ext_os.elevation.posterior_elevation_thinnest_um or 0.0
    delta_elev = abs(elev_od - elev_os)

    badd_od = ext_od.bad_display.final_bad_d_index or 0.0
    badd_os = ext_os.bad_display.final_bad_d_index or 0.0
    delta_badd = abs(badd_od - badd_os)

    # Kmax Asymmetry
    if delta_kmax >= 1.5:
        k_status, k_cls, k_icon = "Significant", "status-pill-abnormal", "🔴"
    elif delta_kmax >= 1.0:
        k_status, k_cls, k_icon = "Borderline", "status-pill-borderline", "🟡"
    else:
        k_status, k_cls, k_icon = "Symmetric", "status-pill-normal", "🟢"

    # Thickness Asymmetry
    if delta_thin >= 30.0:
        t_status, t_cls, t_icon = "Significant", "status-pill-abnormal", "🔴"
    elif delta_thin >= 20.0:
        t_status, t_cls, t_icon = "Borderline", "status-pill-borderline", "🟡"
    else:
        t_status, t_cls, t_icon = "Symmetric", "status-pill-normal", "🟢"

    # Elevation Asymmetry
    if delta_elev >= 8.0:
        e_status, e_cls, e_icon = "Significant", "status-pill-abnormal", "🔴"
    elif delta_elev >= 5.0:
        e_status, e_cls, e_icon = "Borderline", "status-pill-borderline", "🟡"
    else:
        e_status, e_cls, e_icon = "Symmetric", "status-pill-normal", "🟢"

    # BAD-D Asymmetry
    if delta_badd >= 1.0:
        b_status, b_cls, b_icon = "Significant", "status-pill-abnormal", "🔴"
    elif delta_badd >= 0.5:
        b_status, b_cls, b_icon = "Borderline", "status-pill-borderline", "🟡"
    else:
        b_status, b_cls, b_icon = "Symmetric", "status-pill-normal", "🟢"

    has_significant = any(s == "Significant" for s in [k_status, t_status, e_status, b_status])
    has_borderline = any(s == "Borderline" for s in [k_status, t_status, e_status, b_status])

    return {
        "kmax_od": kmax_od, "kmax_os": kmax_os, "delta_kmax": delta_kmax, "k_status": k_status, "k_cls": k_cls, "k_icon": k_icon,
        "thin_od": thin_od, "thin_os": thin_os, "delta_thin": delta_thin, "t_status": t_status, "t_cls": t_cls, "t_icon": t_icon,
        "elev_od": elev_od, "elev_os": elev_os, "delta_elev": delta_elev, "e_status": e_status, "e_cls": e_cls, "e_icon": e_icon,
        "badd_od": badd_od, "badd_os": badd_os, "delta_badd": delta_badd, "b_status": b_status, "b_cls": b_cls, "b_icon": b_icon,
        "has_significant": has_significant, "has_borderline": has_borderline
    }

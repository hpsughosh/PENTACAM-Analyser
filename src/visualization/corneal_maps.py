"""
Corneal Visualization Module using Plotly.
Generates interactive, metric-tailored 3D surface meshes and 2D tomographic maps.
Styled in Editorial Royal Cobalt Navy & Clean White (Studio Aesthetic).
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from src.schemas.stage1_extraction import PentacamExtractionResult


# Editorial Royal Navy & Clean White Constants
BG_COLOR = "#FFFFFF"
CARD_BG = "#F8F9FA"
NAVY_PRIMARY = "#1829B0"
TEXT_COLOR = "#0F172A"
MUTED_TEXT = "#64748B"
GRID_COLOR = "#E2E8F0"


def create_3d_curvature_figure(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False
) -> go.Figure:
    """
    Creates an interactive 3D surface mesh of the anterior corneal surface
    with curvature power (Diopters) mapped across the dome.
    """
    if reference_mode:
        kmax = 43.8
        km = 43.5
        thinnest_x = 0.0
        thinnest_y = 0.0
        cone_amp = 0.0
        title_suffix = "Healthy Reference Model (Symmetric)"
    else:
        kmax = extraction.keratometry.kmax_diopters or 44.0
        km = extraction.keratometry.km_mean_diopters or 43.5
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        cone_amp = max(0.0, (kmax - 47.0) * 0.04) if kmax > 47.0 else 0.0
        title_suffix = f"Patient Scan ({extraction.metadata.eye or 'Eye'})"

    # 8mm optical zone mesh
    r = np.linspace(0, 4.0, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    r0 = 337.5 / max(30.0, km)
    cone_dist_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    cone_sag = cone_amp * np.exp(-cone_dist_sq / 1.5)
    base_sag = (R**2) / (2 * r0)
    Z = -(base_sag + cone_sag)

    Curvature_Map = km + (kmax - km) * np.exp(-cone_dist_sq / 1.8)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=Curvature_Map,
                colorscale="Jet",
                colorbar=dict(
                    title=dict(text="Curvature (D)", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    len=0.75,
                    thickness=14
                ),
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor=NAVY_PRIMARY,
                    project_z=True
                ),
                hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Height: %{z:.3f}mm<br>Curvature: %{surfacecolor:.1f} D<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"3D Anterior Curvature Surface • {title_suffix}",
            font=dict(size=14, color=NAVY_PRIMARY),
            x=0.04
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            zaxis=dict(title=dict(text="Sag Height (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.95))
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=420
    )
    return fig


def create_3d_corneal_dome_figure(extraction: PentacamExtractionResult) -> go.Figure:
    """Alias for backwards compatibility with tests and callers."""
    return create_3d_curvature_figure(extraction, reference_mode=False)


def create_2d_topography_heatmap(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False,
    eye_label: str = ""
) -> go.Figure:
    """Creates a 2D Sagittal Curvature Heatmap resembling Pentacam front map."""
    if reference_mode:
        kmax = 43.8
        km = 43.5
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference"
    else:
        kmax = extraction.keratometry.kmax_diopters or 44.0
        km = extraction.keratometry.km_mean_diopters or 43.5
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        lbl = eye_label or (extraction.metadata.eye if extraction.metadata and extraction.metadata.eye else "")
        title_suffix = f"Patient Scan ({lbl})" if lbl else "Patient Scan"

    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    dist_cone_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    cone_steepening = (kmax - km) * np.exp(-dist_cone_sq / 1.8)
    Z = km + cone_steepening
    Z[R > 4.0] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale="Jet",
            zmin=max(35.0, km - 4),
            zmax=max(48.0, kmax + 2),
            colorbar=dict(
                title=dict(text="Curvature (D)", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR)
            ),
            hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Curvature: %{z:.2f} D<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[thinnest_x],
            y=[thinnest_y],
            mode="markers+text",
            marker=dict(symbol="cross", size=10, color="white", line=dict(width=2, color=NAVY_PRIMARY)),
            text=[f"Kmax {kmax:.1f}D"],
            textposition="top right",
            textfont=dict(color=NAVY_PRIMARY, size=11, family="Arial Black"),
            name="Kmax Apex",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text=f"Sagittal Curvature Map (2D) • {title_suffix}", font=dict(size=14, color=NAVY_PRIMARY)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1", scaleanchor="x"),
        margin=dict(l=10, r=10, b=10, t=35),
        height=320
    )
    return fig


def create_3d_pachymetry_figure(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False
) -> go.Figure:
    """
    Creates an interactive 3D corneal thickness model showing concentric
    peripheral thickening and central / focal thinning.
    """
    if reference_mode:
        cct = 550.0
        thinnest = 545.0
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference Model"
    else:
        cct = extraction.pachymetry.central_corneal_thickness_cct_um or 540.0
        thinnest = extraction.pachymetry.thinnest_pachymetry_um or 535.0
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        title_suffix = f"Patient Scan ({extraction.metadata.eye or 'Eye'})"

    r = np.linspace(0, 4.0, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    peripheral_gain = 16.0 * (R**1.6)
    dist_thin_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    thin_depression = (cct - thinnest) * np.exp(-dist_thin_sq / 1.5)
    Thickness_Map = cct + peripheral_gain - thin_depression

    Z = -(R**2) / 16.0 - (650.0 - Thickness_Map) * 0.001

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=Thickness_Map,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="Thickness (µm)", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    len=0.75,
                    thickness=14
                ),
                contours_z=dict(show=True, usecolormap=True, project_z=True),
                hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Thickness: %{surfacecolor:.0f} µm<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"3D Corneal Thickness Profile • {title_suffix}",
            font=dict(size=14, color=NAVY_PRIMARY),
            x=0.04
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            zaxis=dict(title=dict(text="Depth (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.95))
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=420
    )
    return fig


def create_2d_pachymetry_heatmap(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False,
    eye_label: str = ""
) -> go.Figure:
    """Creates a 2D Pachymetric Thickness Distribution Heatmap."""
    if reference_mode:
        cct = 550.0
        thinnest = 545.0
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference"
    else:
        cct = extraction.pachymetry.central_corneal_thickness_cct_um or 540.0
        thinnest = extraction.pachymetry.thinnest_pachymetry_um or 535.0
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        lbl = eye_label or (extraction.metadata.eye if extraction.metadata and extraction.metadata.eye else "")
        title_suffix = f"Patient Scan ({lbl})" if lbl else "Patient Scan"

    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    peripheral_thickening = 15.0 * (R**1.6)
    dist_thin_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    thin_depression = (cct - thinnest) * np.exp(-dist_thin_sq / 1.5)
    
    Z = cct + peripheral_thickening - thin_depression
    Z[R > 4.0] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale="Viridis",
            zmin=max(380, thinnest - 20),
            zmax=700,
            colorbar=dict(
                title=dict(text="Thickness (µm)", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR)
            ),
            hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Thickness: %{z:.0f} µm<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[thinnest_x],
            y=[thinnest_y],
            mode="markers+text",
            marker=dict(symbol="star", size=11, color="#EF4444", line=dict(width=1.5, color=NAVY_PRIMARY)),
            text=[f"Thinnest {thinnest:.0f}µm"],
            textposition="bottom right",
            textfont=dict(color="#DC2626", size=11, family="Arial Black"),
            name="Thinnest Point",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text=f"Pachymetry Distribution Map (2D) • {title_suffix}", font=dict(size=14, color=NAVY_PRIMARY)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1", scaleanchor="x"),
        margin=dict(l=10, r=10, b=10, t=35),
        height=320
    )
    return fig


def create_3d_elevation_figure(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False
) -> go.Figure:
    """
    Creates an interactive 3D model of posterior elevation relative to Best Fit Sphere (BFS).
    """
    if reference_mode:
        post_elev = 6.0
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference Model"
    else:
        post_elev = extraction.elevation.posterior_elevation_thinnest_um or 8.0
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        title_suffix = f"Patient Scan ({extraction.metadata.eye or 'Eye'})"

    r = np.linspace(0, 4.0, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    dist_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    cone_elev = post_elev * np.exp(-dist_sq / 1.6)
    Elevation_Map = cone_elev - (R**1.3) * 1.5

    Z = -(R**2) / 14.0 + (Elevation_Map * 0.003)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=Elevation_Map,
                colorscale="Spectral_r",
                colorbar=dict(
                    title=dict(text="Back Elev (µm)", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    len=0.75,
                    thickness=14
                ),
                contours_z=dict(show=True, usecolormap=True, project_z=True),
                hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Back Elevation: +%{surfacecolor:.1f} µm<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"3D Posterior Elevation Map (BFS) • {title_suffix}",
            font=dict(size=14, color=NAVY_PRIMARY),
            x=0.04
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            zaxis=dict(title=dict(text="Relative Sag (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.95))
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=420
    )
    return fig


def create_2d_elevation_heatmap(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False,
    eye_label: str = ""
) -> go.Figure:
    """Creates a 2D Posterior Elevation Heatmap."""
    if reference_mode:
        post_elev = 6.0
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference"
    else:
        post_elev = extraction.elevation.posterior_elevation_thinnest_um or 8.0
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        lbl = eye_label or (extraction.metadata.eye if extraction.metadata and extraction.metadata.eye else "")
        title_suffix = f"Patient Scan ({lbl})" if lbl else "Patient Scan"

    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    dist_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    Z = post_elev * np.exp(-dist_sq / 1.6) - (R**1.3) * 1.5
    Z[R > 4.0] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale="Spectral_r",
            colorbar=dict(
                title=dict(text="Elevation (µm)", font=dict(color=TEXT_COLOR)),
                tickfont=dict(color=TEXT_COLOR)
            ),
            hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Elevation: +%{z:.1f} µm<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[thinnest_x],
            y=[thinnest_y],
            mode="markers+text",
            marker=dict(symbol="circle-dot", size=10, color=NAVY_PRIMARY, line=dict(width=2, color="#DC2626")),
            text=[f"Peak +{post_elev:.1f}µm"],
            textposition="top right",
            textfont=dict(color=NAVY_PRIMARY, size=11, family="Arial Black"),
            name="Peak Elevation",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text=f"Posterior Elevation Map (2D) • {title_suffix}", font=dict(size=14, color=NAVY_PRIMARY)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, zerolinecolor="#CBD5E1", scaleanchor="x"),
        margin=dict(l=10, r=10, b=10, t=35),
        height=320
    )
    return fig


def create_3d_badd_figure(
    extraction: PentacamExtractionResult,
    reference_mode: bool = False
) -> go.Figure:
    """
    Creates an interactive 3D model showing Belin-Ambrósio standard deviation distribution.
    """
    if reference_mode:
        bad_d = 0.85
        thinnest_x = 0.0
        thinnest_y = 0.0
        title_suffix = "Healthy Reference Model"
    else:
        bad_d = extraction.bad_display.final_bad_d_index or 1.0
        thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
        thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
        title_suffix = f"Patient Scan ({extraction.metadata.eye or 'Eye'})"

    r = np.linspace(0, 4.0, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    dist_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    Deviation_Map = 0.6 + (bad_d - 0.6) * np.exp(-dist_sq / 2.0)
    Z = -(R**2) / 15.0 - (Deviation_Map * 0.03)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=Deviation_Map,
                colorscale="YlOrRd",
                cmin=0.0,
                cmax=max(4.0, bad_d + 0.5),
                colorbar=dict(
                    title=dict(text="BAD-D (SD)", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                    len=0.75,
                    thickness=14
                ),
                contours_z=dict(show=True, usecolormap=True, project_z=True),
                hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>BAD-D Deviation: %{surfacecolor:.2f} SD<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"3D Belin-Ambrósio Deviation Surface • {title_suffix}",
            font=dict(size=14, color=NAVY_PRIMARY),
            x=0.04
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            zaxis=dict(title=dict(text="Deviation Sag", font=dict(color=MUTED_TEXT)), gridcolor=GRID_COLOR, showbackground=False),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.95))
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=420
    )
    return fig


def create_bad_d_gauge_figure(
    bad_d_value: Optional[float],
    reference_mode: bool = False,
    eye_label: str = ""
) -> go.Figure:
    """Creates an editorial styled Belin-Ambrósio BAD-D speedometer gauge."""
    if reference_mode:
        val = 0.85
        title_text = "BAD-D Reference Index (Healthy Baseline)"
    else:
        val = bad_d_value if bad_d_value is not None else 0.8
        title_text = f"BAD-D Final Index (SD) • {eye_label}" if eye_label else "BAD-D Final Index (SD)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=val,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title_text, font=dict(size=14, color=NAVY_PRIMARY)),
            delta=dict(reference=1.60, increasing=dict(color="#EF4444"), decreasing=dict(color="#10B981")),
            gauge=dict(
                axis=dict(range=[0, 6.0], tickwidth=1, tickcolor=MUTED_TEXT, tickfont=dict(color=TEXT_COLOR)),
                bar=dict(color=NAVY_PRIMARY, thickness=0.25),
                bgcolor="#F8FAFC",
                borderwidth=1.5,
                bordercolor=GRID_COLOR,
                steps=[
                    dict(range=[0, 1.60], color="#D1FAE5"),    # Normal (Light Green)
                    dict(range=[1.60, 2.60], color="#FEF3C7"),  # Suspect (Light Yellow)
                    dict(range=[2.60, 6.0], color="#FEE2E2"),   # Pathological (Light Red)
                ],
                threshold=dict(
                    line=dict(color="#EF4444", width=3),
                    thickness=0.75,
                    value=2.60
                )
            )
        )
    )
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=25, r=25, b=20, t=40),
        height=220
    )
    return fig

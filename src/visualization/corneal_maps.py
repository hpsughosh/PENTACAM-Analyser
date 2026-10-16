"""
Corneal Visualization Module using Plotly.
Generates interactive 3D and 2D tomographic maps to provide intuitive visual scaffolding for patients and clinicians.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from src.schemas.stage1_extraction import PentacamExtractionResult


def create_3d_corneal_dome_figure(extraction: PentacamExtractionResult) -> go.Figure:
    """
    Creates an interactive 3D surface mesh of the anterior corneal surface
    with color mapping representing refractive power or elevation.
    """
    kmax = extraction.keratometry.kmax_diopters or 44.0
    km = extraction.keratometry.km_mean_diopters or 43.5
    thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
    thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0
    q_val = extraction.keratometry.q_value_asphericity or -0.25

    # Grid across 8mm optical zone
    r = np.linspace(0, 4.0, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    # Base prolate ellipsoid radius in mm (R0 = 337.5 / Km)
    r0 = 337.5 / max(30.0, km)
    
    # Aspheric corneal sag height Z = (r^2 / r0) / (1 + sqrt(1 - (1+Q)*(r/r0)^2))
    # Add localized cone steepening if Kmax > 47
    cone_amplitude = max(0.0, (kmax - 47.0) * 0.04) if kmax > 47.0 else 0.0
    cone_dist_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    cone_sag = cone_amplitude * np.exp(-cone_dist_sq / 1.5)

    base_sag = (R**2) / (2 * r0)
    Z = -(base_sag + cone_sag)  # Dome pointing upward

    # Curvature mapping in Diopters across the surface
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
                    title=dict(text="Curvature (D)", font=dict(color="#ffffff")),
                    tickfont=dict(color="#ffffff"),
                    len=0.75,
                    thickness=15
                ),
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True
                ),
                hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Height: %{z:.3f}mm<br>Curvature: %{surfacecolor:.1f} D<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"Interactive 3D Corneal Dome Mesh ({extraction.metadata.eye})",
            font=dict(size=15, color="#ffffff"),
            x=0.05
        ),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        scene=dict(
            xaxis=dict(title=dict(text="X (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", showbackground=False),
            yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", showbackground=False),
            zaxis=dict(title=dict(text="Sag Height (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", showbackground=False),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9))
        ),
        margin=dict(l=10, r=10, b=10, t=40),
        height=450
    )
    return fig


def create_2d_topography_heatmap(extraction: PentacamExtractionResult) -> go.Figure:
    """Creates a 2D Sagittal Curvature Heatmap resembling the Pentacam front map."""
    kmax = extraction.keratometry.kmax_diopters or 44.0
    km = extraction.keratometry.km_mean_diopters or 43.5
    thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
    thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0

    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    dist_cone_sq = (X - thinnest_x)**2 + (Y - thinnest_y)**2
    cone_steepening = (kmax - km) * np.exp(-dist_cone_sq / 1.8)
    Z = km + cone_steepening
    Z[R > 4.0] = np.nan  # Mask outside 8mm zone

    fig = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale="Jet",
            zmin=max(35.0, km - 4),
            zmax=max(48.0, kmax + 2),
            colorbar=dict(
                title=dict(text="Diopters (D)", font=dict(color="#ffffff")),
                tickfont=dict(color="#ffffff")
            ),
            hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Curvature: %{z:.2f} D<extra></extra>"
        )
    )

    # Add marker for Kmax
    fig.add_trace(
        go.Scatter(
            x=[thinnest_x],
            y=[thinnest_y],
            mode="markers+text",
            marker=dict(symbol="cross", size=10, color="white", line=dict(width=2, color="black")),
            text=[f"Kmax: {kmax:.1f}D"],
            textposition="top right",
            textfont=dict(color="white", size=11),
            name="Kmax Apex",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text="Sagittal Curvature Map (2D)", font=dict(size=14, color="#ffffff")),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", zerolinecolor="#4B5563"),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", zerolinecolor="#4B5563", scaleanchor="x"),
        margin=dict(l=10, r=10, b=10, t=35),
        height=320
    )
    return fig


def create_2d_pachymetry_heatmap(extraction: PentacamExtractionResult) -> go.Figure:
    """Creates a 2D Pachymetric Thickness Distribution Heatmap."""
    cct = extraction.pachymetry.central_corneal_thickness_cct_um or 540.0
    thinnest = extraction.pachymetry.thinnest_pachymetry_um or 535.0
    thinnest_x = extraction.pachymetry.thinnest_loc_x_mm or 0.0
    thinnest_y = extraction.pachymetry.thinnest_loc_y_mm or 0.0

    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Thickness increases concentrically to ~650-700um at periphery
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
                title=dict(text="Thickness (µm)", font=dict(color="#ffffff")),
                tickfont=dict(color="#ffffff")
            ),
            hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>Thickness: %{z:.0f} µm<extra></extra>"
        )
    )

    # Add marker for thinnest point
    fig.add_trace(
        go.Scatter(
            x=[thinnest_x],
            y=[thinnest_y],
            mode="markers+text",
            marker=dict(symbol="star", size=10, color="#EF4444", line=dict(width=1, color="white")),
            text=[f"Thinnest: {thinnest:.0f}µm"],
            textposition="bottom right",
            textfont=dict(color="#F87171", size=11),
            name="Thinnest Point",
            showlegend=False
        )
    )

    fig.update_layout(
        title=dict(text="Pachymetry Thickness Map (2D)", font=dict(size=14, color="#ffffff")),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(title=dict(text="X (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", zerolinecolor="#4B5563"),
        yaxis=dict(title=dict(text="Y (mm)", font=dict(color="#9CA3AF")), gridcolor="#374151", zerolinecolor="#4B5563", scaleanchor="x"),
        margin=dict(l=10, r=10, b=10, t=35),
        height=320
    )
    return fig


def create_bad_d_gauge_figure(bad_d_value: Optional[float]) -> go.Figure:
    """Creates an intuitive Belin-Ambrósio BAD-D speedometer gauge."""
    val = bad_d_value if bad_d_value is not None else 0.8

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=val,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text="BAD-D Final Index (SD)", font=dict(size=15, color="#ffffff")),
            delta=dict(reference=1.60, increasing=dict(color="#EF4444"), decreasing=dict(color="#10B981")),
            gauge=dict(
                axis=dict(range=[0, 6.0], tickwidth=1, tickcolor="#9CA3AF", tickfont=dict(color="#ffffff")),
                bar=dict(color="#38BDF8", thickness=0.25),
                bgcolor="#1F2937",
                borderwidth=2,
                bordercolor="#374151",
                steps=[
                    dict(range=[0, 1.60], color="#065F46"),   # Normal (Green)
                    dict(range=[1.60, 2.60], color="#854D0E"), # Suspect (Yellow/Orange)
                    dict(range=[2.60, 6.0], color="#991B1B"),  # Pathological (Red)
                ],
                threshold=dict(
                    line=dict(color="#F87171", width=4),
                    thickness=0.75,
                    value=2.60
                )
            )
        )
    )
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=25, r=25, b=20, t=40),
        height=220
    )
    return fig

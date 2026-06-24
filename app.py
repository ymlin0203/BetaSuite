import io
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, anosim, mantel
from skbio.stats.ordination import pcoa
from streamlit.runtime.uploaded_file_manager import UploadedFile


APP_VERSION = "Biological Ecosystem Trajectory and Association Suite"

st.set_page_config(
    page_title="BetaSuite | Ecosystem Trajectory Analysis",
    page_icon="🧬",
    layout="wide",
)


def safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --apple-bg: #f5f5f7;
                --apple-card: #ffffff;
                --apple-text: #1d1d1f;
                --apple-muted: #6e6e73;
                --apple-border: #d2d2d7;
                --apple-blue: #0071e3;
                --apple-blue-dark: #005bb5;
                --apple-green: #34c759;
                --apple-shadow: 0 10px 28px rgba(0, 0, 0, 0.045);
            }

            .stApp {
                background: var(--apple-bg);
                color: var(--apple-text);
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
            }

            .block-container {
                max-width: 1280px;
                padding-top: 5.8rem !important;
                padding-bottom: 8rem !important;
            }

            header[data-testid="stHeader"] {
                background: rgba(245, 245, 247, 0.92);
                backdrop-filter: saturate(180%) blur(20px);
                border-bottom: 1px solid rgba(0, 0, 0, 0.06);
            }

            #MainMenu, footer, div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0;
            }

            .stDeployButton {
                display: none !important;
            }

            .landing-hero {
                position: relative;
                overflow: hidden;
                background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
                border-radius: 34px;
                padding: 64px 60px 60px 60px;
                margin: 0 auto 22px auto;
                min-height: 340px;
                border: 1px solid rgba(0, 0, 0, 0.055);
                box-shadow: var(--apple-shadow);
            }

            .landing-hero::after {
                content: "";
                position: absolute;
                right: -105px;
                top: -120px;
                width: 420px;
                height: 420px;
                border-radius: 50%;
                background:
                    radial-gradient(circle at 35% 35%, rgba(0, 113, 227, 0.22), transparent 44%),
                    radial-gradient(circle at 65% 65%, rgba(175, 82, 222, 0.15), transparent 45%);
                opacity: 0.9;
            }

            .eyebrow {
                position: relative;
                z-index: 2;
                color: var(--apple-blue);
                font-size: 12px;
                font-weight: 850;
                letter-spacing: 0.13em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .landing-hero h1 {
                position: relative;
                z-index: 2;
                max-width: 860px;
                margin: 0;
                color: var(--apple-text);
                font-size: clamp(44px, 6vw, 76px);
                line-height: 0.98;
                letter-spacing: -0.06em;
                font-weight: 850;
            }

            .landing-hero p {
                position: relative;
                z-index: 2;
                max-width: 780px;
                margin: 22px 0 0 0;
                color: var(--apple-muted);
                font-size: clamp(18px, 2vw, 23px);
                line-height: 1.38;
                letter-spacing: -0.02em;
                font-weight: 500;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
                margin: 0 0 22px 0;
            }

            .feature-card,
            .panel-card {
                background: var(--apple-card);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 24px;
                padding: 20px 22px;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.035);
            }

            .feature-card .icon {
                font-size: 26px;
                margin-bottom: 10px;
            }

            .feature-card h3 {
                margin: 0 0 8px 0;
                color: var(--apple-text);
                font-size: 20px;
                line-height: 1.15;
                letter-spacing: -0.035em;
                font-weight: 850;
            }

            .feature-card p {
                margin: 0;
                color: var(--apple-muted);
                font-size: 14.5px;
                line-height: 1.48;
            }

            .workspace-hero {
                background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 20px;
                padding: 13px 18px;
                margin-bottom: 10px;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.025);
            }

            .workspace-hero h1 {
                margin: 0;
                color: var(--apple-text);
                font-size: clamp(23px, 2.6vw, 32px);
                line-height: 1.05;
                letter-spacing: -0.045em;
                font-weight: 850;
            }

            .workspace-hero p {
                margin: 5px 0 0 0;
                color: var(--apple-muted);
                font-size: 13.8px;
                line-height: 1.42;
                max-width: 940px;
            }

            .summary-strip,
            .settings-strip,
            .overview-strip {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                margin: 0 0 12px 0;
            }

            .summary-pill,
            .setting-chip,
            .overview-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 7px 11px;
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 999px;
                color: var(--apple-muted);
                font-size: 12.8px;
                font-weight: 750;
                box-shadow: 0 5px 14px rgba(0, 0, 0, 0.022);
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--apple-green);
                display: inline-block;
            }

            .setting-chip strong,
            .overview-chip strong,
            .summary-pill strong {
                color: var(--apple-text);
            }

            .section-title {
                color: var(--apple-text);
                font-size: 24px;
                line-height: 1.1;
                letter-spacing: -0.04em;
                font-weight: 850;
                margin: 14px 0 9px 0;
            }

            .section-subtitle {
                color: var(--apple-muted);
                font-size: 14.5px;
                line-height: 1.48;
                margin-bottom: 12px;
            }

            div[data-testid="metric-container"] {
                background: var(--apple-card);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 18px;
                padding: 12px 14px;
                box-shadow: 0 7px 20px rgba(0, 0, 0, 0.03);
            }

            div[data-testid="stMetricLabel"] p {
                color: var(--apple-muted) !important;
                font-weight: 750;
                font-size: 13px !important;
            }

            div[data-testid="stMetricValue"] {
                color: var(--apple-text);
                letter-spacing: -0.04em;
                font-size: 31px !important;
            }

            .interpretation-card {
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 20px;
                padding: 16px 18px;
                margin-top: 12px;
                color: var(--apple-muted);
                font-size: 14.5px;
                line-height: 1.55;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.03);
            }

            .interpretation-card strong {
                color: var(--apple-text);
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e5e5ea;
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 5.3rem !important;
                padding-left: 1.15rem;
                padding-right: 1.15rem;
            }

            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p {
                color: var(--apple-text) !important;
            }

            section[data-testid="stSidebar"] small,
            section[data-testid="stSidebar"] .stCaptionContainer {
                color: var(--apple-muted) !important;
            }

            div[data-testid="stFileUploader"] {
                background: #ffffff;
                border: 1px solid #e5e5ea;
                border-radius: 18px;
                padding: 8px 10px;
                box-shadow: 0 7px 18px rgba(0, 0, 0, 0.025);
            }

            div[data-testid="stFileUploaderDropzone"] {
                background: #fbfbfd !important;
                border: 1px dashed #d2d2d7 !important;
                border-radius: 16px !important;
            }

            div[data-testid="stFileUploaderDropzone"] * {
                color: var(--apple-text) !important;
            }

            div.stDownloadButton > button,
            div.stButton > button {
                border-radius: 999px;
                border: none;
                background: var(--apple-blue);
                color: #ffffff;
                font-weight: 750;
                padding: 0.56rem 1.12rem;
                box-shadow: 0 9px 20px rgba(0, 113, 227, 0.18);
            }

            div.stDownloadButton > button:hover,
            div.stButton > button:hover {
                background: var(--apple-blue-dark);
                color: #ffffff;
                border: none;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                border-radius: 14px !important;
            }

            div[data-testid="stAlert"] {
                border-radius: 18px;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                padding: 7px;
                border-radius: 999px;
                box-shadow: 0 5px 14px rgba(0, 0, 0, 0.022);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 8px 16px;
                color: var(--apple-muted);
                font-weight: 750;
            }

            .stTabs [aria-selected="true"] {
                background: #f5f5f7;
                color: var(--apple-text) !important;
            }

            @media (max-width: 900px) {
                .feature-grid {
                    grid-template-columns: 1fr;
                }

                .block-container {
                    padding-top: 6.2rem !important;
                }

                .landing-hero {
                    padding: 42px 30px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page() -> None:
    st.markdown(
        """
        <section class="landing-hero">
            <div class="eyebrow">BetaSuite</div>
            <h1>Biological Ecosystem<br>Trajectory and Association Suite</h1>
            <p>
                An integrated microbiome analytics workspace for beta diversity, ecological trajectory,
                metadata association testing, and publication-ready visualization.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">🧬</div>
                <h3>Biological ecosystems.</h3>
                <p>Match microbiome distance matrices with metadata and inspect sample compatibility instantly.</p>
            </div>
            <div class="feature-card">
                <div class="icon">📈</div>
                <h3>Trajectory visualization.</h3>
                <p>Generate fixed-frame 2D PCoA plots with consistent legends, axes, and 1200 dpi export.</p>
            </div>
            <div class="feature-card">
                <div class="icon">🧪</div>
                <h3>Association testing.</h3>
                <p>Run ANOSIM for categorical variables and Mantel tests for continuous clinical variables.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_header(n_common: int, n_variables: int, distance_name: str, color_var: str, test_name: str) -> None:
    st.markdown(
        f"""
        <section class="workspace-hero">
            <div class="eyebrow">BetaSuite</div>
            <h1>Biological Ecosystem Trajectory and Association Suite</h1>
            <p>{n_common} matched samples · {n_variables} metadata variables · Distance: {distance_name} · Color: {color_var} · Test: {test_name}</p>
        </section>

        <div class="summary-strip">
            <div class="summary-pill"><span class="status-dot"></span> <strong>{n_common}</strong> matched samples</div>
            <div class="summary-pill"><strong>{n_variables}</strong> metadata variables</div>
            <div class="summary-pill"><strong>{distance_name}</strong> distance matrix</div>
            <div class="summary-pill"><span class="status-dot"></span> PCoA ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_strip(n_dist: int, n_meta: int, n_common: int, n_vars: int) -> None:
    st.markdown(
        f"""
        <div class="overview-strip">
            <div class="overview-chip"><strong>{n_dist}</strong> distance samples</div>
            <div class="overview-chip"><strong>{n_meta}</strong> metadata samples</div>
            <div class="overview-chip"><strong>{n_common}</strong> matched samples</div>
            <div class="overview-chip"><strong>{n_vars}</strong> metadata variables</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_strip(
    distance_name: str,
    metadata_name: str,
    color_var: str,
    x_axis: str,
    y_axis: str,
    plot_kind: str,
    test_name: str,
    perm_count: int,
) -> None:
    st.markdown(
        f"""
        <div class="settings-strip">
            <div class="setting-chip"><strong>Distance</strong> {distance_name}</div>
            <div class="setting-chip"><strong>Metadata</strong> {metadata_name}</div>
            <div class="setting-chip"><strong>Color</strong> {color_var}</div>
            <div class="setting-chip"><strong>Axes</strong> {x_axis} × {y_axis}</div>
            <div class="setting-chip"><strong>Type</strong> {plot_kind.title()}</div>
            <div class="setting-chip"><strong>Test</strong> {test_name}</div>
            <div class="setting-chip"><strong>Permutations</strong> {perm_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def interpretation_text(method: str, stat: float, p_value: float) -> str:
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    if method == "ANOSIM":
        if abs(stat) < 0.1:
            strength = "very weak"
        elif abs(stat) < 0.3:
            strength = "weak to moderate"
        elif abs(stat) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        return (
            f"The selected grouping variable is <strong>{significance}</strong> "
            f"(p = {p_value:.4g}). The ANOSIM R value suggests a <strong>{strength}</strong> "
            "degree of between-group separation in beta diversity."
        )

    return (
        f"The selected continuous variable is <strong>{significance}</strong> "
        f"(p = {p_value:.4g}). The Mantel R value indicates the correlation strength "
        "between the microbiome distance matrix and the clinical-distance matrix."
    )


def sidebar_inputs():
    with st.sidebar:
        st.markdown("### 🧬 BetaSuite")
        st.caption(APP_VERSION)
        st.markdown("---")
        st.markdown("### 1. Data Input")

        distance_file = st.file_uploader(
            "Distance matrix (.tsv / .csv)",
            type=["tsv", "csv"],
            key="distance_matrix_uploader",
        )

        metadata_file = st.file_uploader(
            "Metadata (.xlsx / .csv)",
            type=["xlsx", "csv"],
            key="metadata_uploader",
        )

    return distance_file, metadata_file


def read_distance_matrix(distance_file: UploadedFile) -> pd.DataFrame:
    try:
        df_dist = pd.read_csv(distance_file, sep=None, engine="python", index_col=0)
    except Exception as e:
        st.error(f"Distance matrix loading failed: {e}")
        st.stop()

    df_dist.index = df_dist.index.astype(str).str.strip()
    df_dist.columns = df_dist.columns.astype(str).str.strip()

    try:
        df_dist = df_dist.loc[df_dist.index, df_dist.columns]
    except Exception:
        pass

    try:
        df_dist = df_dist.apply(pd.to_numeric)
    except Exception:
        st.error("The distance matrix contains non-numeric values. Please check the matrix content.")
        st.stop()

    return df_dist


def read_metadata(metadata_file: UploadedFile) -> pd.DataFrame:
    try:
        if metadata_file.name.endswith(".xlsx"):
            df_meta = pd.read_excel(metadata_file, engine="openpyxl", dtype=str)
        else:
            df_meta = pd.read_csv(metadata_file, dtype=str)
    except Exception as e:
        st.error(f"Metadata loading failed: {e}")
        st.stop()

    if df_meta.empty or df_meta.shape[1] < 2:
        st.error("Metadata must contain a SampleID column and at least one variable column.")
        st.stop()

    df_meta.columns.values[0] = "SampleID"
    df_meta["SampleID"] = df_meta["SampleID"].astype(str).str.strip()
    return df_meta.set_index("SampleID")


def sidebar_analysis_controls(pc_cols: list[str], meta_cols: list[str]) -> dict:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 2. Visualization")

        x_axis = st.selectbox("X axis", pc_cols, index=0, key="x_axis")
        y_axis = st.selectbox("Y axis", pc_cols, index=1, key="y_axis")
        color_var = st.selectbox("Color variable", meta_cols, key="color_var")

        mode = st.radio(
            "Variable type",
            ["Auto detect", "Categorical", "Continuous"],
            index=0,
            key="data_type_mode",
        )

        view_mode = st.radio(
            "Display mode",
            ["2D", "3D"],
            index=0,
            key="view_mode",
        )

        with st.expander("3. Figure Style", expanded=False):
            lock_aspect = st.checkbox("Publication frame", value=True, key="lock_aspect")
            fig_width = st.number_input("Figure width", min_value=4.0, max_value=30.0, value=6.0, step=0.5, key="fig_width")
            fig_height = st.number_input("Figure height", min_value=4.0, max_value=30.0, value=6.0, step=0.5, key="fig_height")
            marker_size = st.number_input("Marker size", min_value=10, max_value=600, value=100, step=10, key="marker_size")
            spine_width = st.number_input("Frame line width", min_value=0.0, max_value=10.0, value=1.5, step=0.5, key="spine_width")
            edge_color = st.selectbox("Marker edge color", ["white", "black", "none"], index=0, key="edge_color")
            point_alpha = st.number_input("Marker opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="point_alpha")
            show_title = st.checkbox("Show plot title", value=False, key="show_title")
            show_legend_box = st.checkbox("Show legend frame", value=False, key="show_legend_box")
            force_equal = st.checkbox("Force equal axis ratio", value=False, key="force_equal")
            reverse_x = st.checkbox("Reverse X axis", value=False, key="reverse_x")
            reverse_y = st.checkbox("Reverse Y axis", value=False, key="reverse_y")

            axis_mode = st.radio(
                "Axis range",
                ["Auto", "Fixed equal scale", "Manual"],
                index=0,
                key="axis_mode",
            )

        with st.expander("4. Statistics", expanded=False):
            perm_count = st.number_input("Permutations", min_value=10, step=100, value=999, key="perm_count")
            random_seed = st.number_input("Random seed", min_value=1, value=42, step=1, key="random_seed")

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "color_var": color_var,
        "mode": mode,
        "view_mode": view_mode,
        "lock_aspect": lock_aspect,
        "fig_width": fig_width,
        "fig_height": fig_height,
        "marker_size": marker_size,
        "spine_width": spine_width,
        "edge_color": edge_color,
        "point_alpha": point_alpha,
        "show_title": show_title,
        "show_legend_box": show_legend_box,
        "force_equal": force_equal,
        "reverse_x": reverse_x,
        "reverse_y": reverse_y,
        "axis_mode": axis_mode,
        "perm_count": perm_count,
        "random_seed": random_seed,
    }


def prepare_color_variable(df_merged: pd.DataFrame, color_var: str, mode: str):
    if mode == "Categorical":
        df_merged[color_var] = df_merged[color_var].astype(str)
        palette = st.sidebar.selectbox("Categorical palette", ["Set1", "Set2", "tab10", "Dark2"], index=0, key="categorical_palette")
        return "categorical", palette, {}, df_merged

    if mode == "Continuous":
        df_merged[color_var] = pd.to_numeric(df_merged[color_var], errors="coerce")
        df_merged = df_merged[df_merged[color_var].notna()].copy()
        if df_merged.empty:
            st.error("The selected continuous variable cannot be converted to valid numeric values.")
            st.stop()
        palette = st.sidebar.selectbox("Continuous palette", ["viridis", "plasma", "cividis"], index=0, key="continuous_palette")
        return "continuous", palette, {"color_continuous_scale": palette}, df_merged

    numeric_series = pd.to_numeric(df_merged[color_var], errors="coerce")
    numeric_ratio = numeric_series.notna().sum() / max(len(numeric_series), 1)

    if numeric_ratio >= 0.8 and numeric_series.nunique(dropna=True) > 10:
        df_merged[color_var] = numeric_series
        df_merged = df_merged[df_merged[color_var].notna()].copy()
        palette = st.sidebar.selectbox("Continuous palette", ["viridis", "plasma", "cividis"], index=0, key="continuous_palette")
        return "continuous", palette, {"color_continuous_scale": palette}, df_merged

    df_merged[color_var] = df_merged[color_var].astype(str)
    palette = st.sidebar.selectbox("Categorical palette", ["Set1", "Set2", "tab10", "Dark2"], index=0, key="categorical_palette")
    return "categorical", palette, {}, df_merged


def get_axis_limits(axis_mode: str, x_vals_tmp: pd.Series, y_vals_tmp: pd.Series):
    if axis_mode == "Fixed equal scale":
        pad_ratio = st.sidebar.slider("Axis padding", 0.0, 0.5, 0.10, 0.01, key="pad_ratio")
        limit = float(np.max(np.abs(np.r_[x_vals_tmp.values, y_vals_tmp.values])) * (1.0 + pad_ratio))
        return (-limit, limit), (-limit, limit)

    if axis_mode == "Manual":
        x_min = st.sidebar.number_input("X minimum", value=float(np.min(x_vals_tmp.values)), key="x_min")
        x_max = st.sidebar.number_input("X maximum", value=float(np.max(x_vals_tmp.values)), key="x_max")
        y_min = st.sidebar.number_input("Y minimum", value=float(np.min(y_vals_tmp.values)), key="y_min")
        y_max = st.sidebar.number_input("Y maximum", value=float(np.max(y_vals_tmp.values)), key="y_max")
        return (x_min, x_max), (y_min, y_max)

    return None, None


def render_2d_plot(
    df_merged: pd.DataFrame,
    pcoa_results,
    x_vals_tmp: pd.Series,
    y_vals_tmp: pd.Series,
    x_axis: str,
    y_axis: str,
    color_var: str,
    plot_kind: str,
    palette: str,
    xlim,
    ylim,
    controls: dict,
):
    fig_width = controls["fig_width"]
    fig_height = controls["fig_height"]
    lock_aspect = controls["lock_aspect"]

    if lock_aspect:
        fig = plt.figure(figsize=(fig_width, fig_height))

        # 保留原本方框固定邏輯
        square_size = fig_height * 0.75
        ax_w = square_size / fig_width
        ax_h = square_size / fig_height
        left_margin = 0.15
        bottom_margin = 0.15
        ax = fig.add_axes([left_margin, bottom_margin, ax_w, ax_h])
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        left_margin = 0.15
        bottom_margin = 0.15
        ax_w = 0.75
        ax_h = 0.75

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    edge_color = controls["edge_color"]
    marker_size = controls["marker_size"]
    point_alpha = controls["point_alpha"]

    if plot_kind == "categorical":
        sns.scatterplot(
            x=x_vals_tmp,
            y=y_vals_tmp,
            hue=df_merged[color_var],
            palette=palette,
            s=marker_size,
            edgecolor=edge_color if edge_color != "none" else None,
            linewidth=1.2 if edge_color != "none" else 0,
            alpha=point_alpha,
            ax=ax,
            zorder=3,
        )

        ax.legend(
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            frameon=controls["show_legend_box"],
            fontsize=16,
            markerscale=1.0,
        )
    else:
        sc = ax.scatter(
            x_vals_tmp,
            y_vals_tmp,
            c=df_merged[color_var],
            cmap=palette,
            s=marker_size,
            edgecolors=edge_color if edge_color != "none" else "none",
            linewidths=1.2 if edge_color != "none" else 0,
            alpha=point_alpha,
            zorder=3,
        )

        if lock_aspect:
            cax_left = left_margin + ax_w + 0.03
            cax = fig.add_axes([cax_left, bottom_margin, 0.03, ax_h])
            cbar = fig.colorbar(sc, cax=cax)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = fig.colorbar(sc, cax=cax)

        cbar.set_label(color_var, fontsize=22)
        cbar.ax.tick_params(labelsize=18)
        cbar.outline.set_linewidth(2.5)

    var_x = float(pcoa_results.proportion_explained[x_axis] * 100)
    var_y = float(pcoa_results.proportion_explained[y_axis] * 100)

    ax.set_xlabel(f"{x_axis} ({var_x:.2f}%)", fontsize=24)
    ax.set_ylabel(f"{y_axis} ({var_y:.2f}%)", fontsize=24)

    if controls["show_title"]:
        ax.set_title(f"PCoA colored by {color_var}", fontsize=24, pad=15)

    ax.tick_params(axis="both", which="major", labelsize=18, length=8, width=2.5, direction="out")

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if controls["force_equal"]:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    for spine in ax.spines.values():
        if controls["spine_width"] > 0:
            spine.set_visible(True)
            spine.set_linewidth(controls["spine_width"])
        else:
            spine.set_visible(False)

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=1200, bbox_inches="tight", transparent=False)
    buf.seek(0)

    st.download_button(
        "Download 2D PCoA PNG (1200 dpi)",
        data=buf,
        file_name=f"{safe_key(color_var)}_PCoA.png",
        mime="image/png",
        key=f"download_2d_png_{safe_key(color_var)}",
    )

    plt.close(fig)


def render_3d_plot(df_merged: pd.DataFrame, color_var: str, color_args: dict):
    has_pc1_to_3 = all(pc in df_merged.columns for pc in ["PC1", "PC2", "PC3"])

    if not has_pc1_to_3:
        st.info("3D plotting is unavailable because PC1, PC2, or PC3 is missing.")
        return

    fig3d = px.scatter_3d(
        df_merged,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_var,
        title=f"PCoA colored by {color_var}",
        labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
        **color_args,
    )

    st.plotly_chart(fig3d, use_container_width=True)

    html_bytes = fig3d.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

    st.download_button(
        label="Download 3D interactive PCoA HTML",
        data=html_bytes,
        file_name=f"{safe_key(color_var)}_3D_PCoA.html",
        mime="text/html",
        key=f"download_3d_html_{safe_key(color_var)}",
    )

    st.info("3D export uses HTML format to avoid Kaleido or Chrome dependency issues on Streamlit Cloud.")


def run_statistical_test(
    df_merged: pd.DataFrame,
    full_distance_matrix: DistanceMatrix,
    color_var: str,
    plot_kind: str,
    perm_count: int,
    random_seed: int,
):
    distance_matrix = full_distance_matrix.filter(df_merged["SampleID"].tolist())

    random.seed(random_seed)
    np.random.seed(random_seed)

    if plot_kind == "categorical":
        group_series = df_merged.set_index("SampleID")[color_var]

        if group_series.nunique() < 2:
            return "ANOSIM", None, None, "This categorical variable has fewer than two groups."

        try:
            result = anosim(distance_matrix, group_series, permutations=perm_count)
            return "ANOSIM", float(result["test statistic"]), float(result["p-value"]), None
        except Exception as e:
            return "ANOSIM", None, None, f"ANOSIM failed: {e}"

    try:
        meta_dist = squareform(pdist(df_merged[[color_var]].values, metric="euclidean"))
        meta_matrix = DistanceMatrix(meta_dist, ids=df_merged["SampleID"])
        stat, p_value, _ = mantel(distance_matrix, meta_matrix, permutations=perm_count)
        return "Mantel test", float(stat), float(p_value), None
    except Exception as e:
        return "Mantel test", None, None, f"Mantel test failed: {e}"


def render_statistics(method: str, stat: float | None, p_value: float | None, error_msg: str | None):
    st.markdown('<div class="section-title">Statistical Result</div>', unsafe_allow_html=True)
    st.caption("Categorical variables use ANOSIM; continuous variables use Mantel test.")

    if error_msg:
        st.warning(error_msg)
        return

    if stat is None or p_value is None:
        st.warning("No statistical result is available.")
        return

    r1, r2, r3 = st.columns(3)
    r1.metric("Method", method)
    r2.metric("Statistic R", f"{stat:.4f}")
    r3.metric("P-value", f"{p_value:.4g}")

    st.markdown(
        f"""
        <div class="interpretation-card">
            <strong>Interpretation.</strong> {interpretation_text(method, stat, p_value)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_batch_analysis(df_meta: pd.DataFrame, meta_cols: list[str], full_distance_matrix: DistanceMatrix, perm_count: int, random_seed: int):
    st.markdown('<div class="section-title">Batch Clinical Association Analysis</div>', unsafe_allow_html=True)
    st.write("Automatically screen all metadata variables for potential beta diversity associations.")

    if not st.button("Run batch analysis", key="batch_analysis_button"):
        return None

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_vars = len(meta_cols)

    for i, var in enumerate(meta_cols):
        status_text.text(f"Analyzing ({i + 1}/{total_vars}): {var} ...")

        df_curr = df_meta[["SampleID", var]].copy()
        df_curr = df_curr[df_curr[var].notna() & (df_curr[var].astype(str).str.strip() != "")]
        valid_ids = df_curr["SampleID"].tolist()

        if len(valid_ids) < 3:
            progress_bar.progress((i + 1) / total_vars)
            continue

        curr_dist = full_distance_matrix.filter(valid_ids)
        df_curr = df_curr.set_index("SampleID")

        if df_curr[var].nunique() < 2:
            progress_bar.progress((i + 1) / total_vars)
            continue

        is_numeric = False

        try:
            df_curr_numeric = pd.to_numeric(df_curr[var], errors="coerce")
            if df_curr_numeric.notna().sum() > len(df_curr) * 0.5:
                is_numeric = True
                df_curr[var] = df_curr_numeric
        except Exception:
            pass

        random.seed(random_seed)
        np.random.seed(random_seed)

        if not is_numeric or df_curr[var].nunique() <= 10:
            try:
                group_series = df_curr[var].astype(str)
                if group_series.nunique() < 2:
                    progress_bar.progress((i + 1) / total_vars)
                    continue

                res = anosim(curr_dist, group_series, permutations=perm_count)
                stat_val = res["test statistic"]
                p_val = res["p-value"]
                n_val = len(group_series)
                method = "ANOSIM"
                dtype = "Categorical"
            except Exception:
                progress_bar.progress((i + 1) / total_vars)
                continue

        else:
            try:
                df_curr = df_curr.dropna(subset=[var])
                if len(df_curr) < 3:
                    progress_bar.progress((i + 1) / total_vars)
                    continue

                curr_dist = full_distance_matrix.filter(df_curr.index.tolist())
                meta_dist_arr = squareform(pdist(df_curr[[var]].values, metric="euclidean"))
                meta_dist_mat = DistanceMatrix(meta_dist_arr, ids=df_curr.index)
                stat_val, p_val, _ = mantel(curr_dist, meta_dist_mat, permutations=perm_count)
                n_val = len(df_curr)
                method = "Mantel test"
                dtype = "Continuous"
            except Exception:
                progress_bar.progress((i + 1) / total_vars)
                continue

        results.append(
            {
                "Variable": var,
                "Data type": dtype,
                "Method": method,
                "N": n_val,
                "Statistic R": stat_val,
                "P-value": p_val,
            }
        )

        progress_bar.progress((i + 1) / total_vars)

    status_text.text("Batch analysis completed.")

    if not results:
        st.warning("No variables were successfully analyzed. This may be due to insufficient sample size or invalid grouping.")
        return None

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("P-value", ascending=True).reset_index(drop=True)

    st.dataframe(df_results, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Statistical_Results")
    output.seek(0)

    st.download_button(
        label="Download batch statistics Excel",
        data=output,
        file_name="Batch_PCoA_Statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_batch_excel",
    )

    return df_results


def main() -> None:
    inject_css()

    distance_file, metadata_file = sidebar_inputs()

    if distance_file is None or metadata_file is None:
        render_landing_page()
        st.info("Upload a distance matrix and metadata file from the sidebar to start.")
        return

    df_dist = read_distance_matrix(distance_file)
    df_meta = read_metadata(metadata_file)

    dist_ids_set = set(df_dist.index)
    meta_ids_set = set(df_meta.index)
    common_ids = [i for i in df_dist.index if i in df_meta.index]

    in_dist_not_meta = dist_ids_set - meta_ids_set
    in_meta_not_dist = meta_ids_set - dist_ids_set

    if len(common_ids) < 3:
        st.error("At least three matched SampleIDs are required for PCoA.")
        st.stop()

    df_meta = df_meta.loc[common_ids].reset_index()
    df_dist = df_dist.loc[common_ids, common_ids]

    try:
        full_distance_matrix = DistanceMatrix(df_dist.values, ids=common_ids)
        pcoa_results = pcoa(full_distance_matrix)
    except Exception as e:
        st.error(f"PCoA calculation failed: {e}")
        st.stop()

    coords = pcoa_results.samples.reset_index().rename(columns={"index": "SampleID"})
    df_merged = pd.merge(coords, df_meta, on="SampleID", how="inner")

    pc_cols = [col for col in df_merged.columns if col.startswith("PC")]
    if len(pc_cols) < 2:
        st.error("PCoA results do not contain enough PC axes.")
        st.stop()

    meta_cols = [col for col in df_meta.columns if col != "SampleID"]
    if len(meta_cols) == 0:
        st.error("Metadata has no usable variables.")
        st.stop()

    controls = sidebar_analysis_controls(pc_cols, meta_cols)
    color_var = controls["color_var"]

    df_merged = df_merged[df_merged[color_var].notna() & (df_merged[color_var].astype(str).str.strip() != "")].copy()
    if df_merged.empty:
        st.error("The selected color variable has no valid data.")
        st.stop()

    plot_kind, palette, color_args, df_merged = prepare_color_variable(
        df_merged=df_merged,
        color_var=color_var,
        mode=controls["mode"],
    )

    x_axis = controls["x_axis"]
    y_axis = controls["y_axis"]

    x_vals_tmp = (df_merged[x_axis] * (-1 if controls["reverse_x"] else 1)).astype(float)
    y_vals_tmp = (df_merged[y_axis] * (-1 if controls["reverse_y"] else 1)).astype(float)
    xlim, ylim = get_axis_limits(controls["axis_mode"], x_vals_tmp, y_vals_tmp)

    test_name = "ANOSIM" if plot_kind == "categorical" else "Mantel test"

    render_workspace_header(
        n_common=len(common_ids),
        n_variables=len(meta_cols),
        distance_name=distance_file.name,
        color_var=color_var,
        test_name=test_name,
    )

    overview_tab, pcoa_tab, stats_tab, batch_tab, export_tab = st.tabs(
        ["Overview", "PCoA", "Statistics", "Batch Analysis", "Export"]
    )

    with overview_tab:
        st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
        render_overview_strip(len(df_dist.index), len(df_meta.index), len(common_ids), len(meta_cols))

        if in_dist_not_meta or in_meta_not_dist:
            with st.expander("Unmatched sample IDs", expanded=False):
                st.write(f"Matched samples: **{len(common_ids)}**")
                if in_dist_not_meta:
                    st.warning(
                        f"{len(in_dist_not_meta)} samples were found in the distance matrix but not in metadata: "
                        + ", ".join(sorted(in_dist_not_meta))
                    )
                if in_meta_not_dist:
                    st.warning(
                        f"{len(in_meta_not_dist)} samples were found in metadata but not in the distance matrix: "
                        + ", ".join(sorted(in_meta_not_dist))
                    )
        else:
            st.success("All sample IDs are matched between the distance matrix and metadata.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance samples", len(df_dist.index))
        c2.metric("Metadata samples", len(df_meta.index))
        c3.metric("Matched samples", len(common_ids))
        c4.metric("Variables", len(meta_cols))

    with pcoa_tab:
        st.markdown('<div class="section-title">PCoA Visualization</div>', unsafe_allow_html=True)
        render_settings_strip(
            distance_name=distance_file.name,
            metadata_name=metadata_file.name,
            color_var=color_var,
            x_axis=x_axis,
            y_axis=y_axis,
            plot_kind=plot_kind,
            test_name=test_name,
            perm_count=controls["perm_count"],
        )

        if controls["view_mode"] == "2D":
            render_2d_plot(
                df_merged=df_merged,
                pcoa_results=pcoa_results,
                x_vals_tmp=x_vals_tmp,
                y_vals_tmp=y_vals_tmp,
                x_axis=x_axis,
                y_axis=y_axis,
                color_var=color_var,
                plot_kind=plot_kind,
                palette=palette,
                xlim=xlim,
                ylim=ylim,
                controls=controls,
            )
        else:
            render_3d_plot(df_merged, color_var, color_args)

    method, stat, p_value, error_msg = run_statistical_test(
        df_merged=df_merged,
        full_distance_matrix=full_distance_matrix,
        color_var=color_var,
        plot_kind=plot_kind,
        perm_count=controls["perm_count"],
        random_seed=controls["random_seed"],
    )

    with stats_tab:
        render_statistics(method, stat, p_value, error_msg)

    with batch_tab:
        run_batch_analysis(
            df_meta=df_meta,
            meta_cols=meta_cols,
            full_distance_matrix=full_distance_matrix,
            perm_count=controls["perm_count"],
            random_seed=controls["random_seed"],
        )

    with export_tab:
        st.markdown('<div class="section-title">Export Center</div>', unsafe_allow_html=True)
        st.write("Export options are available in the relevant analysis tabs.")
        st.markdown(
            """
            <div class="panel-card">
                <strong>2D PCoA PNG</strong><br>
                Open the PCoA tab and use the 1200 dpi PNG download button.
                <br><br>
                <strong>3D PCoA HTML</strong><br>
                Switch display mode to 3D and download the interactive HTML file.
                <br><br>
                <strong>Batch statistics Excel</strong><br>
                Run batch analysis and download the Excel report from the Batch Analysis tab.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()


import io
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, anosim, mantel
from skbio.stats.ordination import pcoa
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Keep colorbar aligned with the main plot.
from mpl_toolkits.axes_grid1 import make_axes_locatable


APP_VERSION = "BetaSuite Dashboard 2026-06-22"

st.set_page_config(
    page_title="BetaSuite | Beta Diversity Analysis",
    page_icon="🧬",
    layout="wide",
)


def safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def inject_apple_like_css() -> None:
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
                --apple-soft-blue: #f5faff;
                --apple-shadow: 0 14px 36px rgba(0, 0, 0, 0.055);
            }

            .stApp {
                background: var(--apple-bg);
                color: var(--apple-text);
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
            }

            header[data-testid="stHeader"] {
                background: rgba(245, 245, 247, 0.88);
                backdrop-filter: saturate(180%) blur(20px);
                border-bottom: 1px solid rgba(0, 0, 0, 0.06);
            }

            #MainMenu, footer {
                visibility: hidden;
            }

            .block-container {
                max-width: 1320px;
                padding-top: 1.15rem;
                padding-bottom: 4.5rem;
            }

            .apple-mini-nav {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                max-width: 980px;
                margin: 0 auto 22px auto;
                padding: 10px 18px;
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 999px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.035);
                backdrop-filter: saturate(180%) blur(18px);
            }

            .apple-mini-nav .brand {
                font-size: 15px;
                font-weight: 800;
                color: var(--apple-text);
                letter-spacing: -0.015em;
                white-space: nowrap;
            }

            .apple-mini-nav .items {
                display: flex;
                align-items: center;
                gap: 18px;
                font-size: 12.5px;
                color: var(--apple-muted);
                font-weight: 650;
                white-space: nowrap;
            }

            .landing-hero {
                position: relative;
                overflow: hidden;
                background: linear-gradient(180deg, #ffffff 0%, #fbfbfd 100%);
                border-radius: 36px;
                padding: 70px 64px 66px 64px;
                margin: 0 auto 26px auto;
                min-height: 360px;
                border: 1px solid rgba(0, 0, 0, 0.055);
                box-shadow: var(--apple-shadow);
            }

            .landing-hero::after {
                content: "";
                position: absolute;
                right: -110px;
                top: -120px;
                width: 440px;
                height: 440px;
                border-radius: 50%;
                background:
                    radial-gradient(circle at 35% 35%, rgba(0, 113, 227, 0.25), transparent 44%),
                    radial-gradient(circle at 65% 65%, rgba(175, 82, 222, 0.16), transparent 45%);
                opacity: 0.92;
            }

            .eyebrow {
                position: relative;
                z-index: 2;
                color: var(--apple-blue);
                font-size: 14px;
                font-weight: 850;
                letter-spacing: 0.13em;
                text-transform: uppercase;
                margin-bottom: 14px;
            }

            .landing-hero h1 {
                position: relative;
                z-index: 2;
                max-width: 860px;
                margin: 0;
                color: var(--apple-text);
                font-size: clamp(46px, 6vw, 78px);
                line-height: 0.98;
                letter-spacing: -0.06em;
                font-weight: 850;
            }

            .landing-hero p {
                position: relative;
                z-index: 2;
                max-width: 780px;
                margin: 24px 0 0 0;
                color: var(--apple-muted);
                font-size: clamp(19px, 2vw, 25px);
                line-height: 1.36;
                letter-spacing: -0.02em;
                font-weight: 500;
            }

            .hero-actions {
                position: relative;
                z-index: 2;
                display: flex;
                gap: 12px;
                align-items: center;
                margin-top: 28px;
                flex-wrap: wrap;
            }

            .pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 18px;
                border-radius: 999px;
                background: var(--apple-blue);
                color: #fff;
                font-size: 15px;
                font-weight: 750;
                text-decoration: none;
            }

            .pill.secondary {
                background: #f5f5f7;
                color: var(--apple-blue);
                border: 1px solid #e5e5ea;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 16px;
                margin: 0 0 24px 0;
            }

            .feature-card,
            .panel-card {
                background: var(--apple-card);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 26px;
                padding: 22px 24px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.04);
            }

            .feature-card {
                min-height: 150px;
            }

            .feature-card .icon {
                font-size: 27px;
                margin-bottom: 12px;
            }

            .feature-card h3 {
                margin: 0 0 8px 0;
                color: var(--apple-text);
                font-size: 21px;
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
                border-radius: 26px;
                padding: 20px 24px;
                margin-bottom: 16px;
                box-shadow: 0 9px 24px rgba(0, 0, 0, 0.035);
            }

            .workspace-hero .eyebrow {
                font-size: 12px;
                margin-bottom: 6px;
            }

            .workspace-hero h1 {
                margin: 0;
                color: var(--apple-text);
                font-size: clamp(26px, 3vw, 40px);
                line-height: 1.04;
                letter-spacing: -0.045em;
                font-weight: 850;
            }

            .workspace-hero p {
                margin: 8px 0 0 0;
                color: var(--apple-muted);
                font-size: 15.5px;
                line-height: 1.45;
                max-width: 900px;
            }

            .status-bar {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 0 0 18px 0;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 13px;
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 999px;
                color: var(--apple-muted);
                font-size: 13.5px;
                font-weight: 750;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.026);
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--apple-green);
                display: inline-block;
            }

            .section-title {
                color: var(--apple-text);
                font-size: 27px;
                line-height: 1.1;
                letter-spacing: -0.04em;
                font-weight: 850;
                margin: 26px 0 12px 0;
            }

            .section-subtitle {
                color: var(--apple-muted);
                font-size: 15px;
                line-height: 1.48;
                margin-bottom: 14px;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
                margin: 8px 0 16px 0;
            }

            .metric-card-clean {
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 22px;
                padding: 16px 18px;
                min-height: 104px;
                box-shadow: 0 9px 24px rgba(0, 0, 0, 0.035);
            }

            .metric-card-clean .label {
                color: var(--apple-muted);
                font-size: 13.5px;
                font-weight: 750;
                line-height: 1.2;
                margin-bottom: 10px;
            }

            .metric-card-clean .value {
                color: var(--apple-text);
                font-size: clamp(30px, 3.6vw, 46px);
                line-height: 0.95;
                letter-spacing: -0.055em;
                font-weight: 850;
            }

            .settings-strip {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 0 0 16px 0;
            }

            .setting-chip {
                background: #ffffff;
                border: 1px solid rgba(0, 0, 0, 0.055);
                color: var(--apple-muted);
                border-radius: 999px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: 700;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.025);
            }

            .setting-chip strong {
                color: var(--apple-text);
            }

            div[data-testid="metric-container"] {
                background: var(--apple-card);
                border: 1px solid rgba(0, 0, 0, 0.055);
                border-radius: 22px;
                padding: 16px 18px;
                box-shadow: 0 9px 24px rgba(0, 0, 0, 0.035);
            }

            div[data-testid="stMetricLabel"] p {
                color: var(--apple-muted) !important;
                font-weight: 750;
            }

            div[data-testid="stMetricValue"] {
                color: var(--apple-text) !important;
                font-weight: 850;
                letter-spacing: -0.04em;
                font-size: 34px !important;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid rgba(0, 0, 0, 0.08);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 2.1rem;
                padding-left: 1.05rem;
                padding-right: 1.05rem;
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
                border-radius: 20px;
                padding: 8px 10px 12px 10px;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.026);
            }

            div[data-testid="stFileUploader"] section {
                background: #fbfbfd !important;
                border: 1.5px dashed #d2d2d7 !important;
                border-radius: 16px !important;
                color: var(--apple-text) !important;
                min-height: 92px !important;
            }

            div[data-testid="stFileUploader"] button {
                border-radius: 999px !important;
                background: #1d1d1f !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 750 !important;
            }

            div[data-testid="stFileUploader"] span,
            div[data-testid="stFileUploader"] p,
            div[data-testid="stFileUploader"] small {
                color: var(--apple-muted) !important;
            }

            div.stDownloadButton > button,
            div.stButton > button {
                border-radius: 999px;
                border: none;
                background: var(--apple-blue);
                color: white;
                font-weight: 750;
                padding: 0.56rem 1.14rem;
                box-shadow: 0 8px 18px rgba(0, 113, 227, 0.16);
            }

            div.stDownloadButton > button:hover,
            div.stButton > button:hover {
                background: var(--apple-blue-dark);
                color: white;
                border: none;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                border-radius: 14px !important;
                border-color: #d2d2d7 !important;
            }

            div[data-testid="stAlert"] {
                border-radius: 18px;
                border: 1px solid #dce8f7;
                background: #f5faff;
            }

            hr {
                margin-top: 1.15rem !important;
                margin-bottom: 1.15rem !important;
            }

            .js-plotly-plot,
            .stPlotlyChart,
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"] {
                border-radius: 22px;
            }

            @media (max-width: 900px) {
                .feature-grid,
                .metric-grid {
                    grid-template-columns: 1fr;
                }

                .landing-hero {
                    padding: 42px 30px;
                }

                .apple-mini-nav .items {
                    display: none;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page() -> None:
    st.markdown(
        """
        <div class="apple-mini-nav">
            <div class="brand">🧬 BetaSuite</div>
            <div class="items">
                <span>PCoA</span>
                <span>ANOSIM</span>
                <span>Mantel</span>
                <span>Export</span>
            </div>
        </div>

        <div class="landing-hero">
            <div class="eyebrow">BetaSuite</div>
            <h1>Beta diversity. Beautifully analyzed.</h1>
            <p>
                Upload your distance matrix and metadata to create publication-ready PCoA figures,
                inspect ANOSIM and Mantel statistics, and export clean analysis reports.
            </p>
            <div class="hero-actions">
                <span class="pill">Upload files from the sidebar</span>
                <span class="pill secondary">Publication-ready PNG · Interactive HTML · Excel report</span>
            </div>
        </div>

        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">📁</div>
                <h3>Clean input workflow.</h3>
                <p>Match distance matrices and metadata by SampleID with automatic mismatch checks.</p>
            </div>
            <div class="feature-card">
                <div class="icon">📊</div>
                <h3>Publication-ready PCoA.</h3>
                <p>Keep fixed figure frames, custom markers, legends, and 1200 dpi PNG export.</p>
            </div>
            <div class="feature-card">
                <div class="icon">🧪</div>
                <h3>Statistics in one place.</h3>
                <p>Run ANOSIM for categorical variables and Mantel tests for continuous variables.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_header(matched_samples: int, metadata_variables: int) -> None:
    st.markdown(
        f"""
        <div class="workspace-hero">
            <div class="eyebrow">Analysis Workspace</div>
            <h1>BetaSuite Dashboard</h1>
            <p>
                Data are loaded. Review the matched samples, generate PCoA figures,
                inspect ANOSIM or Mantel statistics, and export publication-ready outputs.
            </p>
        </div>

        <div class="status-bar">
            <span class="status-pill"><span class="status-dot"></span> Data loaded</span>
            <span class="status-pill"><span class="status-dot"></span> {matched_samples} matched samples</span>
            <span class="status-pill"><span class="status-dot"></span> {metadata_variables} metadata variables</span>
            <span class="status-pill"><span class="status-dot"></span> PCoA ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(distance_samples: int, metadata_samples: int, matched_samples: int, variables: int) -> None:
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card-clean">
                <div class="label">Distance samples</div>
                <div class="value">{distance_samples}</div>
            </div>
            <div class="metric-card-clean">
                <div class="label">Metadata samples</div>
                <div class="value">{metadata_samples}</div>
            </div>
            <div class="metric-card-clean">
                <div class="label">Matched samples</div>
                <div class="value">{matched_samples}</div>
            </div>
            <div class="metric-card-clean">
                <div class="label">Variables</div>
                <div class="value">{variables}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_strip(
    x_axis: str,
    y_axis: str,
    color_var: str,
    plot_kind: str,
    view_mode: str,
    perm_count: int,
) -> None:
    st.markdown(
        f"""
        <div class="settings-strip">
            <span class="setting-chip">View <strong>{view_mode}</strong></span>
            <span class="setting-chip">X <strong>{x_axis}</strong></span>
            <span class="setting-chip">Y <strong>{y_axis}</strong></span>
            <span class="setting-chip">Color <strong>{color_var}</strong></span>
            <span class="setting-chip">Variable <strong>{plot_kind}</strong></span>
            <span class="setting-chip">Permutation <strong>{perm_count}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def interpret_statistics(method: str, stat: float, p_value: float) -> str:
    if p_value < 0.001:
        sig_text = "highly significant"
    elif p_value < 0.05:
        sig_text = "statistically significant"
    else:
        sig_text = "not statistically significant"

    abs_stat = abs(float(stat))
    if abs_stat < 0.1:
        strength = "weak"
    elif abs_stat < 0.3:
        strength = "modest"
    elif abs_stat < 0.5:
        strength = "moderate"
    else:
        strength = "strong"

    if method == "ANOSIM":
        return (
            f"Interpretation: The selected grouping variable is {sig_text} "
            f"with a {strength} beta-diversity separation pattern."
        )

    return (
        f"Interpretation: The selected continuous variable shows a {sig_text} "
        f"association with the distance matrix, with a {strength} correlation strength."
    )


def main() -> None:
    inject_apple_like_css()

    with st.sidebar:
        st.markdown("### 🧬 BetaSuite")
        st.markdown(
            f'<p style="color:#6e6e73;font-size:13px;margin-top:-4px;">{APP_VERSION}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 1. Data Input")

        distance_file: UploadedFile = st.file_uploader(
            "Distance matrix (.tsv / .csv)",
            type=["tsv", "csv"],
            key="distance_matrix_uploader",
        )

        metadata_file: UploadedFile = st.file_uploader(
            "Metadata (.xlsx / .csv)",
            type=["xlsx", "csv"],
            key="metadata_uploader",
        )

    if distance_file is None or metadata_file is None:
        render_landing_page()
        st.info("📥 請先從左側 sidebar 上傳距離矩陣與 metadata 檔案。")
        return

    Pipeline().main(distance_file, metadata_file)


class Pipeline:
    def main(self, distance_file: UploadedFile, metadata_file: UploadedFile) -> None:
        df_dist = self._read_distance_matrix(distance_file)
        df_meta = self._read_metadata(metadata_file)

        dist_ids_set = set(df_dist.index)
        meta_ids_set = set(df_meta.index)
        common_ids = [i for i in df_dist.index if i in df_meta.index]

        in_dist_not_meta = dist_ids_set - meta_ids_set
        in_meta_not_dist = meta_ids_set - dist_ids_set

        if len(common_ids) < 3:
            st.error("🚩 距離矩陣與 metadata 可配對樣本數不足，至少需要 3 個共同 SampleID 才能進行 PCoA。")
            st.stop()

        render_workspace_header(matched_samples=len(common_ids), metadata_variables=len(df_meta.columns))

        st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)
        render_metric_grid(
            distance_samples=len(df_dist.index),
            metadata_samples=len(df_meta.index),
            matched_samples=len(common_ids),
            variables=len(df_meta.columns),
        )

        if in_dist_not_meta or in_meta_not_dist:
            with st.expander("⚠️ SampleID mismatch report", expanded=False):
                st.write(f"✅ 成功配對出的樣本數：**{len(common_ids)}** 個")

                if in_dist_not_meta:
                    st.error(
                        f"📍 以下 {len(in_dist_not_meta)} 個樣本出現在【距離矩陣】，但【Metadata】裡找不到：\n\n"
                        + ", ".join(sorted(in_dist_not_meta))
                    )

                if in_meta_not_dist:
                    st.warning(
                        f"📍 以下 {len(in_meta_not_dist)} 個樣本出現在【Metadata】，但【距離矩陣】裡找不到：\n\n"
                        + ", ".join(sorted(in_meta_not_dist))
                    )

        df_meta = df_meta.loc[common_ids].reset_index()
        df_dist = df_dist.loc[common_ids, common_ids]

        try:
            df_dist = df_dist.apply(pd.to_numeric)
        except Exception:
            st.error("🚩 距離矩陣包含無法轉換成數值的內容，請確認矩陣內皆為數字。")
            st.stop()

        try:
            full_distance_matrix = DistanceMatrix(df_dist.values, ids=common_ids)
            pcoa_results = pcoa(full_distance_matrix)
        except Exception as e:
            st.error(f"🚩 PCoA 計算失敗：{e}")
            st.stop()

        coords = pcoa_results.samples.reset_index().rename(columns={"index": "SampleID"})
        df_merged = pd.merge(coords, df_meta, on="SampleID", how="inner")

        pc_cols = [col for col in df_merged.columns if col.startswith("PC")]
        if len(pc_cols) < 2:
            st.error("🚩 PCoA 結果中找不到足夠的 PC 軸。")
            st.stop()

        meta_cols = [col for col in df_meta.columns if col != "SampleID"]
        if len(meta_cols) == 0:
            st.error("🚩 Metadata 沒有可用的分組或臨床變數欄位。")
            st.stop()

        controls = self._sidebar_controls(pc_cols, meta_cols)
        color_var = controls["color_var"]

        df_merged = df_merged[
            df_merged[color_var].notna()
            & (df_merged[color_var].astype(str).str.strip() != "")
        ].copy()

        if df_merged.empty:
            st.error("🚩 上色變數無有效資料。")
            st.stop()

        plot_kind, palette, color_args, df_merged = self._prepare_color_variable(
            df_merged=df_merged,
            color_var=color_var,
            mode=controls["mode"],
        )

        x_axis = controls["x_axis"]
        y_axis = controls["y_axis"]

        x_vals_tmp = (df_merged[x_axis] * (-1 if controls["reverse_x"] else 1)).astype(float)
        y_vals_tmp = (df_merged[y_axis] * (-1 if controls["reverse_y"] else 1)).astype(float)

        xlim, ylim = self._get_axis_limits(
            axis_mode=controls["axis_mode"],
            x_vals_tmp=x_vals_tmp,
            y_vals_tmp=y_vals_tmp,
        )

        render_settings_strip(
            x_axis=x_axis,
            y_axis=y_axis,
            color_var=color_var,
            plot_kind=plot_kind,
            view_mode=controls["view_mode"],
            perm_count=controls["perm_count"],
        )

        self._render_visualization(
            df_merged=df_merged,
            pcoa_results=pcoa_results,
            x_vals_tmp=x_vals_tmp,
            y_vals_tmp=y_vals_tmp,
            x_axis=x_axis,
            y_axis=y_axis,
            color_var=color_var,
            plot_kind=plot_kind,
            palette=palette,
            color_args=color_args,
            xlim=xlim,
            ylim=ylim,
            controls=controls,
        )

        self._run_statistics(
            df_merged=df_merged,
            full_distance_matrix=full_distance_matrix,
            x_axis=x_axis,
            y_axis=y_axis,
            color_var=color_var,
            plot_kind=plot_kind,
            perm_count=controls["perm_count"],
            random_seed=controls["random_seed"],
        )

        self._batch_analysis(
            df_meta=df_meta,
            meta_cols=meta_cols,
            full_distance_matrix=full_distance_matrix,
            perm_count=controls["perm_count"],
            random_seed=controls["random_seed"],
        )

    def _read_distance_matrix(self, distance_file: UploadedFile) -> pd.DataFrame:
        try:
            df_dist = pd.read_csv(distance_file, sep=None, engine="python", index_col=0)
        except Exception as e:
            st.error(f"🚩 距離矩陣讀取失敗：{e}")
            st.stop()

        df_dist.index = df_dist.index.astype(str).str.strip()
        df_dist.columns = df_dist.columns.astype(str).str.strip()

        try:
            df_dist = df_dist.loc[df_dist.index, df_dist.columns]
        except Exception:
            pass

        return df_dist

    def _read_metadata(self, metadata_file: UploadedFile) -> pd.DataFrame:
        try:
            if metadata_file.name.endswith(".xlsx"):
                df_meta = pd.read_excel(metadata_file, engine="openpyxl", dtype=str)
            else:
                df_meta = pd.read_csv(metadata_file, dtype=str)
        except Exception as e:
            st.error(f"🚩 Metadata 讀取失敗：{e}")
            st.stop()

        if df_meta.empty or df_meta.shape[1] < 2:
            st.error("🚩 Metadata 至少需要 SampleID 欄位與一個分組或臨床變數欄位。")
            st.stop()

        df_meta.columns.values[0] = "SampleID"
        df_meta["SampleID"] = df_meta["SampleID"].astype(str).str.strip()
        return df_meta.set_index("SampleID")

    def _sidebar_controls(self, pc_cols: list[str], meta_cols: list[str]) -> dict[str, Any]:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 2. Visualization")

            x_axis = st.selectbox("X axis", pc_cols, index=0, key="x_axis")
            y_axis = st.selectbox("Y axis", pc_cols, index=1, key="y_axis")
            reverse_x = st.checkbox("Reverse X axis", value=False, key="reverse_x")
            reverse_y = st.checkbox("Reverse Y axis", value=False, key="reverse_y")
            color_var = st.selectbox("Color variable", meta_cols, key="color_var")

            mode = st.radio(
                "Variable type",
                ["自動偵測", "類別型", "連續型"],
                index=0,
                key="data_type_mode",
            )

            view_mode = st.radio("Display mode", ["2D", "3D"], index=0, key="view_mode")

            st.markdown("---")
            st.markdown("### 3. Figure Style")

            lock_aspect = st.checkbox("Publication frame", value=True, key="lock_aspect")
            fig_width = st.number_input("Figure width", min_value=4.0, max_value=30.0, value=6.0, step=0.5, key="fig_width")
            fig_height = st.number_input("Figure height", min_value=4.0, max_value=30.0, value=6.0, step=0.5, key="fig_height")
            marker_size = st.number_input("Marker size", min_value=10, max_value=600, value=100, step=10, key="marker_size")
            spine_width = st.number_input("Frame line width", min_value=0.0, max_value=10.0, value=1.5, step=0.5, key="spine_width")

            edge_color = st.selectbox("Marker edge color", ["white", "black", "none"], index=0, key="edge_color")
            point_alpha = st.number_input("Point transparency", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="point_alpha")
            show_title = st.checkbox("Show title", value=False, key="show_title")
            show_legend_box = st.checkbox("Legend frame", value=False, key="show_legend_box")
            force_equal = st.checkbox("Equal PCoA aspect", value=False, key="force_equal")

            axis_mode = st.radio(
                "Axis range",
                ["自動（每次資料不同）", "固定等比例（推薦）", "手動固定"],
                index=0,
                key="axis_mode",
            )

            st.markdown("---")
            st.markdown("### 4. Statistics")

            perm_count = st.number_input("Permutation", min_value=10, step=100, value=999, key="perm_count")
            random_seed = st.number_input("Random seed", min_value=1, value=42, step=1, key="random_seed")

        return {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "reverse_x": reverse_x,
            "reverse_y": reverse_y,
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
            "axis_mode": axis_mode,
            "perm_count": perm_count,
            "random_seed": random_seed,
        }

    def _prepare_color_variable(
        self,
        df_merged: pd.DataFrame,
        color_var: str,
        mode: str,
    ) -> tuple[str, str, dict[str, Any], pd.DataFrame]:
        if mode == "類別型" or (mode == "自動偵測" and df_merged[color_var].nunique() <= 10):
            df_merged[color_var] = df_merged[color_var].astype(str)
            palette = st.sidebar.selectbox("Categorical palette", ["Set1", "Set2", "tab10", "Dark2"], index=0, key="categorical_palette")
            return "categorical", palette, {}, df_merged

        df_merged[color_var] = pd.to_numeric(df_merged[color_var], errors="coerce")
        df_merged = df_merged[df_merged[color_var].notna()].copy()

        if df_merged.empty:
            st.error("🚩 此連續變數無法轉換成有效數值。")
            st.stop()

        palette = st.sidebar.selectbox("Continuous palette", ["viridis", "plasma", "cividis"], index=0, key="continuous_palette")
        return "continuous", palette, {"color_continuous_scale": palette}, df_merged

    def _get_axis_limits(
        self,
        axis_mode: str,
        x_vals_tmp: pd.Series,
        y_vals_tmp: pd.Series,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if axis_mode == "固定等比例（推薦）":
            pad_ratio = st.sidebar.slider("Padding ratio", 0.0, 0.5, 0.10, 0.01, key="pad_ratio")
            limit = float(np.max(np.abs(np.r_[x_vals_tmp.values, y_vals_tmp.values])) * (1.0 + pad_ratio))
            return (-limit, limit), (-limit, limit)

        if axis_mode == "手動固定":
            x_min = st.sidebar.number_input("X min", value=float(np.min(x_vals_tmp.values)), key="x_min")
            x_max = st.sidebar.number_input("X max", value=float(np.max(x_vals_tmp.values)), key="x_max")
            y_min = st.sidebar.number_input("Y min", value=float(np.min(y_vals_tmp.values)), key="y_min")
            y_max = st.sidebar.number_input("Y max", value=float(np.max(y_vals_tmp.values)), key="y_max")
            return (x_min, x_max), (y_min, y_max)

        return None, None

    def _render_visualization(
        self,
        df_merged: pd.DataFrame,
        pcoa_results: Any,
        x_vals_tmp: pd.Series,
        y_vals_tmp: pd.Series,
        x_axis: str,
        y_axis: str,
        color_var: str,
        plot_kind: str,
        palette: str,
        color_args: dict[str, Any],
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        controls: dict[str, Any],
    ) -> None:
        st.markdown('<div class="section-title">PCoA Visualization</div>', unsafe_allow_html=True)
        st.caption("2D output preserves your original publication settings: fixed frame, marker style, legend position, and 1200 dpi PNG export.")

        if controls["view_mode"] == "2D":
            self._render_2d_plot(
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
            self._render_3d_plot(df_merged, color_var, color_args)

    def _render_2d_plot(
        self,
        df_merged: pd.DataFrame,
        pcoa_results: Any,
        x_vals_tmp: pd.Series,
        y_vals_tmp: pd.Series,
        x_axis: str,
        y_axis: str,
        color_var: str,
        plot_kind: str,
        palette: str,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        controls: dict[str, Any],
    ) -> None:
        fig_width = controls["fig_width"]
        fig_height = controls["fig_height"]
        lock_aspect = controls["lock_aspect"]

        if lock_aspect:
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Preserve the original fixed-frame logic.
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
            "📎 Download 2D PNG, 1200 dpi",
            data=buf,
            file_name=f"{safe_key(color_var)}_PCoA.png",
            mime="image/png",
            key=f"download_2d_png_{safe_key(color_var)}",
        )

        plt.close(fig)

    def _render_3d_plot(self, df_merged: pd.DataFrame, color_var: str, color_args: dict[str, Any]) -> None:
        has_pc1_to_3 = all(pc in df_merged.columns for pc in ["PC1", "PC2", "PC3"])

        if not has_pc1_to_3:
            st.info("⚠️ 無法進行 3D 繪圖，因為缺少 PC1、PC2 或 PC3。")
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
            label="📎 Download 3D interactive HTML",
            data=html_bytes,
            file_name=f"{safe_key(color_var)}_3D_PCoA.html",
            mime="text/html",
            key=f"download_3d_html_{safe_key(color_var)}",
        )

        st.info("3D export uses interactive HTML to avoid Kaleido / Chrome errors on Streamlit Cloud.")

    def _run_statistics(
        self,
        df_merged: pd.DataFrame,
        full_distance_matrix: DistanceMatrix,
        x_axis: str,
        y_axis: str,
        color_var: str,
        plot_kind: str,
        perm_count: int,
        random_seed: int,
    ) -> None:
        st.markdown("---")
        st.markdown('<div class="section-title">Statistical Result</div>', unsafe_allow_html=True)
        st.caption("Categorical variables use ANOSIM; continuous variables use Mantel test.")

        selected_coords = df_merged[["SampleID", x_axis, y_axis]].copy()
        distance_matrix = full_distance_matrix.filter(df_merged["SampleID"].tolist())

        random.seed(random_seed)
        np.random.seed(random_seed)

        if plot_kind == "categorical":
            group_series = df_merged.set_index("SampleID").loc[selected_coords["SampleID"], color_var]

            if group_series.nunique() < 2:
                st.warning("⚠️ 此類別變數少於 2 組，無法進行 ANOSIM。")
                return

            try:
                result = anosim(distance_matrix, group_series, permutations=perm_count)
                stat = float(result["test statistic"])
                p_value = float(result["p-value"])

                r1, r2, r3 = st.columns(3)
                r1.metric("Method", "ANOSIM")
                r2.metric("Statistic R", f"{stat:.4f}")
                r3.metric("P-value", f"{p_value:.4g}")

                st.info(interpret_statistics("ANOSIM", stat, p_value))
            except Exception as e:
                st.error(f"🚩 ANOSIM 計算失敗：{e}")

        else:
            try:
                meta_dist = squareform(pdist(df_merged[[color_var]].values, metric="euclidean"))
                meta_matrix = DistanceMatrix(meta_dist, ids=df_merged["SampleID"])
                stat, p_value, _ = mantel(distance_matrix, meta_matrix, permutations=perm_count)
                stat = float(stat)
                p_value = float(p_value)

                r1, r2, r3 = st.columns(3)
                r1.metric("Method", "Mantel test")
                r2.metric("Statistic R", f"{stat:.4f}")
                r3.metric("P-value", f"{p_value:.4g}")

                st.info(interpret_statistics("Mantel", stat, p_value))
            except Exception as e:
                st.error(f"🚩 Mantel test 計算失敗：{e}")

    def _batch_analysis(
        self,
        df_meta: pd.DataFrame,
        meta_cols: list[str],
        full_distance_matrix: DistanceMatrix,
        perm_count: int,
        random_seed: int,
    ) -> None:
        st.markdown("---")
        st.markdown('<div class="section-title">Batch Clinical Association Analysis</div>', unsafe_allow_html=True)
        st.write("Automatically scan all metadata variables to identify potential beta-diversity associations.")

        if not st.button("🚀 Run batch analysis", key="batch_analysis_button"):
            return

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
                    dtype = "類別型 (Categorical)"

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
                    dtype = "連續型 (Continuous)"

                except Exception:
                    progress_bar.progress((i + 1) / total_vars)
                    continue

            results.append(
                {
                    "變數名稱 (Variable)": var,
                    "變數型態 (Data Type)": dtype,
                    "統計方法 (Method)": method,
                    "有效樣本數 (N)": n_val,
                    "Statistic (R)": stat_val,
                    "P-value": p_val,
                }
            )

            progress_bar.progress((i + 1) / total_vars)

        status_text.text("✅ Batch analysis completed.")

        if not results:
            st.warning("沒有成功完成任何變數的分析，可能是樣本數不足、分群異常或資料型態不適合。")
            return

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values("P-value", ascending=True).reset_index(drop=True)

        st.dataframe(df_results, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_results.to_excel(writer, index=False, sheet_name="Statistical_Results")

        output.seek(0)

        st.download_button(
            label="📥 Download Excel report",
            data=output,
            file_name="Batch_PCoA_Statistics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_batch_excel",
        )


if __name__ == "__main__":
    main()

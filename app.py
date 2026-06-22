import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix, anosim, mantel
import io

# 讓 colorbar 跟主圖同高、完全對齊

from mpl_toolkits.axes_grid1 import make_axes_locatable

APP_VERSION = "BetaSuite Apple-like Cloud Safe 2026-06-22"

st.set_page_config(
page_title="BetaSuite | PCoA Analysis",
page_icon="🧬",
layout="wide"
)

def inject_apple_like_css():
st.markdown(
""" <style>
.stApp {
background: linear-gradient(180deg, #F5F5F7 0%, #FFFFFF 100%);
color: #1D1D1F;
}

```
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        .hero {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(0, 0, 0, 0.06);
            border-radius: 30px;
            padding: 42px 46px;
            margin-bottom: 28px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(20px);
        }

        .hero .eyebrow {
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #007AFF;
            margin-bottom: 12px;
        }

        .hero h1 {
            font-size: 48px;
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.045em;
            margin: 0 0 16px 0;
            color: #1D1D1F;
        }

        .hero p {
            font-size: 19px;
            line-height: 1.55;
            color: #6E6E73;
            max-width: 900px;
            margin: 0;
        }

        .section-title {
            font-size: 25px;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #1D1D1F;
            margin-top: 10px;
            margin-bottom: 14px;
        }

        .section-subtitle {
            color: #6E6E73;
            font-size: 15px;
            margin-bottom: 18px;
        }

        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(0, 0, 0, 0.06);
            border-radius: 22px;
            padding: 18px 20px;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.045);
        }

        div.stDownloadButton > button,
        div.stButton > button {
            border-radius: 999px;
            border: none;
            background: #007AFF;
            color: white;
            font-weight: 700;
            padding: 0.58rem 1.18rem;
            box-shadow: 0 10px 24px rgba(0, 122, 255, 0.20);
        }

        div.stDownloadButton > button:hover,
        div.stButton > button:hover {
            background: #0066CC;
            color: white;
            border: none;
        }

        section[data-testid="stSidebar"] {
            background: #F5F5F7;
            border-right: 1px solid rgba(0, 0, 0, 0.06);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #1D1D1F;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 14px;
        }

        .small-caption {
            color: #86868B;
            font-size: 13px;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
```

def safe_key(text: str) -> str:
return "".join(ch if ch.isalnum() else "_" for ch in str(text))

def main():
inject_apple_like_css()

```
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">BetaSuite</div>
        <h1>Microbiome Beta Diversity Analysis Platform</h1>
        <p>
            Upload your distance matrix and metadata to generate publication-ready PCoA plots,
            ANOSIM statistics, Mantel tests, and batch clinical association reports.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 🧬 BetaSuite")
    st.caption(APP_VERSION)

    st.markdown("---")
    st.markdown("### 1. Data Input")

    distance_file: UploadedFile = st.file_uploader(
        "📂 上傳距離矩陣 (.tsv / .csv)",
        type=["tsv", "csv"],
        key="distance_matrix_uploader"
    )

    metadata_file: UploadedFile = st.file_uploader(
        "📂 上傳 metadata (.xlsx / .csv)",
        type=["xlsx", "csv"],
        key="metadata_uploader"
    )

if distance_file is None or metadata_file is None:
    st.info("📥 請依序上傳距離矩陣與 metadata 檔案。")
    st.markdown(
        """
        <div class="section-title">Workflow</div>
        <div class="section-subtitle">
            Distance matrix + metadata → PCoA visualization → ANOSIM / Mantel test → batch clinical association report.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Step 1", "Upload files")
    c2.metric("Step 2", "Generate PCoA")
    c3.metric("Step 3", "Export results")
    return

Pipeline().main(distance_file, metadata_file)
```

class Pipeline:
def main(self, distance_file: UploadedFile, metadata_file: UploadedFile):
try:
df_dist = pd.read_csv(distance_file, sep=None, engine="python", index_col=0)
except Exception as e:
st.error(f"🚩 距離矩陣讀取失敗：{e}")
st.stop()

```
    df_dist.index = df_dist.index.astype(str).str.strip()
    df_dist.columns = df_dist.columns.astype(str).str.strip()

    try:
        df_dist = df_dist.loc[df_dist.index, df_dist.columns]
    except Exception:
        pass

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
    df_meta = df_meta.set_index("SampleID")

    dist_ids_set = set(df_dist.index)
    meta_ids_set = set(df_meta.index)
    common_ids = [i for i in df_dist.index if i in df_meta.index]

    in_dist_not_meta = dist_ids_set - meta_ids_set
    in_meta_not_dist = meta_ids_set - dist_ids_set

    st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Distance samples", len(df_dist.index))
    d2.metric("Metadata samples", len(df_meta.index))
    d3.metric("Matched samples", len(common_ids))
    d4.metric("Variables", len(df_meta.columns))

    if in_dist_not_meta or in_meta_not_dist:
        with st.expander("⚠️ 警告：部分樣本因為兩份檔案的 ID 對不上，已被自動剔除", expanded=True):
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

    if len(common_ids) < 3:
        st.error("🚩 距離矩陣與 metadata 可配對樣本數不足，至少需要 3 個共同 SampleID 才能進行 PCoA。")
        st.stop()

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

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 2. Visualization")

        x_axis = st.selectbox("選擇 X 軸", pc_cols, index=0, key="x_axis")
        y_axis = st.selectbox("選擇 Y 軸", pc_cols, index=1, key="y_axis")
        reverse_x = st.checkbox("反轉 X 軸", value=False, key="reverse_x")
        reverse_y = st.checkbox("反轉 Y 軸", value=False, key="reverse_y")

        color_var = st.selectbox("選擇上色變數", meta_cols, key="color_var")

        mode = st.radio(
            "📌 選擇變數型態",
            ["自動偵測", "類別型", "連續型"],
            index=0,
            key="data_type_mode"
        )

        view_mode = st.radio(
            "📐 顯示模式",
            ["2D", "3D"],
            index=0,
            key="view_mode"
        )

        st.markdown("---")
        st.markdown("### 3. Figure Settings")

        lock_aspect = st.checkbox(
            "🔒 啟用頂級發表格式排版",
            value=True,
            key="lock_aspect"
        )

        fig_width = st.number_input(
            "圖形基礎寬度",
            min_value=4.0,
            max_value=30.0,
            value=6.0,
            step=0.5,
            key="fig_width"
        )

        fig_height = st.number_input(
            "圖形基礎高度",
            min_value=4.0,
            max_value=30.0,
            value=6.0,
            step=0.5,
            key="fig_height"
        )

        marker_size = st.number_input(
            "點的尺寸",
            min_value=10,
            max_value=600,
            value=100,
            step=10,
            key="marker_size"
        )

        spine_width = st.number_input(
            "外框線粗細",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            key="spine_width"
        )

        edge_color = st.selectbox(
            "點的外框顏色",
            ["white", "black", "none"],
            index=0,
            key="edge_color"
        )

        point_alpha = st.number_input(
            "點的透明度",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="point_alpha"
        )

        show_title = st.checkbox("顯示上方主標題", value=False, key="show_title")
        show_legend_box = st.checkbox("顯示圖例周圍方框", value=False, key="show_legend_box")

        force_equal = st.checkbox(
            "📐 鎖定 PCoA 真實等比例",
            value=False,
            key="force_equal"
        )

        axis_mode = st.radio(
            "座標範圍模式",
            ["自動（每次資料不同）", "固定等比例（推薦）", "手動固定"],
            index=0,
            key="axis_mode"
        )

        st.markdown("---")
        st.markdown("### 4. Statistics")

        perm_count = st.number_input(
            "Permutation 次數",
            min_value=10,
            step=100,
            value=999,
            key="perm_count"
        )

        random_seed = st.number_input(
            "隨機種子",
            min_value=1,
            value=42,
            step=1,
            key="random_seed"
        )

    chart_title = f"PCoA colored by {color_var}"

    df_merged = df_merged[
        df_merged[color_var].notna()
        & (df_merged[color_var].astype(str).str.strip() != "")
    ].copy()

    if df_merged.empty:
        st.error("🚩 上色變數無有效資料。")
        st.stop()

    if mode == "類別型" or (mode == "自動偵測" and df_merged[color_var].nunique() <= 10):
        df_merged[color_var] = df_merged[color_var].astype(str)
        palette = st.sidebar.selectbox(
            "🎨 色盤（類別型）",
            ["Set1", "Set2", "tab10", "Dark2"],
            index=0,
            key="categorical_palette"
        )
        plot_kind = "categorical"
        color_args = {}
    else:
        df_merged[color_var] = pd.to_numeric(df_merged[color_var], errors="coerce")
        df_merged = df_merged[df_merged[color_var].notna()].copy()

        if df_merged.empty:
            st.error("🚩 此連續變數無法轉換成有效數值。")
            st.stop()

        palette = st.sidebar.selectbox(
            "🎨 色盤（連續型）",
            ["viridis", "plasma", "cividis"],
            index=0,
            key="continuous_palette"
        )
        plot_kind = "continuous"
        color_args = {"color_continuous_scale": palette}

    x_vals_tmp = (df_merged[x_axis] * (-1 if reverse_x else 1)).astype(float)
    y_vals_tmp = (df_merged[y_axis] * (-1 if reverse_y else 1)).astype(float)

    if axis_mode == "固定等比例（推薦）":
        pad_ratio = st.sidebar.slider(
            "邊界留白比例",
            0.0,
            0.5,
            0.10,
            0.01,
            key="pad_ratio"
        )
        limit = float(np.max(np.abs(np.r_[x_vals_tmp.values, y_vals_tmp.values])) * (1.0 + pad_ratio))
        xlim = (-limit, limit)
        ylim = (-limit, limit)

    elif axis_mode == "手動固定":
        x_min = st.sidebar.number_input("X 最小值", value=float(np.min(x_vals_tmp.values)), key="x_min")
        x_max = st.sidebar.number_input("X 最大值", value=float(np.max(x_vals_tmp.values)), key="x_max")
        y_min = st.sidebar.number_input("Y 最小值", value=float(np.min(y_vals_tmp.values)), key="y_min")
        y_max = st.sidebar.number_input("Y 最大值", value=float(np.max(y_vals_tmp.values)), key="y_max")
        xlim = (x_min, x_max)
        ylim = (y_min, y_max)

    else:
        xlim = None
        ylim = None

    st.markdown('<div class="section-title">PCoA Visualization</div>', unsafe_allow_html=True)
    st.caption("2D 圖形輸出格式已保留原本設定：固定方框、外框線、點大小、legend 與 1200 dpi PNG。")

    if view_mode == "2D":
        if lock_aspect:
            fig = plt.figure(figsize=(fig_width, fig_height))

            square_size = fig_height * 0.75
            ax_w = square_size / fig_width
            ax_h = square_size / fig_height
            left_margin = 0.15
            bottom_margin = 0.15
            ax = fig.add_axes([left_margin, bottom_margin, ax_w, ax_h])

        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        x_vals = x_vals_tmp
        y_vals = y_vals_tmp

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

        if plot_kind == "categorical":
            sns.scatterplot(
                x=x_vals,
                y=y_vals,
                hue=df_merged[color_var],
                palette=palette,
                s=marker_size,
                edgecolor=edge_color if edge_color != "none" else None,
                linewidth=1.2 if edge_color != "none" else 0,
                alpha=point_alpha,
                ax=ax,
                zorder=3
            )

            ax.legend(
                bbox_to_anchor=(1.05, 1.0),
                loc="upper left",
                borderaxespad=0.0,
                frameon=show_legend_box,
                fontsize=16,
                markerscale=1.0
            )

        else:
            sc = ax.scatter(
                x_vals,
                y_vals,
                c=df_merged[color_var],
                cmap=palette,
                s=marker_size,
                edgecolors=edge_color if edge_color != "none" else "none",
                linewidths=1.2 if edge_color != "none" else 0,
                alpha=point_alpha,
                zorder=3
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

        if show_title:
            ax.set_title(chart_title, fontsize=24, pad=15)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
            length=8,
            width=2.5,
            direction="out"
        )

        if xlim is not None and ylim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        if force_equal:
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")

        for spine in ax.spines.values():
            if spine_width > 0:
                spine.set_visible(True)
                spine.set_linewidth(spine_width)
            else:
                spine.set_visible(False)

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=1200, bbox_inches="tight", transparent=False)
        buf.seek(0)

        st.download_button(
            "📎 下載 2D 圖檔 (PNG, 1200 dpi)",
            data=buf,
            file_name=f"{safe_key(color_var)}_PCoA.png",
            mime="image/png",
            key=f"download_2d_png_{safe_key(color_var)}"
        )

        plt.close(fig)

    elif view_mode == "3D":
        has_pc1_to_3 = all(pc in df_merged.columns for pc in ["PC1", "PC2", "PC3"])

        if has_pc1_to_3:
            fig3d = px.scatter_3d(
                df_merged,
                x="PC1",
                y="PC2",
                z="PC3",
                color=color_var,
                title=chart_title,
                labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"},
                **color_args
            )

            st.plotly_chart(fig3d, use_container_width=True)

            html_bytes = fig3d.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

            st.download_button(
                label="📎 下載 3D 互動圖檔 (HTML)",
                data=html_bytes,
                file_name=f"{safe_key(color_var)}_3D_PCoA.html",
                mime="text/html",
                key=f"download_3d_html_{safe_key(color_var)}"
            )

            st.info("3D 圖已改為 HTML 互動格式下載，可避免 Streamlit Cloud 因 Kaleido / Chrome 缺少而發生錯誤。")

        else:
            st.info("⚠️ 無法進行 3D 繪圖，因為缺少 PC1、PC2 或 PC3。")

    st.markdown("---")
    st.markdown('<div class="section-title">Statistical Result</div>', unsafe_allow_html=True)
    st.caption("✅ 類別變因 → ANOSIM；連續變因 → Mantel test")

    selected_coords = df_merged[["SampleID", x_axis, y_axis]].copy()
    distance_matrix = full_distance_matrix.filter(df_merged["SampleID"].tolist())

    random.seed(random_seed)
    np.random.seed(random_seed)

    if plot_kind == "categorical":
        group_series = df_merged.set_index("SampleID").loc[selected_coords["SampleID"], color_var]

        if group_series.nunique() < 2:
            st.warning("⚠️ 此類別變數少於 2 組，無法進行 ANOSIM。")

        else:
            try:
                result = anosim(distance_matrix, group_series, permutations=perm_count)

                r1, r2, r3 = st.columns(3)
                r1.metric("Method", "ANOSIM")
                r2.metric("Statistic R", f'{result["test statistic"]:.4f}')
                r3.metric("P-value", f'{result["p-value"]:.4g}')

            except Exception as e:
                st.error(f"🚩 ANOSIM 計算失敗：{e}")

    else:
        try:
            meta_dist = squareform(pdist(df_merged[[color_var]].values, metric="euclidean"))
            meta_matrix = DistanceMatrix(meta_dist, ids=df_merged["SampleID"])
            stat, p_value, _ = mantel(distance_matrix, meta_matrix, permutations=perm_count)

            r1, r2, r3 = st.columns(3)
            r1.metric("Method", "Mantel test")
            r2.metric("Statistic R", f"{stat:.4f}")
            r3.metric("P-value", f"{p_value:.4g}")

            st.caption("🔍 Mantel test 是用來檢驗兩個距離矩陣之間的相關性，適用於連續變數。")

        except Exception as e:
            st.error(f"🚩 Mantel test 計算失敗：{e}")

    st.markdown("---")
    st.markdown('<div class="section-title">Batch Clinical Association Analysis</div>', unsafe_allow_html=True)
    st.write("可自動遍歷所有 metadata 變數，找出與 beta diversity 可能相關的臨床或分組變項。")

    if st.button("🚀 開始批次檢定", key="batch_analysis_button"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_vars = len(meta_cols)

        for i, var in enumerate(meta_cols):
            status_text.text(f"正在分析 ({i + 1}/{total_vars}): {var} ...")

            df_curr = df_meta[["SampleID", var]].copy()
            df_curr = df_curr[
                df_curr[var].notna()
                & (df_curr[var].astype(str).str.strip() != "")
            ]

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
                test_type = "Categorical (ANOSIM)"

                try:
                    group_series = df_curr[var].astype(str)

                    if group_series.nunique() < 2:
                        progress_bar.progress((i + 1) / total_vars)
                        continue

                    res = anosim(curr_dist, group_series, permutations=perm_count)
                    stat_val = res["test statistic"]
                    p_val = res["p-value"]
                    n_val = len(group_series)

                except Exception:
                    progress_bar.progress((i + 1) / total_vars)
                    continue

            else:
                test_type = "Continuous (Mantel)"

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

                except Exception:
                    progress_bar.progress((i + 1) / total_vars)
                    continue

            results.append(
                {
                    "變數名稱 (Variable)": var,
                    "變數型態 (Data Type)": "類別型 (Categorical)" if "Categorical" in test_type else "連續型 (Continuous)",
                    "統計方法 (Method)": "ANOSIM" if "Categorical" in test_type else "Mantel test",
                    "有效樣本數 (N)": n_val,
                    "Statistic (R)": stat_val,
                    "P-value": p_val,
                }
            )

            progress_bar.progress((i + 1) / total_vars)

        status_text.text("✅ 批次分析完成！")

        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values("P-value", ascending=True).reset_index(drop=True)

            st.dataframe(df_results, use_container_width=True)

            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_results.to_excel(writer, index=False, sheet_name="Statistical_Results")

            output.seek(0)

            st.download_button(
                label="📥 下載 Excel 分析報告",
                data=output,
                file_name="Batch_PCoA_Statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_batch_excel"
            )

        else:
            st.warning("沒有成功完成任何變數的分析，可能是樣本數不足、分群異常或資料型態不適合。")
```

if **name** == "**main**":
main()

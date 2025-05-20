import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix, anosim, mantel
import io


def main():

    st.set_page_config(page_title='PCoA GUI v27', layout='wide')
    st.title('🧬 PCoA GUI v27 (ANOSIM / Mantel 自動判斷 + 高畫質圖表輸出)')
    st.caption('✨ 自動 PCoA + 變數型態統計分析 + 高品質 2D / 3D 輸出')

    distance_file: UploadedFile = st.file_uploader('📂 上傳距離矩陣 (.tsv / .csv)', type=['tsv', 'csv'])
    metadata_file: UploadedFile = st.file_uploader('📂 上傳 metadata (.xlsx / .csv)', type=['xlsx', 'csv'])

    if distance_file is None or metadata_file is None:
        st.info('📥 請依序上傳距離矩陣與 metadata 檔案')
        return

    try:
        pipeline(distance_file=distance_file, metadata_file=metadata_file)
    except Exception as e:
        st.error(f'❗ 發生錯誤：{e}')


def pipeline(distance_file: UploadedFile, metadata_file: UploadedFile):
    # Read and process distance matrix
    df_dist = pd.read_csv(distance_file, sep=None, engine='python', index_col=0)
    df_dist.index = df_dist.index.astype(str).str.strip()
    df_dist.columns = df_dist.columns.astype(str).str.strip()
    df_dist = df_dist.loc[df_dist.index, df_dist.columns]

    # Read and process metadata
    if metadata_file.name.endswith('.xlsx'):
        df_meta = pd.read_excel(metadata_file, engine='openpyxl', dtype=str)
    else:
        df_meta = pd.read_csv(metadata_file, dtype=str)
    df_meta.columns.values[0] = 'SampleID'
    df_meta['SampleID'] = df_meta['SampleID'].astype(str).str.strip()
    df_meta = df_meta.set_index('SampleID')

    # Intersect distance matrix and metadata with common IDs
    common_ids = [i for i in df_dist.index if i in df_meta.index]
    df_meta = df_meta.loc[common_ids].reset_index()
    df_dist = df_dist.loc[common_ids, common_ids]

    # Run PCoA
    full_distance_matrix = DistanceMatrix(df_dist.values, ids=common_ids)
    pcoa_results = pcoa(full_distance_matrix)
    coords = pcoa_results.samples.reset_index().rename(columns={'index': 'SampleID'})
    df_merged = pd.merge(coords, df_meta, on='SampleID', how='inner')

    # Show PC columns
    pc_cols = [col for col in df_merged.columns if col.startswith('PC')]
    x_axis = st.selectbox('選擇 X 軸', pc_cols, index=0)
    y_axis = st.selectbox('選擇 Y 軸', pc_cols, index=1)
    reverse_x = st.checkbox('反轉 X 軸', value=False)
    reverse_y = st.checkbox('反轉 Y 軸', value=False)

    # Show metadata columns
    meta_cols = [col for col in df_meta.columns if col != 'SampleID']
    st.subheader('🧩 上色變數')
    color_var = st.selectbox('選擇上色變數', meta_cols)

    mode = st.radio('📌 選擇變數型態', ['自動偵測', '類別型', '連續型'], index=0)

    # Filter out empty or whitespace-only values
    df_merged = df_merged[df_merged[color_var].notna() & (df_merged[color_var].astype(str).str.strip() != '')]

    if df_merged.empty:
        st.error('🚩 上色變數無有效資料')
        st.stop()

    if mode == '類別型' or (mode == '自動偵測' and df_merged[color_var].nunique() <= 10):
        df_merged[color_var] = df_merged[color_var].astype(str)
        palette = st.selectbox('🎨 色盤（類別型）', ['Set1', 'Set2', 'tab10', 'Dark2'])
        plot_kind = 'categorical'
        color_args = {}
    else:
        df_merged[color_var] = pd.to_numeric(df_merged[color_var], errors='coerce')
        palette = st.selectbox('🎨 色盤（連續型）', ['viridis', 'plasma', 'cividis'])
        plot_kind = 'continuous'
        color_args = {'color_continuous_scale': palette}

    view_mode = st.radio('📐 顯示模式', ['2D', '3D'], index=0)
    chart_title = f'PCoA colored by {color_var}'

    if view_mode == '2D':
        fig, ax = plt.subplots(figsize=(8, 6))
        x_vals = df_merged[x_axis] * (-1 if reverse_x else 1)
        y_vals = df_merged[y_axis] * (-1 if reverse_y else 1)

        if plot_kind == 'categorical':
            sns.scatterplot(
                x=x_vals,
                y=y_vals,
                hue=df_merged[color_var],
                palette=palette,
                s=60,
                edgecolor='black',
                ax=ax
            )
            ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sc = ax.scatter(
                x_vals,
                y_vals,
                c=df_merged[color_var],
                cmap=palette,
                s=60,
                edgecolors='black'
            )
            plt.colorbar(sc, ax=ax).set_label(color_var)

        var_x = pcoa_results.proportion_explained[x_axis] * 100
        var_y = pcoa_results.proportion_explained[y_axis] * 100
        ax.set_xlabel(f'{x_axis} ({var_x:.1f}%)', fontsize=13)
        ax.set_ylabel(f'{y_axis} ({var_y:.1f}%)', fontsize=13)
        ax.set_title(chart_title, fontsize=14)
        sns.despine()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=1200)
        st.download_button(
            '📎 下載 2D 圖檔 (PNG, 1200 dpi)',
            data=buf.getvalue(),
            file_name=f'{color_var}_PCoA.png',
            mime='image/png'
        )

    elif view_mode == '3D':

        has_pc1_to_3 = True
        for pc in ['PC1', 'PC2', 'PC3']:
            if pc not in df_merged.columns:
                has_pc1_to_3 = False
                break

        if has_pc1_to_3:
            fig3d = px.scatter_3d(
                df_merged, x='PC1', y='PC2', z='PC3',
                color=color_var,
                title=chart_title,
                labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
                **color_args
            )
            st.plotly_chart(fig3d, use_container_width=True)

            for fmt in ['png', 'pdf', 'svg']:
                image_bytes = fig3d.to_image(format=fmt, scale=5)
                st.download_button(
                    f'📎 下載 3D 圖檔 ({fmt.upper()})',
                    data=image_bytes,
                    file_name=f"{color_var}_3D_PCoA.{fmt}",
                    mime='image/svg+xml' if fmt == 'svg' else f'image/{fmt}',
                )
        else:
            st.info('⚠️ 無法進行 3D 繪圖（缺少 PC1~PC3）')

    st.subheader('📊 統計分析結果')
    perm_count = st.number_input('Permutation 次數', min_value=10, step=100, value=999)
    random_seed = st.number_input('隨機種子', min_value=1, value=42, step=1)
    st.caption('✅ 類別變因 → ANOSIM；連續變因 → Mantel test')
    try:
        selected_coords = df_merged[['SampleID', x_axis, y_axis]].copy()
        coord_matrix = squareform(pdist(selected_coords[[x_axis, y_axis]], metric='euclidean'))
        distance_matrix = DistanceMatrix(coord_matrix, ids=selected_coords['SampleID'])

        if plot_kind == 'categorical':
            group_series = df_merged.set_index('SampleID').loc[selected_coords['SampleID'], color_var]
            result = anosim(distance_matrix, group_series, permutations=perm_count)
            st.success(f'ANOSIM R = {result["test statistic"]:.4f}, p = {result["p-value"]:.4g}')
        else:
            np.random.seed(random_seed)
            meta_dist = squareform(pdist(df_merged[[color_var]].values, metric='euclidean'))
            meta_matrix = DistanceMatrix(meta_dist, ids=df_merged['SampleID'])
            stat, p_value, _ = mantel(distance_matrix, meta_matrix, permutations=perm_count)
            st.success(f'Mantel test R = {stat:.4f}, p = {p_value:.4g}')
    except Exception as e:
        st.warning(f'⚠️ 統計分析失敗：{e}')


main()

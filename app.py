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

# ✅ NEW: 讓 colorbar 跟主圖同高、完全對齊
from mpl_toolkits.axes_grid1 import make_axes_locatable

st.set_page_config(page_title='PCoA GUI', layout='wide')


def main():
    st.title('🧬 PCoA GUI ')

    distance_file: UploadedFile = st.file_uploader(
        '📂 上傳距離矩陣 (.tsv / .csv)',
        type=['tsv', 'csv'],
        key='distance_matrix_uploader'
    )
    metadata_file: UploadedFile = st.file_uploader(
        '📂 上傳 metadata (.xlsx / .csv)',
        type=['xlsx', 'csv'],
        key='metadata_uploader'
    )

    if distance_file is None or metadata_file is None:
        st.info('📥 請依序上傳距離矩陣與 metadata 檔案')
        return

    Pipeline().main(distance_file, metadata_file)


class Pipeline:
    def main(self, distance_file: UploadedFile, metadata_file: UploadedFile):
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
        dist_ids_set = set(df_dist.index)
        meta_ids_set = set(df_meta.index)
        common_ids = [i for i in df_dist.index if i in df_meta.index]

        # ✅ NEW: 聰明的錯誤提示系統，直接在畫面上印出誰對不起來
        in_dist_not_meta = dist_ids_set - meta_ids_set
        in_meta_not_dist = meta_ids_set - dist_ids_set
        
        if in_dist_not_meta or in_meta_not_dist:
            with st.expander("⚠️ 警告：部分樣本因為兩份檔案的 ID 對不上，已被自動剔除 (點擊展開看是誰)", expanded=True):
                st.write(f"✅ 成功完美配對出的樣本數: **{len(common_ids)}** 個")
                if in_dist_not_meta:
                    st.error(f"📍 以下 {len(in_dist_not_meta)} 個樣本出現在【距離矩陣】，但【Metadata】裡找不到 (請檢查大小寫與空白)：\n\n" + ", ".join(in_dist_not_meta))
                if in_meta_not_dist:
                    st.warning(f"📍 以下 {len(in_meta_not_dist)} 個樣本出現在【Metadata】，但【距離矩陣】裡找不到：\n\n" + ", ".join(in_meta_not_dist))

        if len(common_ids) < 3:
            st.error('🚩 距離矩陣與 metadata 可配對樣本數不足，至少需要 3 個共同 SampleID 才能進行 PCoA。')
            st.stop()

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

        # ✅ 2D 顯示範圍與客製化設定
        st.subheader('📏 2D 圖形顯示與進階美化 (期刊格式)')
        lock_aspect = st.checkbox('🔒 啟用「頂級發表格式」排版 (主圖不受圖例擠壓)', value=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig_width = st.number_input('圖形基礎寬度', min_value=4.0, max_value=30.0, value=6.0, step=0.5)
        with col2:
            fig_height = st.number_input('圖形基礎高度', min_value=4.0, max_value=30.0, value=6.0, step=0.5)
        with col3:
            marker_size = st.number_input('點的尺寸 (Marker Size)', min_value=10, max_value=600, value=100, step=10)
        with col4:
            spine_width = st.number_input('外框線粗細', min_value=0.0, max_value=10.0, value=1.5, step=0.5)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            edge_color = st.selectbox('點的外框顏色', ['white', 'black', 'none'], index=0)
        with c2:
            point_alpha = st.number_input('點的透明度 (防止重疊)', min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        with c3:
            show_title = st.checkbox('顯示上方主標題', value=False)
        with c4:
            show_legend_box = st.checkbox('顯示圖例周圍方框', value=False)

        force_equal = st.checkbox('📐 鎖定 PCoA 真實等比例 (若打勾：可能導致圖形變長方形；若不勾：圖形會完美填滿正方形)', value=False)
        axis_mode = st.radio('座標範圍模式', ['自動（每次資料不同）', '固定等比例（推薦）', '手動固定'], index=0)

        x_vals_tmp = (df_merged[x_axis] * (-1 if reverse_x else 1)).astype(float)
        y_vals_tmp = (df_merged[y_axis] * (-1 if reverse_y else 1)).astype(float)

        if axis_mode == '固定等比例（推薦）':
            pad_ratio = st.slider('邊界留白比例', 0.0, 0.5, 0.10, 0.01)
            limit = float(np.max(np.abs(np.r_[x_vals_tmp.values, y_vals_tmp.values])) * (1.0 + pad_ratio))
            xlim = (-limit, limit)
            ylim = (-limit, limit)
        elif axis_mode == '手動固定':
            x_min = st.number_input('X 最小值', value=float(np.min(x_vals_tmp.values)))
            x_max = st.number_input('X 最大值', value=float(np.max(x_vals_tmp.values)))
            y_min = st.number_input('Y 最小值', value=float(np.min(y_vals_tmp.values)))
            y_max = st.number_input('Y 最大值', value=float(np.max(y_vals_tmp.values)))
            xlim = (x_min, x_max)
            ylim = (y_min, y_max)
        else:
            xlim = None
            ylim = None

        if view_mode == '2D':
            if lock_aspect:
                fig = plt.figure(figsize=(fig_width, fig_height))
                # 讓方塊以高度為主，佔據畫面約 75% 高度
                square_size = fig_height * 0.75  
                ax_w = square_size / fig_width
                ax_h = square_size / fig_height
                left_margin = 0.15  # 固定左側留白給 Y 軸名稱
                bottom_margin = 0.15 # 固定底側留白給 X 軸名稱
                ax = fig.add_axes([left_margin, bottom_margin, ax_w, ax_h])
            else:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

            x_vals = x_vals_tmp
            y_vals = y_vals_tmp

            # 全局設定使用無襯線字體，看起來更像專業期刊
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

            if plot_kind == 'categorical':
                sns.scatterplot(
                    x=x_vals,
                    y=y_vals,
                    hue=df_merged[color_var],
                    palette=palette,
                    s=marker_size,
                    edgecolor=edge_color if edge_color != 'none' else None,
                    linewidth=1.2 if edge_color != 'none' else 0,
                    alpha=point_alpha,
                    ax=ax,
                    zorder=3
                )
                ax.legend(
                    # 不強制顯示 title 讓畫面更乾淨
                    bbox_to_anchor=(1.05, 1.0),
                    loc='upper left',
                    borderaxespad=0.0,
                    frameon=show_legend_box,
                    fontsize=16,
                    markerscale=1.0 # 因為點的大小設定已經在上面了
                )
            else:
                sc = ax.scatter(
                    x_vals,
                    y_vals,
                    c=df_merged[color_var],
                    cmap=palette,
                    s=marker_size,
                    edgecolors=edge_color if edge_color != 'none' else 'none',
                    linewidths=1.2 if edge_color != 'none' else 0,
                    alpha=point_alpha,
                    zorder=3
                )

                # ✅ 連續型 Colorbar 字體放大與防擠壓處理
                if lock_aspect:
                    # 絕對坐標系添加 Colorbar，百分之百不干擾主圖方框的長寬比例
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
            ax.set_xlabel(f'{x_axis} ({var_x:.2f}%)', fontsize=24)
            ax.set_ylabel(f'{y_axis} ({var_y:.2f}%)', fontsize=24)
            
            if show_title:
                ax.set_title(chart_title, fontsize=24, pad=15)
                
            ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=2.5, direction='out')

            if xlim is not None and ylim is not None:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

            if force_equal:
                ax.set_aspect('equal', adjustable='box')
            else:
                ax.set_aspect('auto')

            for spine in ax.spines.values():
                if spine_width > 0:
                    spine.set_visible(True)
                    spine.set_linewidth(spine_width)
                else:
                    spine.set_visible(False)

            st.pyplot(fig)

            buf = io.BytesIO()
            # 由於我們已經在上面透過鎖定排版保護好內部比例，現在可以安全地開啟 tight 切除討厭的大白邊！
            fig.savefig(buf, format='png', dpi=1200, bbox_inches='tight', transparent=False)
            buf.seek(0)
            st.download_button(
                '📎 下載 2D 圖檔 (PNG, 1200 dpi)',
                data=buf,
                file_name=f'{color_var}_PCoA.png',
                mime='image/png'
            )
            plt.close(fig)

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

        perm_count = st.number_input('Permutation 次數', min_value=10, step=100, value=999)
        random_seed = st.number_input('隨機種子', min_value=1, value=42, step=1)
        st.caption('✅ 類別變因 → ANOSIM；連續變因 → Mantel test')

        if x_axis not in df_merged.columns:
            st.warning(f"⚠️ 無此欄位: {x_axis} 在資料中找不到，請確認欄位名稱。")
        elif y_axis not in df_merged.columns:
            st.warning(f"⚠️ 無此欄位: {y_axis} 在資料中找不到，請確認欄位名稱。")
        elif color_var not in df_merged.columns:
            st.warning(f"⚠️ 無此欄位: {color_var} 在資料中找不到，請確認欄位名稱。")
        else:
            selected_coords = df_merged[['SampleID', x_axis, y_axis]].copy()
            distance_matrix = full_distance_matrix.filter(df_merged['SampleID'].tolist())

            random.seed(random_seed)
            np.random.seed(random_seed)

            if plot_kind == 'categorical':
                group_series = df_merged.set_index('SampleID').loc[selected_coords['SampleID'], color_var]
                result = anosim(distance_matrix, group_series, permutations=perm_count)
                st.success(f'ANOSIM R = {result["test statistic"]:.4f}, p = {result["p-value"]:.4g}')
            else:
                meta_dist = squareform(pdist(df_merged[[color_var]].values, metric='euclidean'))
                meta_matrix = DistanceMatrix(meta_dist, ids=df_merged['SampleID'])
                stat, p_value, _ = mantel(distance_matrix, meta_matrix, permutations=perm_count)
                st.success(f'Mantel test R = {stat:.4f}, p = {p_value:.4g}')
                st.caption('🔍 Mantel test 是用來檢驗兩個距離矩陣之間的相關性，適用於連續變數。')
        # --- Batch Analysis Section ---
        st.markdown('---')
        st.subheader('📊 批次自動統計分析 (一鍵跑完所有變數)')
        st.write('由於臨床變數眾多，您可以使用這個功能自動遍歷所有變數，找出具有顯著差異的組別。')
        if st.button('🚀 開始批次檢定 (這可能需要一段時間)'):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_vars = len(meta_cols)
            for i, var in enumerate(meta_cols):
                status_text.text(f'正在分析 ({i+1}/{total_vars}): {var} ...')
                
                # Copy and filter metadata for the current variable
                df_curr = df_meta[['SampleID', var]].copy()
                df_curr = df_curr[df_curr[var].notna() & (df_curr[var].astype(str).str.strip() != '')]
                
                # Get valid IDs and filter distance matrix
                valid_ids = df_curr['SampleID'].tolist()
                if len(valid_ids) < 3:
                    progress_bar.progress((i + 1) / total_vars)
                    continue # Not enough samples
                
                curr_dist = full_distance_matrix.filter(valid_ids)
                df_curr = df_curr.set_index('SampleID')
                
                if df_curr[var].nunique() < 2:
                    progress_bar.progress((i + 1) / total_vars)
                    continue # Need at least 2 groups for any meaningful comparison
                
                # Determine categorical vs continuous
                is_numeric = False
                try:
                    df_curr_numeric = pd.to_numeric(df_curr[var], errors='coerce')
                    if df_curr_numeric.notna().sum() > len(df_curr) * 0.5: # Mostly numeric
                        is_numeric = True
                        df_curr[var] = df_curr_numeric
                except Exception:
                    pass
                
                random.seed(random_seed)
                np.random.seed(random_seed)
                
                # Our rule: if not numeric OR nunique <= 10 => Categorical (ANOSIM)
                if not is_numeric or df_curr[var].nunique() <= 10:
                    test_type = 'Categorical (ANOSIM)'
                    try:
                        group_series = df_curr[var].astype(str)
                        res = anosim(curr_dist, group_series, permutations=perm_count)
                        stat_val = res['test statistic']
                        p_val = res['p-value']
                    except Exception as e:
                        progress_bar.progress((i + 1) / total_vars)
                        continue
                else:
                    test_type = 'Continuous (Mantel)'
                    try:
                        df_curr = df_curr.dropna(subset=[var])
                        if len(df_curr) < 3: 
                            progress_bar.progress((i + 1) / total_vars)
                            continue
                        curr_dist = full_distance_matrix.filter(df_curr.index.tolist())
                        
                        meta_dist_arr = squareform(pdist(df_curr[[var]].values, metric='euclidean'))
                        meta_dist_mat = DistanceMatrix(meta_dist_arr, ids=df_curr.index)
                        stat_val, p_val, _ = mantel(curr_dist, meta_dist_mat, permutations=perm_count)
                    except Exception as e:
                        progress_bar.progress((i + 1) / total_vars)
                        continue
                
                results.append({
                    '變數名稱 (Variable)': var,
                    '變數型態 (Data Type)': '類別型 (Categorical)' if 'Categorical' in test_type else '連續型 (Continuous)',
                    '統計方法 (Method)': 'ANOSIM' if 'Categorical' in test_type else 'Mantel test',
                    '有效樣本數 (N)': len(valid_ids),
                    'Statistic (R)': stat_val,
                    'P-value': p_val
                })
                
                progress_bar.progress((i + 1) / total_vars)
            
            status_text.text('✅ 批次分析完成！')
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('P-value', ascending=True).reset_index(drop=True)
                st.dataframe(df_results)
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Statistical_Results')
                output.seek(0)
                
                st.download_button(
                    label="📥 下載 Excel 分析報告",
                    data=output,
                    file_name="Batch_PCoA_Statistics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("沒有成功完成任何變數的分析（可能是樣本數不足或分群異常）。")


if __name__ == '__main__':
    main()

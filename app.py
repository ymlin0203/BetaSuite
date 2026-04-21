"""
core_microbiota_heatmap.py  –  Universal core microbiota analysis
==================================================================

Run WITHOUT arguments → interactive GUI (folder picker pops up)
Run WITH  arguments  → command-line mode

Step 1 │ Full-dataset heatmap  (all samples)
Step 2 │ Per-group elbow plot  (fixed abundance = FIXED_ABUNDANCE %)
Step 3 │ Per-group species CSV  (all prevalence levels)

Usage
─────
# GUI mode (double-click or run without args):
python core_microbiota_heatmap.py

# Command-line – point to a folder:
python core_microbiota_heatmap.py --data-dir 26_0101_CMC_all_data/

# Command-line – specify files manually:
python core_microbiota_heatmap.py \\
    --tsv  species-table.tsv \\
    --meta sample-sheet.csv \\
    --meta-sample-col "Sample " \\
    --meta-group-col  "Group" \\
    --out  output/
"""

import argparse
import glob
import itertools
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, mannwhitneyu
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("\n[錯誤] 缺少必需的統計套件。")
    print("請先安裝 scipy 與 statsmodels。在終端機執行以下指令：")
    print("  pip install scipy statsmodels\n")
    sys.exit(1)
# ──────────────────────────────────────────────────────────────────────────────
# Cutoff settings  (edit here to change defaults)
# ──────────────────────────────────────────────────────────────────────────────

HEATMAP_ABUNDANCE_CUTOFFS  = [0, 0.01, 0.1, 1, 5, 10]   # %
HEATMAP_PREVALENCE_CUTOFFS = list(range(0, 101, 5))       # 0, 5 … 100 %

FIXED_ABUNDANCE          = 0.01                           # % (Steps 2 & 3)
ELBOW_PREVALENCE_CUTOFFS = list(range(0, 101, 5))         # 0, 5 … 100 %

# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# File helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_files(folder: str, ext: str) -> list[str]:
    """Find all files with *ext* in folder and one level of subfolders."""
    found = glob.glob(os.path.join(folder, f'*{ext}'))
    found += glob.glob(os.path.join(folder, '*', f'*{ext}'))
    return sorted(set(found))


def _auto_discover(data_dir: str) -> tuple[str, str]:
    """
    Auto-discover TSV and CSV inside data_dir.
    - Exactly 1 found → auto-select
    - Multiple found  → list them and exit with instructions
    """
    tsv_files = _find_files(data_dir, '.tsv')
    csv_files = _find_files(data_dir, '.csv')

    if len(tsv_files) == 0:
        raise FileNotFoundError(
            f'\n  No .tsv found in: {data_dir}\n'
            f'  → Put your taxon table there, or use --tsv.')
    elif len(tsv_files) == 1:
        tsv_path = tsv_files[0]
        print(f'  [auto] TSV  → {tsv_path}')
    else:
        print(f'\n  Multiple .tsv files in {data_dir}:')
        for i, f in enumerate(tsv_files, 1):
            print(f'    [{i}] {f}')
        print('\n  → Use --tsv <path> to specify which one.')
        raise SystemExit(1)

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f'\n  No .csv found in: {data_dir}\n'
            f'  → Put your sample sheet there, or use --meta.')
    elif len(csv_files) == 1:
        meta_path = csv_files[0]
        print(f'  [auto] Meta → {meta_path}')
    else:
        print(f'\n  Multiple .csv files in {data_dir}:')
        for i, f in enumerate(csv_files, 1):
            print(f'    [{i}] {f}')
        print('\n  → Use --meta <path> to specify which one.')
        raise SystemExit(1)

    return tsv_path, meta_path


# ══════════════════════════════════════════════════════════════════════════════
# GUI interactive mode
# ══════════════════════════════════════════════════════════════════════════════

def _pick_from_list(root, items: list[str], title: str, prompt: str) -> str:
    """Listbox dialog; returns the selected item path."""
    import tkinter as tk

    selected = tk.StringVar(value='')
    win = tk.Toplevel(root)
    win.title(title)
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=prompt, wraplength=520,
             justify='left', padx=12, pady=8).pack(anchor='w')

    frame = tk.Frame(win)
    frame.pack(fill='both', expand=True, padx=12)

    sb = tk.Scrollbar(frame, orient='vertical')
    lb = tk.Listbox(frame, yscrollcommand=sb.set, selectmode='single',
                    width=90, height=min(len(items), 12),
                    font=('Consolas', 9))
    sb.config(command=lb.yview)
    sb.pack(side='right', fill='y')
    lb.pack(side='left', fill='both', expand=True)

    base = os.path.commonpath(items) if len(items) > 1 else os.path.dirname(items[0])
    for item in items:
        lb.insert('end', os.path.relpath(item, base))
    lb.selection_set(0)

    def confirm():
        idx = lb.curselection()
        if idx:
            selected.set(items[idx[0]])
        win.destroy()

    tk.Button(win, text='確認選擇', command=confirm,
              width=14, pady=4).pack(pady=10)
    win.wait_window()
    return selected.get()


def _gui_pick() -> tuple[str, str, str]:
    """
    Full GUI flow.  Returns (tsv_path, meta_path, out_dir).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()

    # Step 1 – choose folder
    messagebox.showinfo(
        '核心菌相分析',
        '步驟 1／3\n\n請選擇實驗資料夾\n'
        '（內含 .tsv 菌相表  和  .csv 樣本分組表）')
    data_dir = filedialog.askdirectory(title='選擇實驗資料夾')
    if not data_dir:
        raise SystemExit('未選擇資料夾，程式結束。')

    tsv_files = _find_files(data_dir, '.tsv')
    csv_files = _find_files(data_dir, '.csv')

    # Step 2 – pick TSV
    if len(tsv_files) == 0:
        messagebox.showerror('找不到 TSV',
                             f'在以下路徑找不到任何 .tsv 檔案：\n{data_dir}')
        raise SystemExit(1)
    elif len(tsv_files) == 1:
        tsv_path = tsv_files[0]
        print(f'  [GUI] TSV 自動選取 → {tsv_path}')
    else:
        tsv_path = _pick_from_list(
            root, tsv_files,
            title='選擇菌相表',
            prompt='步驟 2／3  ─  找到多個 .tsv，請選擇要分析的菌相表：')
        if not tsv_path:
            raise SystemExit('未選擇 TSV，程式結束。')

    # Step 3 – pick CSV
    if len(csv_files) == 0:
        messagebox.showerror('找不到 CSV',
                             f'在以下路徑找不到任何 .csv 檔案：\n{data_dir}')
        raise SystemExit(1)
    elif len(csv_files) == 1:
        meta_path = csv_files[0]
        print(f'  [GUI] Meta 自動選取 → {meta_path}')
    else:
        meta_path = _pick_from_list(
            root, csv_files,
            title='選擇樣本分組表',
            prompt='步驟 3／3  ─  找到多個 .csv，請選擇樣本分組表（sample sheet）：')
        if not meta_path:
            raise SystemExit('未選擇樣本分組表，程式結束。')

    out_dir = os.path.join(data_dir, 'core_microbiota_output')
    root.destroy()

    print(f'  [GUI] 資料夾 : {data_dir}')
    print(f'  [GUI] TSV    : {tsv_path}')
    print(f'  [GUI] Meta   : {meta_path}')
    print(f'  [GUI] 輸出至 : {out_dir}')
    return tsv_path, meta_path, out_dir


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', index_col=0)


def load_meta(path: str, sample_col: str, group_col: str
              ) -> tuple[pd.DataFrame, str, str]:
    sep  = '\t' if path.endswith('.tsv') else ','
    meta = pd.read_csv(path, sep=sep)
    meta.columns = meta.columns.str.strip()
    sc, gc = sample_col.strip(), group_col.strip()
    stripped = {c.strip(): c for c in meta.columns}
    if sc not in meta.columns and sc in stripped:
        meta = meta.rename(columns={stripped[sc]: sc})
    if gc not in meta.columns and gc in stripped:
        meta = meta.rename(columns={stripped[gc]: gc})
    meta[sc] = meta[sc].astype(str).str.strip()
    meta[gc] = meta[gc].astype(str).str.strip()
    return meta[[sc, gc]].dropna(), sc, gc


def normalize_pct(df: pd.DataFrame) -> pd.DataFrame:
    col_sum = df.sum(axis=0).replace(0, np.nan)
    return df.div(col_sum, axis=1).multiply(100).fillna(0)


def count_species(df_pct: pd.DataFrame,
                  abundance_cutoff: float,
                  prevalence_cutoff: float) -> int:
    present    = df_pct > abundance_cutoff
    prevalence = present.sum(axis=1) / present.shape[1] * 100
    return int((prevalence > prevalence_cutoff).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Pairing summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_pairing_summary(meta: pd.DataFrame, sample_col: str,
                            group_col: str, tsv_cols: pd.Index) -> None:
    tsv_set = set(tsv_cols)
    print()
    print('  ┌──────────────────────────────────────────────┐')
    print('  │         Sample–TSV pairing summary           │')
    print('  ├──────────────┬──────────┬────────────────────┤')
    print('  │ Group        │ In meta  │ Matched in TSV     │')
    print('  ├──────────────┼──────────┼────────────────────┤')
    for group in sorted(meta[group_col].unique()):
        rows    = meta[meta[group_col] == group]
        n_meta  = len(rows)
        matched = rows[sample_col].isin(tsv_set).sum()
        print(f'  │ {group:<12s} │ {n_meta:>8d} │ {matched:>18d} │')
    total   = len(meta)
    matched = meta[sample_col].isin(tsv_set).sum()
    print('  ├──────────────┼──────────┼────────────────────┤')
    print(f'  │ {"TOTAL":<12s} │ {total:>8d} │ {matched:>18d} │')
    print('  └──────────────┴──────────┴────────────────────┘')
    unmatched = set(meta[sample_col]) - tsv_set
    if unmatched:
        print(f'\n  [warn] {len(unmatched)} samples in metadata NOT found in TSV:')
        for s in sorted(unmatched)[:8]:
            print(f'         • {s}')
        if len(unmatched) > 8:
            print(f'         … and {len(unmatched)-8} more')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Full heatmap
# ══════════════════════════════════════════════════════════════════════════════

def run_full_heatmap(df_pct: pd.DataFrame, out_dir: str) -> None:
    print('[Step 1] Building full-dataset heatmap …')
    n_taxa, n_samples = df_pct.shape

    row_labels = [f'> {a}%' for a in HEATMAP_ABUNDANCE_CUTOFFS]
    col_labels  = [f'> {p}%' for p in HEATMAP_PREVALENCE_CUTOFFS]
    mat = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)

    for a in HEATMAP_ABUNDANCE_CUTOFFS:
        for p in HEATMAP_PREVALENCE_CUTOFFS:
            mat.at[f'> {a}%', f'> {p}%'] = count_species(df_pct, a, p)

    csv_path = os.path.join(out_dir, 'step1_all_heatmap.csv')
    png_path = os.path.join(out_dir, 'step1_all_heatmap.png')
    mat.to_csv(csv_path, encoding='utf-8-sig')
    _plot_heatmap(mat,
                  title=f'All samples (n={n_samples}) – '
                        f'species count across cutoffs',
                  png_path=png_path)
    print(f'  Taxa: {n_taxa}  |  Samples: {n_samples}')
    print(f'  PNG → {png_path}')


def _plot_heatmap(mat: pd.DataFrame, title: str, png_path: str) -> None:
    data  = mat.astype(float).values
    ncols, nrows = len(mat.columns), len(mat.index)

    fig, ax = plt.subplots(figsize=(max(12, ncols * 0.6), max(3, nrows * 0.9)))

    # YlOrRd: low values = light yellow, high values = dark red (matches reference)
    im   = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Taxon count', fontsize=10)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_xlabel('Prevalence cutoff (%)', fontsize=10)
    ax.set_ylabel('Abundance cutoff (per-sample %)', fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    # Text colour: white on dark cells, black on light cells
    for i in range(nrows):
        for j in range(ncols):
            v     = int(data[i, j])
            norm_v = im.norm(data[i, j])
            # YlOrRd: high norm = dark red → white text; low norm = light → black text
            color = 'white' if norm_v > 0.65 else 'black'
            ax.text(j, i, str(v), ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Step 1b – Full-dataset elbow plot  (all samples, one line per abundance)
# ══════════════════════════════════════════════════════════════════════════════

# Colours for each abundance level line (matches reference image palette)
_ABUND_LINE_COLORS = {
    0:    '#1f77b4',   # blue
    0.01: '#ff7f0e',   # orange
    0.1:  '#2ca02c',   # teal-green
    1:    '#d62728',   # red
    5:    '#17becf',   # light blue / cyan
    10:   '#e377c2',   # pink
}
_ABUND_SHADE_COLORS = [
    '#aec7e8', '#c7e9c0', '#f5c6aa', '#fdd0a2', '#fde0ef'
]

ANNOTATE_PREV = 5   # prevalence % at which to place the highlight dot + label


def run_full_elbow(df_pct: pd.DataFrame, out_dir: str) -> None:
    """
    Overview elbow plot for ALL samples combined.
    X = prevalence cutoff (%)
    Y = taxon count
    One line per abundance cutoff (same style as reference image).
    Shaded bands between adjacent lines + annotation dot at ANNOTATE_PREV.
    """
    print(f'\n[Step 1b] Full-dataset overview elbow plot …')

    n_samples = df_pct.shape[1]

    # Build count matrix: abundance → list of counts per prevalence cutoff
    line_data = {}
    for a in HEATMAP_ABUNDANCE_CUTOFFS:
        line_data[a] = [count_species(df_pct, a, p)
                        for p in ELBOW_PREVALENCE_CUTOFFS]

    # Find annotation index
    try:
        annot_idx = ELBOW_PREVALENCE_CUTOFFS.index(ANNOTATE_PREV)
    except ValueError:
        annot_idx = 1

    fig, ax = plt.subplots(figsize=(14, 7))

    levels = HEATMAP_ABUNDANCE_CUTOFFS

    # ── Shaded bands between adjacent lines (bottom-most first) ──────────────
    for i in range(len(levels) - 1):
        a_top = levels[i]
        a_bot = levels[i + 1]
        color = _ABUND_SHADE_COLORS[i % len(_ABUND_SHADE_COLORS)]
        ax.fill_between(ELBOW_PREVALENCE_CUTOFFS,
                        line_data[a_top], line_data[a_bot],
                        alpha=0.18, color=color)

    # ── Lines + annotation dots ───────────────────────────────────────────────
    for a in levels:
        counts = line_data[a]
        color  = _ABUND_LINE_COLORS.get(a, 'gray')
        label  = f'abund > {a}%'

        ax.plot(ELBOW_PREVALENCE_CUTOFFS, counts,
                marker='o', markersize=4, linewidth=1.8,
                color=color, label=label)

        # Highlight dot + label at ANNOTATE_PREV
        xv = ELBOW_PREVALENCE_CUTOFFS[annot_idx]
        yv = counts[annot_idx]
        ax.plot(xv, yv, 'o', color=color, markersize=9, zorder=5)
        ax.annotate(
            f'prev={ANNOTATE_PREV}% n={yv}',
            xy=(xv, yv),
            xytext=(xv + 1.5, yv + max(line_data[levels[0]]) * 0.02),
            fontsize=8, color=color, fontweight='bold',
        )

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xticks(ELBOW_PREVALENCE_CUTOFFS)
    ax.set_xticklabels([f'> {p}%' for p in ELBOW_PREVALENCE_CUTOFFS],
                       rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Prevalence cutoff (%)', fontsize=12)
    ax.set_ylabel('Taxon count', fontsize=12)
    ax.set_title(
        f'Elbow plot – Taxon count vs prevalence cutoff\n'
        f'All samples (n={n_samples})',
        fontsize=13)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlim(-1, max(ELBOW_PREVALENCE_CUTOFFS) + 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    png_path = os.path.join(out_dir, 'step1b_full_elbow.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  PNG → {png_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Per-group elbow plot
# ══════════════════════════════════════════════════════════════════════════════

def run_group_elbow(df_pct: pd.DataFrame, meta: pd.DataFrame,
                    sample_col: str, group_col: str, out_dir: str) -> None:
    print(f'\n[Step 2] Per-group elbow plot (abundance > {FIXED_ABUNDANCE}%) …')

    groups = sorted(meta[group_col].unique())
    colors = cm.tab10(np.linspace(0, 1, max(len(groups), 1)))

    fig, ax = plt.subplots(figsize=(12, 6))
    records = []

    for group, color in zip(groups, colors):
        samples = [s for s in meta.loc[meta[group_col] == group, sample_col]
                   if s in df_pct.columns]
        if not samples:
            print(f'  [warn] No matched samples for: {group}')
            continue
        sub    = df_pct[samples]
        n      = len(samples)
        counts = [count_species(sub, FIXED_ABUNDANCE, p)
                  for p in ELBOW_PREVALENCE_CUTOFFS]
        ax.plot(ELBOW_PREVALENCE_CUTOFFS, counts,
                marker='o', markersize=4, linewidth=2,
                label=f'{group}  (n={n})', color=color)
        for p, c in zip(ELBOW_PREVALENCE_CUTOFFS, counts):
            records.append({'Group': group, 'n_samples': n,
                            'Prevalence_cutoff_%': p, 'Species_count': c})

    ax.set_xticks(ELBOW_PREVALENCE_CUTOFFS)
    ax.set_xticklabels([f'{p}%' for p in ELBOW_PREVALENCE_CUTOFFS],
                       rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Prevalence cutoff (%)', fontsize=12)
    ax.set_ylabel('Taxon count', fontsize=12)
    ax.set_title(f'Elbow plot – taxon count vs prevalence cutoff\n'
                 f'(abundance > {FIXED_ABUNDANCE}%)', fontsize=13)
    ax.legend(title='Group', bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, max(ELBOW_PREVALENCE_CUTOFFS) + 2)
    plt.tight_layout()

    png_path = os.path.join(out_dir, 'step2_group_elbow.png')
    csv_path = os.path.join(out_dir, 'step2_group_elbow.csv')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'  Groups: {groups}')
    print(f'  PNG → {png_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Per-group species CSV
# ══════════════════════════════════════════════════════════════════════════════

def run_group_species_csv(df_pct: pd.DataFrame, meta: pd.DataFrame,
                           sample_col: str, group_col: str,
                           out_dir: str) -> None:
    print(f'\n[Step 3] Per-group species lists '
          f'(abundance > {FIXED_ABUNDANCE}%, all prevalence levels) …')

    core_dir = os.path.join(out_dir, 'group_species_lists')
    os.makedirs(core_dir, exist_ok=True)

    for group in sorted(meta[group_col].unique()):
        samples = [s for s in meta.loc[meta[group_col] == group, sample_col]
                   if s in df_pct.columns]
        if not samples:
            continue
        sub = df_pct[samples]
        n   = len(samples)
        present    = sub > FIXED_ABUNDANCE
        prevalence = present.sum(axis=1) / n * 100
        detected   = prevalence > 0

        result = pd.DataFrame({
            'Taxon':               sub.index[detected],
            'n_samples_present':   present.loc[detected].sum(axis=1).values,
            'n_samples_total':     n,
            'Prevalence_%':        prevalence[detected].round(2).values,
            'Mean_RelAbun_%':      sub.loc[detected].mean(axis=1).round(4).values,
            'Median_RelAbun_%':    sub.loc[detected].median(axis=1).round(4).values,
            'Max_RelAbun_%':       sub.loc[detected].max(axis=1).round(4).values,
        }).sort_values('Prevalence_%', ascending=False).reset_index(drop=True)

        safe  = group.replace('/', '-').replace(' ', '_')
        path  = os.path.join(core_dir,
                             f'{safe}_abun{FIXED_ABUNDANCE}_all_prev.csv')
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'  [{group:<12s}]  n={n:3d}  →  {len(result):3d} species  {path}')

    print(f'  Lists saved → {core_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Pairwise Statistical Comparisons
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_comparisons(df_pct: pd.DataFrame, meta: pd.DataFrame,
                                sample_col: str, group_col: str, out_dir: str) -> None:
    print(f'\n[Step 4] Pairwise statistical comparisons (Prevalence & Abundance) …')

    stats_dir = os.path.join(out_dir, 'statistical_comparisons')
    os.makedirs(stats_dir, exist_ok=True)

    groups = sorted(meta[group_col].unique())
    if len(groups) < 2:
        print("  [warn] Not enough groups to perform statistical comparisons.")
        return

    # Generate all pairwise combinations
    pairs = list(itertools.combinations(groups, 2))

    for g1, g2 in pairs:
        print(f'  Comparing [{g1}] vs [{g2}] ...')

        # Get samples for each group
        samples1 = [s for s in meta.loc[meta[group_col] == g1, sample_col] if s in df_pct.columns]
        samples2 = [s for s in meta.loc[meta[group_col] == g2, sample_col] if s in df_pct.columns]

        n1, n2 = len(samples1), len(samples2)
        if n1 == 0 or n2 == 0:
            print(f'    [warn] Missing samples for one of the groups. Skipping.')
            continue

        sub1 = df_pct[samples1]
        sub2 = df_pct[samples2]

        # Calculate presence status for Fisher's exact test (using FIXED_ABUNDANCE, e.g. 0.01%)
        present1_df = sub1 > FIXED_ABUNDANCE
        present2_df = sub2 > FIXED_ABUNDANCE

        records = []
        for taxon in df_pct.index:
            # Species profile
            abun1 = sub1.loc[taxon].values
            abun2 = sub2.loc[taxon].values

            # Presence based on the predefined threshold
            is_present1 = present1_df.loc[taxon].values
            is_present2 = present2_df.loc[taxon].values

            p1_count = is_present1.sum()
            p2_count = is_present2.sum()

            # Skip the test for this taxon if it's completely absent in BOTH groups
            if p1_count == 0 and p2_count == 0:
                continue

            prev1 = (p1_count / n1) * 100
            prev2 = (p2_count / n2) * 100

            mean1, mean2 = np.mean(abun1), np.mean(abun2)
            med1, med2 = np.median(abun1), np.median(abun2)

            # 1. Prevalence test (Fisher's exact test)
            A, B = p1_count, p2_count
            C, D = n1 - A, n2 - B
            table = [[A, B], [C, D]]
            _, fish_p = fisher_exact(table)

            # 2. Abundance test (Mann-Whitney U test)
            try:
                # If both array distributions are perfectly identical (e.g. all 0, all exactly same number)
                if np.array_equal(abun1, abun2):
                    mwu_p = 1.0
                else:
                    _, mwu_p = mannwhitneyu(abun1, abun2, alternative='two-sided')
            except ValueError:
                mwu_p = 1.0

            # Log2FoldChange calculation (Adding a pseudo-count of 1e-5 to avoid log(0))
            log2fc = np.log2((mean1 + 1e-5) / (mean2 + 1e-5))

            records.append({
                'Taxon': taxon,
                f'Prevalence_{g1}_%': round(prev1, 2),
                f'Prevalence_{g2}_%': round(prev2, 2),
                'Prevalence_Diff_%': round(prev1 - prev2, 2),
                f'Mean_Abun_{g1}_%': round(mean1, 4),
                f'Mean_Abun_{g2}_%': round(mean2, 4),
                'Log2FC_Mean_Abun': round(log2fc, 4),
                f'Median_Abun_{g1}_%': round(med1, 4),
                f'Median_Abun_{g2}_%': round(med2, 4),
                'Fisher_pvalue': fish_p,
                'MWU_pvalue': mwu_p
            })

        if not records:
            continue

        res_df = pd.DataFrame(records)

        # FDR correction based on Benjamini-Hochberg for the extracted p-values
        fish_pvals = res_df['Fisher_pvalue'].fillna(1.0).values
        mwu_pvals = res_df['MWU_pvalue'].fillna(1.0).values

        try:
            _, fish_fdr, _, _ = multipletests(fish_pvals, method='fdr_bh')
            _, mwu_fdr, _, _ = multipletests(mwu_pvals, method='fdr_bh')
        except Exception:
            fish_fdr, mwu_fdr = fish_pvals, mwu_pvals

        res_df['Fisher_FDR_qvalue'] = fish_fdr
        res_df['MWU_FDR_qvalue'] = mwu_fdr

        # Sort the rows by Abundance MWU p-value descending by default
        res_df = res_df.sort_values(['MWU_pvalue', 'Fisher_pvalue']).reset_index(drop=True)

        safe1 = g1.replace('/', '-').replace(' ', '_')
        safe2 = g2.replace('/', '-').replace(' ', '_')
        path = os.path.join(stats_dir, f'{safe1}_vs_{safe2}_stats.csv')
        res_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'    → Saved {len(res_df)} taxon comparisons to {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Core microbiota analysis  '
                    '(run without arguments for interactive GUI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--data-dir',
        help='Experiment folder – auto-discovers .tsv and .csv inside')
    parser.add_argument('--tsv',
        help='Taxon table (TSV, rows=taxa, columns=samples)')
    parser.add_argument('--meta',
        help='Sample metadata (CSV or TSV)')
    parser.add_argument('--meta-sample-col', default='Sample',
        help='Sample-ID column in metadata  (default: "Sample")')
    parser.add_argument('--meta-group-col', default='Group',
        help='Group column in metadata  (default: "Group")')
    parser.add_argument('--out', default=None,
        help='Output directory  (default: <data-dir>/core_microbiota_output/)')

    args = parser.parse_args()
    no_args = not args.data_dir and not args.tsv and not args.meta

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if no_args:
        print('No arguments → opening GUI file picker …')
        tsv_path, meta_path, out_dir = _gui_pick()
        if args.out:
            out_dir = args.out

    elif args.data_dir:
        tsv_path, meta_path = _auto_discover(args.data_dir)
        out_dir = args.out or os.path.join(
            args.data_dir, 'core_microbiota_output')

    elif args.tsv and args.meta:
        tsv_path  = args.tsv
        meta_path = args.meta
        out_dir   = args.out or 'core_microbiota_output'

    else:
        parser.error(
            'Run without arguments for GUI mode, or provide:\n'
            '  --data-dir <folder>\n'
            '  --tsv <file> --meta <file>')

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print('=' * 62)
    print('Loading data …')
    df_raw = load_tsv(tsv_path)
    meta, sample_col, group_col = load_meta(
        meta_path, args.meta_sample_col, args.meta_group_col)

    print(f'  Taxa          : {len(df_raw)}')
    print(f'  Samples (TSV) : {len(df_raw.columns)}')
    print(f'  Samples (meta): {len(meta)}')
    print(f'  Groups        : {sorted(meta[group_col].unique())}')
    _print_pairing_summary(meta, sample_col, group_col, df_raw.columns)

    df_pct = normalize_pct(df_raw)

    # ── Run ───────────────────────────────────────────────────────────────────
    print('=' * 62)
    run_full_heatmap(df_pct, out_dir)
    print('=' * 62)
    run_full_elbow(df_pct, out_dir)
    print('=' * 62)
    run_group_elbow(df_pct, meta, sample_col, group_col, out_dir)
    print('=' * 62)
    run_group_species_csv(df_pct, meta, sample_col, group_col, out_dir)
    print('=' * 62)
    run_statistical_comparisons(df_pct, meta, sample_col, group_col, out_dir)
    print('=' * 62)
    print('Done!  Outputs saved to:', os.path.abspath(out_dir))
    print('=' * 62)


if __name__ == '__main__':
    main()

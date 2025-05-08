# BetaSuite

**BetaSuite** is an interactive Streamlit-based web application for advanced beta diversity analysis.  
It supports fully automated PCoA computation from distance matrices, flexible visualization in both 2D and 3D formats, and built-in statistical testing (ANOSIM / Mantel test) based on variable type.

## 🔍 Features

- ✅ Upload your own distance matrix (`.tsv` or `.csv`) and metadata (`.xlsx` or `.csv`)
- ✅ Automatic PCoA computation using scikit-bio
- ✅ Metadata-based coloring with automatic detection of variable type:
  - Categorical → ANOSIM
  - Continuous → Mantel test
- ✅ Support for 2D & 3D interactive PCoA plots
- ✅ Axis flipping for PC1/PC2 to refine visual interpretation
- ✅ High-resolution export options:
  - 2D PNG (1200 dpi via matplotlib)
  - 3D PNG / PDF / SVG (via plotly)
- ✅ Custom color palette selection
- ✅ Reproducible statistical results with permutation seed control

## 🧪 Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

```bash
streamlit run app.py
```

Then open the app in your browser as prompted.

## 📁 Input Files

- **Distance Matrix**: `.tsv` or `.csv` format, symmetric matrix with sample IDs as both rows and columns
- **Metadata**: `.xlsx` or `.csv` file, first column must be `SampleID`, additional columns are treated as variables for coloring/statistics

## 📊 Statistical Tests

| Variable Type | Test Used | Description |
|---------------|-----------|-------------|
| Categorical   | ANOSIM    | Non-parametric test for group separation based on ranks |
| Continuous    | Mantel    | Correlation between distance matrices (e.g., PCoA vs metadata) |

## 📈 Output

- 🖼️ Static high-quality 2D plot (downloadable as PNG, 1200 dpi)
- 🌐 Interactive 3D PCoA plot (with export to PNG / PDF / SVG)
- 📎 Summary of test statistic and p-value
- 🔁 Permutation count and random seed are adjustable

## 📚 Citation

If you use **BetaSuite** in your research, please cite:

> Lin Y-M, et al. BetaSuite: A visual and statistical interface for beta diversity analysis using PCoA and metadata-driven testing. 2025. *(In prep.)*

---

Created with ❤️ by Yu-Min Lin and contributors.

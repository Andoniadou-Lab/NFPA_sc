# ==================================================================================================
# NFPA_sc: all code for the figures and supplementary tables in
#   Kaufman-Cook, Vamvoukaki et al. (2026). A senescence-associated programme marks aggressive
#   non-functioning pituitary neuroendocrine tumours.
#
# One script, run top to bottom (e.g. paste into a single Google Colab cell, or `python NFPA_all_figures.py`).
# Fig. 1A is made separately in R (Fig1A_sample_overview.R).
#
#   Section 1   Setup: packages, paths, data, shared colours and helpers
#   Section 2   Fig. 1B, 1C
#   Section 3   Supplementary Fig. 1A, 1B
#   Section 4   Supplementary Fig. 2A-C (composition model, inferCNV, assembly)
#   Section 5   Supplementary Table 1
#   Section 6   Fig. 2A-E
#   Section 7   Supplementary Fig. 3A-F
#   Section 8   Supplementary Fig. 4A-C
#   Section 9   Tissue cohort: Figs. 3A, 3C, 4A, 4C, 4D; Supplementary Figs. 6, 7
#   Section 10  Software versions
#
# Inputs : the integrated atlas (.h5ad, Zenodo https://doi.org/10.5281/zenodo.23936955)
#          Supplementary_Table_3.xlsx (tissue cohort)
# Outputs: PDF / PNG (600 dpi) for every panel, CSV tables, Supplementary_Table_1.xlsx, all in OUT_DIR
# ==================================================================================================

# ---- paths: edit these -----------------------------------------------------------------------------
DATA_PATH   = "/content/drive/MyDrive/Colab Notebooks/merged_object_2026_10_27.h5ad"   # integrated AnnData object
OUT_DIR     = "/content/drive/MyDrive/NFPA_figures_healthy"                            # all outputs written here
SUPP_TABLE3 = "/content/drive/MyDrive/NFPA_figures_healthy/Supplementary_Table_3.xlsx"  # tissue cohort table

# ---- packages (installed only if missing) -------------------------------------------------------------
import importlib.util, subprocess, sys as _sys
_need = [p for m, p in [("scanpy", "scanpy"), ("gseapy", "gseapy"), ("statsmodels", "statsmodels"),
                        ("pydeseq2", "pydeseq2"), ("infercnvpy", "infercnvpy"), ("openpyxl", "openpyxl"),
                        ("lxml", "lxml"), ("sklearn", "scikit-learn")] if importlib.util.find_spec(m) is None]
if _need:
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q"] + _need, check=True)

try:                                   # mount Google Drive when running in Colab
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass


####################################################################################################
# SECTION 1  Setup: data, condition, colours and helpers
####################################################################################################
# ---- Load data, condition, colours and helpers ----------------------------------------------
import os, itertools, math
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata, wilcoxon

adata = sc.read(DATA_PATH)
OUT = OUT_DIR                                                # all figures and tables go here
os.makedirs(OUT, exist_ok=True); os.chdir(OUT)

o = adata.obs
o["condition"] = np.where(o["Cell type"].astype(str).str.lower() == "normal", "Healthy", "Tumour")
print(pd.crosstab(o["cell_type"], o["condition"], margins=True))

# ---- colours ------------------------------------------------------------------
cats = list(o["cell_type"].cat.categories)
CT_PAL = dict(zip(cats, adata.uns["cell_type_colors"])) if "cell_type_colors" in adata.uns else {}
for i, c in enumerate(c for c in cats if c not in CT_PAL):
    CT_PAL[c] = plt.get_cmap("tab20")(i % 20)
CT_PAL.update({"Stem_cells": "#8c564b", "Gonadotrophs": "#7f7f7f"})   # same as the paper
COND_PAL = {"Healthy": "#4393C3", "Tumour": "#8B1A1A"}
nice = lambda s: str(s).replace("_", " ")
MM = 1 / 25.4

def pub_style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.labelsize": 7,
                         "mathtext.fontset": "custom", "mathtext.it": "Liberation Sans:italic"})
pub_style()

def save(fig, name):
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        fig.savefig(name + ext, bbox_inches="tight", pad_inches=0.02, **kw)
    print("saved", name)

def mwu_exact(a, b, n_mc=200000, seed=0):
    """Two-sided Mann-Whitney P: exact permutation with ties (Monte Carlo when large) - as in Prism."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = np.concatenate([a, b]); r = rankdata(x); n1, N = len(a), len(x)
    mu = n1 * (N + 1) / 2; d = abs(r[:n1].sum() - mu) - 1e-9
    if math.comb(N, n1) <= 2_000_000:
        sums = np.array([r[list(i)].sum() for i in itertools.combinations(range(N), n1)])
        return float(np.mean(np.abs(sums - mu) >= d))
    rng = np.random.default_rng(seed)
    s = np.array([r[rng.permutation(N)[:n1]].sum() for _ in range(n_mc)])
    return float((np.sum(np.abs(s - mu) >= d) + 1) / (n_mc + 1))

def fmt_p(p):
    return "n.a." if p is None or np.isnan(p) else "<0.0001" if p < 0.0001 else \
           f"{p:.4f}".rstrip("0") if p < 0.001 else f"{p:.2g}"

# ---- UMAP coordinates and limits shared by all UMAP figures --------------------------------------
full = adata.obsm["X_umap"]
idx = np.random.default_rng(0).permutation(adata.n_obs)
x0, x1 = full[:, 0].min() - 0.3, full[:, 0].max() + 0.3
y0, y1 = full[:, 1].min() - 0.3, full[:, 1].max() + 0.3


####################################################################################################
# SECTION 2a  Fig. 1B: UMAPs by major cell class, condition and study
####################################################################################################
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Setup (style, colours, UMAP limits) -------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6})
MM = 1 / 25.4
o = adata.obs
o["condition"] = np.where(o["Cell type"].astype(str).str.lower() == "normal", "Healthy", "Tumour")
cats = list(o["cell_type"].cat.categories)
CT_PAL = dict(zip(cats, adata.uns["cell_type_colors"])) if "cell_type_colors" in adata.uns else {}
COND_PAL = {"Healthy": "#4393C3", "Tumour": "#8B1A1A"}
full = adata.obsm["X_umap"]
idx = np.random.default_rng(0).permutation(adata.n_obs)          # random draw order
x0, x1 = full[:, 0].min() - 0.3, full[:, 0].max() + 0.3
y0, y1 = full[:, 1].min() - 0.3, full[:, 1].max() + 0.3

def save(fig, name):
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        fig.savefig(name + ext, bbox_inches="tight", pad_inches=0.02, **kw)
    print("saved", name)

# ---- Fig 1A: integrated atlas UMAPs coloured by cell type, condition and study --------------------
# Immune subtypes are merged for this panel only (detailed labels: Supplementary Figure 1);
# the legend is grouped by compartment.
groups_ct = {
    "Stem / endocrine": {"Stem cells": ["Stem_cells"], "Gonadotrophs / tumour cells": ["Gonadotrophs"],
                         "Somatotrophs": ["Somatotrophs"], "Lactotrophs": ["Lactotrophs"],
                         "Corticotrophs": ["Corticotrophs"], "Thyrotrophs": ["Thyrotrophs"]},
    "Stromal":          {"Pituicytes": ["Pituicytes"], "Endothelial cells": ["Endothelial_cells"],
                         "Mesenchymal cells": ["Mesenchymal_cells"]},
    "Myeloid":          {"Macrophages": ["Macrophages"], "Monocytes": ["Monocytes"],
                         "Other myeloid": ["Dendritic_cells", "pDC", "Mast_cells", "Neutrophils"]},
    "Lymphoid":         {"T/NK cells": ["CD4_T_cells", "CD4_T_regs", "CD8_T_cells", "NK_cells", "ILCs"],
                         "B/plasma cells": ["B_cells", "Plasma_cells"]},
}
to_panel = {raw: lab for grp in groups_ct.values() for lab, raws in grp.items() for raw in raws}
o["cell_type_panel"] = o["cell_type"].astype(str).map(to_panel).fillna(o["cell_type"].astype(str))
PANEL_PAL = {"Stem cells": "#8c564b", "Gonadotrophs / tumour cells": "#7f7f7f",
             "Somatotrophs": CT_PAL.get("Somatotrophs", "#17becf"), "Lactotrophs": CT_PAL.get("Lactotrophs", "#f7b6d2"),
             "Corticotrophs": CT_PAL.get("Corticotrophs", "#c49c94"), "Thyrotrophs": CT_PAL.get("Thyrotrophs", "#9edae5"),
             "Pituicytes": "#f0b27a", "Endothelial cells": "#1f77b4", "Mesenchymal cells": "#2ca02c",
             "Macrophages": "#ff7f0e", "Monocytes": "#d62728", "Other myeloid": "#bcbd22",
             "T/NK cells": "#e377c2", "B/plasma cells": "#393b79"}
present = set(o["cell_type_panel"])
studies = sorted(o["Author"].astype(str).unique())
study_pal = {s: plt.get_cmap("tab10")(i % 10) for i, s in enumerate(studies)}

fig_w, gap = 180 * MM, 0.12
side = (fig_w - 2 * gap) / 3                     # square UMAP panels, as large as the page allows
leg_h, title_h = 1.45, 0.16
fig_h = title_h + side + leg_h
fig = plt.figure(figsize=(fig_w, fig_h))
blank = Line2D([], [], ls="", marker="", label=" ")
def header(t): return Line2D([], [], ls="", marker="", label=t)
def dot(lab, col): return Line2D([], [], marker="o", ls="", markersize=3.3, markerfacecolor=col,
                                 markeredgecolor="none", label=lab)

def umap_panel(i, col, pal, title):
    left, bottom = i * (side + gap) / fig_w, leg_h / fig_h
    ax = fig.add_axes([left, bottom, side / fig_w, side / fig_h])
    labs = o[col].astype(str).to_numpy()[idx]
    ax.scatter(full[idx, 0], full[idx, 1], c=[pal.get(l, "#cccccc") for l in labs], s=0.3,
               linewidths=0, rasterized=True)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.5); s.set_color("#888888")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    ax.set_xlabel("UMAP1", fontsize=6, labelpad=1)
    if i == 0: ax.set_ylabel("UMAP2", fontsize=6, labelpad=1)
    return left

def add_legend(left, handles, ncol):
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(left + side / fig_w / 2, (leg_h - 0.08) / fig_h),
                     ncol=ncol, frameon=False, fontsize=5.2, handletextpad=0.15, columnspacing=0.8,
                     labelspacing=0.22, handlelength=1.0)
    for t in leg.get_texts():                                       # group headers in bold
        if t.get_text() in groups_ct: t.set_fontweight("bold")

# Cell type: two legend columns (stem/endocrine + stromal | myeloid + lymphoid), headers in bold
cols = []
for grp_names in [["Stem / endocrine", "Stromal"], ["Myeloid", "Lymphoid"]]:
    col_h = []
    for gname in grp_names:
        labs = [l for l in groups_ct[gname] if l in present]
        if labs: col_h += [header(gname)] + [dot(l, PANEL_PAL[l]) for l in labs]
    cols.append(col_h)
n = max(len(c) for c in cols)
cols = [c + [blank] * (n - len(c)) for c in cols]
add_legend(umap_panel(0, "cell_type_panel", PANEL_PAL, "Cell type"), cols[0] + cols[1], 2)
add_legend(umap_panel(1, "condition", COND_PAL, "Condition"), [dot(k, COND_PAL[k]) for k in ["Healthy", "Tumour"]], 1)
add_legend(umap_panel(2, "Author", study_pal, "Study"), [dot(s, study_pal[s]) for s in studies], 1)
save(fig, "Fig1B_atlas_umaps")
plt.show()
print(pd.crosstab(o["Author"], o["condition"]).assign(samples=o.groupby("Author", observed=True)["SRA_ID"].nunique()))
print("Legend note for the figure caption: gonadotroph-lineage cells from tumour samples are the tumour cells.")


####################################################################################################
# SECTION 2b  Fig. 1C: cell-type composition per sample
####################################################################################################
# ---- Fig 1C: cell-type composition per sample, with mean healthy / tumour bars ----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

LABEL_MIN = 3            # label inside mean-bar segments at least this size (%)
MEAN_W = 2.3             # width of the mean bar (sample bars are 0.85)
GAP = 2.0                # space between the mean bar and the first sample (room for outside labels)
ALWAYS_LABEL = ["Stem_cells"]   # labelled on the mean bars however small (outside if < LABEL_MIN)
EXCLUDE = []             # e.g. ["SRX21896003", "SRX21896004", "SRX21896005", "SRX21896006", "SRX21896007"]

sid = o["SRA_ID"].astype(str)
comp = pd.crosstab(sid, o["cell_type"].astype(str), normalize="index") * 100
cats_present = [c for c in cats if c in comp.columns]
comp = comp[cats_present]
meta = (pd.DataFrame({"sample": sid.values, "condition": o["condition"].astype(str).values,
                      "study": o["Author"].astype(str).values})
        .groupby("sample").first())
order = {c: sorted([s for s in meta.index[meta["condition"] == c] if s not in EXCLUDE],
                   key=lambda s: (meta.loc[s, "study"], s))
         for c in ["Healthy", "Tumour"]}

# mean of per-sample proportions (each sample weighted equally)
means = pd.DataFrame({c: comp.loc[order[c]].mean() for c in ["Healthy", "Tumour"]})
summ = pd.concat({c: comp.loc[order[c]].agg(["mean", "median"]).T for c in ["Healthy", "Tumour"]},
                 axis=1)
summ.round(2).to_csv("Fig1C_composition_mean_median_healthy_tumour.csv")
comp.round(2).join(meta).to_csv("Fig1C_composition_per_sample_percent.csv")
print(summ.round(1).to_string())

label_name = lambda c: "Gonadotrophs / tumour cells" if c == "Gonadotrophs" else nice(c)
fmt = lambda v: "<0.1" if 0 < v < 0.05 else f"{v:.1f}"

def text_colour(col):
    r, g, b = to_rgb(col)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#222222"

def stack(ax, x, values, width, label=False):
    bottom = 0.0
    for c in cats_present:
        v = values[c]
        col = CT_PAL.get(c, "#cccccc")
        if v > 0:
            ax.bar(x, v, bottom=bottom, width=width, color=col, edgecolor="white", linewidth=0.2)
            if label and v >= LABEL_MIN:
                ax.text(x, bottom + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=4.4, color=text_colour(col), fontweight="bold")
            elif label and c in ALWAYS_LABEL:          # too small: label outside, in the gap
                yc = bottom + v / 2
                x0 = x + width / 2
                ax.plot([x0, x0 + 0.5], [yc, yc], color=col, lw=0.6, clip_on=False)
                ax.text(x0 + 0.55, yc, f"{v:.1f}%", ha="center", va="top", rotation=90,
                        fontsize=4.4, color=col, fontweight="bold", clip_on=False)
        bottom += v

fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM), sharey=True,
                         gridspec_kw={"width_ratios": [len(order["Healthy"]) + MEAN_W + GAP,
                                                       len(order["Tumour"]) + MEAN_W + GAP],
                                      "wspace": 0.04})
for ax, cnd in zip(axes, ["Healthy", "Tumour"]):
    d = comp.loc[order[cnd]]
    stack(ax, 0, means[cnd], MEAN_W, label=True)          # mean bar
    xs = np.arange(len(d)) + MEAN_W / 2 + GAP
    for xi, s in zip(xs, d.index):
        stack(ax, xi, d.loc[s], 0.85)
    ax.set_xticks([0, *xs])
    ax.set_xticklabels(["Mean", *d.index], rotation=90, fontsize=5)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.get_xticklabels()[0].set_fontsize(6.5)
    ax.set_xlim(-MEAN_W / 2 - 0.3, xs[-1] + 0.6)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    lo, hi = ax.get_xlim()
    ax.plot([lo + 0.2, hi - 0.2], [104, 104], color=COND_PAL[cnd], lw=2.5, clip_on=False)
    ax.text((lo + hi) / 2, 107, f"{cnd} (n = {len(d)})", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color=COND_PAL[cnd])
axes[1].spines["left"].set_visible(False)
axes[1].tick_params(axis="y", length=0)
axes[0].set_ylabel("Cells (%)")

handles = [Patch(fc=CT_PAL.get(c, "#cccccc"), ec="none",
                 label=f"{label_name(c)} ({fmt(means.loc[c, 'Healthy'])} · {fmt(means.loc[c, 'Tumour'])})")
           for c in cats_present]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), frameon=False,
           fontsize=5.5, handlelength=1.0, labelspacing=0.35,
           title="Mean % (healthy · tumour)", title_fontsize=6)
save(fig, "Fig1C_composition_mean")
plt.show()


####################################################################################################
# SECTION 3a  Supplementary Fig. 1A: canonical marker dot plot
####################################################################################################
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Setup (style, save) -----------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6})
MM = 1 / 25.4
o = adata.obs

def save(fig, name):
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        fig.savefig(name + ext, bbox_inches="tight", pad_inches=0.02, **kw)
    print("saved", name)

# ---- Supp. Fig. 1A: marker dot plot for every cell type ---------------------------------------------------
markers = {
    "Stem_cells": ["SOX2", "SOX9", "LHX3"], "Gonadotrophs": ["NR5A1", "CGA", "FSHB", "LHB"],
    "Somatotrophs": ["GH1", "POU1F1"], "Lactotrophs": ["PRL"], "Thyrotrophs": ["TSHB"],
    "Corticotrophs": ["POMC", "TBX19"], "Pituicytes": ["NKX2-1", "S100B"],
    "Endothelial_cells": ["CDH5", "VWF"], "Mesenchymal_cells": ["PDGFRB", "COL1A2"],
    "Macrophages": ["CD163", "C1QA"], "Monocytes": ["FCN1", "S100A8"], "Dendritic_cells": ["CLEC10A"],
    "pDC": ["LILRA4"], "Neutrophils": ["CSF3R"], "Mast_cells": ["CPA3"],
    "CD4_T_cells": ["CD3D", "CD4"], "CD4_T_regs": ["FOXP3"], "CD8_T_cells": ["CD8A"],
    "NK_cells": ["GNLY", "NKG7"], "ILCs": ["KIT"], "B_cells": ["MS4A1"], "Plasma_cells": ["JCHAIN"]}

order = [c for c in markers if c in set(o["cell_type"].astype(str))]
genes = list(dict.fromkeys(g for c in order for g in markers[c] if g in adata.var_names))   # flat list: no group brackets
missing = [g for c in order for g in markers[c] if g not in adata.var_names]
if missing:
    print("Not in object:", missing)

# row labels: underscores removed; tumour-sample gonadotrophs are the tumour cells
display = {c: ("Gonadotrophs / tumour cells" if c == "Gonadotrophs" else c.replace("_", " ")) for c in order}
adata.obs["cell_type_ordered"] = pd.Categorical(o["cell_type"].astype(str).map(display),
                                                categories=[display[c] for c in order])

dp = sc.pl.dotplot(adata, genes, groupby="cell_type_ordered", standard_scale="var", cmap="Reds",
                   show=False, return_fig=True, figsize=(180 * MM, 105 * MM))
dp.legend(colorbar_title="Mean expression\n(scaled per gene)", size_title="Cells expressing (%)")
dp.make_figure()
ax = dp.get_axes()["mainplot_ax"]
ax.tick_params(axis="y", labelsize=6.5)
ax.tick_params(axis="x", labelsize=6)
for t in ax.get_xticklabels():
    t.set_fontstyle("italic")
save(dp.fig, "SuppFig1A_marker_dotplot")
plt.show()


####################################################################################################
# SECTION 3b  Supplementary Fig. 1B: lineage markers in tumour cells
####################################################################################################
# ---- Supp. Fig. 1B: lineage markers in NF-PitNET tumour cells — standalone -----------
import logging
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)   # silence aspect warnings

CT_COL      = "cell_type"
TUMOUR_CT   = "Gonadotrophs"   # tumour cells = this cell type in tumour samples
MIN_CELLS   = 20               # min tumour cells for a sample to be shown
EXPR_LAYER  = None             # None = adata.X (log-normalised)
ZOOM_PCT    = 1                # trim this % of outlying cells on each side for the UMAP window
SMAX        = 90               # dot area at 100% expressing
HIGH_OTHER  = 20               # flag samples with > this % tumour cells expressing other-lineage markers
GENES = {"Gonadotroph": ["NR5A1", "CGA", "LHB", "FSHB"],
         "PIT1":        ["POU1F1", "GH1", "PRL", "TSHB"],
         "TPIT":        ["TBX19", "POMC"]}

# study colours — matched to Fig. 1 (edit hex codes if needed, or define STUDY_PAL beforehand)
SPAL = globals().get("STUDY_PAL", {
    "Asuzu et al., (2022)":    "#1f77b4",
    "Huang et al., (2025)":    "#ff7f0e",
    "Ilie et al. (2025)":      "#2ca02c",
    "Potthoff et al., (2024)": "#d62728",
    "Yan et al., (2024)":      "#9467bd",
    "Zhang et al., (2022c)":   "#8c564b",
    "Zhang et al. (2026)":     "#e377c2",
    "Zhang et al., (2022b)":   "#7f7f7f"})

MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6})

# reds, consistent with Supp. Fig. 1A
DOT_CMAP  = plt.get_cmap("Reds")
UMAP_CMAP = LinearSegmentedColormap.from_list(
    "grey_reds", ["#e6e6e6"] + [plt.get_cmap("Reds")(t) for t in np.linspace(0.25, 1, 6)])
EDGE = "#4d4d4d"

def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")

# ---- tumour cells ----------------------------------------------------------------------------
obs  = adata.obs
ct   = obs[CT_COL].astype(str).to_numpy()
cond = obs["condition"].astype(str).to_numpy()
sid  = obs["SRA_ID"].astype(str).to_numpy()
tmask = (ct == TUMOUR_CT) & (cond == "Tumour")

n_t = pd.Series(sid[tmask]).value_counts()
all_tum = pd.unique(sid[cond == "Tumour"])
dropped = sorted(set(all_tum) - set(n_t.index[n_t >= MIN_CELLS]))
print(f"tumour samples shown: {(n_t >= MIN_CELLS).sum()} of {len(all_tum)}; "
      f"excluded (< {MIN_CELLS} tumour cells): {dropped}")
tmask &= np.isin(sid, n_t.index[n_t >= MIN_CELLS])

genes = [g for grp in GENES.values() for g in grp if g in adata.var_names]
missing = [g for grp in GENES.values() for g in grp if g not in adata.var_names]
if missing: print("not in var_names (skipped):", missing)

X = adata.X if EXPR_LAYER is None else adata.layers[EXPR_LAYER]
idx = np.flatnonzero(tmask)
E = X[idx][:, [adata.var_names.get_loc(g) for g in genes]]
E = pd.DataFrame(E.toarray() if sp.issparse(E) else np.asarray(E), columns=genes)
s_t = sid[tmask]

# ---- per-sample summaries --------------------------------------------------------------------
pct  = (E > 0).groupby(s_t).mean() * 100
mean = E.groupby(s_t).mean()
author = obs.loc[tmask, "Author"].astype(str).groupby(s_t).first()
order = author.sort_values(kind="stable").index           # samples grouped by study
pct, mean, author = pct.loc[order], mean.loc[order], author.loc[order]

gon = [g for g in GENES["Gonadotroph"] if g in genes]
oth = [g for g in genes if g not in gon]
summ = pd.DataFrame({"Author": author, "n_tumour_cells": n_t.loc[order],
                     "pct_any_gonadotroph": (E[gon] > 0).any(axis=1).groupby(s_t).mean().loc[order] * 100,
                     "pct_any_other_lineage": (E[oth] > 0).any(axis=1).groupby(s_t).mean().loc[order] * 100})
summ = summ.join(pct.add_prefix("pct_")).round(1)
summ.to_csv("SuppFig1B_tumour_lineage_markers_per_sample.csv")
print(summ[["Author", "n_tumour_cells", "pct_any_gonadotroph", "pct_any_other_lineage"]].to_string())
print(f"\nany gonadotroph marker: median {summ.pct_any_gonadotroph.median():.1f}%, "
      f"range {summ.pct_any_gonadotroph.min():.1f}–{summ.pct_any_gonadotroph.max():.1f}%")
print(f"any other-lineage marker: median {summ.pct_any_other_lineage.median():.1f}%, "
      f"range {summ.pct_any_other_lineage.min():.1f}–{summ.pct_any_other_lineage.max():.1f}%")
print("gonadotroph > other lineage in every sample:",
      bool((summ.pct_any_gonadotroph > summ.pct_any_other_lineage).all()))

high = summ[summ.pct_any_other_lineage > HIGH_OTHER]
if len(high):
    print(f"\nsamples with > {HIGH_OTHER}% other-lineage expression — % tumour cells per gene:")
    print(high[["Author", "pct_any_other_lineage"] + [f"pct_{g}" for g in oth]].to_string())

# study colours: fall back to grey for any study not in the palette
studies = [s for s in SPAL if s in set(author)] + sorted(set(author) - set(SPAL))
unmatched = sorted(set(author) - set(SPAL))
if unmatched: print("studies not in SPAL (shown grey — check spelling):", unmatched)
col_of = lambda s: SPAL.get(s, "#bbbbbb")

# ---- figure ----------------------------------------------------------------------------------
fig = plt.figure(figsize=(180 * MM_, 135 * MM_))
outer = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.0], hspace=0.22,
                         left=0.1, right=0.97, top=0.96, bottom=0.13)
g0 = outer[0].subgridspec(1, 4, wspace=0.4)
g1 = outer[1].subgridspec(1, 2, width_ratios=[1, 0.14], wspace=0.04)

# UMAPs of gonadotroph markers on tumour cells
um = adata.obsm["X_umap"][tmask]
lo, hi = np.percentile(um, ZOOM_PCT, axis=0), np.percentile(um, 100 - ZOOM_PCT, axis=0)
pad = 0.05 * (hi - lo)
pt = max(0.3, min(4, 30000 / tmask.sum()))
for j, g in enumerate(gon):
    ax = fig.add_subplot(g0[j])
    v = E[g].to_numpy()
    pos = v[v > 0]
    norm = Normalize(0, np.percentile(pos, 99) if pos.size else 1, clip=True)
    o = np.argsort(v, kind="stable")                      # expressing cells on top
    ax.scatter(um[o, 0], um[o, 1], c=UMAP_CMAP(norm(v[o])), s=pt, linewidths=0, rasterized=True)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"$\\it{{{g}}}$  ({(v > 0).mean() * 100:.0f}% of cells)", fontsize=6.5, pad=2)
    cax = inset_axes(ax, width="4%", height="70%", loc="center right",
                     bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=UMAP_CMAP), cax=cax, extend="max")
    cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)

# dot plot, samples x markers
ax = fig.add_subplot(g1[0])
ns, ng = len(order), len(genes)
yy = {g: ng - 1 - i for i, g in enumerate(genes)}
vmax = np.percentile(mean.values, 99) or 1
dnorm = Normalize(0, vmax, clip=True)
for g in genes:
    ax.scatter(np.arange(ns), np.full(ns, yy[g]), s=pct[g].values / 100 * SMAX,
               c=DOT_CMAP(dnorm(mean[g].values)), linewidths=0.3, edgecolors=EDGE)

# lineage separators and labels
first = True
for lin, gl in GENES.items():
    gl = [g for g in gl if g in genes]
    if not gl: continue
    ytop, ybot = yy[gl[0]] + 0.5, yy[gl[-1]] - 0.5
    if not first: ax.axhline(ytop, c="k", lw=0.4)
    ax.annotate(lin, xy=(-0.075, (ytop + ybot) / 2), xycoords=("axes fraction", "data"),
                rotation=90, ha="center", va="center", fontsize=5.5)
    first = False

# study strip on top of the plot
ax.scatter(np.arange(ns), np.full(ns, ng - 0.15), marker="s", s=16,
           c=[col_of(a) for a in author], linewidths=0, clip_on=False)

ax.set_xlim(-0.7, ns - 0.3); ax.set_ylim(-0.6, ng + 0.1)
ax.set_xticks(np.arange(ns)); ax.set_xticklabels(order, rotation=90, fontsize=4.5)
ax.set_yticks([yy[g] for g in genes]); ax.set_yticklabels(genes, fontsize=5.5, style="italic")
ax.tick_params(length=1.5, width=0.4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_xlabel("Tumour sample", fontsize=6)

# legends
axL = fig.add_subplot(g1[1]); axL.axis("off")
h_size = [Line2D([], [], marker="o", ls="", mfc="#9e9e9e", mec=EDGE, mew=0.3,
                 ms=np.sqrt(p / 100 * SMAX), label=f"{p}%") for p in (10, 25, 50, 75)]
l1 = axL.legend(handles=h_size, title="% tumour cells\nexpressing", title_fontsize=5,
                fontsize=5, loc="upper left", frameon=False, labelspacing=0.9, borderaxespad=0)
axL.add_artist(l1)
axL.legend(handles=[Patch(fc=col_of(s), label=s) for s in studies], title="Study",
           title_fontsize=5, fontsize=4.5, loc="lower left", frameon=False,
           handlelength=0.8, handleheight=0.8, borderaxespad=0)
cax = inset_axes(axL, width="12%", height="22%", loc="center left",
                 bbox_to_anchor=(0.05, 0.02, 1, 1), bbox_transform=axL.transAxes, borderpad=0)
cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap=DOT_CMAP), cax=cax, extend="max")
cb.set_label("Mean expression", fontsize=5); cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4)
cb.outline.set_linewidth(0.4)

save_fig(fig, "SuppFig1B_tumour_lineage_markers")
plt.show()


####################################################################################################
# SECTION 4a  Supplementary Fig. 2C: composition model (run before the assembly)
####################################################################################################
# ---- Supp. Fig. 2C: compositional differences, tumour vs healthy (propeller-style) — standalone
import numpy as np, pandas as pd, re
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

CT_COL      = "cell_type"
DENOM       = "non_endocrine"   # "non_endocrine" = proportions among non-endocrine cells (recommended)
                                # "all"           = proportions among all cells
ENDOCRINE   = ["Gonadotrophs", "Somatotrophs", "Lactotrophs", "Corticotrophs", "Thyrotrophs"]
MIN_TOTAL   = 50
MIN_SAMPLES = 3
FDR_CUT     = 0.05
SHOW_UNADJ  = True
MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6})
COND_PAL = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
pretty = lambda s: s.replace("_", " ").replace("Gonadotrophs", "Gonadotrophs / tumour cells")
IMMUNE = re.compile(r"macrophage|monocyte|dendritic|pdc|mast|neutrophil|t_cell|t_reg|nk_cell|ilc|b_cell|plasma", re.I)

def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")

# ---- per-sample counts and proportions -------------------------------------------------------
obs  = adata.obs
sid  = obs["SRA_ID"].astype(str)
ct   = obs[CT_COL].astype(str)
counts = pd.crosstab(sid, ct)
if DENOM == "non_endocrine":
    counts = counts.drop(columns=[c for c in ENDOCRINE if c in counts.columns])
meta = obs.assign(_s=sid).groupby("_s")[["condition", "Author"]].first().astype(str).loc[counts.index]
meta["tumour"] = (meta["condition"] == "Tumour").astype(int)
meta["study"]  = meta["Author"].astype(str)

immune_types = [c for c in counts.columns if IMMUNE.search(c)]
counts["All immune cells"] = counts[immune_types].sum(axis=1)
N = counts.drop(columns="All immune cells").sum(axis=1)
print(f"denominator: {DENOM}; cells per sample in denominator: min {N.min()}, median {int(N.median())}")
low = N[N < 50]
if len(low): print("samples with < 50 cells in the denominator (noisy):", low.to_dict())

keep = [c for c in counts.columns
        if counts[c].sum() >= MIN_TOTAL and (counts[c] > 0).sum() >= MIN_SAMPLES]
skipped = sorted(set(counts.columns) - set(keep))
if skipped: print("not tested (too few cells/samples):", skipped)

props = (counts[keep] + 0.5).div(N + 1, axis=0)
logit = np.log(props / (1 - props))

mixed = meta.groupby("study")["condition"].nunique()
print("studies with both conditions (drive the study-adjusted estimate):", list(mixed[mixed > 1].index))

# ---- models ----------------------------------------------------------------------------------
TERM = "tumour"
def fit(c, formula):
    d = meta.assign(y=logit[c].values)
    f = smf.ols(formula, d).fit()
    est, (lo, hi), p = f.params[TERM], f.conf_int().loc[TERM], f.pvalues[TERM]
    return est / np.log(2), lo / np.log(2), hi / np.log(2), p

rows = []
isH, isT = meta.condition == "Healthy", meta.condition == "Tumour"
for c in keep:
    e, lo, hi, p = fit(c, "y ~ tumour + study")
    eu, lou, hiu, pu = fit(c, "y ~ tumour")
    rows.append(dict(cell_type=c,
                     mean_pct_healthy=counts.loc[isH, c].div(N[isH]).mean() * 100,
                     mean_pct_tumour=counts.loc[isT, c].div(N[isT]).mean() * 100,
                     log2OR=e, ci_low=lo, ci_high=hi, p=p,
                     log2OR_unadj=eu, ci_low_unadj=lou, ci_high_unadj=hiu, p_unadj=pu))
res = pd.DataFrame(rows).set_index("cell_type")
res["FDR"] = multipletests(res.p, method="fdr_bh")[1]
res["FDR_unadj"] = multipletests(res.p_unadj, method="fdr_bh")[1]
res = res.sort_values("log2OR")
res.round(4).to_csv(f"SuppFig1c_composition_stats_{DENOM}.csv")
print(res[["mean_pct_healthy", "mean_pct_tumour", "log2OR", "ci_low", "ci_high", "FDR",
           "log2OR_unadj", "FDR_unadj"]].round(3).iloc[::-1].to_string())

# ---- plot ------------------------------------------------------------------------------------
order = [c for c in res.index if c != "All immune cells"] + \
        (["All immune cells"] if "All immune cells" in res.index else [])
r = res.loc[order]
fig, ax = plt.subplots(figsize=(90 * MM_, (len(r) * 4.2 + 22) * MM_))
fig.subplots_adjust(left=0.36, right=0.78, top=0.94, bottom=0.14)
for i, (k, x) in enumerate(r.iterrows()):
    sig = x.FDR < FDR_CUT
    col = (COND_PAL["Tumour"] if x.log2OR > 0 else COND_PAL["Healthy"]) if sig else "#9a9a9a"
    if SHOW_UNADJ:
        ax.plot([x.ci_low_unadj, x.ci_high_unadj], [i - 0.22] * 2, c="#cccccc", lw=0.7, zorder=1)
        ax.scatter(x.log2OR_unadj, i - 0.22, s=10, marker="D", fc="white", ec="#aaaaaa", lw=0.5, zorder=2)
    ax.plot([x.ci_low, x.ci_high], [i, i], c=col, lw=1.0, zorder=3)
    ax.scatter(x.log2OR, i, s=20, c=col, ec="white", lw=0.4, zorder=4)
    f = x.FDR
    ax.text(1.03, i, ("FDR < 0.001" if f < 0.001 else f"FDR = {f:.3f}") + (" *" if sig else ""),
            transform=ax.get_yaxis_transform(), va="center", fontsize=5,
            color="#111111" if sig else "#777777")
if "All immune cells" in r.index:
    ax.axhline(len(r) - 1.5, c="k", lw=0.4)
ax.axvline(0, c="k", lw=0.6, ls="--", zorder=0)
ax.set_yticks(range(len(r)))
ax.set_yticklabels([pretty(k) for k in r.index], fontsize=5.5)
for t in ax.get_yticklabels():
    if t.get_text() == "All immune cells": t.set_fontweight("bold")
ax.set_ylim(-0.7, len(r) - 0.4)
denom_lbl = "proportion of non-endocrine cells" if DENOM == "non_endocrine" else "proportion of all cells"
ax.set_xlabel(f"log$_2$ odds ratio, tumour vs healthy\n({denom_lbl}; study-adjusted, 95% CI)", fontsize=6)
ax.tick_params(axis="x", labelsize=5.5, length=2, width=0.5); ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
xl = ax.get_xlim()
ax.text(xl[1], len(r) - 0.1, "Enriched in tumour →", ha="right", fontsize=5, color=COND_PAL["Tumour"])
ax.text(xl[0], len(r) - 0.1, "← Depleted in tumour", ha="left", fontsize=5, color=COND_PAL["Healthy"])
if SHOW_UNADJ:
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="-", c="#555555", ms=3.5, lw=1, label="Study-adjusted"),
                       plt.Line2D([], [], marker="D", ls="-", c="#bbbbbb", mfc="white", ms=3, lw=0.7,
                                  label="Unadjusted")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5, frameon=False)
save_fig(fig, f"SuppFig2C_composition_forest_{DENOM}")
plt.show()

# ---- draft results sentences -----------------------------------------------------------------
def fmt(k, adj=True):
    x = res.loc[k]
    e, lo, hi, f = (x.log2OR, x.ci_low, x.ci_high, x.FDR) if adj else \
                   (x.log2OR_unadj, x.ci_low_unadj, x.ci_high_unadj, x.FDR_unadj)
    fs = "FDR < 0.001" if f < 0.001 else f"FDR = {f:.3f}"
    return f"{pretty(k).lower()} (log2 OR {e:.2f}, 95% CI {lo:.2f} to {hi:.2f}; {fs})"
sig = res[(res.FDR < FDR_CUT) & (res.index != "All immune cells")]
print("\n--- draft text (study-adjusted) ---")
if "All immune cells" in res.index:
    print("Immune cells overall:", fmt("All immune cells"), "| unadjusted:", fmt("All immune cells", False))
print("Enriched:", "; ".join(fmt(k) for k in sig[sig.log2OR > 0].index[::-1]) or "none")
print("Depleted:", "; ".join(fmt(k) for k in sig[sig.log2OR < 0].index) or "none")
robust = [k for k in sig.index if res.loc[k, "FDR_unadj"] < FDR_CUT and
          np.sign(res.loc[k, "log2OR_unadj"]) == np.sign(res.loc[k, "log2OR"])]
print("Robust (significant, same direction in both models):", [pretty(k) for k in robust])


####################################################################################################
# SECTION 4b  Supplementary Fig. 2A, 2B: inferred copy-number variation
####################################################################################################
# ---- Supp. Fig.: inferred copy-number variation in tumour cells — standalone -----------------
import os, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp, re
try:
    pd.options.future.infer_string = False      # infercnvpy can't handle Arrow-backed strings
except Exception:
    pass
import infercnvpy as cnv
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgba
from scipy.stats import mannwhitneyu, wilcoxon

CT_COL        = "cell_type"
GONAD_CT      = "Gonadotrophs"   # tumour cells in tumour samples; reference in healthy samples
EXPR_LAYER    = None             # None = adata.X (log-normalised)
GTF_URL       = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
                 "gencode.v44.basic.annotation.gtf.gz")
GTF_PATH      = "gencode.v44.basic.annotation.gtf.gz"
MIN_TUMOUR    = 20               # min tumour cells for a sample to be included
MAX_TUMOUR    = 300              # cells sampled per tumour sample
MAX_REF       = 300              # healthy gonadotrophs sampled per healthy sample (split ref / held-out)
MAX_CTRL      = 150              # immune control cells sampled per tumour sample
MIN_CTRL_PAIR = 10               # min control cells for a within-sample paired comparison
WINDOW        = 250
EVENT_SD      = 3                # call a chromosome event if > this many SDs from healthy samples
N_JOBS        = 4
SEED = 0
rng = np.random.default_rng(SEED)
MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6})
COND_PAL = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
IMMUNE = re.compile(r"macrophage|monocyte|dendritic|t_cell|t_reg|nk_cell|b_cell", re.I)

def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")

def plain(x):
    """Plain Python-object array of strings (no Arrow)."""
    return np.asarray([str(v) for v in x], dtype=object)

# ---- pick cells ------------------------------------------------------------------------------
obs   = adata.obs
ct    = plain(obs[CT_COL])
cond  = plain(obs["condition"])
sid   = plain(obs["SRA_ID"])

def sample_idx(mask, cap):
    out = []
    for s in np.unique(sid[mask]):
        i = np.flatnonzero(mask & (sid == s))
        out.append(rng.choice(i, min(cap, len(i)), replace=False))
    return np.concatenate(out) if out else np.array([], int)

tum_mask = (ct == GONAD_CT) & (cond == "Tumour")
n_tum = pd.Series(sid[tum_mask]).value_counts()
tum_mask &= np.isin(sid, n_tum.index[n_tum >= MIN_TUMOUR])
i_tum  = sample_idx(tum_mask, MAX_TUMOUR)
i_ctrl = sample_idx((cond == "Tumour") & np.array([bool(IMMUNE.search(c)) for c in ct]), MAX_CTRL)
i_hg   = sample_idx((ct == GONAD_CT) & (cond == "Healthy"), MAX_REF)

# split healthy gonadotrophs 50/50 within each sample: reference vs held-out
is_ref = np.zeros(len(i_hg), bool)
for s in np.unique(sid[i_hg]):
    j = np.flatnonzero(sid[i_hg] == s)
    is_ref[rng.choice(j, len(j) // 2, replace=False)] = True

idx = np.concatenate([i_tum, i_ctrl, i_hg])
group = (["Tumour cells"] * len(i_tum) + ["Tumour-sample immune (control)"] * len(i_ctrl) +
         ["Reference" if r else "Healthy gonadotrophs (held-out)" for r in is_ref])
ad = adata[idx].copy()
if EXPR_LAYER is not None: ad.X = ad.layers[EXPR_LAYER].copy()
ad.obs_names = pd.Index(plain(ad.obs_names), dtype=object)
ad.obs["cnv_group"] = pd.Categorical(group)
ad.obs["cnv_ref"] = pd.Categorical(np.where(np.array(group) == "Reference",
                                            "ref_" + plain(ad.obs["Author"]), "query").astype(object))
ref_cats = sorted(c for c in ad.obs["cnv_ref"].cat.categories if c.startswith("ref_"))
print(ad.obs["cnv_group"].value_counts().to_dict())
print("reference categories (one per healthy study):", ref_cats)

# ---- gene positions (from GENCODE GTF) -------------------------------------------------------
if not os.path.exists(GTF_PATH):
    print("downloading GENCODE v44 annotation...")
    urllib.request.urlretrieve(GTF_URL, GTF_PATH)
gtf = pd.read_csv(GTF_PATH, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 8],
                  names=["chromosome", "feature", "start", "end", "attr"], dtype={0: object, 2: object, 8: object})
gtf = gtf[gtf.feature == "gene"]
gtf["gene_name"] = gtf.attr.str.extract(r'gene_name "([^"]+)"', expand=False)
gtf = gtf[gtf.chromosome.isin([f"chr{i}" for i in range(1, 23)] + ["chrX"])]
gtf = gtf.drop_duplicates("gene_name").set_index("gene_name")

genes = plain(ad.var_names)
ad.var_names = pd.Index(genes, dtype=object)
ad.var["chromosome"] = gtf["chromosome"].reindex(genes).values
ad.var["start"] = gtf["start"].reindex(genes).values
ad.var["end"]   = gtf["end"].reindex(genes).values
ad = ad[:, pd.notna(ad.var["chromosome"]).values].copy()

# force plain object dtype for everything infercnvpy indexes
ad.var = pd.DataFrame({"chromosome": plain(ad.var["chromosome"]),
                       "start": ad.var["start"].astype(int).values,
                       "end": ad.var["end"].astype(int).values},
                      index=pd.Index(plain(ad.var_names), dtype=object))
print(f"{ad.n_vars:,} of {adata.n_vars:,} genes matched to genomic positions (GENCODE v44)")

# ---- inferCNV --------------------------------------------------------------------------------
cnv.tl.infercnv(ad, reference_key="cnv_ref", reference_cat=ref_cats,
                window_size=WINDOW, n_jobs=N_JOBS)
Xc = ad.obsm["X_cnv"]; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
ad.obs["cnv_score"] = (Xc ** 2).mean(axis=1)

q = ad[ad.obs["cnv_group"] != "Reference"].copy()
q.obs["cnv_group"] = q.obs["cnv_group"].cat.remove_unused_categories()
cnv.pl.chromosome_heatmap(q, groupby="cnv_group", show=False)
plt.savefig("SuppFig_CNV_chromosome_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig("SuppFig_CNV_chromosome_heatmap.pdf", bbox_inches="tight")
plt.show()

# ---- per-sample CNV scores -------------------------------------------------------------------
qo = q.obs.assign(sample=plain(q.obs["SRA_ID"]))
per = qo.groupby(["sample", "cnv_group"], observed=True)["cnv_score"].agg(["median", "size"]).reset_index()
per.to_csv("SuppFig_CNV_scores_per_sample.csv", index=False)
wide = per.pivot(index="sample", columns="cnv_group", values="median")
size = per.pivot(index="sample", columns="cnv_group", values="size")

T, Cn, H = "Tumour cells", "Tumour-sample immune (control)", "Healthy gonadotrophs (held-out)"
p_th = mannwhitneyu(wide[T].dropna(), wide[H].dropna()).pvalue
pair = wide[[T, Cn]].dropna()
pair = pair[size.loc[pair.index, Cn] >= MIN_CTRL_PAIR]
p_tc = wilcoxon(pair[T], pair[Cn]).pvalue if len(pair) >= 5 else np.nan
print(f"\nmedian CNV score — tumour: {wide[T].median():.4g}, own-sample immune: {wide[Cn].median():.4g}, "
      f"healthy gonadotrophs: {wide[H].median():.4g}")
print(f"tumour vs held-out healthy gonadotrophs (per-sample medians, Mann–Whitney): p = {p_th:.3g}")
print(f"tumour vs own-sample immune cells (paired Wilcoxon, n = {len(pair)} samples): p = {p_tc:.3g}")
print(f"samples where tumour > own immune: {(pair[T] > pair[Cn]).sum()} of {len(pair)}")

# ---- per-sample x chromosome matrix and event calls ------------------------------------------
pos = sorted(ad.uns["cnv"]["chr_pos"].items(), key=lambda kv: kv[1])
bounds = [(ch, s, pos[k + 1][1] if k + 1 < len(pos) else Xc.shape[1]) for k, (ch, s) in enumerate(pos)]
chr_cols = [ch for ch, _, _ in bounds]
chrm = pd.DataFrame({ch: Xc[:, s:e].mean(axis=1) for ch, s, e in bounds}, index=ad.obs_names)
chrm["sample"] = plain(ad.obs["SRA_ID"])
chrm["grp"] = plain(ad.obs["cnv_group"])

tum_chr = chrm[chrm.grp == T].groupby("sample")[chr_cols].mean()
hlt_chr = chrm[chrm.grp == H].groupby("sample")[chr_cols].mean()
mu, sd = hlt_chr.mean(), hlt_chr.std().replace(0, np.nan)
z = (tum_chr - mu) / sd
events = z.stack().loc[lambda s: s.abs() > EVENT_SD].rename("z").reset_index()
events.columns = ["sample", "chromosome", "z"]
events["call"] = np.where(events.z > 0, "gain", "loss")
auth = pd.Series(plain(obs["Author"]), index=sid).groupby(level=0).first()
events["Author"] = events["sample"].map(auth)
events.to_csv("SuppFig_CNV_chromosome_events.csv", index=False)
print(f"\nchromosome-level events (|z| > {EVENT_SD} vs healthy gonadotrophs): "
      f"{len(events)} in {events['sample'].nunique()} of {len(tum_chr)} tumour samples")
print(events.sort_values(["sample", "chromosome"]).to_string(index=False) if len(events) else "none")

# ---- figure: per-sample chromosome heatmap + CNV scores --------------------------------------
fig = plt.figure(figsize=(180 * MM_, 80 * MM_))
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.35, left=0.1, right=0.97, top=0.9, bottom=0.2)

ax = fig.add_subplot(gs[0])
order = tum_chr.index.to_series().map(auth).sort_values(kind="stable").index
d = (tum_chr - mu).loc[order]
lim = np.nanpercentile(np.abs(d.values), 99) or 0.01
im = ax.imshow(d.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
if len(events):
    rmap = {s: i for i, s in enumerate(order)}; cmap_ = {ch: i for i, ch in enumerate(chr_cols)}
    for _, e in events.iterrows():
        ax.scatter(cmap_[e.chromosome], rmap[e["sample"]], s=4, c="k", marker="*", linewidths=0)
ax.set_xticks(range(len(chr_cols))); ax.set_xticklabels([c.replace("chr", "") for c in chr_cols], fontsize=4.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=4.5)
ax.set_xlabel("Chromosome", fontsize=6)
ax.set_title("Tumour cells: mean CNV signal relative to healthy gonadotrophs", fontsize=6.5, pad=3)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="both")
cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label("Inferred CNV (gain > 0 > loss)", fontsize=5)

ax2 = fig.add_subplot(gs[1])
cols = {H: COND_PAL["Healthy"], Cn: "#888888", T: COND_PAL["Tumour"]}
labels = {H: "Healthy\ngonadotrophs", Cn: "Tumour-sample\nimmune", T: "Tumour\ncells"}
for i, g in enumerate([H, Cn, T]):
    v = wide[g].dropna()
    ax2.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                boxprops=dict(fc=to_rgba(cols[g], 0.2), ec=cols[g], lw=0.8),
                medianprops=dict(c=cols[g], lw=1.2), whiskerprops=dict(c=cols[g], lw=0.8),
                capprops=dict(c=cols[g], lw=0.8))
    ax2.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=6, c=[cols[g]], ec="white", lw=0.3, zorder=3)
for _, r in pair.iterrows():
    ax2.plot([1, 2], [r[Cn], r[T]], c="#bbbbbb", lw=0.4, zorder=1)
fp = lambda p: "n.s." if not np.isfinite(p) or p >= 0.05 else (f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}")
top = np.nanmax(wide.values); rngv = top - np.nanmin(wide.values)
for (a, b, p, k) in [(1, 2, p_tc, 1), (0, 2, p_th, 2)]:
    y0 = top + (0.06 + 0.12 * (k - 1)) * rngv
    ax2.plot([a, a, b, b], [y0, y0 + 0.03 * rngv, y0 + 0.03 * rngv, y0], c="k", lw=0.5)
    ax2.text((a + b) / 2, y0 + 0.04 * rngv, fp(p), ha="center", fontsize=5)
ax2.set_xticks(range(3)); ax2.set_xticklabels([labels[g] for g in [H, Cn, T]], fontsize=5)
ax2.set_ylabel("CNV score per sample\n(median of cells)", fontsize=6)
ax2.tick_params(axis="y", labelsize=5)
for s_ in ("top", "right"): ax2.spines[s_].set_visible(False)

save_fig(fig, "SuppFig_CNV_per_sample")
plt.show()


####################################################################################################
# SECTION 4c  Supplementary Fig. 2A-C: assembled
####################################################################################################
# ---- Supp. Fig. 2 a–c at print size (run after the CNV cell) ---------------------------------
import os, re, numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

SF_N_PER_GROUP = 2500        # cells drawn per group in the heatmap
SF_VMAX_PCT    = 99.5        # colour limit percentile for the heatmap
SF_CANDIDATES  = [("SRX21219342", "chr13"), ("SRX16175537", "chr18"), ("SRX21896007", "chr18")]
SF_FOREST_CSV  = "SuppFig1c_composition_stats_non_endocrine.csv"
SF_TUMOUR_CT   = "Gonadotrophs"
SF_BAR_LABELS  = {"1", "5", "10", "13", "18", "22"}   # chromosomes labelled in the panel b bar charts
SF_FDR = 0.05
SF_SEED = 0
MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6})
sf_rng = np.random.default_rng(SF_SEED)

SF_COND = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
_SF_STUDY_RAW = {
    "Asuzu et al., (2022)": "#1f77b4", "Huang et al., (2025)": "#ff7f0e", "Ilie et al. (2025)": "#2ca02c",
    "Potthoff et al., (2024)": "#d62728", "Yan et al., (2024)": "#9467bd", "Zhang et al., (2022c)": "#8c564b",
    "Zhang et al. (2026)": "#e377c2", "Zhang et al., (2022b)": "#7f7f7f"}
_SF_STUDY_RAW.update(globals().get("STUDY_PAL", {}))          # your Fig. 1 palette wins if defined
_sf_norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # ignore punctuation, spaces, case
SF_STUDY = {_sf_norm(k): v for k, v in _SF_STUDY_RAW.items()}
sf_study_col = lambda a, default="#bbbbbb": SF_STUDY.get(_sf_norm(a), default)
sf_study_txt = lambda a: re.sub(r"[(),]", "", str(a))         # "Ilie et al. (2025)" -> "Ilie et al. 2025"
H_, T_ = "Healthy gonadotrophs (held-out)", "Tumour cells"

assert "ad" in globals() and "Xc" in globals() and "cnv" in ad.uns, "Run the CNV cell first."
assert Xc.shape[0] == ad.n_obs, "`Xc` does not match `ad` — rerun the CNV cell."
sf_plain = lambda x: np.asarray([str(v) for v in x], dtype=object)
sf_grp, sf_smp = sf_plain(ad.obs["cnv_group"]), sf_plain(ad.obs["SRA_ID"])
sf_auth = pd.Series(sf_plain(ad.obs["Author"]), index=sf_smp).groupby(level=0).first()
unmatched = sorted({a for a in sf_auth.values if _sf_norm(a) not in SF_STUDY})
if unmatched: print("studies without a colour (shown grey):", unmatched)

sf_pos = sorted(ad.uns["cnv"]["chr_pos"].items(), key=lambda kv: kv[1])
sf_bounds = {ch: (s, sf_pos[k + 1][1] if k + 1 < len(sf_pos) else Xc.shape[1]) for k, (ch, s) in enumerate(sf_pos)}
sf_chroms = list(sf_bounds)

# true tumour-cell counts per sample (from the full atlas)
_all_sid = sf_plain(adata.obs["SRA_ID"])
_all_tum = (sf_plain(adata.obs["cell_type"]) == SF_TUMOUR_CT) & (sf_plain(adata.obs["condition"]) == "Tumour")
sf_ntotal = pd.Series(_all_sid[_all_tum]).value_counts()

def sf_save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")

def sf_box(ax, data, pos, cols):
    for v, x, c in zip(data, pos, cols):
        v = pd.Series(v).dropna()
        ax.boxplot(v, positions=[x], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=to_rgba(c, 0.2), ec=c, lw=0.8), medianprops=dict(c=c, lw=1.2),
                   whiskerprops=dict(c=c, lw=0.8), capprops=dict(c=c, lw=0.8))
        ax.scatter(x + sf_rng.uniform(-0.14, 0.14, len(v)), v, s=7, c=[c], ec="white", lw=0.3, zorder=3)

sf_fp = lambda p: "n.s." if not np.isfinite(p) or p >= 0.05 else (f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}")

# =============================== panel a ======================================================
rows_by_grp = []
for g in (H_, T_):
    i = np.flatnonzero(sf_grp == g)
    i = sf_rng.choice(i, min(SF_N_PER_GROUP, len(i)), replace=False)
    k = pd.DataFrame({"i": i, "a": sf_auth.reindex(sf_smp[i]).values, "s": sf_smp[i]}).sort_values(["a", "s"])
    rows_by_grp.append(k["i"].to_numpy())
rows = np.concatenate(rows_by_grp); n_h = len(rows_by_grp[0])
M = Xc[rows]
vmax = np.percentile(np.abs(M), SF_VMAX_PCT) or 0.05

fig = plt.figure(figsize=(180 * MM_, 80 * MM_))
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.42, left=0.09, right=0.98, top=0.88, bottom=0.22)
gl = gs[0].subgridspec(1, 3, width_ratios=[0.022, 0.022, 1], wspace=0.03)
ax_c, ax_s, ax_h = (fig.add_subplot(gl[j]) for j in range(3))

ax_c.imshow(np.array([to_rgba(SF_COND["Healthy"] if sf_grp[r] == H_ else SF_COND["Tumour"]) for r in rows])[:, None, :],
            aspect="auto", interpolation="nearest")
ax_s.imshow(np.array([to_rgba(sf_study_col(sf_auth.get(sf_smp[r], ""))) for r in rows])[:, None, :],
            aspect="auto", interpolation="nearest")
im = ax_h.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                 interpolation="nearest", rasterized=True)
for ch, (s, e) in sf_bounds.items():
    ax_h.axvline(s - 0.5, c="k", lw=0.3)
for a in (ax_c, ax_s, ax_h):
    a.axhline(n_h - 0.5, c="k", lw=0.7)
ax_c.set_xticks([]); ax_s.set_xticks([]); ax_s.set_yticks([])
ax_c.set_yticks([n_h / 2, n_h + len(rows_by_grp[1]) / 2])
ax_c.set_yticklabels(["Healthy\ngonadotrophs", "Tumour\ncells"], fontsize=5.5)
ax_c.tick_params(axis="y", length=0)
mids = [(s + e) / 2 for s, e in sf_bounds.values()]
ax_h.xaxis.tick_top()
ax_h.set_xticks(mids)
ax_h.set_xticklabels(["" if c in ("chr21",) else c.replace("chr", "") for c in sf_chroms], fontsize=4.8)
ax_h.tick_params(axis="x", length=0, pad=1); ax_h.set_yticks([])
ax_h.set_xlabel("Chromosome", fontsize=6); ax_h.xaxis.set_label_position("top")
for a in (ax_c, ax_s, ax_h):
    for sp_ in a.spines.values(): sp_.set_linewidth(0.4)
cax = ax_h.inset_axes([1.012, 0.3, 0.012, 0.4])
cb = fig.colorbar(im, cax=cax, extend="both"); cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4)
cb.outline.set_linewidth(0.4); cb.ax.set_title("CNV", fontsize=5, pad=3)

# study key under the heatmap (only studies present, in palette order)
present = {_sf_norm(sf_auth.get(sf_smp[r], "")) for r in rows}
key_studies = [k for k in _SF_STUDY_RAW if _sf_norm(k) in present]
key_studies += sorted({sf_auth.get(sf_smp[r], "") for r in rows} - set(key_studies), key=str) \
               if any(_sf_norm(a) not in SF_STUDY for a in key_studies) else []
ax_h.legend(handles=[Patch(fc=sf_study_col(s), ec="none", label=sf_study_txt(s)) for s in dict.fromkeys(key_studies)],
            title="Study", title_fontsize=5, fontsize=4.8, ncol=4, frameon=False,
            loc="upper left", bbox_to_anchor=(-0.06, -0.03), handlelength=0.9, handleheight=0.9,
            columnspacing=1.2, borderaxespad=0)

ax_b = fig.add_subplot(gs[1])
sc_per = (ad.obs.assign(_s=sf_smp, _g=sf_grp).query("_g in @H_ or _g in @T_")
          .groupby(["_s", "_g"])["cnv_score"].median().unstack())
vh, vt = sc_per[H_].dropna(), sc_per[T_].dropna()
p_ht = mannwhitneyu(vt, vh).pvalue
sf_box(ax_b, [vh, vt], [0, 1], [SF_COND["Healthy"], SF_COND["Tumour"]])
top = max(vh.max(), vt.max()); span = top - min(vh.min(), vt.min())
ax_b.plot([0, 0, 1, 1], [top + .06 * span, top + .09 * span, top + .09 * span, top + .06 * span], c="k", lw=0.5)
ax_b.text(0.5, top + .11 * span, sf_fp(p_ht), ha="center", fontsize=5.5)
ax_b.set_xticks([0, 1])
ax_b.set_xticklabels([f"Healthy\ngonadotrophs\n(n = {len(vh)})", f"Tumour\ncells\n(n = {len(vt)})"], fontsize=5.3)
ax_b.set_ylabel("CNV score per sample\n(median of cells)", fontsize=6)
ax_b.tick_params(axis="y", labelsize=5); ax_b.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
ax_b.yaxis.get_offset_text().set_fontsize(5)
for s_ in ("top", "right"): ax_b.spines[s_].set_visible(False)

sf_save(fig, "SuppFig2a_CNV")
plt.show()
print(f"panel a: CNV score tumour vs healthy gonadotrophs (Mann–Whitney, per-sample medians): p = {p_ht:.3g}")

# =============================== panels b + c =================================================
Xg = ad.X
def sf_gmean(mask):
    return np.asarray(Xg[np.flatnonzero(mask)].mean(axis=0)).ravel()
gchr = sf_plain(ad.var["chromosome"])
h_samples, t_samples = np.unique(sf_smp[sf_grp == H_]), np.unique(sf_smp[sf_grp == T_])

fig = plt.figure(figsize=(180 * MM_, 112 * MM_))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.62, left=0.07, right=0.97, top=0.93, bottom=0.16)
gd = gs[0].subgridspec(len(SF_CANDIDATES), 3, width_ratios=[1.5, 1, 1.35], hspace=0.95, wspace=0.6)
for r, (S, ch) in enumerate(SF_CANDIDATES):
    s0, e0 = sf_bounds[ch]
    tmask = (sf_grp == T_) & (sf_smp == S)
    col = sf_study_col(sf_auth.get(S, ""), "#8B1A1A")
    n_shown, n_tot = int(tmask.sum()), int(sf_ntotal.get(S, tmask.sum()))
    cnum = ch.replace("chr", "")
    # profile
    ax = fig.add_subplot(gd[r, 0])
    hmeans = np.array([Xc[(sf_grp == H_) & (sf_smp == h), s0:e0].mean(axis=0) for h in h_samples])
    lo5, hi95 = np.percentile(hmeans, [5, 95], axis=0)
    x = np.arange(e0 - s0)
    for o in t_samples:
        if o != S: ax.plot(x, Xc[(sf_grp == T_) & (sf_smp == o), s0:e0].mean(axis=0), c="#cccccc", lw=0.35, zorder=1)
    ax.fill_between(x, lo5, hi95, color="#cfdcec", lw=0, zorder=2)
    ax.plot(x, Xc[tmask, s0:e0].mean(axis=0), c=col, lw=1.1, zorder=3)
    ax.axhline(0, c="k", lw=0.4)
    cells_txt = f"{n_tot:,} tumour cells" if n_shown == n_tot else f"{n_shown} of {n_tot:,} tumour cells"
    ax.set_title(f"{S} ({sf_study_txt(sf_auth.get(S, ''))}), chromosome {cnum}; {cells_txt}",
                 fontsize=5.3, loc="left", pad=2)
    ax.set_xticks([]); ax.set_xlabel(f"position along chr{cnum} →", fontsize=5)
    ax.set_ylabel("Mean CNV", fontsize=5); ax.tick_params(labelsize=4.5)
    # cell distribution
    ax = fig.add_subplot(gd[r, 1])
    cm_ = Xc[:, s0:e0].mean(axis=1)
    h5 = np.percentile(cm_[sf_grp == H_], 5)
    bins = np.linspace(np.percentile(cm_, 0.5), np.percentile(cm_, 99.5), 35)
    ax.hist(cm_[sf_grp == H_], bins=bins, density=True, color="#9fbad6", alpha=0.75)
    ax.hist(cm_[tmask], bins=bins, density=True, color=col, alpha=0.65)
    ax.axvline(h5, c="k", lw=0.5, ls="--"); ax.set_yticks([])
    ax.set_xlabel("Per-cell mean CNV", fontsize=5); ax.tick_params(labelsize=4.5)
    ax.text(0.02, 0.97, f"{(cm_[tmask] < h5).mean() * 100:.0f}% of cells\n< healthy 5th pct",
            transform=ax.transAxes, ha="left", va="top", fontsize=4.5)
    # gene-level shift vs other tumours
    ax = fig.add_subplot(gd[r, 2])
    m_s, m_o = sf_gmean(tmask), sf_gmean((sf_grp == T_) & (sf_smp != S))
    expr = m_o > 0.10; diff = m_s - m_o; gmed = np.median(diff[expr])
    shift = pd.Series({c: np.median(diff[expr & (gchr == c)]) - gmed
                       for c in sf_chroms if (expr & (gchr == c)).sum() >= 10})
    ax.bar(range(len(shift)), shift.values, width=0.75,
           color=[col if c == ch else "#bbbbbb" for c in shift.index])
    ax.axhline(0, c="k", lw=0.4)
    ax.set_xticks(range(len(shift)))
    ax.set_xticklabels([c.replace("chr", "") if c.replace("chr", "") in SF_BAR_LABELS else ""
                        for c in shift.index], fontsize=4.2)
    ax.set_ylabel("Expr. shift vs\nother tumours", fontsize=5); ax.tick_params(axis="y", labelsize=4.5)
    for a in fig.axes[-3:]:
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)

# panel b key: once, below all three rows
fig.legend(handles=[plt.Line2D([], [], c="#444444", lw=1.1, label="This tumour (coloured by study)"),
                    plt.Line2D([], [], c="#cccccc", lw=0.8, label="Other tumours"),
                    Patch(fc="#cfdcec", ec="none", label="Healthy gonadotrophs (5–95%)"),
                    Patch(fc="#9fbad6", ec="none", label="Healthy gonadotroph cells"),
                    plt.Line2D([], [], c="k", lw=0.5, ls="--", label="Healthy 5th percentile")],
           loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, fontsize=4.6, frameon=False,
           handlelength=1.4, columnspacing=1.2)

# panel c: forest plot from saved composition stats
ax = fig.add_subplot(gs[1])
if os.path.exists(SF_FOREST_CSV):
    fr = pd.read_csv(SF_FOREST_CSV, index_col=0).sort_values("log2OR")
    order = [c for c in fr.index if c != "All immune cells"] + (["All immune cells"] if "All immune cells" in fr.index else [])
    fr = fr.loc[order]
    pretty = lambda s: s.replace("_", " ")
    for i, (k, x) in enumerate(fr.iterrows()):
        sig = x.FDR < SF_FDR
        c = (SF_COND["Tumour"] if x.log2OR > 0 else SF_COND["Healthy"]) if sig else "#9a9a9a"
        ax.plot([x.ci_low_unadj, x.ci_high_unadj], [i - 0.22] * 2, c="#cccccc", lw=0.7, zorder=1)
        ax.scatter(x.log2OR_unadj, i - 0.22, s=9, marker="D", fc="white", ec="#aaaaaa", lw=0.5, zorder=2)
        ax.plot([x.ci_low, x.ci_high], [i, i], c=c, lw=1.0, zorder=3)
        ax.scatter(x.log2OR, i, s=18, c=c, ec="white", lw=0.4, zorder=4)
        f = x.FDR
        ax.text(1.02, i, ("FDR < 0.001" if f < 0.001 else f"FDR = {f:.3f}"), transform=ax.get_yaxis_transform(),
                va="center", fontsize=4.6, color="#111111" if sig else "#777777")
    if "All immune cells" in fr.index: ax.axhline(len(fr) - 1.5, c="k", lw=0.4)
    ax.axvline(0, c="k", lw=0.6, ls="--", zorder=0)
    ax.set_yticks(range(len(fr))); ax.set_yticklabels([pretty(k) for k in fr.index], fontsize=5.3)
    for t in ax.get_yticklabels():
        if t.get_text() == "All immune cells": t.set_fontweight("bold")
    ax.set_ylim(-0.7, len(fr) - 0.3)
    ax.set_xlabel("log$_2$ odds ratio, tumour vs healthy\n(non-endocrine cells; 95% CI)", fontsize=5.5)
    ax.tick_params(axis="x", labelsize=5, length=2, width=0.5); ax.tick_params(axis="y", length=0)
    xl = ax.get_xlim()
    ax.text(xl[1], len(fr) - 0.2, "Enriched →", ha="right", fontsize=4.8, color=SF_COND["Tumour"])
    ax.text(xl[0], len(fr) - 0.2, "← Depleted", ha="left", fontsize=4.8, color=SF_COND["Healthy"])
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="-", c="#555555", ms=3.2, lw=1, label="Study-adjusted"),
                       plt.Line2D([], [], marker="D", ls="-", c="#bbbbbb", mfc="white", ms=2.8, lw=0.7, label="Unadjusted")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=4.8, frameon=False)
else:
    ax.text(0.5, 0.5, f"{SF_FOREST_CSV}\nnot found — rerun the composition cell", ha="center", va="center", fontsize=6)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)

sf_save(fig, "SuppFig2bc_CNV_losses_and_composition")
plt.show()


####################################################################################################
# SECTION 5a  Supplementary Table 1: collect sample metadata and cell counts
####################################################################################################
# ---- Supplementary Table 1: single-cell datasets and sample-level metadata ----
# !pip install -q scanpy openpyxl
import numpy as np, pandas as pd, scanpy as sc
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side

# adata = sc.read_h5ad("PATH/TO/atlas.h5ad")   # uncomment if adata is not already loaded
T1_RAW = "Supplementary_Table_1_raw.xlsx"
SAMPLE, STUDY, COND_COL, CT = "SRA_ID", "Author", "condition", "cell_type"

# Study-level details to fill in (accession, platform, healthy tissue source).
# Keys must match adata.obs["Author"] exactly - the cell prints them below.
STUDY_INFO = {
    # "Asuzu et al., (2022)":    dict(accession="GSE______", platform="10x Chromium 3' v_", healthy_source="..."),
    # "Huang et al., (2025)":    dict(accession="______",    platform="SeekOne scRNA-seq", healthy_source="..."),
    # "Zhang et al. (2026)":     dict(accession="______",    platform="10x Multiome (snRNA)", healthy_source="..."),
}

obs = adata.obs.copy()
print("studies:", sorted(obs[STUDY].astype(str).unique()))

# ---- cell counts per sample, by compartment ----
ENDO = ["Gonadotrophs", "Somatotrophs", "Lactotrophs", "Corticotrophs", "Thyrotrophs"]
STROMA = ["Endothelial_cells", "Mesenchymal_cells"]
def compartment(ct):
    if ct in ENDO: return "Endocrine"
    if ct == "Stem_cells": return "Stem cells"
    if ct == "Pituicytes": return "Pituicytes"
    if ct in STROMA: return "Stromal"
    return "Immune"
obs["_comp"] = obs[CT].astype(str).map(compartment)
counts = pd.crosstab(obs[SAMPLE].astype(str), obs["_comp"])
counts = counts.reindex(columns=["Endocrine", "Stem cells", "Pituicytes", "Stromal", "Immune"], fill_value=0)
counts.insert(0, "Cells after QC", counts.sum(axis=1))
# tumour cells = gonadotroph-lineage cells in tumour samples
gon = obs[obs[CT].astype(str) == "Gonadotrophs"].groupby(obs[SAMPLE].astype(str)).size()

# ---- sample-level metadata: any obs column constant within every sample ----
skip = {SAMPLE, CT, "_comp"}
skip |= {c for c in obs.columns if any(k in c.lower() for k in
         ["score", "leiden", "louvain", "n_genes", "n_counts", "total_counts", "pct_", "doublet", "umap", "_hi", "cnv"])}
cand = [c for c in obs.columns if c not in skip]
const = [c for c in cand if (obs.groupby(obs[SAMPLE].astype(str), observed=True)[c].nunique(dropna=False) <= 1).all()]
print("sample-level columns found:", const)
meta = obs.groupby(obs[SAMPLE].astype(str), observed=True)[const].first()

tab = meta.join(counts)
tab["Tumour cells"] = np.where(tab[COND_COL].astype(str) == "Tumour", gon.reindex(tab.index).fillna(0).astype(int), np.nan)
tab.index.name = "Sample"
tab = tab.reset_index()
for k in ["accession", "platform", "healthy_source"]:
    tab[k] = tab[STUDY].astype(str).map(lambda s: STUDY_INFO.get(s, {}).get(k, ""))

# order and names
front = ["Sample", STUDY, "accession", "platform", COND_COL]
rest = [c for c in const if c not in (STUDY, COND_COL)]
num = ["Cells after QC", "Tumour cells", "Endocrine", "Stem cells", "Pituicytes", "Stromal", "Immune"]
tab = tab[front + rest + num].sort_values([COND_COL, STUDY, "Sample"]).rename(columns={
    STUDY: "Study", "accession": "Accession", "platform": "Platform", COND_COL: "Condition"})

# ---- study summary ----
studies = (tab.groupby("Study").agg(Accession=("Accession", "first"), Platform=("Platform", "first"),
           **{"Healthy samples": ("Condition", lambda s: (s == "Healthy").sum()),
              "Tumour samples": ("Condition", lambda s: (s == "Tumour").sum()),
              "Cells after QC": ("Cells after QC", "sum")}).reset_index())
studies["Healthy tissue source"] = studies["Study"].map(lambda s: STUDY_INFO.get(s, {}).get("healthy_source", ""))

# ---- checks against the manuscript ----
print(f"samples: {len(tab)} (healthy {sum(tab.Condition=='Healthy')}, tumour {sum(tab.Condition=='Tumour')}); "
      f"cells: {tab['Cells after QC'].sum():,}  [manuscript: 50 (29 + 21), 285,907]")
print("tumour samples with >= 20 tumour cells:", int((tab['Tumour cells'] >= 20).sum()), "[manuscript: 19]")

# ---- write formatted workbook ----
F = lambda **k: Font(**{"name": "Arial", "size": 10, **k}); thin = Side(style="thin", color="000000")
def write(ws, title, df, notes, widths):
    ws["A1"] = title; ws["A1"].font = F(bold=True, size=11)
    for j, h in enumerate(df.columns, 1):
        c = ws.cell(3, j, h); c.font = F(bold=True); c.border = Border(top=thin, bottom=thin)
        c.alignment = Alignment(wrap_text=True, vertical="bottom")
    for i, row in enumerate(df.itertuples(index=False), 4):
        for j, v in enumerate(row, 1):
            v = None if (isinstance(v, float) and np.isnan(v)) else (int(v) if isinstance(v, (np.integer, float)) and float(v).is_integer() else v)
            c = ws.cell(i, j, v if v is not None else "–"); c.font = F()
            if isinstance(v, (int, float)): c.number_format = "#,##0"; c.alignment = Alignment(horizontal="right")
    last = 3 + len(df)
    for j in range(1, len(df.columns) + 1): ws.cell(last, j).border = Border(bottom=thin)
    for k, n in enumerate(notes):
        r = last + 2 + k; ws.cell(r, 1, n).font = F(size=9)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(df.columns))
        ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top"); ws.row_dimensions[r].height = 26
    for j, w in enumerate(widths, 1): ws.column_dimensions[ws.cell(3, j).column_letter].width = w
    ws.row_dimensions[3].height = 28; ws.freeze_panes = "B4"; ws.sheet_view.showGridLines = False

wb = Workbook()
write(wb.active, "Supplementary Table 1a Single-cell and single-nucleus RNA sequencing datasets", studies,
      ["Cell numbers are after quality control."], [26, 16, 24, 10, 10, 12, 30])
wb.active.title = "Studies"
write(wb.create_sheet("Samples"), "Supplementary Table 1b Sample-level metadata and cell numbers", tab,
      ["Tumour cells are gonadotroph-lineage cells in tumour samples. Endocrine: gonadotrophs (including tumour cells), "
       "somatotrophs, lactotrophs, corticotrophs and thyrotrophs; stromal: endothelial and mesenchymal cells; "
       "immune: all myeloid and lymphoid cells.", "–, not applicable or not available."],
      [16, 24, 14, 22, 10] + [12] * len(rest) + [10] * len(num))
wb.save(T1_RAW); print("saved", T1_RAW)
# from google.colab import files; files.download(T1_RAW)   # uncomment to download from Colab


####################################################################################################
# SECTION 5b  Supplementary Table 1: tidy and format
####################################################################################################
# ---- Supplementary Table 1: tidy and format (run after the previous cell) -------------------
import pandas as pd, numpy as np, re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
d=pd.read_excel(T1_RAW,'Samples',header=2)
d=d[d['Condition'].isin(['Healthy','Tumour'])].replace('–',np.nan)
study=d.Study.str.replace(r',? *\(?(\d{4}[a-z]?)\)?',r' (\1)',regex=True).str.replace(' et al. ',' et al. ').str.replace(r'et al\.,',"et al.",regex=True)
d['StudyC']=study.str.replace(r'\s+',' ',regex=True).str.strip()
def repo(a): return 'GEO' if str(a).startswith('GSM') else 'GSA-Human' if str(a).startswith('HRS') else 'SRA'
d['Repo']=d.GEO.map(repo)
mod={'sc':'scRNA-seq','sn':'snRNA-seq','multi_rna':'snMultiome (GEX)'}
def platform(r):
    m=mod[r.Modality]; 
    if r.StudyC.startswith('Huang'): return 'SeekOne '+m
    return '10x Genomics '+m
d['Plat']=d.apply(platform,axis=1)
def source(r):
    if r.Condition=='Tumour': return 'Tumour'
    if r.Subtype=='Residual_Normal': return {'Cushings':"Residual normal gland (Cushing's disease surgery)",'Acromegaly':'Residual normal gland (acromegaly surgery)'}[r.Disease]
    return 'Donor pituitary'
d['Source']=d.apply(source,axis=1)
d['TType']=d.apply(lambda r: '' if r.Condition=='Healthy' else ('Null cell' if r.Subtype=='Null' else 'Silent gonadotroph' if r.Subtype=='Silent Gonadotroph' else 'Gonadotroph'),axis=1)
sexmap={'m':'M','male':'M','f':'F','female':'F'}
d['SexC']=d.Sex.astype(str).str.lower().map(sexmap)
tum=d.Condition=='Tumour'
def kn(v):
    if pd.isna(v): return np.nan
    v=str(v).upper().replace('.0',''); return v
def ki(v):
    if pd.isna(v): return np.nan
    v=str(v).replace(',','.').strip(); return v if v.endswith('%') else v+'%'
d['Knosp']=np.where(tum,d.KNOSP.map(kn),np.nan); d['Ki']=np.where(tum,d.KI67.map(ki),np.nan)
d['Inv']=np.where(tum,d.Invasion.map({'Invasive':'Invasive','Non_invasive':'Non-invasive'}),np.nan)
d['PR']=np.where(tum,d.Primary_recurrent,np.nan)
d['Pool']=d.Pool.map({'single':'No','multiplexed':'Yes'})
d['Age']=pd.to_numeric(d.Age_numeric,errors='coerce')
for c in ['Cells after QC','Endocrine','Stem cells','Pituicytes','Stromal','Immune']: d[c]=d[c].astype(int)
d['Tumour cells']=pd.to_numeric(d['Tumour cells'],errors='coerce')
d=d.sort_values(['Condition','StudyC','GEO'])
S=pd.DataFrame({'Accession':d.GEO,'Original sample name':d.Name,'Study':d.StudyC,'Repository':d.Repo,'Platform':d.Plat,'Multiplexed':d.Pool,
 'Condition':d.Condition,'Tissue source':d.Source,'Tumour type':d.TType.replace('',np.nan),'Age (years)':d.Age,'Sex':d.SexC,
 'Knosp grade':d.Knosp,'Ki-67':d.Ki,'Invasion':d.Inv,'Primary/recurrent':d.PR,
 'Cells after QC':d['Cells after QC'],'Tumour cells':d['Tumour cells'],'Endocrine':d.Endocrine,'Stem cells':d['Stem cells'],'Pituicytes':d.Pituicytes,'Stromal':d.Stromal,'Immune':d.Immune})
g=d.groupby('StudyC')
T=pd.DataFrame({'Study':list(g.groups),'DOI':[re.sub(r'^(https?://)?(dx\.)?doi\.org/','',g.get_group(s).DOI.iloc[0]) for s in g.groups],
 'Repository':[', '.join(sorted(g.get_group(s).Repo.unique())) for s in g.groups],
 'Platform':['; '.join(sorted(g.get_group(s).Plat.unique())) for s in g.groups],
 'Healthy samples':[int((g.get_group(s).Condition=='Healthy').sum()) for s in g.groups],
 'Tumour samples':[int((g.get_group(s).Condition=='Tumour').sum()) for s in g.groups],
 'Cells after QC':[int(g.get_group(s)['Cells after QC'].sum()) for s in g.groups]})
for c in ['Tumour cells','Endocrine','Stem cells','Pituicytes','Stromal','Immune']: T[c]=[int(np.nansum(g.get_group(s)[c])) for s in g.groups]
tot=pd.DataFrame([{'Study':'Total','DOI':np.nan,'Repository':np.nan,'Platform':np.nan,'Healthy samples':T['Healthy samples'].sum(),'Tumour samples':T['Tumour samples'].sum(),'Cells after QC':T['Cells after QC'].sum()}])
print(T.to_string(index=False))

F=lambda **k: Font(**{"name":"Arial","size":10,**k}); thin=Side(style="thin",color="000000")
def write(ws,title,df,notes,widths,bold_last=False,totals=()):
    ws["A1"]=title; ws["A1"].font=F(bold=True,size=11)
    for j,h in enumerate(df.columns,1):
        c=ws.cell(3,j,h); c.font=F(bold=True); c.border=Border(top=thin,bottom=thin); c.alignment=Alignment(wrap_text=True,vertical="bottom")
    for i,row in enumerate(df.itertuples(index=False),4):
        for j,v in enumerate(row,1):
            if isinstance(v,float) and np.isnan(v): v='–'
            elif isinstance(v,(float,np.floating)) and float(v).is_integer(): v=int(v)
            elif isinstance(v,np.integer): v=int(v)
            c=ws.cell(i,j,v); c.font=F(bold=bold_last and i==3+len(df))
            if isinstance(v,(int,float)): c.number_format="#,##0"
            c.alignment=Alignment(horizontal="right" if isinstance(v,(int,float)) else "left",vertical="top")
    last=3+len(df); first=4; ncol=len(df.columns)
    for j in range(1,ncol+1): ws.cell(last,j).border=Border(bottom=thin)
    from openpyxl.utils import get_column_letter as L
    for k,(label,labcol,kind,cols) in enumerate(totals):
        r=last+1+k; ws.cell(r,1,label).font=F(bold=True)
        for j in range(1,ncol+1): ws.cell(r,j).border=Border(top=thin if k==0 else None,bottom=thin if k==len(totals)-1 else None)
        for name in cols:
            j=list(df.columns).index(name)+1; col=L(j); rng=f"{col}{first}:{col}{last}"
            f=f"=SUM({rng})" if kind is None else f'=SUMIF(${L(labcol)}${first}:${L(labcol)}${last},"{kind}",{rng})'
            c=ws.cell(r,j,f); c.font=F(bold=True); c.number_format="#,##0"; c.alignment=Alignment(horizontal="right")
    last+=len(totals)
    for k,n in enumerate(notes):
        r=last+2+k; ws.cell(r,1,n).font=F(size=9); ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=min(len(df.columns),12))
        ws.cell(r,1).alignment=Alignment(wrap_text=True,vertical="top"); ws.row_dimensions[r].height=28
    for j,w in enumerate(widths,1): ws.column_dimensions[ws.cell(3,j).column_letter].width=w
    ws.row_dimensions[3].height=30; ws.freeze_panes="B4"; ws.sheet_view.showGridLines=False
wb=Workbook()
write(wb.active,"Supplementary Table 1a Single-cell and single-nucleus RNA sequencing datasets included in the atlas",T,
 ["Cell numbers are after quality control; tumour cells are gonadotroph-lineage cells in tumour samples and are included in the endocrine count. GEO, Gene Expression Omnibus; SRA, Sequence Read Archive; GSA-Human, Genome Sequence Archive for Human (National Genomics Data Center, China)."],
 [22,30,16,44,10,10,12,10,10,10,10,10,10],
 totals=[('Total',1,None,['Healthy samples','Tumour samples']+['Cells after QC', 'Tumour cells', 'Endocrine', 'Stem cells', 'Pituicytes', 'Stromal', 'Immune'])]); wb.active.title="Studies"
write(wb.create_sheet("Samples"),"Supplementary Table 1b Sample-level metadata and cell numbers",S,
 ["Accession: GEO sample (GSM), SRA experiment (SRX) or GSA-Human run (HRS) identifier. Tumour type, Knosp grade, Ki-67, invasion and primary/recurrent status are as reported by the original studies. Multiplexed samples are pools of several donors, so age and sex are not given.",
  "Tumour cells are gonadotroph-lineage cells in tumour samples. Endocrine: gonadotrophs (including tumour cells), somatotrophs, lactotrophs, corticotrophs and thyrotrophs; stromal: endothelial and mesenchymal cells; immune: all myeloid and lymphoid cells.",
  "–, not applicable or not available."],
 [13,42,18,11,30,10,9,38,17,8,6,8,8,12,10,9,9,9,8,9,8,8],
 totals=[('Healthy samples',7,'Healthy',['Cells after QC', 'Endocrine', 'Stem cells', 'Pituicytes', 'Stromal', 'Immune']),('Tumour samples',7,'Tumour',['Cells after QC', 'Tumour cells', 'Endocrine', 'Stem cells', 'Pituicytes', 'Stromal', 'Immune']),('All samples',7,None,['Cells after QC', 'Tumour cells', 'Endocrine', 'Stem cells', 'Pituicytes', 'Stromal', 'Immune'])])
wb.save("Supplementary_Table_1.xlsx"); print("saved Supplementary_Table_1.xlsx")


####################################################################################################
# SECTION 6  Fig. 2A-E
####################################################################################################
# ==== Fig. 2 — standalone, assembled A–E (needs only `adata`) ===================================
# A  change in median per-sample senescence scores, tumour − healthy, for every cell type
# B  stem-cell UMAPs: condition, SenMayo, SASP (Reactome)
# C  per-sample stem-cell scores and % SenMayo-high cells
# D  CDKN1A-positive stem cells: UMAP and % per sample
# E  SenMayo and SASP scores in CDKN1A-negative vs CDKN1A-positive stem cells of the same sample (CDKN1A removed)
# If needed first:  !pip install -q gseapy lxml
# install missing packages automatically (gseapy needs lxml to fetch MSigDB)
import importlib.util, subprocess, sys as _sys
_need = [m for m in ("lxml", "gseapy", "pydeseq2") if importlib.util.find_spec(m) is None]
if _need:
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q"] + _need, check=True)
import numpy as np, pandas as pd, scipy.sparse as sp, re
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
CT_COL, MIN_CELLS, MIN_SAMPLES = "cell_type", 20, 3
HIGH_PCT, ZOOM_PCT, SCORE_LO_PCT, SEED = 95, 0.5, 80, 0
GENE, P21_PCT, MIN_GROUP = "CDKN1A", 90, 5   # CDKN1A-positive = above this percentile of all healthy stem cells, pooled
FIG_H_MM = 208           # figure height
GAP_AB   = 0.12          # space between A (heatmap labels) and B UMAPs
GAP_BC   = 0.22          # space between UMAPs and box plots
GAP_CD   = 0.35          # space between box plots and D/E
HIGH_COL, LOW_COL = "#B2182B", "#d9d9d9"
STEM_CT = "Stem_cells"
CT_GROUPS = {"Endocrine":    ["Gonadotrophs", "Somatotrophs", "Lactotrophs", "Corticotrophs", "Thyrotrophs"],
             "Stem / glial": ["Stem_cells", "Pituicytes"],
             "Stromal":      ["Endothelial_cells", "Mesenchymal_cells"],
             "Myeloid":      ["Macrophages", "Monocytes", "Dendritic_cells", "pDC", "Mast_cells", "Neutrophils"],
             "Lymphoid":     ["CD4_T_cells", "CD4_T_regs", "CD8_T_cells", "NK_cells", "ILCs", "B_cells", "Plasma_cells"]}
CT_ORDER = [c for v in CT_GROUPS.values() for c in v]

MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
rng = np.random.default_rng(SEED)
COND = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
CMAP_UMAP = LinearSegmentedColormap.from_list(
    "grey_ylorrd", ["#e9e9e9"] + [plt.get_cmap("YlOrRd")(t) for t in np.linspace(0.25, 1, 6)])
plain  = lambda x: np.asarray([str(v) for v in x], dtype=object)
pretty = lambda s: s.replace("_", " ").replace("Gonadotrophs", "Gonadotrophs / tumour cells")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.show()

def fmt_p(p):
    if not np.isfinite(p) or p >= 0.05: return "n.s."
    if p >= 0.001: return f"p = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"p = {m}$\\times$10$^{{{int(e)}}}$"

# set F2_DONE = False to force recomputation; otherwise reruns only redraw the figure
if not globals().get("F2_DONE", False):
    # ---------------------------------------------------------------- gene sets, per-sample totals
    from gseapy import Msigdb
    c2 = Msigdb().get_gmt(category="c2.all", dbver="2023.2.Hs")
    SETS = {"SenMayo":            c2["SAUL_SEN_MAYO"],
            "Fridman senescence": c2["FRIDMAN_SENESCENCE_UP"],
            "SASP (Reactome)":    c2["REACTOME_SENESCENCE_ASSOCIATED_SECRETORY_PHENOTYPE_SASP"]}
    obs = adata.obs
    cond = (plain(obs["condition"]) if "condition" in obs else
            np.where(obs["Cell type"].astype(str).str.lower() == "normal", "Healthy", "Tumour").astype(object))
    ct, sid = plain(obs[CT_COL]), plain(obs["SRA_ID"])
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    samples = np.unique(sid)
    S = sp.csr_matrix((np.ones(len(sid)), (pd.Categorical(sid, categories=samples).codes, np.arange(len(sid)))),
                      shape=(len(samples), len(sid)))
    tot = S @ counts
    totals = pd.DataFrame(np.asarray(tot.todense() if sp.issparse(tot) else tot), index=samples, columns=adata.var_names)
    umap = adata.obsm["X_umap"]

    def score(sets_genes, prefix):
        cols = {}
        for k, gl in sets_genes.items():
            col = prefix + re.sub(r"\W+", "_", k).strip("_")
            sc.tl.score_genes(adata, gl, score_name=col, use_raw=False, random_state=SEED)
            cols[k] = col
        return cols

    # ---------------------------------------------------------------- A: landscape across all cell types
    meas_all = (totals > 0).all(axis=0)
    COLS_ALL = score({k: [g for g in dict.fromkeys(v) if g in adata.var_names and meas_all.get(g, False)]
                      for k, v in SETS.items()}, "lsc_")
    df = pd.DataFrame({c: adata.obs[c].to_numpy() for c in COLS_ALL.values()})
    df["ct"], df["sid"], df["cond"] = ct, sid, cond
    g = df.groupby(["ct", "sid"])
    ps = g[list(COLS_ALL.values())].mean(); ps["n"] = g.size(); ps["cond"] = g["cond"].first()
    ps = ps[ps.n >= MIN_CELLS].reset_index()
    TYPES = [c for c in CT_ORDER if c in set(ps.ct)] + sorted(set(ps.ct) - set(CT_ORDER))
    rows = []
    for c in TYPES:
        d = ps[ps.ct == c]; h, t = d[d.cond == "Healthy"], d[d.cond == "Tumour"]
        for k, col in COLS_ALL.items():
            p = mannwhitneyu(t[col], h[col], alternative="two-sided").pvalue if min(len(h), len(t)) >= MIN_SAMPLES else np.nan
            rows.append(dict(cell_type=c, signature=k, n_healthy=len(h), n_tumour=len(t),
                             median_healthy=h[col].median() if len(h) else np.nan,
                             median_tumour=t[col].median() if len(t) else np.nan, p=p))
    LAND = pd.DataFrame(rows)
    LAND.loc[LAND.p.notna(), "FDR"] = multipletests(LAND.loc[LAND.p.notna(), "p"], method="fdr_bh")[1]
    LAND["delta"] = LAND.median_tumour - LAND.median_healthy
    LAND.to_csv("Fig2A_senescence_by_celltype.csv", index=False)

    # ---------------------------------------------------------------- B–F: stem cells
    n_stem = pd.Series(sid[ct == STEM_CT]).value_counts()
    keep = n_stem.index[n_stem >= MIN_CELLS]
    m_stem = (ct == STEM_CT) & np.isin(sid, keep)
    meas = (totals.loc[keep] > 0).all(axis=0)
    COLS_STEM = score({k: [x for x in dict.fromkeys(v) if x in adata.var_names and meas.get(x, False)]
                       for k, v in SETS.items()}, "stem_")
    idx = np.flatnonzero(m_stem)
    cell = pd.DataFrame({k: adata.obs[c].to_numpy()[idx] for k, c in COLS_STEM.items()})
    cell["sid"], cell["cond"] = sid[idx], cond[idx]
    thr = np.percentile(cell.loc[cell.cond == "Healthy", "SenMayo"], HIGH_PCT)
    cell["high"] = cell["SenMayo"] > thr
    SAMP = cell.groupby("sid").agg({**{k: "mean" for k in SETS}, "high": "mean", "cond": "first"})
    SAMP["pct_high"] = SAMP.pop("high") * 100
    STATS = {}
    for c in list(SETS) + ["pct_high"]:
        h, t = SAMP.loc[SAMP.cond == "Healthy", c], SAMP.loc[SAMP.cond == "Tumour", c]
        STATS[c] = dict(median_healthy=h.median(), median_tumour=t.median(), n_healthy=len(h), n_tumour=len(t),
                        p=mannwhitneyu(t, h, alternative="two-sided").pvalue)
    STATS = pd.DataFrame(STATS).T
    SAMP.to_csv("Fig2_Stem_cells_per_sample.csv"); STATS.to_csv("Fig2_Stem_cells_stats.csv")
    N_CELLS = {c_: int((cond[m_stem] == c_).sum()) for c_ in ["Healthy", "Tumour"]}


    # ---------------------------------------------------------------- D–E: CDKN1A-positive stem cells
    expr = adata.X[idx][:, adata.var_names.get_loc(GENE)]
    expr = np.asarray(expr.todense() if sp.issparse(expr) else expr).ravel()
    healthy_c = cond[idx] == "Healthy"
    thr_p21 = max(np.percentile(expr[healthy_c], P21_PCT), 0.0)
    high = expr > thr_p21
    print(f"CDKN1A-positive threshold (> {P21_PCT}th pct of healthy stem cells): {thr_p21:.3f}; "
          f"CDKN1A-positive: {high[healthy_c].mean()*100:.1f}% of healthy, {high[~healthy_c].mean()*100:.1f}% of tumour stem cells")
    COLS_P21 = score({k: [x for x in dict.fromkeys(v) if x in adata.var_names and meas.get(x, False) and x != GENE]
                      for k, v in SETS.items()}, "p21_")                       # CDKN1A removed
    pc = pd.DataFrame({k: adata.obs[c].to_numpy()[idx] for k, c in COLS_P21.items()})
    pc["sid"], pc["cond"], pc["high"] = sid[idx], cond[idx], high
    frac = pc.groupby("sid").agg(pct_high=("high", "mean"), cond=("cond", "first"))
    frac["pct_high"] *= 100
    grp = pc.groupby(["sid", "high"])
    pair_n = grp.size().unstack(fill_value=0)
    ok = pair_n.index[(pair_n.get(True, 0) >= MIN_GROUP) & (pair_n.get(False, 0) >= MIN_GROUP)]
    means = grp[list(SETS)].mean()
    PAIRED = {}
    for k in SETS:
        lo_, hi_ = means.xs(False, level="high").loc[ok, k], means.xs(True, level="high").loc[ok, k]
        PAIRED[k] = dict(low=lo_, high=hi_, p=wilcoxon(hi_, lo_).pvalue)
        print(f"  {k:20s} CDKN1A-positive vs -negative, {len(ok)} samples: paired Wilcoxon p = {PAIRED[k]['p']:.2g}")
    F2_DONE = True

# ---------------------------------------------------------------- drawing helpers
def zoom(mask, pct=ZOOM_PCT):
    lo, hi = np.percentile(umap[mask], pct, axis=0), np.percentile(umap[mask], 100 - pct, axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

def clean(ax, lims):
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)

def cbar(fig, ax, norm, cmap, label="Score", rect=(1.0, 0.18, 0.035, 0.64)):
    cax = ax.inset_axes(list(rect))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
    cb.ax.set_title(label, fontsize=5, pad=3)

def draw_box(ax, col, title, ylabel=None):
    st = STATS.loc[col]
    data = [SAMP.loc[SAMP.cond == c_, col].dropna() for c_ in ["Healthy", "Tumour"]]
    for i, (v, c_) in enumerate(zip(data, ["Healthy", "Tumour"])):
        k = COND[c_]
        ax.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=to_rgba(k, 0.18), ec=k, lw=0.8), medianprops=dict(c=k, lw=1.3),
                   whiskerprops=dict(c=k, lw=0.8), capprops=dict(c=k, lw=0.8))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, c=k, ec="white", lw=0.3, zorder=3)
    allv = pd.concat(data); top, span = allv.max(), allv.max() - allv.min()
    ax.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
    ax.text(0.5, top + .12 * span, fmt_p(st.p), ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(allv.min() - .06 * span, top + .27 * span)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Healthy\n(n = {int(st.n_healthy)})", f"Tumour\n(n = {int(st.n_tumour)})"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=6.3, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=5.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

# ================================================================ assembled figure
idx_stem = idx

def paired_panel(ax, k, ylabel=None):
    lo, hi, p_ = PAIRED[k]["low"], PAIRED[k]["high"], PAIRED[k]["p"]
    for s in lo.index:
        c_ = COND[frac.loc[s, "cond"]]
        ax.plot([0, 1], [lo[s], hi[s]], c=c_, lw=0.6, alpha=0.7, zorder=2)
        ax.scatter([0, 1], [lo[s], hi[s]], s=6, c=c_, ec="white", lw=0.3, zorder=3)
    allv = pd.concat([lo, hi]); top, span = allv.max(), allv.max() - allv.min()
    ax.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
    ax.text(0.5, top + .12 * span, fmt_p(p_), ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(allv.min() - .06 * span, top + .27 * span); ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"], fontsize=5.2)
    ax.set_xlabel(r"$\it{CDKN1A}$ transcript", fontsize=5.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=0)
    ax.set_title(k, fontsize=6.3, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=5.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

fig = plt.figure(figsize=(180 * MM_, FIG_H_MM * MM_))
outer = fig.add_gridspec(7, 1, height_ratios=[1.0, GAP_AB, 1.0, GAP_BC, 0.95, GAP_CD, 1.0], hspace=0,
                         left=0.1, right=0.96, top=0.972, bottom=0.072)
ROW_A, ROW_U, ROW_BOX, ROW_DE = outer[0], outer[2], outer[4], outer[6]

# ---------------- A: Δ heatmap -------------------------------------------------------------------
gA = ROW_A.subgridspec(2, 2, height_ratios=[1, 0.5], width_ratios=[1, 0.03], hspace=0, wspace=0.03)
axA = fig.add_subplot(gA[0, 0])
sigs = list(SETS)
D = np.full((len(sigs), len(TYPES)), np.nan); F = np.full_like(D, np.nan); present = np.zeros_like(D, bool)
for yi, s in enumerate(sigs):
    for xi, c in enumerate(TYPES):
        r = LAND[(LAND.cell_type == c) & (LAND.signature == s)].iloc[0]
        D[yi, xi], present[yi, xi], F[yi, xi] = r.delta, np.isfinite(r.delta), r.get("FDR", np.nan)
tested = np.isfinite(F)
stars = lambda f: "***" if f < 0.001 else "**" if f < 0.01 else "*" if f < 0.05 else ""
lim = np.nanmax(np.abs(D[tested]))
dnorm = TwoSlopeNorm(0, -lim, lim)
dcmap = plt.get_cmap("RdBu_r").copy(); dcmap.set_bad("#f2f2f2")
axA.imshow(np.ma.masked_where(~tested, D), aspect="auto", cmap=dcmap, norm=dnorm, interpolation="nearest")
for yi in range(len(sigs)):
    for xi in range(len(TYPES)):
        if tested[yi, xi]:
            v = D[yi, xi]; tc = "white" if abs(dnorm(v) - 0.5) > 0.3 else "#222222"
            axA.text(xi, yi - 0.1, f"{v:+.2f}", ha="center", va="center", fontsize=5.2, color=tc)
            axA.text(xi, yi + 0.27, stars(F[yi, xi]), ha="center", va="center", fontsize=6, color=tc)
        elif present[yi, xi]:
            axA.text(xi, yi, "n.t.", ha="center", va="center", fontsize=4.8, color="#9a9a9a")
        else:
            axA.text(xi, yi, "–", ha="center", va="center", fontsize=6.5, color="#bbbbbb")
pos = 0
for grp, members in CT_GROUPS.items():
    n_in = sum(c in TYPES for c in members)
    if not n_in: continue
    if pos: axA.axvline(pos - 0.5, c="k", lw=0.6)
    axA.text(pos + (n_in - 1) / 2, -0.62, grp, ha="center", va="bottom", fontsize=5.8, fontweight="bold")
    pos += n_in
axA.set_xticks(range(len(TYPES)))
axA.set_xticklabels([pretty(c) for c in TYPES], rotation=40, ha="right", fontsize=5.4)
axA.set_yticks(range(len(sigs))); axA.set_yticklabels(sigs, fontsize=5.8); axA.tick_params(length=0)
for s_ in axA.spines.values(): s_.set_linewidth(0.4)
caxA = fig.add_subplot(gA[0, 1])
cb = fig.colorbar(ScalarMappable(norm=dnorm, cmap=dcmap), cax=caxA)
cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label("Δ median score\n(tumour − healthy)", fontsize=5.2)
keyA = fig.add_subplot(gA[1, 1]); keyA.axis("off")
keyA.text(-1.2, 0.95, "* FDR < 0.05\n** FDR < 0.01\n*** FDR < 0.001\nn.t., not tested\n–, absent in tumours",
          transform=keyA.transAxes, fontsize=4.8, va="top", ha="left", color="#444444", linespacing=1.35)
fig.add_subplot(gA[1, 0]).axis("off")

# ---------------- B and C (top): stem-cell UMAPs -------------------------------------------------
gU = ROW_U.subgridspec(1, 3, wspace=0.28)
lims = zoom(m_stem); pt = max(0.25, min(4, 30000 / m_stem.sum()))
axB = fig.add_subplot(gU[0])
for c_ in sorted(["Healthy", "Tumour"], key=lambda c_: -N_CELLS[c_]):
    i = idx_stem[cond[idx_stem] == c_]
    axB.scatter(umap[i, 0], umap[i, 1], s=pt, c=COND[c_], linewidths=0, rasterized=True)
clean(axB, lims); axB.set_anchor("N"); axB.set_title("Condition", fontsize=6.5, pad=2)
axB.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, mfc=COND[c_], mec="none",
                           label=f"{c_}\n({N_CELLS[c_]:,} cells)") for c_ in ["Healthy", "Tumour"]],
           loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=5, frameon=False, handletextpad=0.2,
           borderaxespad=0, labelspacing=0.5)
for j, (k, title) in enumerate([("SenMayo", "SenMayo score"), ("SASP (Reactome)", "SASP (Reactome) score")]):
    ax = fig.add_subplot(gU[j + 1])
    v = adata.obs[COLS_STEM[k]].to_numpy()
    norm = Normalize(np.percentile(v[m_stem & (cond == "Healthy")], SCORE_LO_PCT), np.percentile(v[m_stem], 99), clip=True)
    i = idx_stem[np.argsort(v[idx_stem], kind="stable")]
    ax.scatter(umap[i, 0], umap[i, 1], c=CMAP_UMAP(norm(v[i])), s=pt, linewidths=0, rasterized=True)
    clean(ax, lims); ax.set_anchor("N"); ax.set_title(title, fontsize=6.5, pad=2); cbar(fig, ax, norm, CMAP_UMAP)

# ---------------- C (bottom): per-sample scores --------------------------------------------------
gB = ROW_BOX.subgridspec(1, 4, wspace=0.65)
for j, k in enumerate(SETS):
    draw_box(fig.add_subplot(gB[j]), k, k, ylabel="Mean score per sample" if j == 0 else None)
draw_box(fig.add_subplot(gB[3]), "pct_high", f"SenMayo-high cells\n(> {HIGH_PCT}th pct. of healthy)",
         ylabel="% cells per sample")

# ---------------- D: CDKN1A-positive UMAP + fraction;  E: paired SenMayo and SASP -----------------------
gD = ROW_DE.subgridspec(1, 5, width_ratios=[1.35, 0.75, 0.12, 0.72, 0.72], wspace=0.6)
axD = fig.add_subplot(gD[0])
axD.scatter(umap[idx_stem[~high], 0], umap[idx_stem[~high], 1], s=pt, c=LOW_COL, linewidths=0, rasterized=True)
axD.scatter(umap[idx_stem[high], 0], umap[idx_stem[high], 1], s=pt * 1.4, c=HIGH_COL, linewidths=0, rasterized=True)
clean(axD, lims); axD.set_anchor("N"); axD.set_title("$\\it{CDKN1A}$-positive stem cells", fontsize=6.5, pad=2)
axD.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, mfc=HIGH_COL, mec="none", label=f"$\\it{{CDKN1A}}$-positive\n({high.sum():,} cells)"),
                    Line2D([], [], marker="o", ls="", ms=3.2, mfc=LOW_COL, mec="none", label=f"$\\it{{CDKN1A}}$-negative\n({(~high).sum():,} cells)")],
           loc="lower right", bbox_to_anchor=(1.08, 0.0), fontsize=5, frameon=False, handletextpad=0.2,
           borderaxespad=0, labelspacing=0.5)

axD2 = fig.add_subplot(gD[1])
fh, ft = frac.loc[frac.cond == "Healthy", "pct_high"], frac.loc[frac.cond == "Tumour", "pct_high"]
for i, (v, c_) in enumerate([(fh, "Healthy"), (ft, "Tumour")]):
    k = COND[c_]
    axD2.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                 boxprops=dict(fc=to_rgba(k, 0.18), ec=k, lw=0.8), medianprops=dict(c=k, lw=1.3),
                 whiskerprops=dict(c=k, lw=0.8), capprops=dict(c=k, lw=0.8))
    axD2.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, c=k, ec="white", lw=0.3, zorder=3)
top, span = max(fh.max(), ft.max()), max(fh.max(), ft.max()) - min(fh.min(), ft.min())
axD2.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
axD2.text(0.5, top + .12 * span, fmt_p(mannwhitneyu(ft, fh, alternative="two-sided").pvalue),
          ha="center", va="bottom", fontsize=5.3)
axD2.set_ylim(min(fh.min(), ft.min()) - .06 * span, top + .27 * span)
axD2.set_xticks([0, 1]); axD2.set_xticklabels([f"Healthy\n(n = {len(fh)})", f"Tumour\n(n = {len(ft)})"], fontsize=5)
axD2.set_ylabel("% cells per sample", fontsize=5.5)
axD2.tick_params(axis="y", labelsize=5, length=2); axD2.tick_params(axis="x", length=0)
axD2.set_title("$\\it{CDKN1A}$-positive cells", fontsize=6.3, pad=4)
for s_ in ("top", "right"): axD2.spines[s_].set_visible(False)

axE1 = fig.add_subplot(gD[3]); paired_panel(axE1, "SenMayo", ylabel="Mean score per sample")
axE2 = fig.add_subplot(gD[4]); paired_panel(axE2, "SASP (Reactome)")
fig.canvas.draw()
b1, b2 = axE1.get_position(), axE2.get_position()
fig.legend(handles=[Line2D([], [], c=COND[c_], lw=0.8, marker="o", ms=2.5, label=f"{c_} sample")
                    for c_ in ["Healthy", "Tumour"]],
           loc="upper center", bbox_to_anchor=((b1.x0 + b2.x1) / 2, b1.y0 - 0.042), ncol=2,
           fontsize=5, frameon=False, handlelength=1.4, columnspacing=1.5)

save(fig, "Fig2_assembled")
print("\nStem-cell statistics:"); print(STATS[["median_healthy", "median_tumour", "n_healthy", "n_tumour", "p"]].round(4).to_string())
print(f"CDKN1A-positive stem cells per sample: healthy median {fh.median():.1f}% (n = {len(fh)}), "
      f"tumour median {ft.median():.1f}% (n = {len(ft)}); paired comparison used {len(PAIRED['SenMayo']['low'])} samples")


####################################################################################################
# SECTION 7  Supplementary Fig. 3A-F
####################################################################################################
# ==== Supplementary Fig. 3 (A–F) — standalone, needs only `adata` ===================================
# A gene-set overlap (Venn) | B cell types + SenMayo/SASP UMAPs (all cells) | C score heatmaps by cell type
# D tumour cells vs healthy gonadotrophs (UMAPs) | E per-sample scores | F leave-one-gene-out (stem cells)
# Reuses the Fig. 2 / Supp. Fig. 3 code unchanged; no panel letters (added by hand).
# !pip install -q gseapy statsmodels lxml   (installed at the top of this script)
import os, re, time
import numpy as np, pandas as pd, scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

LOGO_CSV  = "SuppFig_LOGO_stem_cells.csv"     # reused if present in OUT_DIR (from the leave-one-gene-out run)

if "adata" not in globals():
    from google.colab import drive; drive.mount("/content/drive")
    adata = sc.read(DATA_PATH)
os.makedirs(OUT_DIR, exist_ok=True); os.chdir(OUT_DIR)
if "condition" not in adata.obs:
    adata.obs["condition"] = np.where(adata.obs["Cell type"].astype(str).str.lower() == "normal", "Healthy", "Tumour")
COND_PAL = {"Healthy": "#4393C3", "Tumour": "#8B1A1A"}                       # as in Figs 1–2
_cats = list(adata.obs["cell_type"].astype("category").cat.categories)
CT_PAL = dict(zip(_cats, adata.uns["cell_type_colors"])) if "cell_type_colors" in adata.uns else {}

# ------------------------------------------------------------------ scores and statistics (Fig. 2 code)
import numpy as np, pandas as pd, scipy.sparse as sp, re
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
CT_COL, MIN_CELLS, MIN_SAMPLES = "cell_type", 20, 3
HIGH_PCT, ZOOM_PCT, SCORE_LO_PCT, SEED = 95, 0.5, 80, 0
STEM_CT = "Stem_cells"
CT_GROUPS = {"Endocrine":    ["Gonadotrophs", "Somatotrophs", "Lactotrophs", "Corticotrophs", "Thyrotrophs"],
             "Stem / glial": ["Stem_cells", "Pituicytes"],
             "Stromal":      ["Endothelial_cells", "Mesenchymal_cells"],
             "Myeloid":      ["Macrophages", "Monocytes", "Dendritic_cells", "pDC", "Mast_cells", "Neutrophils"],
             "Lymphoid":     ["CD4_T_cells", "CD4_T_regs", "CD8_T_cells", "NK_cells", "ILCs", "B_cells", "Plasma_cells"]}
CT_ORDER = [c for v in CT_GROUPS.values() for c in v]

MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
rng = np.random.default_rng(SEED)
COND = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
CMAP_UMAP = LinearSegmentedColormap.from_list(
    "grey_ylorrd", ["#e9e9e9"] + [plt.get_cmap("YlOrRd")(t) for t in np.linspace(0.25, 1, 6)])
plain  = lambda x: np.asarray([str(v) for v in x], dtype=object)
pretty = lambda s: s.replace("_", " ").replace("Gonadotrophs", "Gonadotrophs / tumour cells")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.show()

def fmt_p(p):
    if not np.isfinite(p) or p >= 0.05: return "n.s."
    if p >= 0.001: return f"p = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"p = {m}$\\times$10$^{{{int(e)}}}$"

# ---------------------------------------------------------------- gene sets, per-sample totals
from gseapy import Msigdb
c2 = Msigdb().get_gmt(category="c2.all", dbver="2023.2.Hs")
SETS = {"SenMayo":            c2["SAUL_SEN_MAYO"],
        "Fridman senescence": c2["FRIDMAN_SENESCENCE_UP"],
        "SASP (Reactome)":    c2["REACTOME_SENESCENCE_ASSOCIATED_SECRETORY_PHENOTYPE_SASP"]}
obs = adata.obs
ct, cond, sid = plain(obs[CT_COL]), plain(obs["condition"]), plain(obs["SRA_ID"])
counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
samples = np.unique(sid)
S = sp.csr_matrix((np.ones(len(sid)), (pd.Categorical(sid, categories=samples).codes, np.arange(len(sid)))),
                  shape=(len(samples), len(sid)))
tot = S @ counts
totals = pd.DataFrame(np.asarray(tot.todense() if sp.issparse(tot) else tot), index=samples, columns=adata.var_names)
umap = adata.obsm["X_umap"]

def score(sets_genes, prefix):
    cols = {}
    for k, gl in sets_genes.items():
        col = prefix + re.sub(r"\W+", "_", k).strip("_")
        sc.tl.score_genes(adata, gl, score_name=col, use_raw=False, random_state=SEED)
        cols[k] = col
    return cols

# ---------------------------------------------------------------- A: landscape across all cell types
meas_all = (totals > 0).all(axis=0)
COLS_ALL = score({k: [g for g in dict.fromkeys(v) if g in adata.var_names and meas_all.get(g, False)]
                  for k, v in SETS.items()}, "lsc_")
df = pd.DataFrame({c: adata.obs[c].to_numpy() for c in COLS_ALL.values()})
df["ct"], df["sid"], df["cond"] = ct, sid, cond
g = df.groupby(["ct", "sid"])
ps = g[list(COLS_ALL.values())].mean(); ps["n"] = g.size(); ps["cond"] = g["cond"].first()
ps = ps[ps.n >= MIN_CELLS].reset_index()
TYPES = [c for c in CT_ORDER if c in set(ps.ct)] + sorted(set(ps.ct) - set(CT_ORDER))
rows = []
for c in TYPES:
    d = ps[ps.ct == c]; h, t = d[d.cond == "Healthy"], d[d.cond == "Tumour"]
    for k, col in COLS_ALL.items():
        p = mannwhitneyu(t[col], h[col], alternative="two-sided").pvalue if min(len(h), len(t)) >= MIN_SAMPLES else np.nan
        rows.append(dict(cell_type=c, signature=k, n_healthy=len(h), n_tumour=len(t),
                         median_healthy=h[col].median() if len(h) else np.nan,
                         median_tumour=t[col].median() if len(t) else np.nan, p=p))
LAND = pd.DataFrame(rows)
LAND.loc[LAND.p.notna(), "FDR"] = multipletests(LAND.loc[LAND.p.notna(), "p"], method="fdr_bh")[1]
LAND["delta"] = LAND.median_tumour - LAND.median_healthy
LAND.to_csv("Fig2A_senescence_by_celltype.csv", index=False)

# ---------------------------------------------------------------- B–F: stem cells
n_stem = pd.Series(sid[ct == STEM_CT]).value_counts()
keep = n_stem.index[n_stem >= MIN_CELLS]
m_stem = (ct == STEM_CT) & np.isin(sid, keep)
meas = (totals.loc[keep] > 0).all(axis=0)
COLS_STEM = score({k: [x for x in dict.fromkeys(v) if x in adata.var_names and meas.get(x, False)]
                   for k, v in SETS.items()}, "stem_")
idx = np.flatnonzero(m_stem)
cell = pd.DataFrame({k: adata.obs[c].to_numpy()[idx] for k, c in COLS_STEM.items()})
cell["sid"], cell["cond"] = sid[idx], cond[idx]
thr = np.percentile(cell.loc[cell.cond == "Healthy", "SenMayo"], HIGH_PCT)
cell["high"] = cell["SenMayo"] > thr
SAMP = cell.groupby("sid").agg({**{k: "mean" for k in SETS}, "high": "mean", "cond": "first"})
SAMP["pct_high"] = SAMP.pop("high") * 100
STATS = {}
for c in list(SETS) + ["pct_high"]:
    h, t = SAMP.loc[SAMP.cond == "Healthy", c], SAMP.loc[SAMP.cond == "Tumour", c]
    STATS[c] = dict(median_healthy=h.median(), median_tumour=t.median(), n_healthy=len(h), n_tumour=len(t),
                    p=mannwhitneyu(t, h, alternative="two-sided").pvalue)
STATS = pd.DataFrame(STATS).T
SAMP.to_csv("Fig2_Stem_cells_per_sample.csv"); STATS.to_csv("Fig2_Stem_cells_stats.csv")
N_CELLS = {c_: int((cond[m_stem] == c_).sum()) for c_ in ["Healthy", "Tumour"]}

# ---------------------------------------------------------------- drawing helpers
def zoom(mask, pct=ZOOM_PCT):
    lo, hi = np.percentile(umap[mask], pct, axis=0), np.percentile(umap[mask], 100 - pct, axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

def clean(ax, lims):
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)

def cbar(fig, ax, norm, cmap, label="Score", rect=(1.0, 0.18, 0.035, 0.64)):
    cax = ax.inset_axes(list(rect))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
    cb.ax.tick_params(labelsize=4.8, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
    cb.ax.set_title(label, fontsize=5, pad=3)

def draw_box(ax, col, title, ylabel=None):
    st = STATS.loc[col]
    data = [SAMP.loc[SAMP.cond == c_, col].dropna() for c_ in ["Healthy", "Tumour"]]
    for i, (v, c_) in enumerate(zip(data, ["Healthy", "Tumour"])):
        k = COND[c_]
        ax.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=to_rgba(k, 0.18), ec=k, lw=0.8), medianprops=dict(c=k, lw=1.3),
                   whiskerprops=dict(c=k, lw=0.8), capprops=dict(c=k, lw=0.8))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=7, c=k, ec="white", lw=0.3, zorder=3)
    allv = pd.concat(data); top, span = allv.max(), allv.max() - allv.min()
    ax.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
    ax.text(0.5, top + .12 * span, fmt_p(st.p), ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(allv.min() - .06 * span, top + .27 * span)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Healthy\n(n = {int(st.n_healthy)})", f"Tumour\n(n = {int(st.n_tumour)})"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=6.3, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=5.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)


# ------------------------------------------------------------------ helpers and tumour-cell analysis (Supp. Fig. 3 code)
MIN_CELLS_S = 20
GONAD_CT = "Gonadotrophs"
ID_GENES = ["SOX2", "SOX9", "NR5A1", "CGA", "LHB", "FSHB"]
LABEL_MIN_CELLS = 1500          # label cell types with at least this many cells on the cell-type UMAP
LABEL_FS = 4.6                  # label font size
HIGH_COL = "#B2182B"
CMAP_HEAT = plt.get_cmap("YlOrRd").copy(); CMAP_HEAT.set_bad("#f0f0f0")
rng_s = np.random.default_rng(2)
umap = adata.obsm["X_umap"]
dense = lambda m: m.toarray() if sp.issparse(m) else np.asarray(m)

all_types = [c for c in TYPES] + sorted(set(ct) - set(TYPES))
_fallback = plt.get_cmap("tab20")
CT_COLS = {c: _fallback(i % 20) for i, c in enumerate(all_types)}
CT_COLS.update({k: v for k, v in globals().get("CT_PAL", {}).items()})

_STUDY_RAW = {"Asuzu et al., (2022)": "#1f77b4", "Huang et al., (2025)": "#ff7f0e", "Ilie et al. (2025)": "#2ca02c",
              "Potthoff et al., (2024)": "#d62728", "Yan et al., (2024)": "#9467bd", "Zhang et al., (2022c)": "#8c564b",
              "Zhang et al. (2026)": "#e377c2", "Zhang et al., (2022b)": "#7f7f7f"}
_STUDY_RAW.update(globals().get("STUDY_PAL", {}))
_norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
study_col = lambda a: {_norm(k): v for k, v in _STUDY_RAW.items()}.get(_norm(a), "#bbbbbb")
study_txt = lambda a: re.sub(r"[(),]", "", str(a))
author_of = pd.Series(plain(adata.obs["Author"]), index=sid).groupby(level=0).first()
stars = lambda f: "" if not np.isfinite(f) else ("***" if f < 0.001 else "**" if f < 0.01 else "*" if f < 0.05 else "")

def box_generic(ax, samp, st, col, title, ylabel=None):
    data = [samp.loc[samp.cond == c_, col].dropna() for c_ in ["Healthy", "Tumour"]]
    for i, (v, c_) in enumerate(zip(data, ["Healthy", "Tumour"])):
        k = COND[c_]
        ax.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                   boxprops=dict(fc=to_rgba(k, 0.18), ec=k, lw=0.8), medianprops=dict(c=k, lw=1.3),
                   whiskerprops=dict(c=k, lw=0.8), capprops=dict(c=k, lw=0.8))
        ax.scatter(i + rng_s.uniform(-0.15, 0.15, len(v)), v, s=7, c=k, ec="white", lw=0.3, zorder=3)
    allv = pd.concat(data); top, span = allv.max(), allv.max() - allv.min()
    ax.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
    ax.text(0.5, top + .12 * span, fmt_p(st.loc[col, "p"]), ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(allv.min() - .06 * span, top + .27 * span)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Healthy\n(n = {int(st.loc[col, 'n_healthy'])})", f"Tumour\n(n = {int(st.loc[col, 'n_tumour'])})"], fontsize=5)
    ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=6.3, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=5.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

def place_labels(ax, items, fs=LABEL_FS):
    """items: list of (text, x_data, y_data). Places labels in axes coords, pushing overlaps apart."""
    fig_ = ax.figure; fig_.canvas.draw()
    bb = ax.get_window_extent()
    w_ax, h_ax = bb.width, bb.height                                   # pixels
    px_per_pt = fig_.dpi / 72
    xl, yl = ax.get_xlim(), ax.get_ylim()
    texts = [t for t, _, _ in items]
    pos = np.array([[(x - xl[0]) / (xl[1] - xl[0]), (y - yl[0]) / (yl[1] - yl[0])] for _, x, y in items], float)
    orig = pos.copy()
    half_w = np.array([0.5 * 0.55 * fs * px_per_pt * max(len(l) for l in t.split("\n")) / w_ax for t in texts])
    half_h = np.array([0.5 * 1.25 * fs * px_per_pt * (t.count("\n") + 1) / h_ax for t in texts])
    for _ in range(300):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx, dy = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1]
                ox, oy = half_w[i] + half_w[j] - abs(dx), half_h[i] + half_h[j] - abs(dy)
                if ox > 0 and oy > 0:
                    step = oy / 2 + 0.002; sgn = 1 if dy >= 0 else -1
                    pos[j, 1] += sgn * step; pos[i, 1] -= sgn * step; moved = True
        pos[:, 0] = np.clip(pos[:, 0], half_w + 0.01, 1 - half_w - 0.01)
        pos[:, 1] = np.clip(pos[:, 1], half_h + 0.01, 1 - half_h - 0.01)
        if not moved: break
    for (t, _, _), p, o in zip(items, pos, orig):
        if np.hypot(*(p - o)) > 0.03:
            ax.plot([o[0], p[0]], [o[1], p[1]], transform=ax.transAxes, c="#555555", lw=0.35, zorder=4)
        tx = ax.text(p[0], p[1], t, transform=ax.transAxes, fontsize=fs, ha="center", va="center",
                     color="#111111", fontweight="bold", zorder=5, linespacing=0.9)
        tx.set_path_effects([pe.withStroke(linewidth=1.5, foreground="white")])

# ---------------- tumour cells vs healthy gonadotrophs (same rules as Fig. 2) --------------------
g_n = pd.Series(sid[ct == GONAD_CT]).value_counts()
g_keep = g_n.index[g_n >= MIN_CELLS_S]
g_m = (ct == GONAD_CT) & np.isin(sid, g_keep)
g_meas = (totals.loc[g_keep] > 0).all(axis=0)
G_COLS = score({k: [x for x in dict.fromkeys(v) if x in adata.var_names and g_meas.get(x, False)]
                for k, v in SETS.items()}, "s3g_")
g_idx = np.flatnonzero(g_m)
gcell = pd.DataFrame({k: adata.obs[c].to_numpy()[g_idx] for k, c in G_COLS.items()})
gcell["sid"], gcell["cond"] = sid[g_idx], cond[g_idx]
g_thr = np.percentile(gcell.loc[gcell.cond == "Healthy", "SenMayo"], HIGH_PCT)
gcell["high"] = gcell["SenMayo"] > g_thr
G_SAMP = gcell.groupby("sid").agg({**{k: "mean" for k in SETS}, "high": "mean", "cond": "first"})
G_SAMP["pct_high"] = G_SAMP.pop("high") * 100
G_STATS = {}
for c in list(SETS) + ["pct_high"]:
    h, t = G_SAMP.loc[G_SAMP.cond == "Healthy", c], G_SAMP.loc[G_SAMP.cond == "Tumour", c]
    G_STATS[c] = dict(median_healthy=h.median(), median_tumour=t.median(), n_healthy=len(h), n_tumour=len(t),
                      p=mannwhitneyu(t, h, alternative="two-sided").pvalue)
G_STATS = pd.DataFrame(G_STATS).T
G_N = {c_: int((cond[g_m] == c_).sum()) for c_ in ["Healthy", "Tumour"]}
print("Tumour cells vs healthy gonadotrophs:", G_N); print(G_STATS.round(4).to_string())


# ------------------------------------------------------------------ A: gene-set overlap among genes used for scoring
VENN_SETS = {k: set(g for g in v if g in adata.var_names and meas_all.get(g, False)) for k, v in SETS.items()}
VENN_COLS = ["#E6550D", "#3182BD", "#31A354"]

def draw_venn(ax):
    """Three-set Venn (equal circles) of genes detected in all samples; counts per region."""
    from matplotlib.patches import Circle
    (na, A), (nb, B), (nc, C) = list(VENN_SETS.items())
    r, ctr = 0.40, [(-0.22, 0.13), (0.22, 0.13), (0.0, -0.25)]
    for (x, y), col in zip(ctr, VENN_COLS):
        ax.add_patch(Circle((x, y), r, fc=to_rgba(col, 0.28), ec=col, lw=0.8))
    regions = {(-0.40, 0.24): len(A - B - C), (0.40, 0.24): len(B - A - C), (0.0, -0.45): len(C - A - B),
               (0.0, 0.30): len((A & B) - C), (-0.21, -0.10): len((A & C) - B), (0.21, -0.10): len((B & C) - A),
               (0.0, 0.03): len(A & B & C)}
    for (x, y), n in regions.items():
        ax.text(x, y, str(n), ha="center", va="center", fontsize=5.5 if abs(y) > 0.2 or abs(x) > 0.3 else 5)
    lab = lambda k, n: f"{k}\n({n} genes)"
    ax.text(-0.42, 0.60, lab("SenMayo", len(A)), ha="center", va="bottom", fontsize=5.3, color=VENN_COLS[0], linespacing=0.95)
    ax.text(0.42, 0.60, lab("Fridman\nsenescence", len(B)), ha="center", va="bottom", fontsize=5.3, color=VENN_COLS[1], linespacing=0.95)
    ax.text(0.0, -0.69, lab("SASP (Reactome)", len(C)), ha="center", va="top", fontsize=5.3, color=VENN_COLS[2], linespacing=0.95)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.95, 0.95); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Genes used for scoring", fontsize=6.5, pad=2)
    print(f"Venn: {len(A)}, {len(B)}, {len(C)} genes; union {len(A | B | C)}; shared by ≥2 sets "
          f"{len((A & B) | (A & C) | (B & C))}; all three: {sorted(map(str, A & B & C))}")

# ------------------------------------------------------------------ F: leave-one-gene-out (stem cells, tumour vs healthy)
LOGO_GENES = {k: [x for x in dict.fromkeys(v) if x in adata.var_names and meas.get(x, False)] for k, v in SETS.items()}
s_idx = np.flatnonzero(m_stem)

def stem_test(genes):
    sc.tl.score_genes(adata, genes, score_name="_logo", use_raw=False, random_state=SEED)
    v = adata.obs["_logo"].to_numpy()[s_idx]
    per = pd.DataFrame({"v": v, "sid": sid[s_idx], "cond": cond[s_idx]}).groupby("sid").agg(v=("v", "mean"), cond=("cond", "first"))
    t, h = per.loc[per.cond == "Tumour", "v"], per.loc[per.cond == "Healthy", "v"]
    return mannwhitneyu(t, h, alternative="two-sided").pvalue

if os.path.exists(LOGO_CSV):
    LOGO = pd.read_csv(LOGO_CSV); print(f"Leave-one-gene-out: loaded {LOGO_CSV} ({len(LOGO)} rows)")
else:
    print("Leave-one-gene-out: computing (~15 min) ..."); rows, t0 = [], time.time()
    for k, genes in LOGO_GENES.items():
        for g in genes:
            rows.append(dict(signature=k, dropped=g, p=stem_test([x for x in genes if x != g])))
        print(f"  {k}: {len(genes)} genes ({time.time() - t0:.0f} s)")
    adata.obs.drop(columns="_logo", inplace=True)
    LOGO = pd.DataFrame(rows); LOGO.to_csv(LOGO_CSV, index=False)
FULL_P = {k: float(STATS.loc[k, "p"]) for k in SETS}           # full-set stem-cell p-values (Fig. 2c)

def draw_logo(spec):
    g = spec.subgridspec(1, 4, wspace=0.65)                       # same 4-column grid as row E
    ymax = max(np.ceil(max((-np.log10(LOGO.p)).max(), max(-np.log10(p) for p in FULL_P.values())) * 2) / 2 + 0.25,
               -np.log10(0.05) + 0.4)   # always show the p = 0.05 line
    DOT, FULL_C, THR_C = "#3a3a3a", "#B2182B", "0.55"
    for j, k in enumerate(SETS):
        ax = fig.add_subplot(g[j])
        d = LOGO[LOGO.signature == k].sort_values("p").reset_index(drop=True)
        y = -np.log10(d.p.to_numpy()); x = np.arange(1, len(d) + 1)
        ax.axhline(-np.log10(0.05), color=THR_C, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.axhline(-np.log10(FULL_P[k]), color=FULL_C, lw=0.8, zorder=2)
        ax.scatter(x, y, s=4, color=DOT, lw=0, zorder=3)
        pv = d.p.to_numpy(); i = len(d) - 1                       # largest p = most influential gene
        tied = d.dropped[np.isclose(pv, pv.max(), rtol=1e-9, atol=0)].tolist()
        lab = d.dropped[i] if len(tied) == 1 else f"{len(tied)} genes, tied"
        xt = x[np.isclose(pv, pv.max(), rtol=1e-9, atol=0)].mean() if len(tied) > 1 else x[i]
        room = y[i] - (-np.log10(0.05))                            # space above the p = 0.05 line
        if room > 0.45:                                            # enough room: label below the point(s)
            ax.annotate(lab, (xt, y[i]), xytext=(0, -5), textcoords="offset points",
                        ha="center" if len(tied) > 1 else "right", va="top",
                        fontsize=5, fontstyle="italic" if len(tied) == 1 else "normal", color="#222222")
        else:                                                      # close to the line: label to the left
            ax.annotate(lab, (x[i], y[i]), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=5, fontstyle="italic" if len(tied) == 1 else "normal", color="#222222")
        print(f"  {k}: most influential = {', '.join(tied)} (p = {pv.max():.3g})")
        ax.set_xlim(0, len(d) + 1); ax.set_ylim(0, ymax)
        ax.set_xticks([1, len(d)]); ax.set_xticklabels(["1", str(len(d))], fontsize=5)
        ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=2)
        ax.set_xlabel("Gene removed (ranked by p)", fontsize=5.5, labelpad=1.5)
        ax.set_title(f"{k} ({len(d)} genes)", fontsize=6.3, pad=4)
        if j == 0: ax.set_ylabel("−log$_{10}$ p, tumour vs\nhealthy stem cells", fontsize=5.5)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axk = fig.add_subplot(g[3]); axk.axis("off")
    axk.legend(handles=[Line2D([], [], color=FULL_C, lw=0.8, label="All genes"),
                        Line2D([], [], color=THR_C, lw=0.6, ls=(0, (3, 2)), label="p = 0.05"),
                        Line2D([], [], marker="o", ls="", ms=2.2, mfc=DOT, mec="none", label="One gene removed")],
               loc="center left", bbox_to_anchor=(-0.1, 0.55), fontsize=5.3, frameon=False,
               handlelength=1.8, handletextpad=0.5, labelspacing=0.6)
    print("Leave-one-gene-out p ranges:",
          {k: f"{LOGO[LOGO.signature == k].p.min():.2g}–{LOGO[LOGO.signature == k].p.max():.2g}" for k in SETS})

# =====================================================================================================
# SUPPLEMENTARY FIG. 3 (A–F)
# =====================================================================================================
S3_H_MM = 285
GAP = 0.24
fig = plt.figure(figsize=(180 * MM_, S3_H_MM * MM_))
outer = fig.add_gridspec(9, 1, height_ratios=[0.9, GAP, 1.75, 0.40, 0.95, GAP, 0.95, 0.3, 0.8], hspace=0,
                         left=0.06, right=0.97, top=0.978, bottom=0.04)
# ---- A: gene-set overlap (Venn);  B: cell types and full-atlas SenMayo / SASP UMAPs
gTop = outer[0].subgridspec(1, 4, width_ratios=[0.8, 1, 1, 1], wspace=0.28)
draw_venn(fig.add_subplot(gTop[0]))
all_mask = np.ones(len(umap), bool); lims_all = zoom(all_mask, 0.2)
pt_all = max(0.03, min(1, 30000 / len(umap)))

axA = fig.add_subplot(gTop[1])
order = rng_s.permutation(len(umap))
axA.scatter(umap[order, 0], umap[order, 1], s=pt_all, c=[CT_COLS.get(c, "#bbbbbb") for c in ct[order]],
            linewidths=0, rasterized=True)
clean(axA, lims_all); axA.set_title("Cell types (all cells)", fontsize=6.5, pad=2)
counts_ct = pd.Series(ct).value_counts()
items = []
for c in counts_ct.index[counts_ct >= LABEL_MIN_CELLS]:
    x0, y0 = np.median(umap[ct == c], axis=0)
    items.append((pretty(c).replace("Gonadotrophs / tumour cells", "Gonadotrophs /\ntumour cells"), x0, y0))

for j, k in enumerate(["SenMayo", "SASP (Reactome)"]):
    ax = fig.add_subplot(gTop[j + 2])
    v = adata.obs[COLS_ALL[k]].to_numpy()
    norm = Normalize(np.percentile(v[cond == "Healthy"], SCORE_LO_PCT), np.percentile(v, 99), clip=True)
    o = np.argsort(v, kind="stable")
    ax.scatter(umap[o, 0], umap[o, 1], c=CMAP_UMAP(norm(v[o])), s=pt_all, linewidths=0, rasterized=True)
    clean(ax, lims_all); ax.set_title(f"{k} score (all cells)", fontsize=6.5, pad=2)
    cbar(fig, ax, norm, CMAP_UMAP, rect=(0.98, 0.2, 0.035, 0.6))

# ---- C: absolute-score heatmaps
gC = outer[2].subgridspec(1, 5, width_ratios=[0.42, 1, 1, 1, 0.48], wspace=0.12)
heat = []
for j, k in enumerate(SETS):
    ax = fig.add_subplot(gC[j + 1]); heat.append(ax)
    r = LAND[LAND.signature == k].set_index("cell_type").loc[TYPES]
    M = r[["median_healthy", "median_tumour"]].to_numpy(dtype=float)
    vals = M[np.isfinite(M)]
    norm = Normalize(np.percentile(vals, 2), np.percentile(vals, 98), clip=True)
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=CMAP_HEAT, norm=norm, interpolation="nearest")
    for i, c in enumerate(TYPES):
        for jj in (0, 1):
            if np.isfinite(M[i, jj]):
                f = r.loc[c, "FDR"] if "FDR" in r else np.nan
                ax.text(jj, i, f"{M[i, jj]:.2f}" + (stars(f) if jj == 1 else ""), ha="center", va="center",
                        fontsize=4.4, color="white" if norm(M[i, jj]) > 0.6 else "#222222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "Tumour"], fontsize=5.3)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=1); ax.set_title(k, fontsize=6.3, pad=12)
    ax.set_yticks(range(len(TYPES))); ax.set_yticklabels([pretty(c) for c in TYPES] if j == 0 else [], fontsize=5.3)
    ax.tick_params(axis="y", length=0)
    for sp_ in ax.spines.values(): sp_.set_linewidth(0.4)
    pos = 0
    for grp, members in CT_GROUPS.items():
        n_in = sum(c in TYPES for c in members)
        if not n_in: continue
        if pos: ax.axhline(pos - 0.5, c="k", lw=0.5)
        pos += n_in
    cax = ax.inset_axes([0.15, -0.06, 0.7, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP_HEAT), cax=cax, orientation="horizontal", extend="both")
    cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
    cb.set_label("Median of per-sample mean score", fontsize=4.8)
axn = fig.add_subplot(gC[4]); axn.set_ylim(len(TYPES) - 0.5, -0.5); axn.axis("off")
nn = LAND[LAND.signature == "SenMayo"].set_index("cell_type").loc[TYPES]
axn.text(0.08, -1.3, "Samples\n(healthy / tumour)", fontsize=5.3, va="bottom")
for i, c in enumerate(TYPES):
    axn.text(0.08, i, f"{int(nn.loc[c, 'n_healthy'])} / {int(nn.loc[c, 'n_tumour'])}", fontsize=5, va="center")

# ---- D: tumour-cell UMAPs
gDF = outer[4].subgridspec(1, 3, wspace=0.28)
g_lims = zoom(g_m); g_pt = max(0.25, min(4, 30000 / g_m.sum()))
ax = fig.add_subplot(gDF[0])
for c_ in sorted(["Healthy", "Tumour"], key=lambda c_: -G_N[c_]):
    i = g_idx[cond[g_idx] == c_]
    ax.scatter(umap[i, 0], umap[i, 1], s=g_pt, c=COND[c_], linewidths=0, rasterized=True)
clean(ax, g_lims); ax.set_title("Condition", fontsize=6.5, pad=2)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, mfc=COND[c_], mec="none",
                          label=f"{'Healthy gonadotrophs' if c_ == 'Healthy' else 'Tumour cells'}\n({G_N[c_]:,} cells)")
                   for c_ in ["Healthy", "Tumour"]],
          loc="lower left", bbox_to_anchor=(-0.02, 0.0), fontsize=5, frameon=False, handletextpad=0.2,
          borderaxespad=0, labelspacing=0.5)
for j, (k, title) in enumerate([("SenMayo", "SenMayo score"), ("SASP (Reactome)", "SASP (Reactome) score")]):
    ax = fig.add_subplot(gDF[j + 1])
    v = adata.obs[G_COLS[k]].to_numpy()
    norm = Normalize(np.percentile(v[g_m & (cond == "Healthy")], SCORE_LO_PCT), np.percentile(v[g_m], 99), clip=True)
    i = g_idx[np.argsort(v[g_idx], kind="stable")]
    ax.scatter(umap[i, 0], umap[i, 1], c=CMAP_UMAP(norm(v[i])), s=g_pt, linewidths=0, rasterized=True)
    clean(ax, g_lims); ax.set_title(title, fontsize=6.5, pad=2); cbar(fig, ax, norm, CMAP_UMAP)

# ---- E: tumour-cell per-sample scores
gGH = outer[6].subgridspec(1, 4, wspace=0.65)
for j, k in enumerate(SETS):
    box_generic(fig.add_subplot(gGH[j]), G_SAMP, G_STATS, k, k, ylabel="Mean score per sample" if j == 0 else None)
box_generic(fig.add_subplot(gGH[3]), G_SAMP, G_STATS, "pct_high",
            f"SenMayo-high cells\n(> {HIGH_PCT}th pct. of healthy)", ylabel="% cells per sample")

# ---- cell-type labels for B (after layout is final) and group labels for C
place_labels(axA, items)
fig.canvas.draw()
bb = heat[0].get_position(); pos = 0
for grp, members in CT_GROUPS.items():
    n_in = sum(c in TYPES for c in members)
    if not n_in: continue
    y_mid = bb.y1 - (pos + n_in / 2) / len(TYPES) * (bb.y1 - bb.y0)
    fig.text(0.02, y_mid, grp.replace(" / ", "/\n"), rotation=90, ha="center", va="center",
             fontsize=5 if n_in <= 2 else 5.5, fontweight="bold", linespacing=0.9)
    pos += n_in

# ---- F: leave-one-gene-out
draw_logo(outer[8])
save(fig, "SuppFig3_assembled")


####################################################################################################
# SECTION 8  Supplementary Fig. 4A-C
####################################################################################################
# ==== Supplementary Fig. 4 — standalone (needs only `adata`) =====================================
# A  Fridman senescence score in CDKN1A-negative vs CDKN1A-positive stem cells, paired within sample (CDKN1A removed)
# B  Pre-ranked GSEA of senescence-associated gene sets (paired pseudobulk DESeq2, CDKN1A-positive vs CDKN1A-negative)
# C  Top 15 MSigDB Hallmark pathways from the same ranking
# If needed first:  !pip install -q gseapy pydeseq2 lxml
# install missing packages automatically (gseapy needs lxml to fetch MSigDB)
import importlib.util, subprocess, sys as _sys
_need = [m for m in ("lxml", "gseapy", "pydeseq2") if importlib.util.find_spec(m) is None]
if _need:
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q"] + _need, check=True)
import re
import numpy as np, pandas as pd, scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
import gseapy as gp
from gseapy import Msigdb
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ---------------------------------------------------------------- settings
FOCUS_CT   = "Stem_cells"
CT_COL     = "cell_type"
SAMPLE_COL = "SRA_ID"
GENE       = "CDKN1A"
MIN_CELLS  = 20        # min stem cells per sample (as in Fig. 2)
P21_PCT    = 90        # CDKN1A-positive = CDKN1A above this percentile of all healthy stem cells, pooled
MIN_GROUP  = 5         # min CDKN1A-positive and CDKN1A-negative cells per sample for the paired analyses
PB_MIN_COUNTS, PB_MIN_PROFILES = 10, 10   # pseudobulk gene filter: >= 10 counts in >= 10 profiles
SEED, GSEA_SEED, N_PERM = 0, 0, 1000
TOP_N   = 15           # Hallmark pathways shown (highest |NES|)
FIG_H   = 122          # figure height (mm)
ROW_GAP = 0.36         # space between the two rows (C title + B axis label)
ROW1_WIDTHS = [0.55, 1.05, 1.0]   # A, space for legend + B's pathway labels, B
ROW2_WIDTHS = [0.42, 1.0]         # space for C's pathway labels, C
UP_COL, DOWN_COL, FDR_CUT = "#B2182B", "#2166AC", 0.05
COND = globals().get("COND_PAL", {"Healthy": "#4C8BC0", "Tumour": "#8B1A1A"})
MM_ = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
plain = lambda x: np.asarray([str(v) for v in x], dtype=object)
dense = lambda m: m.toarray() if sp.issparse(m) else np.asarray(m)

def fmt_p(p):
    if not np.isfinite(p) or p >= 0.05: return "n.s."
    if p >= 0.001: return f"p = {p:.3f}"
    m_, e = f"{p:.1e}".split("e")
    return f"p = {m_}$\\times$10$^{{{int(e)}}}$"

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.show()

# set RECOMPUTE = False to redraw the figure only, reusing results from a previous run of this cell
RECOMPUTE = not globals().get("S4_DONE", False)
if RECOMPUTE:
    # ---------------------------------------------------------------- stem cells and p21 status
    obs = adata.obs
    if "condition" in obs:
        cond = plain(obs["condition"])
    else:
        cond = np.where(obs["Cell type"].astype(str).str.lower() == "normal", "Healthy", "Tumour").astype(object)
    ct, sid = plain(obs[CT_COL]), plain(obs[SAMPLE_COL])
    n_focus = pd.Series(sid[ct == FOCUS_CT]).value_counts()
    keep = n_focus.index[n_focus >= MIN_CELLS]
    idx = np.flatnonzero((ct == FOCUS_CT) & np.isin(sid, keep))
    expr = dense(adata.X[idx][:, adata.var_names.get_loc(GENE)]).ravel()
    healthy_c = cond[idx] == "Healthy"
    thr = max(np.percentile(expr[healthy_c], P21_PCT), 0.0)
    high = expr > thr
    print(f"{len(keep)} samples with >= {MIN_CELLS} stem cells; CDKN1A-positive threshold {thr:.3f}; "
          f"CDKN1A-positive: {high[healthy_c].mean()*100:.1f}% of healthy, {high[~healthy_c].mean()*100:.1f}% of tumour stem cells")

    # ---------------------------------------------------------------- Fridman score, CDKN1A removed (panel A)
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    samples = np.unique(sid)
    S_mat = sp.csr_matrix((np.ones(len(sid)), (pd.Categorical(sid, categories=samples).codes, np.arange(len(sid)))),
                          shape=(len(samples), len(sid)))
    totals = pd.DataFrame(dense(S_mat @ counts), index=samples, columns=adata.var_names)
    meas = (totals.loc[keep] > 0).all(axis=0)                     # detected in every included sample
    c2 = Msigdb().get_gmt(category="c2.all", dbver="2023.2.Hs")
    hallmark = Msigdb().get_gmt(category="h.all", dbver="2023.2.Hs")
    fr_genes = [g for g in dict.fromkeys(c2["FRIDMAN_SENESCENCE_UP"])
                if g in adata.var_names and meas.get(g, False) and g != GENE]
    sc.tl.score_genes(adata, fr_genes, score_name="s4_fridman", use_raw=False, random_state=SEED)
    print(f"Fridman senescence: {len(fr_genes)} genes scored (CDKN1A removed)")

    cell = pd.DataFrame({"sid": sid[idx], "cond": cond[idx], "high": high,
                         "Fridman senescence": adata.obs["s4_fridman"].to_numpy()[idx]})
    sample_cond = cell.groupby("sid")["cond"].first()

    # ---------------------------------------------------------------- within-sample paired comparison
    grp = cell.groupby(["sid", "high"])
    pair_n = grp.size().unstack(fill_value=0)
    ok = pair_n.index[(pair_n.get(True, 0) >= MIN_GROUP) & (pair_n.get(False, 0) >= MIN_GROUP)]
    means = grp["Fridman senescence"].mean()
    lo_A, hi_A = means.xs(False, level="high").loc[ok], means.xs(True, level="high").loc[ok]
    p_A = wilcoxon(hi_A, lo_A).pvalue
    print(f"Paired comparison: {len(ok)} samples with >= {MIN_GROUP} CDKN1A-positive and CDKN1A-negative cells; "
          f"Fridman higher in CDKN1A-positive in {(hi_A > lo_A).sum()}/{len(ok)}; paired Wilcoxon p = {p_A:.2g}")

    # ---------------------------------------------------------------- paired pseudobulk + DESeq2
    pb_hi, pb_lo = [], []
    for s in ok:
        sel = cell.sid.values == s
        pb_hi.append(np.asarray(counts[idx[sel & high]].sum(axis=0)).ravel())
        pb_lo.append(np.asarray(counts[idx[sel & ~high]].sum(axis=0)).ravel())
    pb = np.vstack(pb_hi + pb_lo)
    keep_g = ((pb >= PB_MIN_COUNTS).sum(axis=0) >= PB_MIN_PROFILES) & (np.asarray(adata.var_names) != GENE)
    genes = np.asarray(adata.var_names[keep_g])
    sample_ids = [f"s{i + 1}" for i in range(len(ok))]
    rows = [f"{s}_high" for s in sample_ids] + [f"{s}_low" for s in sample_ids]
    counts_pb = pd.DataFrame(np.rint(pb[:, keep_g]).astype(int), index=rows, columns=genes)
    meta_pb = pd.DataFrame({"sample": sample_ids * 2, "group": ["high"] * len(ok) + ["low"] * len(ok)}, index=rows)
    meta_pb["group"] = pd.Categorical(meta_pb["group"], categories=["low", "high"])
    try:                                                                # PyDESeq2 >= 0.5
        dds = DeseqDataSet(counts=counts_pb, metadata=meta_pb, design="~ sample + group", quiet=True)
    except TypeError:                                                   # older PyDESeq2
        meta_pb = meta_pb.astype(str)
        dds = DeseqDataSet(counts=counts_pb, metadata=meta_pb, design_factors=["sample", "group"],
                           ref_level=["group", "low"], quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["group", "high", "low"], quiet=True)
    ds.summary()
    DESEQ = ds.results_df.sort_values("pvalue")
    DESEQ.to_csv("SuppFig4_pseudobulk_DESeq2_p21high_vs_low.csv")
    print(f"DESeq2: {len(ok)} paired samples, {len(genes):,} genes (CDKN1A excluded); "
          f"{((DESEQ.padj < 0.05) & (DESEQ.log2FoldChange > 0)).sum()} up, "
          f"{((DESEQ.padj < 0.05) & (DESEQ.log2FoldChange < 0)).sum()} down in CDKN1A-positive (FDR < 0.05)")

    # ---------------------------------------------------------------- pre-ranked GSEA on the Wald statistic
    rank = DESEQ["stat"].dropna()
    rank = rank[~rank.index.duplicated()].sort_values(ascending=False)
    SEN_SETS = {"SAUL_SEN_MAYO": "SenMayo",
                "FRIDMAN_SENESCENCE_UP": "Fridman senescence (up)",
                "REACTOME_SENESCENCE_ASSOCIATED_SECRETORY_PHENOTYPE_SASP": "SASP (Reactome)",
                "REACTOME_CELLULAR_SENESCENCE": "Cellular senescence (Reactome)",
                "REACTOME_OXIDATIVE_STRESS_INDUCED_SENESCENCE": "Oxidative stress-induced senescence",
                "REACTOME_ONCOGENE_INDUCED_SENESCENCE": "Oncogene-induced senescence",
                "REACTOME_DNA_DAMAGE_TELOMERE_STRESS_INDUCED_SENESCENCE": "DNA damage/telomere-induced senescence"}
    HALLMARK_LABELS = {"TNFA_SIGNALING_VIA_NFKB": "TNF-α signalling via NF-κB", "INTERFERON_GAMMA_RESPONSE": "Interferon-γ response",
                       "INTERFERON_ALPHA_RESPONSE": "Interferon-α response", "IL6_JAK_STAT3_SIGNALING": "IL-6/JAK/STAT3 signalling",
                       "IL2_STAT5_SIGNALING": "IL-2/STAT5 signalling", "TGF_BETA_SIGNALING": "TGF-β signalling",
                       "MTORC1_SIGNALING": "mTORC1 signalling", "P53_PATHWAY": "p53 pathway",
                       "PI3K_AKT_MTOR_SIGNALING": "PI3K/AKT/mTOR signalling", "WNT_BETA_CATENIN_SIGNALING": "WNT/β-catenin signalling",
                       "UV_RESPONSE_UP": "UV response up", "UV_RESPONSE_DN": "UV response down", "DNA_REPAIR": "DNA repair",
                       "E2F_TARGETS": "E2F targets", "G2M_CHECKPOINT": "G2M checkpoint", "MYC_TARGETS_V1": "MYC targets v1",
                       "MYC_TARGETS_V2": "MYC targets v2", "KRAS_SIGNALING_UP": "KRAS signalling up",
                       "KRAS_SIGNALING_DN": "KRAS signalling down", "PROTEIN_SECRETION": "Protein secretion",
                       "EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial-mesenchymal transition"}

    def run_gsea(sets, labels, min_size):
        pre = gp.prerank(rnk=rank, gene_sets=sets, min_size=min_size, max_size=500, permutation_num=N_PERM,
                         seed=GSEA_SEED, threads=4, outdir=None, verbose=False)
        R = pre.res2d.copy()
        fdr_col = next(c for c in R.columns if "FDR" in c)
        R["NES"], R["FDR"] = R["NES"].astype(float), R[fdr_col].astype(float)
        R["Pathway"] = R["Term"].map(labels)
        return R[["Term", "Pathway", "NES", "FDR"]].sort_values("NES", ascending=False).reset_index(drop=True)

    S = run_gsea({k: c2[k] for k in SEN_SETS}, SEN_SETS, min_size=10)
    hl = lambda t: HALLMARK_LABELS.get(t.replace("HALLMARK_", ""), t.replace("HALLMARK_", "").replace("_", " ").capitalize()).replace("signaling", "signalling")
    H = run_gsea(hallmark, {t: hl(t) for t in hallmark}, min_size=15)
    S.to_csv("SuppFig4_GSEA_senescence_sets.csv", index=False); H.to_csv("SuppFig4_GSEA_hallmark.csv", index=False)
    print("\nSenescence gene sets:"); print(S[["Pathway", "NES", "FDR"]].round(3).to_string(index=False))
    print(f"\nHallmark: {((H.FDR < FDR_CUT) & (H.NES > 0)).sum()} enriched and {((H.FDR < FDR_CUT) & (H.NES < 0)).sum()} "
          f"depleted in CDKN1A-positive (FDR < 0.05) of {len(H)} tested")

    S4_DONE = True

# British spelling and hyphenated EMT, applied here so a redraw-only rerun picks them up
H["Pathway"] = (H["Pathway"].str.replace("signaling", "signalling", regex=False)
                .str.replace("Epithelial mesenchymal transition", "Epithelial-mesenchymal transition", regex=False))

# ================================================================ figure
def paired_panel(ax, lo, hi, p_, title, ylabel=None):
    for s in lo.index:
        c_ = COND[sample_cond[s]]
        ax.plot([0, 1], [lo[s], hi[s]], c=c_, lw=0.6, alpha=0.7, zorder=2)
        ax.scatter([0, 1], [lo[s], hi[s]], s=6, c=c_, ec="white", lw=0.3, zorder=3)
    allv = pd.concat([lo, hi]); top, span = allv.max(), allv.max() - allv.min()
    ax.plot([0, 0, 1, 1], [top + .07 * span, top + .10 * span, top + .10 * span, top + .07 * span], c="k", lw=0.5)
    ax.text(0.5, top + .12 * span, fmt_p(p_), ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(allv.min() - .06 * span, top + .27 * span); ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"], fontsize=5.2)
    ax.set_xlabel(r"$\it{CDKN1A}$ transcript", fontsize=5.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=5, length=2); ax.tick_params(axis="x", length=0)
    ax.set_title(title, fontsize=6.3, pad=4)
    if ylabel: ax.set_ylabel(ylabel, fontsize=5.5)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)

def gsea_bars(ax, df, title, label_fs=5.1):
    y = np.arange(len(df))
    cols = [(UP_COL if n > 0 else DOWN_COL) if f < FDR_CUT else "#bdbdbd" for n, f in zip(df.NES, df.FDR)]
    ax.barh(y, df.NES, color=cols, height=0.7)
    for yi, (n, f) in enumerate(zip(df.NES, df.FDR)):
        ax.text(n + (0.06 if n > 0 else -0.06), yi, "FDR < 0.001" if f < 0.001 else f"FDR = {f:.3f}",
                va="center", ha="left" if n > 0 else "right", fontsize=4.4, color="#333333")
    lo = min(0, df.NES.min()) - 0.15; hi = max(0, df.NES.max()) + 0.95
    ax.set_xlim(lo, hi); ax.axvline(0, c="k", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df.Pathway, fontsize=label_fs)
    ax.set_xlabel("Normalised enrichment score\n($\\it{CDKN1A}$-positive vs -negative stem cells)", fontsize=5.5)
    ax.tick_params(axis="x", labelsize=5, length=2); ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=6.3, pad=4)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)

S_plot = S.sort_values("NES")
H_top = H.reindex(H.NES.abs().sort_values(ascending=False).index).head(TOP_N).sort_values("NES")

fig = plt.figure(figsize=(180 * MM_, FIG_H * MM_))
outer = fig.add_gridspec(3, 1, height_ratios=[1.0, ROW_GAP, 1.25], hspace=0,
                         left=0.07, right=0.99, top=0.965, bottom=0.085)

# row 1: A (paired Fridman) and B (senescence gene sets)
g1 = outer[0].subgridspec(1, 3, width_ratios=ROW1_WIDTHS, wspace=0.05)
axA = fig.add_subplot(g1[0])
paired_panel(axA, lo_A, hi_A, p_A, "Fridman senescence", ylabel="Mean score per sample")
axB = fig.add_subplot(g1[2]); gsea_bars(axB, S_plot, "Senescence gene sets")

# row 2: C (top Hallmark pathways)
g2 = outer[2].subgridspec(1, 2, width_ratios=ROW2_WIDTHS, wspace=0.05)
axC = fig.add_subplot(g2[1])
gsea_bars(axC, H_top, f"Hallmark pathways (top {TOP_N} by enrichment)")

fig.canvas.draw()
bA = axA.get_position()
fig.legend(handles=[Line2D([], [], c=COND[c_], lw=0.8, marker="o", ms=2.5, label=f"{c_} sample")
                    for c_ in ["Healthy", "Tumour"]],
           loc="upper left", bbox_to_anchor=(bA.x1 + 0.015, bA.y1), ncol=1,
           fontsize=5, frameon=False, handlelength=1.4, borderaxespad=0)
save(fig, "SuppFig4_assembled")
print(H_top[["Pathway", "NES", "FDR"]].sort_values("NES", ascending=False).round(3).to_string(index=False))


####################################################################################################
# SECTION 9  Tissue cohort: Figs. 3A, 3C, 4A, 4C, 4D; Supplementary Figs. 6, 7
####################################################################################################
import os
os.chdir(OUT_DIR)
# ==== Tissue cohort figures (Fig. 3A, 3C, 4A, 4C, 4D; Supp. Fig. 6; Supp. Fig. 7) ====================
# Aggressive = repeat surgery for regrowth or recurrence; non-aggressive = no repeat surgery (>= 2 years' follow-up).
# Input: Supplementary_Table_3.xlsx (one row per tumour). Output: PDF + PNG for each panel. No panel letters.
import sys, numpy as np, pandas as pd
from math import comb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, norm, spearmanr

TABLE = SUPP_TABLE3
OUT = OUT_DIR
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# ---------------------------------------------------------------- palette (sampled from the original figures)
PAL = {
    "Centre": {"UCLH": "#D9D2E9", "UoC": "#5E4FA2"},
    "Sex": {"M": "#2E4E4E", "F": "#FF4040"},
    "Age": {"≤30": "#440154", "31-50": "#31688E", "51-70": "#35B679", "70+": "#FDE725"},
    "Knosp": {"0": "#F0F921", "1": "#FDB42E", "2": "#ED7952", "3": "#F2B8C6", "3A": "#CC4678", "3B": "#9C169E", "4": "#3B0F70"},
    "Ki-67": {"<3%": "#9DC9E0", "3-5%": "#4191C5", "5-10%": "#2171B4", ">10%": "#08306B"},
    "Tumour size": {"Micro": "#C7E9C0", "Macro": "#41AA5D", "Giant": "#00441B", "Unknown": "#989898"},
    "Surgical revision": {"Yes": "#8C5109", "No": "#F5E8C2"},
    "Score (0-6)": {"0": "#FEE4D8", "1": "#FCBAA0", "2": "#FC9171", "3": "#FB6949", "4": "#EF3B2C", "5": "#CB181D",
                    "6": "#66000C", "Not done": "#D9D9D9"},
}
NA_COL = "#D9D9D9"
NON_COL, AGG_COL = "#8C8C8C", "#9B1C1C"

# ---------------------------------------------------------------- data
d = pd.read_excel(TABLE, header=2).dropna(subset=["Tumour type"])
d = d[d["Patient"].astype(str).str.match(r"^[A-Z]+ \d+$")].copy()
for c in ["CDKN1A score", "CTGF score", "Age (years)"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["G"] = np.where(d["Surgical revision"].astype(str).str.lower().str.startswith("y"), "Aggressive", "Non-aggressive")
d["num"] = d["Patient"].str.extract(r"(\d+)").astype(int)
def age_bin(a):
    if pd.isna(a): return np.nan
    return "≤30" if a <= 30 else "31-50" if a <= 50 else "51-70" if a <= 70 else "70+"
d["Age bin"] = d["Age (years)"].map(age_bin)
d["Size"] = d["Tumour size"].map({"Microadenoma": "Micro", "Macroadenoma": "Macro", "Giant adenoma": "Giant"}).fillna("Unknown")
d["Ki"] = d["Ki-67"].astype(str).str.replace("–", "-")
score_lab = lambda v: "Not done" if pd.isna(v) else str(int(v))
GORDER = ["Non-aggressive", "Aggressive"]

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- statistics
def mw_exact(a, b):
    """Exact two-sided Mann-Whitney p with tied mid-ranks by full enumeration (matches GraphPad Prism)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = rankdata(np.r_[a, b]); n1, n = len(a), len(a) + len(b)
    obs, mu = r[:n1].sum(), n1 * (n + 1) / 2
    vals, cnt = np.unique(r, return_counts=True)
    dist = {(0, 0.0): 1}
    for v, c in zip(vals, cnt):
        new = {}
        for (k, s), w in dist.items():
            for j in range(0, min(c, n1 - k) + 1):
                key = (k + j, round(s + j * v, 6)); new[key] = new.get(key, 0) + w * comb(int(c), j)
        dist = new
    tot = sum(w for (k, s), w in dist.items() if k == n1)
    return sum(w for (k, s), w in dist.items() if k == n1 and abs(s - mu) >= abs(obs - mu) - 1e-9) / tot

def van_elteren(df, col):
    W = E = V = 0
    for _, s in df.groupby("Centre"):
        s = s[s[col].notna()]; n = len(s); g = (s.G == "Aggressive").values; n1 = g.sum(); n0 = n - n1
        if not n1 or not n0: continue
        r = rankdata(s[col]); w = 1 / (n + 1); t = np.unique(s[col], return_counts=True)[1]
        W += w * r[g].sum(); E += w * n1 * (n + 1) / 2
        V += w * w * n1 * n0 / 12 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1)))
    return 2 * norm.sf(abs((W - E) / np.sqrt(V)))

def auc_ci(a, b, seed=0, B=5000):
    a, b = np.asarray(a), np.asarray(b)
    f = lambda a, b: ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).mean()
    rng = np.random.default_rng(seed)
    bs = [f(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]
    return f(a, b), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def fmt_p(p):
    if p < 0.0001: return "<0.0001"
    if p < 0.001: return f"{p:.4f}"
    return f"{p:.2g}"

# ---------------------------------------------------------------- clinical heatmaps
def heatmap(df, rows, blocks, title_label="Patient", w_mm=170, cell_h=3.4, fname="heatmap", legend_cols=None):
    """blocks: list of (block title or None, dataframe) drawn left to right; each split Non-aggressive | Aggressive.
    Layout is in millimetres: label margin, heatmap, patient numbers, then a compact legend."""
    multi = len(blocks) > 1
    cols, xpos, brackets, block_spans, x = [], [], [], [], 0.0
    for bt, sub in blocks:
        start = x
        for g in GORDER:
            s = sub[sub.G == g].sort_values(["Centre", "num"])
            if s.empty: continue
            brackets.append((x, x + len(s), g)); cols.append(s); xpos += list(np.arange(x, x + len(s))); x += len(s)
        block_spans.append((start, x, bt)); x += 1.5 if multi else 0
    ncol = x - (1.5 if multi else 0)
    allc = pd.concat(cols)
    left, right = 24, 3
    cw = (w_mm - left - right) / ncol
    top = 13 if multi else 8
    lab_h = 7 if ncol > 60 else 4
    legend_cols = legend_cols or [[k] for k in PAL]
    leg_rows = max(sum(len(PAL[k]) + 1.6 for k in grp) for grp in legend_cols)
    leg_h = leg_rows * 3.0 + 2
    H = top + len(rows) * cell_h + lab_h + 4 + leg_h
    fig = plt.figure(figsize=(w_mm * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w_mm); ax.set_ylim(H, 0); ax.axis("off")
    X = lambda c: left + c * cw
    Y = lambda r: top + r * cell_h
    for ri, (lab, fn, pal) in enumerate(rows):
        for xi, (_, r) in zip(xpos, allc.iterrows()):
            v = fn(r); col = PAL[pal].get(v, NA_COL) if isinstance(v, str) else NA_COL
            ax.add_patch(Rectangle((X(xi), Y(ri)), cw, cell_h, fc=col, ec="white", lw=0.4))
        ax.text(left - 1, Y(ri) + cell_h / 2, lab, ha="right", va="center", fontsize=6,
                style="italic" if lab.split()[0] in ("CDKN1A", "CTGF") else "normal")
    yl = Y(len(rows)) + 0.8
    for xi, (_, r) in zip(xpos, allc.iterrows()):
        ax.text(X(xi) + cw / 2, yl, str(r.num), ha="center", va="top", fontsize=4.4 if ncol > 60 else 5.5,
                rotation=90 if ncol > 60 else 0)
    ax.text(left - 1, yl, title_label, ha="right", va="top", fontsize=6)
    for (a, b, g) in brackets:
        ax.plot([X(a) + 0.3, X(b) - 0.3], [top - 1.2] * 2, c="k", lw=0.6)
        short_lab = ("Non-agg" if g == "Non-aggressive" else ("Agg" if b - a < 5 else "Aggressive"))
        ax.text((X(a) + X(b)) / 2, top - 1.6, short_lab if multi else g,
                ha="center", va="bottom", fontsize=5.5 if multi else 6.5, fontweight="normal" if multi else "bold")
    for bs, be, bt in block_spans:
        inner = [bb for bb in brackets if bs <= bb[0] < be]
        if len(inner) == 2:
            ax.plot([X(inner[0][1])] * 2, [top - 0.6, Y(len(rows))], c="k", lw=0.8, ls=(0, (3, 2)))
        if bt:
            ax.plot([X(bs) + 0.3, X(be) - 0.3], [top - 5.6] * 2, c="k", lw=0.8)
            ax.text((X(bs) + X(be)) / 2, top - 6.0, bt, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    # compact legend
    y0 = Y(len(rows)) + lab_h + 4
    widths = {"Surgical revision": 21, "Score (0-6)": 16, "Tumour size": 17}
    total = sum(max(widths.get(k, 12.5) for k in grp) for grp in legend_cols)
    xl = left + max(0, (w_mm - left - right - total) / 2)
    for grp in legend_cols:
        yy = y0
        for k in grp:
            ax.text(xl, yy, k, fontsize=6, fontweight="bold", va="top"); yy += 3.2
            for lab, col in PAL[k].items():
                ax.add_patch(Rectangle((xl, yy), 2.4, 2.4, fc=col, ec="none"))
                ax.text(xl + 3.2, yy + 1.2, lab, fontsize=5.5, va="center"); yy += 3.0
            yy += 1.6
        xl += max(widths.get(k, 12.5) for k in grp)
    save(fig, fname)

ROWS_NF = lambda score: [("Centre", lambda r: r.Centre, "Centre"), ("Sex", lambda r: r.Sex, "Sex"),
                         ("Age", lambda r: r["Age bin"], "Age"), ("Knosp", lambda r: str(r["Knosp grade"]), "Knosp"),
                         ("Ki-67", lambda r: r.Ki, "Ki-67"), ("Surgical revision", lambda r: r["Surgical revision"], "Surgical revision"),
                         (f"{score.split()[0]} score", lambda r, s=score: score_lab(r[s]), "Score (0-6)")]
LEG_NF = [["Centre", "Sex"], ["Age"], ["Knosp"], ["Ki-67"], ["Surgical revision"], ["Score (0-6)"]]
nf = d[d["Tumour type"] == "NF-PitNET"]
heatmap(nf, ROWS_NF("CDKN1A score"), [(None, nf)], fname="Fig3A_heatmap_CDKN1A", legend_cols=LEG_NF)
heatmap(nf, ROWS_NF("CTGF score"), [(None, nf)], fname="Fig4A_heatmap_CTGF", legend_cols=LEG_NF)
fu = d[d["Tumour type"] != "NF-PitNET"]
ROWS_FU = ROWS_NF("CDKN1A score")[:5] + [("Tumour size", lambda r: r.Size, "Tumour size"),
          ("Surgical revision", lambda r: r["Surgical revision"], "Surgical revision"),
          ("CDKN1A score", lambda r: score_lab(r["CDKN1A score"]), "Score (0-6)"),
          ("CTGF score", lambda r: score_lab(r["CTGF score"]), "Score (0-6)")]
SUB = [("Acromegaly", "Somatotroph"), ("Cushing's disease", "Corticotroph"), ("Prolactinoma", "Lactotroph")]
heatmap(fu, ROWS_FU, [(t, fu[fu["Tumour type"].str.startswith(k)]) for t, k in SUB], title_label="Case",
        fname="SuppFig6A_heatmap_functioning", w_mm=180,
        legend_cols=[["Centre", "Sex"], ["Age"], ["Knosp"], ["Ki-67"], ["Tumour size"], ["Surgical revision"], ["Score (0-6)"]])

# ---------------------------------------------------------------- score dot plots (Prism style)
def spread(vals, x0, width=0.55):
    xs = np.zeros(len(vals))
    for v in np.unique(vals):
        idx = np.where(vals == v)[0]; k = len(idx)
        xs[idx] = x0 + (np.arange(k) - (k - 1) / 2) * min(width / max(k - 1, 1), 0.12)
    return xs

def dotplot(ax, groups, ylabel, pairs=(), title=None):
    """groups: list of (label, values, colour, marker)."""
    for i, (lab, v, col, mk) in enumerate(groups):
        v = np.asarray(v, float)
        if len(v) == 0: continue
        ax.scatter(spread(v, i), v, s=6, c=col, marker=mk, lw=0, zorder=3)
        med = np.median(v); q1, q3 = np.percentile(v, [25, 75])
        ax.hlines(med, i - 0.28, i + 0.28, color=col, lw=1.0, zorder=4)
        ax.vlines(i, q1, q3, color=col, lw=0.8, zorder=4)
        ax.hlines([q1, q3], i - 0.1, i + 0.1, color=col, lw=0.8, zorder=4)
    for (i, j, p) in pairs:
        y = 6.55
        ax.plot([i, i, j, j], [y - 0.15, y, y, y - 0.15], c="k", lw=0.6)
        ax.text((i + j) / 2, y + 0.1, fmt_p(p), ha="center", va="bottom", fontsize=5.5)
    ax.set_ylim(-0.4, 7.4); ax.set_yticks([0, 2, 4, 6]); ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=5.5)
    ax.set_ylabel(ylabel, fontsize=6); ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=5.5, length=2)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    if title: ax.set_title(title, fontsize=6.3, pad=8)

def grp(df, col, g):
    return df[(df.G == g) & df[col].notna()][col].values

STATS = []
# Fig. 3C: CDKN1A, UCLH NF-PitNETs
u = nf[nf.Centre == "UCLH"]
a, b = grp(u, "CDKN1A score", "Aggressive"), grp(u, "CDKN1A score", "Non-aggressive")
p = mw_exact(a, b); STATS.append(("Fig 3C CDKN1A UCLH", len(b), len(a), np.median(b), np.median(a), p))
fig, ax = plt.subplots(figsize=(28 * MM, 42 * MM))
dotplot(ax, [(f"Non-agg\n(n = {len(b)})", b, NON_COL, "s"), (f"Agg\n(n = {len(a)})", a, AGG_COL, "^")],
        "$\\it{CDKN1A}$ total score (0-6)", pairs=[(0, 1, p)])
save(fig, "Fig3C_CDKN1A_scores")

# Fig. 4C: CTGF by centre
groups, pairs = [], []
for c in ["UCLH", "UoC"]:
    s = nf[nf.Centre == c]
    b, a = grp(s, "CTGF score", "Non-aggressive"), grp(s, "CTGF score", "Aggressive")
    p = mw_exact(a, b); STATS.append((f"Fig 4C CTGF {c}", len(b), len(a), np.median(b), np.median(a), p))
    pairs.append((len(groups), len(groups) + 1, p))
    groups += [(f"Non-agg\n{c}\n(n = {len(b)})", b, NON_COL, "s"), (f"Agg\n{c}\n(n = {len(a)})", a, AGG_COL, "^")]
p_ve = van_elteren(nf, "CTGF score"); STATS.append(("Fig 4C CTGF van Elteren", None, None, None, None, p_ve))
def draw_4c(ax):
    dotplot(ax, groups, "$\\it{CTGF}$ total score (0-6)", pairs=pairs)

rho, prho = spearmanr(u["CDKN1A score"], u["CTGF score"])
def draw_4d(ax):
    cnt = u.groupby(["CDKN1A score", "CTGF score", "G"]).size().reset_index(name="n")
    shared = cnt.groupby(["CDKN1A score", "CTGF score"]).G.transform("nunique") > 1
    for (_, r), sh in zip(cnt.iterrows(), shared):
        dx, dy = (-0.33, 0.33) if (sh and r.G == "Non-aggressive") else (0.0, 0.0)
        col = NON_COL if r.G == "Non-aggressive" else AGG_COL
        ax.scatter(r["CDKN1A score"] + dx, r["CTGF score"] + dy, s=28 * r.n, c=col, ec="white", lw=0.6,
                   zorder=3 + (sh and r.G == "Non-aggressive"))
        ax.text(r["CDKN1A score"] + dx, r["CTGF score"] + dy, str(r.n), ha="center", va="center", fontsize=4.8,
                color="white", zorder=5, fontweight="bold")
    ax.set_xlim(-0.3, 6.8); ax.set_ylim(-0.4, 7.4); ax.set_xticks(range(0, 7, 2)); ax.set_yticks(range(0, 7, 2))
    ax.set_xlabel("$\\it{CDKN1A}$ total score (0-6)", fontsize=6); ax.set_ylabel("$\\it{CTGF}$ total score (0-6)", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.text(0.02, 1.0, (f"Spearman ρ = {rho:.2f}, P < 0.0001" if prho < 1e-4 else f"Spearman ρ = {rho:.2f}, P = {prho:.2g}")
            + f"\nn = {len(u)} (UCLH)", transform=ax.transAxes, fontsize=5.5, va="top")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, mfc=c_, mec="none", label=l) for l, c_ in
                       [("Non-aggressive", NON_COL), ("Aggressive", AGG_COL)]], title="Bubble size = no. of tumours",
              title_fontsize=5.2, fontsize=5.2, frameon=False, loc="lower right")

for fn, name in [(draw_4c, "Fig4C_CTGF_scores"), (draw_4d, "Fig4D_CDKN1A_vs_CTGF")]:
    fig, ax = plt.subplots(figsize=(52 * MM, 46 * MM)); fn(ax); save(fig, name)

# Fig. 4C and 4D together: identical plotting areas side by side (positions in mm)
W_, H_ = 170, 58
PW, PH, BOT, L1, GAP = 62, 36, 15, 12, 26
fig = plt.figure(figsize=(W_ * MM, H_ * MM))
ax_c = fig.add_axes([L1 / W_, BOT / H_, PW / W_, PH / H_]); draw_4c(ax_c)
ax_d = fig.add_axes([(L1 + PW + GAP) / W_, BOT / H_, PW / W_, PH / H_]); draw_4d(ax_d)
fig.savefig(f"{OUT}/Fig4CD_combined.pdf"); fig.savefig(f"{OUT}/Fig4CD_combined.png", dpi=600); plt.close(fig)

# Supp. Fig. 6B-G: functioning tumours
fig = plt.figure(figsize=(150 * MM, 150 * MM))
gs = fig.add_gridspec(3, 2, width_ratios=[0.8, 1.6], hspace=0.75, wspace=0.45)
for ri, (title, key) in enumerate(SUB):
    s = fu[fu["Tumour type"].str.startswith(key)]
    su = s[s.Centre == "UCLH"]
    b, a = grp(su, "CDKN1A score", "Non-aggressive"), grp(su, "CDKN1A score", "Aggressive")
    ax = fig.add_subplot(gs[ri, 0]); p = mw_exact(a, b) if len(a) and len(b) else np.nan
    STATS.append((f"Supp 6 {key} CDKN1A UCLH", len(b), len(a), np.median(b) if len(b) else None, np.median(a) if len(a) else None, p))
    dotplot(ax, [(f"Non-agg\n(n = {len(b)})", b, NON_COL, "s"), (f"Agg\n(n = {len(a)})", a, AGG_COL, "^")],
            "$\\it{CDKN1A}$ total score (0-6)", pairs=[(0, 1, p)] if np.isfinite(p) else [], title="$\\it{CDKN1A}$ (UCLH)")
    ax.text(-0.75, 0.5, title, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.5, fontweight="bold")
    groups, pairs = [], []
    for c in ["UCLH", "UoC"]:
        sc_ = s[s.Centre == c]
        b, a = grp(sc_, "CTGF score", "Non-aggressive"), grp(sc_, "CTGF score", "Aggressive")
        p = mw_exact(a, b) if len(a) and len(b) else np.nan
        STATS.append((f"Supp 6 {key} CTGF {c}", len(b), len(a), np.median(b) if len(b) else None, np.median(a) if len(a) else None, p))
        i0 = len(groups)
        if len(b): groups.append((f"Non-agg\n{c}\n(n = {len(b)})", b, NON_COL, "s"))
        if len(a): groups.append((f"Agg\n{c}\n(n = {len(a)})", a, AGG_COL, "^"))
        if np.isfinite(p): pairs.append((i0, i0 + 1, p))
    ax = fig.add_subplot(gs[ri, 1])
    dotplot(ax, groups, "$\\it{CTGF}$ total score (0-6)", pairs=pairs, title="$\\it{CTGF}$ (UCLH and UoC)")
save(fig, "SuppFig6BG_functioning_scores")

# Supp. Fig. 7: ROC curves
def roc(a, b):
    ts = np.r_[np.inf, np.unique(np.r_[a, b])[::-1], -np.inf]
    return [(np.mean(b >= t), np.mean(a >= t)) for t in ts]
fig, ax = plt.subplots(figsize=(70 * MM, 52 * MM))
for lab, s, col, ls, mfc in [("$\\it{CDKN1A}$, UCLH", ("UCLH", "CDKN1A score"), "#B2182B", "-", "#B2182B"),
                             ("$\\it{CTGF}$, UCLH", ("UCLH", "CTGF score"), "#2166AC", "-", "#2166AC"),
                             ("$\\it{CTGF}$, UoC", ("UoC", "CTGF score"), "#2166AC", "--", "white")]:
    x = nf[(nf.Centre == s[0]) & nf[s[1]].notna()]
    a, b = grp(x, s[1], "Aggressive"), grp(x, s[1], "Non-aggressive")
    e, lo, hi = auc_ci(a, b); STATS.append((f"Supp 7 AUC {s[1]} {s[0]}", len(b), len(a), None, None, (e, lo, hi)))
    pts = np.array(roc(a, b))
    ax.plot(pts[:, 0], pts[:, 1], ls=ls, c=col, lw=1.2, marker="o", ms=3, mfc=mfc, mec=col, mew=0.8,
            label=f"{lab}: AUC {e:.2f} ({lo:.2f}-{hi:.2f})")
ax.plot([0, 1], [0, 1], ls=":", c="#AAAAAA", lw=0.8)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("1 - specificity", fontsize=6); ax.set_ylabel("Sensitivity", fontsize=6); ax.tick_params(labelsize=5.5, length=2)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(fontsize=5.2, frameon=False, loc="lower right")
save(fig, "SuppFig7_ROC")

print("panel, n non-aggressive, n aggressive, median non-agg, median agg, p (or AUC with 95% CI)")
for r in STATS: print(r)


####################################################################################################
# SECTION 10  Software versions
####################################################################################################
import importlib, sys
print("Python", sys.version.split()[0])
for pkg in ["scanpy", "anndata", "numpy", "pandas", "scipy", "statsmodels", "matplotlib",
            "gseapy", "pydeseq2", "infercnvpy", "openpyxl", "sklearn"]:
    try:
        print(f"{pkg:12s}", importlib.import_module(pkg).__version__)
    except Exception as e:
        print(f"{pkg:12s} not available ({e.__class__.__name__})")

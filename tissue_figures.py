# ==== Tissue cohort figures (Fig. 3A, 3C, 4A, 4C, 4D; Supp. Fig. 6; Supp. Fig. 7) ====================
# Aggressive = repeat surgery for regrowth or recurrence; non-aggressive = no repeat surgery (>= 2 years' follow-up).
# Input: Supplementary_Table_3.xlsx (one row per tumour). Output: PDF + PNG for each panel. No panel letters.
import sys, numpy as np, pandas as pd
from math import comb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, norm, spearmanr

TABLE = sys.argv[1] if len(sys.argv) > 1 else "Supplementary_Table_3.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
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

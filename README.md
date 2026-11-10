# NFPA_sc

Code to reproduce the single-cell atlas figures in:

> Kaufman-Cook et al. (2026). *A senescence-associated programme marks aggressive non-functioning pituitary neuroendocrine tumours.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andoniadou-Lab/NFPA_sc/blob/main/NFPitNET_senescence_atlas_figures.ipynb)

## Contents

| File | Description |
|---|---|
| `Fig1A_sample_overview.R` | Figure 1A: overview of the 50 atlas samples (R) |
| `NFPitNET_senescence_atlas_figures.ipynb` | Figures 1B–C and 2, Supplementary Figures 1–4 and Supplementary Table 1 (Python, Colab) |

## Data

The analyses use an integrated AnnData object of 21 NF-PitNETs and 29 healthy pituitary samples (285,907 cells) from eight published studies, subset from the Pituitary Tumour Atlas (Kövér et al., in preparation). Accession numbers for every sample are listed in Supplementary Table 1. [Add how to obtain the integrated object and the sample metadata CSV.]

## Running

**Fig. 1A.** In R, set `METADATA_CSV` to the atlas sample metadata and run `Fig1A_sample_overview.R`.

**All other figures.** Open the notebook in Google Colab, set `DATA_PATH` and `OUT_DIR` in the Configuration cell, and run all cells from top to bottom. Each figure section depends on the sections above it. Figures are written as PDF and PNG (600 dpi), with the underlying tables as CSV. The final cell prints the package versions used.

## Licence

MIT (see `LICENSE`).

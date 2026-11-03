# ==== Fig. 1A: overview of the 50 samples in the single-cell atlas ==============================
# Senescence and the SASP in non-functioning PitNETs (Kaufman-Cook, Vamvoukaki et al.)
#
# Input : sample-level metadata of the Pituitary Tumour Atlas (one row per sample; columns include
#         Internal_ID, SRA_ID, Author, `Cell type`, Subtype, `10X version`, Modality, Final_sex,
#         Age_numeric, KNOSP, KI67, Invasion)
# Output: Fig1A_sample_overview.pdf / .png (600 dpi) / .svg, 7.1 x 4.6 in (full page width)
#
# Packages: dplyr, tidyr, stringr, ggplot2, ggtext, svglite (for the .svg)

METADATA_CSV <- "nfpa_pituitary_tumor_atlas_2026_10_27.csv"
OUT_NAME     <- "Fig1A_sample_overview"

# Load necessary packages
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(ggtext)  # bold legend headers

# Samples in the metadata but not in the integrated object
excluded_samples <- c("HRS1189473")

# ---- Step 1: Read in the data -------------------------------------------------
raw <- read.csv(METADATA_CSV, colClasses = "character",
                check.names = FALSE, na.strings = c("", "NA")) %>%
  filter(!is.na(Internal_ID))

# ---- Step 2: Clean and harmonise each field (one row per sample) -------------------
samples <- raw %>%
  mutate(
    Patient_ID = str_squish(SRA_ID),
    Condition  = ifelse(`Cell type` == "Normal", "Healthy", "Tumour"),
    
    `Data Source` = str_replace(str_squish(Author), ",\\s*\\(", " ("),
    
    Technology = paste(
      case_when(`10X version` == "10XV2"   ~ "10x v2",
                `10X version` == "10XV3"   ~ "10x v3",
                `10X version` == "SeekOne" ~ "SeekOne"),
      case_when(str_detect(Modality, "^multi") ~ "snMultiome",
                str_detect(Modality, "^sn")    ~ "snRNA-seq",
                TRUE                           ~ "scRNA-seq")
    ),
    
    # Final_sex is complete (1 = male, 0 = female); the Sex text column has gaps
    Sex = case_when(Final_sex %in% c("1", "1.0") ~ "Male",
                    Final_sex %in% c("0", "0.0") ~ "Female"),
    
    age_num = suppressWarnings(as.numeric(Age_numeric)),
    Age = case_when(age_num <  18 ~ "<18",
                    age_num <= 30 ~ "18-30",
                    age_num <= 40 ~ "31-40",
                    age_num <= 50 ~ "41-50",
                    age_num <= 60 ~ "51-60",
                    age_num <= 70 ~ "61-70",
                    age_num >  70 ~ "71+"),
    
    `Reported subtype` = case_when(
      Subtype == "Residual_Normal" ~ "Residual normal gland",
      `Cell type` == "Normal"      ~ "Healthy donor",
      `Cell type` == "Null"        ~ "Null cell",
      `Cell type` == "Gonadotroph" ~ "Gonadotroph"),
    
    `Knosp Grade` = KNOSP %>% str_squish() %>% str_to_upper() %>%
      str_replace("^([0-9])\\.0+$", "\\1"),
    
    ki67_num = sapply(
      str_extract_all(str_replace_all(KI67, ",", "."), "[0-9.]+"),
      function(x) if (length(x) == 0 || all(is.na(suppressWarnings(as.numeric(x))))) NA_real_
      else mean(as.numeric(x), na.rm = TRUE)
    ),
    `Ki-67` = case_when(ki67_num <  3 ~ "<3%",
                        ki67_num >= 3 ~ "\u22653%"),
    
    Invasion = case_when(Invasion %in% c("Invasive", "Yes", "Invasion") ~ "Invasive",
                         Invasion %in% c("Non_invasive", "No")          ~ "Non-invasive")
  ) %>%
  # tumour-only variables are not applicable to healthy / residual normal glands
  # (Yan residual-normal rows carry the paired tumour's Ki-67 and invasion, which must not be shown)
  mutate(across(c(`Knosp Grade`, `Ki-67`, Invasion),
                ~ ifelse(Condition == "Healthy", "Not applicable", .x))) %>%
  distinct(Patient_ID, .keep_all = TRUE) %>%
  filter(!Patient_ID %in% excluded_samples) %>%
  # order: healthy first (healthy donors, then residual normal), then tumours; by study within each
  arrange(factor(Condition, levels = c("Healthy", "Tumour")),
          factor(`Reported subtype`, levels = c("Healthy donor", "Residual normal gland",
                                                "Gonadotroph", "Null cell")),
          `Data Source`, Patient_ID)

message(nrow(samples), " samples plotted (", sum(samples$Condition == "Healthy"), " healthy, ",
        sum(samples$Condition == "Tumour"), " tumour)")   # should be 50 (29 healthy, 21 tumour)

# ---- Step 3: Reshape to long format --------------------------------------------
meta_rows <- c("Condition", "Data Source", "Technology", "Sex", "Age", "Reported subtype",
               "Knosp Grade", "Ki-67", "Invasion")

metadata_long <- samples %>%
  select(Patient_ID, all_of(meta_rows)) %>%
  pivot_longer(-Patient_ID, names_to = "Metadata", values_to = "Value") %>%
  mutate(
    Value = ifelse(is.na(Value) | grepl("^\\s*$", Value), "Unknown", Value),
    Value_grouped = ifelse(Value %in% c("Unknown", "Not applicable"), Value, paste0(Metadata, ": ", Value)),
    Metadata   = factor(Metadata, levels = rev(meta_rows)),
    Patient_ID = factor(Patient_ID, levels = samples$Patient_ID)
  )

# ---- Step 4: Colours and legend layout -------------------------------------------
# Data Source uses the same colours as the study UMAP in Fig. 1B
legend_spec <- list(
  "Condition"   = c("Healthy" = "#4393C3", "Tumour" = "#8B1A1A"),
  "Data Source" = c("Asuzu et al. (2022)"    = "#1f77b4",
                    "Huang et al. (2025)"    = "#ff7f0e",
                    "Ilie et al. (2025)"     = "#2ca02c",
                    "Potthoff et al. (2024)" = "#d62728",
                    "Yan et al. (2024)"      = "#9467bd",
                    "Zhang et al. (2022c)"   = "#8c564b",
                    "Zhang et al. (2026)"    = "#e377c2",
                    "Zhang et al. (2022b)"   = "#7f7f7f"),
  "Technology"  = c("10x v2 scRNA-seq"  = "#F28E2B",
                    "10x v3 scRNA-seq"  = "#B07AA1",
                    "SeekOne scRNA-seq" = "#76B7B2",
                    "10x v3 snRNA-seq"  = "#EDC948",
                    "10x v3 snMultiome" = "#59A14F"),
  "Sex"         = c("Female" = "#D7191C", "Male" = "#2C7BB6"),
  "Age"         = c("<18" = "#FDE0EF", "18-30" = "#440154", "31-40" = "#414487", "41-50" = "#2A788E",
                    "51-60" = "#22A884", "61-70" = "#7AD151", "71+" = "#FDE725"),
  "Reported subtype" = c("Healthy donor" = "#C6DBEF", "Residual normal gland" = "#6BAED6",
                         "Gonadotroph" = "#DFC27D", "Null cell" = "#8C510A"),
  "Knosp Grade" = c("0" = "#C6DBEF", "1" = "#9ECAE1", "2" = "#6BAED6", "3" = "#4292C6",
                    "3A" = "#2171B5", "3B" = "#08519C", "4" = "#08306B"),
  "Ki-67"       = setNames(c("#FCBBA1", "#CB181D"), c("<3%", "\u22653%")),
  "Invasion"    = c("Non-invasive" = "#A1D99B", "Invasive" = "#E6550D")
)
unknown_col <- "grey80"
na_col      <- "grey96"

# Keep only legend entries that appear in the data
present <- split(metadata_long$Value, as.character(metadata_long$Metadata))
legend_spec <- setNames(lapply(names(legend_spec), function(m)
  legend_spec[[m]][names(legend_spec[[m]]) %in% present[[m]]]), names(legend_spec))

# Legend columns (groups are never split across columns)
legend_cols <- list(
  c("Condition", "Sex", "Ki-67"),
  c("Data Source"),
  c("Technology", "Invasion"),
  c("Age"),
  c("Reported subtype"),
  c("Knosp Grade", "Not applicable", "Unknown")
)

group_entries <- function(m) {
  if (m %in% c("Unknown", "Not applicable")) {
    return(list(levels = m, labels = paste0("<b>", m, "</b>"),
                fills = setNames(ifelse(m == "Unknown", unknown_col, na_col), m)))
  }
  item_lv <- paste0(m, ": ", names(legend_spec[[m]]))
  list(
    levels = c(paste(m, "Header"), item_lv),
    labels = c(paste0("<b>", m, "</b>"), paste0("  ", names(legend_spec[[m]]))),
    fills  = setNames(c("white", unname(legend_spec[[m]])), c(paste(m, "Header"), item_lv))
  )
}

col_entries <- lapply(legend_cols, function(groups) {
  e <- lapply(groups, group_entries)
  list(levels = unlist(lapply(e, `[[`, "levels")),
       labels = unlist(lapply(e, `[[`, "labels")),
       fills  = unlist(lapply(e, `[[`, "fills")))
})
n_rows <- max(sapply(col_entries, function(e) length(e$levels)))

# Pad shorter columns with invisible spacer entries
legend_levels <- character(0); legend_labels <- character(0); fill_colors <- character(0)
n_spacers <- 0
for (e in col_entries) {
  n_pad  <- n_rows - length(e$levels)
  spacer <- if (n_pad > 0) paste0("spacer_", n_spacers + seq_len(n_pad)) else character(0)
  n_spacers <- n_spacers + n_pad
  legend_levels <- c(legend_levels, e$levels, spacer)
  legend_labels <- c(legend_labels, e$labels, rep(" ", n_pad))
  fill_colors   <- c(fill_colors, e$fills, setNames(rep("white", n_pad), spacer))
}

unmatched <- setdiff(unique(metadata_long$Value_grouped), legend_levels)
if (length(unmatched)) warning("No colour for: ", paste(unmatched, collapse = ", "))

# ---- Step 5: Apply factor levels -------------------------------------------------
metadata_long$Value_grouped <- factor(metadata_long$Value_grouped, levels = legend_levels)

n_healthy <- sum(samples$Condition == "Healthy")
n_tumour  <- sum(samples$Condition == "Tumour")
n_meta    <- length(meta_rows)

# ---- Step 6: Plot --------------------------------------------------------------
p <- ggplot(metadata_long, aes(x = Patient_ID, y = Metadata, fill = Value_grouped)) +
  geom_tile(color = "white", linewidth = 0.4, show.legend = TRUE) +
  geom_vline(xintercept = n_healthy + 0.5, linetype = "dashed", linewidth = 0.4) +
  annotate("segment", x = 0.6, xend = n_healthy + 0.4, y = n_meta + 0.75, yend = n_meta + 0.75,
           linewidth = 0.4, colour = "#4393C3") +
  annotate("segment", x = n_healthy + 0.6, xend = n_healthy + n_tumour + 0.4,
           y = n_meta + 0.75, yend = n_meta + 0.75, linewidth = 0.4, colour = "#8B1A1A") +
  annotate("text", x = (n_healthy + 1) / 2, y = n_meta + 1.4, fontface = "bold", size = 3,
           colour = "#4393C3", label = paste0("Healthy (n = ", n_healthy, ")")) +
  annotate("text", x = n_healthy + (n_tumour + 1) / 2, y = n_meta + 1.4, fontface = "bold", size = 3,
           colour = "#8B1A1A", label = paste0("Tumour (n = ", n_tumour, ")")) +
  scale_fill_manual(values = fill_colors, labels = legend_labels, breaks = legend_levels, drop = FALSE) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_fixed(clip = "off") +
  labs(x = "Accession ID") +
  theme_minimal(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 5.5, colour = "black"),
    axis.text.y = element_text(size = 8, colour = "black"),
    axis.title.x = element_text(size = 8, colour = "black", margin = margin(t = 4)),
    axis.title.y = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    legend.justification = "center",
    legend.title = element_blank(),
    legend.key.size = unit(0.25, "cm"),
    legend.text = ggtext::element_markdown(size = 6),
    legend.margin = margin(t = 4),
    plot.margin = margin(18, 5, 5, 5)          # top room for the Healthy / Tumour labels
  ) +
  guides(fill = guide_legend(nrow = n_rows, byrow = FALSE, override.aes = list(color = NA)))

print(p)

# ---- Step 7: Save (full page width) -----------------------------------------------
ggsave(paste0(OUT_NAME, ".pdf"), p, device = cairo_pdf, width = 7.1, height = 4.6, units = "in")
ggsave(paste0(OUT_NAME, ".png"), p, device = "png", dpi = 600, width = 7.1, height = 4.6, units = "in")
ggsave(paste0(OUT_NAME, ".svg"), p, device = "svg", width = 7.1, height = 4.6, units = "in")

# Open Replication Repository & Reviewer Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24202251.svg)](https://doi.org/10.5281/zenodo.24202251)

**Manuscript Title**: Information-Enriched Forecast Combinations for Nepalese Bank Deposits: Retrospective Evidence and Reliability Limits  
**Journal Target**: *Applied Stochastic Models in Business and Industry* (ASMBI, Wiley)  
**Track**: Practitioner's Corner (Practitioner Paper)  
**Permanent Zenodo DOI**: [https://doi.org/10.5281/zenodo.24202251](https://doi.org/10.5281/zenodo.24202251)  
**Public GitHub Repository**: [https://github.com/Dips7/deposit-hybrid-forecast](https://github.com/Dips7/deposit-hybrid-forecast)  
**Authors**: Dipesh Gurung¹, Deepika Sharma²*, Rashmi Dahiya²  
1. *Department of Computer Science, Noida International University, Greater Noida, India*  
2. *Department of Computer Science, Noida International University, Greater Noida, India*  
\* *Correspondence: Deepika Sharma, Assistant Professor, Noida International University (Email: deepika.sharma@niu.edu.in; Alternative: Sharmadeepyan123@gmail.com; ORCID: https://orcid.org/0009-0001-1048-1448)*  
*ORCID IDs: Dipesh Gurung (https://orcid.org/0009-0009-0335-2267) | Rashmi Dahiya (https://orcid.org/0009-0004-7806-3340)*  

---

## 1. Overview and Replication Scope

This directory serves as the self-contained replication archive for readers and peer reviewers of *Applied Stochastic Models in Business and Industry* (Practitioner's Corner).

### Replication Scope:
1. **Independent Mathematical Recalculation**: The archive provides the complete origin-by-horizon evaluation ledger (`data/evaluation_DEVELOPMENT_FIT_V1.csv`, 1,337 scored predictions across 191 rolling origins). The included replication script (`scripts/replicate_from_osf.py`) performs 100% independent mathematical recalculation of:
   - Aggregate sample performance metrics (MAE, RMSE, MedAE, Mean Error / Bias, and operational fallback rates) for all 7 candidate procedures;
   - Outlier squared loss concentration curves (Lorenz-style cumulative loss across top-loss quantiles);
   - Primary centered moving-block bootstrap hypothesis tests ($L = 12$ months, $B = 10,000$ resamples, seed 20261029) and step-down Holm familywise error rate (FWER) adjusted p-values;
   - Block length sensitivity checks across $L = 6$ and $L = 18$ months.
2. **Tabular and Graphical Assets**: The package includes all 11 manuscript and supplementary tables in machine-readable CSV format, as well as publication-ready vector (PDF, EPS) and high-resolution raster (300 DPI PNG, 600 DPI PNG, TIFF) figures.
3. **Refitting Scope Distinction**: This archive enables deterministic recalculation, statistical inference reproduction, and artifact auditing from the complete frozen evaluation records. Full model retraining and tuning grids from raw historical scrapes are preserved in the institutional git repository.

---

## 2. Directory Structure

```
reviewer_osf_archive/
├── HASH_MANIFEST.json                   # SHA-256 cryptographic manifest of all archive contents
├── README.md                            # This replication guide
├── data/                                # Primary evaluation receipts, forecasts, and metrics
│   ├── evaluation_DEVELOPMENT_FIT_V1.csv  # 1,337 out-of-sample origin-by-horizon predictions
│   ├── forecasts_DEVELOPMENT_FIT_V1.csv   # Model forecast outputs
│   ├── metrics_DEVELOPMENT_FIT_V1.csv     # Aggregate performance metrics (MAE, RMSE, bias)
│   ├── inference_summary_DEVELOPMENT_FIT_V1.json # Moving-block bootstrap inference ledgers
│   ├── RECEIPT_DEVELOPMENT_FIT_V1.json    # Cryptographic execution receipt
│   ├── TAIL_METRICS.csv                   # Quantile-ordered squared loss concentration
│   ├── YEARLY_METRICS.csv                 # Annual origin- and target-year metrics (2010–2025)
│   ├── WIN_COUNTS.csv                     # Pairwise origin-by-origin tournament win tallies
│   ├── ERROR_CORRELATION.csv              # Inter-model forecast error cross-correlation matrix
│   └── OMO_INCREMENT_ORIGIN_LOSSES.csv    # Origin-by-origin loss differentials
├── tables/                              # Manuscript tables in machine-readable CSV format
│   ├── Table1_Data_and_Availability.csv
│   ├── Table2_Procedures_and_Components.csv
│   ├── Table3_Aggregate_Performance.csv
│   ├── Table4_Primary_Inference_Contrasts.csv
│   ├── TableS1_Order_Selection_and_Hyperparameters.csv
│   ├── TableS2_Bootstrap_Block_Sensitivity.csv
│   ├── TableS3_Origin_Year_Annual_Metrics.csv
│   ├── TableS4_Target_Year_Annual_Metrics.csv
│   ├── TableS5_Tail_Loss_Quantiles.csv
│   ├── TableS6_Error_Correlation_Matrix.csv
│   └── TableS7_Pairwise_Win_Counts.csv
├── figures/                             # Publication figures in PDF, EPS, 300/600 DPI PNG, TIFF
│   ├── Figure1_Information_Chronology.*
│   ├── Figure2_Cumulative_Loss_Differential.*
│   ├── Figure3_Annual_MAE_Tournament.*
│   ├── Figure4_Tail_Loss_Concentration.*
│   └── FigureS1_Error_Distributions_Boxplot.*
└── scripts/                             # Verification and extraction scripts
    ├── replicate_from_osf.py            # Standalone 0-dependency verification script
    └── build_tables_and_figures.py      # Table and figure generator
```

---

## 3. Replication Quick Start

To independently recalculate all metrics, loss quantiles, and moving-block bootstrap tests:

```bash
# Requires only Python >= 3.9, numpy, and pandas
python3 scripts/replicate_from_osf.py
```

### Verified Empirical Outputs:
- **Sample Origins**: 191 contiguous rolling origins (2010-01 to 2025-11, cutoff $c = t-2$, target $t+1 = c+3$).
- **Sample MAE (NPR million)**: $H_{\text{OMO}} = 23,546.37$ (lowest sample MAE; 11.13% reduction vs $E_{\text{S}}$, 8.13% vs $E_{\text{EQ}}$, 2.43% vs $H_{\text{Y}}$).
- **Sample RMSE (NPR million)**: $H_{\text{Y}} = 34,501.11$ vs $H_{\text{OMO}} = 34,515.16$ (edge of 14.05 mn NPR / 0.041% favoring target-only hybrid due to quadratic loss penalization of mid-2020 underprediction).
- **Primary Moving-Block Bootstrap Test ($L = 12$ months, $B = 10,000$ resamples)**:
  - Contrast $H_{\text{OMO}}$ vs $E_{\text{X}}$: Raw $p = 0.0441$, Step-down Holm $p_{\text{Holm}} = 0.1764$ (non-significant at $\alpha = 0.05$).
  - Contrast $H_{\text{OMO}}$ vs $E_{\text{S}}$: Raw $p = 0.0837$, Step-down Holm $p_{\text{Holm}} = 0.2511$.
  - Contrast $H_{\text{OMO}}$ vs $E_{\text{EQ}}$: Raw $p = 0.2284$, Step-down Holm $p_{\text{Holm}} = 0.4568$.
  - Contrast $H_{\text{OMO}}$ vs $H_{\text{Y}}$: Raw $p = 0.2398$, Step-down Holm $p_{\text{Holm}} = 0.4568$.
  - *Conclusion*: Zero contrasts achieve statistical significance after familywise error rate adjustment.
- **Block Length Sensitivity**:
  - $L = 6$: minimum $p_{\text{Holm}} = 0.0720 \ge 0.05$ (non-significant).
  - $L = 18$: minimum $p_{\text{Holm}} = 0.1996 \ge 0.05$ (non-significant).
- **Tail Squared Loss Concentration**:
  - The worst 9 evaluation origins (4.71% of origins) account for 51.25% of $H_{\text{OMO}}$'s cumulative squared loss and 47.06% of $H_{\text{Y}}$'s.

---

## 4. Cryptographic Integrity

Every file in this archive is cataloged in `HASH_MANIFEST.json` with its cryptographic SHA-256 hash. To verify file integrity:

```bash
python3 -c "
import json, hashlib, pathlib
manifest = json.loads(open('HASH_MANIFEST.json').read())
for fpath, exp_hash in manifest.items():
    p = pathlib.Path(fpath)
    if not p.exists():
        print(f'MISSING: {fpath}')
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == exp_hash, f'HASH MISMATCH: {fpath}'
print('ALL ARCHIVE FILES PASS SHA-256 INTEGRITY VERIFICATION.')
"
```

---

## 5. Repository and Deposit Status

This archive represents the self-contained replication bundle staged locally in the manuscript submission package (`reviewer_osf_archive/` and `reviewer_osf_archive.zip`).
- **Live GitHub Repository**: `https://github.com/Dips7/deposit-hybrid-forecast`
- **Permanent Zenodo DOI**: `https://doi.org/10.5281/zenodo.24202251`
- **Replication Scope**: Enables 100% deterministic recalculation, statistical inference reproduction, and artifact auditing directly from the complete frozen evaluation records (`data/evaluation_DEVELOPMENT_FIT_V1.csv`). It does not perform raw model re-scraping from preliminary central bank records.
- **Deposit Status**: Publicly active on GitHub and permanently archived on Zenodo with citable DOI (`10.5281/zenodo.24202251`).

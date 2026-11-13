[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24202251.svg)](https://doi.org/10.5281/zenodo.24202251)

# Open Replication Archive: Information-Enriched Forecast Combinations for Nepalese Bank Deposits

Official replication package accompanying the revised manuscript submitted to the **Practitioner's Corner** of ***Applied Stochastic Models in Business and Industry*** (ASMBI, Wiley).

- **Permanent Zenodo DOI**: [https://doi.org/10.5281/zenodo.24202251](https://doi.org/10.5281/zenodo.24202251)
- **Manuscript ID**: `ASMB-26-00640`
- **Track**: Practitioner's Corner (Practitioner Paper)
- **Authors**: Dipesh Gurung, Deepika Sharma*, Rashmi Dahiya (*Noida International University*)

---

## Package Contents

This release provides the complete, self-contained replication bundle for peer reviewers, central bank practitioners, and researchers:
- `data/`: All 1,337 out-of-sample origin-by-horizon predictions across 191 rolling origins (`evaluation_DEVELOPMENT_FIT_V1.csv`), aggregate metrics, tail squared-loss concentration records, and moving-block bootstrap ledgers.
- `tables/`: All 11 manuscript and supplementary tables in machine-readable CSV format.
- `figures/`: High-resolution figures in vector (PDF, EPS) and publication raster formats (300/600 DPI PNG, TIFF).
- `scripts/`: Zero-dependency standalone replication script (`scripts/replicate_from_osf.py`).
- `HASH_MANIFEST.json`: SHA-256 cryptographic hashes for 100% bit-for-bit file verification.

---

## Quick Start (Deterministic Recalculation)

Requires only standard **Python >= 3.9**, `numpy`, and `pandas`:

```bash
git clone https://github.com/Dips7/deposit-hybrid-forecast.git
cd deposit-hybrid-forecast
python3 scripts/replicate_from_osf.py
```

The script independently recalculates all sample metrics, confirms error identities, executes paired moving-block bootstrap tests ($L=12$ months, $B=10,000$ resamples, seed 20261029), and computes step-down Holm familywise error adjustments in under 15 seconds.

---

## Key Empirical Findings

1. **Sample MAE (NPR million)**: The open market operation-enriched hybrid ($H_{\text{OMO}}$) achieves the lowest sample MAE ($23,546.37$; an 11.13% reduction vs. SARIMA).
2. **Bootstrap Hypothesis Tests**: After step-down Holm correction, none of the primary performance contrasts is statistically significant ($p_{\text{Holm}} = 0.1764$).
3. **Tail Risk & Loss Asymmetry**: Under quadratic loss (RMSE), a target-only hybrid ($H_{\text{Y}}$) outperforms $H_{\text{OMO}}$ because 4.71% of origins (mid-2020 deposit surge) account for 51.25% of cumulative squared loss.

---

## Citation

```bibtex
@software{gurung2026deposit_hybrid,
  author       = {Dipesh Gurung and Deepika Sharma and Rashmi Dahiya},
  title        = {Replication Archive for Information-Enriched Forecast Combinations for Nepalese Bank Deposits},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24202251},
  url          = {https://doi.org/10.5281/zenodo.24202251}
}
```

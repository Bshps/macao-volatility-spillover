# Volatility Spillovers in Macao's Gaming Economy before and after COVID-19

**Read this in other languages:** [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文（香港）](README.zh-HK.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20523832.svg)](https://doi.org/10.5281/zenodo.20523832)

**Author**: Lin Yihuan (ORCID: [0000-0002-8391-7732](https://orcid.org/0000-0002-8391-7732))
**Affiliation**: School of Business and Economics, Universiti Putra Malaysia
**Journal**: *International Journal of Economic Performance* (المجلة الدولية للأداء الاقتصادي) — ISSN 2661-7161 (Print) / 2716-9073 (Online) — Volume 9, Numéro 1, pp. 113–128 (2026)
**Published article**: [https://asjp.cerist.dz/en/article/299154](https://asjp.cerist.dz/en/article/299154)

---

## Overview

This repository contains the replication package for the paper:

> Lin, Y. (2026). Volatility Spillovers in Macao's Gaming Economy before and after COVID-19. *International Journal of Economic Performance*, 9(1), 113–128. https://asjp.cerist.dz/en/article/299154

The study examines the dynamic relationship between overnight visitor arrivals and gaming revenue in Macao using **GJR-GARCH**, **Dynamic Conditional Correlation (DCC)**, and the **Diebold-Yilmaz spillover framework**. Key findings include:

- The DCC correlation between overnight arrivals and gaming revenue was positive pre-COVID (ρ̄=0.160), collapsed during the border closure (ρ̄=0.009), and turned persistently negative post-2022 (ρ̄=-0.071)
- Bad volatility spillovers (30.6%) substantially exceed good volatility spillovers (8.2%), a gap of 22.4 percentage points
- The standard modelling assumptions of stability and symmetry both fail when evaluated over the full sample

## Data

| File | Description | Source |
|------|-------------|--------|
| `dsec.xlsx` | Monthly overnight and same-day visitor arrivals | DSEC Macao SAR (https://www.dsec.gov.mo) |
| `gaming_revenue.csv` | Monthly gaming tax revenue (million MOP) | DSEC Macao SAR |
| | Sample period: January 2008 – July 2025 | |

The overnight and same-day visitor data are extracted from the DSEC tourism statistics table. Gaming revenue data are from the DSEC public accounts table.

## Requirements

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

Tested with Python 3.12.

## Reproduction

```bash
# Step 1: Run the main analysis (GJR-GARCH, DCC, Diebold-Yilmaz spillovers)
python paper2_analysis.py

# Step 2: Generate publication-quality figures
python make_figures_p2.py
```

All outputs are saved to the `output/` directory:
- `descriptive_stats.csv` — Table 2
- `gjr_garch_params.csv` — Table 3 (GJR-GARCH estimates)
- `correlation_matrix.csv` — Correlation matrix of log-returns
- `unit_root_tests.csv` — ADF and KPSS test results
- `arch_lm_tests.csv` — ARCH-LM test results
- `fig1_series_overview.png` — Overnight share and log-returns
- `fig2_cond_vol.png` — Conditional volatility estimates
- `fig3_dcc_correlations.png` — DCC time-varying correlation
- `fig4_rolling_spillover.png` — Rolling Diebold-Yilmaz spillover index
- `fig5_asymmetric_spillover.png` — Bad vs good volatility spillovers
- `paper2_results.pkl` — Full results (for use by `make_figures_p2.py`)

## Methodology

1. **GJR-GARCH(1,1)** with AR(1) mean equation and structural-break dummies (COVID-19, reopening, GFC)
2. **DCC-GARCH** (Engle, 2002) — bivariate and trivariate specifications
3. **Diebold-Yilmaz (2012)** spillover index with generalised FEVD (Koop, Pesaran & Potter, 1996; Pesaran & Shin, 1998)
4. **Asymmetric spillover decomposition** — bad vs good volatility (Barunik, Kocenda & Vacha, 2017)
5. **Rolling-window estimation** (60-month window)

All models are implemented from scratch using `scipy.optimize` (BFGS/L-BFGS-B maximum likelihood). The code does not depend on the `arch` or `rugarch` packages.

## Citation

```bibtex
@article{lin2026volatility,
  title   = {Volatility Spillovers in Macao's Gaming Economy before and after COVID-19},
  author  = {Lin, Yihuan},
  journal = {International Journal of Economic Performance},
  volume  = {9},
  number  = {1},
  pages   = {113--128},
  year    = {2026}
}
```

## License

Code: MIT License
Data: Sourced from the Statistics and Census Service (DSEC), Macao SAR Government. Reuse with attribution.

---

*Data sourced from the Statistics and Census Service, Government of the Macao Special Administrative Region, People's Republic of China (https://www.dsec.gov.mo).*

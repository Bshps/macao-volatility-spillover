# Spillover Turun Naik dalam Ekonomi Perjudian Makau Sebelum dan Selepas COVID-19

**Bahasa lain:** [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文（香港）](README.zh-HK.md) | [العربية](README.ar.md) | [Melayu](README.ms.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20523832.svg)](https://doi.org/10.5281/zenodo.20523832)

**Pengarang**: Lin Yihuan (ORCID: [0000-0002-8391-7732](https://orcid.org/0000-0002-8391-7732))
**Afiliasi**: Sekolah Perniagaan dan Ekonomi, Universiti Putra Malaysia
**Jurnal**: *International Journal of Economic Performance* — ISSN 2661-7161 — Jilid 9, Nombor 1, ms. 113–128 (2026)
**Artikel diterbitkan**: [https://asjp.cerist.dz/en/article/299154](https://asjp.cerist.dz/en/article/299154)

---

## Gambaran Keseluruhan

Kajian ini meneliti hubungan dinamik antara ketibaan pelawat bermalam dan hasil perjudian di Makau menggunakan **GJR-GARCH**, **Korelasi Bersyarat Dinamik (DCC)**, dan rangka kerja **Indeks Spillover Diebold-Yilmaz**. Dapatan utama termasuk:

- Korelasi DCC antara pelawat bermalam dan hasil perjudian adalah positif sebelum COVID-19 (ρ̄=0.160), runtuh semasa penutupan sempadan (ρ̄=0.009), dan kekal negatif selepas 2022 (ρ̄=-0.071)
- Spillover turun naik buruk (30.6%) jauh melebihi spillover turun naik baik (8.2%), jurang sebanyak 22.4 mata peratusan
- Kedua-dua andaian kestabilan dan simetri gagal apabila dinilai ke atas keseluruhan sampel

## Data

| Fail | Penerangan | Sumber |
|------|------------|--------|
| `dsec.xlsx` | Ketibaan pelawat bermalam dan harian bulanan | DSEC Makau SAR (https://www.dsec.gov.mo) |
| `gaming_revenue.csv` | Hasil cukai perjudian bulanan (juta MOP) | DSEC Makau SAR |
| | Tempoh sampel: Januari 2008 – Julai 2025 | |

## Keperluan

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

Diuji dengan Python 3.12.

## Pembiakan Semula

```bash
# Langkah 1: Jalankan analisis utama (GJR-GARCH, DCC, Diebold-Yilmaz)
python paper2_analysis.py

# Langkah 2: Hasilkan angka berkualiti penerbitan
python make_figures_p2.py
```

Semua output disimpan dalam direktori `output/`.

## Petikan

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

## Lesen

Kod: Lesen MIT
Data: Bersumber daripada Perkhidmatan Statistik dan Banci (DSEC), Kerajaan Wilayah Pentadbiran Khas Makau. Guna semula dengan atribusi.

---

*Sumber data: Perkhidmatan Statistik dan Banci, Kerajaan Wilayah Pentadbiran Khas Makau, Republik Rakyat China (https://www.dsec.gov.mo)*

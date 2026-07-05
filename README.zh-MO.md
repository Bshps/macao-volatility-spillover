# 新冠疫情前後澳門博彩經濟的波動率溢出效應

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20523832.svg)](https://doi.org/10.5281/zenodo.20523832)

**作者**: Lin Yihuan (ORCID: [0000-0002-8391-7732](https://orcid.org/0000-0002-8391-7732))
**所屬單位**: 博特拉大學經濟與管理學院 (School of Business and Economics, Universiti Putra Malaysia)
**期刊**: *International Journal of Economic Performance* — ISSN 2661-7161 — 第 9 卷第 1 期, pp. 113–128 (2026)
**已發佈文章**: [https://asjp.cerist.dz/en/article/299154](https://asjp.cerist.dz/en/article/299154)

---

## 概述

本研究使用 **GJR-GARCH**、**動態條件相關（DCC）** 及 **Diebold-Yilmaz 溢出指數框架**，分析澳門過夜旅客人次與博彩收入之間的動態關係。主要發現包括：

- 過夜旅客與博彩收入的 DCC 相關性在疫情前為正（ρ̄=0.160），封關期間趨近於零（ρ̄=0.009），2022 年後持續轉為負值（ρ̄=-0.071）
- 壞波動溢出（30.6%）大幅超過好波動溢出（8.2%），差距達 22.4 個百分點
- 穩定性與對稱性這兩個標準建模假設在完整樣本中均不成立

## 數據

| 檔案 | 說明 | 來源 |
|------|------|------|
| `dsec.xlsx` | 月度過夜及即日旅客人次 | 澳門統計暨普查局 (https://www.dsec.gov.mo) |
| `gaming_revenue.csv` | 月度博彩稅收（百萬澳門元） | 澳門統計暨普查局 |
| | 樣本區間: 2008 年 1 月 – 2025 年 7 月 | |

## 運行環境

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

已在 Python 3.12 下測試。

## 複現步驟

```bash
# 第一步：運行主分析（GJR-GARCH、DCC、Diebold-Yilmaz 溢出）
python paper2_analysis.py

# 第二步：生成期刊級圖表
python make_figures_p2.py
```

所有輸出保存在 `output/` 目錄下，包括描述性統計、GJR-GARCH 參數估計、條件波動率、DCC 動態相關性、滾動溢出指數、不對稱溢出分解等。

## 引用

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

## 許可協議

代碼：MIT License
數據：來源於澳門特別行政區政府統計暨普查局（DSEC），引用時請註明出處。

---

*數據來源：中華人民共和國澳門特別行政區政府統計暨普查局 (https://www.dsec.gov.mo)*

# 新冠疫情前后澳门博彩经济的波动率溢出效应

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20523832.svg)](https://doi.org/10.5281/zenodo.20523832)

**作者**: Lin Yihuan (ORCID: [0000-0002-8391-7732](https://orcid.org/0000-0002-8391-7732))
**单位**: 博特拉大学经济与管理学院 (School of Business and Economics, Universiti Putra Malaysia)
**期刊**: *International Journal of Economic Performance* — ISSN 2661-7161 — 第 9 卷第 1 期, pp. 113–128 (2026)
**已发表文章**: [https://asjp.cerist.dz/en/article/299154](https://asjp.cerist.dz/en/article/299154)

---

## 概述

本研究使用 **GJR-GARCH**、**动态条件相关（DCC）** 和 **Diebold-Yilmaz 溢出指数框架**，分析了澳门过夜旅客人次与博彩收入之间的动态关系。主要发现包括：

- 过夜旅客与博彩收入的 DCC 相关性在疫情前为正（ρ̄=0.160），封关期间趋近于零（ρ̄=0.009），2022 年后持续转为负值（ρ̄=-0.071）
- 坏波动溢出（30.6%）大幅超过好波动溢出（8.2%），差距达 22.4 个百分点
- 稳定性和对称性这两个标准建模假设在全样本中均不成立

## 数据

| 文件 | 说明 | 来源 |
|------|------|------|
| `dsec.xlsx` | 月度过夜及即日旅客人次 | 澳门统计暨普查局 (https://www.dsec.gov.mo) |
| `gaming_revenue.csv` | 月度博彩税收（百万澳门元） | 澳门统计暨普查局 |
| | 样本区间: 2008 年 1 月 – 2025 年 7 月 | |

## 运行环境

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

已在 Python 3.12 下测试。

## 复现步骤

```bash
# 第一步：运行主分析（GJR-GARCH、DCC、Diebold-Yilmaz 溢出）
python paper2_analysis.py

# 第二步：生成论文级图表
python make_figures_p2.py
```

所有输出保存在 `output/` 目录下，包括描述性统计、GJR-GARCH 参数估计、条件波动率、DCC 动态相关性、滚动溢出指数、不对称溢出分解等。

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

## 许可协议

代码：MIT License
数据：来源于澳门特别行政区政府统计暨普查局（DSEC），引用时请注明出处。

---

*数据来源：中华人民共和国澳门特别行政区政府统计暨普查局 (https://www.dsec.gov.mo)*

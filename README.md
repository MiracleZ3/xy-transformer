# xy-transformer

理财产品资金流预测（Wealth-Management Product Cash-Flow Forecasting）。

基于 PANTHER（*Generative Pretraining Beyond Language for Sequential User Behavior Modeling*,
NeurIPS 2025）的「结构化分词 + 行为序列生成式预训练 + SPRM 周期 motif + 产品画像对比学习」框架，
把序列主体从「用户」迁移到「理财产品」，对每个产品预测未来 1 / 7 / 30 天的**申购额与赎回额**，
再合成净现金流。

## 当前阶段

数据 schema 与分词可行性的验证脚本（已用真实公开数据跑通），核心模型代码待后续接入。

## 目录结构

```
├── build_demo_data.py        把公开数据集映射到建模 schema（流水宽表 + 产品×日聚合表）
├── tokenize_dryrun.py        PANTHER Eq.(4) 结构化分词 + 词表裁剪 + 序列打包 dry-run
├── data_source/              原始样本数据（<10MB 切片）
│   ├── Online_Retail_sample.xlsx   UCI Online Retail 切片（~4MB / 80k 行）
│   └── creditcard_sample.csv       ULB 信用卡切片（~6MB / 12k 行，备份路径）
└── data_sample/              脚本输出（txns / daily_p parquet + csv 采样）
```

## 快速开始

```bash
pip install pandas numpy openpyxl scikit-learn
python3 build_demo_data.py      # 生成 data_sample/*.parquet
python3 tokenize_dryrun.py      # 分词覆盖率与序列打包报告
```

无需 GPU / torch，全流程 < 30 秒。

## 数据说明

仓内只携带 <10MB 的切片，确保开箱即用。完整数据请按需自行下载到 `data_source/`：

| 文件 | 完整数据下载地址 |
|---|---|
| `Online Retail.xlsx`（~24MB） | https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx |
| `creditcard.csv`（~144MB） | https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv |

> ⚠️ 切片仅为流程验证用途：UCI 是电商小额（£1~£10²）数据，**金额量级与申赎比例不代表真实理财产品**。
> 真实理财数据接入后需重新校准分布。

## License

MIT

<!-- ci-verify: credential helper end-to-end check -->

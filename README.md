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
├── data_source/              原始数据集（不进仓，本地按需放置）
└── data_sample/              50 行最小样例（parquet + csv），开箱即用
    ├── txns.parquet / txns_sample.csv       UCI 流水宽表（50 行）
    ├── daily_p.parquet / daily_sample.csv   产品×日聚合（50 行）
    ├── txns_real.parquet / txns_real_sample.csv  ETF 真实流水（50 行）
    ├── xy_txns.parquet                      真实 schema 申赎流水（50 行）
    ├── fact_txn.parquet                     客户级流水（50 行）
    └── dim_product/dim_customer.parquet     产品/客户维度词典（完整保留）
```

## 快速开始

```bash
pip install pandas numpy openpyxl scikit-learn
python3 build_demo_data.py      # 生成 data_sample/*.parquet
python3 tokenize_dryrun.py      # 分词覆盖率与序列打包报告
```

无需 GPU / torch，全流程 < 30 秒。

## 数据说明

仓内 **只携带每类 50 行的最小样例**（`data_sample/`，约 80KB），保证开箱即用且仓库轻量。
维表（`dim_product` / `dim_customer`）与产品画像（`xy_product_meta.json`）为结构词典，按真实规模保留。
需要完整数据时，按需下载到本地 `data_source/`：

| 文件 | 完整数据下载地址 |
|---|---|
| `Online Retail.xlsx`（~24MB） | https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx |
| `creditcard.csv`（~144MB） | https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv |

> `data_source/` 全量及切片均已通过 `.gitignore` 排除，不进仓。
> 模拟类数据（`xy_txns` / `fact_txn`）可用 `simulate_xy_real_schema.py` / `simulate_customer_txns.py` 重新生成全量版本。

> ⚠️ 样例仅为流程验证用途：UCI 是电商小额（£1~£10²）数据，**金额量级与申赎比例不代表真实理财产品**。
> 真实理财数据接入后需重新校准分布。

## License

MIT

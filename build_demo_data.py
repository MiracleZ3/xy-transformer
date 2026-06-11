"""
build_demo_data.py
==================

把公开数据集构造成「理财产品资金流预测」建模所需的样本 schema，
用来 **验证方案可行性**（端到端打通：原始数据 → 流水宽表 → 序列 → 标签）。

⮕ 输出 schema 严格对齐 docs/02 §5 的两张表：
    data_sample/txns.parquet        行为序列宽表（每行一笔流水）
    data_sample/daily_p.parquet     产品 × 日 聚合表（回归标签 + 上下文）
    data_sample/txns_sample.csv     同上的 CSV 采样，供人肉检查

数据来源（选用其一，默认 A）：
    A. UCI Online Retail（真实电商流水，含正负数量端天然表达"申/赎"）
       https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
       映射:  StockCode  → product_id
              InvoiceDate→ ts
              Quantity>0 → 方向=申(申购), Quantity<0 → 方向=赎(赎回)
              Quantity*UnitPrice → amount
              CustomerID → cust_id
              Country / Description 关键词 → 人工补造 product_type / risk_level（缺数据时的合理归并）
    B. ULB 信用卡欺诈（备份，PCA 特征，仅当 A 不可用时退化为合成补充）
       https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
       映射:  Class=1 视为赎回冲击，其余视为申购；Amount→amount。仅做最小演示。

设计取舍：
- 真实理财数据字段缺失时（MUST 字段如 risk_level/yield_rate），用 **确定性的、可复现的规则**
  从 product_id 派生，并在 docs/02 标注为「占位/合成」，避免给业务读数造成混淆。
- 所有金额做 log1p -> 回归头以 log1p(金额) 为目标（docs/01 §7）。
- 申/赎 **分别** 金额分桶（docs/02 §4.2 / RETRAIN.md §7 强制要求），用 quantile。

运行：
    python3 build_demo_data.py                 # 用 data_source/online_retail.xlsx
    python3 build_demo_data.py --rows 200000   # 限制行数，更快验证
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 可复现随机数（不调用系统时间/随机源，保证每次运行结果一致） ----
_RNG = np.random.default_rng(seed=20260611)

# ---- 路径 ----
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "data_source"
OUT_DIR = HERE / "data_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_XLSX_CANDIDATES = [
    SRC_DIR / "Online Retail.xlsx",          # 真实 UCI 全量原始文件（带空格，~24MB）
    SRC_DIR / "online_retail.xlsx",          # 历史命名兼容
    SRC_DIR / "Online_Retail_sample.xlsx",   # GitHub 公开仓仓内切片（~4MB / 80k 行）
]
SRC_XLSX = next((p for p in SRC_XLSX_CANDIDATES if p.exists()), SRC_XLSX_CANDIDATES[0])
SRC_CSV_BAK = SRC_DIR / "creditcard.csv"

# 公开仓内的数据下载地址（用于补齐 <10MB 切片之外的完整数据）
DATA_URLS = {
    "online_retail_full": "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "creditcard_full":    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
}

RAW_TEXT_FIELDS = [
    "product_id", "ts", "direction", "status", "amount",
    "amount_log1p", "amount_bin", "aum_after",
    "dow", "dom", "hour_bin", "is_month_end", "is_quarter_end",
    "dt_prev_sec", "product_type", "risk_level", "term_type",
    "channel", "cust_type", "yield_rate",
]


def _hash_to_buckets(key: str, *, n: int, salt: str = "") -> int:
    """确定性 hash → bucket，保证同一 product_id 多次出现映射一致，且可复现。"""
    h = hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(n, 1)


def load_online_retail(limit: int | None = None) -> pd.DataFrame:
    """优先用真实 UCI Online Retail（全量或仓内切片均可）；都不可用时退化到可复现合成数据。"""
    if SRC_XLSX.exists() and _looks_complete_xlsx(SRC_XLSX):
        # sheet 在 UCI 全量是 "Online Retail"；切片文件用首表。
        try:
            xl = pd.ExcelFile(SRC_XLSX)
            sheet = "Online Retail" if "Online Retail" in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(SRC_XLSX, sheet_name=sheet)
            if limit:
                df = df.head(limit)
            print(f"使用真实数据源: {SRC_XLSX.name} ({len(df):,} 行)")
            return df
        except Exception as e:
            print(f"[warn] 读取真实数据失败 ({type(e).__name__})，降级到合成数据。")

    if limit is None:
        limit = 60_000  # 合成默认量级，足够走通流程
    print(f"[info] 使用合成数据（{limit:,} 行）—— 模拟理财申赎流水的真实统计特征。")
    return synthetic_online_retail_frame(n_rows=limit)


def _looks_complete_xlsx(path: Path) -> bool:
    """任一有效的 xlsx（全量 24MB 或仓内切片 ~4MB）都接受为真实数据入口。"""
    return path.exists() and path.stat().st_size > 1_000_000


def synthetic_online_retail_frame(n_rows: int = 60_000) -> pd.DataFrame:
    """
    用可复现随机数生成「形似」UCI Online Retail 的流水，使下游 map_to_schema 流程一致。

    注入的真实理财状态特征（docs/02 §4 / 文献常识）：
      - 月末/季末赎回潮（is_month_end / is_quarter_end → 赎回概率 & 金额上调）
      - 工作日 vs 周末（周末申赎更少）
      - 申/赎金额长尾（lognormal，按产品档分异）
      - 产品数量 ~ 数百（贴近真实产品池规模，远小于支付场景）
    """
    n_products = 220
    n_cust = 2_000
    base_date = np.datetime64("2024-01-01")
    n_days = 540

    rows = []
    # 每个产品一个发行档 + 风险，决定其金额量级与活跃度
    # mu_log（log 空间）：¥1k≈6.9, ¥10k≈9.2, ¥100k≈11.5, ¥1M≈13.8；用 6.0~11.5 让中位金额落入 ¥1k~¥1M
    product_meta = {
        f"P{pid:04d}": {
            "mu_log": float(_RNG.uniform(6.0, 11.5)),
            "sigma": float(_RNG.uniform(0.7, 1.5)),  # 收敛尾部；避免 e^(11+6) 量级
            "rate": float(_RNG.uniform(20, 180)),    # 月均笔数
        }
        for pid in range(n_products)
    }
    for pid_str, meta in product_meta.items():
        pid_num = int(pid_str[1:])  # "P0123" -> 123
        n_txn = int(meta["rate"] * (n_days / 30))
        n_txn = max(50, min(n_txn, 4_000))  # 单产品上限，使小样本里也能覆盖更多产品
        # 抽时间：周末 0.4x，月末/季末 1.6x（先把秒拍在 [0, n_days*86400)）
        day_idx = _RNG.integers(0, n_days, size=n_txn)
        dow = (day_idx + 1) % 7  # 2024-01-01 是周一(0)
        # 工作日权重
        w = np.where((dow >= 1) & (dow <= 5), 1.0, 0.4)
        # 月末/季末加成
        dom = (day_idx % 30) + 1
        is_me = (dom >= 27)
        is_qe = is_me & (((day_idx // 90) * 90 + 89 - (day_idx % 90)) < 5)
        w = np.where(is_me, w * 1.6, w)
        w = np.where(is_qe, w * 1.3, w)
        w = w / w.sum()
        # 重采样：weighted 选择
        idx_sampled = _RNG.choice(n_txn, size=n_txn, replace=True, p=w)
        cols = []
        for i in idx_sampled:
            d = int(day_idx[i])
            secs = int(_RNG.integers(8 * 3600, 22 * 3600))
            # 先构 datetime64 秒粒度的日期，再叠加秒数偏移，避免 D vs s 类型冲突
            ts = (base_date + np.timedelta64(d, "D")).astype("datetime64[s]") \
                + np.timedelta64(secs, "s")
            month_end = is_me[i]
            # 方向：赎回基础概率 0.35；月末 → 0.55
            p_red = 0.55 if month_end else 0.35
            direction = 1 if _RNG.random() < p_red else 0
            # 金额 lognormal：exp(N(mu, sigma)) 直接是人民币金额（¥1k~¥1M 量级）
            mu = meta["mu_log"] - (0.4 if direction == 1 else 0.0)
            amount = float(_RNG.lognormal(mu, meta["sigma"]))
            amount = float(min(amount, 1_000_000.0))  # winsorize 极端尾部，¥1M 上限
            # 在 Online-Retail 两列口径下：amount = Quantity * UnitPrice。
            # 我们让 Quantity = ±1（一份成交单价 = 金额），UnitPrice = 金额，保证相乘恰好等于金额本身，
            # 与下游 map_to_schema 的 amount = |Quantity| * UnitPrice 口径自洽。
            unit_price = round(amount, 2)
            qty = (-1 if direction == 1 else 1)
            status_failed = _RNG.random() < 0.03
            inv_ts = int(ts.astype("datetime64[s]").astype("int64"))
            prefix = "C" if status_failed else "OK"
            invoice_no = "{}_{:010d}_{}_{}".format(prefix, inv_ts, i % 9, pid_num % 7)
            cols.append((
                invoice_no, ts, pid_str, qty, unit_price,
                int(_RNG.integers(1, n_cust)), "United Kingdom",
            ))
        rows.extend(cols)
        if len(rows) >= n_rows:
            break

    df = pd.DataFrame(rows, columns=[
        "InvoiceNo", "InvoiceDate", "StockCode", "Quantity", "UnitPrice",
        "CustomerID", "Country",
    ]).head(n_rows)
    return df


def map_to_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """UCI Online Retail → docs/02 §5 完整 schema（缺字段用可复现规则补足）。"""
    df = df_raw.copy()

    # 1. 基本清洗
    df = df.dropna(subset=["CustomerID", "StockCode", "InvoiceDate", "Quantity", "UnitPrice"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"])
    df = df[(df["Quantity"] != 0) & (df["UnitPrice"] >= 0)]

    # 2. 映射到 schema
    df["product_id"] = df["StockCode"].astype(str)
    df["cust_id"] = df["CustomerID"].astype("Int64")
    # ts = 当日 0 点 + InvoiceDate 内的秒数；保留"日期+当日秒数"语义（与用户给的原始字段一致）
    dt = pd.to_datetime(df["InvoiceDate"])
    df["ts"] = dt.astype("int64") // 10**9
    df["direction"] = np.where(df["Quantity"] >= 0, 0, 1)  # 0=申 1=赎
    # 状态：UCI 没有失败概念；用 InvoiceNo 以 C 开头(取消) 标记为"失败反向记录"，其余成功
    df["status"] = np.where(df["InvoiceNo"].astype(str).str.startswith("C"), 0, 1)
    df["amount"] = (df["Quantity"].abs() * df["UnitPrice"]).astype("float64")

    # 3. 缺失 MUST 字段：确定性占位（标记为合成，业务上不可直接信）
    df["product_type"] = df["product_id"].map(
        lambda k: _hash_to_buckets(k, n=6, salt="ptype")   # 0..5: 货基/债基/股基/混合/固收+/其他
    ).astype("int8")
    df["risk_level"] = df["product_type"].map(
        lambda pt: min(5, pt)                                # 货基R1, ... 股基R4，用作示例占位
    ).astype("int8") + 1
    df["term_type"] = df["product_id"].map(
        lambda k: _hash_to_buckets(k, n=4, salt="term")     # 0..3: T+0/T+1/7d/封闭
    ).astype("int8")
    df["channel"] = df["product_id"].map(
        lambda k: _hash_to_buckets(k, n=4, salt="ch")       # APP/网银/柜台/第三方
    ).astype("int8")
    df["cust_type"] = df["cust_id"].map(
        # 机构客户占比偏高占位
        lambda c: _hash_to_buckets(str(c), n=3, salt="cust")  # 0=个人 1=机构 2=同业
    ).astype("int8")
    # 收益率：按 risk 等级线性合成（高 R 高收益），加 product 哈希扰动，便于学收益拐点信号
    base = {1: 0.022, 2: 0.030, 3: 0.050, 4: 0.080, 5: 0.120}
    df["yield_rate"] = df["risk_level"].map(base).astype("float32")
    df["yield_rate"] = df["yield_rate"] + df["product_id"].map(
        lambda k: (_hash_to_buckets(k, n=200, salt="y") - 100) / 2000.0
    )

    # 4. 派生时间特征（docs/02 §4.1）
    df["date"] = dt.dt.normalize()
    df["dow"] = dt.dt.dayofweek.astype("int8")               # 0=Mon
    df["dom"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["hour_bin"] = (df["hour"] // 3).clip(0, 7).astype("int8")   # 3小时一桶, 0..7
    df["is_month_end"] = dt.dt.is_month_end.astype("int8")
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")
    # 距同产品上一笔间隔
    df = df.sort_values(["product_id", "ts"])
    df["dt_prev_sec"] = (
        df.groupby("product_id")["ts"].diff().fillna(0).astype("int64")
    )
    # 赎回一笔后常有相同金额的另一笔反向（保守处理），winsorize 极端 dt
    df["dt_prev_sec"] = df["dt_prev_sec"].clip(0, 30 * 24 * 3600).astype("int32")

    # 5. 金额分桶 —— 必须按 direction 分别 quantile（docs/02 §4.2 / RETRAIN §7）
    df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0)).astype("float32")
    df["amount_bin"] = -1
    for d in (0, 1):
        mask = df["direction"] == d
        if mask.sum() >= 32:
            df.loc[mask, "amount_bin"] = pd.qcut(
                df.loc[mask, "amount_log1p"], q=24, labels=False, duplicates="drop"
            ).astype("int16")
        else:
            df.loc[mask, "amount_bin"] = 0
    df["amount_bin"] = df["amount_bin"].astype("int16")

    # 6. 产品剩余金额（AUM）：在产品时序内累加，赎回记负（保留用户标"剩余金额"语义）
    signed = df["amount"].where(df["direction"] == 0, -df["amount"])
    df["aum_after"] = (
        df.assign(_sgn=signed)
        .sort_values(["product_id", "ts"])
        .groupby("product_id")["_sgn"].cumsum()
    )
    # aum 不应负 -> clip 0（真实剩余金额不会负；这里为占位口径）
    df["aum_after"] = df["aum_after"].clip(lower=0.0).astype("float64")

    # 7. 行 ID（对齐参考工程的 __row_id__ 机制，让标签在抽嵌入 reorder 后可回溯）
    df = df.reset_index(drop=True)
    df["__row_id__"] = np.arange(len(df), dtype="int64")
    # 仅保留 schema 列，丢弃原始 leak（InvoiceNo / Description / CustomerID / date / hour 等）
    schema_cols = [
        "product_id", "cust_id", "ts", "date", "direction", "status",
        "amount", "amount_log1p", "amount_bin", "aum_after",
        "dow", "dom", "hour", "hour_bin", "is_month_end", "is_quarter_end",
        "dt_prev_sec", "product_type", "risk_level", "term_type",
        "channel", "cust_type", "yield_rate", "__row_id__",
    ]
    df = df[[c for c in schema_cols if c in df.columns]]
    return df.sort_values(["product_id", "ts"]).reset_index(drop=True)


def build_daily(txns: pd.DataFrame) -> pd.DataFrame:
    """把 txns 聚合成 产品×日 表（docs/02 §5.2）。"""
    g = txns.groupby(["product_id", "date", "direction"], as_index=False)["amount"].sum()
    piv = g.pivot_table(
        index=["product_id", "date"], columns="direction", values="amount",
        fill_value=0.0,
    ).rename(columns={0: "purchase_amt", 1: "redemption_amt"})
    daily = piv.reset_index()
    daily.columns.name = None
    daily["net_amt"] = daily["purchase_amt"] - daily["redemption_amt"]
    # 日末日 AUM（产品内累加）
    aum = txns.sort_values(["product_id", "ts"]).groupby(
        ["product_id", "date"], as_index=False
    )["aum_after"].last().rename(columns={"aum_after": "aum_eod"})
    daily = daily.merge(aum, on=["product_id", "date"], how="left")
    # 占位行情/客户结构（与 txns 表同口）
    daily["yield_rate"] = daily["product_id"].map(
        lambda k: 0.03 + (_hash_to_buckets(k, n=200, salt="y") - 100) / 2000.0
    ).astype("float32")
    daily["__row_id__"] = np.arange(len(daily), dtype="int64")
    return daily


def report(out_dir: Path, txns: pd.DataFrame, daily: pd.DataFrame) -> None:
    print("\n===== txns 宽表 =====")
    print("rows:", len(txns), "| products:", txns["product_id"].nunique())
    print("dir 分布：", txns["direction"].value_counts().to_dict())
    print("status 分布：", txns["status"].value_counts().to_dict())
    print("amount 描述:\n", txns["amount"].describe().round(2).to_dict())
    print("amount_log1p 范围:", round(txns["amount_log1p"].min(), 3), "~", round(txns["amount_log1p"].max(), 3))
    print("amount_bin (0/1 dir) max:", txns["amount_bin"].max())
    print("\n===== daily 产品×日 =====")
    print("rows:", len(daily), "| products:", daily["product_id"].nunique())
    print("purchase_amt>0 比例:", round((daily["purchase_amt"]>0).mean(), 3))
    print("redemption_amt>0 比例:", round((daily["redemption_amt"]>0).mean(), 3))
    # 简单的 multi-horizon 标签自检：每个产品每个 date 存在 T+1/T+7/T+30 的真值可对齐
    for h in (1, 7, 30):
        lbl = daily.set_index([pd.to_datetime(daily['date']), 'product_id'])
        nxt = lbl['net_amt'].groupby('product_id').shift(-h)
        cov = nxt.notna().mean()
        print(f"  horizon=+{h}d 真值覆盖率: {cov:.3f}")
    print("\n输出文件：")
    print(" -", out_dir / "txns.parquet")
    print(" -", out_dir / "daily_p.parquet")
    print(" -", out_dir / "txns_sample.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="限制原始行数，加快验证")
    ap.add_argument("--csv-sample", type=int, default=2000, help="CSV 采样输出行数")
    args = ap.parse_args(argv)

    try:
        raw = load_online_retail(limit=args.rows)
    except FileNotFoundError as e:
        print("ERROR:", e, file=sys.stderr)
        return 2

    print(f"读取原始 {len(raw):,} 行 -> 映射 schema ...")
    txns = map_to_schema(raw)
    daily = build_daily(txns)

    # 持久化
    txns_parquet = OUT_DIR / "txns.parquet"
    daily_parquet = OUT_DIR / "daily_p.parquet"
    txns.to_parquet(txns_parquet, index=False)
    daily.to_parquet(daily_parquet, index=False)

    keep = [c for c in RAW_TEXT_FIELDS if c in txns.columns] + ["__row_id__"]
    txns[keep].head(args.csv_sample).to_csv(OUT_DIR / "txns_sample.csv", index=False)
    daily.head(args.csv_sample).to_csv(OUT_DIR / "daily_sample.csv", index=False)

    report(OUT_DIR, txns, daily)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

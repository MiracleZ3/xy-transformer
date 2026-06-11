"""
preprocess.py
=============

数据工程脚本（实现 docs/05 全部步骤）。把原始三层（产品维/客户维/逐笔流水）加工成
训练可用的样本：流水宽表 + 多窗口标签表 + 分桶器状态。

数据流：
  data_sample/  (simulate 或真实的 data_raw/)
    dim_product.parquet
    dim_customer.parquet
    fact_txn.parquet
        │
        ▼ preprocess.py
  data_processed/
    txns_{train,val,test}.parquet
    daily_{train,val,test}.parquet
    tokenizer_state.json

功能映射 docs/05 章节：
  §1-§2 字段处理/选择        → join_dims() + select_drop_fields()
  §3   分桶（含分方向金额）   → build_amount_bin() / build_dt_bin() / build_hour_bin() / build_min_purch_bin()
  §4   防泄漏                → temporal_split() + 持仓 T-1 滞后 (在 ψ 阶段，本脚本打标)
  §5   多窗口标签 (1/7/30d)  → build_daily_labels()
  §6   时间切分              → temporal_split()
  §7   数据质量门禁          → run_quality_gates()

设计原则：脚本对真实/模拟数据完全无感知——只要列名对齐，跑同一套逻辑。
文档详细决策见 docs/05-data-engineering.md。

运行：
  python3 preprocess.py                           # 默认从 data_sample/，输出 data_processed/
  python3 preprocess.py --input data_raw --output data_processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "data_sample"
DEFAULT_OUT = HERE / "data_processed"

# ---- 全局配置 ----
AMOUNT_BINS = 16        # 金额桶（对齐 PANTHER Eq.4 第2维）
DT_BINS = 8             # 时间间隔桶
HOUR_BINS = 8           # 时刻桶（3小时一桶）
MIN_PURCH_BINS = 4
HORIZONS = [1, 7, 30]   # 多窗口预测
GAP_DAYS = 7            # train/val 边界缓冲
TRAIN_RATIO, VAL_RATIO = 0.80, 0.10


# ============================================================
# §1 三表 JOIN + §2 字段选择
# ============================================================
def join_dims(fact: pd.DataFrame, dim_product: pd.DataFrame,
              dim_customer: pd.DataFrame) -> pd.DataFrame:
    """流水 join 产品/客户维度，补全画像字段。"""
    n0 = len(fact)
    df = fact.merge(dim_product, on="product_id", how="left")
    df = df.merge(dim_customer, on="cust_id_hash", how="left")
    dropped = n0 - len(df[df["product_type"].notna() & df["cust_type"].notna()])
    if dropped / n0 > 0.05:
        print(f"  [WARN] join 后丢失 {dropped}/{n0} ({dropped/n0:.1%}) 行：产品/客户维度缺失 >5%")
    return df


def select_drop_fields(df: pd.DataFrame) -> pd.DataFrame:
    """§1.1 product_type_name 仅审计用，丢弃。"""
    drop = [c for c in ["product_type_name"] if c in df.columns]
    return df.drop(columns=drop, errors="ignore")


# ============================================================
# §2 清洗
# ============================================================
def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """§2 + §7：去空/0金额，校正持仓非负。"""
    df = df.copy()
    # 金额空 / 0
    before = len(df)
    df = df[df["amount"].notna() & (df["amount"] > 0)].copy()
    if before - len(df):
        print(f"  丢弃空/0金额: {before - len(df)} 行")
    # 持仓非负
    df["aum_after"] = df["aum_after"].clip(lower=0.0)
    # monotonic ts per (cust,product) 排序保证
    df["date"] = pd.to_datetime(df["txn_ts"], unit="s").dt.normalize()
    df = df.sort_values(["cust_id_hash", "product_id", "txn_ts"]).reset_index(drop=True)
    return df


# ============================================================
# §3 时间派生
# ============================================================
def derive_time(df: pd.DataFrame) -> pd.DataFrame:
    """§1.4：dow/dom/hour_bin/is_month_end/quarter_end/dt_prev_sec。"""
    df = df.copy()
    dt = pd.to_datetime(df["txn_ts"], unit="s")
    df["dow"] = dt.dt.dayofweek.astype("int8")
    df["dom"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["hour_bin"] = (df["hour"] // 3).clip(0, HOUR_BINS - 1).astype("int8")
    df["is_month_end"] = dt.dt.is_month_end.astype("int8")
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")
    # dt_prev_sec：同 (cust,product) 上一笔间隔
    df = df.sort_values(["cust_id_hash", "product_id", "txn_ts"])
    df["dt_prev_sec"] = (
        df.groupby(["cust_id_hash", "product_id"])["txn_ts"]
        .diff().fillna(0).astype("int64").clip(0, 30 * 86400).astype("int32")
    )
    return df


# ============================================================
# §3 分桶
# ============================================================
def fit_amount_bin(df_train: pd.DataFrame, bins: int = AMOUNT_BINS) -> dict:
    """§3.1 在 train 段按 direction 分别 quantile fit，返回边界。"""
    edges = {}
    for d, name in [(0, "purchase"), (1, "redemption")]:
        sub = df_train.loc[df_train["direction"] == d, "amount_log1p"]
        if len(sub) < bins * 2:
            print(f"  [WARN] direction={d} 样本 {len(sub)} < {bins*2}，分桶可能不稳定")
        _, e = pd.qcut(sub, q=bins, labels=False, retbins=True, duplicates="drop")
        edges[f"bin_edges_{name}"] = [float(x) for x in e.tolist()]
    return {"amount_bins": bins, **edges}


def apply_amount_bin(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """用 fit 出的边界 transform 到任意 split。"""
    df = df.copy()
    df["amount_bin"] = -1
    for d, name in [(0, "purchase"), (1, "redemption")]:
        m = df["direction"] == d
        edges = state[f"bin_edges_{name}"]
        df.loc[m, "amount_bin"] = pd.cut(
            df.loc[m, "amount_log1p"], bins=edges, labels=False,
            include_lowest=True,
        )
    df["amount_bin"] = df["amount_bin"].fillna(-1).astype("int16")
    return df


def fit_dt_bin(df_train: pd.DataFrame, bins: int = DT_BINS) -> dict:
    sub = np.log1p(df_train["dt_prev_sec"].clip(lower=0))
    _, e = pd.qcut(sub, q=bins, labels=False, retbins=True, duplicates="drop")
    return {"dt_bins": bins, "edges": [float(x) for x in e.tolist()]}


def apply_dt_bin(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = df.copy()
    sub = np.log1p(df["dt_prev_sec"].clip(lower=0))
    df["dt_bin"] = pd.cut(sub, bins=state["edges"], labels=False, include_lowest=True)
    df["dt_bin"] = df["dt_bin"].fillna(-1).astype("int16")
    return df


def build_min_purch_bin(df: pd.DataFrame, bins: int = MIN_PURCH_BINS) -> pd.DataFrame:
    """§3.4：起购金额分桶（无需防泄漏，产品级静态字段）。"""
    df = df.copy()
    if "min_purchase" not in df.columns:
        df["min_purchase_bin"] = 0
        return df
    sub = np.log1p(df["min_purchase"].clip(lower=0))
    df["min_purchase_bin"] = pd.qcut(sub, q=bins, labels=False, duplicates="drop")
    df["min_purchase_bin"] = df["min_purchase_bin"].fillna(0).astype("int8")
    return df


# ============================================================
# §5 daily 聚合 + 多窗口标签
# ============================================================
def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """§5：聚合成功流水到 产品×日 + 多窗口标签。"""
    ok = df[df["status"] == 1].copy()
    ok["date"] = ok["date"]
    # 单产品×日 聚合（客户维度不进标签，因为预测粒度是产品×日）
    g = ok.groupby(["product_id", "date", "direction"], as_index=False)["amount"].sum()
    piv = g.pivot_table(
        index=["product_id", "date"], columns="direction",
        values="amount", fill_value=0.0,
    )
    for d, col in [(0, "purchase_amt"), (1, "redemption_amt")]:
        if d not in piv.columns:
            piv[d] = 0.0
    daily = piv.rename(columns={0: "purchase_amt", 1: "redemption_amt"}).reset_index()
    daily.columns.name = None
    daily["net_amt"] = daily["purchase_amt"] - daily["redemption_amt"]
    # 日末持仓
    aum = df.sort_values(["product_id", "txn_ts"]).groupby(
        ["product_id", "date"], as_index=False
    )["aum_after"].last().rename(columns={"aum_after": "aum_eod"})
    daily = daily.merge(aum, on=["product_id", "date"], how="left")
    # 节假日/失败率上下文（用 T-1 滞后避免泄漏）
    dt = daily["date"]
    daily["is_month_end"] = dt.dt.is_month_end.astype("int8")
    daily["is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")
    # 失败率（产品 7d 滚动，T-1 滞后）
    if "status" in df.columns:
        fr = df.copy()
        fr["fail"] = (fr["status"] == 0).astype(int)
        fr_by_day = fr.groupby(["product_id", pd.to_datetime(fr["txn_ts"], unit="s").dt.normalize()])["fail"].mean()
        fr_lag = fr_by_day.groupby(level=0).shift(1).rolling(7).mean().reset_index()
        fr_lag.columns = ["product_id", "date", "fail_rate_7d"]
        daily = daily.merge(fr_lag, on=["product_id", "date"], how="left")
        daily["fail_rate_7d"] = daily["fail_rate_7d"].fillna(0.0)
    # 占位收益率（真实数据需外联 yield）
    if "yield_rate" not in daily.columns:
        daily["yield_rate"] = 0.0
    # 多窗口标签
    daily = daily.sort_values(["product_id", "date"])
    for h in HORIZONS:
        daily[f"label_purchase_{h}d"] = daily.groupby("product_id")["purchase_amt"].shift(-h)
        daily[f"label_redemption_{h}d"] = daily.groupby("product_id")["redemption_amt"].shift(-h)
    daily["__row_id__"] = np.arange(len(daily), dtype="int64")
    return daily.reset_index(drop=True)


# ============================================================
# §6 时间切分（防泄漏）
# ============================================================
def temporal_split_dates(df: pd.DataFrame, train_ratio=TRAIN_RATIO,
                         val_ratio=VAL_RATIO, gap_days=GAP_DAYS):
    """§6：全局时间切 80/10/10，中间留 gap_days 天防泄漏。"""
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_end = dates.iloc[n_train - 1]
    val_start = train_end + pd.Timedelta(days=gap_days)
    val_end = dates.iloc[n_train + n_val - 1]
    test_start = val_end + pd.Timedelta(days=1)
    return train_end, val_start, val_end, test_start


def assign_split(df: pd.DataFrame, splits) -> pd.DataFrame:
    train_end, val_start, val_end, test_start = splits
    df = df.copy()
    def fn(d):
        if d <= train_end: return "train"
        if val_start <= d <= val_end: return "val"
        if d >= test_start: return "test"
        return "gap"
    df["split"] = df["date"].map(fn)
    return df


# ============================================================
# §7 数据质量门禁
# ============================================================
def run_quality_gates(df: pd.DataFrame, daily: pd.DataFrame) -> None:
    print("\n===== §7 数据质量门禁 =====")
    fail_rate = (df["status"] == 0).mean()
    print(f"失败率: {fail_rate:.2%}  (期望 1%-10%)")
    if not (0.01 <= fail_rate <= 0.50):
        print("  [WARN] 失败率异常")
    red_ratio = (df["direction"] == 1).mean()
    print(f"赎回占比: {red_ratio:.1%}  (期望 20%-80%)")
    if not (0.20 <= red_ratio <= 0.80):
        print("  [WARN] 申赎方向失衡")
    assert (df["aum_after"] >= 0).all(), "持仓出现负值"
    print("✓ 持仓非负")
    seq_len = df.groupby("product_id").size()
    print(f"单产品序列长度: min={seq_len.min()} 中位={int(seq_len.median())} max={seq_len.max()}")
    for h in HORIZONS:
        cov = daily[f"label_purchase_{h}d"].notna().mean()
        print(f"  horizon=+{h}d 真值覆盖率: {cov:.1%}")
    # 切分无重叠
    splits = df["split"].unique()
    print(f"切分集合: {sorted(set(splits))}")


# ============================================================
# 主流程
# ============================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN,
                    help="输入目录（含 dim_product/dim_customer/fact_txn parquet）")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="输出目录")
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    # 读三层
    print(f"读取原始三层 from {args.input} ...")
    fact = pd.read_parquet(args.input / "fact_txn.parquet")
    dim_p = pd.read_parquet(args.input / "dim_product.parquet")
    dim_c = pd.read_parquet(args.input / "dim_customer.parquet")
    print(f"  fact_txn: {len(fact):,} 笔")

    # §1-§2 join + 字段选择 + 清洗
    print("\n§1-2 三表 JOIN + 选字段 ...")
    df = join_dims(fact, dim_p, dim_c)
    df = select_drop_fields(df)
    print("\n§2 清洗 ...")
    df = clean_basic(df)
    df = df.rename(columns={"txn_type": "direction", "txn_status": "status"})

    # §3 派生 + 分桶
    print("\n§3 时间派生 + 分桶 ...")
    df = derive_time(df)
    df["amount_log1p"] = np.log1p(df["amount"]).astype("float32")
    df = build_min_purch_bin(df)

    # §6 先切分 train，§3 在 train 上 fit 分桶器
    print("\n§6 时间全局切分 ...")
    splits = temporal_split_dates(df)
    df = assign_split(df, splits)
    df_train = df[df["split"] == "train"]
    if len(df_train) == 0:
        print("ERROR: train 段为空，调整切分比例或增长数据周期", file=sys.stderr)
        return 2
    print(f"  train={int((df['split']=='train').sum()):,}  "
          f"val={int((df['split']=='val').sum()):,}  "
          f"test={int((df['split']=='test').sum()):,}  "
          f"gap={int((df['split']=='gap').sum()):,}")

    print("\n§3 分桶器 fit (仅 train, 防泄漏) ...")
    amt_state = fit_amount_bin(df_train)
    dt_state = fit_dt_bin(df_train)
    state = {
        "amount": amt_state,
        "dt": dt_state,
        "hour_bin": {"strategy": "fixed_floor_3h", "bins": HOUR_BINS},
        "horizons": HORIZONS,
    }
    df = apply_amount_bin(df, amt_state)
    df = apply_dt_bin(df, dt_state)
    df["__row_id__"] = np.arange(len(df), dtype="int64")

    # §5 daily 聚合 + 标签
    print("\n§5 daily 聚合 + 多窗口标签 ...")
    daily = build_daily(df)
    daily = assign_split(daily, splits)

    # §7 质量门禁
    run_quality_gates(df, daily)

    # 输出
    print(f"\n>> 落盘到 {args.output} ...")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].drop(columns=["split"], errors="ignore")
        sub.to_parquet(args.output / f"txns_{split}.parquet", index=False)
        dsub = daily[daily["split"] == split].drop(columns=["split"], errors="ignore")
        dsub.to_parquet(args.output / f"daily_{split}.parquet", index=False)
    with open(args.output / "tokenizer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  txns_{{train,val,test}}.parquet")
    print(f"  daily_{{train,val,test}}.parquet")
    print(f"  tokenizer_state.json  (含分桶边界，线上推理复用)")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

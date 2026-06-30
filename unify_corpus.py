"""
unify_corpus.py
================
合并预训练语料的两类来源，产出 PANTHER Stage-1 喂入用的统一 schema parquet：

  data_sample/pretrain_corpus.parquet
    列: product_id, date, direction, amount_bin, product_type, risk_level

设计要点 (与 train_xy_model.py::PretrainDataset 对齐):
  1. 预训练只消费 PANTHER Eq.(4) 的 4 维 token (direction/amount_bin/product_type/risk_level);
     **不接触任何回归标签 (purchase/redemption 金额、horizon y)**，因此与 SFT
     train/val/test 的监督信号完全隔离，不存在数据泄漏。
  2. 两类来源的字段名 / 量级 / 分桶口径存在差异，本脚本负责 coax：
     - 仿真 xy_txns.parquet: product_id (9K../9T..), group_id, amount_bin 已 fit
     - akshare txns_real.parquet: product_id (510300/159915..), cust_type 作 group,
                                   product_type/risk_level 来自名称映射 (0..6 / 1..5)
  3. 金额分桶: 跨两类语料重新统一 fit qcut(log1p(amount), 16)，按 direction 分别 fit
     (PANTHER 强制要求 — docs/02 §4.2 / RETRAIN.md §7)。仿真源若已有 amount_bin
     也重新 fit, 避免与 akshare 的桶口径不一致。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM_PATH = HERE / "data_sample" / "xy_txns.parquet"
AKSHARE_PATH = HERE / "data_sample" / "txns_real.parquet"
OUT_PATH = HERE / "data_sample" / "pretrain_corpus.parquet"


# harmonize 后的 token 维度词表上限 (与 train_xy_model.py::CashFlowTransformer 对齐)
N_AMOUNT_BINS = 16


def _fit_amount_bins_global(df: pd.DataFrame) -> pd.DataFrame:
    """跨语料统一按 direction 分组 quantile 分桶 (PANTHER Eq.4 第 2 维)。

    direction ∈ {0=申购, 1=赎}。申赎金额分布差异极大，必须分别 fit。
    幅度 (log1p) 后 16 桶，缺失 / 越界兜底回 0。
    """
    df = df.copy()
    df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0).astype("float64"))
    df["amount_bin"] = -1
    for d in (0, 1):
        m = df["direction"] == d
        if m.sum() < N_AMOUNT_BINS:
            df.loc[m, "amount_bin"] = 0
            continue
        try:
            df.loc[m, "amount_bin"] = pd.qcut(
                df.loc[m, "amount_log1p"], q=N_AMOUNT_BINS,
                labels=False, duplicates="drop",
            ).astype("int16")
        except ValueError:
            df.loc[m, "amount_bin"] = 0
    # 任何残留 NaN/unk 落 0; clamp 到 [0, N_AMOUNT_BINS-1]
    df["amount_bin"] = df["amount_bin"].fillna(0).astype("int16").clip(0, N_AMOUNT_BINS - 1)
    return df


def from_simulate(path: Path) -> pd.DataFrame:
    """case A: 仿真 xy_txns.parquet。

    该路径源自 simulate_xy_real_schema.py，schema（节选）：
      product_id, group_id, txn_ts, txn_type(0申/1赎), amount, yield_rate
    product_type/risk_level 不在 parquet 里 (在 xy_product_meta.json)，需要查表回填。
    统一映射到 (direction, amount, product_type, risk_level)。
    """
    df = pd.read_parquet(path)
    if len(df) == 0:
        return df
    df = df.rename(columns={"txn_type": "direction"})
    if "txn_ts" in df.columns:
        df["date"] = pd.to_datetime(df["txn_ts"], unit="s").dt.normalize()
    elif "date" not in df.columns:
        df["date"] = pd.NaT

    # product_type / risk_level 回填：来自 xy_product_meta.json
    meta_path = HERE / "data_sample" / "xy_product_meta.json"
    import json
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        df["product_type"] = (df["product_id"]
                              .map(lambda p: int(meta.get(p, {}).get("product_type_id", 1)))
                              .fillna(1).astype("int8"))
        df["risk_level"] = (df["product_id"]
                            .map(lambda p: int(meta.get(p, {}).get("risk_level", 2)))
                            .fillna(2).astype("int8"))
    else:
        df["product_type"] = 1
        df["risk_level"] = 2

    df["direction"] = df["direction"].astype("int8")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df[["product_id", "date", "direction", "amount",
               "product_type", "risk_level"]]


def from_akshare(path: Path) -> pd.DataFrame:
    """case B: akshare txns_real.parquet。

    该路径源自 fetch_fund_flow.py，schema 已携带：
      product_id (ETF 代码), date, direction, amount,
      product_type (0..6), risk_level (1..5), cust_type (0零售/1机构)
    无需再查表回填；cust_type 不进 token，不用。
    """
    df = pd.read_parquet(path)
    if len(df) == 0:
        return df
    keep = ["product_id", "date", "direction", "amount",
            "product_type", "risk_level"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"akshare 语料缺列 {missing}，确认 fetch_fund_flow.py 是否最新版")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["direction"] = df["direction"].astype("int8")
    df["product_type"] = df["product_type"].astype("int8").clip(0, 5)
    df["risk_level"] = df["risk_level"].astype("int8").clip(1, 5)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df[keep]


def harmonize_and_merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """合并 + 统一分桶 + 词表 clamp。"""
    df = pd.concat([f for f in frames if f is not None and len(f) > 0],
                   ignore_index=True)
    if len(df) == 0:
        raise RuntimeError("两类语料都为空，请先跑 simulate_xy_real_schema.py "
                           "和/或 fetch_fund_flow.py 生成至少一类")
    df = _fit_amount_bins_global(df)
    # 防御：4 维 token 全部 clamp 到 CashFlowTransformer 词表范围内。
    # 注意 risk_level 在 xy 仿真里取自 PRODUCTS (值=2 即 R2), akshare 里取自名字映射 (值 1..5 即 R1..R5)。
    # 我们 **不做 -1 偏移**, 直接保留 1..5 索引 —— 与 train_xy_model.py SFT 路径里
    # daily["risk_level"].fillna(2) 的索引空间完全一致 (都是 1..5), 这样预训练学到的 risk_emb
    # 权重能直接迁移给 SFT。对应预训练头定义成 nn.Linear(dim, 6) (索引 0..5) 容纳全部可能值。
    df["direction"] = df["direction"].clip(0, 1)
    df["amount_bin"] = df["amount_bin"].clip(0, N_AMOUNT_BINS - 1)
    df["product_type"] = df["product_type"].clip(0, 5)
    df["risk_level"] = df["risk_level"].clip(1, 5)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df[["product_id", "date", "direction", "amount_bin",
               "product_type", "risk_level"]]


def report(df: pd.DataFrame) -> None:
    print(f"\n===== unify_corpus 自检 =====")
    print(f"总行数: {len(df):,}")
    print(f"唯一产品数: {df['product_id'].nunique()}")
    if df["date"].notna().any():
        print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"direction: 申={int((df['direction']==0).sum()):,} "
          f"({(df['direction']==0).mean():.1%}) / "
          f"赎={int((df['direction']==1).sum()):,}")
    print(f"amount_bin 分布 (均匀为佳): {df.groupby('amount_bin').size().to_dict()}")
    print(f"product_type 分布: {df.groupby('product_type').size().to_dict()}")
    print(f"risk_level 分布:   {df.groupby('risk_level').size().to_dict()}")
    print(f"\n>> 输出: {OUT_PATH} (预训练专用，不含回归标签)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=Path, default=SIM_PATH,
                    help=f"仿真 xy_txns.parquet 路径 (默认 {SIM_PATH})")
    ap.add_argument("--akshare", type=Path, default=AKSHARE_PATH,
                    help=f"akshare txns_real.parquet 路径 (默认 {AKSHARE_PATH})")
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help=f"统一预训练语料输出路径 (默认 {OUT_PATH})")
    args = ap.parse_args(argv)

    frames = []
    if args.sim.exists():
        print(f"[load] 仿真语料 {args.sim} ...")
        f = from_simulate(args.sim)
        print(f"  → {len(f):,} 行")
        frames.append(f)
    else:
        print(f"[skip] 仿真语料不存在 ({args.sim})")
    if args.akshare.exists():
        print(f"[load] akshare 语料 {args.akshare} ...")
        f = from_akshare(args.akshare)
        print(f"  → {len(f):,} 行")
        frames.append(f)
    else:
        print(f"[skip] akshare 语料不存在 ({args.akshare})")

    if not frames:
        print("ERROR: 至少需要一类预训练语料。请先跑 "
              "simulate_xy_real_schema.py 或 fetch_fund_flow.py", flush=True)
        return 2

    merged = harmonize_and_merge(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    report(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
verify_data.py
==============

数据 sanity check 脚本（避免在 shell 里跑 ad-hoc inline 代码触发 pandas FutureWarning）。
跑完 simulate_xy_real_schema.py 后用它核对数据是否符合预期。

用法:
  python3 verify_data.py
  python3 verify_data.py --txns data_sample/xy_txns.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txns", type=Path, default=Path("data_sample/xy_txns.parquet"))
    args = ap.parse_args(argv)

    if not args.txns.exists():
        print(f"[ERR] {args.txns} 不存在，请先跑 simulate_xy_real_schema.py", file=sys.stderr)
        return 2

    df = pd.read_parquet(args.txns)
    print(f"==== 数据 sanity check ({args.txns.name}) ====")
    print(f"行数: {len(df):,}")
    print(f"列: {list(df.columns)}")
    print(f"产品 ID: {sorted(df['product_id'].unique())}")

    # ★ v4: group 维度校验
    if "group_id" in df.columns:
        print(f"\n== group 维度（细粒度维度，决定样本量量级）==")
        print(f"group 数: {df['group_id'].nunique()}")
        for gid, gg in df.groupby("group_id"):
            name = gg["group_name"].iloc[0] if "group_name" in gg.columns else f"G{gid}"
            print(f"  group={gid} ({name}): 笔数={len(gg):,}  中位=¥{gg['amount'].median():,.0f}")

    print(f"\n时间跨度: {df['txn_ts'].min()} -> {df['txn_ts'].max()}")

    # 用 groupby + 标量聚合（不用 apply，避免 FutureWarning）
    print("\n按产品 × 方向分布:")
    pivot = df.groupby(["product_id", "txn_type"], as_index=False).size()
    pivot["direction"] = pivot["txn_type"].map({0: "申", 1: "赎"})
    for _, row in pivot.iterrows():
        print(f"  {row['product_id']} {row['direction']}: {int(row['size']):>8,} 笔")

    print("\n每产品 申/赎 占比（用向量化 mean，不用 apply）:")
    for pid, g in df.groupby("product_id"):
        pur_share = float((g["txn_type"] == 0).mean())
        red_share = float((g["txn_type"] == 1).mean())
        amt_median = float(g["amount"].median())
        print(f"  {pid}: 申={pur_share:.1%}  赎={red_share:.1%}  金额中位=¥{amt_median:,.0f}")

    print("\n金额分布统计:")
    print(df["amount"].describe().round(0).to_string())

    # 关键 invariant
    print("\n==== invariant 检查 ====")
    issues = []
    if len(df) < 10_000:
        issues.append(f"⚠ 行数 {len(df)} 偏少（A800 训练期望 >10万）")
    if df["product_id"].nunique() != 2:
        issues.append(f"⚠ 产品数不是 2（实际 {df['product_id'].nunique()}）")
    # 持有期差异
    red_per_pid = df.groupby("product_id").apply(
        lambda g: (g["txn_type"] == 1).mean(),
        include_groups=False   # 显式关掉 FutureWarning
    )
    if len(red_per_pid) == 2:
        a, b = red_per_pid.iloc[0], red_per_pid.iloc[1]
        if a > b:
            print(f"  赎占比: {red_per_pid.index[0]}={a:.1%} > {red_per_pid.index[1]}={b:.1%}")
            print("  注意：持有期长(180d)的产品理论上赎回更稀，请确认 PRODUCTS 配置正确")
        else:
            print(f"  ✓ 赎占比: {red_per_pid.index[0]}={a:.1%} < {red_per_pid.index[1]}={b:.1%} "
                  f"(持有期差异保留)")

    if issues:
        print("\n".join(issues))
        return 1
    print("\n✓ 数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

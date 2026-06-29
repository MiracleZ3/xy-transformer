"""
simulate_xy_real_schema.py
==========================

严格对齐客户真实 schema 的脱敏交易流水模拟器（data_sample/xy_sample.md）。

真实结构（用户确认）:
  - 产品 2 个：9K73101A (固收, 最短持有 180 天, 近 6 月年化 1.922%, 业绩基准 2%~3.8%)
              9T32001A (固收, 最短持有  30 天, 近 1 年年化 2.303%, 基准 0-3 年 AAA 科创债×20%+中债新综合 1-3y×80%)
              二者均为 R2 谨慎型固收
  - 时间：yyyymmddhhmmss（14 位整数）
  - 交易类型：申/赎（两值）
  - 交易状态：只用成功
  - 确认金额：需模拟（基金交易量级，参考固收理财真实水平）
  - 剩余金额：恒 0，直接丢

输出（严格对齐真实可用字段，不留模型用不到的列）：
  data_sample/xy_txns.parquet  每行一笔成功流水
    列: product_id, txn_time(str yyyymmddhhmmss), txn_ts(int Unix 秒), txn_type(0申/1赎),
        amount(float, ¥)
  data_sample/xy_product_meta.json  两个产品的静态信息（type/risk/持有期/年化/基准）

设计取舍 —— 因为产品只有 2 个，本模拟的核心目标是验证：
  ✅ PANTHER Eq.(4) 结构化分词在本真实 schema 上能落地（4 维里 product_type / risk 都是常量,
     但不会失败，词表自动收窄）
  ✅ 序列 Transformer + 多窗口回归头能把多个时间窗口的金额学出来
  ❌ 对比学习的"跨产品迁移"价值无法在 2 个产品池上展示（属真实数据天然限制，非方案缺陷）

数据真实性注入（让模拟数据像真实固收理财）：
  - 申购笔数 >> 赎回（固收产品长期净申购常态）
  - 赎回受持有期约束：9K73101A(180天) 赎回更稀疏、单笔更大；9T32001A(30天) 频次更高
  - 月末/季末赎回节律（资金回流、机构调仓）
  - 金额长尾 lognormal：固收理财典型 ¥1k ~ ¥10M，中位落在 ¥1w-¥10w

运行：
  python3 simulate_xy_real_schema.py                        # 默认全规模 3 年 ~百万级流水
  python3 simulate_xy_real_schema.py --years 5              # 5 年
  python3 simulate_xy_real_schema.py --rate-multiplier 2    # 月笔数再翻倍
  python3 simulate_xy_real_schema.py --small                # 烟雾测试（千分之一量级，秒级）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_RNG = np.random.default_rng(seed=20260617)

# 两个产品的真实画像（来自 xy_sample.md）
# ★ 数据量已大幅放大以适配 A800×8 训练规模：
#   原始零售口径月均 60/120 笔 → 放大 50× 到 3,000 / 7,500 笔/月
#   对应"全市场客户池"口径（含机构/同业/跨分行），不是单一渠道。
#   翻 50–100 倍的逻辑依据:
#     - 单产品 1 年 ~36k-90k 笔 × 3 年 = 数十万笔流水
#     - 这正是 PANTHER 论文 38M 用户 5.3B 交易的"小型版本"
PRODUCTS = {
    "9K73101A": {
        "product_type_name": "固收",
        "product_type_id":   1,        # BOND/FIXED 等价
        "risk_level":        2,        # R2
        "min_holding_days":  180,
        "is_t0":             0,
        "annualized_yield":  0.01922,  # 1.922%
        "benchmark":         "2%~3.8% 年化基准",
        "mu_amount_log":     9.8,      # lognormal mu（中位 ≈ ¥18k）
        "sigma_amount":      1.2,
        "monthly_txn_rate":  3000,     # ★ 60 → 3000（放大 50×，全市场口径）
        "p_redemption_base": 0.30,     # 申/赎基础偏赎率
    },
    "9T32001A": {
        "product_type_name": "固收",
        "product_type_id":   1,
        "risk_level":        2,
        "min_holding_days":  30,
        "is_t0":             0,
        "annualized_yield":  0.02303,  # 2.303%
        "benchmark":         "AAA 科创债×20% + 中债新综合 1-3y×80%",
        "mu_amount_log":     9.3,
        "sigma_amount":      1.0,
        "monthly_txn_rate":  7500,     # ★ 120 → 7500（放大 ~60×，持有期短换手更高）
        "p_redemption_base": 0.35,
    },
}


def _to_yyyymmddhhmmss(ts: pd.Timestamp) -> str:
    """Unix 秒 -> yyyymmddhhmmss 14 位（真实时间字段格式）。"""
    return ts.strftime("%Y%m%d%H%M%S")


def simulate_product(pid: str, meta: dict, base_date: pd.Timestamp, n_days: int) -> pd.DataFrame:
    """为单个产品按其画像生成申赎流水（只生成成功记录）。

    性能优化（百万级数据用）：
      ① 时间/方向/金额 用 numpy 数组批量生成，避免逐笔 Python 循环
      ② 持仓约束（pool_granted 不可负）按时间排序后一次性扫描修正
    """
    # 总笔数
    n_txn = int(meta["monthly_txn_rate"] * n_days / 30)
    # 抽时间：工作日权重 1.0，周末 0.3；月末权重 1.5；季末 1.8
    day_idx = _RNG.integers(0, n_days, size=n_txn)
    dow = (day_idx + base_date.dayofweek) % 7
    w = np.where((dow >= 0) & (dow <= 4), 1.0, 0.3)
    dom = (day_idx % 30) + 1
    is_me = dom >= 27
    is_qe = is_me & ((day_idx % 90) >= 85)
    w = np.where(is_qe, w * 1.8, np.where(is_me, w * 1.5, w))
    w = w / w.sum()
    picked = _RNG.choice(n_txn, size=n_txn, replace=True, p=w)

    # ===== 向量化批量生成（避免百万级的逐笔循环）=====
    day_idx_picked = day_idx[picked]
    secs_picked = _RNG.integers(9 * 3600, 16 * 3600, size=n_txn)
    is_me_picked = is_me[picked]
    is_qe_picked = is_qe[picked]

    # 方向批量：base / 月末 / 季末 / 持有期折扣合并成 per-sample 的赎回概率
    p_red = np.full(n_txn, meta["p_redemption_base"], dtype=np.float64)
    p_red = np.where(is_qe_picked, np.minimum(0.85, p_red + 0.25), p_red)
    p_red = np.where(is_me_picked & ~is_qe_picked, np.minimum(0.75, p_red + 0.15), p_red)
    if meta["min_holding_days"] >= 180:
        p_red = p_red * 0.8
    u = _RNG.random(n_txn)
    direction = (u < p_red).astype(np.int8)   # 0=申, 1=赎

    # 金额批量 lognormal + winsorize
    amount = _RNG.lognormal(meta["mu_amount_log"], meta["sigma_amount"], size=n_txn)
    amount = np.minimum(amount, 10_000_000.0).astype(np.float64)

    # 全量 batch-生成 timestamp（性能关键，不走 pd.Timedelta 逐笔）
    ts_int = (base_date.value // 10**9) + day_idx_picked * 86400 + secs_picked

    # ===== 持仓约束修正（按 ts 升序扫描，O(n)，但只在 Python loop 里跑 pool 更新）=====
    order = np.argsort(ts_int, kind="stable")
    direction = direction[order]
    amount = amount[order]
    ts_int = ts_int[order]
    pool = 0.0
    # 用 numpy buffer 避免逐笔 dict append
    keep_mask = np.ones(n_txn, dtype=bool)
    for i in range(n_txn):
        d = int(direction[i]); amt = float(amount[i])
        if d == 1:   # 赎回
            if pool <= amt:
                # 转小额申购，并 50% 概率跳过
                if _RNG.random() < 0.5:
                    keep_mask[i] = False
                    continue
                direction[i] = 0
                d = 0
        pool = pool + amt if d == 0 else max(0.0, pool - amt)

    # ===== 落盘前最后格式化（向量化生成 txn_time 字符串）=====
    ts_kept = ts_int[keep_mask]
    ts_pd = pd.Series(pd.to_datetime(ts_kept, unit="s"))
    direction_final = direction[keep_mask]
    amount_final = amount[keep_mask]
    df = pd.DataFrame({
        "product_id": pid,
        "txn_time": ts_pd.dt.strftime("%Y%m%d%H%M%S"),
        "txn_ts": ts_kept.astype("int64"),
        "txn_type": direction_final.astype("int8"),
        "amount": np.round(amount_final, 2),
    })
    if len(df) == 0:
        return df
    # 全部视为成功（用户要求"只用成功的"）
    df["status"] = 1
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3, help="历史年数")
    ap.add_argument("--rate-multiplier", type=float, default=1.0,
                    help="月均笔数整体倍率（默认 1.0 = 全规模 ~30 万笔/年；"
                         "调小做 smoke-test，调大堆量）")
    ap.add_argument("--small", action="store_true",
                    help="烟雾测试模式：rate_multiplier=0.01（千分之一量级），秒级跑完")
    args = ap.parse_args(argv)

    mult = 0.01 if args.small else args.rate_multiplier
    if mult != 1.0:
        for pid in PRODUCTS:
            base = PRODUCTS[pid]["monthly_txn_rate"]
            PRODUCTS[pid]["monthly_txn_rate"] = max(10, int(base * mult))
        print(f"  [scale] rate_multiplier={mult} -> 月笔数: "
              f"{PRODUCTS['9K73101A']['monthly_txn_rate']} / "
              f"{PRODUCTS['9T32001A']['monthly_txn_rate']}")

    n_days = 365 * args.years
    base_date = pd.Timestamp("2022-01-03")  # 周一

    frames = []
    for pid, meta in PRODUCTS.items():
        df = simulate_product(pid, meta, base_date, n_days)
        print(f"  {pid}: {len(df):,} 笔成功流水")
        frames.append(df)
    txns = pd.concat(frames, ignore_index=True)
    # 全局时间排序（保留原顺序的 id）
    txns = txns.sort_values(["txn_ts"]).reset_index(drop=True)

    out_parquet = OUT_DIR / "xy_txns.parquet"
    txns.to_parquet(out_parquet, index=False)
    # 静态画像 JSON
    with open(OUT_DIR / "xy_product_meta.json", "w", encoding="utf-8") as f:
        json.dump(PRODUCTS, f, ensure_ascii=False, indent=2)

    # 报告
    print(f"\n===== 模拟数据自检（真实 schema）=====")
    print(f"流水总笔数: {len(txns):,}")
    print(f"产品数: {txns['product_id'].nunique()}  ({list(txns['product_id'].unique())})")
    print(f"时间范围: {txns['txn_time'].min()} ~ {txns['txn_time'].max()}")
    print(f"方向分布: 申={int((txns['txn_type']==0).sum()):,} "
          f"({(txns['txn_type']==0).mean():.1%}) / "
          f"赎={int((txns['txn_type']==1).sum()):,} "
          f"({(txns['txn_type']==1).mean():.1%})")
    amt = txns["amount"]
    print(f"金额分布: 中位=¥{amt.median():,.0f}  P75=¥{amt.quantile(.75):,.0f}  "
          f"P99=¥{amt.quantile(.99):,.0f}  max=¥{amt.max():,.0f}")
    # 按产品分组
    print("\n按产品:")
    for pid, g in txns.groupby("product_id"):
        print(f"  {pid}: 笔数={len(g):,}  赎占比={float((g['txn_type']==1).mean()):.1%}  "
              f"金额中位=¥{g['amount'].median():,.0f}")
    print(f"\n输出: {out_parquet}, {OUT_DIR/'xy_product_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

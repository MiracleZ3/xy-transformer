"""
simulate_customer_txns.py
=========================

生成「真实理财语义」的客户级逐笔脱敏流水，填补 docs/04 §7 唯一剩余缺口
（"客户级逐笔粒度"——ETF/UCI 都是聚合数据）。

输出三张表（数据工程标准三层模型），全部对齐 docs/02 schema：

  data_sample/dim_product.parquet    产品维表（数十个理财产品）
  data_sample/dim_customer.parquet   客户维表（数百~数千脱敏客户）
  data_sample/fact_txn.parquet       逐笔申赎流水事实表（主体）

关键真实性注入（让数据像真实理财而非随机噪声）：
  - 长尾金额：lognormal，¥1k~¥10M 区间（真实货基/股基/固收+ 量级）
  - 产品/客户异质性：不同 risk_level 的 mu 不同（高 R 高金额高波动）
  - 申赎节律：
      * 月末/季末 → 赎回概率升高（机构调仓、个人季末提现）
      * 工作日 vs 周末 → 工作日量大
      * 客户类型：
          - 机构：单笔大、频次低、月末集中
          - 同业：极单笔大、月末+季末集中
          - 个人：单笔小、频次高、波动平稳
      * 持仓约束：赎回金额 ≤ 持仓（不会负持仓）
  - 失败交易：~3% 概率（失败多出现在大额赎回）
  - 失败特征：失败 txns 的 amount 略大（与真实一致——大额易失败/受限）

⚠️ 脱敏口径（重要）：
  - cust_id 用稳定的单向哈希（"CUST_xxxxxx" 前缀），不暴露真实客户号
  - 不生成姓名/身份证/手机/地址；只有 edad band / region_code 这种聚合维度
  - 文件里没有真实可识别个人信息，可作为样本流通

运行:
  python3 simulate_customer_txns.py                    # 默认 300 客户 × 60 产品 × 2 年
  python3 simulate_customer_txns.py --n-cust 2000      # 加大客户规模
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 可复现随机
_RNG = np.random.default_rng(seed=20260612)

# ============================================================
# 产品池（贴近真实财管产品分类）
# ============================================================
PRODUCT_TYPES = [
    # (type_id, name, risk_level, base_mu_log, sigma, is_t0, term_type, min_purchase)
    (0, "MM",         1, 7.0,  0.8, True,  0, 1.0),      # 货基 R1，¥1k 起，T+0
    (1, "BOND",       2, 9.0,  1.0, False, 1, 10_000),   # 债基 R2，¥10000 起
    (4, "FIXEDPLUS",  4, 10.0, 1.2, False, 2, 10_000),   # 固收+ R4
    (3, "MIX",        4, 10.5, 1.4, False, 1, 10_000),   # 混合 R4
    (2, "EQUITY",     5, 11.0, 1.6, False, 1, 1_000),    # 股基 R5，¥1000 起
    (2, "EQUITY",     4, 10.8, 1.3, False, 1, 1_000),    # 指数股基 R4
]

def build_product_table(n_products: int) -> pd.DataFrame:
    """生成 n_products 个产品，循环按 PRODUCT_TYPES 分配。"""
    rows = []
    for i in range(n_products):
        tid, tname, rl, mu, sigma, is_t0, term, min_p = PRODUCT_TYPES[i % len(PRODUCT_TYPES)]
        pid = f"P{i:05d}"
        rows.append({
            "product_id": pid,
            "product_type": tid,
            "product_type_name": tname,
            "risk_level": rl,
            "is_t0": int(is_t0),
            "term_type": term,                           # 0=T+0, 1=开放, 2=封闭
            "min_purchase": float(min_p),
            "cust_segment_focus": _hash_bucket(pid, n=3, salt="seg"),  # 0=零售 1=私行 2=机构
        })
    return pd.DataFrame(rows)


# ============================================================
# 客户池（脱敏）
# ============================================================
def build_customer_table(n_cust: int) -> pd.DataFrame:
    """生成 n_cust 个脱敏客户。

    cust_id_hash: 单向 MD5 哈希成 "CUST_<6hex>"，不可逆推。
    """
    rows = []
    for i in range(n_cust):
        cid_raw = f"RAW-CUST-{i:07d}-{_RNG.integers(1<<31)}"
        cid_hash = "CUST_" + hashlib.md5(cid_raw.encode()).hexdigest()[:6].upper()
        ctype = _RNG.choice([0, 1, 2], p=[0.78, 0.18, 0.04])  # 个人 78%，机构 18%，同业 2%
        # 客户分级：机构/同业都"高净"；个人细分
        if ctype == 0:
            seg = _RNG.choice([0, 1, 2], p=[0.65, 0.25, 0.10])  # 普通/金葵/私行
        else:
            seg = 3 if ctype == 1 else 4   # 机构=3, 同业=4
        rows.append({
            "cust_id_hash": cid_hash,
            "cust_type": int(ctype),
            "cust_segment": int(seg),
            "age_band": int(_hash_bucket(cid_hash, n=7, salt="age")),  # 0..6 年龄段
            "region_code": int(_hash_bucket(cid_hash, n=10, salt="reg")),  # 0..9 地区代理
            "is_highworth": int(seg >= 2),
        })
    return pd.DataFrame(rows)


def _hash_bucket(key: str, n: int, salt: str = "") -> int:
    return int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) % n


# ============================================================
# 流水生成（核心）
# ============================================================
def build_txn_table(products: pd.DataFrame, customers: pd.DataFrame,
                    n_years: int) -> pd.DataFrame:
    """为每个 (客户, 产品) 生成多重序列的申赎流水。"""
    n_days = 365 * n_years
    base_date = pd.Timestamp("2024-01-01")

    # 每个产品的基础参数查表
    pmeta = products.set_index("product_id")[["product_type", "risk_level", "is_t0"]].to_dict("index")

    # base_mu_log / sigma 通过 product_type 查（PRODUCT_TYPES）
    type_to_mu = {pt[0]: pt[3] for pt in PRODUCT_TYPES}
    type_to_sigma = {pt[0]: pt[4] for pt in PRODUCT_TYPES}

    def mu_lookup(pid: str) -> float:
        t = pmeta.get(pid, {}).get("product_type", 2)
        return type_to_mu.get(t, 10.0)

    def sigma_lookup(pid: str) -> float:
        t = pmeta.get(pid, {}).get("product_type", 2)
        return type_to_sigma.get(t, 1.2)

    rows = []
    # 跟踪每个 (cust, product) 的累积持仓，保证赎回不超持仓
    holding = {}   # (cust_id, product_id) -> 累积净额

    cid_to_type = dict(zip(customers["cust_id_hash"], customers["cust_type"]))
    for cust in customers.itertuples(index=False):
        # 每个客户持仓 1~6 个产品（机构少，个人多）
        n_hold = int(np.clip(_RNG.normal(4, 1.5), 1, len(products))) if cust.cust_type == 0 \
            else int(np.clip(_RNG.normal(2, 0.8), 1, len(products)))
        prods = _RNG.choice(products["product_id"].values, size=min(n_hold, len(products)), replace=False)
        for pid in prods:
            ctype = cust.cust_type
            mu_p, sig_p = mu_lookup(pid), sigma_lookup(pid)
            # 月度交易笔数：个人高频小额，机构/同业中频大额（不是"少频"——而是"大额"）
            monthly_rate = {0: 6.0, 1: 4.0, 2: 3.0}[ctype]
            n_txn = max(3, int(monthly_rate * n_years * (n_days / 365)))
            for _ in range(n_txn):
                # 时间：周末 0.4x，月末 1.6x，季末 1.8x
                day_idx = int(_RNG.integers(0, n_days))
                dow = (day_idx) % 7
                w = 1.0 if 0 <= dow <= 4 else 0.4
                dom = (day_idx % 30) + 1
                is_me = dom >= 27
                is_qe = is_me and ((day_idx % 90) >= 85)
                w *= 1.7 if is_qe else (1.4 if is_me else 1.0)
                # 按 w 做加权采样：w>1 月末/季末更易命中，w<1 周末更易跳过
                if _RNG.random() > min(1.0, w) * 0.75:
                    continue
                secs = int(_RNG.integers(9 * 3600, 16 * 3600))
                ts = (base_date + pd.Timedelta(days=day_idx, seconds=secs))

                # 方向：机构/同业 赎回基础概率更高（这是真实规律）
                p_red = {0: 0.35, 1: 0.48, 2: 0.55}[ctype]
                if is_qe:
                    p_red = min(0.85, p_red + 0.25)
                elif is_me:
                    p_red = min(0.78, p_red + 0.15)
                direction = 1 if _RNG.random() < p_red else 0

                # 金额：lognormal（机构/同业 额度上调）
                scale = {0: 1.0, 1: 3.0, 2: 8.0}[ctype]
                amount = float(_RNG.lognormal(mu_p, sig_p)) * scale
                amount = min(amount, 50_000_000.0)  # 单笔上限 ¥5kw

                # 持仓约束：赎回不能超过累积持仓
                key = (cust.cust_id_hash, pid)
                cur_hold = holding.get(key, 0.0)
                if direction == 1:
                    if cur_hold <= 0:
                        continue   # 无持仓跳过赎回
                    amount = min(amount, cur_hold)

                # 状态：失败 ~3%，且大额/高 R 倾向失败
                p_fail = 0.03
                if amount > 1_000_000:
                    p_fail = 0.06
                status_failed = _RNG.random() < p_fail

                # 更新持仓（失败不更新）
                if not status_failed:
                    holding[key] = cur_hold + amount if direction == 0 else max(0.0, cur_hold - amount)
                aum_after = holding.get(key, 0.0)

                rows.append({
                    "txn_id": f"T{len(rows):08d}",
                    "cust_id_hash": cust.cust_id_hash,
                    "product_id": pid,
                    "txn_ts": int(ts.value // 10**9),
                    "txn_type": int(direction),         # 0=申, 1=赎
                    "txn_status": int(not status_failed),  # 1=成功, 0=失败/取消
                    "amount": round(amount, 2),
                    "aum_after": round(aum_after, 2),
                })

    df = pd.DataFrame(rows)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cust", type=int, default=300)
    ap.add_argument("--n-products", type=int, default=40)
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args(argv)

    print(f"生成模拟数据: {args.n_cust} 客户 × {args.n_products} 产品 × {args.years} 年 ...")
    products = build_product_table(args.n_products)
    customers = build_customer_table(args.n_cust)
    print(f"  产品维表: {len(products)} 个产品")
    print(f"  客户维表: {len(customers)} 个客户"
          f"（个人={int((customers['cust_type']==0).sum())}, "
          f"机构={int((customers['cust_type']==1).sum())}, "
          f"同业={int((customers['cust_type']==2).sum())}）")
    txns = build_txn_table(products, customers, args.years)
    print(f"  流水事实表: {len(txns):,} 笔")

    products.to_parquet(OUT_DIR / "dim_product.parquet", index=False)
    customers.to_parquet(OUT_DIR / "dim_customer.parquet", index=False)
    txns.to_parquet(OUT_DIR / "fact_txn.parquet", index=False)

    # 报告
    print("\n===== 模拟数据自检 =====")
    print(f"流水笔数: {len(txns):,}")
    print(f"日期范围: {pd.Timestamp(txns['txn_ts'].min(), unit='s').date()} ~ "
          f"{pd.Timestamp(txns['txn_ts'].max(), unit='s').date()}")
    print(f"客户数: {txns['cust_id_hash'].nunique()} | 产品数: {txns['product_id'].nunique()}")
    print(f"方向分布: 申={int((txns['txn_type']==0).sum()):,} ({(txns['txn_type']==0).mean():.1%}) "
          f"/ 赎={int((txns['txn_type']==1).sum()):,} ({(txns['txn_type']==1).mean():.1%})")
    print(f"状态分布: 成功={int((txns['txn_status']==1).sum()):,} "
          f"失败={int((txns['txn_status']==0).sum()):,} "
          f"(失败率 {(txns['txn_status']==0).mean():.2%})")
    amt = txns["amount"]
    print(f"金额分布: 中位=¥{amt.median():,.0f}  P75=¥{amt.quantile(.75):,.0f}  "
          f"P99=¥{amt.quantile(.99):,.0f}  max=¥{amt.max():,.0f}")
    # 客户类型 vs 方向（验证机构赎回更多）
    pvt = txns.merge(customers[["cust_id_hash","cust_type"]], on="cust_id_hash")
    print("\n客户类型 × 方向 交叉表 (验证机构/同业赎回占比更高):")
    tab = pvt.groupby(["cust_type","txn_type"]).size().unstack(fill_value=0)
    tab.columns = ["申", "赎"]
    tab.index = ["个人", "机构", "同业"]
    print(tab.to_string())
    tab_pct = tab.div(tab.sum(axis=1), axis=0)
    print("\n赎回占比:")
    print((tab_pct["赎"]*100).round(1).astype(str) + "%")
    print(f"\n输出:")
    print(f"  {OUT_DIR/'dim_product.parquet'}")
    print(f"  {OUT_DIR/'dim_customer.parquet'}")
    print(f"  {OUT_DIR/'fact_txn.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

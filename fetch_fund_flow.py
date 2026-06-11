"""
fetch_fund_flow.py
==================

从 akshare 拉取真实中国 ETF / 公募基金数据，构造符合本仓库 schema 的资金流样本，
用作 docs/04 可行性校准（解决 UCI 电商数据"金额量级/申赎比例不匹配"的 ⚠️ 项）。

数据源（已实测可达）:
  - ak.fund_etf_spot_em()           1500+ ETF 实时行情 + 资金流分级
                                    (主力/超大单/大单/中单/小单 净流入 → 机构/零售申赎代理)
  - ak.fund_etf_hist_sina(code)     单只 ETF 日 K (新浪源；东财 push2his 在本沙箱被代理拦截)
                                    返回 date/open/high/low/close/volume/amount (无换手率，够用)
  - ak.fund_etf_category_sina(...)  ETF 分类清单

映射到 schema (对齐 docs/02 §5.1 txns 宽表 + docs/04 真实量级):
  product_id    ← ETF 代码 (510300 / 159915 ...)
  ts            ← 交易日 0 点 Unix 秒 (日级粒度)
  direction     ← 由"成交额" 拆申/赎: 收盘涨 -> 申购为主 (0), 跌 -> 赎回为主 (1)
  status        ← 全部成功 (历史已成交)
  amount        ← 当日成交额 (真实 A 股 ETF 是亿元级，落在 ¥1k-¥1B 区间)
  amount_log1p  ← log1p(amount)
  product_type  ← 从 ETF 名称关键词映射 (货基/债基/股基/混合/商品/跨境)
  risk_level    ← type 推导 (R1 货基 ~ R5 股基/商品)
  yield_rate    ← 当日涨跌幅 (单位 %)
  aum_after     ← 当日成交额累积 (近似规模代理)
  机构vs零售    ← 来自 spot_em 的 主力净流入 / 散户净流入 (作为 cust_type 代理)

设计取舍 / 局限:
  - 真实理财"逐笔申赎"流水不公开；ETF 是最贴近的等价物 (产品×日 资金流)。
  - ETF 二级市场成交 ≠ 一级申赎，但量级、节奏、节假日效应、机构/散户结构都真实。
  - 单次运行默认抽 ~60 只高活跃 ETF × 近 3 年日 K，规模控制在 <50MB。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# ETF 名称 → 产品类型 / 风险等级 关键词映射（用真实命名惯例）
# ------------------------------------------------------------
TYPE_RULES = [   # (regex, type_id, type_name, risk_level)
    (r"货币|现金|钱包",           0, "MM",         1),  # 货基 R1
    (r"债|信用|利率",             1, "BOND",       2),  # 债基 R2
    (r"红利|价值| Berk",          4, "FIXEDPLUS",  4),  # 固收+/红利 R4
    (r"混合|灵活|配置|平衡",      3, "MIX",        4),  # 混合 R4
    (r"50ETF|300|500|800|1000|宽基|综合|A50|创业|科创|上证|深证|中证",
                                  2, "EQUITY",     4),  # 宽基股基 R4
    (r"芯片|半导体|新能源|光伏|电池|医药|医疗|生科|军工|国防|消费|食品|白酒"
     r"|科技|计算机|电子|人工智能|AI|机器人|5G|通信|传媒|游戏|金融|券商|银行"
     r"|地产|基建|有色|煤炭|钢铁|化工|环保|光伏|车|制造|机械|电力",
                                  2, "EQUITY",     5),  # 行业股基 R5
    (r"黄金|白银|原油|商品|资源的|豆粕|有色",  5, "COMMODITY", 5),  # 商品 R5
    (r"美国|纳斯达克|纳指|标普|日经|德国|法国|恒生|港股|海外|跨境|QDII",
                                  6, "CROSSBORDER",5),  # 跨境 R5
]
TYPE_FALLBACK = (2, "EQUITY", 4)

def map_type(name: str) -> tuple[int, str, int]:
    if not isinstance(name, str):
        return TYPE_FALLBACK
    for pat, tid, tn, rl in TYPE_RULES:
        if re.search(pat, name):
            return tid, tn, rl
    return TYPE_FALLBACK


# ------------------------------------------------------------
# 渠道 (channel) / 客户类型 (cust_type) 占位
# ------------------------------------------------------------
def hash_bucket(key: str, n: int, salt: str = "") -> int:
    import hashlib
    return int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) % n


def spot_snapshot() -> pd.DataFrame:
    """拉 ETF 实时快照，用于挑活跃标的 + 拿资金流分级作 cust_type 代理。"""
    import akshare as ak
    print("拉取 ETF 实时快照 (fund_etf_spot_em)...")
    df = ak.fund_etf_spot_em()
    # 取有成交额的，按成交额降序
    df = df[df["成交额"].notna() & (df["成交额"] > 0)].copy()
    df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
    df = df.dropna(subset=["成交额"]).sort_values("成交额", ascending=False)
    print(f"  共 {len(df):,} 只活跃 ETF；头部成交额: {df['成交额'].iloc[0]/1e8:.1f} 亿元")
    return df.reset_index(drop=True)


def hist_one(code: str, start: str, end: str) -> pd.DataFrame | None:
    """拉单只 ETF 日 K（新浪源；东财 push2his 在本沙箱被代理拦截）。

    code 必须带前缀 sz/sh（新浪接口要求）。
    start/end 用于后续裁剪；新浪接口本身返回全历史。
    """
    import akshare as ak
    # 新浪要求 sz/sh + 6 位代码
    c6 = code.lstrip("shsz").lstrip("SHSZ")
    symbol = ("sh" if c6.startswith(("5", "6", "9")) else "sz") + c6
    for attempt in range(3):
        try:
            df = ak.fund_etf_hist_sina(symbol=symbol)
            if df is not None and len(df):
                df["code"] = c6
                # 按窗口裁剪
                df["date"] = pd.to_datetime(df["date"])
                df = df[(df["date"] >= start) & (df["date"] <= end)]
                if len(df):
                    return df
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] {symbol} hist 拉取失败: {type(e).__name__}")
            time.sleep(0.6)
    return None


def build_txns(spot: pd.DataFrame, n_etfs: int, years: int) -> pd.DataFrame:
    """主流程：抽出 n_etfs 只活跃 ETF × 近 years 年日 K，组装成 schema 流水。"""
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y%m%d")
    print(f"窗口: {start} -> {end}, 抽样 {n_etfs} 只活跃 ETF")

    # 选样本：最高成交额的 n_etf 只，且过滤掉代码异常的
    picks = spot.head(n_etfs * 2).copy()
    picks["code6"] = picks["代码"].astype(str).str.strip()
    picks = picks[picks["code6"].str.fullmatch(r"\d{6}")].head(n_etfs)
    print(f"实际抽样: {len(picks)} 只")

    # 把 spot 里的资金流分级 (按 code6 索引) 作 cust_type 代理
    spot_meta = picks.set_index("code6")[["名称", "主力净流入-净额", "小单净流入-净额"]].copy()
    spot_meta.columns = ["name", "main_net", "retail_net"]

    frames = []
    for i, row in enumerate(picks.itertuples(index=False)):
        code = row.code6
        hdf = hist_one(code, start, end)
        if hdf is None or len(hdf) == 0:
            continue
        frames.append(hdf)
        if (i + 1) % 10 == 0:
            print(f"  ...已拉 {i+1}/{len(picks)}")
        time.sleep(0.25)  # 限频，避免被 ban

    if not frames:
        raise RuntimeError("没有任何 ETF 历史数据拉取成功，检查网络。")
    raw = pd.concat(frames, ignore_index=True)
    print(f"原始日 K 合计: {len(raw):,} 行 / {raw['code'].nunique()} 只 ETF")

    # ---- 映射到 schema ----
    spot_lookup = spot.set_index("代码").to_dict("index")
    # fund_etf_hist_sina 已是英文小写列；统一字段名
    df = raw.rename(columns={
        "date": "date", "open": "open", "close": "close",
        "high": "high", "low": "low",
        "volume": "volume", "amount": "amount",
    })
    df["product_id"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["ts"] = df["date"].astype("int64") // 10**9
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[df["amount"].notna() & (df["amount"] > 0)].copy()
    # 申/赎方向：当日 close >= open 视为净申购 (0)，否则为净赎回 (1)。
    # 新浪源无涨跌幅字段，用日内涨跌 (close vs open) 作方向代理
    df["direction"] = np.where(
        pd.to_numeric(df["close"], errors="coerce")
        >= pd.to_numeric(df["open"], errors="coerce"),
        0, 1,
    )
    df["status"] = 1  # 历史已成交，全部成功

    # 金额：ETF 真实成交额 (元)，量级落在 ¥1k ~ ¥1B
    df["amount_log1p"] = np.log1p(df["amount"]).astype("float32")
    # 按方向分桶（申、赎各自 quantile）
    df["amount_bin"] = -1
    for d in (0, 1):
        m = df["direction"] == d
        if m.sum() >= 32:
            df.loc[m, "amount_bin"] = pd.qcut(
                df.loc[m, "amount_log1p"], q=24, labels=False, duplicates="drop"
            ).astype("int16")
        else:
            df.loc[m, "amount_bin"] = 0
    df["amount_bin"] = df["amount_bin"].astype("int16")

    # 产品画像（来自名称映射 + spot 资金流代理）
    meta = df["product_id"].map(lambda c: map_type(spot_lookup.get(c, {}).get("名称", "")))
    df["product_type"] = meta.map(lambda x: x[0]).astype("int8")
    df["risk_level"] = meta.map(lambda x: x[2]).astype("int8")
    df["term_type"] = df["product_id"].map(lambda c: hash_bucket(c, 4, "term")).astype("int8")
    df["channel"] = df["product_id"].map(lambda c: hash_bucket(c, 4, "ch")).astype("int8")
    # 客户类型：用 spot 里的主力净流入信号 — 主力净流入>0 偏机构主导(1)，否则偏零售(0)
    def cust_type(c):
        s = spot_lookup.get(c, {})
        v = s.get("主力净流入-净额")
        try:
            return 1 if pd.notna(v) and float(v) > 0 else 0
        except Exception:
            return 0
    df["cust_type"] = df["product_id"].map(cust_type).astype("int8")

    # 收益率：日线收益率 = (close - open) / open（新浪源无 pct_chg，用日内推算）
    op = pd.to_numeric(df["open"], errors="coerce")
    cl = pd.to_numeric(df["close"], errors="coerce")
    df["yield_rate"] = ((cl - op) / op.where(op > 0, np.nan)).astype("float32")

    # 时间派生
    df["dow"] = df["date"].dt.dayofweek.astype("int8")
    df["dom"] = df["date"].dt.day.astype("int8")
    df["hour"] = 15  # 收盘价统一 15:00
    df["hour_bin"] = 5  # 收盘桶
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype("int8")
    df = df.sort_values(["product_id", "ts"])
    df["dt_prev_sec"] = (
        df.groupby("product_id")["ts"].diff().fillna(0).astype("int64").clip(0, 30 * 86400)
    ).astype("int32")

    # 产品剩余成交额累积（近似规模代理）
    sgn = df["amount"].where(df["direction"] == 0, -df["amount"])
    df["aum_after"] = (
        df.assign(_s=sgn).sort_values(["product_id", "ts"])
        .groupby("product_id")["_s"].cumsum().clip(lower=0)
    ).astype("float64")

    # cust_id (ETF 日级无客户维度，但保留占位以对齐 schema)
    df["cust_id"] = pd.NA

    df = df.reset_index(drop=True)
    df["__row_id__"] = np.arange(len(df), dtype="int64")

    schema_cols = [
        "product_id", "cust_id", "ts", "date", "direction", "status",
        "amount", "amount_log1p", "amount_bin", "aum_after",
        "dow", "dom", "hour", "hour_bin", "is_month_end", "is_quarter_end",
        "dt_prev_sec", "product_type", "risk_level", "term_type",
        "channel", "cust_type", "yield_rate", "__row_id__",
    ]
    return df[[c for c in schema_cols if c in df.columns]]


def report(df: pd.DataFrame) -> None:
    print("\n===== 真实 ETF 资金流数据自检 =====")
    print(f"流水总笔数: {len(df):,}")
    print(f"产品数 (ETF): {df['product_id'].nunique()}")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"direction 分布: 申={int((df['direction']==0).sum()):,} ({(df['direction']==0).mean():.1%}) /"
          f" 赎={int((df['direction']==1).sum()):,} ({(df['direction']==1).mean():.1%})")
    amt = df["amount"]
    print(f"amount (真实成交额) 分布:")
    print(f"  中位: ¥{amt.median():,.0f}  P75: ¥{amt.quantile(.75):,.0f}  "
          f"P99: ¥{amt.quantile(.99):,.0f}  max: ¥{amt.max():,.0f}")
    print(f"amount_log1p 范围: {df['amount_log1p'].min():.2f} ~ {df['amount_log1p'].max():.2f}")
    print(f"product_type 分布: {df.groupby('product_type').size().to_dict()}")
    print(f"cust_type (机构vs零售代理) 分布: 机构={int((df['cust_type']==1).sum()):,} "
          f"零售={int((df['cust_type']==0).sum()):,}")
    print(f"\n输出: {OUT_DIR/'txns_real.parquet'}, {OUT_DIR/'txns_real_sample.csv'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-etfs", type=int, default=60, help="抽样 ETF 数量（按成交额排序）")
    ap.add_argument("--years", type=int, default=3, help="历史回看年数")
    args = ap.parse_args(argv)
    try:
        spot = spot_snapshot()
    except Exception as e:
        print(f"ERROR 拉取快照失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    df = build_txns(spot, n_etfs=args.n_etfs, years=args.years)

    out_parquet = OUT_DIR / "txns_real.parquet"
    df.to_parquet(out_parquet, index=False)
    df.head(2000).to_csv(OUT_DIR / "txns_real_sample.csv", index=False)

    report(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

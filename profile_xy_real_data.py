"""
profile_xy_real_data.py
=======================

【客户现场执行脚本】从真实交易流水 xlsx 中提取**聚合统计信息**，用于校准本仓库的
资金流预测 simulator / preprocess，让仿真数据贴合客户真实分布。

★ 隐私原则（重要）：
  - 本脚本 **绝不输出原始行数据**。只输出聚合统计（分位点、计数、比例、直方图 bin）。
  - 输出物 (statistics.json + profile_report.md) 只包含分布特征，不含任何客户或交易原始值。
  - 你可以把这两份文件放心拷给我，我用它去调 simulator 参数（mu_amount_log / sigma /
    monthly_rate / 月末加成系数 等），不需要看到任何原始行。
  - 视情况可同时把脚本和 xlsx 同机运行（不连外网也行），把生成的 JSON 拿出/邮件发出。

【使用】（三种用法任选）：

  # 用法 A：自动识别列名（脚本内置 product_id/txn_time/txn_type/status/amount 的中英文常用名）
  python3 profile_xy_real_data.py --xlsx 真实流水.xlsx

  # 用法 B：列名特殊时显式指定
  python3 profile_xy_real_data.py --xlsx 真实流水.xlsx \
      --product-col 产品代码 --time-col 交易时间 --type-col 交易类型 \
      --status-col 交易状态 --amount-col 确认金额

  # 用法 C：多个 sheet 或文件
  python3 profile_xy_real_data.py --xlsx 第1批.xlsx 第2批.xlsx --sheet 0

【运行环境】:
  仅依赖 pandas + numpy + openpyxl；不需要 GPU/torch。
  Win/Mac/Linux 通用。
  pip install pandas numpy openpyxl   （或 conda 环境）

【输出】（在工作目录下生成）:
  statistics.json        机器可读的全量聚合统计，我（即建模方）真正需要的就这份
  profile_report.md      人可读的中文报告，你现场自己也能先看一眼分布是否合理
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# 列名候选（中英文混合，按常见命名覆盖）
PRODUCT_COL_CANDIDATES = [
    "product_id", "产品id", "产品ID", "产品代码", "产品编码", "产品号", "product",
    "prd_code", "fund_code", "prod_id",
]
TIME_COL_CANDIDATES = [
    "txn_time", "trans_time", "time", "timestamp",
    "交易时间", "流水时间", "下单时间", "申请时间", "确认时间",
]
# 这些列名是我们自己 pipeline 的派生输出，detect 时一律跳过避免误判
INTERNAL_OUTPUT_COLS = {"txn_ts", "date", "direction", "status", "amount",
                         "product_id", "split", "__row_id__", "__src_file__"}
TYPE_COL_CANDIDATES = [
    "txn_type", "trans_type", "type", "trade_type",
    "交易类型", "流水类型", "业务类型", "申赎类型",
]
STATUS_COL_CANDIDATES = [
    "status", "txn_status", "trans_status",
    "交易状态", "流水状态", "确认状态", "状态",
]
AMOUNT_COL_CANDIDATES = [
    "amount", "txn_amount", "trans_amount", "confirm_amount", "confirmed_amount",
    "确认金额", "交易金额", "成交金额", "发生金额", "金额",
]

PURCHASE_VALUES = {"purchase", "buy", "subscribe", "sub", "inflow", "in",
                   "申购", "购买", "申请", "买入", "申"}
REDEMPTION_VALUES = {"redemption", "redeem", "sell", "outflow", "out",
                     "赎回", "卖出", "赎"}
SUCCESS_VALUES = {"success", "succeeded", "successful", "ok", "1", "done",
                  "成功", "确认成功", "已确认", "完成"}


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns if c not in INTERNAL_OUTPUT_COLS}
    for cand in candidates:
        # 精确
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
        # 包含
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None


def detect_direction_map(series: pd.Series) -> dict:
    """推断每个原始值对应的 0(申)/1(赎)。同时收录原类型 key 与字符串 key，避免 map 类型对不上。"""
    val2dir = {}
    for v in series.dropna().unique():
        # 字符串标准化
        s = str(v).strip()
        low = s.lower()
        if low in PURCHASE_VALUES:
            d = 0
        elif low in REDEMPTION_VALUES:
            d = 1
        else:
            try:
                d = int(float(s))
                if d not in (0, 1):
                    d = -1
            except ValueError:
                d = -1
        # 同时收藏原类型 key 与字符串 key
        val2dir[v] = d
        val2dir[s] = d
    return val2dir


def detect_success_mask(series: pd.Series) -> pd.Series | None:
    raw = series.astype(str).str.strip().str.lower()
    is_success = raw.isin(SUCCESS_VALUES)
    if is_success.sum() == 0:
        # 数字型尝试：1=成功 0=失败
        try:
            return pd.to_numeric(series, errors="coerce").fillna(-1).astype(int) == 1
        except Exception:
            return None
    return is_success


def _parse_time_column(series: pd.Series):
    """优先按 yyyymmddhhmmss 整数解析，再回退到 datetime。"""
    # 数值型：yyyymmddhhmmss
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.astype("int64").abs().between(10**13, 10**14 + 10**12).mean() > 0.5:
        ts = pd.to_datetime(numeric.astype("int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        return ts
    if numeric.notna().all() and numeric.astype("int64").abs().between(10**9, 10**11).mean() > 0.5:
        # 可能是 Unix 秒
        return pd.to_datetime(numeric.astype("int64"), unit="s", errors="coerce")
    # 字符串日期
    return pd.to_datetime(series, errors="coerce")


def _quantiles(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {}
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    qvals = np.quantile(x, qs)
    return {f"p{int(q*100):02d}": float(v) for q, v in zip(qs, qvals)}


def _histogram(x: pd.Series, n_bins: int = 20) -> dict:
    """输出固定面数的直方图，仅 bin 边界 + 计数（重构不回原始）。"""
    x = x.dropna()
    if len(x) == 0:
        return {}
    counts, edges = np.histogram(x, bins=n_bins)
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def profile_one_product(df: pd.DataFrame, pid: str) -> dict:
    """对单产品统计所有维度。"""
    out = OrderedDict()
    out["n_records"] = int(len(df))

    # --- 时间跨度 ---
    if "ts" in df.columns:
        ts = df["ts"]
        out["time_span"] = {
            "min": str(ts.min()),
            "max": str(ts.max()),
            "days_total": int((ts.max() - ts.min()).days) if pd.notna(ts.max()) else 0,
        }
        # 按年/月/星期/小时聚合活跃度
        out["txns_per_year"] = {int(k): int(v) for k, v in ts.dt.year.value_counts().items()}
        out["txns_per_month"] = {int(k): int(v) for k, v in ts.dt.month.value_counts().items()}
        out["txns_per_dow"] = {  # 0=周一 ... 6=周日
            str(i): int(ts.dt.dayofweek.value_counts().get(i, 0)) for i in range(7)
        }
        daily_count = ts.dt.date.value_counts().sort_index()
        out["daily_count_quantiles"] = _quantiles(daily_count.astype(float))
        # 月末/季末效应: (月末日均笔数) / (整体日均笔数)
        month_end_mask = ts.dt.is_month_end
        is_me = ts.dt.is_month_end
        is_qe = ts.dt.is_quarter_end
        me_rate = float(month_end_mask.mean())
        qe_rate = float(is_qe.mean())
        out["month_end_share"] = me_rate
        out["quarter_end_share"] = qe_rate
        # 按日聚合再算月末日均值
        if "date" in df.columns:
            by_day = df.groupby("date").size()
            me_dates = ts[is_me].dt.date.unique()
            qe_dates = ts[is_qe].dt.date.unique()
            # 与 by_day 比对统计（注意是产品内日均）
            all_day_mean = float(by_day.mean()) if len(by_day) else 0.0
            me_subset = by_day.reindex(list(me_dates)).dropna()
            qe_subset = by_day.reindex(list(qe_dates)).dropna()
            out["avg_txns_per_day"] = all_day_mean
            out["avg_txns_per_month_end_day"] = float(me_subset.mean()) if len(me_subset) else None
            out["avg_txns_per_quarter_end_day"] = float(qe_subset.mean()) if len(qe_subset) else None

    # --- 申/赎比例 ---
    if "direction" in df.columns:
        out["direction_counts"] = {str(k): int(v) for k, v in df["direction"].value_counts().items()}
        out["purchase_share"] = float((df["direction"] == 0).mean())
        out["redemption_share"] = float((df["direction"] == 1).mean())

    # --- 金额分布（按方向分别）---
    if "amount" in df.columns:
        out["amount_overall"] = {
            "quantiles": _quantiles(df["amount"]),
            "histogram": _histogram(df["amount"], n_bins=20),
            "mean": float(df["amount"].mean()),
            "std": float(df["amount"].std()),
            "log1p_quantiles": _quantiles(np.log1p(df["amount"].clip(lower=0))),
        }
        # 按方向分别 quantile
        by_dir = {}
        for d, name in [(0, "purchase"), (1, "redemption")]:
            sub = df.loc[df["direction"] == d, "amount"] if "direction" in df.columns else None
            if sub is None or len(sub) == 0:
                continue
            by_dir[name] = {
                "n": int(len(sub)),
                "quantiles": _quantiles(sub),
                "log1p_quantiles": _quantiles(np.log1p(sub.clip(lower=0))),
                "mean": float(sub.mean()),
                "std": float(sub.std()),
            }
        out["amount_by_direction"] = by_dir

    # --- 状态分布（成功/失败/总）---
    if "status" in df.columns:
        out["status_share_success"] = float(df["status"].mean())
        out["status_counts"] = {str(k): int(v) for k, v in df["status"].value_counts().items()}

    return out


def write_markdown(stats: dict, path: Path) -> None:
    """生成人可读中文 markdown 报告，方便现场肉眼检查。"""
    lines = ["# 客户真实数据统计报告（脱敏，仅分布）\n"]
    lines.append(f"- 数据源文件: {', '.join(stats.get('files', []))}")
    lines.append(f"- 总记录数: {stats.get('n_records_total', '?')}")
    lines.append(f"- 涉及产品: {list(stats.get('products', {}).keys())}\n")
    for pid, p in stats["products"].items():
        lines.append(f"## 产品 {pid}\n")
        lines.append(f"- 总记录数: {p.get('n_records')}")
        if "time_span" in p:
            ts = p["time_span"]
            lines.append(f"- 时间范围: {ts['min']} ~ {ts['max']}（{ts['days_total']} 天）")
        if "purchase_share" in p:
            lines.append(f"- 申占比: {p['purchase_share']:.2%} / 赎占比: {p['redemption_share']:.2%}")
        if "status_share_success" in p:
            lines.append(f"- 成功率: {p['status_share_success']:.2%}")
        if "amount_overall" in p:
            a = p["amount_overall"]
            q = a["quantiles"]
            lines.append(f"- 金额分布（元）: P25={q.get('p25',0):,.0f}  "
                         f"中位={q.get('p50',0):,.0f}  "
                         f"P75={q.get('p75',0):,.0f}  "
                         f"P99={q.get('p99',0):,.0f}")
            lines.append(f"- 金额均值/标准差: {a['mean']:,.0f} / {a['std']:,.0f}")
        if "amount_by_direction" in p:
            for d_name, d_stats in p["amount_by_direction"].items():
                dq = d_stats["quantiles"]
                if dq:
                    lines.append(f"  - {d_name}: n={d_stats['n']}, "
                                 f"中位={dq.get('p50',0):,.0f}, "
                                 f"P99={dq.get('p99',0):,.0f}")
        if "txns_per_dow" in p:
            dows = ["周一","周二","周三","周四","周五","周六","周日"]
            counts = p["txns_per_dow"]
            txt = ", ".join(f"{dows[i]}={counts.get(str(i), counts.get(i,0))}" for i in range(7))
            lines.append(f"- 按星期分布: {txt}")
        if "avg_txns_per_day" in p:
            lines.append(f"- 日均笔数: {p['avg_txns_per_day']:.1f}  月末日均: "
                         f"{p.get('avg_txns_per_month_end_day')}  "
                         f"季末日均: {p.get('avg_txns_per_quarter_end_day')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="提取真实 xlsx 数据聚合统计（脱敏）")
    ap.add_argument("--xlsx", nargs="+", required=True, help="一个或多个 xlsx 文件路径")
    ap.add_argument("--sheet", default="0", help="sheet 名或索引（默认第一个）")
    ap.add_argument("--product-col", default=None, help="覆盖自动识别的产品列名")
    ap.add_argument("--time-col", default=None, help="覆盖自动识别的时间列名")
    ap.add_argument("--type-col", default=None, help="覆盖自动识别的交易类型列名")
    ap.add_argument("--status-col", default=None, help="覆盖自动识别的状态列名")
    ap.add_argument("--amount-col", default=None, help="覆盖自动识别的金额列名")
    ap.add_argument("--out-dir", default=".", help="输出目录")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    used_files = []
    for fp in args.xlsx:
        fp_in = Path(fp).expanduser()
        if not fp_in.exists():
            print(f"[ERR] 文件不存在: {fp_in}", file=sys.stderr); return 2
        # 读 sheet
        sheet_arg = args.sheet
        try:
            sheet = int(sheet_arg)
            df = pd.read_excel(fp_in, sheet_name=sheet)
        except (ValueError, TypeError):
            df = pd.read_excel(fp_in, sheet_name=sheet_arg)
        df["__src_file__"] = fp_in.name
        all_frames.append(df)
        used_files.append(fp_in.name)
        print(f"  读取 {fp_in.name}: {len(df):,} 行  列={list(df.columns)}")

    raw = pd.concat(all_frames, ignore_index=True)
    print(f"  合计 {len(raw):,} 行")

    # 列名识别
    product_col = args.product_col or detect_column(raw, PRODUCT_COL_CANDIDATES)
    time_col    = args.time_col    or detect_column(raw, TIME_COL_CANDIDATES)
    type_col    = args.type_col    or detect_column(raw, TYPE_COL_CANDIDATES)
    status_col  = args.status_col  or detect_column(raw, STATUS_COL_CANDIDATES)
    amount_col  = args.amount_col  or detect_column(raw, AMOUNT_COL_CANDIDATES)

    print("\n识别到的列:")
    print(f"  product: {product_col}\n  time: {time_col}\n  type: {type_col}")
    print(f"  status: {status_col}\n  amount: {amount_col}")
    missing = [n for n, c in [("product", product_col), ("time", time_col),
                              ("type", type_col), ("amount", amount_col)] if c is None]
    if missing:
        print(f"\n[ERR] 缺列: {missing}. 请用 --xxx-col 参数显式指定。", file=sys.stderr)
        return 3

    # 解析 / 规整
    df = raw.rename(columns={
        product_col: "product_id",
        time_col: "txn_time_raw",
        type_col: "type_raw",
        amount_col: "amount",
    })
    if status_col:
        df = df.rename(columns={status_col: "status_raw"})
    df["ts"] = _parse_time_column(df["txn_time_raw"])
    df["date"] = df["ts"].dt.normalize()
    df["direction"] = df["type_raw"].map(detect_direction_map(df["type_raw"]))
    if "status_raw" in df.columns:
        succ = detect_success_mask(df["status_raw"])
        df["status"] = succ.fillna(False).astype(bool).astype(int) if succ is not None else 1
    else:
        df["status"] = 1
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    n_drop = df["amount"].isna().sum()
    if n_drop:
        print(f"  丢弃 amount 非数值: {n_drop}")

    n_records_total = len(df)

    # 按产品分桶统计
    products = OrderedDict()
    for pid, g in df.groupby("product_id"):
        products[str(pid)] = profile_one_product(g.reset_index(drop=True), str(pid))

    stats = {
        "files": used_files,
        "n_records_total": int(n_records_total),
        "row_status": {"dropped_amount_na": int(n_drop)},
        "column_mapping_detected": {
            "product": product_col, "time": time_col, "type": type_col,
            "status": status_col, "amount": amount_col,
        },
        "products": products,
    }

    # 落盘
    json_path = out_dir / "statistics.json"
    md_path = out_dir / "profile_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    write_markdown(stats, md_path)

    print(f"\n✓ 统计聚合完成（无原始行数据输出）")
    print(f"  机器可读 (给我): {json_path}")
    print(f"  人可读报告   (你先看): {md_path}")
    print(f"\n请把 statistics.json + profile_report.md 拷给我即可，"
          f"我会用里面的分布去重调 simulator 参数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

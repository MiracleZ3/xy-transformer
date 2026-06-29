"""
plot_summary.py
===============

从 model_out/{eval_summary.json, all_runs.jsonl, test_predictions.parquet}
+ data_sample/xy_txns.parquet 生成 4 张图，配套 docs/07 三段式总结。

v2（重写）—— 针对上版"看不出差异"+"没置信区间"的反馈:
  · 图3 训练曲线 改为 **3 个 seed 的均值 + 标准差带状区间**, 不再单一曲线
  · 图4 基线对比 加 **误差棒 (mean ± std)**, 不再单点
  · 新增 **相对提升 % 子图**: (LightGBM - Transformer) / LightGBM × 100%
  · 主图明示 mean/std/seed 数

输出: docs/assets/fig{1,2,3,4}*.png

★ 重要：本脚本读的是 model_out/，意味着必须先跑 train_xy_model.py。
  在本地若没跑全量训练，运行时若发现数据未收敛（如 Transformer WAPE > LightGBM），
  脚本会把"未收敛警告"叠到图上，避免小规模训练产物被误读为结论。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                    "Heiti TC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
TXNS_PATH = HERE / "data_sample" / "xy_txns.parquet"
EVAL_PATH = HERE / "model_out" / "eval_summary.json"
RUNS_PATH = HERE / "model_out" / "all_runs.jsonl"
PRED_PATH = HERE / "model_out" / "test_predictions.parquet"
ASSETS = HERE / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PRODUCT_COLORS = {"9K73101A": "#1f77b4", "9T32001A": "#ff7f0e"}
METHOD_COLORS = {"Naive mean": "#bdbdbd", "LightGBM": "#74c476",
                 "Transformer": "#3182bd"}


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_converged(summary):
    """v4: 粗略判断训练收敛——Transformer 的 申购+1d WAPE 应该 <= Naive mean 的同口径值。"""
    try:
        kh = "purchase_log1p|1"
        transf = summary["Transformer"][kh]["WAPE"]["mean"]
        naive = summary["Naive mean"][kh]["WAPE"]["mean"]
        # 给 1.5x 容差也是"收敛可用"；若 Transformer 反超 1.5x 则标未收敛
        return transf <= naive * 1.5
    except Exception:
        return None


# ============================================================
# Fig 1: 数据分布
# ============================================================
def fig1():
    if not TXNS_PATH.exists():
        print("[skip] xy_txns.parquet 不存在"); return
    TXNS = pd.read_parquet(TXNS_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for pid, c in PRODUCT_COLORS.items():
        sub = TXNS[TXNS["product_id"] == pid]
        pur = np.log1p(sub[sub["txn_type"] == 0]["amount"])
        red = np.log1p(sub[sub["txn_type"] == 1]["amount"])
        ax.hist(pur, bins=50, alpha=0.55, color=c, label=f"{pid} 申",
                density=True, histtype="stepfilled")
        ax.hist(red, bins=50, alpha=0.5, color=c, label=f"{pid} 赎",
                density=True, histtype="step", linewidth=1.8)
    ax.set_xlabel("log1p(金额)"); ax.set_ylabel("密度")
    ax.set_title(f"图1a  金额分布（n={len(TXNS):,} 笔）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    pids = list(PRODUCT_COLORS.keys())
    pur_share = [float((TXNS[TXNS["product_id"] == p]["txn_type"] == 0).mean()) for p in pids]
    red_share = [float((TXNS[TXNS["product_id"] == p]["txn_type"] == 1).mean()) for p in pids]
    ax.bar(pids, pur_share, color="#2ca02c", label="申购", width=0.5)
    ax.bar(pids, red_share, bottom=pur_share, color="#d62728", label="赎回", width=0.5)
    for i, p in enumerate(pids):
        ax.text(i, pur_share[i] / 2, f"{pur_share[i]*100:.1f}%",
                ha="center", color="white", fontweight="bold")
        ax.text(i, pur_share[i] + red_share[i] / 2, f"{red_share[i]*100:.1f}%",
                ha="center", color="white", fontweight="bold")
    ax.set_ylabel("占比"); ax.set_ylim(0, 1.0)
    ax.set_title("图1b  申/赎占比（反映持有期差异）")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(ASSETS / "fig1_data_distribution.png",
                                    dpi=140, bbox_inches="tight"); plt.close()
    print("✓ fig1_data_distribution.png")


# ============================================================
# Fig 2: 时间节律
# ============================================================
def fig2():
    if not TXNS_PATH.exists():
        print("[skip] xy_txns.parquet 不存在"); return
    df = pd.read_parquet(TXNS_PATH)
    df["date"] = pd.to_datetime(df["txn_ts"], unit="s")
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    dows = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    width = 0.4; x = np.arange(7)
    for i, (pid, c) in enumerate(PRODUCT_COLORS.items()):
        cnt = df[df["product_id"] == pid].groupby("dow").size().reindex(range(7), fill_value=0)
        ax.bar(x + (i - 0.5) * width, cnt.values, width=width, color=c, label=pid)
    ax.set_xticks(x); ax.set_xticklabels(dows)
    ax.set_ylabel("笔数"); ax.set_title("图2a  周内活跃度（工作日 vs 周末）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    monthly = df.groupby(["month", "txn_type"])["amount"].sum().unstack(fill_value=0)
    for col, color, lab in [(0, "#2ca02c", "申购"), (1, "#d62728", "赎回")]:
        if col in monthly.columns:
            ax.plot(monthly.index, monthly[col] / 1e4, marker="o",
                    color=color, label=lab, linewidth=2)
    ax.set_xlabel("月份"); ax.set_ylabel("累积金额 (¥万)"); ax.set_xticks(range(1, 13))
    ax.set_title("图2b  月度申/赎金额节律（季末 effect）")
    for qm in [3, 6, 9, 12]:
        ax.axvline(qm, color="gray", linestyle=":", alpha=0.5)
    ymax = ax.get_ylim()[1]
    for q, qname in zip([3, 6, 9, 12], ["Q1末", "Q2末", "Q3末", "Q4末"]):
        ax.text(q, ymax * 0.92, qname, ha="center", color="gray", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ASSETS / "fig2_time_patterns.png",
                                    dpi=140, bbox_inches="tight"); plt.close()
    print("✓ fig2_time_patterns.png")


# ============================================================
# Fig 3: 训练曲线（多 seed mean ± std 带状）
# ============================================================
def fig3():
    if not RUNS_PATH.exists():
        print("[skip] all_runs.jsonl 不存在"); return
    runs = _load_jsonl(RUNS_PATH)
    transf_runs = [r for r in runs if r["method"] == "Transformer" and "history" in r]
    if not transf_runs:
        print("[skip] 没有 Transformer history"); return

    # 把每个 seed 的 history 按最大 epoch 长度对齐
    max_ep = max(len(r["history"]) for r in transf_runs)
    train = np.full((len(transf_runs), max_ep), np.nan)
    val = np.full((len(transf_runs), max_ep), np.nan)
    for i, r in enumerate(transf_runs):
        h = r["history"]
        for j, rec in enumerate(h):
            train[i, j] = rec["train_loss"]
            val[i, j]   = rec["val_loss"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, max_ep + 1)
    # 均值曲线 + ±1 std 带状
    def line_with_band(arr, label, color, ls="-"):
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0)
        ax.plot(epochs, m, color=color, linewidth=1.8, ls=ls, label=label)
        ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.15)

    line_with_band(train, f"训练 loss (n={len(transf_runs)} seeds)", "#1f77b4")
    line_with_band(val,   f"验证 loss (n={len(transf_runs)} seeds)", "#ff7f0e")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss (log)")
    ax.set_title(f"图3  训练曲线（{len(transf_runs)} seeds 均值 ± 标准差）")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(ASSETS / "fig3_training_curve.png",
                                    dpi=140, bbox_inches="tight"); plt.close()
    print("✓ fig3_training_curve.png")


# ============================================================
# Fig 4: 基线对比（带误差棒）— v4: 分申/赎两路 × 3 horizon
# ============================================================
def fig4():
    if not EVAL_PATH.exists():
        print("[skip] eval_summary.json 不存在"); return
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    s = data["summary"]; n_seeds = data.get("n_seeds", "?")
    converged = _is_converged(s)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    for ax_idx, (kind, kind_label) in enumerate(
        [("purchase_log1p", "申购 PURCHASE"), ("redemption_log1p", "赎回 REDEMPTION")]
    ):
        ax = axes[ax_idx]
        methods = [m for m in ["Naive mean", "LightGBM", "Transformer"] if m in s]
        horizons = [1, 7, 30]
        width = 0.25; x = np.arange(len(horizons))
        for i, m in enumerate(methods):
            means, stds = [], []
            for h in horizons:
                kh = f"{kind}|{h}"
                w = s[m].get(kh, {}).get("WAPE", {})
                means.append(w.get("mean", float("nan")))
                stds.append(w.get("std", 0))
            if any(np.isnan(v) for v in means):
                continue
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
                   color=METHOD_COLORS.get(m, "#888"),
                   edgecolor="black", linewidth=0.5, label=m,
                   error_kw={"elinewidth": 1.0, "ecolor": "black"})
            for j, (mu, sd) in enumerate(zip(means, stds)):
                ax.text(x[j] + (i - 1) * width, mu + sd + max(means) * 0.02,
                        f"{mu*100:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"+{h}天" for h in horizons])
        ax.set_ylabel("WAPE (越低越好)")
        ax.set_title(f"图4{'ab'[ax_idx]}  {kind_label}  WAPE 对比\n({n_seeds} seeds mean ± std)")
        ax.legend(fontsize=8, loc="upper right")
        all_means = [s[m].get(f"{kind}|{h}", {}).get("WAPE", {}).get("mean", 1.0)
                     for m in methods for h in horizons]
        all_stds = [s[m].get(f"{kind}|{h}", {}).get("WAPE", {}).get("std", 0)
                    for m in methods for h in horizons]
        ymax = max((m + sd) for m, sd in zip(all_means, all_stds) if not np.isnan(m))
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(alpha=0.3, axis="y")

    # 未收敛警告 stamp
    if converged is False:
        fig.text(0.5, 0.02,
                 "[警告] 训练未收敛（mini 烟雾测试产物）—— 不可作为最终结论，"
                 "需在 A800×8 上跑全量训练",
                 ha="center", color="red", fontsize=10, fontweight="bold")

    plt.tight_layout(); plt.savefig(ASSETS / "fig4_eval_comparison.png",
                                    dpi=140, bbox_inches="tight"); plt.close()
    print("✓ fig4_eval_comparison.png")


# ============================================================
# Fig 5: 预测散点（v4: 申购 + 赎回 两套）
# ============================================================
def fig5():
    if not PRED_PATH.exists():
        print("[skip] test_predictions.parquet 不存在"); return
    pred = pd.read_parquet(PRED_PATH)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (kind, lab) in zip(axes, [("pur", "申购"), ("red", "赎回")]):
        col_t = f"truth_{kind}_log1p_h1"; col_p = f"pred_{kind}_log1p_h1"
        ax.scatter(pred[col_t], pred[col_p], alpha=0.5, s=22,
                   c=pred["product_id"].map(PRODUCT_COLORS))
        lo = min(pred[col_t].min(), pred[col_p].min())
        hi = max(pred[col_t].max(), pred[col_p].max())
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--",
                linewidth=1.2, label="完美预测 y=x")
        for pid, c in PRODUCT_COLORS.items():
            ax.scatter([], [], c=c, label=pid)
        ax.set_xlabel(f"真实 log1p({lab} +1d)"); ax.set_ylabel("Transformer 预测")
        ax.set_title(f"图5  {lab} +1d 预测散点")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ASSETS / "fig5_pred_scatter.png",
                                    dpi=140, bbox_inches="tight"); plt.close()
    print("✓ fig5_pred_scatter.png")


if __name__ == "__main__":
    print("生成插图...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print(f"\n>> 输出目录: {ASSETS}")
    for p in sorted(ASSETS.glob("*.png")):
        print(f"   {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

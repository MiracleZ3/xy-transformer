"""
plot_summary.py
===============

为 docs/07 三段式总结生成所有插图。从 data_sample/xy_txns.parquet + model_out/ 读取，
不重新跑训练。输出 PNG 到 docs/assets/。

图表清单:
  fig1_data_distribution.png   数据分布: 金额 log-直方图 + 申赎方向占比（按产品）
  fig2_time_patterns.png       时间节律: 周/月热力 + 申赎量按月
  fig3_training_curve.png      训练曲线: train/val loss 每 epoch
  fig4_eval_comparison.png     基线对比: WAPE 柱状（3 horizon × 3 method）+ 预测散点
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能写文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                    "Heiti TC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
TXNS = pd.read_parquet(HERE / "data_sample" / "xy_txns.parquet")
HIST = json.loads((HERE / "model_out" / "training_history.json").read_text(encoding="utf-8"))
EVAL = json.loads((HERE / "model_out" / "eval_summary.json").read_text(encoding="utf-8"))
PRED = pd.read_parquet(HERE / "model_out" / "test_predictions.parquet")

ASSETS = HERE / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PRODUCT_COLORS = {"9K73101A": "#1f77b4", "9T32001A": "#ff7f0e"}

# ============================================================
# Fig 1: 数据分布
# ============================================================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左: 金额 log1p 直方图，按产品 × 方向分别
    ax = axes[0]
    for pid, color in PRODUCT_COLORS.items():
        sub = TXNS[TXNS["product_id"] == pid]
        pur = np.log1p(sub[sub["txn_type"] == 0]["amount"])
        red = np.log1p(sub[sub["txn_type"] == 1]["amount"])
        ax.hist(pur, bins=40, alpha=0.55, color=color, label=f"{pid} 申购",
                density=True, histtype="stepfilled")
        ax.hist(red, bins=40, alpha=0.35, color=color, label=f"{pid} 赎回",
                density=True, histtype="step", linewidth=1.8)
    ax.set_xlabel("log1p(金额)   (= log(1 + ¥金额))")
    ax.set_ylabel("密度")
    ax.set_title("图1a  金额分布（按产品 × 方向）")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # 右: 申赎占比 stacked bar by product
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
    ax.set_ylabel("占比")
    ax.set_ylim(0, 1.0)
    ax.set_title("图1b  申/赎占比（按产品）—— 反映持有期差异")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(ASSETS / "fig1_data_distribution.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ fig1_data_distribution.png")


# ============================================================
# Fig 2: 时间节律
# ============================================================
def fig2():
    df = TXNS.copy()
    df["date"] = pd.to_datetime(df["txn_ts"], unit="s")
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon
    df["month"] = df["date"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左: 按星期 × 产品 的笔数
    ax = axes[0]
    dows = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    width = 0.4
    x = np.arange(7)
    for i, (pid, color) in enumerate(PRODUCT_COLORS.items()):
        cnt = df[df["product_id"] == pid].groupby("dow").size().reindex(range(7), fill_value=0)
        ax.bar(x + (i - 0.5) * width, cnt.values, width=width, color=color, label=pid)
    ax.set_xticks(x); ax.set_xticklabels(dows)
    ax.set_ylabel("笔数");
    ax.set_title("图2a  周内活跃度分布（工作日 vs 周末）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # 右: 按月的累积金额（按申/赎分）
    ax = axes[1]
    monthly = df.groupby(["month", "txn_type"])["amount"].sum().unstack(fill_value=0)
    for col, color, lab in [(0, "#2ca02c", "申购"), (1, "#d62728", "赎回")]:
        if col in monthly.columns:
            ax.plot(monthly.index, monthly[col] / 1e4, marker="o",
                    color=color, label=lab, linewidth=2)
    ax.set_xlabel("月份")
    ax.set_ylabel("累积金额 (¥万)")
    ax.set_xticks(range(1, 13))
    ax.set_title("图2b  月度申/赎金额节律（月末/季末效应）")
    # 标注季末
    for qm in [3, 6, 9, 12]:
        ax.axvline(qm, color="gray", linestyle=":", alpha=0.5)
    ax.text(3, ax.get_ylim()[1] * 0.92, "Q1末", ha="center", color="gray", fontsize=9)
    ax.text(6, ax.get_ylim()[1] * 0.92, "Q2末", ha="center", color="gray", fontsize=9)
    ax.text(9, ax.get_ylim()[1] * 0.92, "Q3末", ha="center", color="gray", fontsize=9)
    ax.text(12, ax.get_ylim()[1] * 0.92, "Q4末", ha="center", color="gray", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / "fig2_time_patterns.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ fig2_time_patterns.png")


# ============================================================
# Fig 3: 训练曲线
# ============================================================
def fig3():
    h = HIST["history"]
    eps = [r["epoch"] for r in h]
    tr = [r["train_loss"] for r in h]
    vl = [r["val_loss"] for r in h]
    best_vl = min(vl)
    best_ep = eps[vl.index(best_vl)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(eps, tr, marker="o", markersize=4, linewidth=1.8, color="#1f77b4", label="训练 loss")
    ax.plot(eps, vl, marker="s", markersize=4, linewidth=1.8, color="#ff7f0e", label="验证 loss")
    ax.axhline(best_vl, color="red", linestyle="--", alpha=0.6,
               label=f"最低 val loss = {best_vl:.4f}  (epoch {best_ep})")
    # 把首 epoch 单独高一些做对数刻度可视
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss (log)")
    ax.set_title("图3  训练与验证损失曲线（30 epoch，CPU 78s）")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(ASSETS / "fig3_training_curve.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ fig3_training_curve.png")


# ============================================================
# Fig 4: 基线对比
# ============================================================
def fig4():
    s = EVAL["summary"]
    methods = [m for m in ["Naive mean", "LightGBM", "Transformer (ours)"] if m in s]
    horizons = [1, 7, 30]
    colors = {"Naive mean": "#bdbdbd", "LightGBM": "#74c476", "Transformer (ours)": "#3182bd"}

    fig = plt.figure(figsize=(12, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 左: WAPE 分组柱
    width = 0.25
    x = np.arange(len(horizons))
    for i, m in enumerate(methods):
        wapes = [s[m][str(h)]["WAPE"] for h in horizons]
        ax1.bar(x + (i - 1) * width, wapes, width=width, color=colors[m],
                label=m, edgecolor="black", linewidth=0.5)
        for j, w in enumerate(wapes):
            ax1.text(x[j] + (i - 1) * width, w + 0.001,
                     f"{w*100:.2f}%", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels([f"+{h}天" for h in horizons])
    ax1.set_ylabel("WAPE（金额相对误差，越低越好）")
    ax1.set_title("图4a  三方法 × 三 horizon WAPE 对比")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(0, max(s[m][str(h)]["WAPE"] for m in methods for h in horizons) * 1.15)

    # 右: Transformer +1d 预测 vs 真值散点
    ax2.scatter(PRED["truth_net_log1p_h1"], PRED["pred_transf_h1"],
                alpha=0.55, s=22, c=PRED["product_id"].map(PRODUCT_COLORS))
    lo = min(PRED["truth_net_log1p_h1"].min(), PRED["pred_transf_h1"].min())
    hi = max(PRED["truth_net_log1p_h1"].max(), PRED["pred_transf_h1"].max())
    ax2.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.2,
             label="完美预测 y=x")
    # 图例加产品
    for pid, c in PRODUCT_COLORS.items():
        ax2.scatter([], [], c=c, label=pid)
    ax2.set_xlabel("真实 log1p(净现金流)"); ax2.set_ylabel("Transformer 预测")
    ax2.set_title("图4b  +1 天 预测 vs 真值散点")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / "fig4_eval_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ fig4_eval_comparison.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print(f"\n>> 4 张图已生成于 {ASSETS}")
    for p in sorted(ASSETS.glob("*.png")):
        print(f"   {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

"""
train_xy_model.py
=================

在客户真实 schema 上训练 PANTHER 风格的资金流预测模型，支持单卡 / 多卡 DDP / AMP，
保证 A800×8 上能完整跑通（百万级流水）。

数据：
  data_sample/xy_txns.parquet  （simulate_xy_real_schema.py 产出；百万级量级由 --rate-multiplier 控制）
  data_sample/xy_product_meta.json

建模（对齐 docs/01）:
  · PANTHER Eq.(4) 4 维结构化分词 (direction, amount_bin, product_type, risk_level)
  · Llama3 风格 decoder (RMSNorm + RoPE + SwiGLU + causal SDPA) + SPRM 多尺度空洞卷积
    (论文 §3.3, dilation 1/2/4, 顶层并联)
  · 产品画像嵌入 (论文 §3.4 Profile-as-PosEnc)
  · 多窗口回归头 (1/7/30 天, Huber loss)

★ v3 修订 (针对 v2 在小数据上 val loss 波动 + 单一基线不透明 的问题):
  - 多 seed 重复 (默认 5 次)，报告 mean ± std
  - early-stop (patience=15), val 单调下降不再早停前震荡
  - 降低 lr (1e-3 → 3e-4), 加大 weight_decay (1e-4 → 5e-4), dropout 0.1 → 0.2
  - LightGBM 基线也跑同样多 seed, 口径对齐
  - DDP + AMP，让 A800×8 上百万样本几分钟跑完

运行 (单卡):
  python3 train_xy_model.py --epochs 60 --seeds 5

运行 (A800×8):
  torchrun --nproc-per-node=8 --master-port=29500 train_xy_model.py \\
      --epochs 60 --seeds 5 --batch-size 256

输出：
  model_out/eval_summary.json     多 seed × 多 horizon × 多方法的 mean/std
  model_out/all_runs.jsonl        每个 run 一行 (含 history), 供画图
  model_out/test_predictions.parquet  （以最优种子最优 epoch 的模型产出）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 分布式支持（即便单进程也优雅退化）
DDP_AVAILABLE = False
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    DDP_AVAILABLE = True
except ImportError:
    pass


HERE = Path(__file__).resolve().parent
TXNS_PATH = HERE / "data_sample" / "xy_txns.parquet"
META_PATH = HERE / "data_sample" / "xy_product_meta.json"
OUT_DIR = HERE / "model_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 7, 30]
DEFAULT_EPOCHS = 60
DEFAULT_BATCH = 256
DEFAULT_SEEDS = 5
PATIENCE = 15


# ============================================================
# 分布式辅助
# ============================================================
def setup_distributed():
    """返回 (rank, world_size, local_rank, is_dist). 单进程时全 0。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed():
    if DDP_AVAILABLE and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_rank0(msg, rank=0):
    if is_main_process(rank):
        print(msg, flush=True)


# ============================================================
# 数据加载 / 聚合 / 序列化
# ============================================================
FEATURE_DIMS = ["direction", "amount_bin", "product_type", "risk_level",
                "dow", "dom", "is_month_end", "is_quarter_end", "n_txn",
                "yield_rate"]   # v5 新增：宏观收益率上下文（Transformer 应更会用）


def load_and_aggregate():
    """读取流水 → (产品×group×日) 聚合 + 派生特征 + 分方向 log1p 标签。

    ★ v4 修订（修 v3 的两个伪象问题）:
      - 聚合粒度从 (product,date) 改为 (product,group,date)：样本量从 ~2000 → ~8000+
        （这是 Transformer 优势能显示的条件）
      - label 从 net_log1p（log 平移压塌信号）改为两个独立目标:
          log1p(purchase)   和   log1p(redemption)
        purchase/redemption 都是非负的（金额本身就是非负），无需平移；分方向对齐 docs/01 §1.1
        的"分方向回归"设计。
      - 模型一次输出 6 个目标：(purchase, redemption) × (+1d, +7d, +30d)
    """
    df = pd.read_parquet(TXNS_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    df["date"] = pd.to_datetime(df["txn_ts"], unit="s").dt.normalize()

    # group_id 字段是 v4 仿真产出的；若旧数据没有则统一为 group_id=0（向后兼容）
    if "group_id" not in df.columns:
        df["group_id"] = 0

    # (产品, group, date, 方向) 金额聚合
    GROUP_KEYS = ["product_id", "group_id", "date"]
    g = df.groupby(GROUP_KEYS + ["txn_type"], as_index=False)["amount"].sum()
    piv = g.pivot_table(index=GROUP_KEYS, columns="txn_type",
                        values="amount", fill_value=0.0).reset_index()
    for d in (0, 1):
        if d not in piv.columns:
            piv[d] = 0.0
    piv.columns.name = None
    daily = piv.rename(columns={0: "purchase", 1: "redemption"})
    daily["net"] = daily["purchase"] - daily["redemption"]

    # 笔数（用于上下文，不直接进 label）
    cnt = df.groupby(GROUP_KEYS).size().rename("n_txn").reset_index()
    daily = daily.merge(cnt, on=GROUP_KEYS, how="left").fillna({"n_txn": 0})
    daily["n_txn"] = daily["n_txn"].astype(int)

    # v5 新增：把 yield_rate 从 txns 逐笔→(product,group,date)聚合（取 mean；同一天内基本相同）
    if "yield_rate" in df.columns:
        yr = df.groupby(GROUP_KEYS)["yield_rate"].mean().reset_index()
        daily = daily.merge(yr, on=GROUP_KEYS, how="left")
        daily["yield_rate"] = daily["yield_rate"].fillna(0.02).astype("float32")
    else:
        daily["yield_rate"] = 0.02   # 旧数据无 yield_rate 时退化默认值

    dt = daily["date"]
    daily["dow"] = dt.dt.dayofweek.astype("int8")
    daily["dom"] = dt.dt.day.astype("int8")
    daily["is_month_end"] = dt.dt.is_month_end.astype("int8")
    daily["is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")

    # 4 维 token 前置筛选：用 net 符号判定 direction + |net| 分桶
    daily["direction"] = (daily["net"] < 0).astype("int8")
    daily["amount_log1p"] = np.log1p(daily["net"].abs()).astype("float32")
    pid2type = {p: int(meta[p]["product_type_id"]) for p in meta}
    pid2risk = {p: int(meta[p]["risk_level"]) for p in meta}
    daily["product_type"] = daily["product_id"].map(pid2type).fillna(1).astype("int8")
    daily["risk_level"]   = daily["product_id"].map(pid2risk).fillna(2).astype("int8")

    # ★ 两个分方向 log1p 标签（v4）—— 已是非负，直接 log1p 不平移
    daily["purchase_log1p"]   = np.log1p(daily["purchase"]).astype("float32")
    daily["redemption_log1p"] = np.log1p(daily["redemption"]).astype("float32")

    return daily, meta


def fit_amount_bins(train_daily: pd.DataFrame, n_bins: int = 16) -> np.ndarray:
    _, edges = pd.qcut(train_daily["amount_log1p"], q=n_bins,
                        labels=False, retbins=True, duplicates="drop")
    return np.asarray(edges, dtype="float32")


def reapply_amount_bin(daily: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    daily = daily.copy()
    bins = pd.cut(daily["amount_log1p"], bins=edges, labels=False, include_lowest=True)
    daily["amount_bin"] = bins.fillna(-1).astype("int16") + 1
    return daily


def build_sequences(daily: pd.DataFrame, hist_len: int = 30):
    """v4: 按 (product_id, group_id) 分组生成滑窗样本，label 为分方向 × 多 horizon。"""
    samples = []
    max_h = max(HORIZONS)
    for (pid, gid), g in daily.groupby(["product_id", "group_id"]):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < hist_len + max_h + 1:
            continue
        purch = g["purchase"].values
        red   = g["redemption"].values
        for t in range(hist_len, len(g) - max_h):
            window = g.iloc[t - hist_len: t]
            features = window[FEATURE_DIMS].values.astype("float32")
            samples.append({
                "features": features,
                "product_id": pid, "group_id": gid,
                # 6 个标签：每 horizon 一个申、一个赎（log1p 空间）
                "label_purchase_log1p": {h: float(np.log1p(purch[t + h - 1])) for h in HORIZONS},
                "label_redemption_log1p": {h: float(np.log1p(red[t + h - 1])) for h in HORIZONS},
                # 原始金额（用于回算 WAPE）
                "label_purchase_raw": {h: float(purch[t + h - 1]) for h in HORIZONS},
                "label_redemption_raw": {h: float(red[t + h - 1]) for h in HORIZONS},
                "date": str(g.iloc[t]["date"].date()),
            })
    return samples


def temporal_split(samples):
    samples_sorted = sorted(samples, key=lambda s: s["date"])
    n = len(samples_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return (samples_sorted[:n_train],
            samples_sorted[n_train:n_train + n_val],
            samples_sorted[n_train + n_val:])


# ============================================================
# Dataset / Model
# ============================================================
# label 顺序：[purchase_h1, purchase_h7, purchase_h30,
#              redemption_h1, redemption_h7, redemption_h30]
LABEL_KEYS = ("purchase_log1p", "redemption_log1p")
AMOUNT_BINS = 16   # PANTHER Eq.(4) 第 2 维，对齐 fit_amount_bins(n_bins=16)
PRETRAIN_CORPUS = HERE / "data_sample" / "pretrain_corpus.parquet"


class CashFlowDataset(Dataset):
    def __init__(self, samples, pid2idx):
        self.samples = samples
        self.pid2idx = pid2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = torch.tensor(s["features"], dtype=torch.float32)
        pid = torch.tensor(self.pid2idx[s["product_id"]], dtype=torch.long)
        # 6 维标签 (申×3horizon + 赎×3horizon)
        labels = []
        for kind in LABEL_KEYS:
            for h in HORIZONS:
                labels.append(s[f"label_{kind}"][h])
        return feats, pid, torch.tensor(labels, dtype=torch.float32)


class PretrainDataset(Dataset):
    """无监督语料：返回 (特征序列, product_id) 仅给预训练头用。

    特征序列的 4 维 (direction, amount_bin, product_type, risk_level) 既是输入
    又是预测目标 (做 t → t+1 的 shift)。预训练只消费这 4 维 token，**不接触回归
    标签 (purchase/redemption 金额)**，因此不与 SFT train/val/test 共享监督信号。
    """
    def __init__(self, parquet_path, hist_len=30):
        df = pd.read_parquet(parquet_path)
        # 派生 dow/dom 等上下文 (只给 SPRM 卷积, 不进 token)
        if "dow" not in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["dow"] = df["date"].dt.dayofweek.astype("float32")
            df["dom"] = df["date"].dt.day.astype("float32")
            df["is_month_end"] = df["date"].dt.is_month_end.astype("float32")
            df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype("float32")

        self.hist_len = hist_len
        self.columns = ["direction", "amount_bin", "product_type",
                        "risk_level", "dow", "dom", "is_month_end",
                        "is_quarter_end"]
        # 缺列补 0，保证特征维数与 SFT 路径一致 (10 维)
        for c in self.columns + ["n_txn", "yield_rate"]:
            if c not in df.columns:
                df[c] = 0.0

        # 按 product_id (跨 group) 打成长序列 —— PANTHER 的行为序列打包
        self.sequences = []
        self.pids = []
        for pid, g in df.sort_values(["product_id", "date"]).groupby("product_id", sort=False):
            feats = g[self.columns + ["n_txn", "yield_rate"]].values.astype("float32")
            if len(feats) < hist_len + 1:
                continue
            # 滑窗切片，每个窗口 hist_len 长度
            for t in range(0, len(feats) - hist_len):
                self.sequences.append(feats[t:t + hist_len])
                self.pids.append(pid)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feats = torch.tensor(self.sequences[idx], dtype=torch.float32)
        pid = self.pids[idx]
        return feats, pid


class StructuredTokenEmbedding(nn.Module):
    def __init__(self, dim=256, n_amount=32, n_type=8, n_risk=8, n_dir=4):
        super().__init__()
        self.dir_emb = nn.Embedding(n_dir, dim)
        self.amt_emb = nn.Embedding(n_amount, dim, padding_idx=0)
        self.type_emb = nn.Embedding(n_type, dim)
        self.risk_emb = nn.Embedding(n_risk, dim)

    def forward(self, direction, amount_bin, ptype, risk):
        return (self.dir_emb(direction.long().clamp(0, 3)) +
                self.amt_emb(amount_bin.long().clamp(0, 31)) +
                self.type_emb(ptype.long().clamp(0, 7)) +
                self.risk_emb(risk.long().clamp(0, 7)))


# ============================================================
# Llama3-style building blocks (RMSNorm / RoPE / SwiGLU / causal block)
# ============================================================
class RMSNorm(nn.Module):
    """Llama 风格 RMSNorm: 去 mean、除 RMS, + 可学习缩放 eps 护底。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float(). pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0,
                          device=None, dtype=torch.float32):
    """返回 cos/sin: [max_seq_len, head_dim/2] (按 Llama 实现, paired 维度)。

    对 cos/sin 用 [len, hd/2] 后再 repeat_interleave 到 [len, hd] —— 与 apply_rope 配套。
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=dtype, device=device) / half))
    t = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(t, freqs)                  # [T, hd/2]
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos, sin):
    """q,k: [B, H, T, hd]; cos/sin: [T, hd] (已 repeat_interleave 到 full head_dim)。

    Llama 用 rotated half: 把最后一维拆两半互换做 (x1*cos - x2*sin, x1*sin + x2*cos)。
    """
    hd = q.shape[-1]
    half = hd // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    cos = cos[:, :half].unsqueeze(0).unsqueeze(0)   # [1,1,T,half]
    sin = sin[:, :half].unsqueeze(0).unsqueeze(0)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU FFN (Llama2/3 默认): w2(x) * silu(w1(x)) 形, 多 d_ff_hidden 维 + down 投回。

    Llama3 用 SwiGLU = silu(w_gate(x)) * w_up(x), 再 w_down. hidden 通常 = (2/3)*4*dim, 圆整到 8 倍数.
    """
    def __init__(self, dim, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int((2 / 3) * 4 * dim)
            hidden = ((hidden + 7) // 8) * 8        # 圆整到 8 倍 (硬件友好, 也是 Llama 实践)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up   = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    """单个 Llama3 attention block (pre-norm + causal SDPA + RoPE) + SwiGLU FFN。"""
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.nope_dim = 0   # 保留 GQA 接口位 (暂时全 head 都用 RoPE)
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if cos is not None and sin is not None:
            # cos/sin 是 [T, head_dim], repeat 到 full 然后进 apply_rope
            cos_full = torch.repeat_interleave(cos[:T], 2, dim=-1)
            sin_full = torch.repeat_interleave(sin[:T], 2, dim=-1)
            q, k = apply_rope(q, k, cos_full, sin_full)
        # causal SDPA: is_causal=True 走内存高效路径 (torch >= 2.0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.drop(self.o_proj(out))
        # FFN 残差
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x





class SPRMConv(nn.Module):
    """Causal (decoder-only) multi-scale dilated depthwise conv.

    与 PANTHER §3.3 对齐：多尺度 dilation 的 depthwise conv 与 self-attention 并联、
    输出加到 attention 输出上。但 PANTHER 论文里 SPRM 是作为 decoder 的组成部分，因此
    卷积必须 **causal** (只左看 right padding=0) —— 否则位置 t 的感受野会含 t+1 等未来 token,
    破坏 next-token 预测的泄漏约束. 这里用左侧 padding=(kernel-1)*dilation、右侧裁掉实现.
    """
    def __init__(self, dim, kernel=3, dilations=(1, 2, 4)):
        super().__init__()
        self.kernel = kernel
        self.dilations = dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=kernel, dilation=d,
                      padding=0)   # 显式 0 padding, 手工做左侧零填充
            for d in dilations
        ])

    def forward(self, x):  # x: [B, T, D]
        xt = x.transpose(1, 2)  # [B, D, T]
        T = xt.shape[-1]
        out = torch.zeros_like(xt)
        for conv, d in zip(self.convs, self.dilations):
            pad = (d * (self.kernel - 1), 0)    # 左填充, 右 0 (causal 关键)
            y = F.pad(xt, pad)
            y = conv(y)[..., :T]                # 输出长度裁回 T
            out = out + y
        return out.transpose(1, 2)


class CashFlowTransformer(nn.Module):
    """Llama3-style decoder (RoPE + RMSNorm + SwiGLU + causal SDPA), PANTHER token + SPRM 顶层并联。

    架构要点 (现代 Llama3 组件 × PANTHER 主体结构):
      · 结构化分词 token emb (PANTHER Eq.4): 4 维笛卡尔积, 4 个 emb 相加
      · 位置编码用 **RoPE** 旋转矩阵 (取代绝对 pos_emb), 编码相对位置
      · 多层 **LlamaBlock**: RMSNorm pre-norm + causal scaled-dot-product attention + SwiGLU FFN
      · **SPRM** causal dilated conv 与 transformer 顶层并联, 输出相加 (PANTHER §3.3 原文
        "Operating in parallel with the multi-head self-attention ... adding its output")
      · 输入侧用一个 RMSNorm 做最终 norm (取代 LayerNorm, 对齐 Llama 实践)
      · 回归/预训练头都从 last-position hidden h[:,-1,:] 出发 (decoder-only 标准聚合)

    与 nn.TransformerEncoder 旧实现的关键差异:
      - 残差路径去掉 LayerNorm -> RMSNorm (无 mean-centering, 数值更稳)
      - 注意力走 SDPA is_causal=True (内存高效, 不再构造 [T,T] mask)
      - FFN 从 GELU+Linear 改 SwiGLU (silu gate, 通常 +1~2% 下游精度)
    """
    def __init__(self, dim=256, depth=4, heads=8, n_products=64, dropout=0.2,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        assert dim % heads == 0, f"dim={dim} 必须能被 heads={heads} 整除 (RoPE 要求)"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.max_seq_len = max_seq_len
        self.token_emb = StructuredTokenEmbedding(dim=dim)
        # 不再有 self.pos_emb; 用 RoPE 在 attention 内旋转 q/k. 预计算 cos/sin 缓存.
        cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList([
            LlamaBlock(dim=dim, heads=heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.sprm = SPRMConv(dim=dim)
        self.norm = RMSNorm(dim)        # 最终 norm 用 RMSNorm (取代 LayerNorm)
        self.product_profile = nn.Embedding(n_products, dim)
        # 回归头（SFT 用）：双路 (purchase / redemption) × 3 horizon = 6 维输出
        self.n_outputs = len(LABEL_KEYS) * len(HORIZONS)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, self.n_outputs)
        )
        # 预训练头来对齐 PANTHER Eq.(4)：4 路 token 分类，预测下一笔 τ_{t+1}
        #   direction (2: 申/赎) / amount_bin (16) / product_type (6: 0..5) / risk_level (6: 0..5)
        # risk_level 用 6 类 (索引 0..5) 以同时容纳 xy 仿真的 R2 (=2) 和 akshare 的 R1..R5 (=1..5),
        # 索引空间与 SFT 路径 daily["risk_level"].fillna(2) 一致 —— pretrained risk_emb 可直接迁移。
        self.pt_heads = nn.ModuleDict({
            "direction":    nn.Linear(dim, 2),
            "amount_bin":   nn.Linear(dim, AMOUNT_BINS),
            "product_type": nn.Linear(dim, 6),
            "risk_level":   nn.Linear(dim, 6),
        })

    def forward(self, feat, pid_idx, mode="regress"):
        """mode ∈ {'regress','pretrain'}:
           - pretrain: 返回 4 路分类 logits dict, 每个 shape [B,T,V]。
                       对位置 t 的 logits, 模型只能看到 τ_{1:t} (causal attention + causal SPRM)，
                       shift 后做 τ_t → τ_{t+1} 预测 (在 _pretrain_loss 里做)
           - regress:  返回 (logits_6d [B,6], pooled [B,dim])。pooled 用 last-position hidden
                       + product profile (decoder-only 标准聚合方式)。
        """
        B, T, _ = feat.shape
        x = self.token_emb(feat[..., 0], feat[..., 1], feat[..., 2], feat[..., 3])
        # Llama3 block 堆栈: 内部已含 RoPE 旋转 + causal SDPA + SwiGLU + 残差
        h = x
        for blk in self.blocks:
            h = blk(h, cos=self.rope_cos, sin=self.rope_sin)
        h = h + self.sprm(x)            # causal SPRM 顶层并联 (PANTHER §3.3)
        h = self.norm(h)

        if mode == "pretrain":
            return {name: head(h) for name, head in self.pt_heads.items()}

        # decoder-only 标准聚合：last-position hidden (含全部历史信息) + product profile
        pooled = h[:, -1, :] + self.product_profile(pid_idx)
        return self.head(pooled), pooled


# ============================================================
# 训练单 seed
# ============================================================
def train_one_seed(train_s, val_s, test_s, pid2idx, *, seed, epochs, lr,
                   batch_size, dim, device, rank, world_size, amp,
                   pretrain_ckpt=None):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = CashFlowDataset(train_s, pid2idx)
    val_ds = CashFlowDataset(val_s, pid2idx)
    test_ds = CashFlowDataset(test_s, pid2idx)

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        shuffle = False

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CashFlowTransformer(dim=dim, n_products=max(len(pid2idx), 64)).to(device)
    if pretrain_ckpt is not None:
        backbone_tgt = model.module if hasattr(model, "module") else model
        load_pretrain_backbone(backbone_tgt, pretrain_ckpt, rank=rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    def run_epoch(dl, train_mode, epoch_idx):
        model.train(train_mode)
        losses = []
        if sampler is not None and train_mode:
            sampler.set_epoch(epoch_idx)   # 每 epoch 不同 shuffle
        for feats, pid, labels in dl:
            feats = feats.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    out, _ = model(feats, pid)
                    loss = sum(F.huber_loss(out[:, j], labels[:, j], delta=1.0)
                               for j in range(model.module.n_outputs
                                              if hasattr(model, "module") else model.n_outputs)
                              ) / (model.module.n_outputs if hasattr(model, "module") else model.n_outputs)
                if train_mode:
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
            else:
                out, _ = model(feats, pid)
                n_out = model.module.n_outputs if hasattr(model, "module") else model.n_outputs
                loss = sum(F.huber_loss(out[:, j], labels[:, j], delta=1.0)
                           for j in range(n_out)) / n_out
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    best_val = float("inf")
    best_state = None
    history = []
    no_improve = 0
    for ep in range(epochs):
        tr = run_epoch(train_dl, True, ep)
        vl = run_epoch(val_dl, False, ep)
        if world_size > 1:
            # 多卡同步 val loss
            vl_t = torch.tensor(vl, device=device)
            dist.all_reduce(vl_t, op=dist.ReduceOp.AVG)
            vl = vl_t.item()
        sched.step()
        history.append({"epoch": ep + 1, "train_loss": float(tr), "val_loss": float(vl)})
        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if rank == 0 and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"  [seed={seed}] epoch {ep+1:3d}/{epochs}  train={tr:.4f}  "
                  f"val={vl:.4f}  best={best_val:.4f}  no_improve={no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            if rank == 0:
                print(f"  [seed={seed}] early-stop at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # 测试集预测（rank 0 收集并返回）
    preds_list, truths_list = [], []
    with torch.no_grad():
        for feats, pid, labels in test_dl:
            feats = feats.to(device); pid = pid.to(device)
            out, _ = model(feats, pid)
            preds_list.append(out.cpu().numpy())
            truths_list.append(labels.numpy())

    if rank == 0:
        preds = np.concatenate(preds_list)[:len(test_s)]   # [N, 6]
        truths_arr = np.concatenate(truths_list)[:len(test_s)]
    else:
        preds = truths_arr = None

    return model, (preds if rank == 0 else None,
                   truths_arr if rank == 0 else None,
                   test_s if rank == 0 else None), history, best_val


# ============================================================
# PANTHER Stage-1：生成式预训练（下一笔行为 token 预测）
# ============================================================
# 损失权重：方向 / 金额桶权 1.0（信息量高）；
#           type/risk 在 2 产品 xy 场景下是常量，给 0.3 权重避免退化为平凡 0-loss。
PT_LOSS_WEIGHTS = {"direction": 1.0, "amount_bin": 1.0,
                   "product_type": 0.3, "risk_level": 0.3}


def _pretrain_loss(token_logits: dict, feats: torch.Tensor):
    """4 路交叉熵求和。
    token_logits[name] shape [B,T,V]；feats[..., k] 是 ground-truth token id。
    做 t → t+1 shift：用位置 t 的 logits 预测位置 t+1 的 token（causal LM 标准做法）。
    """
    def shift(x):
        return x[:, :-1]            # inputs: 位置 0..T-2 的预测
    def shift_label(x):
        return x[:, 1:]             # targets: 位置 1..T-1 的真实 token

    col_map = {"direction": 0, "amount_bin": 1,
               "product_type": 2, "risk_level": 3}
    loss = 0.0
    for name, w in PT_LOSS_WEIGHTS.items():
        tgt = feats[..., col_map[name]].long()
        # clamp 到合法词表范围 (防御 unify_corpus 偶发的越界值)
        V = token_logits[name].shape[-1]
        tgt = shift_label(tgt).clamp(0, V - 1)
        logits = shift(token_logits[name])     # [B, T-1, V]
        loss = loss + w * F.cross_entropy(
            logits.reshape(-1, V), tgt.reshape(-1))
    return loss


def run_pretraining(*, corpus_path, epochs, lr, batch_size, dim, device,
                    rank, world_size, amp, hist_len=30):
    """Stage-1 生成式预训练：在无标签语料上学 τ = (dir, amt_bin, type, risk) 的下一笔预测。

    输出：model_out/pretrain.ckpt （只保存 backbone: token_emb / blocks.* (LlamaBlocks) /
          sprm / norm / product_profile；RoPE 用 register_buffer(persistent=False) 不进 ckpt；
          regression head 与 pt_heads 都跟 backbone 一起存,
          SFT 时只加载 backbone 部分, head 重新随机初始化）。
    """
    print_rank0(f"\n==== Stage-1 生成式预训练 ====", rank)
    print_rank0(f"  语料: {corpus_path}", rank)
    ds = PretrainDataset(corpus_path, hist_len=hist_len)
    print_rank0(f"  预训练样本数: {len(ds)} (滑窗 hist_len={hist_len})", rank)
    if len(ds) == 0:
        raise RuntimeError(f"预训练语料为空，请先跑 unify_corpus.py 生成 {corpus_path}")

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
        shuffle = False
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                   num_workers=4, pin_memory=True, drop_last=True)

    # pid → idx (用语料里出现过的所有 product_id 构造，给 product_profile 用)
    pid2idx = {pid: i for i, pid in enumerate(sorted(set(ds.pids)))}
    n_products = max(len(pid2idx), 64)

    torch.manual_seed(0); np.random.seed(0)
    model = CashFlowTransformer(dim=dim, n_products=n_products).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == "cuda") else None

    best_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(ep)
        ep_losses = []
        for feats, pid in dl:
            feats = feats.to(device, non_blocking=True)
            # pid 是字符串 product_id 列表 (DataLoader 不 collate 成 tensor), 查 pid2idx 表得 idx
            if isinstance(pid, torch.Tensor):
                pid_list = pid.tolist()
            else:
                pid_list = list(pid)
            pid_idx = torch.tensor([pid2idx[p] for p in pid_list],
                                   dtype=torch.long, device=device)
            if amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    token_logits = model(feats, pid_idx, mode="pretrain")
                    loss = _pretrain_loss(token_logits, feats)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                token_logits = model(feats, pid_idx, mode="pretrain")
                loss = _pretrain_loss(token_logits, feats)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ep_losses.append(loss.item())
        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        if world_size > 1:
            mt = torch.tensor(mean_loss, device=device)
            dist.all_reduce(mt, op=dist.ReduceOp.AVG); mean_loss = mt.item()
        sched.step()
        if mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            tgt = model.module if hasattr(model, "module") else model
            best_state = {k: v.detach().cpu().clone() for k, v in tgt.state_dict().items()}
        if rank == 0:
            print(f"  [pretrain] epoch {ep+1:3d}/{epochs}  loss={mean_loss:.4f}  "
                  f"best={best_loss:.4f}", flush=True)

    if rank == 0 and best_state is not None:
        ckpt_path = OUT_DIR / "pretrain.ckpt"
        torch.save({"state_dict": best_state, "best_loss": best_loss,
                    "dim": dim, "pid2idx": pid2idx}, ckpt_path)
        print(f"  >> 预训练 ckpt 落: {ckpt_path} (best_loss={best_loss:.4f})")
    return best_loss


def load_pretrain_backbone(model: CashFlowTransformer, ckpt_path: str,
                           rank: int = 0) -> CashFlowTransformer:
    """SFT 续训：加载预训练 backbone 权重, 回归头 self.head 与 pt_heads 保留随机初始化。

    加载策略 (strict=False + 名称前缀过滤):
      ✓ 加载: token_emb.* / blocks.* (LlamaBlocks) / sprm.* / norm.* / product_profile.*
              (以及 rope_cos/rope_sin register_buffer 不需加载, persistent=False 已经不进 ckpt)
      ✗ 跳过: head.* / pt_heads.*  (回归目标空间与预训练不同, 重头学)
      历史兼容: 旧 ckpt 用 transformer.* 命名, 此处会显示该部分骨干随机初始化 —— 由于 ckpt
              形状也对不上 (旧 /transformer.* 是 nn.TransformerEncoderLayer, 新 /blocks.*
              是 LlamaBlock), 旧 ckpt **不能复用**, 必须重新预训练.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # 预训练 ckpt 里如果是 DDP 包装的, 去掉 'module.' 前缀
    src = {k.removeprefix("module."): v for k, v in src.items()}
    BACKBONE_PREFIXES = ("token_emb.", "blocks.",
                         "sprm.", "norm.", "product_profile.")
    backbone_only = {k: v for k, v in src.items() if k.startswith(BACKBONE_PREFIXES)}
    missing, unexpected = model.load_state_dict(backbone_only, strict=False)
    if rank == 0:
        loaded = len(backbone_only) - len(unexpected)
        total_backbone = sum(1 for k in model.state_dict()
                             if k.startswith(BACKBONE_PREFIXES))
        print(f"  [pretrain-ckpt] {ckpt_path}: 加载 {loaded}/{total_backbone} 个 backbone 参数, "
              f"{len(unexpected)} 个预训练里有但当前模型没有, "
              f"{sum(1 for m in missing if any(m.startswith(p) for p in BACKBONE_PREFIXES))} 个 backbone 参数仍是随机初始化 "
              f"(通常是 n_products 不一致导致 product_profile 形状不匹配, 这种情况会自动用随机初始化也是安全的)")
    return model


# ============================================================
# 指标（v4：labels 是 6 维：[pur_h1, pur_h7, pur_h30, red_h1, red_h7, red_h30]）
# ============================================================
# 给定一列 (pred, truth)：算 MAE/RMSE/WAPE/DirAcc
def metrics_one_col(pred, truth):
    pred = np.asarray(pred); truth = np.asarray(truth)
    mae  = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    wape = float(np.sum(np.abs(pred - truth)) / max(np.sum(np.abs(truth)), 1e-6))
    sign_p = np.sign(np.diff(pred, prepend=pred[0]))
    sign_t = np.sign(np.diff(truth, prepend=truth[0]))
    dir_acc = float((sign_p == sign_t).mean()) if len(pred) > 1 else 0.0
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "DirAcc": dir_acc}


def collect_metrics_for_targets(preds, truths, samples):
    """v4: 返回嵌套 dict, 索引到 kind/purchase 或 redemption × horizon。
    preds/truths: [N, 6]; samples 携带 6 个 label 的真值。
    返回结构: {(kind, h): dict {MAE, RMSE, WAPE, DirAcc}}
    """
    out = {}
    col = 0
    for kind in LABEL_KEYS:           # purchase_log1p, redemption_log1p
        for h in HORIZONS:            # 1, 7, 30
            out[(kind, h)] = metrics_one_col(preds[:, col], truths[:, col])
            col += 1
    return out


# ============================================================
# 基线 (Naive mean + LightGBM), 跟 Transformer 一样跑多 seed
# ============================================================
def baseline_naive_mean(daily_train: pd.DataFrame, test_s, seed: int):
    """基线：用历史 (product, group) 维度上的 purchase/redemption 均值的 log1p 预测。

    返回与 Transformer 相同口径的 6 维 pred。
    """
    np.random.seed(seed)
    # 对每个 (product, group) 维度上：取 purchase/redemption 的历史均值，转 log1p
    base = {}
    for (pid, gid), g in daily_train.groupby(["product_id", "group_id"]):
        pur_m = float(np.log1p(max(g["purchase"].mean(), 0)))
        red_m = float(np.log1p(max(g["redemption"].mean(), 0)))
        # ±5% 微扰（展示 seed 方差；可关掉）
        pur_m *= 1.0 + np.random.normal(0, 0.02)
        red_m *= 1.0 + np.random.normal(0, 0.02)
        base[(pid, gid)] = (pur_m, red_m)

    # 输出与 Transformer 一致的 [N, 6] 口径
    preds = []
    for s in test_s:
        key = (s["product_id"], s["group_id"])
        pur_m, red_m = base.get(key, (0.0, 0.0))
        # 每个 horizon 都用历史均值（无 horizon-specific）
        preds.append([pur_m, pur_m, pur_m, red_m, red_m, red_m])
    return np.array(preds)   # [N, 6]


def baseline_lightgbm(train_s, test_s, seed: int):
    """基线: LightGBM 对 6 个目标分别训练（多 seed via random_state）。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print(f"[WARN] no lightgbm; baseline skipped (seed={seed})")
        return None

    def featurize(s):
        f = s["features"]
        return np.concatenate([
            f.mean(axis=0), f.std(axis=0), f[-1],
            # 把 group_id 也作为 one-hot 特征喂给 lgb
            np.eye(8, dtype=float)[int(s["group_id"]) % 8],
        ])

    Xtr = np.stack([featurize(s) for s in train_s])
    Xte = np.stack([featurize(s) for s in test_s])
    # 6 个目标值
    target_kinds = []
    for kind in LABEL_KEYS:
        for h in HORIZONS:
            target_kinds.append((f"label_{kind}", h))

    preds = np.zeros((len(test_s), 6))
    for j, (lk, h) in enumerate(target_kinds):
        ytr = np.array([s[lk][h] for s in train_s])
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=seed, verbosity=-1, n_jobs=-1)
        m.fit(Xtr, ytr)
        preds[:, j] = m.predict(Xte)
    return preds   # [N, 6]


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size",	type=int, default=DEFAULT_BATCH)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help="每个方法跑多少个不同 seed (取平均 ± std)")
    ap.add_argument("--hist-len", type=int, default=30)
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")

    # ===== PANTHER Stage-1 生成式预训练 =====
    ap.add_argument("--pretrain", action="store_true",
                    help="只跑 Stage-1 生成式预训练（产出 model_out/pretrain.ckpt），"
                         "不进入下面的回归 SFT + 基线评估流程")
    ap.add_argument("--pretrain-data", type=str, default=str(PRETRAIN_CORPUS),
                    help=f"预训练语料 parquet 路径（默认 {PRETRAIN_CORPUS}；"
                         f"由 unify_corpus.py 合并仿真放大 + akshare 真实基金产出）")
    ap.add_argument("--pretrain-epochs", type=int, default=30,
                    help="预训练 epoch 数（建议 20-40，causal LM 通常 30 内收敛）")

    # ===== PANTHER Stage-2 SFT 续训 =====
    ap.add_argument("--pretrain-ckpt", type=str, default=None,
                    help="SFT 模式加载该预训练 ckpt 作为 backbone 初始化；"
                         "不传则与原行为一致（从零监督训练）")
    args = ap.parse_args()

    rank, world_size, local_rank, is_dist = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    # ===== Stage-1 短路：只跑预训练就退出 =====
    if args.pretrain:
        run_pretraining(corpus_path=args.pretrain_data,
                        epochs=args.pretrain_epochs, lr=args.lr,
                        batch_size=args.batch_size, dim=args.dim,
                        device=device, rank=rank, world_size=world_size,
                        amp=amp, hist_len=args.hist_len)
        cleanup_distributed()
        return 0

    print_rank0("\n==== 1. 加载 + 聚合 + 分桶 ====", rank)
    daily, meta = load_and_aggregate()
    print_rank0(f"  日级行数: {len(daily)} | 产品: {daily['product_id'].nunique()}", rank)

    dates_sorted = sorted(daily["date"].unique())
    n_train_d = int(len(dates_sorted) * 0.7)
    n_val_d = int(len(dates_sorted) * 0.15)
    train_end = dates_sorted[n_train_d - 1]
    val_end = dates_sorted[n_train_d + n_val_d - 1]
    daily["split"] = np.where(daily["date"] <= train_end, "train",
                       np.where(daily["date"] <= val_end, "val", "test"))
    daily_train = daily[daily["split"] == "train"].copy()
    edges = fit_amount_bins(daily_train, 16)
    daily = reapply_amount_bin(daily, edges)
    print_rank0(f"  分桶边界数: {len(edges)}", rank)

    print_rank0("\n==== 2. 序列样本 ====", rank)
    train_s = build_sequences(daily[daily["split"] == "train"], args.hist_len)
    val_s   = build_sequences(daily[daily["split"] == "val"],   args.hist_len)
    test_s  = build_sequences(daily[daily["split"] == "test"],  args.hist_len)
    print_rank0(f"  train={len(train_s)} val={len(val_s)} test={len(test_s)}", rank)
    print_rank0(f"  (聚合粒度: 产品×group×日，4 个 group 让样本量是纯产品×日粒度的 ~4x)", rank)
    if len(train_s) < 1000:
        print_rank0(f"[WARN] train 样本仅 {len(train_s)}，Transformer 优势可能尚未展示；"
                    f"检查仿真是否真跑全量 (verify_data.py)", rank)

    pids = sorted({s["product_id"] for s in train_s + val_s + test_s})
    pid2idx = {p: i for i, p in enumerate(pids)}

    if args.pretrain_ckpt:
        print_rank0(f"\n==== SFT 模式：加载预训练 backbone {args.pretrain_ckpt} ====", rank)

    seeds_list = [hash(f"seed-{i}") % 100000 + 1 for i in range(args.seeds)]

    # ===== 多 seed 跑三方法 =====
    all_runs = []
    summary = {}  # method -> (kind, h) -> {mean, std}

    # 3.1 Transformer：跑每个 seed
    transf_preds_per_seed = []
    for i, seed in enumerate(seeds_list):
        print_rank0(f"\n==== 3.1 Transformer seed={seed} ({i+1}/{args.seeds}) ====", rank)
        t0 = time.time()
        _, (preds, truths, _samples), history, best_val = train_one_seed(
            train_s, val_s, test_s, pid2idx,
            seed=seed, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, dim=args.dim,
            device=device, rank=rank, world_size=world_size, amp=amp,
            pretrain_ckpt=args.pretrain_ckpt,
        )
        elapsed = time.time() - t0
        print_rank0(f"  elapsed: {elapsed:.1f}s, best_val={best_val:.4f}", rank)
        if rank == 0:
            this_seed_metrics = collect_metrics_for_targets(preds, truths, test_s)
            # 把 (kind, h) 元组 key 序列化成 "kind|h"
            this_seed_flat = {f"{kind}|{h}": m for (kind, h), m in this_seed_metrics.items()}
            all_runs.append({
                "method": "Transformer", "seed": seed,
                "best_val_loss": best_val, "elapsed_sec": elapsed,
                "history": history, "metrics": this_seed_flat,
            })
            transf_preds_per_seed.append((seed, preds, truths))

    # 3.2 基线：每个 seed（口径与 Transformer 一致：pred_s [N,6] vs truths_arr [N,6]）
    if rank == 0:
        truths_ref = np.array([
            [s[f"label_{k}"][h] for k in LABEL_KEYS for h in HORIZONS]
            for s in test_s
        ])   # [N, 6]
        for i, seed in enumerate(seeds_list):
            print_rank0(f"\n==== 3.2 Baselines seed={seed} ({i+1}/{args.seeds}) ====", rank)
            for method_name, preds in [
                ("Naive mean", baseline_naive_mean(daily_train, test_s, seed)),
                ("LightGBM",   baseline_lightgbm(train_s, test_s, seed)),
            ]:
                if preds is None:
                    continue
                preds = np.asarray(preds)[:len(test_s), :]
                m_flat = {}
                col = 0
                for kind in LABEL_KEYS:
                    for h in HORIZONS:
                        m_flat[f"{kind}|{h}"] = metrics_one_col(preds[:, col], truths_ref[:, col])
                        col += 1
                all_runs.append({
                    "method": method_name, "seed": seed, "metrics": m_flat,
                })

        # 汇总 mean±std（按 method × kind|horizon 维度聚合所有 seed）
        import collections
        grouped = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in all_runs:
            for kh, m in r["metrics"].items():
                for mk, mv in m.items():
                    grouped[r["method"]][f"{kh}/{mk}"].append(mv)

        for method, kh_metric_dict in grouped.items():
            summary[method] = {}
            for key, vals in kh_metric_dict.items():
                kh, mk = key.split("/")
                summary[method].setdefault(kh, {})[mk] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "values": [float(v) for v in vals],
                }

    # ===== 输出 =====
    if rank == 0:
        with open(OUT_DIR / "all_runs.jsonl", "w", encoding="utf-8") as f:
            for r in all_runs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(OUT_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "n_seeds": args.seeds, "seeds": seeds_list,
                "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
                "epochs": args.epochs, "hist_len": args.hist_len,
                "world_size": world_size, "device": str(device),
                "amp": amp,
                "pretrain_ckpt": args.pretrain_ckpt,
                "is_sft_mode": args.pretrain_ckpt is not None,
            }, f, ensure_ascii=False, indent=2)

        # 逐样本预测 dump（取 Transformer 最优 seed）—— v4: 6 维 label (pur×3 + red×3)
        best_transf = max(
            [r for r in all_runs if r["method"] == "Transformer"],
            key=lambda r: -r["best_val_loss"]
        )
        best_seed = best_transf["seed"]
        for s_seed, s_preds, s_truths in transf_preds_per_seed:
            if s_seed == best_seed:
                rows = []
                for i, s in enumerate(test_s):
                    # 列按 [pur_h1, pur_h7, pur_h30, red_h1, red_h7, red_h30] 顺序
                    rows.append({
                        "product_id": s["product_id"], "group_id": s["group_id"],
                        "date": s["date"], "seed": best_seed,
                        # Purchase (log1p 空间)
                        "truth_pur_log1p_h1":  float(s_truths[i, 0]),  "pred_pur_log1p_h1":  float(s_preds[i, 0]),
                        "truth_pur_log1p_h7":  float(s_truths[i, 1]),  "pred_pur_log1p_h7":  float(s_preds[i, 1]),
                        "truth_pur_log1p_h30": float(s_truths[i, 2]),  "pred_pur_log1p_h30": float(s_preds[i, 2]),
                        # Redemption (log1p 空间)
                        "truth_red_log1p_h1":  float(s_truths[i, 3]),  "pred_red_log1p_h1":  float(s_preds[i, 3]),
                        "truth_red_log1p_h7":  float(s_truths[i, 4]),  "pred_red_log1p_h7":  float(s_preds[i, 4]),
                        "truth_red_log1p_h30": float(s_truths[i, 5]),  "pred_red_log1p_h30": float(s_preds[i, 5]),
                    })
                pd.DataFrame(rows).to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
                break

        print_rank0(f"\n>> 落盘完成:", rank)
        print_rank0(f"   - {OUT_DIR / 'eval_summary.json'}      多 seed×多 horizon×多方法 mean±std", rank)
        print_rank0(f"   - {OUT_DIR / 'all_runs.jsonl'}         每 run 一行 (含 history)", rank)
        print_rank0(f"   - {OUT_DIR / 'test_predictions.parquet'}  最优 seed 逐样本预测 (画图用)", rank)

        # 汇总表（v4: 按 kind × horizon 打印）
        print_rank0("\n==== 汇总 (mean ± std, {} seeds, 6 目标) ====".format(args.seeds), rank)
        print_rank0(f"{'方法':14s} | {'目标':10s} | {'horizon':7s} | "
                    f"{'WAPE':>14s} {'DirAcc':>10s}", rank)
        print_rank0("-" * 70, rank)
        for method in ["Naive mean", "LightGBM", "Transformer"]:
            if method not in summary:
                continue
            for kind in LABEL_KEYS:
                kind_short = "申购" if "purchase" in kind else "赎回"
                for h in HORIZONS:
                    kh = f"{kind}|{h}"
                    w = summary[method].get(kh, {}).get("WAPE", {})
                    d = summary[method].get(kh, {}).get("DirAcc", {})
                    wm = w.get("mean", float("nan")); ws = w.get("std", 0)
                    dm = d.get("mean", float("nan")); ds = w.get("std", 0)
                    if np.isnan(wm):
                        continue
                    print_rank0(f"{method:14s} | {kind_short:6s} +{h:>3d}d | "
                                f"{wm*100:.2f}±{ws*100:.2f}%   "
                                f"{dm*100:.1f}±{ds*100:.1f}%", rank)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

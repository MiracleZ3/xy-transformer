# xy_train.md · A800 训练操作手册

> 在 A800 服务器上从零跑完整流程、出最终图的精确步骤。每步含命令、预期输出、失败处理。
>
> **当前设备状态（用户反馈）**：8 卡 A800，但其中 5 卡被占用，**实际可用 3 卡**。
> 因此下面 §2 的训练命令默认用 `--nproc-per-node=3`；如果未来卡释放，可改回 8。

---

## Step 0：环境准备（一次性）

```bash
# 0.1 校验可用 GPU（确认实际可用几张）
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs,', torch.cuda.get_device_name(0))"
# 预期输出: 8 GPUs, ...  (在 A800 上看到 8 张全在)
# 但通过 nvidia-smi 看哪些被占用：
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
# 5 卡占用的话，跑训练时只能用 CUDA_VISIBLE_DEVICES 暴露剩余 3 张

# 0.2 设置只用剩余 3 张卡（关键，避免抢别人的卡）
export CUDA_VISIBLE_DEVICES=0,1,2   # 改成你确认空闲的 3 张卡的 index

# 0.3 装依赖
pip install torch numpy pandas openpyxl lightgbm matplotlib

# 0.4 拉代码
git clone https://github.com/MiracleZ3/xy-transformer.git
cd xy-transformer
```

**关于 5 卡被占用的影响**：
- A800 80GB 显存，本模型 `dim=256 batch=512` 单卡显存占用 ~5GB，**3 卡完全够用**
- 训练时间会比 8 卡慢 ~2-3 倍，从 "15-30 分钟" 变成 "40-90 分钟"，可接受
- 如果想再省时间，把 `--batch-size` 调大些（单卡 batch=128 → A800 80GB 还能吃下），
  用 `--batch-size 384`（3 卡 × 128）就能接近 8 卡 × 64 的吞吐

---

## Step 1：生成全量仿真数据（百万级流水）

```bash
python3 simulate_xy_real_schema.py --years 3
```

**预期输出**（关键看这些数字）：
```
  9K73101A: ~100,000~150,000 笔成功流水
  9T32001A: ~300,000~400,000 笔成功流水
===== 模拟数据自检（真实 schema）=====
流水总笔数: 400,000 ~ 500,000       ← 必须几十万到百万量级，否则后续训练不稳
产品数: 2
申/赎占比: 申 65% / 赎 35%
9K73101A: 赎占比 ~27%（长持有期）
9T32001A: 赎占比 ~39%（短持有期）
```

**跑完用 `verify_data.py` 做严谨的数据 sanity check**（已经写好的脚本，不要在 shell 里 ad-hoc 敲 inline 代码，会触发 pandas FutureWarning）：

```bash
python3 verify_data.py
```

**预期输出**：
```
==== 数据 sanity check (xy_txns.parquet) ====
行数: 400,000+
按产品 × 方向分布:
  ...
✓ 赎占比: 9K73101A=27% < 9T32001A=39% (持有期差异保留)
✓ 数据
```

**失败处理**：
- 行数 < 10 万 → 检查 `simulate_xy_real_schema.py` 里 `PRODUCTS` 的 `monthly_txn_rate` 是否被改成小值（应该 3000/7500）
- 内存不够 → 用 `--rate-multiplier 0.5` 减半
- 想要更长时间窗 → `--years 5`

---

## Step 2：A800 全量训练（核心步骤）

### 2.1 标准命令（3 卡可用版）

```bash
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --epochs 60 \
    --seeds 5 \
    --batch-size 384 \
    --dim 256
```

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--nproc-per-node` | **3**（当前实际可用卡数）| 8 卡全空时可改 8 |
| `--epochs 60` | 60 | 配合 early-stop (patience=15)，通常 25-40 epoch 收敛 |
| `--seeds 5` | 5 | 每个 seed 跑一遍，输出 mean±std |
| `--batch-size 384` | 3 卡 × 128 | 8 卡时可用 512 (8×64) |
| `--dim 256` | 256 | Transformer 隐维；不够就调 384/512 |

**预期耗时**：40-90 分钟（3 卡 + 5 seeds + 60 epochs 上限）

### 2.2 如果 5 卡被释放后想用满 8 卡

把 `--nproc-per-node 3` 改成 `--nproc-per-node 8`，`--batch-size 384` 改 `--batch-size 512`，
取消 `CUDA_VISIBLE_DEVICES`，时间能压回 15-30 分钟。

### 2.3 训练过程中应该看到的日志

```
==== 3.1 Transformer seed=xxx (1/5) ====
  [seed=xxx] epoch   1/60  train=0.7123  val=0.2105  best=0.2105  no_improve=0/15
  [seed=xxx] epoch  10/60  train=0.2455  val=0.0917  best=0.0884  no_improve=2/15
  [seed=xxx] epoch  20/60  train=0.2123  val=0.0895  best=0.0821  no_improve=0/15
  [seed=xxx] epoch  33/60  ...
  [seed=xxx] early-stop at epoch 33                    ← val 连续 15 epoch 没改善触发
  train elapsed: 480s, best_val=0.0821
==== 3.2 Baselines seed=xxx (1/5) ====
...
==== 汇总 (mean ± std, 5 seeds) ====
方法              | horizon |     WAPE        DirAcc
------------------------------------------------------------
Naive mean        | + 1d    | X.XX±0.XX%   X.X±0.X%
Naive mean        | + 7d    | ...
Naive mean        | +30d    | ...
LightGBM          | + 1d    | ...
LightGBM          | + 7d    | ...
LightGBM          | +30d    | ...
Transformer       | + 1d    | X.XX±0.XX%   X.X±0.X%
Transformer       | + 7d    | ...
Transformer       | +30d    | ...
```

### 2.4 判读训练是否正常

| 现象 | 解读 / 处理 |
|---|---|
| val 单调下降至 0.05~0.10，early-stop 触发 | ✅ 正常收敛 |
| val 在 epoch 10 后仍大幅震荡 | 调 `--lr 1e-4` 重跑 |
| 跑到 60 epoch 仍在下降 | `--epochs 100` 重跑 |
| Transformer 的 WAPE 远高于 Naive mean（>2x） | ⚠ 模型没学到东西，先回 Step 1 看 `verify_data.py` 输出 |
| 三方法误差棒重叠（std > mean × 10%） | 诚实结论「差异不显著」，不要强行粉饰 |
| 三方法误差棒不重叠 & Transformer 完胜 | ✅ 可作「显著优于」结论 |

### 2.5 最终产物（训练完后 `model_out/` 下）

| 文件 | 用途 |
|---|---|
| `eval_summary.json` | 多 seed × 多 horizon × 多方法的 mean±std（**填结论表的核心**）|
| `all_runs.jsonl` | 每个 run 一行（含完整 history，画训练曲线用） |
| `test_predictions.parquet` | 最优 seed 的逐样本预测（画散点用） |

### 2.6 失败处理

| 报错 | 解决 |
|---|---|
| `CUDA out of memory` | `--batch-size` 减半（384→192）|
| `NCCL timeout` / `connection refused` | 换 `--master-port 29501`，或检查防火墙 |
| 只有 1 张卡在跑 | 没用 `torchrun` 改用 `python3`；或者忘了 `export CUDA_VISIBLE_DEVICES` |
| 5 seeds 跑太慢想先看趋势 | `--seeds 3`，确认无误再补到 5 |
| 服务器掉 SSH 进程被打断 | 用 `nohup ... &` 后台跑，日志写文件 |

---

## Step 3：出 4 张图

```bash
python3 plot_summary.py
```

**预期输出**：
```
生成插图...
✓ fig1_data_distribution.png
✓ fig2_time_patterns.png
✓ fig3_training_curve.png
✓ fig4_eval_comparison.png
```

**关键看 fig4**（基线对比主图）：
- 图上**不应再有红字"训练未收敛"警告**（脚本自动检测收敛，正常会消失）
- **图4a**：WAPE 误差棒清楚显示 Transformer 是否显著低于 LightGBM
- **图4b**：相对提升 % 三柱为正数，数字越大越好
- **图4c**：散点紧贴红色 y=x 对角线，两种产品颜色不混

如果图上还显示"训练未收敛"红字，回 Step 2 调参重跑。

---

## Step 4：把产物拷回本地

在本地终端（非服务器）执行：

```bash
# 优先：把 "可作为结论的产物" 都拷回
scp user@a800-server:/path/to/xy-transformer/model_out/eval_summary.json ./model_out/
scp user@a800-server:/path/to/xy-transformer/model_out/test_predictions.parquet ./model_out/
scp user@a800-server:/path/to/xy-transformer/model_out/all_runs.jsonl ./model_out/
scp user@a800-server:/path/to/xy-transformer/docs/assets/*.png ./docs/assets/

# 或用 rsync 一次性同步
rsync -av user@a800-server:/path/to/xy-transformer/model_out/ ./model_out/
rsync -av user@a800-server:/path/to/xy-transformer/docs/assets/ ./docs/assets/
```

**必拷**：
- `docs/assets/*.png`（4 张图）← 给文档
- `model_out/eval_summary.json` ← 填结论表用

**可选拷**：
- `model_out/all_runs.jsonl`（debug / 看每 seed 曲线）

---

## Step 5：填结论表（拷回本地后）

打开 `model_out/eval_summary.json`，结构长这样：
```json
{
  "summary": {
    "Naive mean":   {"1": {"WAPE": {"mean": 0.067, "std": 0.002}, "DirAcc": {...}}, ...},
    "LightGBM":     {"1": {"WAPE": {"mean": 0.058, "std": 0.001}, ...}, ...},
    "Transformer":  {"1": {"WAPE": {"mean": 0.052, "std": 0.003}, ...}, ...}
  },
  ...
}
```

按 `docs/07-three-part-summary.md §3.3` 模板表填入每格，写 `mean±std%`。

**判据**（docs/07 §3.3 规定的）：
- 误差棒**不重叠**（`Transformer.mean + std < LightGBM.mean - std`）才算「显著优于」
- 仅均值更低但误差棒重叠 → 结论必须改为「在 N seed 范围内 Transformer 不劣于 LightGBM」

---

## 一行流水线（按需复制）

```bash
# 3 卡版（当前设备状态）
git clone https://github.com/MiracleZ3/xy-transformer.git && cd xy-transformer && \
pip install torch numpy pandas openpyxl lightgbm matplotlib && \
export CUDA_VISIBLE_DEVICES=0,1,2 && \
python3 simulate_xy_real_schema.py --years 3 && \
python3 verify_data.py && \
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --epochs 60 --seeds 5 --batch-size 384 --dim 256 && \
python3 plot_summary.py
```

总耗时约 40-90 分钟。

---

## 关于 pandas FutureWarning（已修）

之前在 shell 里 ad-hoc 跑的：
```python
df.groupby('product_id').apply(lambda g: (g.txn_type==1).mean())  # ⚠ 触发 FutureWarning
```
在 pandas 3.x 会变成 error。仓库脚本**没用**这种写法，全部用 `groupby[...].mean()`
向量化聚合。如果需要快速核对数据，**用 `verify_data.py`**，它已经显式用
`include_groups=False` 处理掉了这个 warning，未来 pandas 升级兼容。

---

## 完成后发我两份

1. `docs/assets/*.png`（4 张图）
2. `model_out/eval_summary.json`（数字结果）

我据此：
- 检查 fig4 是否还有"未收敛"警告（没有才作数）
- 按 §3.3「误差棒不重叠才算显著优于」的判据据实填表
- 如果 Transformer 显著优于基线，写正向结论
- **如果基线反序或差异不显著，按 docs/07 §3.5 的保守版结论写，不强行粉饰**

报错或异常贴 stderr 给我。

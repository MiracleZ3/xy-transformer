# 在 A800×8 上跑全量训练的操作手册（run_on_a800.md）

> 目标：把仿真 + 训练 + 出图三段代码在 A800×8 服务器上完整跑通，产出可读为结论的图。
>
> **为什么需要这一步**：本机 CPU smoke-test 只验证"代码正确性"，不可作结论。
> 只有在 A800 上跑全量仿真（百万级流水）+ 多 seed 全量训练（60 epoch × 5 seed + early-stop）
> + DDP 加速（A800×8 把 hours 级压到 minutes 级），出来的数字才能进 docs/07 作结论。

---

## 0. 环境

```bash
# 推荐：最新稳定 torch（带 CUDA 12.x、NCCL）
pip install torch numpy pandas openpyxl lightgbm matplotlib

# 校验 GPU 可见
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs', torch.cuda.get_device_name(0))"
# 应该输出: 8 GPUs NVIDIA A800 ...
```

## 1. 拉代码 + 数据生成（前置）

```bash
git clone https://github.com/MiracleZ3/xy-transformer.git
cd xy-transformer

# 全规模仿真：百万级流水（默认 rate_multiplier=1.0）
python3 simulate_xy_real_schema.py --years 3
# 预计输出: xy_txns.parquet（数十~数百万行）
```

- 想要再堆量：`--rate-multiplier 3`（约 3 倍数据）
- 想跑更久历史：`--years 5`
- **不要用 `--small`**——那是 CPU smoke-test 用的

## 2. 全量训练（核心命令）

### 2.1 单卡跑（可作 fallback；会慢但能跑完）

```bash
python3 train_xy_model.py \
    --epochs 60 \
    --seeds 5 \
    --batch-size 256 \
    --dim 256
```

### 2.2 A800×8 DDP 跑（推荐）

```bash
torchrun --nproc-per-node=8 --master-port=29500 train_xy_model.py \
    --epochs 60 \
    --seeds 5 \
    --batch-size 512 \
    --dim 256
```

**参数含义**：
- `--epochs 60`：最多 60 epoch，配合 `--patience 15`（内置）早停，通常 25–40 epoch 收敛
- `--seeds 5`：每个方法跑 5 个不同 seed，输出 mean±std
- `--batch-size 512`：8 卡每卡 ~64，A800 80GB 显存绰绰有余
- `--dim 256`：Transformer 隐维（默认值）

**预期耗时**：
- 单卡（A800 单卡）：约 10–20 分钟/seed，总 1–2 小时
- **A800×8 DDP**：约 2–4 分钟/seed，**总 15–30 分钟**

## 3. 输出产物（训练完后）

`model_out/` 下三份关键文件：

| 文件 | 用途 |
|---|---|
| `eval_summary.json` | 多 seed × 多 horizon × 多方法的 mean/std（**作结论的核心数据**）|
| `all_runs.jsonl` | 每个 run 一行（含完整 history，画训练曲线用） |
| `test_predictions.parquet` | 最优 seed 的逐样本预测（画散点用） |

控制台同时打印汇总表，类似：

```
方法              | horizon |     WAPE     DirAcc
------------------------------------------------
Naive mean        | + 1d    | 6.7±0.2%   2.6±0.0%
LightGBM          | + 1d    | 5.8±0.1%   55±2%
Transformer       | + 1d    | 5.2±0.2%   55±3%
...
```

## 4. 出图

```bash
python3 plot_summary.py
# → docs/assets/fig{1,2,3,4}*.png
```

四张图：
1. **fig1_data_distribution.png**：金额分布 + 申赎占比（数据 sanity check）
2. **fig2_time_patterns.png**：周内活跃 + 月度节律（仿真质量证明）
3. **fig3_training_curve.png**：5 seed 训练曲线均值±std 带（审视收敛）
4. **fig4_eval_comparison.png**：WAPE 误差棒 + 相对提升% + 预测散点（**作结论的主图**）

## 5. 怎么把图拷回来

A800 服务器一般没图形界面，把 PNG 拷回本地看：

```bash
# 在本地执行
scp user@a800-server:/path/to/xy-transformer/docs/assets/*.png ~/Desktop/
# 或 rsync
rsync -av user@a800-server:/path/to/xy-transformer/docs/assets/ docs/assets/
```

## 6. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| OOM（显存溢出） | `--batch-size` 减半（256→128）|
| NCCL timeout | 检查 `--master-port` 是否被占用；防火墙开放 29500 |
| 只有 1 个 GPU 也在跑 | 没用 `torchrun`，改成 `python3 train_xy_model.py` |
| `--no-amp` 关闭混合精度 | 默认开启 AMP；如不兼容加 `--no-amp` |
| Transformer WAPE 远高于基线 | epoch 没跑够，加大 `--epochs 100`；或样本太少（看日志 `[WARN]`） |
| 训练曲线震荡不收敛 | 学习率偏大；改 `--lr 1e-4` |

## 7. 一行命令流水线（推荐）

```bash
# 仿真 → 训练 → 出图
python3 simulate_xy_real_schema.py --years 3 && \
torchrun --nproc-per-node=8 --master-port=29500 train_xy_model.py \
    --epochs 60 --seeds 5 --batch-size 512 --dim 256 && \
python3 plot_summary.py
```

完整跑完约 20–40 分钟（含仿真），然后把 `docs/assets/*.png` + `model_out/eval_summary.json` 拷回来即可。

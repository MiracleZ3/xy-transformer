# xy_train.md · A800 训练操作手册（两段式：生成式预训练 → 回归 SFT）

> 在 A800 服务器上跑完 PANTHER 两段式训练、出最终图的精确步骤。每步含命令、预期输出、失败处理。
>
> **当前设备状态**：8 卡 A800，其中 5 卡被占用，**实际可用 3 卡**。下面训练命令默认 `--nproc-per-node=3`；
> 卡释放后改 8 即可。
>
> **本手册当前路径**：先做 **Stage-1 生成式预训练**（在放大仿真 + akshare 真实基金语料上学"下一笔行为 token"），
> 再做 **Stage-2 回归 SFT**（用预训练 ckpt 初始化 backbone，5 seed × LightGBM 基线对比 + 多窗口回归）。
> 与 `docs/07-three-part-summary.md` 的当前结论 (LightGBM 6/6 WAPE 显著胜出，但 Transformer 方向命中率基本持平) 对齐；
> 本轮目标是验证"加上 PANTHER Stage-1 预训练之后能否把误差棒收窄、方向命中率能否反超"。
>
> **模型架构（Llama3 风格 decoder + PANTHER 主体）**：
> - 主体改用 **Llama3 组件**：RMSNorm（取代 LayerNorm）+ RoPE 旋转位置编码（取代绝对 pos_emb）
>   + SwiGLU FFN（取代 GELU）+ causal SDPA（`is_causal=True` 内存高效路径）
> - 自注意力恒走 causal mask（decoder-only，无开关）
> - SPRM 是 **causal** dilated conv（左 padding，只看历史），与 attention 块顶层并联、输出相加（PANTHER §3.3）
> - 回归头与预训练头都从最后一位置 hidden `h[:, -1, :]` 出发（decoder-only 标准聚合）
> - 本地已验证：RoPE 平移等变、各 block 与全模型 causal（扰动未来位置 → 过去位置输出 diff = 0）
> - ⚠ **架构变更后老 ckpt 不可复用**：上一版用 `nn.TransformerEncoder` + pos_emb，本版用 LlamaBlock + RoPE，
>   权重名称/形状都对不上。A800 上**必须重新跑 Stage-1 预训练**，不能加载上一版的 `model_out/pretrain.ckpt`

---

## Step −1：拉取新代码（每次代码更新都按这个走，**不会动 `model_out/` 训练产物**）

`model_out/` 在 `.gitignore` 里，git **不会**碰它——`eval_summary.json` / `all_runs.jsonl` / `pretrain.ckpt` / `*.png`
都安全。唯一可能挡住 `git pull` 的是 `data_sample/xy_txns.parquet` / `txns_real.parquet` 这两个**被跟踪**的小样本
（如果你在 A800 上重跑过仿真/akshare，新版会盖掉它们，造成"本地修改与远程冲突"）。下面三段按你情况选一：

### 如果只是想"先看看会不会冲突"（dry-run，不改任何文件）

```bash
cd xy-transformer
git fetch origin
git diff --stat HEAD origin/main      # 预览远程带来什么改动；输出空 = 已经最新
git status                            # 看本地有没有未提交的脏改动
```

### 安全拉取（最稳，**保留 `model_out/` 不删**）

```bash
cd xy-transformer

# 1) 把本地所有脏改动（含可能被改过的 data_sample/*.parquet）暂存隔离到 stash，**不删任何东西**
git stash push -u -m "a800-local-changes-before-pull-$(date +%Y%m%d)"

# 2) 此时工作区干净，git pull 丝滑通过
git pull origin main

# 3) (可选) 想看 stash 里存了什么：
git stash list
git stash show -p stash@{0}           # 看具体 diff
# 想恢复某个 stash 的改动(比如你想保留某次跑出来的 config)：git stash pop  或 git checkout stash@{0} -- <file>
```

`stash` 是非破坏性的：默认的 `git stash` 不会清掉 `.gitignore` 里的文件（`model_out/`、`data_sample/pretrain_corpus.parquet`
一直留在原地）；只有被 git **跟踪**的文件（`data_sample/*.parquet` 小样本 + 代码）改动会被暂存。

### 如果完全确定要"清重跑"（激进：丢弃所有本地脏改动）

```bash
cd xy-transformer
git fetch origin
git reset --hard origin/main          # ⚠ 会丢弃所有跟踪文件的本地修改 (不清 model_out/)
```

> `model_out/` 由于 gitignore 保护，**任何**拉取姿势（包括 `reset --hard`）都不会动它。
> 想清旧训练产物 → 在 §3 开跑前手动做：`mv model_out/* model_out_archive/ 2>/dev/null || true`。

---

## Step 0：环境准备（一次性）

```bash
# 0.1 校验可用 GPU（确认实际可用几张）
python3 -c "import torch; print(torch.cuda.device_count(), 'GPUs,', torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# 0.2 设置只用剩余 3 张卡（5 卡被占用时）
export CUDA_VISIBLE_DEVICES=0,1,2   # 改成你确认空闲的 3 张卡的 index

# 0.3 装依赖
pip install torch numpy pandas openpyxl lightgbm matplotlib akshare

# 0.4 拉代码
git clone https://github.com/MiracleZ3/xy-transformer.git
cd xy-transformer
```

**关于 5 卡被占用的影响**：
- A800 80GB 显存，本模型 `dim=256 batch=512` 单卡显存占用 ~5GB，**3 卡完全够用**
- 训练时间会比 8 卡慢 ~2-3 倍，可接受
- 想再省时间把 `--batch-size` 调大些，A800 80GB 还能吃下

---

## Step 1：生成预训练语料（两条线并行）

PANTHER Stage-1 是**无监督**的，需要比 SFT 大得多的语料。本仓用 **放大仿真 + akshare 真实基金** 两条并行：

### 1.1 放大仿真语料（主语料，~7.7M 笔）

```bash
python3 simulate_xy_real_schema.py --rate-multiplier 5 --years 3
```

把 `monthly_txn_rate` 整体 ×5（原本 12000/30000 → 6万/15万），3 年合计产出 **~7.7M 笔** xy schema 流水。
落盘到 `data_sample/xy_txns.parquet`。

> ⚠ **规模选择**：rate-multiplier 控制 pretrain 语料量。**不要用 rate-multiplier 50**——那会产出 76M 笔，
> 单 epoch 在 3 卡 A800 上要几小时。推荐 5 (8M 笔) 起步；想更快收敛用 2 (~3M) 也够；要更大可用 10 (~15M)。
> 默认 rate-multiplier=1 时是 ~1.5M 笔，对 pretrain 偏小但能跑通 smoke。

**预期输出关键点**：
```
流水总笔数: ~7,700,000（应到百万级；几十万也行但 pretrain 收敛会慢一些）
申/赎占比: 申 ~65% / 赎 ~35%
```

### 1.2 akshare 真实基金语料（领域对齐微语料，~38k 笔）

```bash
python3 fetch_fund_flow.py --n-etfs 60 --years 3
```

从 akshare 拉取成交额排名前 60 的 ETF 日 K，3 年回看，产出 ~38k 行真实市场资金流（落 `data_sample/txns_real.parquet`）。

**注意**：
- 这是**真实市场数据**，给预训练模型"固收产品的资金流动力学"先验，但与 xy 的真实客户层申赎**分布不完全一致**
- akshare 拉取耗时 ~10-15 分钟（限频 + 60 次 HTTP），建议 `nohup ... &` 后台跑
- 如果网络不通或被 ban：本语料**可以跳过**，只靠放大仿真也能完成 Stage-1（详见 §1.4）

### 1.3 合并两类语料成预训练统一 schema

```bash
python3 unify_corpus.py
```

把两类语料 coax 进同一个 4 维 token schema（`direction / amount_bin / product_type / risk_level`），
统一 16 桶 quantile 分桶（按 direction 分别 fit），落 `data_sample/pretrain_corpus.parquet`。

**预期输出关键点**：
```
[load] 仿真语料 ... → ~7,700,000 行
[load] akshare 语料 ... → ~38,000 行
总行数: ~7,740,000
direction: 申 ~65% / 赎 ~35%
amount_bin 分布 (均匀为佳): {0: ..., 1: ..., ..., 15: ...}    ← 各桶笔数接近
product_type 分布: {1: 大头(固收), 2: 中等(ETF 股基), ...}
risk_level 分布:   {2: 大头(xy R2), 5: 中等(ETF R5), ...}
```

只要 `amount_bin` 各桶大致均匀（不集中在某一个桶），就说明分桶成功。

### 1.4 数据完整性校验

```bash
python3 verify_data.py
```

校 `xy_txns.parquet` 的 4 个 group 维度（RETAIL_APP/RETAIL_OTC/HNW/INSTITUTIONAL）+ 收益率字段。
只要**4 个 group 的中位金额呈递增**（零售 < 高净值 < 机构）就说明仿真合理。

### 1.5 防数据泄漏（关键合规说明）

PANTHER 两段式天然的隔离机制，本流程严格执行：

| 阶段 | 数据 | 信号 |
|---|---|---|
| **Stage-1 预训练** | `pretrain_corpus.parquet`（~7.7M 仿真 + 38k ETF） | **只吃 4 维 token**（方向/金额桶/产品类型/风险等级），**不接触任何金额回归标签** |
| **Stage-2 SFT** | `xy_txns.parquet` 的时间切分 70/15/15（同 SFT 流程），n_train≈5-20k | 6 维 `log1p(purchase)/log1p(redemption)` × 3 horizon |

预训练语料和 SFT 数据**在产品 ID 上可能重叠**（仿真语料用了相同的 9K.../9T... 产品），但这不是泄漏：
PANTHER 的设计前提就是 "无监督学行为表示 → 监督任务上微调"，监督信号只来自 Stage-2。预训练阶段**永远不会看到 SFT 标签**。

---

## Step 2：Stage-1 生成式预训练（A800 核心）

### 2.1 标准命令（3 卡可用版）

```bash
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --pretrain \
    --pretrain-data data_sample/pretrain_corpus.parquet \
    --pretrain-epochs 30 \
    --dim 256 \
    --batch-size 384
```

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--pretrain` | (flag) | 触发 Stage-1 短路：跑预训练然后退出，不进 SFT 评估流程 |
| `--pretrain-data` | 默认值即可 | 上一 step 产出的统一语料 |
| `--pretrain-epochs 30` | 20–40 | causal LM 在 ~7.7M+38k 语料上通常 20–30 内收敛 |
| `--dim 256` | 256 | 与 Stage-2 SFT **必须一致**，否则 backbone 权重形状不匹配 |
| `--batch-size 384` | 3 卡 × 128 | 8 卡可用 512 |
| `--no-amp` | 默认开 AMP | 出 NaN 才关 |

> **必须与 SFT 共用 `--dim`**——预训练 ckpt 是按 dim 形状存的，加载时会校验形状，dim 不一致会跳过加载 backbone。

### 2.2 训练过程中应该看到的日志

```
==== Stage-1 生成式预训练 ====
  语料: .../pretrain_corpus.parquet
  预训练样本数: ~790,000 / hist_len=30
  [pretrain] epoch   1/30  loss=3.51  best=3.51
  [pretrain] epoch   5/30  loss=2.83  best=2.83
  [pretrain] epoch  15/30  loss=2.41  best=2.41
  [pretrain] epoch  25/30  loss=2.27  best=2.27
  [pretrain] epoch  30/30  loss=2.24  best=2.24
  >> 预训练 ckpt 落: model_out/pretrain.ckpt (best_loss=2.24)
```

### 2.3 判读预训练是否正常

| 现象 | 解读 / 处理 |
|---|---|
| loss 从 3.5 单调下降到 ~2.2–2.5 | ✅ 正常收敛 |
| loss 停在 3.5 不动 | 语料突然 collapse：检查 unify_corpus.py 输出某桶是否占 100% |
| loss 降到 ~3.0 以下 | 方向/金额桶预测能学了，迁移到 SFT 通常有效果 |
| 30 epoch 没触底 | `--pretrain-epochs 50` 重跑 |
| `dim mismatch` 报错 | SFT 用了不同的 `--dim`，重训 SFT 时保持一致 |

**预训练目标的下界参考**：4 路 cross-entropy 求和；理论随机基线 = `log(2)+log(16)+log(6)+log(6) ≈ 5.40`。
收敛到 ~2.2 表示"比随机猜下一笔显著好"；稳态 loss 越低通常 SFT 收益越大，但不绝对（预训练 ≠ 下游 task gain）。

### 2.4 失败处理

| 报错 | 解决 |
|---|---|
| `CUDA out of memory` | `--batch-size` 减半 |
| `预训练语料为空` | Step 1 没产出 `pretrain_corpus.parquet`，重新跑 unify_corpus.py |
| `MMCV / NCCL timeout` | 换 `--master-port 29501` |

---

## Step 3：Stage-2 回归 SFT（A800 核心）

### 3.1 标准命令（用预训练 ckpt 初始化）

```bash
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --pretrain-ckpt model_out/pretrain.ckpt \
    --epochs 60 \
    --seeds 5 \
    --batch-size 384 \
    --dim 256
```

> 与原"从零训练"的唯一区别就是加 `--pretrain-ckpt`。其余照旧：5 seed × LightGBM × Naive mean + 6 目标评估。

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--pretrain-ckpt` | 必填 | 加载 Stage-1 backbone (token_emb / blocks / sprm / norm / product_profile)；回归头与 pt_heads 都重新随机初始化 |
| `--epochs 60` | 60 | 配合 early-stop (patience=15)，通常 25-40 epoch 收敛 |
| `--seeds 5` | 5 | 每个 seed 重新加载同一份 ckpt，输出 mean±std |
| `--batch-size 384` `--dim 256` | 与 Stage-1 一致 | |

**预期耗时**：40-90 分钟（3 卡 + 5 seeds + 60 epochs 上限）

### 3.2 训练过程中应该看到的日志

```
==== SFT 模式：加载预训练 backbone model_out/pretrain.ckpt ====
  [pretrain-ckpt] ...: 加载 62/62 个 backbone 参数, 0 个预训练里有但当前模型没有,
                        0 个 backbone 参数仍是随机初始化 (...)
==== 1. 加载 + 聚合 + 分桶 ====
  日级行数: ~31,800 | 产品: 2
==== 2. 序列样本 ====
  train=~5,650 val=~832 test=~840
==== 3.1 Transformer seed=xxx (1/5) ====
  [seed=xxx] epoch  1/60  train=0.41  val=0.10  best=0.10  no_improve=0/15
  ...
==== 汇总 (mean ± std, 5 seeds) ====
...
```

ckpt-load 日志那一行**必须显示 `加载 62/62 个 backbone 参数`**，否则权重没迁上。

### 3.3 判读 SFT 是否有效

参考 `docs/07-three-part-summary.md §3.3 / §3.4` 的判据：

| 现象 | 解读 |
|---|---|
| Transformer 6/6 目标 WAPE **误差棒与 LightGBM 接近重叠** | ✅ 预训练把差距收窄；可写入"两种方法在该规模下能力接近"的结论 |
| 任一目标 WAPE **误差棒完全不重叠且 Transformer 更优** | ✅ 显著优于；写正向结论 |
| 所有目标仍 LightGBM 显著优于 Transformer、方向命中率无变化 | ⚠ 预训练在这个规模/规律下收益不足；承认当前设置仍对 Tree 最有利 |
| 6 目标方向命中率 Transformer 全部接近或反超 LightGBM | "方向信号上学到了真实信息"——即使幅值输掉，方向判断上的价值已有 |

**不预设结论**：误差棒判据是客观的（Transformer.mean + std < LightGBM.mean - std 才算显著优于），据实写。

### 3.4 失败处理

| 报错 | 解决 |
|---|---|
| `dim mismatch` 加载跳过 | Stage-1/Stage-2 用了不同 `--dim`，重训保证一致 |
| `加载 0/62 个 backbone 参数` | ckpt 路径错或被覆写；重训 Stage-1 |
| val loss 比从零训练还差 | 预训练语料 collapse；回 Step 1 看 unify 输出 |
| 5 seeds 太慢 | `--seeds 3` 先跑通 |

---

## Step 4：出图

```bash
python3 plot_summary.py
```

**4 张图**（fig1/2 用 data_sample/xy_txns.parquet，fig3/4/5 用 model_out 的 SFT 产物）：
- fig3 训练曲线
- fig4 三方法 WAPE 对比 + 误差棒
- fig5 散点（最优 seed 逐样本预测 vs 真值）

**关键看 fig4**：误差棒是否**变窄了**（比 docs/07 §3.3 旧结果），以及 Transformer 是否与 LightGBM 误差棒更接近重叠。

---

## Step 5：把产物拷回本地

在本地终端（非服务器）：

```bash
# 必拷（含预训练 ckpt 用于复现 SFT）
scp user@a800-server:/path/to/xy-transformer/model_out/eval_summary.json ./model_out/
scp user@a800-server:/path/to/xy-transformer/model_out/test_predictions.parquet ./model_out/
scp user@a800-server:/path/to/xy-transformer/model_out/all_runs.jsonl ./model_out/
scp user@a800-server:/path/to/xy-transformer/model_out/pretrain.ckpt ./model_out/
scp user@a800-server:/path/to/xy-transformer/docs/assets/*.png ./docs/assets/
```

---

## Step 6：填结论表（拷回本地后）

打开 `model_out/eval_summary.json`，**先看是否含 `is_sft_mode: true`**（确认本轮是 SFT 模式而非误跑从零训练）。
然后按 `docs/07-three-part-summary.md §3.3` 模板表填 6 目标 × 3 方法的 mean±std。

**判据**（docs/07 §3.4）：
- 误差棒**不重叠**（`Transformer.mean + std < LightGBM.mean - std`）才算「显著优于」
- 仅均值更低但误差棒重叠 → 必须改为「在 N seed 范围内 Transformer 不劣于 LightGBM」

---

## 一行流水线（按需复制）

```bash
# 完整两段式（3 卡）
git clone https://github.com/MiracleZ3/xy-transformer.git && cd xy-transformer && \
pip install torch numpy pandas openpyxl lightgbm matplotlib akshare && \
export CUDA_VISIBLE_DEVICES=0,1,2 && \
# Stage-1 语料
python3 simulate_xy_real_schema.py --rate-multiplier 5 --years 3 && \
python3 fetch_fund_flow.py --n-etfs 60 --years 3 && \
python3 unify_corpus.py && \
python3 verify_data.py && \
# Stage-1 预训练
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --pretrain --pretrain-data data_sample/pretrain_corpus.parquet \
    --pretrain-epochs 30 --dim 256 --batch-size 384 && \
# Stage-2 SFT
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --pretrain-ckpt model_out/pretrain.ckpt \
    --epochs 60 --seeds 5 --batch-size 384 --dim 256 && \
python3 plot_summary.py
```

总耗时 ~90 分钟（语料生成 ~15 分钟 + 预训练 ~30 分钟 + SFT ~40 分钟 + 出图 ~1 分钟）。

---

## 关于 pandas FutureWarning（已规避）

仓库脚本全部用 `groupby[...].mean()` 向量化聚合，不触发 pandas 3.x 的 `groupby.apply` FutureWarning。
如需快速核对数据，用 `verify_data.py`（已显式 `include_groups=False`）。

---

## 完成后发我两份

1. `docs/assets/*.png`（4 张图，含 fig4 误差棒对比）
2. `model_out/eval_summary.json`（数字结果，**需含 `is_sft_mode: true`**）

我据此：
- 检查 fig4 是否有"未收敛"警告（没有才作数）
- 按 §3.4「误差棒不重叠才算显著优于」的判据据实填 §3.3 表
- 与 docs/07 当前结论（without pretrain）做 A/B 对比，看预训练带来的 delta
- **如果基线反序或差异不显著，按 docs/07 §3.5 保守版结论写，不强行粉饰**

报错或异常贴 stderr 给我。

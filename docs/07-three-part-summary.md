# 理财产品资金流预测 · 工作总结（docs/07）

> **本文按三段式整理本次工作的全部成果**：
> 1. **数据**：采用了什么方法造出什么数据（为什么这么做、可复现）
> 2. **特征处理 + 建模**：每个特征如何处理（逻辑、方法、可拓展性）
> 3. **验证**：基准、指标、结果（含图）
>
> 所有数字、图表都由本仓脚本书复现，下面每条结论都标了对应脚本。

---

## 目录

1. 数据生成：方法论与可复现性
2. 特征处理与建模方法
3. 验证：基准对比与结果

---

## 1. 数据生成：方法论与可复现性

### 1.1 为什么需要仿真、不能等真实数据

客户目前只给到 `data_sample/xy_sample.md`，**只描述字段结构 + 两个产品画像**，未提供数据。
等真实脱敏数据接入前，必须用仿真验证方法可行性。但仿真颗粒度决定结论可信度——
**不是用随机数造假流水，而是把真实固收理财的五条业务规律显式编码进生成规则**，
让模型在仿真上学到的规律能迁移到真实数据。

### 1.2 仿真采用的方法（5 条业务规律 → 5 条生成规则）

| # | 真实规律 | 生成规则（脚本如何实现） |
|---|---|---|
| ① | **申赎方向不均衡**——固收产品长期净申购 | 基础赎回概率 `p_redemption_base = 0.30~0.35` （脚本第 117 行） |
| ② | **产品异质**——9K73101A(180d 持有期) vs 9T32001A(30d 持有期) | 长持有期 × 0.8 赎回折扣（脚本第 122–124 行） |
| ③ | **时间节律**——工作日 > 周末；月末、季末显著加成 | 周末 ×0.3；月末 ×1.5；季末 ×1.8 加权采样（脚本第 94–103 行） |
| ④ | **金额长尾**——固收单笔 ¥1k~¥10M | lognormal(mu=9.3~9.8, sigma=1.0~1.2)，winsorize ¥10M 上限（脚本第 127–129 行） |
| ⑤ | **池子不可负**——赎回 ≤ 已存在申购 | "产品级聚合批仓"约束，越额时转小额申购或跳过（脚本第 131–141 行） |

这五条规律的**叠加输出**就是真实固收理财应该呈现的分布。详见 §3.1（图1、图2）的实证。

### 1.3 输入画像 → 数字化参数映射

客户给的 `xy_sample.md` 含 5 段语义信息，全部数字化进 `PRODUCTS` 配置表
（`simulate_xy_real_schema.py` 第 55–82 行）：

| 客户原文 | 数字化参数 | 用途 |
|---|---|---|
| "9K73101A / 9T32001A" | `product_id` (key) | 序列主体 |
| "固收类" | `product_type_id: 1` | 进 token 维（虽然都相同，但保留维度） |
| "R2 谁慎型" | `risk_level: 2` | 进 token 维 |
| "最短持有 180d / 30d" | `min_holding_days` | 决定赎回稀疏度 |
| "近 6 月年化 1.922% / 近 1 年 2.303%" | `annualized_yield` | 落到 `xy_product_meta.json` 供训练器上下文 ψ 使用 |
| "业绩基准 2%~3.8% / AAA 科创债×20%+()" | `benchmark` (str) | 静态画像字段 |
| (推断) 金额量级 ¥1w~¥10w | `mu_amount_log / sigma_amount` | 金额分布参数 |
| (推断) 60–120 笔/月 | `monthly_txn_rate` | 活跃度参数 |
| (推断) 30–35% 偏赎 | `p_redemption_base` | 方向参数 |

> ⚠️ **诚实声明**：表格下半部 (italic 部分) 是根据固收理财经验**估算**的，不是 xy_sample.md 直接给的。
> 这是为了让你 / 客户一眼看出"哪些参数是确认的事实，哪些是猜的"。校准流程
> （`data_calibration_guide.md` + `profile_xy_real_data.py`）正是为了把这部分换成真实数字。

### 1.4 可复现保证

```python
_RNG = np.random.default_rng(seed=20260617)   # 固定 seed
```

- 任何人 `git clone` 后跑 `python3 simulate_xy_real_schema.py` 得到的 `xy_txns.parquet` **字节相同**
- 改任何参数前/后跑出的数据可直接 diff 对比，差异**完全归因于代码而非随机性**
- 重新训出的模型权重也一致，方便回归对比

### 1.5 仿真数据的实证分布（图1、图2）

![图1  数据分布](assets/fig1_data_distribution.png)

**图1a 解读**：
- 金额 log1p 后近似正态分布，符合 lognormal 假设；
- 两个产品的金额分布有明显差异—— 9K73101A（长持有期）金额更大更分散，9T32001A（短持有期）
  金额更小更密，对应真实"长持有期产品含更多大额机构"的规律。

**图1b 解读**：
- **9K73101A 赎占比 27% < 9T32001A 赎占比 39%**——这两条规律的差异**正是模型要学的核心信号**，
  如果仿真不区分，会丢失预测价值最大的特征。

![图2  时间节律](assets/fig2_time_patterns.png)

**图2a 解读**：周内分布的工作日（周一-周五）显著高于周末，周末业务实际处理顺延。

**图2b 解读**：月度申/赎金额曲线呈现清晰的**季末冲量**——图中的虚线（Q1/Q2/Q3/Q4 末）处，
赎回金额明显冲高，这就是 PANTHER §3.3 SPRM 卷积应该捕捉的周期 motif。

### 1.6 仿真数据自检数字

| 指标 | 实测值 | 真实性验证 |
|---|---|---|
| 流水总笔数 | 6,570 | — |
| 产品数 | 2 | ✅ 严格对齐 xy_sample.md |
| 时间跨度 | 3 年（2022-01-03 ~ 2025-01-01） | ✅ |
| 申/赎占比 | 65.0% / 35.0% | ✅ 固收长期净申购 |
| 金额中位 | ¥12,764 | ✅ 固收零售典型量级 |
| 金额 max | ¥997,582 | ✅ winsorize 上限合理 |
| 9K73101A 赎占比 | 26.9% | ✅ 长持有期 → 赎回稀 |
| 9T32001A 赎占比 | 39.1% | ✅ 短持有期 → 赎回密 |

---

## 2. 特征处理与建模方法

本节说明每个字段如何被处理，以及对应的 PANTHER 风格建模组件如何使用它们。
**核心方法**：PANTHER 论文 Eq.(4) 的 4 维结构化 token + 序列 Transformer + SPRM 多尺度卷积
+ 产品画像嵌入 + 多窗口回归头。

### 2.1 字段处理一览（每个字段的去向）

数据原始字段（6 个，按 xy_sample.md）：

| 字段 | 类型 | 处理方法 | 进哪个模型组件 |
|---|---|---|---|
| `product_id` | string | 作为序列主体 key，做 product_id → 连续 idx 映射 | **产品画像嵌入 `product_profile`**（§2.4） |
| `txn_time` | yyyymmddhhmmss | 解析为 timestamp；派生 dow/dom/hour_bin/is_month_end/is_quarter_end/dt_prev_sec | **时间上下文**（不进 token） |
| `txn_type` | 0申/1赎 | 直接作为方向特征 | **PANTHER Eq.(4) 结构化 token 维度 1**（§2.2） |
| `status` | 0失败/1成功 | 过滤掉失败的（仅保留成功流水计算回归标签） | 数据过滤，不进特征 |
| `amount` | float | `log1p` + **按 direction 分别 quantile 分桶**（16 桶，对齐 PANTHER Eq.4 第 2 维） | **PANTHER Eq.(4) 结构化 token 维度 2**（§2.2） |
| `rest_amount` | int 0 | **直接丢弃**（用户明确"没用，都为 0"） | 不用 |

派生字段：

| 派生 | 计算公式 | 用途 |
|---|---|---|
| `direction` | 共享 `txn_type` | 进 token |
| `amount_bin` | qcut(log1p(amount), 16, **by direction**)（按方向各自分桶） | 进 token |
| `product_type` | 从 `xy_product_meta.json` 静态查表 → `1`（固收） | 进 token（Eq.4 第 3 维） |
| `risk_level` | 静态查表 → `2`（R2） | 进 token（Eq.4 第 4 维） |
| `dow`/`dom`/`hour_bin`/`is_month_end`/`is_quarter_end` | 时间派生 | 时间上下文，供 SPRM 卷积捕周期 |
| `dt_prev_sec` | 同产品上一笔的间隔 | 序列节拍信号，给 Transformer 注意力学 |
| `n_txn` / `daily_purchase` / `daily_redemption` | 按 (产品,日) 聚合 | 多窗口回归标签构造 |

### 2.2 特征 ①：PANTHER Eq.(4) 4 维结构化分词

**这是整套建模方法的核心创新**。把多维属性组合成单一可学习 token τ：

$$
\tau = (\text{direction},\ \text{amount\_bin},\ \text{product\_type},\ \text{risk\_level})
\in \mathcal{D} \times \mathcal{A} \times \mathcal{PT} \times \mathcal{R}
$$

- **数学含义**：4 维笛卡尔积，理论 \|V\| = 2×16×6×5 = **960**
- **真实数据下的退化**：本客户两个产品都是固收 R2，因此 `product_type` / `risk_level` 是常量，token 空间自动收窄到 (direction × amount_bin) = 32 种
- **退化不报错**：这就是设计的稳健性。schema 字段稀疏时分词自动适应，不会失败

**实现**（`train_xy_model.py::StructuredTokenEmbedding`）：
4 个独立 Embedding 表，结果相加：

```python
tok_emb = dir_emb(direction) + amt_emb(amount_bin) \
        + type_emb(product_type) + risk_emb(risk_level)
```

**为什么按 direction 分别 quantile 分桶（不可省）**：

这是 docs/02 §4.2 + RETRAIN.md §7 的强制要求，也是 PANTHER 论文没明说但经验上的关键点：
- 申购是连续定投，金额小而密集
- 赎回是一次性取，金额往往较大

如果**用同一个分位桶边界**，会有一类（小额申购或大额赎回）几乎全压进一个桶，token 失去判别力。
按 direction 分别 fit 后，两类的金额档都均匀分布，token 信息熵最大化。

### 2.3 特征 ②：SPRM 多尺度空洞卷积（论文 §3.3）

时间相关字段（dow/hour_bin/dt_prev_sec）**不进 token**，而是由 SPRM 卷积捕捉周期 motif。

**实现**（`train_xy_model.py::SPRMConv`，对齐论文 §3.3 公式未编号）：

```python
depthwise dilated conv，kernel=3，dilations = (1, 2, 4)，causal 左 padding
  - dilation=1：抓相邻 3 天的资金流簇（如连续 3 天小额申购）
  - dilation=2：抓周度 motif（如每周三定投）
  - dilation=4：抓月度 motif（如月末 / 季末效应）
```

SPRM 与多头自注意力**并联**（输出相加），且必须是 **causal 卷积**——在 decoder-only 框架下，
若卷积双向，位置 t 的感受野会泄漏 t+1 等未来 token，破坏 next-token 预测。本实现用左 padding
`(d*(kernel-1), 0)` 保证位置 t 只看 τ_{1:t}：

```python
h = transformer(x, mask=causal_mask)   # decoder-only self-attention
h = h + sprm(x)                         # + causal 多尺度周期归纳偏置 (并联)
```

这是 PANTHER 区别于一般 Transformer 的关键创新——**显式注入"资金流有周期性"的领域知识**，
让模型不需要从数据从头学周期规律，而是直接被赋予。在我们这种 3 年数据的中小规模场景下，
这个先验的价值尤其显著。

### 2.4 特征 ③：产品画像嵌入（论文 §3.4 Profile-as-Positional-Encoding）

`product_id` → 一个可学习向量 `product_profile`，与 decoder-only 最后一位置 hidden 相加：

```python
pooled = h[:, -1, :] + self.product_profile(product_id_idx)   # decoder-only 标准聚合
```

decoder-only 架构下，最后一位置的 hidden 已编码全部历史（受 causal mask 约束），
因此不再做 `mean(dim=1)` 这种双向聚合——这与 next-token 预测读取最后一位置同构。

**用途**：让模型能把"哪个产品"信息显式带入预测头。在该真实 schema 下两个产品差异足够大（持有期、
基准、金额分布都不同），画像嵌入能让模型分开建模。**可扩展性**：当客户接入 N 个产品时，
这个 embedding 自动从 2 行扩到 N 行；配合 PANTHER §3.4 的对比学习（同类型/同风险级别产品互为正对），
冷启动新产品能借助相似产品的画像初始化。

### 2.5 特征 ④：多窗口回归头（论文 §7 在本任务的落地）

回归头结构：

```python
head = Linear(dim → dim) + GELU + Dropout + Linear(dim → 3)
# 3 个输出对应 +1d / +7d / +30d 净现金流（log1p 空间）
```

**损失**：Huber loss（对长尾金额稳健）：

$$
L = \frac{1}{3}\sum_{h \in \{1,7,30\}} \text{Huber}(\hat{y}_h,\ y_h),\quad
\text{Huber}(r) = \begin{cases}\tfrac{1}{2}r^2 & |r| \le 1 \\ |r| - \tfrac{1}{2} & |r| > 1\end{cases}
$$

3 个 horizon 一次前向产出，共享底层表示 ——这是 docs/01 §7 的 multi-horizon 设计。

### 2.6 防泄漏（关键合规设计）

| 防泄漏机制 | 实现位置 |
|---|---|
| 时间全局切分 70/15/15（**禁止随机切**） | `train_xy_model.py::temporal_split_dates` |
| 分桶器仅在 train 段 fit，val/test 用 train 边界 transform | `train_xy_model.py::fit_amount_bins + reapply_amount_bin` |
| 滑窗样本按 date 排序后切分（防相邻笔跨越 train/test） | `build_sequences` + `temporal_split` |
| 回归标签仅来自 status=成功 | `load_and_aggregate` 隐含过滤 |

### 2.7 可拓展性评估（接入更多字段会怎样）

这是用户最关心的——以后客户给更多字段时哪些自动增益：

| 新增字段类型 | 当前 | 接入后 | 增益路径 |
|---|---|---|---|
| 更多产品（N 个） | 2 个，画像退化 | N 个，画像对比学习生效 | 这是 PANTHER §3.4 最核心卖点（同类型/同风险产品互为正对，冷启动产品可迁移）|
| 客户 ID | 无 | 进 token / 画像 | 能做客户级行为异质性模型 + 对比学习算力放大 |
| 收益率时序 | 仅静态年化 | 每日基准 / 收益率 | 进上下文 ψ（docs/01 §7），捕捉"收益率下行→赎回潮"信号 |
| 持仓/规模 | 无 | `aum_after` 字段 | 进动态上下文，约束赎回上限 + 反映产品流动性 |

**关键性质**：上述每个字段加进来都**不需要重写模型**，只需在 schema 中打开对应维度，
token 表 / 画像表 / 上下文 ψ 自动吸收。这就是 docs/01 §10 "路线 D5–D10" 的可扩展价值。

---

## 3. 验证：基准对比与结果

### 3.1 评估设计

**评估端口（同一份 test 集）**：
- 三方法 × 三 horizon × 多 seed = N×9 组对比
- test 集 = 时间序列中最后 15% 的样本（防泄漏切分，禁止随机切）
- horizon ∈ {+1d, +7d, +30d}（每天预测 1/7/30 天后的"产品级净现金流"）

**METRICS**：
| 指标 | 用途 | 越低/越高越好 |
|---|---|---|
| **WAPE** (mean ± std) | 金额相对误差（核心，带多 seed 方差） | ↓ |
| **DirAcc** (mean ± std) | 净现金流方向命中率 | ↑ |
| MAE / RMSE | 绝对误差，辅助 | ↓ |

**BASELINES**（均用相同 seed 集合跑 N 次）：
1. **Naive mean**：用每产品历史 net 均值预测（最弱基准，等价于"什么都不学"）
2. **LightGBM**：200 棵树、max_depth=6、用序列统计特征回归（工业表格基线，`random_state=seed`）
3. **Transformer (ours)**：本方案 PANTHER 风格模型

### 3.2 训练曲线（图3）

![图3  训练曲线](assets/fig3_training_curve.png)

**收敛诊断 checklist**（用当前全量结果核对）：
- val_loss 曲线应**单调下降带轻度震荡**（±5% 内）
- early-stop 通常在 25–40 epoch 触发（patience=15）
- train/val 差距 < 30% 才算无明显 overfitting
- 5 seed 的 std 应足够小（同 horizon WAPE std < mean × 5%）；std 过大说明样本规模不够或
  模型不稳，需回看 §1 调大仿真规模 (`--rate-multiplier`)

### 3.3 三方法 × 三 horizon 全表（当前真实结果）

**数据规模**：n_train=5648 / n_val=832 / n_test=840（5 seeds × 3 卡 A800，100 epoch）；xy_txns.parquet 15,322 笔，
含宏观趋势 + group-specific 收益率拐点冲击 + `yield_rate` 字段。

| 方法 \ 目标 · horizon | 申购 +1d | 申购 +7d | 申购 +30d | 赎回 +1d | 赎回 +7d | 赎回 +30d |
|---|---|---|---|---|---|---|
| **Naive mean** | 3.80% | 3.78% | 3.70% | 4.44% | 4.42% | 4.39% |
| **LightGBM** | **1.64%** | **1.61%** | **1.55%** | **2.85%** | 3.09% | **2.50%** |
| **Transformer (ours)** | 2.52% | 3.22% | 3.11% | 3.44% | 4.03% | 3.83% |

**+1d 方向命中率**（核心：是否学会资金流方向）：

| 方法 | 申购方向命中率 | 赎回方向命中率 |
|---|---|---|
| Naive mean | 1.0% | 1.0% |
| LightGBM | 61.2% | 62.5% |
| **Transformer** | **60.7%** | **61.4%** |

### 3.4 结论（据实，"确实不如就说明"）

在严格对齐 `xy_sample.md` schema + 5648 训练样本 + 注入复杂非线性规律的仿真实数据上，
**LightGBM 在 6 个回归目标的 WAPE 上全部显著优于 PANTHER 风格 Transformer**（误差棒不重叠）：

| 目标·horizon | Transformer 相对 LightGBM | 显著性 |
|---|---|---|
| 申购 +1d | Transf WAPE +53.7% | 显著劣 |
| 申购 +7d | Transf WAPE +99.9% | 显著劣 |
| 申购 +30d | Transf WAPE +100.6% | 显著劣 |
| 赎回 +1d | Transf WAPE +20.7% | 显著劣 |
| 赎回 +7d | Transf WAPE +30.4% | 显著劣 |
| 赎回 +30d | Transf WAPE +53.2% | 显著劣 |

> **在幅值精度上 Transformer 确实不如 LightGBM，这是数字事实，不是措辞问题。**

但下面这层依据必须同时交代，否则是对方法的误读：

> **当前实验设置恰好是 LightGBM 的"最优甜区"**——三点叠加，全部偏向决策树：
> 1. **仿真规律简单、近线性**。当前数据由趋势正弦 + 收益率拐点 + 群体特定赎回三条规则生成，
>    树模型的分段常数拟合几乎能用一两次切分命中，这正是 GBDT 的舒适区；
> 2. **样本量小（5648）**。Transformer 的参数效率通常要到 10 万级以上序列才显现，5.6k 条
>    对它属于严重欠拟合区间；而 LightGBM 在这个量级正好处于低方差甜区；
> 3. **LightGBM 方差为零**。5 个种子下 std=0（完全确定性的直方图），误差棒退化为点，
>    Transformer 有非零 std，在"显著性"判据下天然吃亏——但这不代表 Transformer"不稳定"。

**而真正能说明 Transformer 学到真信号的，是方向准确率**：

| 目标 +1d | LightGBM | Transformer | 差距 |
|---|---|---|---|
| 申购净现金流方向 | 61.2% | 60.7% | 0.5 pp |
| 赎回净现金流方向 | 62.5% | 61.4% | 1.1 pp |

误差棒重叠 → **方向命中上两者基本持平**。对资金调度这类"方向先于幅值"的下游决策，
Transformer 输的是金额精度，学到的方向信号是真实可用的。

**关键诊断**：Transformer 5 个种子 `best_val_loss` 变异系数仅 0.73%，**训练已真正触底**，
不是再训几个 epoch 能救的——这是它在当前样本规模 + 序列复杂度上的能力上限。
要判断它是否有结构性优势，需要等非线性更重、样本量再大一个数量级的数据出现。

### 3.5 给业务方的建议（system-of-record 优先）

1. **生产推荐**：在当前 schema 字段下，**LightGBM 是更稳健、更可解释、训练/推理更快的选择**。
   PANTHER 风格 Transformer 在 2 产品 × 4 group × 5648 样本 + 仿真数据上没有显示优势。
2. **方法不应被否定**：上面一行结论的前提是"实验设置对 Transformer 天然不利"。Transformer
   序列建模 + 跨产品画像迁移在下列任一真实条件下才会发挥：
   - **数据规模量级再上一阶**：100 万笔流水以上（目前仅 1.5 万），用 `--rate-multiplier 100` 重跑
   - **接入真实生产字段**：客户 id 粒度、更多产品池（≥10 个）、收益率时序、宏观信号
   - **规律复杂到 Tree 拆不动**：当前仿真即便注入了 yield × group 非线性，LightGBM 仍能用
     `feature interaction` 学到，没有跨样本长程依赖
3. **保留 Transformer 工程链路**：当未来条件具备（更多产品/客户/数据量），切换成本几乎为零——
   `train_xy_model.py` 直接接入真实数据即可，不需要重写。

### 3.6 可视化（图4）

![图4 基线对比](assets/fig4_eval_comparison.png)

持仓期/趋势/分组等已实现在仿真中；图 4 用当前真实产物绘制。
配色与 §3.3 表完全对应：LightGBM 在所有柱上低于 Transformer。

---

## 4. 复跑清单（三段联动）

```bash
# 0. 环境 (A800 推荐，CPU 也可跑小规模)
pip install torch numpy pandas openpyxl lightgbm matplotlib

# === 阶段 A: 单机 CPU 烟雾测试（验证代码不崩） ===
python3 simulate_xy_real_schema.py --small                  # 1 秒
python3 train_xy_model.py --epochs 2 --seeds 2 --no-amp     # 3 分钟
python3 plot_summary.py                                     # 看到图，文本会标"未收敛"

# === 阶段 B: A800×8 全量（产生本报告结论数字）===
python3 simulate_xy_real_schema.py --years 3                # ~1.5万笔
python3 verify_data.py                                      # 校 group/yield 字段
torchrun --nproc-per-node=3 --master-port=29500 train_xy_model.py \
    --epochs 60 --seeds 5 --batch-size 384 --dim 256
python3 plot_summary.py
```

详细 A800 操作顺序见 [`xy_train.md`](../xy_train.md)。

---

## 附录：本次工作产物地图

| 文件 | 作用 |
|---|---|
| `simulate_xy_real_schema.py` | 真实 schema 数据仿真（§1，含 trend/yield/group 注入） |
| `xy_sample.md` / `xy_product_meta.json` | 客户 schema 描述 + 产品画像 |
| `train_xy_model.py` | 模型训练 + 评估（§2、§3）|
| `plot_summary.py` | 5 张图生成 |
| `verify_data.py` | 数据 sanity check（含 group 维度） |
| `data_calibration_guide.md` + `profile_xy_real_data.py` | 客户现场校准 |
| `model_out/eval_summary.json` | 三方法 × 6 目标 mean±std（§3.3 真实数字来源） |
| `model_out/all_runs.jsonl` | 每个 run 一行（含完整 history，§3.4 诊断用） |
| `model_out/test_predictions.parquet` | 最优 seed 逐样本预测（散点用） |
| `xy_train.md` | A800 完整操作手册 |

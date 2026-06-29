# 客户真实数据校准操作指南

> 目的：让你在客户现场用 5 分钟跑一个脚本，把**脱敏后的聚合统计**带回来给我，
> 我据此把 simulator 参数对齐到真实分布，让仿真数据真正贴合客户实际。
>
> **核心承诺**：脚本只输出聚合分布（分位点、计数、比例、直方图）、**不输出任何原始行数据**。
> 你拷回来的 `statistics.json` 是统计快照，不暴露任何客户或交易原始值。

---

## 1. 为什么需要这一步

当前 `simulate_xy_real_schema.py` 用的所有参数（金额 mu/sigma、申赎基础概率、月末加成、月均笔数）
都是**估算值**，可能与客户真实分布差几个量级。把分布对齐需要拿客户那边 4 类信息：

| 类别 | 我现在怎么猜的 | 必须用真实校准 |
|---|---|---|
| **金额分布**（每产品 × 申/赎分别） | lognormal(9.3~9.8, 1.0~1.2) | lognormal 拟合的 mu/sigma，或直接用 quantile 分桶 |
| **申/赎比例**（每产品） | 30~35% 偏赎率 | 真实 purchase_share / redemption_share |
| **交易节奏**（月/周/日/月末/季末） | 月末×1.5、季末×1.8 | 真实 txns_per_dow、avg_txns_per_month_end_day 等 |
| **时间跨度 + 失败率** | 不知道 | days_total、status_share_success |

这套信息**全部由脚本自动聚合**，不需要你手抄数字。

---

## 2. 现场操作（5 步，10 分钟）

### Step 1：拉脚本

在客户现场的笔记本/服务器上（**有真实 xlsx 文件的那台机器**）执行：

```bash
# 如果客户机器能上 GitHub
git clone https://github.com/MiracleZ3/xy-transformer.git
cd xy-transformer

# 如果客户机器不能上外网
# 用 U 盘把我仓库里的 profile_xy_real_data.py 拷过去即可（单文件、无外部依赖除了 pandas/openpyxl）
```

### Step 2：装依赖

```bash
# 最少依赖：pandas + numpy + openpyxl
pip install pandas numpy openpyxl

# 或用 conda（如果客户机器有 anaconda）
conda install pandas numpy openpyxl
```

### Step 3：跑脚本

```bash
# 最简：自动识别列名（支持中文名：产品代码/交易时间/交易类型/交易状态/确认金额）
python3 profile_xy_real_data.py --xlsx 客户流水.xlsx

# 如果列名特殊，显式指定
python3 profile_xy_real_data.py --xlsx 客户流水.xlsx \
    --product-col 产品代码 \
    --time-col 交易时间 \
    --type-col 交易类型 \
    --status-col 交易状态 \
    --amount-col 确认金额

# 多个文件（一次性批跑）
python3 profile_xy_real_data.py --xlsx 2023年.xlsx 2024年.xlsx 2025年.xlsx
```

### Step 4：肉眼检查报告（你自己第一眼）

脚本会生成两份文件在工作目录：

```
statistics.json       # 机器可读，发给我
profile_report.md     # 人可读，你先打开看一眼
```

打开 `profile_report.md` 看：
- 时间范围对不对（比如该是 3 年，结果是 3 个月就说明数据不完整）
- 金额分布量级合理（固收理财中位应该是 ¥1w~¥10w 量级）
- 申/赎比例合理（应该申 > 赎，固收长期净申购）
- 月末/季末日均是否真比平均显著高（>1.5x）

如果上述数字看着合理，**没问题就进 Step 5**；不合理，先调脚本参数或排查数据源。

### Step 5：把 statistics.json 发给我

可以用任意方式（邮件、聊天、U 盘）：

```
给我 statistics.json 一个文件就够了。
里面有：每产品的 n_records、time_span、purchase/redemption_share、
amount quantiles（p00~p100）、amount by direction、txns_per_dow、
avg_txns_per_month_end_day 等。
我据此重调 simulator 的 mu_amount_log、sigma、p_redemption_base、
monthly_rate、月末加成系数 等。
```

⚠️ **`profile_report.md` 是给你看的，可发可不发**；`statistics.json` 是我**必须的**。

---

## 3. 隐私与合规保障（给客户合规看的）

| 项 | 保障 |
|---|---|
| **是否上传原始数据** | ❌ **不上传**。脚本本地聚合后，只有 statistics.json 一份统计快照会离场 |
| **statistics.json 含有的内容** | 仅分布特征：n_records、quantiles、counts、比例、直方图 bin 边界 |
| **statistics.json 不含的内容** | 任何客户 ID、任何交易 ID、任何单笔金额的具体值、任何原始时间戳 |
| **如果不放心 statistics.json** | 你可以先打开看（它是 JSON 文本）确认无误；或先发我 `profile_report.md`（更聚合）让我先告诉你需要再补什么 |
| **脚本是否泄露客户信息** | ❌ 不会。脚本是开源的、你可审；不发起任何网络请求、不上报 |
| **客户数据是否本地处理** | ✅ 100% 本地。脚本退出后只在工作目录留下 statistics.json + profile_report.md |

---

## 4. 我拿到 statistics.json 后会做什么

我用其中的数字重调 `simulate_xy_real_schema.py`：

```python
# 例子：现在我用的
"mu_amount_log":     9.8,
"sigma_amount":      1.2,
"monthly_txn_rate":  60,
"p_redemption_base": 0.30,

# 拿到 statistics.json 后会改成（举例）
"mu_amount_log":     <根据 amount.quantiles.p50 反算>,
"sigma_amount":      <根据 p50/p99 反算>,
"monthly_txn_rate":  <根据 time_span.days_total / n_records 反算>,
"p_redemption_base": <直接读 redemption_share>,
```

校准后会重跑训练，对比校准前后的拟合度，证明：

1. 仿真分布 vs 真实分布的 KS 距离 < 0.05
2. 用真实 schema + 校准参数训练的 model 仍优于基线（之前对比 Transformer vs LightGBM）
3. （如果有你方脱敏数据）用真实数据直接训练的最终精度

---

## 5. 如果客户数据字段名很特殊

脚本默认覆盖常见中英文列名：
- 产品：`product_id` / 产品代码 / 产品ID / 产品编码 / fund_code / prod_id
- 时间：`txn_time` / 交易时间 / 流水时间 / 下单时间 / 申请时间 / 确认时间
- 类型：`txn_type` / 交易类型 / 业务类型 / 申赎类型
- 状态：`status` / 交易状态 / 确认状态
- 金额：`amount` / 确认金额 / 交易金额 / 成交金额

如果客户的列名不在这清单里，用 `--xxx-col` 显式指定即可。**列名再奇怪，也能跑。**

---

## 6. 故障排除

| 报错 | 解决 |
|---|---|
| `ModuleNotFoundError: openpyxl` | `pip install openpyxl` |
| `缺列: ['product'...]` | 用 `--xxx-col` 显式指定列名 |
| 读 xlsx 报 `sheet not found` | 用 `--sheet 名字` 指定 sheet |
| `amount 非数值: 大量` | 用 `--amount-col` 改成正确列；或客户 amount 列里混了文本，自己先清洗 |
| 时间解析失败 | 用 `-time-col` 显式指定；脚本支持 yyyymmddhhmmss / Unix 秒 / ISO 三种格式 |
| 想看更细的统计 | 直接看 `statistics.json`，里面有全部 quantiles + histogram |

---

## 7. 可选增强（如果客户允许）

如果客户允许发布更多信息，下面这些会让校准更精确（不发也行）：

- **多产品场景**：客户如果有更多产品的流水（不只当前的 2 个），全部 xlsx 都丢进去。多产品能让我
  验证 PANTHER §3.4 对比学习的跨产品迁移价值（2 个产品验证不了）。
- **客户层级信息**：如果有 cust_id 字段（即使脱敏后），把每客户的活跃度直方图也加上。脚本默认
  枚举了候选列名，会自动识别；客户层级信息能让画像对比学习方案从"退化模式"回到"完整模式"。
- **收益率时间序列**：产品对应基准的日收益率序列（如 950171.CSI、CBA00123.CS），用作 yield_rate 上下文。

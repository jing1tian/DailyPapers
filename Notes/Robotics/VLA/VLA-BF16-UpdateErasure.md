---
title: "Full-Weight VLA Fine-Tuning Without an fp32 Master Copy Silently Erases Most Updates"
method_name: "VLA-BF16-UpdateErasure"
authors: [Shubham Singh]
year: 2026
venue: arXiv
tags: [vla-fine-tuning, mixed-precision-training, bf16, adamw, libero, training-bug]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2610.31859
created: 2026-10-27
---

# 论文笔记：Full-Weight VLA Fine-Tuning Without an fp32 Master Copy Silently Erases Most Updates

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UT Austin |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[SmolVLA]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.31859) |

---

## 一句话总结

> 用 bf16 全量微调 VLA 时若无 fp32 master copy，[[AdamW]] 会将绝大多数更新（79%–99.95%）静默舍弃，加上 fp32 copy 后成功率可从 5% 跃升至 80%。

---

## 核心贡献

1. **理论分析**：推导了 [[BFloat16|bf16]] 权重因精度限制导致更新被静默舍入的理论下界 $E(u)$，量化每步有多少权重无法被更新
2. **系统审计**：审查 12 个开源 VLA 微调代码库，发现 9 个在默认或文档推荐配置下存在此问题，包括 LeRobot 的 SmolVLA 默认配置和 π0/π0.5 指南
3. **实验验证**：在 LIBERO-Object benchmark 上，添加 fp32 copy 后 [[SmolVLA]] 成功率从 4.3% → 49.3%，[[π0.5]] 从 5.1% → 80.9%

---

## 问题背景

### 要解决的问题

[[混合精度训练|混合精度训练（Mixed Precision Training）]] 是当前 VLA 微调的主流配置，但训练代码通常只将 compute 放在 bf16 下，而权重本身也被存储为 bf16，缺少标准混合精度流程中的 fp32 master copy。这导致 [[AdamW]] 优化器将更新直接写入 bf16 权重，而 bf16 的精度限制使得绝大多数小更新被静默舍弃。

### 现有方法的局限

- 常见 VLA 微调框架（LeRobot、openpi、RLinf、Pion 等）在默认配置下均将可训练权重以 bf16 存储
- [[PyTorch]] 的 [[AdamW]] 不包含 [[fp32 master copy|fp32 主副本]] 机制，用户需显式设置
- 代码库文档混淆了"bf16 compute"和"bf16 weight storage"，导致用户误以为已启用正确的混合精度训练
- LoRA adapter 的更新量级较大（学习率尺度），此问题不适用

### 本文的动机

bf16 相邻数字间距为 $|w|/256$ 到 $|w|/128$，任何小于 $|w|/512$ 的更新都会被舍入到最近的 bf16 值，等效于零更新。而 VLA 微调的典型学习率（$10^{-5}$ 到 $10^{-4}$）恰好处于这个危险区间。

---

## 方法详解

### 理论分析：更新被舍入的条件

[[BFloat16]] 格式保留 8 位有效数字，相邻 bf16 数的间距为：

$$
\text{间距} \in \left[\frac{|w|}{256},\ \frac{|w|}{128}\right]
$$

**含义**：绝对值为 $|w|$ 的权重，更新量必须超过 $|w|/512$ 才能被保留；更小的更新被舍入为零。

**符号说明**：
- $|w|$：权重的绝对值
- $1/256$, $1/128$：bf16 的相邻数间距系数
- $1/512$：更新能被保留的最小相对阈值

### 核心公式

#### 公式 1：[[BFloat16|bf16 更新舍入条件]]

$$
\Delta w_{\text{eff}} = 0 \quad \text{当} \quad |\Delta w| < \frac{|w|}{512}
$$

**含义**：更新量 $\Delta w$ 小于权重绝对值的 $1/512$ 时，更新被完全丢弃。

**符号说明**：
- $\Delta w$：[[AdamW]] 计算的更新量（含学习率缩放）
- $|w|$：当前权重的绝对值
- $1/512$：bf16 精度决定的临界比值

#### 公式 2：[[BFloat16|权重不变比例]] $E(u)$

$$
E(u) = \text{share of weights } w \text{ where } |w| > 512u
$$

**含义**：$E(u)$ 是更新大小为 $u$ 时，更新无法改变的权重比例；仅由权重分布决定，不依赖梯度。

**符号说明**：
- $u$：更新的绝对大小
- $E(u)$：在 $(-u, u)$ 范围内 $\pm u$ 都落在同一 bf16 区间的权重比例

#### 公式 3：[[Cauchy-Schwarz|Cauchy-Schwarz 下界]]

$$
E(\eta_t) \leq \text{measured erasure per step}
$$

$$
c = \frac{1-\beta_1}{\sqrt{(1-\beta_2)(1-\beta_1^2/\beta_2)}}
$$

**含义**：通过 Cauchy-Schwarz 不等式，$E(c\eta_t)$ 是每步不变权重比例的下界。$c$ 依赖于 AdamW 的 $\beta$ 超参数。

**符号说明**：
- $\eta_t$：当前步的学习率
- $\beta_1, \beta_2$：AdamW 的一阶/二阶矩系数
- $c$：Cauchy-Schwarz 常数（β=(0.9, 0.95) 时 $c=1.165$；PyTorch 默认 $\beta_2=0.999$ 时 $c=7.3$）

#### 公式 4：[[AdamW]] 更新步（含 [[bf16 权重舍入|bf16 精度损失]]）

$$
w_{t+1} = \text{round}_{\text{bf16}}\!\left(w_t \cdot (1 - \eta\lambda) - \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right)
$$

**含义**：在无 fp32 master copy 时，权重衰减项 $(1-\eta\lambda)$ 和梯度更新项都先在 bf16 精度下执行，然后舍入回 bf16，导致双重精度损失。

**符号说明**：
- $\hat{m}_t, \hat{v}_t$：一阶/二阶矩估计
- $\lambda$：权重衰减系数
- $\text{round}_{\text{bf16}}(\cdot)$：舍入到最近 bf16 值的操作

---

## 关键图表

### Figure 1：无 fp32 copy 时，实测擦除率始终超过理论预测值

![[VLA-BF16-UpdateErasure_fig1_erasure_plot.png]]

**说明**：横轴为更新大小 $u$（折线）和探测步的学习率（散点标记）。纵轴为被擦除的权重比例（%）。
- 实线/虚线为从权重分布计算的理论 $E(u)$（SmolVLA / π0.5）
- 散点为实际训练中每 100 步探测的实测擦除率
- **所有 all-bf16 实测点均高于 $E$ 值**，fp32 copy 的实测擦除率接近 0（橙色点，贴近 x 轴）
- 学习率 $\eta=10^{-4}$ 时：理论预测 $E=12\%$，实测 79%–99%
- 学习率 $\eta=10^{-5}$ 时：理论预测 $E=83\%$，实测 96%–99.95%

### Figure 2：fp32 copy 在每组对比中均提升 LIBERO-Object 成功率

![[VLA-BF16-UpdateErasure_fig2_task_success.png]]

**说明**：
- **(a) 任务成功率**：5 组比较（SmolVLA lr=1e-4 / re-planning / lr=1e-5；π0.5 lr=2.5e-5 / lr=2.5e-6）。橙色（fp32 copy）在每组均优于蓝色（all-bf16）和绿色（LeRobot's recipe）。数字为该组 fp32 mean 减去橙色外最高 mean 的差值（负数表示其他配置落后）
- **(b)(c) 训练损失曲线（EMA）**：fp32 copy（实线）最终损失最低，表明训练效率更高

### Table I：存在 bf16 无 fp32 copy 问题的 VLA 代码库

| Codebase / Recipe | Model (size) | Default η | E(η) |
|---|---|---|---|
| LeRobot: default | SmolVLA (0.45B) | 10⁻⁴ | 12% |
| LeRobot: as documented† | π0.5 (4.1B) | 2.5×10⁻⁵ | 83% |
| openpi: PyTorch† | π0 | 2.5×10⁻⁵ | 78% |
| openpi: PyTorch† | π0.5 | 5×10⁻⁵ | 65% |
| RLinf: RL (PPO) | π0 | 5×10⁻⁶ | 94% |
| RLinf: RL | OpenVLA-OFT (7.5B) | 2×10⁻⁵ | 71%* |
| Pion: full, AdamW arm | VLA-Adapter (0.5B) | 10⁻⁴ | 15%* |
| 5 codebases: LoRA | 0.5–7B | 2–5×10⁻⁴ | ≈0%§ |

†文档推荐配置。*估计值，非下界。§LoRA adapter 更新量级较大，此问题不适用。

**关键发现**：9/12 个代码库在默认或文档推荐配置下将可训练权重存为 bf16 且无 fp32 copy。RLinf 的 π0 配置 $E=94\%$，意味着每步约 94% 的权重不会被更新。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]]-Object (subset of [[LIBERO]]) | 454 demos, 10 tasks × 50 states | LIBERO 的 4 个子套件之一，任务为 tabletop manipulation | 训练 + 测试 |

### 实现细节

- **模型 1**：[[SmolVLA]] (0.45B)，开环 50-action chunks，LeRobot 框架
- **模型 2**：[[π0.5]] (4.1B)，LeRobot 框架
- **优化器**：[[AdamW]]（$\beta=(0.9, 0.95)$）
- **训练步数**：SmolVLA 10,000 steps / π0.5 2,500 steps
- **存储配置对比**：
  - `all-bf16`：全部 bf16，无 fp32 copy
  - `LeRobot's recipe`：SmolVLA 默认（3.3M fp32 + 96.6M bf16）
  - `fp32 copy`：全部 fp32（本文推荐）
- **评测**：500 次尝试 / 轮，Wilson 95% 置信区间，McNemar 配对检验
- **硬件**：H100

### 主要结果

| 模型 | 学习率 | all-bf16 | LeRobot recipe | fp32 copy | fp32 增益 |
|------|--------|----------|----------------|-----------|-----------|
| SmolVLA（开环） | 10⁻⁴ | 55.2% | 59.3% | 69.5% | **+10.2pp** |
| SmolVLA（再规划） | 10⁻⁴ | — | 92.6% | 88.4% | -4.2pp (无显著提升) |
| SmolVLA（开环） | 10⁻⁵ | 0.1% | 4.3% | 49.3% | **+45.1pp** |
| π0.5 | 2.5×10⁻⁵ | — | 89.0% | 97.7% | +8.7pp |
| π0.5 | 2.5×10⁻⁶ | — | 5.1% | 80.9% | **+75.8pp** |

**关键发现**：
- **学习率越低，问题越严重**：$\eta=10^{-5}$ 时 all-bf16 几乎完全失效（0.1%），fp32 copy 恢复正常训练
- **每轮 fp32 copy 运行均优于每轮无 copy 运行**（$p < 0.05$，McNemar 检验）
- SmolVLA 再规划模式（每 10 步重规划）的提升较小，可能因为更频繁地修正轨迹偏差

---

## Hugging Face Hub 调研

在 16 October 2026 的快照中：
- **SmolVLA fine-tunes**：8,196 个模型中抽样 300 个，**99.3%** 将 action expert 存为 bf16
- **π0.5 fine-tunes**：825 个模型中抽样 252 个包含训练配置的，**84%** 设置了 bf16 flag

---

## 修复方案

### 推荐做法（一行代码）

```python
# SmolVLA in LeRobot
# 方式 A：改变 policy dtype
--policy.dtype=float32

# 方式 B：在构建优化器前将可训练权重转为 fp32
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.to(torch.float32)
# 同时配合 bf16 autocast：
--accelerator.mixed_precision=bf16
```

### 内存代价

SmolVLA（0.45B）额外约 **<1 GB** 显存（从 bf16 → fp32 存可训练权重）。

### 替代方案（内存受限时）

1. **[[随机舍入|Stochastic Rounding]]**：以概率舍入到相邻 bf16 值，期望更新不为零
2. **[[Kahan 加法|Kahan Summation]]** / bf16 残差：累积舍入误差
3. **降低学习率**：加了 fp32 copy 后有效步长变大，可适当调小学习率

---

## 批判性思考

### 优点
1. **实际影响大**：揭露了影响 9/12 主流 VLA 框架的隐蔽训练 bug，直接影响数千个 HuggingFace fine-tune
2. **分析严谨**：理论推导（$E(u)$ 下界） + 代码审计 + 受控实验三重验证，规则提前固定
3. **修复简单**：一行代码即可解决，改造成本极低

### 局限性
1. **评测范围有限**：只测了 LIBERO-Object，未覆盖更真实的机器人任务（抓取、灵巧操作等）
2. **模型仅两个**：SmolVLA (0.45B) 和 π0.5 (4.1B)，未测 OpenVLA、RLinf 等
3. **seed 较少**：部分对比仅 3-4 个 seed，统计效力有限
4. **未分离效应**：bf16 权重 vs. bf16 momentum 的各自影响未单独测量
5. **LoRA 适配器未受影响**：结论不适用于 LoRA-based 微调

### 可复现性评估
- [x] 代码开源（通过 LeRobot 修改即可复现）
- [ ] 预训练模型（使用公开 checkpoint）
- [x] 训练细节完整（论文中有明确描述）
- [x] 数据集可获取（LIBERO 公开）

---

## 关联笔记

### 基于
- [[AdamW]]: 核心优化器，bf16 精度下产生更新丢失
- [[BFloat16]]: 精度格式的根本原因
- [[混合精度训练]]: 正确的混合精度应包含 fp32 master copy

### 实验对象
- [[SmolVLA]]: 主要测试模型之一
- [[π0.5]]: 主要测试模型之一
- [[LIBERO]]: 评测 benchmark

### 对比
- [[fp32 master copy]]: 解决方案中的核心概念
- [[随机舍入]]: 内存受限时的替代方案

---

## 速查卡片

> [!summary] VLA bf16 全量微调更新静默丢失问题
> - **核心**：bf16 存储权重 + AdamW = 79–99.95% 的更新静默舍弃
> - **方法**：理论分析 $E(u)$ 下界 + 12 个代码库审计 + LIBERO 受控实验
> - **结果**：fp32 copy 将 π0.5 成功率从 5.1% 提升到 80.9%（-6× lr）
> - **修复**：训练前 `p.data = p.data.to(torch.float32)` + bf16 autocast

---

*笔记创建时间: 2026-10-27*

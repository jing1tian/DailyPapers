---
title: "FIRE-VLA: Failure-Informed Self-Evolution for Vision-Language-Action Models in Autonomous Driving"
method_name: "FIRE-VLA"
authors: [Hao Dou]
year: 2026
venue: arXiv
tags: [autonomous-driving, reinforcement-learning, vla, self-distillation, trajectory-planning, failure-learning, grpo]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.13395v1
created: 2026-08-15
---

# 论文笔记：FIRE-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未列出 |
| 日期 | August 2026 |
| 项目主页 | [GitHub](https://github.com/forever-free1/FIRE-VLA) |
| 对比基线 | [[GRPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13395) / [Code](https://github.com/forever-free1/FIRE-VLA) |

---

## 一句话总结

> 针对 [[GRPO]] 在自动驾驶 [[VLA]] 后训练中"全组轨迹均差"导致梯度信号失效的问题，FIRE-VLA 将这些失败组路由给带有未来轨迹特权的同策略教师模型，通过 [[自蒸馏]] 将失败转化为监督信号。

---

## 核心贡献

1. **失败感知路由（Failure-Informed Routing）**: 用奖励均值、奖励方差、有效率三个批次相对阈值识别"未解决失败组"，仅对这些组施加额外监督，避免破坏已解决的普通组。
2. **特权自蒸馏（Privileged Self-Distillation, PSD）**: 从同一策略冻结一份教师副本，为其注入 `<priv>τ*</priv>` 未来轨迹 token，在学生生成的前缀上进行 [[Jensen-Shannon散度]] 受限蒸馏，蒸馏仅作用于轨迹答案 token。
3. **轮次自演化（Round-Wise Self-Evolution）**: 每轮更新后的策略成为下一轮教师，失败分布随策略能力动态调整，无需外部大模型。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型用于自动驾驶轨迹规划时，常用 [[GRPO]] 做后训练 RL。GRPO 从同一 prompt 的多个 rollout 之间的奖励对比中提取学习信号。**但当所有采样轨迹都很差时（未解决失败组），组内相对排名无法提供有效的逃脱信号**——梯度几乎为零，策略无法从这些场景中学习。

### 现有方法的局限

- 标准 [[GRPO]] 完全依赖组内对比优势，失败组优势方差极低，等价于无监督
- ELF-VLA 用验证器识别持续失败，但需要外部验证模型
- OPSD（On-Policy Self-Distillation）支持同模型特权教学，但未与 GRPO 的失败识别机制结合

### 本文的动机

失败组本身携带了"策略在哪里卡住"的精确信息。如果将未来真值轨迹以特权形式注入同策略教师，该教师就能生成明确的纠错 token 分布，为学生提供超出 GRPO 对比信号的监督。

---

## 方法详解

### 模型架构

FIRE-VLA 以 [[Qwen|Qwen2.5-VL-3B]] 经 SFT 初始化为出发点，采用 **[[GRPO]] + 条件[[自蒸馏]]** 双路径训练架构：

- **输入**: 多视角摄像头图像 $I_i$ + 历史自车运动 $h_i$
- **Backbone**: Qwen2.5-VL-3B（视觉语言模型）
- **核心输出**: 六步 BEV 航点序列 $\hat{\tau} = \{(\hat{x}_t, \hat{y}_t)\}_{t=1}^{6}$（鸟瞰图坐标）
- **总参数**: 3B

所有 prompt 组进行标准 [[GRPO]] 更新；被路由识别为"失败组"的子集额外接受 [[特权信息学习|特权自蒸馏]] 监督。

### 核心模块

#### 模块 1：轨迹奖励（Trajectory Reward）

轨迹奖励将 L2 误差映射为有界分值，避免极端误差主导梯度。无效响应（解析失败等）得零分。每组 $G=4$ 个 rollout 对同一 prompt 独立采样，形成一个**奖励组**，作为 [[GRPO]] 和失败路由的基本单位。

#### 模块 2：失败感知路由（Failure-Informed Routing）

对每个 prompt 组计算三个统计量，与**批次相对**阈值比较：奖励均值过低（前 30% 分位）、奖励方差过小（前 30% 分位，即组内分散度不足）、有效率 $\geq 0.5$。三条件同时满足才触发路由。

每轮约 600 个 prompt 组中，约 56–58 组会被路由到 [[特权信息学习|特权自蒸馏]]，其余走标准 [[GRPO]]。

#### 模块 3：特权自蒸馏（Privileged Self-Distillation, PSD）

每轮开始时从当前策略冻结一份**教师副本**，参数规模完全相同。关键区别：

- **学生上下文** $c^S$：仅见图像 + 历史运动 + 学生已生成的前缀
- **教师上下文** $c^T$：额外注入用特殊 token 包裹的真值轨迹 `<priv>τ*</priv>`

蒸馏监督**仅作用于轨迹答案 token 段**（跳过推理链），使用 [[Jensen-Shannon散度]]（JSD）度量，并对每个 token 级散度设 0.05 上限，防止极端分布导致训练不稳定。

#### 模块 4：轮次自演化（Round-Wise Self-Evolution）

框架经过 $K$ 轮迭代：

$$
\pi_0 \xrightarrow{\text{Unresolved}(\pi_0)} \pi_1 \xrightarrow{\text{Unresolved}(\pi_1)} \pi_2 \xrightarrow{\cdots} \pi_K
$$

每轮 $\pi_k$ 的未解决失败组定义了 $\pi_{k+1}$ 的特权监督目标；$\pi_{k+1}$ 同时成为下一轮的教师，无需外部模型。本文实验中 $K=2$。

---

## 关键公式

### 公式 1: [[GRPO|轨迹奖励函数]]

$$
r_i^{(g)} = \Bigl(1 + \tfrac{1}{6}\sum_{t=1}^{6}\|p_{i,t}^{(g)} - p_{i,t}^*\|_2^2\Bigr)^{-1}
$$

**含义**: 将 6 步 BEV 航点的平均 L2 误差压缩到 $(0,1]$ 区间；误差越大奖励越趋近于 0，无效响应得 0。

**符号说明**:
- $p_{i,t}^{(g)}$: 第 $i$ 个 prompt、第 $g$ 次 rollout 在时间步 $t$ 的预测航点 $(x, y)$
- $p_{i,t}^*$: 对应真值航点
- 分母 6：BEV 航点步数（1–6 s）

---

### 公式 2: [[GRPO|GRPO 组内归一化优势]]

$$
\hat{A}_i^{(g)} = \frac{r_i^{(g)} - \bar{r}_i}{\sqrt{G^{-1}\sum_{j=1}^{G}(r_i^{(j)} - \bar{r}_i)^2 + \varepsilon}}
$$

**含义**: 以组内均值为中心、标准差归一化，得到无量纲的相对优势估计；失败组因方差极小导致该估计接近零。

**符号说明**:
- $\bar{r}_i$: 第 $i$ 组的奖励均值
- $G$: 每组 rollout 数（本文 $G=4$）
- $\varepsilon$: 数值稳定项

---

### 公式 3: [[Advantage Estimation|组奖励统计量]]

$$
\bar{r}_i = G^{-1}\sum_{g=1}^{G} r_i^{(g)}, \qquad \sigma_i = \sqrt{G^{-1}\sum_{g=1}^{G}(r_i^{(g)} - \bar{r}_i)^2}
$$

**含义**: 每个 prompt 组的奖励均值 $\bar{r}_i$ 和标准差 $\sigma_i$，作为失败路由门的两个核心输入。

---

### 公式 4: [[特权信息学习|训练路由门]]

$$
z_i^{\text{train}} = \mathbf{1}\!\left[\bar{r}_i \leq Q_{0.3}^{\text{batch}}(\bar{r}) \;\wedge\; \sigma_i \leq Q_{0.3}^{\text{batch}}(\sigma) \;\wedge\; v_i \geq 0.5\right]
$$

**含义**: 三个条件同时满足时，第 $i$ 组被路由为"失败组"并接受特权自蒸馏；$z_i^{\text{train}}=1$ 意味着该组同时参与 [[GRPO]] 和 PSD。

**符号说明**:
- $Q_{0.3}^{\text{batch}}(\bar{r})$: 当前批次奖励均值的第 30 百分位（批次相对阈值）
- $Q_{0.3}^{\text{batch}}(\sigma)$: 当前批次奖励标准差的第 30 百分位
- $v_i \geq 0.5$: 该组有效响应（可解析）的比例不低于 0.5

---

### 公式 5: [[特权信息学习|教师-学生上下文构造]]

$$
\begin{aligned}
c_{i,g,t}^S &= (I_i,\; h_i,\; y_{i,g,<t}) \\
c_{i,g,t}^T &= (I_i,\; h_i,\; \langle\text{priv}\rangle\tau_i^*\langle/\text{priv}\rangle,\; y_{i,g,<t})
\end{aligned}
$$

**含义**: 教师和学生共享图像 $I_i$、历史运动 $h_i$、以及学生已生成的前缀 $y_{i,g,<t}$；教师额外接收用特殊 token 包裹的真值轨迹 $\tau_i^*$，形成特权信息通道。

**符号说明**:
- $I_i$: 当前帧多摄像头图像
- $h_i$: 历史自车运动条件
- $y_{i,g,<t}$: 学生模型在第 $g$ 次 rollout 中已生成的前 $t-1$ 个 token（学生前缀）
- $\tau_i^*$: 真值 BEV 轨迹序列

---

### 公式 6: [[Jensen-Shannon散度|Token 级 JSD 与上限截断]]

$$
J_{i,g,t} = \beta\, D_{\mathrm{KL}}(P^S_{i,g,t} \| M_{i,g,t}) + (1-\beta)\, D_{\mathrm{KL}}(P^T_{i,g,t} \| M_{i,g,t})
$$

$$
d_{i,g,t} = \min\!\{J_{i,g,t},\; 0.05\}
$$

**含义**: 在教师分布 $P^T$ 和学生分布 $P^S$ 的混合分布 $M$ 上计算对称 KL 散度（即 JSD），再以 0.05 截断，防止分布极度偏离时梯度爆炸。

**符号说明**:
- $P^S_{i,g,t}$: 学生在 token $t$ 处的输出分布（取 top-16 token + 余量桶）
- $P^T_{i,g,t}$: 教师在同一位置的输出分布（冻结）
- $M_{i,g,t} = \tfrac{1}{2}(P^S_{i,g,t} + P^T_{i,g,t})$: 混合分布
- $\beta = 0.5$: 对称权重

---

### 公式 7: [[知识蒸馏|PSD 批次损失聚合]]

$$
\ell_{i,g} = \frac{\displaystyle\sum_{t \in \mathcal{A}_{i,g}} d_{i,g,t}}{|\mathcal{A}_{i,g}|}, \qquad
\ell_i = G^{-1}\sum_{g=1}^{G} \ell_{i,g}
$$

$$
\mathcal{L}_{\mathrm{PSD},\mathcal{B}} = \frac{\displaystyle\sum_{i \in \mathcal{B}} n_i^{\mathrm{ans}}\, \ell_i}{\displaystyle\sum_{i \in \mathcal{B}} n_i^{\mathrm{ans}}}
$$

**含义**: 只对轨迹答案 token 集 $\mathcal{A}_{i,g}$ 内的位置取均值，再以每个 prompt 的答案 token 总数 $n_i^{\mathrm{ans}}$ 加权聚合至批次，避免短响应组获得过大权重。

**符号说明**:
- $\mathcal{A}_{i,g}$: 第 $i$ 个 prompt、第 $g$ 次 rollout 中属于轨迹答案段的 token 位置集合
- $n_i^{\mathrm{ans}}$: $\sum_g |\mathcal{A}_{i,g}|$，第 $i$ 个 prompt 全部 rollout 的答案 token 总数

---

### 公式 8: [[GRPO|联合 Actor 损失]]

$$
\mathcal{L}_{\mathcal{B}} = \mathcal{L}_{\mathrm{GRPO},\mathcal{B}} + \lambda\, z_{\mathcal{B}}^{\mathrm{train}}\, \mathcal{L}_{\mathrm{PSD},\mathcal{B}}, \qquad \lambda = 0.1
$$

**含义**: 所有组均参与 [[GRPO]] 损失；被路由的失败批次额外叠加 PSD 损失，权重 $\lambda=0.1$ 保持 GRPO 主导地位。

**符号说明**:
- $\mathcal{L}_{\mathrm{GRPO},\mathcal{B}}$: 标准 GRPO 策略梯度损失（含裁剪）
- $z_{\mathcal{B}}^{\mathrm{train}}$: 当前批次是否包含任何失败组（0/1）
- $\lambda = 0.1$: PSD 权重系数

---

### 公式 9: [[特权信息学习|轮次自演化循环]]

$$
\text{Unresolved}(\pi_k) \;\xrightarrow{\text{PSD}}\; \pi_{k+1} \;\xrightarrow{\text{defines}}\; \text{Unresolved}(\pi_{k+1})
$$

**含义**: $\pi_k$ 的未解决失败组为 $\pi_{k+1}$ 提供特权监督；$\pi_{k+1}$ 同时成为下一轮 $\pi_{k+2}$ 的教师，失败集随策略能力自适应更新。

---

### 公式 10（附录 C）: [[GRPO|通用轨迹奖励]]

$$
r(\hat{\tau}, \tau^*) = \Bigl(1 + T^{-1}\sum_{t=1}^{T}\|\hat{p}_t - p_t^*\|_2^2\Bigr)^{-1}
$$

**含义**: 附录中给出的通用形式，适用于任意步数 $T$ 的轨迹评估（正文中 $T=6$）。

---

## 关键图表

### Figure 1: FIRE-VLA 一览 / Overview

![Figure 1 — FIRE-VLA Overview](https://arxiv.org/html/2608.13395v1/figures/figure1_overview.png)

**说明**: 上半部分展示代表性 nuScenes 场景中 GRPO 与 FIRE-VLA 的轨迹对比（FIRE-VLA 在极端失败场景中轨迹更接近真值）；下半部分以前视图 + 历史运动为条件，说明六步 BEV 航点预测任务的整体设置。

---

### Figure 2: 失败感知自演化方法框图 / Method Pipeline

![Figure 2 — Failure-Informed Self-Evolution Framework](https://github.com/forever-free1/FIRE-VLA/raw/main/assets/framework.png)

**说明**: 完整的 FIRE-VLA 训练流程。(1) 当前策略生成带奖励的 rollout；(2) 路由门判断哪些组进入 PSD；(3) 冻结教师副本接入特权轨迹 token；(4) JSD 蒸馏仅作用于答案段；(5) 更新策略晋升为下一轮教师。

---

### Figure 3: 定性分析——四个典型场景 / Qualitative Results

![Figure 3 — Qualitative Analysis](https://arxiv.org/html/2608.13395v1/qualitative.png)

**说明**: 四行对应四类场景：
- **(a) RL 导致退化** (样本 5449)：标准 GRPO 训练后轨迹变差，FIRE-VLA 维持正常
- **(b) 持续失败修复** (样本 3491)：GRPO 完全失败的场景，FIRE-VLA 显著改善
- **(c) 随机不稳定改善** (样本 5643)：非灾难性随机噪声被抑制
- **(d) 反例** (样本 4798)：FIRE-VLA 未能修复的场景，说明方法并非全能

---

### Table 1: 单样本低温规划结果

| Method | Reward ↑ | L2@1s ↓ | L2@2s ↓ | L2@3s ↓ | Avg. L2 ↓ |
|--------|:--------:|:-------:|:-------:|:-------:|:---------:|
| Standard GRPO | 0.6788 | 0.2135 | 0.6396 | 1.5302 | 0.6421 |
| **FIRE-VLA** | **0.6785** | **0.1997** | **0.6106** | **1.3898** | **0.6023** |

**说明**: 单次低温采样下两者性能相当，L2 差距置信区间跨零，说明 FIRE-VLA **不损害**正常场景的规划精度。

---

### Table 2: G=4 随机评估结果

| Method | Reward ↑ | Avg. L2 ↓ | Persistent Failures ↓ | Any >10m ↓ |
|--------|:--------:|:---------:|:---------------------:|:----------:|
| Standard GRPO | 0.6370 | 1.8478 | 784/6019 (13.03%) | 1.35% |
| **FIRE-VLA** | 0.6156 | **1.5001** | **674/6019 (11.20%)** | **0.83%** |

**说明**: FIRE-VLA 在四次随机 rollout 评估中将平均 L2 降低 18.8%，持续失败率下降 13.9%，极端错误（>10m）从 1.35% 降至 0.83%。注意 FIRE-VLA 的标量奖励反而更低（0.6156 vs 0.6370），因为极端错误的奖励函数在高误差区已饱和，无法区分"稍差"和"极差"。

---

### Table 3（附录 A）: 完整结果对比（含 SFT 基线）

（附录 A 含 SFT、GRPO、FIRE-VLA 在单样本和 G=4 两种模式下的完整奖励与多时间步 L2 指标，以及场景聚类成对 bootstrap 检验的 95% 置信区间，此处摘录核心数字已在 Table 1/2 中呈现。）

---

### Table 4（附录 A）: 场景聚类成对 Bootstrap 检验（10,000 次重复）

Bootstrap 检验结果显示：
- 随机评估均值 L2 的置信区间**不跨零**（FIRE-VLA 显著更好）
- 单样本规划均值 L2 的置信区间**跨零**（差异不显著）

---

### Table 5（附录 A）: 极端随机错误统计

| Method | CVaR95 | CVaR99 | Worst-of-4 | Intra-Std | Any >10m |
|--------|:------:|:------:|:----------:|:---------:|:--------:|
| Standard GRPO | 25.52 | 118.58 | 5.57 | 2.19 | 1.35% |
| **FIRE-VLA** | **17.72** | **79.21** | **4.13** | **1.56** | **0.83%** |

**说明**: FIRE-VLA 的 CVaR95 改善 31%、CVaR99 改善 33%，组内标准差（随机不稳定性）降低 29%，一致说明**改善集中于尾部极端错误**，而非均匀提升。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[nuScenes]] | 训练 1200 prompts / 测试 6019 样本（150 场景） | 多摄像头城市驾驶，提供 BEV 真值轨迹 | RL 训练 + 评估 |

### 实现细节

- **基础模型**: [[Qwen|Qwen2.5-VL-3B-Instruct]]，从 SFT checkpoint 出发
- **RL 训练**: 1200 个唯一 prompt，每组 G=4 rollout（共 4800 次），150 步策略更新
- **教师前向次数**: 失败组的额外教师前向传播（非等算力对比）
- **PSD 权重**: $\lambda=0.1$，JSD 上限 0.05，top-16 token 压缩分布
- **演化轮数**: $K=2$
- **评估**: 6019 样本，低温单样本 + G=4 随机多次采样两种协议

### 可视化结果

定性分析（Figure 3）证明 FIRE-VLA 在持续失败和随机不稳定两类场景均有改善，但存在无法修复的反例，表明改善来自失败分布的局部抑制，不是全局提升。

---

## 批判性思考

### 优点

1. **问题定位精准**: 直接切中 GRPO 在全组失败时信号退化的本质缺陷，而非工程化打补丁
2. **同策略设计**: 教师与学生同参数规模，避免了需要外部大模型或独立预训练教师的额外开销
3. **最小侵入**: 仅对失败组添加辅助损失，不修改 GRPO 结构，易于集成
4. **分析诚实**: 作者主动揭示奖励函数尾部饱和导致标量奖励下降的反直觉现象，并做了详细分布分析（CVaR、winsorized mean）

### 局限性

1. **单 seed 评估**: 没有方差测量，无法判断改善是否稳健
2. **非等算力对比**: 教师前向传播带来额外计算，与标准 GRPO 不公平对比
3. **开环评估**: 未测闭环安全指标，轨迹 L2 改善未必等价于驾驶安全性提升
4. **演化轮次有限**: 仅做 2 轮，单调性未验证
5. **prompt 子集不连续**: 各轮使用不相交的 prompt 子集，无法追踪单个场景的纵向改善

### 潜在改进方向

1. 将特权信息扩展到更丰富的场景语义（障碍物预测、意图标注），而非仅轨迹坐标
2. 设计等算力基线（如更多 GRPO rollout 轮次）做公平对比
3. 结合闭环仿真（CARLA 等）验证尾部错误减少是否转化为安全指标提升

### 可复现性评估

- [x] 代码开源（GitHub 承诺提供）
- [x] 训练细节完整（batch size、学习率、rollout 数量均有列出）
- [ ] 预训练模型（未提供 checkpoint）
- [x] 数据集可获取（nuScenes 公开）

---

## 关联笔记

### 基于

- [[GRPO]]: 核心 RL 算法，FIRE-VLA 在其基础上增加失败路由
- [[Qwen]]: 使用 Qwen2.5-VL-3B 作为基础模型
- [[自蒸馏]]: 特权自蒸馏的方法论出处（On-Policy Self-Distillation / OPSD）

### 对比

- [[Shared-Prefix GRPO]]: 相关的 GRPO 变体，探索共享前缀对组内对比的影响

### 方法相关

- [[Jensen-Shannon散度]]: token 级蒸馏损失的核心度量
- [[知识蒸馏]]: PSD 的上位概念
- [[特权信息学习]]: 教师获取未来信息的设计范式

### 硬件/数据相关

- [[nuScenes]]: 自动驾驶评估数据集

---

## 速查卡片

> [!summary] FIRE-VLA
> - **核心**: GRPO 失败组全局差 → 路由给特权教师做自蒸馏
> - **方法**: 批次相对路由门 + 同策略冻结教师 + JSD 答案段蒸馏 + 轮次演化
> - **结果**: G=4 随机评估均值 L2 -18.8%，CVaR99 -33%，持续失败 -13.9%；单样本性能持平
> - **代码**: [github.com/forever-free1/FIRE-VLA](https://github.com/forever-free1/FIRE-VLA)

---

*笔记创建时间: 2026-08-15*

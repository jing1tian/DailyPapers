---
title: "WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation"
method_name: "WEAVER"
authors: [Arnav Kumar Jain, Yilin Wu, Jesse Farebrother, Gokul Swamy, Andrea Bajcsy]
year: 2026
venue: arXiv
tags: [world-model, robotic-manipulation, diffusion-forcing, flow-matching, policy-evaluation, policy-improvement, test-time-planning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.13672
created: 2026-06-13
---

# 论文笔记：WEAVER, Better, Faster, Longer

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila / Université de Montréal, Carnegie Mellon University |
| 日期 | June 2026 |
| 项目主页 | [arnavkj1995.github.io/WEAVER](https://arnavkj1995.github.io/WEAVER/) |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13672) |

---

## 一句话总结

> WEAVER 是一个同时满足高保真度、长时序一致性和高效推理三大目标的机器人操作[[世界模型]]，在策略评估、策略提升和测试时规划三类下游任务中均超越先前方法。

---

## 核心贡献

1. **统一三大目标的世界模型架构**: 融合[[扩散强制]](Diffusion Forcing)、[[流匹配]](Flow Matching)、预训练编码器与多视角记忆机制，在保真度、一致性、效率上同时 Pareto 超越先前方法
2. **高效推理加速**: 通过 KV 缓存、余弦噪声调度和[[整流流]](Rectified Flow)后训练蒸馏，实现比 Ctrl-World 快 5–10× 的推理速度
3. **全流程下游验证**: 在真实 Franka 机械臂上验证策略评估（ρ=0.870）、策略提升（+38% 无真实交互）和测试时规划（+14%）三类应用

---

## 问题背景

### 要解决的问题

机器人操作中的[[世界模型]]需要同时满足三个目标：
- **保真度（Fidelity）**: 仿真轨迹与真实世界高度相关
- **一致性（Consistency）**: 长时序预测保持时序连贯性
- **效率（Efficiency）**: 推理速度满足实际部署需求

### 现有方法的局限

- **视频生成模型**（如 Ctrl-World）：保真度高但推理效率低，无法用于实时规划
- **JEPA 风格的潜在模型**：效率高但潜在表示无法解码为图像，难以驱动视觉运动策略
- **Dreamer-v4**：从头学习编码器，分布外鲁棒性差
- 上述方法无法同时满足三个目标，形成固有权衡

### 本文的动机

通过融合视频生成与潜在模型的优势——利用预训练 [[VAE]] 编码器保留视觉语义先验，结合[[扩散强制]]实现一致性，再用 KV 缓存和整流流后训练提升效率——WEAVER 打破了"三选二"的权衡。

---

## 方法详解

### 模型架构

WEAVER 采用 **编码器-动力学-奖励-解码器** 四级架构：

- **输入**: 语言指令 $\ell \in \mathcal{L}$ + 多视角 RGB 图像 $\mathbf{I}_t := (I^1_t, \ldots, I^n_t)$ + 本体感知状态 $q_t \in \mathbb{R}^8$ + 动作 $a_t \in \mathcal{A}$
- **Backbone**: 预训练 Stable Diffusion 3 VAE 编码器（冻结权重）
- **核心模块**: 32 层 2D [[Transformer]] + [[稀疏记忆]](Sparse Memory) + [[扩散强制]] 动力学预测
- **输出**: 预测的未来潜在状态 $\hat{\mathbf{z}}_t$、奖励 $\hat{r}_t$、以及可选的解码图像 $\hat{\mathbf{o}}_t$
- **总参数**: 928M

### 核心模块

#### 模块 1: 多视角稀疏记忆（Multi-View Sparse Memory）

**设计动机**: 操作任务需要长时序上下文（如任务初始配置）与短期运动细节（如接触状态），[[稀疏记忆]]以低开销提供两者

**具体实现**:
- **记忆** $\mathbf{z}^{mem}_t := (\ldots, z_{t-2k}, z_{t-k})$：每隔 $k$ 帧采样一次，提供长时序上下文（默认 6 帧，步长 5）
- **历史** $\mathbf{z}^{hist}_t := (z_{t-1}, z_t)$：最近两帧，捕捉短期运动细节
- 使用 Stable Diffusion 3 VAE 将每个视角 $I^j_t$ 编码为 $H \times W$ 个图块 token

#### 模块 2: 潜在动力学模型（Latent Dynamics Model）

**设计动机**: 在潜在空间做预测而非像素空间，兼顾效率与表达力

**具体实现**:
- 32 层 2D [[Transformer]]，包含 $L$ 个动力学块
- 每块含：空间注意力（图块间）+ 因果时序注意力（帧间）
- 归一化使用 [[RMSNorm]]，位置编码使用 [[RoPE]]，注意力头归一化使用 [[QK Normalization]]，前馈网络使用 [[SwiGLU]]
- [[SPRINT]] token 剪枝：随机 drop patch token（概率 0.5），训练时加速而不损失质量

#### 模块 3: 流匹配训练目标（Flow Matching Objective）

**设计动机**: [[流匹配]]比传统扩散更高效，且支持少步推理

**具体实现**:
- 定义噪声插值轨迹，结合[[扩散强制]]对不同时间步独立采样噪声水平，提升长时序一致性

#### 模块 4: 潜在空间奖励与价值估计

**设计动机**: 直接在潜在空间估计奖励，无需解码到像素空间，大幅降低计算开销

**具体实现**:
- **奖励模型**：AdaPool 聚合 + MLP，蒸馏 Robometer 奖励信号，MSE 目标训练
- **价值网络**：与奖励模型共享架构，基于自举 λ-回报训练

### 推理加速三机制

1. **KV 缓存**：跨去噪步骤缓存记忆和历史 token，减少重复前向计算
2. **余弦噪声调度**：$k = 1 - \cos(i\pi / (2K))$，比线性调度更高效地集中采样步数
3. **整流流后训练**（Rectified Flow Post-Training）：将教师模型的轨迹蒸馏为学生模型，以极少前向次数（NFE）实现高质量生成

---

## 关键公式

### 公式 1: [[流匹配|噪声插值轨迹]]

$$
x^\tau_t = \tau x^1_t + (1 - \tau) x^0_t, \quad \tau \in [0, 1)
$$

**含义**: 在数据样本 $x^1_t$（干净潜在）和噪声 $x^0_t$ 之间按比例 $\tau$ 插值，构造[[流匹配]]训练所需的中间状态

**符号说明**:
- $x^1_t$: 目标干净潜在表示（来自编码器）
- $x^0_t$: 标准高斯噪声
- $\tau \sim \mathcal{U}(0,1)$: 插值系数，[[扩散强制]]中对每个时间步独立采样

### 公式 2: [[扩散强制|世界模型训练损失]]

$$
\mathcal{L}^{WM}(\phi) = \mathbb{E}\left[\left\|(x^1_t - x^0_t) - f_\phi(\mathbf{z}^{hist}_t, \mathbf{z}^{mem}_t, \mathbf{a}_t, x^\tau_t, \tau)\right\|^2_2\right]
$$

**含义**: 最小化动力学网络 $f_\phi$ 对"干净-噪声方向"的预测误差，结合[[扩散强制]]逐步序列去噪

**符号说明**:
- $f_\phi$: 潜在动力学模型（32 层 Transformer）
- $\mathbf{z}^{hist}_t$: 短期历史潜在
- $\mathbf{z}^{mem}_t$: 稀疏记忆潜在
- $\mathbf{a}_t$: 动作序列

### 公式 3: [[整流流|后训练蒸馏损失]]

$$
\mathcal{L}^{ReFlow}(\phi) = \mathbb{E}\left[\left\|(\hat{x}^1_t - x^0_t) - f_\phi(\mathbf{z}^{hist}_t, \mathbf{z}^{mem}_t, \mathbf{a}_t, x^\tau_t, \tau)\right\|^2_2\right]
$$

**含义**: 以教师模型生成的样本 $\hat{x}^1_t$ 替换真实数据，蒸馏出可用少步推理的学生模型

**符号说明**:
- $\hat{x}^1_t$: 教师模型（多步 NFE）生成的高质量样本
- 学生模型继承教师架构，仅需 1–4 步 NFE 推理

### 公式 4: [[余弦噪声调度]]

$$
k = 1 - \cos\!\left(\frac{i\pi}{2K}\right)
$$

**含义**: 推理时按余弦曲线非均匀分配 $K$ 个去噪步骤，使早期（高噪声）步骤间隔小、后期（低噪声）步骤间隔大，提升生成质量

**符号说明**:
- $i$: 当前去噪步编号，$i \in \{0, \ldots, K\}$
- $K$: 总去噪步数（NFE）
- $k$: 对应的噪声水平

### 公式 5: [[TD(λ)|自举 λ-回报]]（价值网络训练）

$$
\mathbf{v}^k_t = R(z_t, \ell) + \gamma\!\left((1-\lambda)V(z_{t+1}, \ell) + \lambda\, \mathbf{v}^k_{t+1}\right), \quad \mathbf{v}^k_{t+K} = V(z_{t+K}, \ell)
$$

$$
\mathcal{L}^{critic}(V) = \left\|V(z_t, \ell) - \mathbf{v}^k_t\right\|^2_2
$$

**含义**: 用折扣 λ-回报自举训练价值网络，平衡即时奖励与长远价值估计

**符号说明**:
- $R(z_t, \ell)$: 奖励模型在潜在 $z_t$ 和指令 $\ell$ 下的预测奖励
- $\gamma = 0.995$: 折扣因子
- $\lambda = 0.95$: 回报混合系数
- $V(z_t, \ell)$: 价值函数

### 公式 6: [[Advantage Estimation|优势加权策略提升]]

$$
\hat{A}^b_t = \sum_{l=1}^{H} \gamma^{l-1} R(\hat{z}^b_{t+l}, \ell) + \gamma^H V(\hat{z}^b_{t+H}, \ell) - V(z_t, \ell)
$$

**含义**: 计算每条合成轨迹相对基线（当前价值）的优势，选取最高优势轨迹蒸馏回策略，同时用阈值 $\epsilon_{adv}$ 过滤无收益状态

**符号说明**:
- $\hat{z}^b_{t+l}$: 第 $b$ 条世界模型想象轨迹的第 $l$ 步潜在
- $H = Kh$: 总想象步长（$K$ 次 rollout × $h$ 步 action chunk）
- $\epsilon_{adv}$: 优势过滤阈值，防止在无法获胜状态退化

### 公式 7: [[动作适配器|联合位置预测损失]]

$$
\mathcal{L} = \mathcal{L}_{joint} + 5.0\, \mathcal{L}_{gripper}
$$

$$
q_{t+k} = q_t + \hat{\Delta}_{joint,k}, \quad g_{t+k} = g_t + \hat{\Delta}_{gripper,k}
$$

**含义**: 动作适配器（3 层 MLP）将关节速度命令转换为关节位置增量预测，夹爪损失权重为 5× 以保证精确夹持

**符号说明**:
- $q_t \in \mathbb{R}^7$: 当前关节角度
- $g_t$: 当前夹爪状态
- $\hat{\Delta}_{joint,k}$: 预测的第 $k$ 步关节增量

---

## 关键图表

### Figure 1: WEAVER 三目标全景

![Figure 1](https://arxiv.org/html/2606.13672v1/x1.png)

**说明**: WEAVER 同时满足高保真度（Fidelity）、长时序一致性（Consistency）和高效生成（Efficiency）三个[[世界模型]]目标，解锁三类下游应用：策略评估、策略提升和测试时规划。

### Figure 2: WEAVER 整体架构

![Figure 2](https://arxiv.org/html/2606.13672v1/x2.png)

**说明**: 左侧：编码记忆、历史和动作序列，在潜在空间生成未来 rollout；中间：潜在空间验证器（奖励头 + 价值头）选择高优势样本；右侧：解码器生成可视化图像。整个推理流程主要在潜在空间进行，仅在需要时解码。

### Figure 3: 长时序一致性对比

![Figure 3](https://arxiv.org/html/2606.13672v1/x3.png)

**说明**: 不同时序长度下的 FID 对比。WEAVER 在所有时序长度均保持比 Ctrl-World 更低的 FID，即使使用更少推理预算（NFE=16），说明[[扩散强制]]有效提升了长时序生成一致性。

### Figure 4: 奖励预测与测试时规划

![Figure 4](https://arxiv.org/html/2606.13672v1/x4.png)

**说明**: 左侧：预测奖励曲线与 Robometer 真实进度奖励高度吻合；右侧：基于优势的最佳动作样本筛选，证明潜在空间奖励估计可有效区分不同动作候选。

### Figure 5: FVD vs 推理时间 Pareto 前沿

![Figure 5](https://arxiv.org/html/2606.13672v1/x5.png)

**说明**: WEAVER 在 NFE=8、16、32、50 各点均 Pareto 主导 Ctrl-World，推理时间低 30–50× 的同时实现相当甚至更优的 FVD 质量。两个视角（外置相机、腕部相机）均成立。

### Figure 6: 策略评估结果

![Figure 6](https://arxiv.org/html/2606.13672v1/x6.png)

**说明**: 左侧：PnP Towel、Pour Beans 任务的定性预测对比，WEAVER-FT 预测更贴合真实执行；右侧：相关系数散点图，WEAVER-FT 达到 ρ=0.870，显著优于 Ctrl-World 基线。

### Figure 7: 策略提升结果与数据扩展

![Figure 7 (a)](https://arxiv.org/html/2606.13672v1/x7.png)

![Figure 7 (b)](https://arxiv.org/html/2606.13672v1/x8.png)

**说明**: 左图：各数据来源（真实/合成/混合）的成功率对比，混合数据提升最显著（+37%）；右图：Pour Beans 任务上合成数据量（1K→2K→5K 片段）的扩展规律，更多合成数据持续提升效果并超越纯真实数据。

### Figure 9: 测试时规划结果

![Figure 9](https://arxiv.org/html/2606.13672v1/x10.png)

**说明**: WEAVER 测试时规划（Best-of-4）在 5 个真实任务上平均提升 14%，最高提升 20%（弱策略基线）。与 Ctrl-World 相比速度快约 20×。

### Figure 10: 硬件配置与任务

![Figure 10](https://arxiv.org/html/2606.13672v1/x11.png)

**说明**: 左侧：Franka Emika Panda 机械臂 + 2 个 Zed 2i 外置相机 + 1 个 Zed Mini 腕部相机三视角配置；右侧：5 个操作任务示例（叠碗、PnP 薯片袋、PnP 记号笔、PnP 毛巾、倒咖啡豆）。

### Table 1: 视频生成质量（FID/FVD）对比

| 数据集 | 方法 | NFE | Ext. FID↓ | Ext. FVD↓ | Wrist FID↓ | Wrist FVD↓ | 时间(s)↓ |
|--------|------|-----|-----------|-----------|-----------|-----------|---------|
| DROID (val) | Ctrl-World | 16 | 26.09 | 78.73 | 33.83 | 195.37 | 14.65 |
| DROID (val) | Ctrl-World | 50 | 22.44 | 55.05 | 25.32 | 91.77 | 42.33 |
| DROID (val) | **WEAVER** | 16 | **10.20** | **27.83** | **21.50** | 90.72 | **4.78** |
| DROID (val) | **WEAVER** | 50 | **9.51** | **26.54** | **16.75** | **66.89** | 14.25 |
| Task (OOD) | Ctrl-World | 16 | 36.16 | 139.54 | 38.76 | 277.13 | 14.65 |
| Task (OOD) | Ctrl-World | 50 | 31.44 | 91.48 | 33.47 | 145.86 | 42.33 |
| Task (OOD) | **WEAVER** | 16 | **23.95** | **88.27** | **30.77** | 184.62 | **4.78** |
| Task (OOD) | **WEAVER** | 50 | **23.48** | **87.03** | **27.37** | **145.04** | 14.25 |

**关键发现**: WEAVER 在 DROID 分布内和 OOD 任务数据上均以更低推理时间（3× 更快）实现更优 FID/FVD。

### Table 2: 模型超参数

| 组件 | 参数 | 值 |
|------|------|-----|
| 架构 | 层数 | 32 |
| | 注意力头数 | 16 |
| | 嵌入维度 | 1536 |
| | 头维度 | 96 |
| | SPRINT drop 概率 | 0.5 |
| 预训练 | Batch size | 32 |
| | Batch length | 8 |
| | 记忆帧数 | 6 |
| | 记忆步长 | 5 |
| | 学习率 | 1e-4 |
| | Warmup steps | 10000 |
| | EMA decay | 0.9999 |
| | 训练步数 | 1,000,000 |
| 微调 | 学习率 | 2e-5 |
| | 训练步数 | 16,000 |
| 策略 | 折扣 γ | 0.995 |
| | 回报 λ | 0.95 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | 大规模（76K 轨迹） | 多机器人、多场景、多语言 | 预训练世界模型 |
| $\mathcal{D}^{FT}_{real}$ | 50 rollouts × 5 任务 | 目标场景特定数据 | 微调世界模型 |
| $\mathcal{D}^{val}_{real}$ | 20 rollouts × 5 任务 | 含策略质量多样性 | 策略评估验证 |

### 实现细节

- **Backbone**: Stable Diffusion 3 VAE（冻结，仅用于编码/解码）
- **动力学模型**: 32 层 2D Transformer，928M 参数
- **优化器**: AdamW，lr=1e-4（预训练），lr=2e-5（微调），EMA decay=0.9999
- **Batch Size**: 32，Batch Length 8
- **训练轮数**: 1M steps（预训练，10 天）+ 16K steps（微调）
- **硬件**: 4 × H100 GPU（预训练），Franka Emika Panda（实体实验）
- **基础策略**: π₀.₅ VLA（在 DROID 数据集上训练）
- **预测频率**: 5Hz

### 定量结果摘要

- **策略评估**: WEAVER-FT 达到 ρ=0.870 皮尔逊相关（vs. 真实成功率）
- **策略提升**: 混合数据（真实+合成）微调提升 +37%；纯合成数据提升 +22%（与纯真实数据差距仅 4%）
- **测试时规划**: Best-of-4 平均提升 +14%，最大提升 +20%，推理速度比 Ctrl-World 快 ~20×

---

## 批判性思考

### 优点
1. **三目标全面突破**: 同时在保真度、一致性和效率三个维度超越先前 SOTA，非三选二
2. **工程实用性强**: 928M 参数在 4 × H100 上 10 天预训练，微调仅需 16K steps，部署可行
3. **无需额外真实数据**: 策略提升仅用合成数据可达到真实数据 96% 的效果，大幅降低数据采集成本
4. **下游验证全面**: 在评估、提升、规划三类应用场景均在真实机器人上验证

### 局限性
1. **部分可观测性**: 纯视觉世界模型无法感知接触力/触觉信息，限制接触密集任务的准确性
2. **柔性物体建模弱**: 缺乏物理先验，对可变形物体（毛巾、薯片袋）的轨迹预测精度有限
3. **单块规划局限**: 测试时规划仅选最优单 action chunk，无法做多轮迭代长时序规划
4. **奖励噪声依赖**: 依赖 Robometer 作为监督信号，Robometer 本身存在噪声，限制奖励预测精度

### 潜在改进方向
1. 引入触觉传感器作为额外观测通道，增强接触状态建模
2. 结合物理先验（FEM/PBD）改善柔性物体预测
3. 扩展为多轮树搜索规划，发挥长时序一致性优势

### 可复现性评估
- [ ] 代码开源（项目主页提及，截至论文提交时未确认）
- [x] 预训练模型（项目主页提及）
- [x] 训练细节完整（附录提供完整超参数）
- [x] 数据集可获取（DROID 公开数据集）

---

## 关联笔记

### 基于
- [[Ctrl-World]]: 主要对比基线，同为视频生成式世界模型
- [[Dreamer-v4]]: 潜在空间世界模型代表，WEAVER 解决其 OOD 鲁棒性问题
- [[扩散强制]]: WEAVER 训练目标的核心技术

### 对比
- [[Ctrl-World]]: 在 FID/FVD 质量相当时，WEAVER 推理速度快 5–10×
- [[JEPA]]: 潜在模型代表，但缺乏解码能力

### 方法相关
- [[流匹配]]: 核心训练目标框架
- [[整流流]]: 推理加速蒸馏方法
- [[扩散强制]]: 长时序一致性关键技术
- [[SPRINT]]: Token 剪枝加速方法
- [[RoPE]]: Transformer 位置编码
- [[SwiGLU]]: 前馈网络激活函数
- [[TD(λ)]]: 价值网络训练方法

### 硬件/数据相关
- [[Franka Emika Panda]]: 实验机械臂平台
- [[DROID]]: 预训练数据集
- [[Zed 2i]]: 外置双目相机

---

## 速查卡片

> [!summary] WEAVER: World Model for Robotic Manipulation
> - **核心**: 同时满足保真度/一致性/效率三目标的机器人操作世界模型
> - **方法**: 扩散强制 + 流匹配 + 稀疏记忆 + KV 缓存 + 整流流蒸馏
> - **结果**: 策略评估 ρ=0.870，策略提升 +38%（无真实交互），测试时规划 +14%，速度比 Ctrl-World 快 5–10×
> - **代码**: https://arnavkj1995.github.io/WEAVER/

---

*笔记创建时间: 2026-06-13*

---
title: "WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning"
method_name: "WCM"
authors: [Senyu Fei, Xiaopeng Yu, Siyin Wang, Xianzhong Zhao, Jingjing Gong, Xipeng Qiu]
year: 2026
venue: arXiv
tags: [vla-rl, world-model, critic-model, robot-manipulation, reinforcement-learning, pomdp, flow-matching]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.29613
created: 2026-08-04
---

# 论文笔记：WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tongji University, Shanghai Innovation Institute, Fudan University |
| 日期 | July 2026 |
| 项目主页 | https://sylvestf.github.io/wcm-homepage/ |
| 对比基线 | [[π₀]], [[OpenVLA-OFT]], [[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.29613) / [Code](https://github.com/sylvestf/WCM) |

---

## 一句话总结

> WCM 通过将世界预测目标与价值估计联合训练，使 critic 显式建模时序动态，解决了 VLA-RL 中 critic 因单帧观测而忽视部分可观测性的根本缺陷。

---

## 核心贡献

1. **诊断 VLA-RL 的 critic 瓶颈**: 证明现有 critic 依赖单帧观测与[[POMDP]]理论矛盾，历史帧信息在没有显式时序学习目标时也无法被有效利用。
2. **WCM 联合训练框架**: 提出基于[[LeJEPA]]的轻量架构，同时完成未来潜在状态预测（世界建模）和价值估计，让 critic 表示具备动态感知能力。
3. **通用 RL 流水线兼容性**: 无缝接入 on-policy（PPO/Flow-SDE）和 off-policy（AWR/RECAP）两类训练范式，与[[π₀]]、[[π₀.₅]]、[[OpenVLA-OFT]]三种 VLA backbone 均验证有效，149 个任务上达到 SOTA。

---

## 问题背景

### 要解决的问题

[[VLA-RL]]训练中 critic（值函数）的表示质量直接决定 RL 信号的准确性。机器人操作是[[部分可观测性|部分可观测]]的序贯决策问题，单帧图像无法还原完整系统状态（如被遮挡物体、速度方向、任务进度）。

### 现有方法的局限

- 主流 VLA-RL critic 仅接受单帧观测回归标量回报，与[[POMDP]]的最优决策依赖历史充分统计量的理论相矛盾。
- 简单地拼接多帧历史输入（如 ViT + 历史帧）在稀疏标量监督下仍难以学到帧间动态，因为高维观测空间的时序依赖性需要显式的学习信号来引导。
- 流匹配类 VLA（如 π₀）使用的[[Flow Matching|Flow-SDE]]存在掉落（dropping）现象，critic 质量更是成为瓶颈。

### 本文的动机

来自[[世界动作模型|WAM]]系列工作的启发——**将预测未来状态作为学习目标**可以让表示编码环境动态。WCM 将此思路迁移至 critic 侧：让 critic 在预测下一时刻潜在状态的同时估计价值，迫使其内部表示捕捉跨帧的因果关系，从而大幅提升价值估计精度和 OOD 泛化能力。

---

## 方法详解

### 模型架构

WCM 采用 **编码器-预测器-双头解码器** 架构（基于[[LeJEPA]]）：

- **输入**: 语言指令 $\ell$ + 历史 K 帧观测 $o_{t-K+1:t}$ + 当前动作 $a_t$
- **Backbone**: [[ViT]]（模拟环境）或 VLM 骨干（如 SigLIP，真实机器人）
- **核心模块**: [[Causal Transformer]]世界预测器 + [[Cross-Attention]]语言融合
- **输出**: 当前时刻价值估计 $\hat{V}_t$ 和下一帧潜在状态预测 $\hat{z}_{t+1}$
- **总参数**: 107.2M（真实机器人场景）

### 核心模块

#### 模块1: 观测编码器（Observation Encoder）

**设计动机**: 将原始图像帧压缩为低维[[潜在表示|潜在向量]]，以供时序建模。

**具体实现**:
- 逐帧使用共享[[ViT]]编码器 $\text{enc}_\varepsilon$ 处理历史 K 帧
- 生成序列潜在状态 $\{z_{t-K+1}, \ldots, z_t\}$，每帧独立编码后拼接为序列
- 真实机器人场景下使用 VLM backbone（SigLIP）提取更丰富的语义特征

#### 模块2: 世界预测器（World Predictor / Causal Transformer Trunk）

**设计动机**: 利用[[Causal Transformer]]聚合多帧历史，通过[[Cross-Attention]]注入语言条件，生成包含时序动态的历史摘要向量 $h_t$。

**具体实现**:
- 语言指令 $\ell$ 经[[CLIP]]编码后再通过语言适配器 $\mathcal{A}_\text{lang}$ 映射为 $\mathbf{u}_\ell \in \mathbb{R}^d$
- 潜在序列与语言向量做[[Cross-Attention]]：$\text{XAttn}(z_{t-K+1:t}, \mathbf{u}_\ell)$
- 再通过因果 Transformer Trunk $\text{Tr}_\phi$ 聚合得到历史摘要 $h_t \in \mathbb{R}^d$

#### 模块3: 双头解码器（Dual Decoder Heads）

**设计动机**: 价值估计头提供 RL 训练信号；世界预测头提供显式时序学习目标，迫使 $h_t$ 包含动态信息。

**具体实现**:
- **价值头** $\mathcal{D}_\text{value}$：2 层 MLP，将 $h_t$ 映射为标量 $\hat{V}_t$
- **世界头** $\mathcal{D}_\text{world}$：以 $h_t$、当前帧 $z_t$、动作 $a_t$ 为条件，通过**残差更新**预测 $\hat{z}_{t+1}$（动作条件残差机制避免简单记忆当前帧）

#### 模块4: SIGReg 正则化

**设计动机**: 联合目标中预测头可能导致[[特征塌缩|特征坍塌]]（所有帧映射到同一点），[[SIGReg]]通过强制潜在空间服从各向同性高斯分布来防止此问题。

**具体实现**:
- 对随机单位向量方向 $\mathbf{a}$ 的一维投影 $\mathbf{a}^\top z$ 计算经验特征函数
- 惩罚其与标准高斯特征函数 $\phi(t) = e^{-t^2/2}$ 的偏差
- 有效防止维度坍塌和模式退化，保持潜在空间的多样性

---

## 关键公式

### 公式1: [[ViT|观测编码]]

$$
\bm{z}_{t-k} = \mathrm{enc}_{\varepsilon}(o_{t-k}); \quad \forall k \in \{0, 1, \cdots, K-1\}
$$

**含义**: 将历史 K 帧图像观测逐帧编码为潜在向量序列。

**符号说明**:
- $o_{t-k}$: 时刻 $t-k$ 的图像观测
- $\mathrm{enc}_\varepsilon$: 共享参数的视觉编码器（ViT 或 VLM backbone）
- $\bm{z}_{t-k} \in \mathbb{R}^d$: 对应的潜在向量

### 公式2: [[CLIP|语言条件编码]]

$$
\mathbf{u}_{\ell} = \mathcal{A}_{\mathrm{lang}}\!\left(\mathrm{CLIP}(\ell)\right) \in \mathbb{R}^{d}
$$

**含义**: 将自然语言任务指令编码为与观测潜在空间维度匹配的语言向量。

**符号说明**:
- $\ell$: 自然语言任务指令
- $\mathrm{CLIP}(\cdot)$: CLIP 文本编码器
- $\mathcal{A}_\mathrm{lang}$: 可学习语言适配层（线性投影）
- $\mathbf{u}_\ell$: 语言条件向量

### 公式3: [[Causal Transformer|历史摘要生成]]

$$
\bm{h}_{t} = \mathrm{Tr}_{\phi}\!\left(\operatorname{XAttn}\!\left(\bm{z}_{t-K+1:t},\, \mathbf{u}_{\ell}\right)\right) \in \mathbb{R}^{d}
$$

**含义**: 以语言为条件，通过因果 Transformer 将历史帧序列聚合为单一历史摘要向量，编码时序动态信息。

**符号说明**:
- $\bm{z}_{t-K+1:t}$: K 帧历史潜在向量序列
- $\operatorname{XAttn}$: 以 $\mathbf{u}_\ell$ 为 key/value 的交叉注意力
- $\mathrm{Tr}_\phi$: 因果 Transformer 主干网络
- $\bm{h}_t$: 包含时序动态的历史摘要向量

### 公式4: [[值函数|价值估计]]

$$
\hat{V}_{t} = \mathcal{D}_{\text{value}}(\bm{h}_{t}) \in \mathbb{R}
$$

**含义**: 将历史摘要向量通过价值解码头映射为当前状态的估计累积回报。

**符号说明**:
- $\bm{h}_t$: 历史摘要向量
- $\mathcal{D}_\text{value}$: 价值解码头（2 层 MLP）
- $\hat{V}_t$: 预测的状态价值

### 公式5: [[世界动作模型|下一帧潜在状态预测]]

$$
\hat{\bm{z}}_{t+1} = \mathcal{D}_{\text{world}}(\bm{h}_{t},\, a_{t},\, \bm{z}_{t}) \in \mathbb{R}^{d}
$$

**含义**: 以历史摘要、当前动作和当前帧为条件，通过动作条件残差预测下一帧潜在状态，形成世界建模目标。

**符号说明**:
- $a_t$: 当前时刻执行的动作
- $\bm{z}_t$: 当前帧潜在向量
- $\hat{\bm{z}}_{t+1}$: 预测的下一帧潜在向量
- $\mathcal{D}_\text{world}$: 动作条件残差世界解码头

### 公式6: [[世界动作模型|预测损失]]

$$
\mathcal{L}_{\text{pred}} = \left\|\hat{\bm{z}}_{t+1} - \bm{z}_{t+1}\right\|_{2}^{2}
$$

**含义**: 世界预测头的 MSE 损失，驱使潜在表示编码真实的帧间动态关系。

**符号说明**:
- $\hat{\bm{z}}_{t+1}$: 世界头预测的下一帧潜在向量
- $\bm{z}_{t+1}$: 真实的下一帧潜在向量（stop-gradient）
- $\|\cdot\|_2^2$: 欧几里德距离的平方

### 公式7: [[SIGReg|各向同性高斯正则化]]

$$
\mathcal{L}_{\text{SIGReg}} = \mathbb{E}_{\mathbf{a} \sim \mathcal{U}(\mathcal{S}^{d-1})}\!\left[\int_{\mathbb{R}} \left|\hat{\phi}_{\mathbf{a}^{\top}\bm{z}}(t) - \phi(t)\right|^{2} e^{-t^{2}} dt\right]
$$

**含义**: 对潜在空间任意方向的投影分布施加各向同性高斯约束，防止特征坍塌和模式退化。

**符号说明**:
- $\mathbf{a} \sim \mathcal{U}(\mathcal{S}^{d-1})$: 在单位超球面上均匀采样的随机方向
- $\hat{\phi}_{\mathbf{a}^{\top}\bm{z}}(t)$: $\mathbf{a}^\top z$ 的经验特征函数
- $\phi(t) = e^{-t^2/2}$: 标准正态分布的特征函数
- $e^{-t^2}$: 高斯权重，使积分收敛

### 公式8: [[奖励函数|稀疏奖励设计]]

$$
r_{t} = \begin{cases} 0 & \text{if } t = T \text{ and success} \\ -C_{\text{fail}} & \text{if } t = T \text{ and failure} \\ -1 & \text{otherwise} \end{cases}, \qquad G_{t} = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}
$$

**含义**: 步骤级时间惩罚 + 终态奖励的稀疏奖励设计，鼓励高效完成任务，$C_\text{fail}$ 在真实机器人实验中设为 300。

**符号说明**:
- $T$: 轨迹终止时刻
- $C_\text{fail}$: 失败惩罚系数（真实机器人实验中 = 300）
- $\gamma$: 折扣因子
- $G_t$: 折扣累积回报（回归目标），归一化至 $[-1, 1]$

### 公式9: [[值函数|价值损失]]

$$
\mathcal{L}_{\text{value}} = \left\|\hat{V}_{t} - G_{t}\right\|_{2}^{2}
$$

**含义**: 价值头的 L2 回归损失，监督 critic 的价值估计准确性。

**符号说明**:
- $\hat{V}_t$: critic 输出的预测价值
- $G_t$: 折扣累积回报（ground truth）

### 公式10: [[WCM|WCM 总训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{value}} + \lambda \cdot \mathcal{L}_{\text{pred}} + \eta \cdot \mathcal{L}_{\text{SIGReg}}
$$

**含义**: WCM 的三目标联合损失，权衡价值估计准确性、时序预测能力和表示多样性。最优 $\lambda$ 范围为 $[0.3, 0.5]$。

**符号说明**:
- $\lambda$: 预测损失权重，最优区间 $[0.3, 0.5]$
- $\eta$: SIGReg 正则化权重
- $\mathcal{L}_\text{value}$: 价值回归损失（主要监督信号）
- $\mathcal{L}_\text{pred}$: 世界预测损失（时序表示增强）
- $\mathcal{L}_\text{SIGReg}$: 各向同性高斯正则化（防止特征坍塌）

### 公式11: [[PPO|PPO Actor 损失（On-Policy 自回归 VLA）]]

$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}_{t}\!\left[\min\!\left(\rho_{t}(\pi)\hat{A}_{t},\; \mathrm{clip}\!\left(\rho_{t}(\pi), 1-\varepsilon, 1+\varepsilon\right)\hat{A}_{t}\right)\right]
$$

**含义**: PPO 裁剪目标函数，配合 WCM 提供的 GAE 优势估计，用于 OpenVLA-OFT 等自回归 VLA 的 on-policy 训练。

**符号说明**:
- $\rho_t(\pi) = \pi(a_t|o_t) / \pi_\text{old}(a_t|o_t)$: 策略比率
- $\hat{A}_t$: WCM 输出的广义优势估计（GAE）
- $\varepsilon$: PPO 裁剪系数

### 公式12: [[行为克隆|AWR Actor 损失（Off-Policy 自回归 VLA）]]

$$
\mathcal{L}_{\text{actor}} = \mathbb{E}_{t}\!\left[-\log \pi(a_{t}|o_{t}) \exp\!\left(\frac{1}{\beta}\left(G_{t} - V(o_{t-K+1:t})\right)\right)\right]
$$

**含义**: AWR（Advantage-Weighted Regression）利用 WCM 估计的价值对 BC 损失加权，通过优势权重过滤低质量演示。

**符号说明**:
- $\beta$: 温度系数，控制优势权重的尖锐程度
- $G_t - V(o_{t-K+1:t})$: 优势估计（回报减去 WCM 价值）

### 公式13: [[RECAP|RECAP 优势估计（Off-Policy 流匹配 VLA）]]

$$
A(o_{t}, a_{t}, \ell) = \mathrm{normalize}(G_{t:t+N}) + \gamma^{N} \hat{V}(o_{t+N-K+1:t+N}) - \hat{V}(o_{t-K+1:t})
$$

**含义**: RECAP 的 N 步优势估计，结合 WCM 提供的多帧历史价值估计，实现 π₀ 等流匹配 VLA 的 off-policy 训练。

**符号说明**:
- $N$: N 步回报窗口
- $G_{t:t+N}$: N 步内的累积折扣回报
- $\hat{V}(o_{t-K+1:t})$: WCM 对当前 K 帧历史的价值估计
- $\hat{V}(o_{t+N-K+1:t+N})$: WCM 对 N 步后历史的价值估计（自举）

---

## 关键图表

### Figure 1: 动机对比 / Motivation Overview

![Figure 1](https://arxiv.org/html/2607.29613v1/x1.png)

**说明**: 对比现有单帧 critic 与 WCM 的核心差异。现有方法在[[部分可观测性|部分可观测]]环境中使用单帧图像回归价值，导致 critic 表示无法捕捉时序结构。WCM 通过联合预测未来潜在状态显式引入时序学习目标。

### Figure 2: WCM 整体架构与训练流水线

![Figure 2](https://arxiv.org/html/2607.29613v1/x2.png)

**说明**: **(a)** 架构：观测编码器将 K 帧图像映射为潜在序列，[[Causal Transformer]]主干聚合历史，双头解码器分别输出价值 $\hat{V}_t$ 和下一帧预测 $\hat{z}_{t+1}$。**(b)** On-policy 流水线：WCM 提供 GAE 优势估计，配合 PPO（自回归）或 Flow-SDE（流匹配）更新 actor。**(c)** Off-policy 流水线：统一数据缓冲区支持 AWR（自回归）或 RECAP（流匹配）的稳定更新。

### Figure 3: VLA-RL 的 POMDP 图模型

![Figure 3](https://arxiv.org/html/2607.29613v1/x3.png)

**说明**: 机器人操作的[[POMDP]]图模型，实心圆为可观测量（图像 $o_t$、动作 $a_t$），空心圆为隐藏系统状态（真实物理状态 $s_t$）。该图直观说明单帧观测无法推断完整隐藏状态的理论依据。

### Figure 4: MetaWorld 和 CALVIN 基准结果

![Figure 4](https://arxiv.org/html/2607.29613v1/x4.png)

**说明**: WCM 在 MetaWorld 和 CALVIN 上的结果，报告成功率或平均序列长度（含误差棒）。WCM 在两个经典操作基准上均达到 SOTA，验证了方法的跨任务泛化性。

### Figure 5: Critic 架构消融与历史长度分析

![Figure 5](https://arxiv.org/html/2607.29613v1/x5.png)

**说明**: 对比三种 critic 配置：(1) **MLP**（单帧基线）；(2) **ViT**（多帧历史，$\lambda=0$，无世界预测）；(3) **WCM**（多帧历史 + 世界预测）。结果显示历史帧本身不足以带来显著提升，世界预测目标是关键。最优历史长度 $K=3$。

| Critic 配置 | ManiSkill IND | ManiSkill OOD | 说明 |
|------------|--------------|--------------|------|
| MLP（单帧基线） | 77.8% | 36.9% | 无历史，无预测 |
| ViT（$\lambda=0$） | 82.1% | ~45% | 有历史，无预测目标 |
| **WCM（完整）** | **84.4%** | **51.5%** | 有历史 + 世界预测 |

**关键发现**: 世界预测目标对 OOD 泛化提升尤为显著（+6.5% vs ViT），说明显式动态建模增强了分布外泛化能力。

### Figure 6: 预测损失权重 λ 的影响

![Figure 6](https://arxiv.org/html/2607.29613v1/x6.png)

**说明**: IND 和 OOD 成功率随 $\lambda$ 变化的曲线（星号标注最优）。最优 $\lambda$ 区间为 $[0.3, 0.5]$，过大的 $\lambda$ 会使世界预测目标压制价值估计，导致性能下降。

### Figure 7: Flow-SDE 掉落现象与 Zero-Value 配置分析

![Figure 7](https://arxiv.org/html/2607.29613v1/x7.png)

**说明**: 展示流匹配 VLA（如 π₀）在使用 Flow-SDE 进行 on-policy 训练时出现的 **dropping phenomenon**（轨迹质量骤降），以及 Zero-Value 初始化（critic 全零初始化）对训练稳定性的影响分析。

### Figure 8: 部分可观测条件下的 Rollout 过程

![Figure 8](https://arxiv.org/html/2607.29613v1/x8.png)

**说明**: 直观展示 rollout 过程中观测如何只能揭示隐藏状态的部分信息，强调[[POMDP]]设定下历史依赖的必要性。

### Figure 9: 真实机器人实验环境

![Figure 9](https://arxiv.org/html/2607.29613v1/x9.png)

**说明**: WidowX-250S 机械臂实验场景，包含 7 个操作任务：Carrot、Banana、Pepper（刚性抓取）、Cloth、Towel（可变形物体）、Stovetop（动态任务）、Sushi（精细操作）。

### Figure 10: 不同策略变体的训练吞吐量对比

![Figure 10](https://arxiv.org/html/2607.29613v1/x10.png)

**说明**: 跨三个任务衡量每小时成功 rollout 数。WCM 的轻量 critic 设计保持了可接受的计算开销，验证了在实际 RL 训练中的可行性。

### Figure 11: 全部 8 种设置下的训练曲线

![Figure 11](https://arxiv.org/html/2607.29613v1/x11.png)

**说明**: 8 种配置（3 个 VLA backbone × 多种 RL 方法）的完整训练曲线，从 SFT 初始化起约 1,000 步 RL 训练后达到最优性能，展示了 WCM 的样本效率。

### Figure 12: 仿真环境中的 WCM 价值曲线

![Figure 12](https://arxiv.org/html/2607.29613v1/x12.png)

**说明**: WCM 在 ManiSkill 仿真环境中对成功轨迹和失败轨迹的价值预测曲线对比，展示 WCM critic 具有强烈的区分能力——成功轨迹价值单调递增，失败轨迹价值提前下跌。

### Figure 13: 真实世界的 WCM 价值曲线

![Figure 13](https://arxiv.org/html/2607.29613v1/x13.png)

**说明**: 真实机器人实验中 WCM 的价值曲线，即使在环境噪声下仍能区分成功/失败轨迹，验证了 critic 表示在真实物理世界的鲁棒性。

### Figure 14: 机器人可达工作空间的价值热力图

![Figure 14](https://arxiv.org/html/2607.29613v1/x14.png)

**说明**: 不同坐标位置对应 WCM 预测价值的热力图，暖色表示高价值区域（靠近目标物体），冷色表示低价值区域（远离或障碍区）。直观展示 WCM 学到了语义上合理的空间价值分布。

### Table 1: ManiSkill IND/OOD 性能对比

| Backbone | 方法 | IND avg. | Δ | Vision | Semantic | Execution | OOD avg. | Δ |
|----------|------|----------|---|--------|----------|-----------|----------|---|
| π₀ | SFT | 38.4 | — | 32.6 | 8.4 | 13.2 | 18.1 | — |
| | + FlowSDE | 78.8 | +40.4 | 61.1 | 25.4 | 31.5 | 39.3 | +21.2 |
| | + FlowNoise | 77.8 | +39.4 | 63.4 | 23.1 | 24.2 | 36.9 | +18.8 |
| | + π-stepNFT | 79.2 | +40.8 | 69.1 | 49.1 | 33.1 | 50.4 | +32.3 |
| | **+ WCM (Ours)** | **84.4±1.2** | **+46.0±1.2** | **69.1±0.7** | **49.8±2.2** | **35.6±1.3** | **51.5±1.5** | **+33.4±1.5** |
| π₀.₅ | SFT | 47.0 | — | 40.2 | 16.6 | 22.4 | 26.4 | — |
| | + FlowSDE | 90.9 | +43.9 | 68.0 | 34.5 | 45.4 | 49.3 | +22.9 |
| | + FlowNoise | 89.7 | +42.7 | 69.9 | 35.5 | 54.9 | 53.4 | +27.0 |
| | + π-stepNFT | 85.4 | +38.4 | 76.9 | 56.6 | 45.1 | 59.5 | +33.1 |
| | **+ WCM (Ours)** | **91.9±0.4** | **+44.9±0.4** | **78.1±1.6** | **58.5±1.0** | **56.5±1.5** | **64.4±1.4** | **+38.0±1.4** |
| OpenVLA-OFT | SFT | 28.1 | — | 27.7 | 13.0 | 11.7 | 18.3 | — |
| | + GRPO | 94.1 | +66.0 | 84.7 | 45.5 | 44.7 | 60.6 | +42.3 |
| | + PPO | 97.7 | +69.6 | 92.1 | 64.8 | 73.6 | 77.1 | +58.8 |
| | **+ WCM (Ours)** | **99.0±0.4** | **+70.9±0.4** | **92.4±0.5** | **65.9±1.0** | **75.5±0.7** | **77.9±0.8** | **+59.6±0.8** |

**说明**: WCM 在三种 VLA backbone 上均刷新 SOTA；尤其在 OOD 泛化上超过 π-stepNFT 等强基线，验证了世界预测带来的泛化能力提升。

### Table 2: LIBERO-Plus 泛化基准详细结果

| Backbone | 方法 | Camera | Env | Init | Language | Noise | Layout | Light | 总计 |
|----------|------|--------|-----|------|----------|-------|--------|-------|------|
| π₀ | Full-SFT | 85.4 | 90.1 | 12.5 | 65.8 | 89.7 | 75.3 | 89.3 | 71.2 |
| | One-SFT | 32.3 | 59.4 | 1.5 | 42.3 | 48.7 | 50.2 | 48.0 | 39.1 |
| | **+ WCM** | **88.3** | **91.0** | **19.3** | **65.9** | **87.5** | **76.0** | **90.9** | **72.8** |
| | Δ（One-SFT→WCM） | +56.0 | +31.6 | +17.8 | +23.6 | +38.8 | +25.8 | +42.9 | **+33.7** |
| π₀.₅ | Full-SFT | 78.8 | 89.9 | 24.7 | 73.8 | 88.2 | 77.0 | 78.3 | 72.9 |
| | One-SFT | 32.8 | 52.7 | 1.9 | 38.7 | 50.5 | 46.1 | 51.2 | 38.0 |
| | **+ WCM** | **80.8** | **90.3** | **31.5** | **65.0** | **86.3** | **79.3** | **91.6** | **73.7** |
| | Δ | +48.0 | +37.6 | +29.6 | +26.3 | +35.8 | +33.2 | +40.4 | **+35.7** |
| OpenVLA-OFT | Full-SFT | 69.4 | 88.5 | 49.6 | 66.3 | 78.7 | 70.3 | 88.2 | 71.7 |
| | One-SFT | 12.8 | 49.6 | 23.0 | 30.0 | 23.3 | 34.5 | 42.0 | 29.3 |
| | **+ WCM** | **74.6** | **94.9** | **51.3** | **65.8** | **84.1** | **63.6** | **94.8** | **74.0** |
| | Δ | +61.8 | +45.3 | +28.3 | +35.8 | +60.8 | +29.1 | +52.8 | **+44.7** |

**说明**: WCM 将 One-SFT（单任务 SFT）的性能提升至接近 Full-SFT（全数据 SFT）水平，在约 250 步 RL 训练后即可达到，展示了强大的数据效率和跨分布泛化。

### Table 3: 真实机器人实验结果（50 次 rollout 成功次数）

| Backbone | 方法 | Carrot | Banana | Pepper | Cloth | Towel | Stovetop | Sushi |
|----------|------|--------|--------|--------|-------|-------|----------|-------|
| OpenVLA-OFT | SFT | 24/50 | 11/50 | 19/50 | 15/50 | 16/50 | 1/50 | 9/50 |
| | + AWR | 29/50 | 23/50 | 24/50 | 29/50 | 35/50 | 10/50 | 17/50 |
| | **+ WCM** | **32/50** | **26/50** | **26/50** | **38/50** | **40/50** | **15/50** | **22/50** |
| π₀.₅ | SFT | 25/50 | 31/50 | 34/50 | 21/50 | 24/50 | 4/50 | 13/50 |
| | + RECAP | 33/50 | 37/50 | 40/50 | 32/50 | 33/50 | 27/50 | 18/50 |
| | **+ WCM** | **44/50** | **38/50** | **43/50** | **38/50** | **35/50** | **33/50** | **24/50** |

**说明**: WCM 在真实机器人 7 项任务上全面超越基线，尤其在 Stovetop（动态任务）和可变形物体（Cloth、Towel）上提升最大，轨迹更流畅且碰撞更少。

### Table 4: 仿真到真实迁移结果

| 模型 | 方法 | Sim-IND | Sim-OOD | Carrot | Banana | Pepper |
|------|------|---------|---------|--------|--------|--------|
| π₀.₅ | Sim SFT | 47.0 | 26.4 | 0/25 | 0/25 | 0/25 |
| | + Sim RL (WCM) | 91.9 | 64.4 | 7/25 | 7/25 | 6/25 |
| | Real SFT | 6.9 | 5.4 | 13/25 | 2/25 | 4/25 |
| | + Sim RL→Real | 73.5 | 37.9 | 11/25 | 8/25 | 9/25 |

**说明**: 仿真 RL 后的模型具备一定 sim-to-real 迁移能力，但直接真实机器人 SFT 仍更有优势，说明 sim-to-real gap 仍是挑战。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ManiSkill | 25 个拾放任务 | IND/OOD 双评估；3 类泛化（视觉/语义/执行） | 主基准，Table 1 |
| MetaWorld | 多样化操作 | 超出拾放范围的复杂操作 | 泛化验证，Figure 4 |
| CALVIN | 长程序列任务 | 0-5 完成长度评估 | 长程泛化，Figure 4 |
| LIBERO-Plus | 7 类泛化变体 | 相机/环境/初始/语言/噪声/布局/光照 | 系统性泛化，Table 2 |
| 真实 WidowX-250S | 7 个任务 × 50 rollout | 刚性+可变形+动态 | 真实机器人验证，Table 3 |

### 实现细节

- **Backbone**: ViT（仿真）/ SigLIP（真实机器人 VLM 特征）
- **历史长度**: $K = 3$（最优）
- **损失权重**: $\lambda \in [0.3, 0.5]$
- **训练步数**: 约 1,000 步 RL（SFT 后）
- **硬件**: 8 × H100（仿真训练），RTX 5090（真实机器人推理）
- **参数量**: 107.2M（真实机器人 WCM critic）
- **RL 方法**: PPO/AWR（自回归 VLA），Flow-SDE/RECAP（流匹配 VLA）

### 可视化结果

- WCM 的价值曲线（Figure 12-13）显示对成功/失败轨迹的强判别力
- 工作空间价值热力图（Figure 14）呈现语义上合理的空间价值分布
- 真实机器人实验展示 WCM 带来更平滑轨迹和更少碰撞

---

## 批判性思考

### 优点

1. **理论有据**: 从[[POMDP]]理论出发严格论证单帧 critic 的局限，而非仅靠经验验证。
2. **高兼容性**: 无需修改 actor 即可插入现有 VLA-RL 流水线，与 on/off-policy 和自回归/流匹配两类范式均兼容。
3. **轻量高效**: 107.2M 参数的 WCM 相比完整 VLA（数十亿参数）极轻量，计算开销可接受。
4. **真实验证充分**: 7 个任务 × 50 rollout 的真实机器人实验在同类工作中属于较完整的验证。
5. **可解释性强**: 价值热力图和曲线对比提供了直觉上易理解的质量验证。

### 局限性

1. **历史长度固定**: $K=3$ 在本文任务上最优，但复杂长程任务可能需要更长历史，动态调整机制未探讨。
2. **Sim-to-Real Gap**: 仿真 RL 结果直接迁移到真实机器人性能下降明显（见 Table 4），世界预测是否能缩小域差距有待研究。
3. **SIGReg 超参数**: $\eta$ 的选择敏感性未详细分析，可能在其他任务上需要重新调参。
4. **仅评估短程任务**: 149 个任务主要是单阶段/短程操作，长程复杂任务中 critic 价值估计的误差累积未被研究。

### 潜在改进方向

1. 将 WCM 的世界预测扩展为多步预测（multi-step rollout），提供更长远的状态预测作为辅助信号。
2. 探索自适应历史长度（可变 $K$），针对任务复杂度动态调整。
3. 结合 Domain Randomization 缩小 WCM 在 sim-to-real 场景下的 gap。
4. 将 WCM 应用于更高层次的 token-level 价值估计，探索与 LLM RLHF 的协同。

### 可复现性评估

- [x] 代码开源（https://github.com/sylvestf/WCM）
- [x] 预训练模型（HuggingFace: Sylvest/WCM）
- [x] 训练细节完整（超参数、GPU 配置均有报告）
- [x] 数据集可获取（ManiSkill/MetaWorld/CALVIN 均为公开基准）

---

## 关联笔记

### 基于

- [[π₀]]: WCM 主要验证 backbone 之一（flow-matching VLA）
- [[π₀.₅]]: WCM 验证 backbone，更强的 VLA 基础
- [[OpenVLA-OFT]]: WCM 验证 backbone（autoregressive VLA）
- [[世界动作模型|WAM]]: WCM 的世界预测思路来源，WAM 在 actor 侧做预测，WCM 在 critic 侧
- [[POMDP]]: WCM 的理论基础，部分可观测马尔可夫决策过程

### 对比

- [[VLA-RL]]: WCM 是 VLA-RL 的 critic 改进，与现有单帧 critic 方法对比
- [[SimpleVLA-RL]]: 同类 VLA-RL 框架，critic 设计对比
- [[PPO]]: WCM 兼容的 on-policy RL 算法

### 方法相关

- [[JEPA]]: WCM 基于 LeJEPA（轻量 JEPA 变体），联合嵌入预测架构
- [[SIGReg]]: WCM 使用 SIGReg 正则化防止特征坍塌
- [[Causal Transformer]]: WCM 世界预测器的核心架构
- [[值函数]]: WCM 改进的核心组件
- [[强化学习]]: WCM 所处的更广泛框架
- [[Flow Matching]]: WCM 兼容的动作生成范式（π₀ 系列使用）

### 硬件/数据相关

- [[WidowX-250S]]: 真实机器人实验平台

---

## 速查卡片

> [!summary] WCM: A World Critic Model for VLA-RL
> - **核心**: 通过联合预测未来潜在状态与估计价值，让 critic 显式建模时序动态，解决 VLA-RL 中单帧 critic 的[[POMDP|部分可观测]]瓶颈
> - **方法**: [[LeJEPA]] 架构 + 双头解码器（价值头 + 世界预测头）+ SIGReg 防坍塌正则化；兼容 on/off-policy × 自回归/流匹配 4 种训练范式
> - **结果**: 149 任务 SOTA；π₀ +46%、π₀.₅ +45%、OpenVLA-OFT +71%（vs SFT）；7 项真实机器人任务全面超越基线
> - **代码**: https://github.com/sylvestf/WCM

---

*笔记创建时间: 2026-08-04*

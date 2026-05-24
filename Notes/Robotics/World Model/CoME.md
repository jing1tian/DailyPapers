---
title: "Composition of Memory Experts for Diffusion World Models"
method_name: "CoME"
authors: [Sebastian Stapf, Pablo Acuaviva Huertos, Aram Davtyan, Paolo Favaro]
year: 2026
venue: ICLR 2026
tags: [world-model, diffusion-model, memory, product-of-experts, video-generation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.18813v1
created: 2026-05-21
---

# 论文笔记：Composition of Memory Experts for Diffusion World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Computer Vision Group, University of Bern |
| 日期 | May 2026 |
| 项目主页 | [wiqzard.github.io/composition-of-memory-experts](https://wiqzard.github.io/composition-of-memory-experts/) |
| 对比基线 | [[NWM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.18813) |

---

## 一句话总结

> 将短期、长期、空间长期三类异构记忆专家通过对比乘积专家（PoCE）框架组合进[[扩散世界模型]]，在不引入模式崩溃的前提下实现超过500帧的时序一致性。

---

## 核心贡献

1. **Composition of Memory Experts (CoME)**: 提出将异构记忆专家组合到单一[[扩散模型]]推理流水线，各专家互补覆盖短期至长期时间尺度
2. **Product of Contrastive Experts (PoCE)**: 引入对比系数 $\alpha_i > 1$ 抑制虚假模式，解决朴素乘积专家的模式崩溃问题
3. **Test-Time LoRA 长期记忆**: 利用 [[LoRA]] 测试时微调将历史帧信息压缩进模型权重，以亚线性计算代价处理1000+帧上下文

---

## 问题背景

### 要解决的问题

[[世界模型]]在预测长视频序列时需要维护跨越数百帧的时序一致性。现有单一架构方案存在根本矛盾：既要精确还原局部细节，又要高效处理极长历史上下文。

### 现有方法的局限

- **[[Transformer]]**：通过注意力机制保留细节，但 $O(n^2)$ 的注意力代价使上下文窗口受限
- **[[SSM]]（State Space Model）/ [[Mamba]]**：线性扩展，但压缩历史时会损失细节保真度
- **朴素 [[Product of Experts]]（PoE）**：将多模型分布相乘容易产生模式崩溃，生成结果退化

### 本文的动机

不同记忆机制各有专长——无需将所有记忆压缩进单一架构，而是让专家并行工作，在[[扩散模型]]的采样阶段通过分数函数加权合成。对比配方在不改变主模式形状的前提下选择性抑制虚假模式。

---

## 方法详解

### 模型架构

CoME 采用 **分布式专家组合** 架构，基础模型为预训练[[扩散模型]]（DiT-S 骨干）：

- **输入**: 历史帧 $c$（条件）+ 当前噪声帧 $x_t$
- **Backbone**: [[Diffusion Transformer (DiT)]]，条件通过 cross-attention 注入
- **核心模块**: [[Product of Contrastive Experts]] 组合三类记忆专家
- **输出**: 去噪预测 $\hat{x}_0$，经 DDPM 反向链采样未来帧
- **总参数**: 基础模型参数 + 每专家 LoRA 增量（rank 8-64）

### 核心模块

#### 模块1: Short-Term Memory Expert (STM)

**设计动机**: 利用[[扩展上下文窗口]]捕捉 10-100 帧内的精细局部动态

**具体实现**:
- 使用 [[DiT]] 在训练时扩展条件帧数至 10-100 帧（基础模型仅用 3 帧）
- 通过 [[分块推理（Chunked Inference）]] 维持计算可控
- 条件向量 $c_{ST}$ 包含最近 N 帧的特征

#### 模块2: Long-Term Memory Expert (LTM)

**设计动机**: 存储 100-1000 帧的情节历史（Episodic History），使用[[测试时微调]]避免二次复杂度

**具体实现**:
- 对预训练模型施加 [[LoRA]] 适配（rank 16-64）
- 每帧在线执行 2-3 次梯度步（test-time finetuning）
- LoRA 隐式正则化防止灾难遗忘
- 条件向量 $c_{LT}$ 为空（历史已编码进权重 $\psi(c_{LT})$）

#### 模块3: Spatial Long-Term Memory Expert (SLTM)

**设计动机**: 通过相机位姿或空间先验强制几何一致性（Geometric Coherence）

**具体实现**:
- 利用相机姿态条件 $S$（位置/朝向）作为额外条件信号
- 与 LTM 共享骨干，针对空间对齐单独训练
- 确保不同视角下同一区域的外观一致

---

## 关键公式

### 公式1: [[扩散模型|扩散前向过程]]

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I})
$$

**含义**: 前向扩散逐步向干净图像添加高斯噪声

**符号说明**:
- $x_t$: 第 $t$ 步的噪声图像
- $\beta_t$: 第 $t$ 步的噪声调度系数
- $x_0$: 原始干净图像

### 公式2: [[扩散模型|扩散反向过程]]

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \tilde{\beta}_t \mathbf{I})
$$

**含义**: 学习参数化的反向去噪分布

**符号说明**:
- $\mu_\theta$: 网络预测的均值
- $\tilde{\beta}_t$: 反向过程方差

### 公式3: [[扩散模型|扩散训练损失]]

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, t, \varepsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]
$$

**含义**: 均方误差噪声预测损失

**符号说明**:
- $\varepsilon$: 真实添加的噪声
- $\varepsilon_\theta$: 网络预测的噪声

### 公式4: [[Product of Experts|朴素乘积专家]]

$$
p(x \mid \mathcal{M}) \propto p_{\text{query}}(x) \prod_i p_i(x)
$$

**含义**: 多专家分布直接相乘，结果为每个专家约束的交集

**符号说明**:
- $p_{\text{query}}$: 查询分布（基础模型）
- $p_i$: 第 $i$ 个专家的分布
- $\mathcal{M}$: 所有记忆条件的集合

### 公式5: [[Product of Contrastive Experts|对比乘积专家（PoCE）]]

$$
p(x \mid \mathcal{M}) \propto \prod_{i=1}^{K} \tilde{p}_i(x), \quad \tilde{p}_i(x) \propto p_i(x)^{\alpha_i} \cdot \bar{p}_i(x)^{1 - \alpha_i}
$$

**含义**: 每个专家使用对比系数 $\alpha_i > 1$ 压制非条件基线，抑制虚假模式而不扭曲主模式形状

**符号说明**:
- $\alpha_i > 1$: 对比系数，控制条件强度
- $\bar{p}_i(x)$: 第 $i$ 个专家的无条件基线分布
- $p_i(x)$: 第 $i$ 个专家的条件分布

### 公式6: [[CoME|完整CoME分布]]

$$
\begin{aligned}
p_{\text{CoME}}(x \mid c, c_{ST}, c_{LT}, S) \propto\;
&\left[p_\theta(x|\emptyset)^{1-\alpha_0} p_\theta(x|c)^{\alpha_0}\right] \\
\cdot\; &\left[p_\phi(x|\emptyset)^{1-\alpha_1} p_\phi(x|c_{ST})^{\alpha_1}\right] \\
\cdot\; &\left[p_{\psi(\emptyset)}(x|c)^{1-\alpha_2} p_{\psi(c_{LT})}(x|c)^{\alpha_2}\right] \\
\cdot\; &\left[p_\lambda(x|\emptyset)^{1-\alpha_3} p_\lambda(x|S)^{\alpha_3}\right]
\end{aligned}
$$

**含义**: CoME 完整分布，四项分别对应基础模型、STM、LTM、SLTM

**符号说明**:
- $\theta$: 基础模型参数
- $\phi$: STM 参数（训练时扩展上下文）
- $\psi(c_{LT})$: 经历史帧微调后的 LTM 参数
- $\lambda$: SLTM 参数（空间条件化）
- $c, c_{ST}, c_{LT}, S$: 基础条件、短期条件、长期条件、空间条件

### 公式7: [[Classifier-Free Guidance|分数函数合成]]

$$
\nabla_{x_t} \log p_{\text{CoM}}(x_t) = \sum_k \left[ \alpha_k \nabla_{x_t} \log p_k(x_t) + (1 - \alpha_k) \nabla_{x_t} \log \bar{p}_k(x_t) \right]
$$

**含义**: 采样时将各专家的分数函数加权求和，等价于在扩散采样步骤中合成各专家的噪声预测

**符号说明**:
- $\nabla_{x_t} \log p_k(x_t)$: 第 $k$ 个专家的条件分数函数
- $\nabla_{x_t} \log \bar{p}_k(x_t)$: 第 $k$ 个专家的无条件分数函数

### 公式8: [[Langevin Dynamics|Langevin 更新步]]

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t \eta
$$

**含义**: DDPM 采样的单步更新，将合成分数代入即得 CoME 的采样步骤

**符号说明**:
- $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$: 累积噪声调度
- $\sigma_t$: 采样噪声标准差
- $\eta \sim \mathcal{N}(0, \mathbf{I})$: 标准高斯噪声

---

## 关键图表

### Figure 1: 系统概览

![Figure 1 - CoME Overview](https://arxiv.org/html/2605.18813v1/sections/image.png)

**说明**: CoME 的整体架构。三类记忆专家（STM 捕捉局部动态、LTM 存储情节历史、SLTM 保证几何一致性）通过[[Product of Contrastive Experts]]（PoCE）合成分数函数，驱动[[扩散模型]]生成与历史一致的未来帧。

### Figure 2: 上下文长度 vs. 重建质量

![Figure 2 - LPIPS vs Context Length](https://arxiv.org/html/2605.18813v1/sections/resub/context4.png)

**说明**: 在 Memory Maze 数据集上，LPIPS（越低越好）随上下文帧数增加线性下降，直至 480 帧未见饱和。说明 CoME 可持续从更长历史中获益，而无需二次计算代价。

### Figure 3: Memory Maze 流式评估

![Figure 3 - Long Streaming Evaluation](https://arxiv.org/html/2605.18813v1/sections/resub/long_gt2.png)

**说明**: 在 Memory Maze 上用 10 个 chunk（每 chunk 20 帧）进行增量记忆评估。CoME 通过持续 test-time finetuning 维护长达 200 帧的场景一致性，而基础模型随时间推移一致性明显退化。

### Figure 4: RealEstate10K 回忆质量

![Figure 4 - RealEstate10K Recall](https://arxiv.org/html/2605.18813v1/x1.png)

**说明**: 在 RealEstate10K 上生成 6 帧前向 rollout 后反转相机轨迹。CoME 能正确回忆初始帧外观（两个示例均成功），而无长期记忆的基础模型无法保持场景一致性，表明 LTM 有效存储了情节历史。

### Figure 5: 对比专家玩具示例

![Figure 5 - Toy Example](https://arxiv.org/html/2605.18813v1/sections/resub/toy_example.png)

**说明**: 以二维高斯混合分布演示 PoCE 的作用机制。朴素 PoE 将所有专家的分布相乘，会压缩方差并产生虚假模式；PoCE 通过对比系数选择性抑制各专家的非主导模式，保留真实分布的主峰形状。

### Table 1: Memory Maze 消融实验

| 方法 | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|------|---------|--------|--------|
| Base | 0.209 | 0.771 | 19.16 |
| + STM | 0.156 | 0.820 | 21.29 |
| + LTM | 0.171 | 0.805 | 19.98 |
| + SLTM | 0.150 | 0.833 | 20.65 |
| + STM + LTM | 0.114 | 0.862 | 22.32 |
| **CoME** | **0.097** | **0.892** | **23.07** |
| Sliding | 0.183 | 0.753 | 19.02 |
| SSM | 0.158 | 0.828 | 20.62 |
| Full | 0.113 | 0.859 | 22.78 |

**关键发现**: CoME（三专家全组合）达到最佳，LPIPS 相比 Base 下降 53.6%；Full-attention 基线（Full）需要 60× 计算量仍不及 CoME。

### Table 2: RECON 导航规划 Benchmark

| 方法 | ATE ↓ | RPE ↓ |
|------|-------|-------|
| GNM | 1.87 | 0.73 |
| NOMAD | 1.93 | 0.52 |
| NWM | 1.13 | 0.35 |
| CoME STM | 1.05 | 0.32 |
| CoME LTM | 1.10 | 0.32 |
| NWM + STM | 0.98 | 0.30 |
| NWM + LTM | 1.07 | 0.33 |
| **CoME** | **0.96** | **0.28** |

**关键发现**: CoME 在绝对轨迹误差（ATE）和相对位姿误差（RPE）上均优于所有 baseline，说明更强的长程一致性可直接提升规划精度。

### Table 3: RealEstate10K 场景回忆评估

| 方法 | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|------|---------|--------|--------|
| Base | 0.405 | 19.8 | 0.794 |
| HG-v | 0.414 | 19.2 | 0.764 |
| HG-t | 0.400 | 19.7 | 0.788 |
| **CoME** | **0.359** | **21.3** | **0.830** |

**关键发现**: CoME 在三指标均优，History-Guided（HG）变体甚至略弱于 Base，说明简单拼接历史帧反而有害，而 PoCE 的结构化组合才是关键。

### Table 4: 对比配方消融（LPIPS ↓）

| 配置 | 无对比 | 有对比 |
|------|--------|--------|
| Base | 0.203 | 0.200 |
| STM | 0.175 | 0.156 |
| LTM | 0.188 | 0.171 |
| SLTM | 0.178 | 0.150 |
| STM + LTM | 0.170 | 0.114 |
| All | 0.192 | **0.097** |

**关键发现**: 对比配方（$\alpha_i > 1$）在所有配置下均带来一致提升，且在三专家全组合时效果最为显著（0.192 → 0.097）。

### Table 5: LoRA Rank × 上下文帧数消融（LPIPS ↓）

| 上下文帧数 | Rank 8 | Rank 32 | Rank 64 | Rank 256 | Full FT |
|-----------|--------|---------|---------|----------|---------|
| 50 | 0.193 | 0.175 | 0.161 | 0.162 | 0.221 |
| 150 | 0.161 | 0.169 | 0.158 | 0.142 | 0.188 |
| 450 | 0.146 | 0.128 | 0.124 | 0.118 | 0.125 |

**关键发现**: Full FT（全参数微调）在短上下文时性能最差，说明 LoRA 的隐式正则化对防止灾难遗忘至关重要；随上下文增长，高 rank 优势扩大。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Memory Maze | — | 3D 迷宫导航，需长距回忆 | 主要消融 |
| RECON | 100 轨迹 | 户外室外导航，真实场景 | 规划评估 |
| RealEstate10K | — | 室内场景漫游视频 | 场景回忆评估 |
| DMLab-40K | 40K 轨迹 | 3D 环境导航 | 预训练 |
| Minecraft-200K | 200K 轨迹 | 开放世界游戏 | 预训练 |
| Memory Cards | 自建 | 记忆卡片离散物体回忆任务 | 回忆能力评估 |

### 实现细节

- **Backbone**: DiT-S（Diffusion Transformer Small）
- **条件注入**: Cross-attention（历史帧特征）
- **LoRA rank**: 8-64（主实验用 16-32）
- **LTM 在线更新步数**: 每帧 2-3 步梯度更新
- **训练步数**: 150k 步
- **STM 上下文长度**: 10-100 帧
- **LTM 上下文长度**: 100-1000 帧
- **对比系数 $\alpha_i$**: 各专家单独调参，$\alpha_i > 1$

### 定性结果

Memory Cards 任务中，CoME 实现 79.2% 的离散物体回忆率，vs. 基础模型 13%，说明 LTM 的 test-time finetuning 能有效记忆具体物体外观。RealEstate10K 的反向漫游测试中，CoME 在两个示例均能成功重建初始帧场景。

---

## 批判性思考

### 优点

1. **优雅的推理期组合**: 不需要重新训练，在采样时通过分数函数加权即可利用多个预训练模型
2. **计算高效**: Full-attention 基线需要 60× 计算量仍不及 CoME，实用性强
3. **理论严谨**: Proposition 1 从核密度估计角度证明了 PoCE 不扭曲主模式形状的理论保证
4. **模块化设计**: 各专家互相独立，便于增减组件

### 局限性

1. **超参数敏感**: 对比系数 $\alpha_i$ 需要对每个专家单独调参，实际部署复杂
2. **LTM 在线计算开销**: 每帧 2-3 步梯度更新在高帧率场景下有延迟
3. **专家数量固定**: 目前仅三类专家，扩展到更多异构专家的理论/实验尚未探索
4. **PoCE 理论假设**: Proposition 1 依赖核密度估计的不相交支撑假设，真实高维分布未必满足

### 潜在改进方向

1. 自动学习对比系数 $\alpha_i$（e.g., 通过元学习或在线适应）
2. 将 LTM 替换为外部记忆库（Key-Value Memory）以避免在线微调
3. 探索更多专家类型（如语义记忆、事件记忆）

### 可复现性评估

- [ ] 代码开源（项目页已创建，代码尚未发布）
- [ ] 预训练模型
- [x] 训练细节完整（论文附录包含超参数）
- [x] 数据集可获取（Memory Maze、RealEstate10K 为公开数据集）

---

## 关联笔记

### 基于

- [[扩散模型]]: CoME 以扩散模型为生成骨干
- [[Diffusion Transformer (DiT)]]: 使用 DiT-S 作为具体实现
- [[LoRA]]: LTM 的测试时微调基于 LoRA

### 对比

- [[NWM]]: Navigation World Model，导航 benchmark 的主要竞争基线
- [[SSM]]: State Space Model，高效长程建模的替代方案
- [[GNM]]: General Navigation Model，传统导航基线

### 方法相关

- [[Product of Experts]]: CoME 的理论基础，Product of Contrastive Experts 是其改进
- [[Classifier-Free Guidance]]: 分数函数合成与 CFG 机制高度相似
- [[测试时微调（Test-Time Finetuning）]]: LTM 的核心实现机制
- [[世界模型]]: CoME 所属的大类方法

### 数据集相关

- [[Memory Maze]]: 主要消融实验数据集
- [[RealEstate10K]]: 场景回忆评估数据集
- [[RECON]]: 导航规划 benchmark

---

## 速查卡片

> [!summary] Composition of Memory Experts (CoME)
> - **核心**: 将短期/长期/空间三类异构记忆专家通过对比乘积专家（PoCE）组合进扩散世界模型
> - **方法**: PoCE 用 $\alpha_i>1$ 对比系数抑制虚假模式；LTM 用 LoRA 测试时微调存储情节历史
> - **结果**: Memory Maze LPIPS 从 0.209→0.097（↓53.6%），优于需 60× 计算量的全注意力基线
> - **代码**: [wiqzard.github.io/composition-of-memory-experts](https://wiqzard.github.io/composition-of-memory-experts/)

---

*笔记创建时间: 2026-05-21*

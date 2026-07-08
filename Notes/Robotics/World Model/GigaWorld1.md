---
title: "GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"
method_name: "GigaWorld1"
authors: [GigaWorld Team, Angyuan Ma, Boyuan Wang, Bohan Li, Chaojun Ni, Guo Li, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jingyu Liu, Jiwen Lu, Qiuping Deng, Tingdong Yu, Xuancheng Xu, Xinyu Zhou, Xiuwei Xu, Xinze Chen, Xiaofeng Wang, Xiaoyu Tian, Yang Wang, Yifan Chang, Yukun Zhou, Yun Ye, Zhenyu Wu, Zhanqian Wu, Zheng Zhu]
year: 2026
venue: arXiv
tags: [world-model, robot-evaluation, benchmark, diffusion-transformer, policy-evaluation, autoregressive-generation, video-generation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.02642
created: 2026-07-08
---

# 论文笔记：GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | GigaWorld Team |
| 日期 | July 2026 |
| 项目主页 | [open-gigaai.github.io/giga-world-1](https://open-gigaai.github.io/giga-world-1/) |
| 对比基线 | [[Cosmos3]], [[Wan]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.02642) |

---

## 一句话总结

> 提出 WMBench 基准和 10 项系统性发现，指导构建 GigaWorld-1 世界模型，作为机器人策略评估器，以 14.9% 的提升超过 SOTA。

---

## 核心贡献

1. **WMBench 基准**: 包含 2,989 条真实机器人轨迹与仿真匹配轨迹、8 类操作任务、32.4 万+标注 rollout，提供四步评估协议。
2. **10 项设计发现**: 系统研究指出评估器质量由长时 action-faithful rollout 一致性主导，而非短期视觉真实性；Subject Consistency (ρ=0.88) 和 Perspectivity (ρ=0.86) 是最强预测指标。
3. **GigaWorld-1 世界模型**: 基于 Wan 骨干的自回归扩散变换器，集成分层历史记忆、统一控制注入和相对 RoPE，专为策略评估优化。

---

## 问题背景

### 要解决的问题

评估具身机器人基础模型仍是关键瓶颈——与语言模型不同，机器人策略需要代价高昂的物理环境测试，阻碍了大规模迭代。如何用[[世界模型|World Model]]作为代理评估器替代真实 rollout，是核心问题。

### 现有方法的局限

- 现有[[视频生成|视频世界模型]]缺乏系统性的评估指标对齐分析，不清楚哪些视频质量指标真正预测策略评估能力
- 背景一致性（ρ=−0.45）等常用指标实际与评估质量负相关——静态视频基线反而分数高
- 通用[[视频扩散模型|视频扩散骨干]]（SVD、LTX 等）在长时 rollout 中出现视角漂移、物体身份崩溃
- 缺乏标准化基准和人工标注数据集

### 本文的动机

作者认为世界模型评估器的质量取决于：长时动作忠实性、空间对齐的控制注入、以及通过持久记忆维护场景一致性——而非追求逐帧的视觉真实感。基于 WMBench 的系统性研究结论直接指导了 GigaWorld-1 的设计选择。

---

## 方法详解

### WMBench 评估基准

**构建流程（四步协议）**：
- Step 1: 收集真实世界策略 rollout（多视角）
- Step 2: 用严格数据划分训练世界模型
- Step 3: 闭环 rollout 生成（以真实数据中的策略动作驱动世界模型）
- Step 4: 指标评估 → 计算 WMES（World Model Evaluator Score）

**WMES 四级量表**：

| 分数 | 名称 | 描述 |
|------|------|------|
| 3 | 准确结果 + 高保真 | 正确预测任务结局，视觉高质量 |
| 2 | 准确结果 + 低保真 | 正确结局但中间帧有明显失真 |
| 1 | 错误结局 + 高保真 | 视频稳定但结局预测错误 |
| 0 | 错误结局 + 低保真 | 完全崩溃，结局和视觉均失败 |

**VLM 评估器**：[[LoRA]] 微调 Qwen3-VL-8B-Instruct，在 5000+ 视频上实现 87.80% 精确一致、99.16% 相邻一致（见 Table 1）。

### GigaWorld-1 整体架构

GigaWorld-1 是一个基于 [[Wan]] 视频骨干（1.3B 或 5B 参数）的[[自回归扩散变换器|Autoregressive Diffusion Transformer]]，通过三大原则专为机器人策略评估设计：

- 保留大规模预训练视频骨干的时空先验
- 通过几何对齐的接口注入机器人动作
- 跨迭代 rollout 维持状态一致性

**输入**：首帧图像锚点 + 历史 latent + 控制信号（EE 位姿图 + Ray 图）+ 任务描述
**骨干**：[[Wan]] DiT（冻结主干 + 可训练 [[LoRA]] 适配器）
**核心模块**：[[分层历史注意力|Hierarchical History Guidance Attention]] + [[相对旋转位置编码|Relative RoPE]] + [[SLERP]] 提示过渡
**输出**：预测未来观测帧序列（策略 rollout 仿真）
**总参数**：1.3B 或 5B

### 核心模块

#### 模块1: 自回归世界生成

**设计动机**：将固定长度的[[视频扩散模型]]转换为任意长度的自回归生成器，支持长时策略 rollout。

**具体实现**：
- 给定历史 latent $X_{hist} = \{x_1, x_2, \ldots, x_t\}$ 和未来窗口 $X_{future} = \{x_{t+1}, \ldots, x_{t+T_f}\}$
- 学习条件分布 $p(X_{future} | X_{hist}, C)$，$C$ 为控制条件
- 推理时每个生成窗口追加到历史缓冲区，驱动下一步预测
- 联合分布分解为：$p(X_{1:T} | C) = \prod_{k=1}^K p(X_k | H_{k-1}, C)$

#### 模块2: 统一控制注入（Unified Control Injection）

**设计动机**：动作控制必须通过空间对齐接口注入，Cross-attention 注入被语义 token 淹没，[[ControlNet]] 式空间注入效果更好，通道拼接最优（见 Finding 9）。

**具体实现**：
- **EE 位姿控制（头部相机）**：将未来末端执行器轨迹投影到图像平面，生成空间对齐的表示，编码臂的位置、朝向和夹爪状态
- **Ray Map 控制（腕部相机）**：存储每像素的光线原点和归一化方向（世界坐标系），提供显式相机几何表示
- 统一控制图：$C_t = \text{Concat}_W(C_t^{ee}, C_t^{ray})$
- 控制序列编码为 latent：$Z_{ctrl} = E(C)$

#### 模块3: 分层历史注入（Hierarchical History Injection）

**设计动机**：自回归生成中短期运动连续性和长期场景一致性须在有限上下文预算内同时满足（Finding 10）。

**具体实现**：
- 多尺度记忆：$H_t = \{H_t^{(L)}, H_t^{(M)}, H_t^{(S)}\}$
  - $H_t^{(S)}$：局部运动连续性和近期场景变化
  - $H_t^{(M)}$：中期动作演化和物体交互
  - $H_t^{(L)}$：全局场景布局、物体身份和环境状态
- 首帧锚点防止身份漂移：$\tilde{H}_t = \{x_{anchor}, H_t^{(L)}, H_t^{(M)}, H_t^{(S)}\}$
- 去噪网络条件：$\varepsilon_\theta = f_\theta(X_\tau, Z_{ctrl}^{(k)}, \tilde{H}_t, \tau)$

#### 模块4: 分层历史引导注意力（Hierarchical History Guidance Attention）

**设计动机**：历史上下文和噪声上下文统计特性根本不同，需差异化处理。

**具体实现**：
- 历史记忆：$X_{Hist} = \{X_{Anchor}, X_{Long}, X_{Mid}, X_{Short}\}$
- Self-attention 同时处理历史和噪声上下文：
  $X_{Self} = \text{Attention}([Q_{Noisy}, Q_{Hist}], [K_{Noisy}, K_{Hist}], [V_{Noisy}, V_{Hist}])$
- Cross-attention 仅对当前噪声窗口注入任务语义（历史已含语义信息）：
  $X_{Cross} = \text{Attention}(Q_{Noisy}, K_{Task}, V_{Task})$
- 最终：$X = X_{Self} + X_{Cross}$

#### 模块5: 相对旋转位置编码（Relative RoPE）

**设计动机**：长时自回归生成中，绝对位置编码导致模型推理时遇到训练期未见的位置范围，引发时间漂移和重复运动。

**具体实现**：
- 每步自回归中重建局部时间坐标系：$P_{hist} = \{0, 1, \ldots, T_h-1\}$，$P_{future} = \{T_h, \ldots, T_h+T_f-1\}$
- 对局部位置 $p$ 处的 token 应用旋转嵌入：$Q'_p = R(p)Q_p$，$K'_p = R(p)K_p$
- 每步重新初始化，不依赖绝对视频时间戳

---

## 关键公式

### 公式1: [[世界模型|世界模型条件预测]]

$$
\hat{o}_{t+1:t+H} \sim M_\theta(\cdot | o_{\leq t}, s_{\leq t}, a_{\leq t}, l)
$$

**含义**：世界模型在给定历史观测 $o$、状态 $s$、动作 $a$ 和语言指令 $l$ 的条件下，预测未来 $H$ 步的观测序列。

**符号说明**：
- $o_{\leq t}$：历史观测序列
- $s_{\leq t}$：历史状态序列
- $a_{\leq t}$：历史动作序列
- $l$：语言任务描述
- $H$：预测视野长度

### 公式2: [[WMES|策略评估相关性指标]]

$$
\rho = \text{Corr}(S^{real}(\pi), S^{wm}(\pi))
$$

**含义**：世界模型评估器质量的核心指标，衡量真实环境策略排名与世界模型仿真排名之间的 Pearson/Spearman 相关性。

**符号说明**：
- $S^{real}(\pi)$：策略 $\pi$ 在真实环境中的得分
- $S^{wm}(\pi)$：策略 $\pi$ 在世界模型仿真中的得分

### 公式3: [[自回归生成|自回归世界生成分解]]

$$
p(X_{1:T} | C) = \prod_{k=1}^K p(X_k | H_{k-1}, C)
$$

**含义**：将任意长度的视频生成分解为逐窗口的条件自回归乘积，每步以历史记忆 $H_{k-1}$ 和控制条件 $C$ 为条件。

**符号说明**：
- $X_k$：第 $k$ 个生成窗口的 latent
- $H_{k-1}$：第 $k$ 步的历史记忆状态
- $C$：控制条件（动作图 + 任务描述）
- $K$：总生成窗口数

### 公式4: [[扩散去噪|训练去噪目标]]

$$
X_\tau = \alpha_\tau X_{future} + \sigma_\tau \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

$$
\mathcal{L}_{AR} = \mathbb{E}\left[\|\varepsilon - \varepsilon_\theta(X_\tau, H_t, C, \tau)\|_2^2\right]
$$

**含义**：训练时对未来窗口施加扩散噪声，网络学习在历史记忆和控制条件下去噪。

**符号说明**：
- $\alpha_\tau, \sigma_\tau$：噪声调度参数
- $\tau$：连续时间步
- $H_t$：历史记忆状态
- $\varepsilon_\theta$：去噪网络预测

### 公式5: [[流匹配|Flow Matching 预训练目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}\left[\|u_t - u_\theta(x_t, t)\|_2^2\right]
$$

**含义**：Stage 1 预训练使用流匹配目标，学习从噪声到数据的速度场，无自回归因果约束。

**符号说明**：
- $u_t$：目标速度场
- $u_\theta$：网络预测速度
- $x_t$：连续时间 $t$ 处的 noisy latent

### 公式6: [[SLERP|球面线性插值提示过渡]]

$$
\theta = \arccos\left(\frac{e_1^T e_2}{\|e_1\| \|e_2\|}\right)
$$

$$
\text{SLERP}(e_1, e_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta} e_1 + \frac{\sin(t\theta)}{\sin\theta} e_2
$$

**含义**：在两个文本 embedding 之间进行球面线性插值，实现长视频生成中语义状态的平滑过渡。

**符号说明**：
- $e_1, e_2 \in \mathbb{R}^d$：两个文本 embedding
- $\theta$：二者之间的夹角
- $t \in [0,1]$：插值系数

### 公式7: [[DMD2|蒸馏训练目标]]

$$
\mathcal{L}_{DMD2} = \lambda_{dm}\mathcal{L}_{distill} + \lambda_{score}\mathcal{L}_{score} + \lambda_{GAN}\mathcal{L}_{GAN}
$$

**含义**：Stage 4 蒸馏阶段的综合损失，结合教师分布匹配、得分一致性和对抗监督，使学生模型用更少去噪步数达到教师质量。

**符号说明**：
- $\lambda_{dm}, \lambda_{score}, \lambda_{GAN}$：各分量权重
- $\mathcal{L}_{distill}$：分布匹配损失
- $\mathcal{L}_{score}$：得分一致性损失
- $\mathcal{L}_{GAN}$：对抗损失

### 公式8: [[相对旋转位置编码|Relative RoPE]]

$$
Q'_p = R(p)Q_p, \quad K'_p = R(p)K_p
$$

**含义**：对每个局部时间位置 $p$ 的 Query 和 Key 应用旋转嵌入，每步自回归重新初始化局部坐标系。

**符号说明**：
- $R(p)$：位置 $p$ 处的旋转矩阵算子
- $p \in P = \{0, 1, \ldots, T_h+T_f-1\}$：局部时间位置（每步重置）

---

## 关键图表

### Figure 1: 整体研究概览

![Figure 1](https://arxiv.org/html/2607.02642/2607.02642v1/x1.png)

**说明**：论文分析了 32.4 万条世界模型仿真 rollout、7 个视频世界模型、4 种动作表示范式，提炼出 10 项关键设计发现，指导 GigaWorld-1 的构建。

### Figure 2: 世界模型作为策略评估器的框架

![Figure 2](https://arxiv.org/html/2607.02642/2607.02642v1/x2.png)

**说明**：世界模型迭代接收策略动作并预测未来观测，通过比较仿真 rollout 质量（WMES）评估策略优劣，替代代价高昂的真实环境测试。

### Figure 3: WMBench 评估流水线

![Figure 3](https://arxiv.org/html/2607.02642/2607.02642v1/x3.png)

**说明**：四步协议——(1) 收集真实策略 rollout，(2) 用严格数据划分训练世界模型，(3) 闭环 rollout 生成，(4) 指标评估。确保评估的可重复性和公平性。

### Figure 4: 指标组与 WMES 的相关性

![Figure 4](https://arxiv.org/html/2607.02642/2607.02642v1/x4.png)

**说明**：视觉保真度（ρ=0.78）和几何（ρ=0.71）主导评估器质量；语义对齐（ρ=0.11）单独相关性弱；背景一致性（ρ=−0.45）为负相关的"欺骗性指标"。

### Figure 5: 所有指标的 Pearson 相关矩阵

![Figure 5](https://arxiv.org/html/2607.02642/2607.02642v1/x5.png)

**说明**：Subject Consistency（ρ=0.88）和 Perspectivity（ρ=0.86）是最强的 WMES 预测指标；Instruction Following（ρ=0.84）紧随其后。完整热力图揭示指标间的相互依赖关系。

### Figure 6: VLM 辅助 Rollout 评估器

![Figure 6](https://arxiv.org/html/2607.02642/2607.02642v1/x6.png)

**说明**：给定三视角 rollout 视频和任务特定评估提示，[[LoRA]] 微调的 Qwen3-VL 评估器预测 WMES 分数。Token 类型感知的损失权重（总分 token 权重 8.0，格式 token 1.0，证据 token 最低 0.05）防止标准语言建模目标低估结果预测。

### Figure 7: 数据构建流水线

![Figure 7](https://arxiv.org/html/2607.02642/2607.02642v1/x7.png)

**说明**：多源数据经过质量过滤、运动过滤、分布均衡和自动标注（语义掩码、深度图、快慢双粒度描述）后纳入训练语料，总计约 12,980 小时。

### Figure 8: GigaWorld-1 整体架构

![Figure 8](https://arxiv.org/html/2607.02642/2607.02642v1/x8.png)

**说明**：自回归扩散变换器架构，以 [[LoRA]] 参数高效适配。左侧为分层历史记忆注入，右侧为统一控制注入（EE 位姿图 + Ray 图通道拼接），中间为带 [[相对旋转位置编码|Relative RoPE]] 的 DiT 骨干。

### Figure 9: 球面线性插值提示过渡

![Figure 9](https://arxiv.org/html/2607.02642/2607.02642v1/x9.png)

**说明**：[[SLERP]] 在文本 embedding 空间中平滑插值，使世界模型在长视频生成中渐进过渡语义状态，避免直接切换导致的外观突变。

### Figure 10: GigaWorld-1 四阶段训练流程

![Figure 10](https://arxiv.org/html/2607.02642/images/Training_Pipeline.png)

**说明**：Stage 1（流匹配预训练）→ Stage 2（自回归世界建模）→ Stage 3（场景 LoRA 适配，可选）→ Stage 4（少步蒸馏）。每阶段在前阶段基础上专门化，最终实现快速高质量 rollout。

### Table 1: VLM 评估器与人工标注一致性

| 指标 | 数值 |
|------|------|
| Exact Accuracy | 0.8780 |
| Adjacent Accuracy | 0.9916 |
| Large Error Rate | 0.0084 |
| MAE | 0.1304 |
| RMSE | 0.3836 |
| Quadratic Weighted Kappa | 0.7349 |
| Spearman Correlation | 0.7574 |
| Kendall τb | 0.7507 |
| Weighted F1 | 0.8744 |

**说明**：VLM 评估器在 5000+ 视频上达到 87.80% 精确一致率，99.16% 相邻一致率，证明其可靠替代人工标注。

### Table 2: 训练数据配比消融

| Data Recipe | Aesthetic | Image Quality | JEPA Sim. | Photo Cons. | Semantic | Subject | Trajectory | Average |
|-------------|-----------|---------------|-----------|-------------|----------|---------|------------|---------|
| GigaData | 0.34355 | 0.6904 | 0.8141 | 0.3187 | 0.8896 | 0.7212 | 0.2566 | 0.5654 |
| GigaData + AgiBot | 0.3802 | 0.7042 | 0.5715 | 0.6218 | 0.8710 | 0.8613 | 0.1482 | 0.5940 |
| Delta | +0.037 | +0.014 | **−0.243** | +0.303 | −0.019 | +0.140 | −0.108 | +0.029 |
| GigaData + PhysData | 0.3388 | 0.6974 | 0.7678 | 0.6261 | 0.8798 | 0.7409 | 0.2497 | **0.6144** |
| Delta | −0.005 | +0.007 | −0.046 | +0.307 | −0.010 | +0.020 | −0.007 | **+0.049** |

**关键发现**：GigaData + PhysData（物理视频）提供最佳整体权衡（+0.049），Photometric Consistency 提升最大（+0.307）；AgiBot（机器人专项数据）提升较小且引入 JEPA Similarity 大幅下降（−0.243）。

### Table 3: 控制注入方式消融

| Method | Control Type | Traj. Acc. | Dynamic | Smooth | Flow | Subject | Photo. |
|--------|--------------|-----------|---------|--------|------|---------|--------|
| Wan 2.1 1.3B I2V | None | 0.1576 | 0.2429 | 0.4997 | 0.0971 | 0.5568 | 0.2185 |
| Wan 2.1 1.3B Control | Cross-attention | 0.1620 | 0.1049 | 0.4525 | 0.0624 | 0.3573 | 0.1853 |
| Wan 2.1 1.3B Control | ControlNet | 0.2566 | 0.3083 | 0.5197 | 0.1412 | 0.7212 | 0.3187 |
| Wan 2.1 1.3B Control | **Channel concat** | **0.3528** | **0.3566** | **0.5747** | **0.2179** | **0.8600** | **0.3206** |

**关键发现**：通道拼接（Channel concat）在所有指标上全面领先，Trajectory Accuracy 达 0.3528（vs Cross-attention 的 0.1620），验证 Finding 9。

### Table 4: 长时 Rollout 质量（0-40 秒）

| Model | 0–8s PSNR | 8–16s PSNR | 16–24s PSNR | 24–32s PSNR | 32–40s PSNR |
|-------|-----------|------------|-------------|-------------|-------------|
| SVD | 14.05 | 8.71 | 7.24 | 6.82 | 6.88 |
| Cosmos2.5 | 13.65 | 13.75 | 13.40 | 13.13 | 12.83 |
| LTX-Video | 13.38 | 13.67 | 12.98 | 12.82 | 12.74 |
| Wan 2.2 | 14.35 | 11.57 | 10.97 | 10.53 | 10.09 |
| Wan 2.1 | 14.46 | 13.63 | 13.63 | 13.49 | 13.37 |
| **Wan 2.1+Mem.** | **19.82** | **17.01** | **17.52** | **17.51** | **17.41** |

**关键发现**：添加记忆机制后 PSNR 在所有时间段均大幅提升（0-8s 从 14.46 → 19.82），证明 Finding 10：持久记忆对长时 rollout 至关重要。SVD 后期严重退化（32-40s PSNR 仅 6.88）。

### Table 5: GigaWorld-1 设计图谱

| 组件 | 设计选择 | 备注 |
|------|----------|------|
| 骨干 | Wan-[1.3B / 5B] | 竞争性表现 + 成熟生态 |
| 训练数据 | 物理视频 + 开源机器人 + 自我中心 + Giga 采集 | 广泛物理先验 + 具身多样性 |
| 数据筛选 | 质量 + 运动 + 分布过滤 | 去除噪声、静态、错位样本 |
| 结构化监督 | 语义掩码 + 深度 + 快慢描述 | 改善几何和任务接地 |
| 动作接口 | 显式像素对齐表示 | EE 位姿图 + Ray 图 |
| 长时模块 | 记忆增强 rollout | 首帧锚点 + 分层历史 |
| 时间编码 | 相对 RoPE | 减少长时自回归中的位置漂移 |
| 训练策略 | 渐进式多阶段训练 | 基础预训练 → AR 学习 → 可选场景 LoRA → 蒸馏 |

### Table 6: 训练语料概览

| 类别 | 代表来源 | 视角 | 机器人类型 | 时长(h) | 模态 |
|------|----------|------|-----------|---------|------|
| 物理数据 | 互联网/物理视频 | 单视角 | N/A | ~1298 | RGB Video |
| 开源机器人数据 | Open X, AgiBot | 单/多视角 | 单臂、双臂、人形 | ~5377 | Robot Demo |
| 以人为中心数据 | EgoDex, SynData | 自我中心 | 人手 | ~2411 | RGB + Hand Pose |
| Giga 采集数据 | Giga Humanoid, Dual-arm | 单/多视角 | 人形、双臂 | ~3894 | Robot Demo |
| **总计** | — | — | — | **~12,980** | Multi-modal |

### Table 7: 非蒸馏阶段超参数

| 配置 | Stage 1 | Stage 2 |
|------|---------|---------|
| Global Batch Size | 32 | 32 |
| 优化器 | AdamW | AdamW |
| 学习率 | 5×10⁻⁵ | 1×10⁻⁴ |
| LR 调度 | Cosine | Cosine |
| 训练步数 | 13k | 36k |
| 梯度裁剪 | 1.0 | 1.0 |
| LoRA Rank | 128 | 256 |
| LoRA Alpha | 128 | 256 |
| 数值精度 | BFloat16 | BFloat16 |
| GPU | 32× NVIDIA H20 | 32× NVIDIA H20 |

### Table 8: 蒸馏阶段超参数

| 配置 | ODE | DMD2 |
|------|-----|------|
| Global Batch Size | 32 | 32 |
| 学习率 (Gθ) | 2.0×10⁻⁶ | 2.0×10⁻⁶ |
| 学习率 (pfake) | — | 4.0×10⁻⁷ |
| LoRA Rank (Gθ) | 256 | 256 |
| LoRA Alpha (Gθ) | 256 | 256 |
| GAN 头层数 | — | 5, 15, 25, 35, 39 |
| TTUR | — | 5 |
| EMA 衰减 | 0.99 | 0.99 |
| 训练步数 | 3759 | 2250 |
| GPU | 32× NVIDIA H20 | 32× NVIDIA H20 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| WMBench | 2,989 条轨迹，32.4 万+ rollout | 8 类操作任务，真实+仿真配对 | 评估基准 |
| Open X-Embodiment | ~5377h（含 AgiBot） | 多具身多任务 | 预训练 |
| EgoDex / SynData | ~2411h | 以人为中心，含手部位姿 | 预训练 |
| 物理/互联网视频 | ~1298h | 通用物理先验 | 预训练 |
| Giga 采集数据 | ~3894h | 人形/双臂机器人 | 预训练 |

### 实现细节

- **骨干**: Wan 2.1（1.3B 或 5B 参数），迁移预训练权重
- **优化器**: AdamW，学习率 5×10⁻⁵（Stage 1）/ 1×10⁻⁴（Stage 2）
- **LoRA Rank**: 128（Stage 1）→ 256（Stage 2/蒸馏）
- **训练步数**: 13k（Stage 1）+ 36k（Stage 2）+ 3759（ODE）+ 2250（DMD2）
- **硬件**: 32× NVIDIA H20，BFloat16 精度
- **效率优化**: [[SageAttention]]（注意力加速）+ TinyVAE Decoding（解码加速）+ 自定义归一化和 RoPE 核

### 主要结果

GigaWorld-1 在核心评估对齐指标上比竞争性 SOTA 基线提升 **14.9%**。WMBench 上，Wan 2.1+Memory 在长时 PSNR 上全面领先所有对比模型（0-8s PSNR: 19.82 vs Wan 2.1 的 14.46）。VLM 评估器达到 87.80% 精确一致、99.16% 相邻一致。

---

## 10 项关键发现（Findings）

| 编号 | 发现 | 核心数据 |
|------|------|----------|
| F1 | 视觉/几何保真度主导评估器质量 | Subject Cons. ρ=0.88, Perspectivity ρ=0.86 |
| F2 | 退化指标误导排名（背景一致性 ρ=−0.45） | 静态视频基线高分但不预测策略效果 |
| F3 | 评估器质量需在长时 rollout（40s）下评估 | 短时强模型长时不一定强 |
| F4 | 结果中心监督对 VLM 评估至关重要 | 总分 token 权重 8.0，证据 token 0.05 |
| F5 | VLM 评估器达近完美人工一致性 | 精确 87.80%，相邻 99.16% |
| F6 | 可迁移物理先验比规模更重要 | Cosmos2.5（专项预训）> Wan 2.2 5B > LTX 22B |
| F7 | 广泛物理视频提供最佳整体权衡 | PhysData +0.049 vs AgiBot +0.029 |
| F8 | 机器人专项数据主要改善具身保真度但引入权衡 | AgiBot JEPA Sim. −0.243 |
| F9 | 动作控制须通过空间对齐接口注入 | Channel concat Traj.Acc.=0.3528 vs Cross-attn 0.1620 |
| F10 | 可靠评估器需要持久记忆支持长时 rollout | Wan+Mem PSNR 19.82 vs Wan 14.46（0-8s） |

---

## 批判性思考

### 优点
1. **系统性 roadmap**：10 项发现形成完整的设计指导，从基准到模型架构逻辑自洽
2. **大规模验证**：32.4 万+ 标注 rollout + 5000+ VLM 评估验证，数据规模充分
3. **开放性**：代码、模型权重、数据集和工具包全部开源，有利于社区跟进
4. **效率导向**：SageAttention + TinyVAE + 自定义核的系统优化，实用价值高

### 局限性
1. **场景覆盖局限**：主要评估桌面操作（8 类任务），足式运动、导航等场景未覆盖
2. **单视角限制**：部分发现基于头部/腕部双摄配置，其他多摄或不同安装方式的泛化性未知
3. **世界模型替代完整性**：仍需真实 rollout 数据构建 WMBench 和训练评估器，并非完全"零真实环境"

### 潜在改进方向
1. 扩展 WMBench 到足式运动、导航等更多任务类别
2. 探索不依赖真实 rollout 标注的自监督评估对齐
3. 进一步压缩蒸馏步数以降低推理延迟

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型（1.3B 和 5B checkpoints）
- [x] 训练细节完整（详细超参数表格）
- [x] 数据集可获取（WMBench + 部分训练数据）

---

## 关联笔记

### 基于
- [[Wan]]: 视频生成骨干，GigaWorld-1 的基础模型
- [[LoRA]]: 参数高效微调方法，所有训练阶段的核心技术

### 对比
- [[Cosmos3]]: Cosmos-Predict 2.5，最强基线（AVG 0.6123），机器人/自动驾驶预训练提供有用物理先验
- [[WBench]]: 不同 benchmark（多轮交互视频世界模型评估），两者均关注世界模型评估但场景不同

### 方法相关
- [[自回归扩散变换器|Autoregressive Diffusion Transformer]]: 核心生成范式
- [[流匹配|Flow Matching]]: Stage 1 预训练目标
- [[DMD2]]: Stage 4 蒸馏方法
- [[SageAttention]]: 注意力加速工具

### 数据相关
- [[Open X-Embodiment]]: 开源机器人数据主要来源之一
- [[AgiBot]]: 机器人专项数据集（Finding 8 分析对象）

---

## 速查卡片

> [!summary] GigaWorld-1
> - **核心**: 用世界模型替代真实环境做机器人策略评估，14.9% 超越 SOTA
> - **方法**: WMBench 基准 + 10 项设计发现 + 基于 Wan 骨干的自回归扩散变换器（分层历史记忆 + 通道拼接控制注入 + 相对 RoPE）
> - **结果**: VLM 评估器 87.80% 人工一致，Wan+Mem 长时 PSNR 19.82（vs 基线 14.46）
> - **代码**: [open-gigaai.github.io/giga-world-1](https://open-gigaai.github.io/giga-world-1/)

---

*笔记创建时间: 2026-07-08*

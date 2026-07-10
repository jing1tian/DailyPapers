---
title: "Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence"
method_name: "LingBot-Video"
authors: [Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang, Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, Ka Leong Cheng]
year: 2026
venue: arXiv
tags: [video-generation, mixture-of-experts, embodied-ai, world-model, video-diffusion, robot-manipulation, physical-plausibility]
zotero_collection: 3-Robotics/World-Model
image_source: local
created: 2026-07-10
---

# 论文笔记：Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RobbyAnt（多人团队） |
| 日期 | July 2026 |
| 项目主页 | [technology.robbyant.com/lingbot-video](https://technology.robbyant.com/lingbot-video) |
| 对比基线 | [[Cosmos3]]、[[Wan 2.2 A14B]]、[[HunyuanVideo]]、[[LongCat-Video]]、[[DreamDojo]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.07675) / [Code](https://github.com/robbyant/lingbot-video) / [Checkpoints](https://huggingface.co/robbyant/lingbot-video) |

---

## 一句话总结

> LingBot-Video 是首个大规模开源 [[MoE|Mixture-of-Experts]] 视频基础模型，专为具身智能设计，通过 MoE 架构、机器人增强数据和多维物理奖励，在保持推理效率的同时大幅提升物理真实性和动作理解能力。

---

## 核心贡献

1. **MoE 视频扩散框架**: 首次将 [[Mixture-of-Experts]] 大规模引入视频 [[DiT]]，在 120B 参数规模下验证可预测的 scaling law，同等计算量下大幅超越 dense 基线。
2. **机器人增强数据基础设施**: 构建 Data Profiling Engine + World-Knowledge Topological Graph + Dense Structured Captioning 三位一体的数据流水线，注入超 70,000 小时具身数据。
3. **多维奖励强化学习**: 设计六维专用奖励模型（视觉质量、文视对齐、动态程度、运动连贯性、人体动作一致性、物理合理性）+ 真实视频对比偏好对齐（RealNFT），超越传统单一标量奖励。

---

## 问题背景

### 要解决的问题

现有视频生成模型（扩散模型/自回归模型）在具身智能场景中存在严重的 domain mismatch：模型优化视觉保真度和创意性，而非物理正确性和计算效率，无法作为可靠的机器人世界模拟器。

### 现有方法的局限

三个维度的短板共同存在：
- **架构**: 大多数视频扩散 Transformer 使用 dense 计算，推理成本高、扩展性差
- **数据**: 训练语料以互联网视频为主，缺乏机器人具身先验和精确交互动态
- **训练目标**: 对齐策略仅优化美学质量和文视对应，不包含物理可行性、任务完成度或长时程奖励信号

### 本文的动机

[[MoE|Mixture-of-Experts]] 在 LLM 中已证明"参数容量-计算量解耦"的有效性：在固定 FLOPs 下大幅扩展模型总参数。视频生成需要捕捉复杂的物理规律（流体、刚体、材质），天然需要大容量模型，而 MoE 可在不膨胀推理开销的前提下实现这一目标。

---

## 方法详解

### 模型架构概览

LingBot-Video 采用 **级联设计**：基础生成器（480p）+ 精炼器（1080p），共享 VAE 和条件编码器。

- **条件编码**: [[Qwen3-VL]] (Qwen3-VL-4B) 提取多模态条件特征
- **视觉压缩**: Wan2.1-VAE 进行潜在空间压缩
- **主干**: Task-Unified Single-Stream [[DiT|Diffusion Transformer]] + [[Sparse MoE|Sparse Mixture-of-Experts]]
- **上采样**: Cascaded Refiner 从 480p 升至 1080p
- **总参数规模**: 最大 120B 总参数（11B 激活参数）

### 核心模块 1：Task-Unified Single-Stream DiT

**统一输入公式**：将视觉 patch token 和条件特征投影到同一隐维度后沿 sequence 维度拼接，在单一框架内统一处理 T2I、T2V、TI2V 三类任务，将图像视为单帧（T=1）的特殊情况。

**关键设计点**：
- 使用 [[Multi-Modal 3D RoPE|3D MM-RoPE]] 分离条件和视觉 token 坐标：条件 token 用时序坐标 $(i, 0, 0)$，视觉 patch 用 $(L+1+f, h, w)$，保持完全单流注意力
- 采用 [[QK-Norm]]（per-head [[RMSNorm]]）稳定深层 Transformer 的注意力
- 使用 [[AdaLN-Single]] 调制：在 Transformer 块前计算一次共享时步调制，每层只添加可训练的层专属偏移表，减少调制开销

**单流 vs. 双流**：单流设计最大化参数复用，避免双流每层内的张量拼接/分割，在分布式序列并行场景下减少通信开销，提升 MFU。

### Figure 2: 单流扩散 Transformer + Sparse MoE 架构

![[LingBot-Video_fig2_architecture.png]]

**说明**: 统一输入经过 $N$ 个 Transformer 块处理，时步 embedding 调制注意力和 Sparse MoE 分支；注意力分支使用 QK-Norm 和 [[Multi-Modal 3D RoPE]]，MoE 分支结合常驻共享专家和 top-$K_r$ 路由专家，最终预测速度场。

### 核心模块 2：Sparse Mixture-of-Experts

**设计动机**：视频生成面临严重的子任务干扰——单一 FFN 参数集必须同时建模空间纹理 vs. 时序运动、不同噪声区间（高噪全局布局 vs. 低噪细节重建）、T2I/T2V/TI2V 等多种任务。[[Sparse MoE]] 通过 token 路由到专门化子网络来缓解这一问题。

**架构设计**（借鉴 [[DeepSeekMoE]]）：
- 保留 FFN 残差分支结构，仅将 dense FFN 替换为 token-choice sparse MoE 层
- 细粒度专家分割（每个专家 FFN 维度比 dense 对应物小）
- 共享专家（$N_s$ 个）+ 路由专家（$N_r$ 个）并行设计

**最终选定配置**：MoE 13B-A1.4B-E128（13B 总参数，1.4B 激活参数，128 路由专家，每 token 激活 $K_r=8$ 个）

**负载均衡**：无辅助损失的在线偏置校正（Online Bias Correction）+ 序列级辅助平衡损失（Sequence-Wise Auxiliary Loss），避免批级别统计掩盖视频内部路由不均衡问题。

### 核心模块 3：级联精炼器（Cascaded Refiner）

精炼器学习从退化低分辨率条件 $x_{lr}$ 到高清目标 $x_0$ 的条件 [[Rectified Flow]]，而非从纯噪声去噪：

- 训练时：对目标视频下采样并添加高斯模糊和压缩等合成降质，像素空间上采样后用 VAE 编码得到条件 latent
- 推理时：将 480p 基础生成结果上采样后编码，在 $t = \tau_{inf}$ 处注入噪声，仅运行反向 ODE 的后段 $[0, \tau_{inf}]$

这一设计保留基础生成的全局语义和运动，将精炼器容量专用于恢复高频细节。

### Figure 8: 精炼器对比

![[LingBot-Video_fig8_refiner.png]]

**说明**: 480p 基础生成 vs. 1080p 精炼后对比。左侧：舞者面部细节显著增强；右侧：机器人场景中高频纹理和可读 OCR 文字得以恢复。

---

## 关键公式

### 公式1：[[Sparse MoE|MoE 分支输出]]

$$
m(u_t) = \sum_{i=1}^{N_s} E_i^{(s)}(u_t) + \sum_{j \in \mathcal{R}_b(u_t)} g_{t,j} E_j^{(r)}(u_t)
$$

**含义**: token $t$ 的 MoE 分支输出由所有共享专家输出之和加上选中路由专家的加权输出组成。

**符号说明**:
- $u_t$: token $t$ 经调制后的 FFN 输入
- $N_s$: 共享专家数量
- $E_i^{(s)}(\cdot)$, $E_j^{(r)}(\cdot)$: 第 $i$ 个共享专家 / 第 $j$ 个路由专家函数（SwiGLU MLP）
- $\mathcal{R}_b(u_t)$: 为 token $t$ 选中的路由专家集合
- $g_{t,j}$: token $t$ 对专家 $j$ 的最终门控权重

### 公式2：[[Sigmoid Router|Sigmoid 路由亲和度]]

$$
\alpha_{t,j} = \text{Sigmoid}\left(u_t^\top r_j\right)
$$

**含义**: 计算 token $t$ 与路由专家 $j$ 之间的亲和度得分，使用 Sigmoid 而非 Softmax 使各专家独立评分（无归一化竞争）。

**符号说明**:
- $\alpha_{t,j}$: token $t$ 与专家 $j$ 的路由亲和度
- $r_j \in \mathbb{R}^d$: 第 $j$ 个路由专家的可学习路由 embedding

### 公式3：[[Load Balancing|在线偏置校正]]

$$
b_j \leftarrow b_j - \eta \cdot \text{sign}(n_j - \bar{n})
$$

$$
b_j \leftarrow b_j - \frac{1}{N_r} \sum_{k=1}^{N_r} b_k \quad \text{（偏置中心化）}
$$

**含义**: 每个优化步骤后，根据专家负载偏差的符号更新偏置，配合均值中心化保持路由器的输入依赖性和稳定性，无需梯度传播（auxiliary-loss-free）。

**符号说明**:
- $b_j$: 专家 $j$ 的动态修正偏置（仅用于 top-K 选择，不用于最终门控值）
- $n_j$: 全局累积的专家 $j$ 的有效 token 分配数
- $\bar{n}$: 平均每专家负载
- $\eta$: 偏置更新学习率

### 公式4：[[Sequence-Wise Auxiliary Loss|序列级辅助平衡损失]]

$$
\mathcal{L}_{\text{seq}} = \frac{1}{S} \sum_{s=1}^{S} \sum_{j=1}^{N_r} f_j^{(s)} P_j^{(s)}
$$

$$
P_j^{(s)} = \frac{1}{T_s} \sum_{t=1}^{T_s} p_{t,j}, \quad p_{t,j} = \frac{\alpha_{t,j}}{\sum_{k=1}^{N_r} \alpha_{t,k}}
$$

$$
f_j^{(s)} = \frac{N_r}{K_r T_s} c_j^{(s)}, \quad c_j^{(s)} = \sum_{t=1}^{T_s} \mathbf{1}\left[\alpha_{t,j} \in \text{Top-}K_r\right]
$$

**含义**: 在每个打包视频序列内部而非批次级别衡量专家负载均衡，避免批统计掩盖单视频内的路由不均衡问题。

**符号说明**:
- $S$: 打包批次中的序列数
- $T_s$: 第 $s$ 个序列的 token 长度
- $P_j^{(s)}$: 序列 $s$ 中专家 $j$ 的平均归一化路由概率
- $f_j^{(s)}$: 序列 $s$ 中专家 $j$ 的归一化分配频率（从计算图 detach）

$$
\mathcal{L}_{\text{aux}} = \lambda_{\text{aux}} \mathcal{L}_{\text{seq}}, \quad \mathcal{L} = \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{aux}}
$$

### 公式5：[[Cascaded Refiner|级联精炼器条件轨迹]]

$$
x_t = \left(1 - \frac{t}{\tau}\right) x_0 + \frac{t}{\tau} x_\tau, \quad v^\star_{\text{ref}} = \frac{x_\tau - x_0}{\tau}
$$

其中 $x_\tau = (1-\tau)x_{lr} + \tau\epsilon$，$\tau \sim \text{Uniform}(0.85, 0.95)$

**含义**: 精炼器不从纯噪声去噪，而是学习从退化条件 $x_{lr}$ 出发的条件 rectified flow，将去噪限制在 $t \in [0, \tau]$ 的噪声区间内。

### 公式6：[[CPS|Coefficients-Preserving Sampling]]

$$
\hat{x}_0 = x_{t_i} - t_i \hat{v}_\theta, \quad \hat{\epsilon} = x_{t_i} + (1-t_i)\hat{v}_\theta
$$

$$
x_{t_{i+1}} = \mu_\theta + s_i \epsilon, \quad \mu_\theta = (1-t_{i+1})\hat{x}_0 + \sqrt{t_{i+1}^2 - s_i^2} \hat{\epsilon}
$$

其中 $s_i = t_{i+1} \sin(\eta\pi/2)$，$\eta = 0.7$

**含义**: 在 GRPO 强化学习的随机步骤中，CPS 注入噪声后保持边际分布完全匹配确定性采样路径（而非 SDE 转化会引入过量噪声），确保奖励始终在干净视频上评估。

**符号说明**:
- $\hat{v}_\theta$: 速度预测网络输出
- $s_i$: 注入噪声量
- $\eta$: 探索强度系数

### 公式7：[[GRPO|时步平衡梯度重加权]]

$$
\kappa_k = 2s_k \left\|\frac{\partial\mu_\theta}{\partial\hat{v}_\theta}\right\|, \quad \lambda_k = \frac{\kappa_k^{-1}}{\frac{1}{N}\sum_{j=0}^{N-1} \kappa_j^{-1}}
$$

**含义**: 用转移增益的倒数对每个时步的策略梯度重加权，均衡不同时步对训练的贡献，防止少数时步主导更新。

### 公式8：[[Multi-Reward Advantage|多奖励优势归一化]]

$$
\hat{A}^{(i)} = \sum_r w_r \frac{R_r(x_0^{(i)}, c) - \mu_r}{\sigma_r + \delta}
$$

**含义**: 六个奖励信号各自独立归一化后加权融合为单一优势估计，避免不同量纲和方差的奖励互相干扰。

**符号说明**:
- $R_r$: 第 $r$ 个奖励函数
- $w_r$: 奖励权重
- $\mu_r, \sigma_r$: 组内均值和标准差
- $\delta$: 数值稳定小常数

### 公式9：[[GRPO]] 最终优化目标

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\lambda_k \hat{A}^{(i)} \log \pi_\theta\left(x_{t_{k+1}}^{(i)} \mid x_{t_k}^{(i)}\right)\right]
$$

**含义**: 时步平衡加权的策略梯度，在单一随机步骤上严格在线更新，无 KL 惩罚，无参考模型。

### 公式10：[[DiffusionNFT|负向感知微调（RealNFT）]]

$$
\hat{v}_{\text{pos}} = \beta\hat{v}_\theta + (1-\beta)\hat{v}_{\text{old}}, \quad \hat{v}_{\text{neg}} = (1+\beta)\hat{v}_{\text{old}} - \beta\hat{v}_\theta
$$

$$
\mathcal{L}_{\text{chosen}} = \|\hat{v}^w_{\text{pos}} - v^w\|^2, \quad \mathcal{L}_{\text{reject}} = \|\hat{v}^l_{\text{neg}} - v^l\|^2
$$

$$
\mathcal{L}_{\text{RealNFT}} = \mathcal{L}_{\text{chosen}} + \mathcal{L}_{\text{reject}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}
$$

**含义**: 用真实视频作为正样本（chosen）、模型生成视频作负样本（rejected），通过隐式正/负策略路径构建偏好对，在单步噪声状态上优化，避免反向传播穿越去噪轨迹。

### 公式11：[[DMD2|分布匹配蒸馏]]

$$
\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_{z,t,\epsilon}\left[-w_t\left(s_{\text{real}}(x_t, t, c) - s_{\text{fake}}(x_t, t, c)\right)\frac{\partial G_\theta(z,c)}{\partial\theta}\right]
$$

**含义**: 将 LingBot-Video 蒸馏为少步生成器，用 teacher 视频扩散模型提供 $s_{\text{real}}$，用在线训练的辅助 score 模型估计 $s_{\text{fake}}$，配合 GAN 目标提升视觉质量。

---

## 关键图表

### Figure 1: LingBot-Video 生成样本

![[LingBot-Video_fig1_samples.png]]

**说明**: T2I 和 T2V 生成示例，展示高视觉保真度、丰富细节和强文本对齐，涵盖多样场景。

### Figure 3 & 4: MoE 配置 Recipe 消融

![[LingBot-Video_fig3_recipe.png]]

![[LingBot-Video_fig4_recipe_compare.png]]

**说明**:
- **专家数量消融**（Fig 3）：固定 1.4B 激活参数，专家数 $E \in \{64, 128, 256\}$，E=128 为最佳性价比
- **细粒度 vs. 粗粒度路由**（Fig 4）：相同 13B 总参数下，细粒度路由（128 专家，$K_r=8$）vs. 粗粒度（64 专家，$K_r=4$），细粒度持续更优——$\binom{128}{8}$ vs. $\binom{64}{4}$ 的路由组合空间差异造就了更强的专家专业化

### Figure 5 & 6: 计算等效 Scaling 对比

![[LingBot-Video_fig5_scaling.png]]

![[LingBot-Video_fig6_7_scaling_speed.png]]

**说明**:
- **Fig 5**：MoE 13B-A1.4B vs. Dense 1.3B（相近激活参数），MoE 整个训练过程均显著领先
- **Fig 6a**：MoE 30B-A3B 接近 Dense 14B 性能，激活参数仅为其约 1/4.7
- **Fig 6b**：MoE 从 13B 到 120B 总参数呈现可预测的 scaling law
- **Fig 7**（合并在 page 10 截图中）：在 1M token 序列长度下，MoE 30B-A3B 相比 Dense 30B 速度快 **3.18×**，相比 Dense 3B（等激活参数）几乎等速（0.97×）

### Figure 9: 数据画像引擎

![[LingBot-Video_fig9_data_profiling.png]]

**说明**: 每个图像/视频样本沿五个维度（结构、语义、运动、相机、质量）标注为结构化档案，驱动下游过滤、平衡采样和描述生成。质量维度包含美学评分（HPSv3）、合成检测（OmniAID）等专用评分器。

### Figure 10: 世界知识拓扑图

![[LingBot-Video_fig10_topological_graph.png]]

**说明**: 语义树（50,000 细粒度叶节点，1,000 中间类别，25 顶层组）+ 动作树联合索引数据，训练反馈（去噪损失按节点聚合）驱动动态采样权重，实现长尾数据上采样和稀缺操作类别的有针对性数据采集。

### Figure 11: 五阶段数据课程

![[LingBot-Video_fig11_curriculum.png]]

**说明**: 从 Stage 1（纯 192p 图像）→ Stage 2（引入 70,000+ 小时具身数据，VLA/导航/第一人称视角）→ Stage 3（480p 双任务 T2V+TI2V）→ Stage 4（具身导向重平衡）→ Stage 5（1080p 高质量精炼集）。图像流（蓝）和视频流（绿）的相对比例在各阶段动态调整。

### Figure 12: 后训练前后通用质量对比

![[LingBot-Video_fig12_posttraining_quality.png]]

**说明**: 后训练显著改善手部/四肢不一致合成、模糊/错误文字渲染和结构物体变形等关键 artifacts。

### Figure 13: 后训练前后具身场景对比

![[LingBot-Video_fig13_embodied_scenarios.png]]

**说明**: 后训练在具身场景中显著提升物理合理性，修复机械臂结构畸变、非物理穿透、过早物体释放和物体复制等 artifacts。

### Figure 14: LingBot-Video-A2V 动作条件架构

![[LingBot-Video_fig14_a2v_architecture.png]]

**说明**: 给定未来 4T 帧的逐帧动作，先转换为相对动作，展平后通过 ActionEmbedder 映射到动作 latent；在初始状态前置零动作以对齐时序；动作 embedding 以残差信号注入每个 Transformer 块，ActionEmbedder 末层零初始化以稳定后训练。

### Figure 15: 内部基准评测结果

![[LingBot-Video_fig15_benchmark.png]]

**说明**: 在 T2V 和 TI2V 两种设定下，对比 NVIDIA Cosmos 3、Wan 2.2 A14B、LongCat-Video、HunyuanVideo 1.5、LTX-2.3。LingBot-Video 在 TI2V 的 General Quality 和 Embodied Domain 两项均获 **最优**；T2V 通用质量排第二，但 Embodied Domain 评分持续领先 Cosmos 3。

### Table 1: RBench 评测结果

| 模型 | 类型 | 平均 | 操作 | 空间 | 多实体 | 长时程 | 推理 | 单臂 | 双臂 | 四足 | 人形 |
|------|------|------|------|------|--------|--------|------|------|------|------|------|
| **LingBot-Video** | 开源 | **0.620** | **0.578** | 0.643 | 0.444 | **0.634** | 0.505 | **0.636** | 0.639 | **0.758** | 0.689 |
| Cosmos3 Super | 开源 | 0.581 | 0.487 | **0.642** | 0.444 | 0.591 | 0.395 | 0.615 | 0.623 | 0.739 | 0.691 |
| Wan 2.2 A14B | 开源 | 0.507 | 0.381 | 0.454 | 0.373 | 0.501 | 0.330 | 0.608 | 0.582 | 0.690 | 0.648 |
| HunyuanVideo 1.5 | 开源 | 0.460 | 0.442 | 0.316 | 0.312 | 0.438 | 0.364 | 0.513 | 0.526 | 0.634 | 0.595 |
| LongCat-Video | 开源 | 0.437 | 0.372 | 0.310 | 0.220 | 0.384 | 0.186 | 0.586 | 0.576 | 0.681 | 0.621 |
| Wan 2.6 | 闭源 | 0.607 | 0.546 | **0.656** | **0.479** | 0.514 | **0.531** | 0.666 | **0.681** | 0.723 | 0.667 |
| Seedance 1.5 pro | 闭源 | 0.584 | **0.577** | 0.495 | 0.484 | 0.570 | 0.470 | 0.648 | 0.641 | 0.680 | **0.692** |
| Veo 3 | 闭源 | 0.563 | 0.521 | 0.508 | 0.430 | 0.530 | 0.504 | 0.634 | 0.610 | 0.689 | 0.637 |

**关键发现**: LingBot-Video 在所有开源模型中综合第一（0.620），超越闭源 Veo 3（0.563），与 Wan 2.6（0.607）和 Seedance 1.5 pro（0.584）接近。四足类别（0.758）和操作类别（0.578）表现尤为突出。

### Figure 16: 公开基准对比

![[LingBot-Video_fig16_public_bench.png]]

**说明**:
- **RBench 平均分**（左）：LingBot-Video 0.620，Wan 2.6（闭源）0.607，Seedance 1.5 pro（闭源）0.584
- **Physics-IQ Verified I2V**（右）：LingBot-Video **40.4**，Cosmos 3 **39.5**，HunyuanVideo 1.5 33.4，Wan 2.2 A14B 32.2——在真实物理实验（固体/流体/热力学/光学/电磁学）的图像到视频预测中排名开源第一

### Figure 17: 用户研究（GSB）

![[LingBot-Video_fig17_user_study.png]]

**说明**: Good-Same-Bad 人类盲评，对比 6 个开源和 4 个商业模型，每对 400 个提示。TI2V 中 LingBot-Video 对所有开源基线均 Good > Bad；T2V 中对多数开源基线占优，与 Cosmos 3 和 HunyuanVideo 相近；落后于 Kling-V3、HappyHorse 1.0 等商业模型。

### Figure 18: DreamDojo vs. LingBot-Video-A2V

![[LingBot-Video_fig18_dreamdojo_compare.png]]

**说明**: 在 EgoDex Eval 和 DreamDojo-HV Eval 上（含训练集外的新物体和动作），A2V 模型展现出更好的物理一致性（如保留黄苹果）和动作跟随能力（如手相对三明治的位置）。

---

## 实验

### 数据集/数据来源

| 类型 | 来源 | 阶段 | 特点 |
|------|------|------|------|
| 通用图像 | 互联网 | Stage 1-4 | 高美学质量 |
| 通用视频 | 互联网 | Stage 2-4 | 多样内容 |
| VLA 视频 | 真实机器人、仿真、开源 | Stage 2-4 | 操作、人形/四足 |
| 导航视频 | 第一人称、户外 | Stage 2-4 | 空间布局 |
| 第一人称视频 | Egocentric | Stage 2-4 | 手部交互 |
| 高清视频 | 精选 | Stage 5 | 1080p 精炼 |
| A2V 后训练 | Fourier GR-1 数据集 | Post | 动作条件 |

具身数据总量：超 **70,000 小时**（Stage 2 注入）

### 实现细节

- **条件编码器**: Qwen3-VL-4B
- **VAE**: Wan2.1-VAE
- **并行训练**: DP + FSDP + Sequence Parallelism (Ulysses) + Expert Parallelism (DeepEP)
- **编译优化**: torch.compile（Inductor backend），相比未编译提升 **1.9× MFU**
- **GRPO 后训练**: 单步随机探索，CPS 采样，$\eta=0.7$，组大小 $G$，无 KL 惩罚
- **A2V 后训练**: 8k 步，全局 batch size 64，LR $10^{-5}$
- **RL 基础设施**: 30B 模型权重同步 20 秒/步，GB 级中间状态节点间 50ms 交换，端到端 MFU 43.9%
- **推理**: SGLang + Diffusers 兼容包，支持 FP8 Triton 核（速度优先模式）

### 可视化结果

定性结果（Fig 19-23）展示 TI2V 和 T2V 在机器人操作、人体动作、自然场景等多类任务下的高质量视频生成，时序一致性和物理合理性明显优于对比模型。

---

## 批判性思考

### 优点

1. **首个大规模开源 MoE 视频模型**：填补了 MoE 在视频扩散领域的空白，公开权重（HuggingFace）和代码（GitHub）
2. **系统性三维改进**：架构（MoE）、数据（机器人增强）、训练目标（物理奖励）三者协同设计，而非单点改进
3. **工程扎实**：Scaling law 验证（13B→120B），推理效率量化（1M token 下 3.18× 加速），RL 基础设施性能数据透明
4. **物理评测覆盖全面**：同时报告 RBench（机器人交互）和 Physics-IQ Verified（真实物理现象），评测维度超过多数对比工作

### 局限性

1. **A2V 数据单一**：Action-to-Video 后训练仅用 Fourier GR-1 数据集，对其他机器人平台的泛化性依赖预训练先验
2. **闭源对比存在差距**：T2V 通用质量落后于 HappyHorse 1.0、Kling-V3 等商业模型，人类评测中明显居下
3. **缺乏消融实验**：三维贡献（MoE/数据/奖励）的单独贡献未完全分离，难以判断各自权重
4. **物理奖励依赖 VLM 评分**：Physical Plausibility 奖励基于 Qwen3.6-27B 评估，存在奖励 hacking 风险（尽管 RealNFT 部分缓解）

### 潜在改进方向

1. 扩展 A2V 训练数据到更多机器人平台，测试跨平台泛化
2. 将六维奖励转化为可插拔的评估框架，供社区使用
3. 探索 MoE 专家专业化分析，验证不同专家是否捕捉到不同物理先验

### 可复现性评估

- [x] 代码开源（GitHub: robbyant/lingbot-video）
- [x] 预训练模型（HuggingFace: robbyant/lingbot-video）
- [x] 训练细节完整（progressive schedule、并行策略、RL 设置）
- [ ] 数据集部分可获取（互联网视频受版权限制；具身数据来源未全部公开）

---

## 关联笔记

### 基于

- [[DiT]]: 单流扩散 Transformer 主干
- [[Rectified Flow]]: 训练和精炼器的核心优化目标
- [[MoE]] / [[DeepSeekMoE]]: Sparse MoE 架构和 fine-grained 专家设计
- [[DMD2]]: 少步蒸馏框架
- [[GRPO]] / [[FlowGRPO]]: 强化学习后训练基础，[[Flash-GRPO]] 的单步随机探索范式
- [[DiffusionNFT]]: RealNFT 的前向过程优化框架
- [[DPO]] / [[RealDPO]]: 真实视频偏好对比的动机

### 对比

- [[Cosmos3]]: NVIDIA 开源世界模型，RBench 0.581 vs 0.620
- [[DreamDojo]]: A2V 世界模型，EgoDex/DreamDojo-HV Eval 对比基线
- [[HunyuanVideo]]: 腾讯开源视频生成模型
- [[LongCat-Video]]: 长视频生成模型，引用其 HPSv3 奖励设计

### 方法相关

- [[Mixture-of-Experts]]: 核心架构范式
- [[Multi-Modal 3D RoPE]]: 统一单流 token 坐标系统
- [[AdaLN-Single]]: 轻量级时步调制设计
- [[CPS]]: GRPO 随机步骤采样策略
- [[FSDP2]]: 训练基础设施权重分片
- [[Sequence Parallelism]]: 长序列分布式训练

### 硬件/数据相关

- [[RBench]]: 具身视频生成专用公开基准
- [[Physics-IQ Verified]]: 真实物理现象预测基准
- [[GR-1]]: A2V 后训练数据集来源

---

## 速查卡片

> [!summary] LingBot-Video (2607.07675)
> - **核心**: 首个开源大规模 MoE 视频扩散模型，专为具身智能 / 机器人世界模拟设计
> - **方法**: 单流 DiT + Sparse MoE（128 专家，E128-A1.4B）+ 5 阶段课程预训练 + 6 维物理奖励 GRPO + RealNFT 对比优化
> - **结果**: RBench 开源第一（0.620），Physics-IQ Verified 开源第一（40.4），TI2V 内部基准开源第一
> - **代码**: [github.com/robbyant/lingbot-video](https://github.com/robbyant/lingbot-video)

---

*笔记创建时间: 2026-07-10*

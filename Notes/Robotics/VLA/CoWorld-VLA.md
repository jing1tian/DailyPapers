---
title: "CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving"
method_name: "CoWorld-VLA"
authors: [Minqing Huang, Yujiao Xiang, Zihan Liang, Jiajie Huang, Jingqi Wang, Yuheng Zhou, Zhi Xu, Feiyang Tan, Hangning Zhou, Mu Yang, Gong Che]
year: 2026
venue: arXiv
tags: [autonomous-driving, vla, world-model, diffusion-planner, latent-reasoning, multi-expert]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.10426
created: 2026-08-27
---

# 论文笔记：CoWorld-VLA: Thinking in a Multi-Expert World Model for Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未披露（AFARI Research） |
| 日期 | May 2026 |
| 项目主页 | https://github.com/AFARI-Research/CoWorld-VLA |
| 对比基线 | [[DriveLaW]], [[Epona]], [[PWM]], [[ResWorld]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.10426) / [Code](https://github.com/AFARI-Research/CoWorld-VLA) |

---

## 一句话总结

> CoWorld-VLA 用四类专家 Token（语义、几何、动态、轨迹）构建[[Latent Chain-of-Thought|潜在思维链]]，通过[[HMEF|层次化多专家融合扩散规划器]]将多源世界知识转化为自动驾驶轨迹，在 NAVSIM v1 上达到 90.0 PDMS。

---

## 核心贡献

1. **多专家潜在世界表示**: 将 [[VLA]] 的隐状态同时对齐四种异构世界知识（语义、几何、动态演化、自车轨迹），形成比单一世界模型更完整的规划潜在表示
2. **三阶段训练流水线**: Stage 1 预训练行为条件视频生成器；Stage 2 多专家表示学习；Stage 3 扩散规划器融合，解耦表示学习与策略学习
3. **HMEF 扩散规划器**: 采用双流[[Diffusion Model|扩散模型]]，以专家 Token 为条件联合去噪，学习可优化的专家融合权重 $\alpha_e$，在推理时动态聚合多专家轨迹预测

---

## 问题背景

### 要解决的问题

现有自动驾驶 [[VLA]] 在感知到规划之间缺乏结构化中间推理。直接映射会丢失连续时空信息，文本 CoT 无法保留稠密几何动态细节。

### 现有方法的局限

- **直接动作预测**（UniAD、DiffusionDrive）：无中间推理，感知噪声直接传播到规划
- **文本 CoT**（LaST-VLA、SGDrive）：语言中间步骤丢失连续时空信息，无法精确表达几何约束
- **单世界模型潜在推理**（PWM、ResWorld、DriveLaW）：依赖单一隐式世界表示，可能不完整或与动作弱耦合

### 本文的动机

多种世界知识（语义意图、几何结构、未来动态、自车轨迹先验）在自然上是互补的，将四者统一到 [[VLM]] 隐空间作为[[Latent Chain-of-Thought|潜在思维链]]，可以在不依赖文本的前提下保留丰富时空信息，再由[[Diffusion Model|扩散规划器]]将多专家信息融合为轨迹。

---

## 方法详解

### 模型架构

CoWorld-VLA 采用**三阶段渐进训练**架构：

- **Backbone**: [[Qwen3-VL]] VLM + [[Wan]] 视频生成基座
- **输入**: 当前帧图像 $o_t$ + 文本条件（速度、导航指令、场景描述）
- **核心模块**:
  - Stage 1: [[Flow Matching]] 视频预测（行为条件生成器）
  - Stage 2: 四路专家监督 — [[JEPA]] 语义 / [[VGGT]] 几何 / [[Wan]] 动态 / MLP 轨迹
  - Stage 3: [[HMEF|HMEF 扩散规划器]] — 双流 Transformer + 专家融合权重
- **输出**: $T$ 步自车轨迹 $A_{t+1:t+T}$

### 核心模块

#### 模块 1: 多专家表示学习（Stage 2）

**设计动机**: 利用 [[JEPA]] 语义特征、[[VGGT]] 3D 几何特征、[[Wan]] 动态特征、MLP 轨迹头，分别监督四类专家 Token，让 VLM 隐状态同时编码多种世界先验

**具体实现**:
- 在 [[Qwen3-VL]] 输入序列中插入四组可学习的专家占位 Token：$t_{sem}, t_{geo}, t_{dyn}, t_{traj}$
- Backbone 输出 $\{H_{ctx}, H_{sem}, H_{geo}, H_{dyn}, H_{traj}\}$
- 各专家监督信号来自对未来帧 $o_{fut}$ 分别抽取的外部特征：
  - $Z_{sem} = \text{Pool}(E_{sem}(o_{fut}))$（[[JEPA]] 编码器）
  - $Z_{geo} = \text{Pool}(E_{geo}(o_{fut}))$（[[VGGT]] 编码器）
  - 动态：冻结 Stage 1 的 [[Wan]] 生成器，用 $H_{dyn}$ 替代文本条件来驱动未来帧重建
  - 轨迹：轻量 MLP 回归 $T$ 个 waypoint

#### 模块 2: HMEF 扩散规划器（Stage 3）

**设计动机**: 利用[[Diffusion Model|扩散模型]]的连续生成能力，以四类专家 Token 为条件联合去噪，学习可优化的专家融合权重，在推理阶段动态聚合

**具体实现**:
- 双流架构：干净场景 Token 流 $C = H_{ctx}$，带噪专家动作 Token 流 $R = \{H_{sem}, H_{geo}, H_{dyn}, H_{traj}\}$
- 时间步嵌入 $e_\tau$ 注入噪声流
- 双流之间执行联合 Self-Attention：$X_0 = D_\theta(C, R, e_\tau)$
- 每路专家独立预测轨迹 $\hat{A}_e$，再以学习权重 $\alpha_e = \text{softmax}(w_e)$ 融合：$\bar{A} = \sum_e \alpha_e \hat{A}_e$
- 推理 10 步去噪

---

## 关键公式

### 公式 1: [[VLA|标准 VLA 建模]]

$$
p_\theta(A_{t+1:t+T} \mid o_t, c_t)
$$

**含义**: 直接将多模态输入 $o_t$（图像）和 $c_t$（语言指令）映射到未来动作序列，无中间推理

**符号说明**:
- $o_t$: 当前时刻观测帧
- $c_t$: 语言条件（导航指令、速度等）
- $A_{t+1:t+T}$: 未来 $T$ 步自车轨迹

---

### 公式 2: [[Latent Chain-of-Thought|多专家潜在推理 VLA]]

$$
p_\theta(A_{t+1:t+T} \mid o_t, c_t, \mathcal{Z})
$$

$$
\mathcal{Z} = \{z_{sem},\, z_{geo},\, z_{dyn},\, z_{traj}\}
$$

**含义**: 引入四类专家 Token 集合 $\mathcal{Z}$ 作为结构化潜在中间推理，分别编码语义交互、几何结构、动态演化、自车轨迹先验

**符号说明**:
- $z_{sem}$: 语义交互 Token（[[JEPA]] 对齐）
- $z_{geo}$: 几何结构 Token（[[VGGT]] 对齐）
- $z_{dyn}$: 动态演化 Token（[[Wan]] 世界模型对齐）
- $z_{traj}$: 自车轨迹 Token（直接轨迹回归监督）

---

### 公式 3: [[Flow Matching|Stage 1 流匹配噪声与速度目标]]

$$
\begin{aligned}
\tilde{z}_{f,\sigma} &= (1-\sigma) z_f + \sigma \varepsilon \\
v_{target} &= \varepsilon - z_f
\end{aligned}
$$

**含义**: 对未来帧 latent $z_f$ 施加噪声扰动构成带噪样本，速度目标为噪声与干净 latent 之差，用于[[Flow Matching]]训练

**符号说明**:
- $\sigma \in (0,1)$: 噪声水平
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $z_f$: 未来帧 VAE latent
- $v_{target}$: 流匹配速度目标

---

### 公式 4: [[Flow Matching|Stage 1 流匹配损失]]

$$
\mathcal{L}_{flow} = \mathbb{E}_{z_h, z_f, \varepsilon, \sigma, c} \left[ \|F_\theta(\tilde{z}_\sigma, c, \sigma)_f - (\varepsilon - z_f)\|_2^2 \right]
$$

**含义**: 只对未来帧输出 token 计算流匹配损失，历史帧 $z_h$ 保持确定性不参与去噪梯度

**符号说明**:
- $F_\theta$: [[Wan]] 视频生成网络
- $c = E_{text}(P)$: 结构化文本条件嵌入
- $(\cdot)_f$: 只取未来帧对应 token

---

### 公式 5: [[JEPA|语义对齐损失]]

$$
\mathcal{L}_{sem} = \lambda_{l1} \cdot \text{SmoothL1}(\hat{Z}_{sem}, Z_{sem}) + \lambda_{cos}(1 - \cos(\hat{Z}_{sem}, Z_{sem}))
$$

**含义**: 组合 SmoothL1（幅值对齐）和余弦相似度（方向对齐）监督语义专家 Token 向 [[JEPA]] 特征靠拢

**符号说明**:
- $\hat{Z}_{sem}$: VLM 预测的语义专家特征
- $Z_{sem} = \text{Pool}(E_{sem}(o_{fut}))$: [[JEPA]] 编码器提取的目标特征
- $\lambda_{l1}, \lambda_{cos}$: 幅值与方向损失权重

---

### 公式 6: [[VGGT|几何对齐损失]]

$$
\mathcal{L}_{geo} = \text{MSE}(\hat{Z}_{geo}, Z_{geo})
$$

**含义**: 均方误差监督几何专家 Token 向 [[VGGT]] 3D 空间特征对齐

---

### 公式 7: 动态演化损失

$$
\hat{o}_{fut} = W_\psi(o_t, H_{dyn}), \quad \mathcal{L}_{dyn} = \mathcal{L}_{flow}(\hat{o}_{fut}, o_{fut};\, o_t, H_{dyn})
$$

**含义**: 以 VLM 动态专家隐状态 $H_{dyn}$ 替代文本条件驱动冻结 [[Wan]] 生成器重建未来帧，流匹配损失监督动态 Token

**符号说明**:
- $W_\psi$: 冻结的 [[Wan]] 视频生成器
- $H_{dyn}$: VLM 动态专家隐状态

---

### 公式 8: 轨迹预测损失

$$
\hat{A}_{t+1:t+T} = \Phi_{traj}(H_{traj}[-1]), \quad \mathcal{L}_{traj} = \text{MSE}(\hat{A}_{t+1:t+T}, A_{t+1:t+T})
$$

**含义**: 轻量 MLP 头 $\Phi_{traj}$ 直接从轨迹专家末尾 Token 回归 $T$ 步 waypoint，MSE 监督

**符号说明**:
- $\Phi_{traj}$: 轨迹回归 MLP 头
- $H_{traj}[-1]$: 轨迹专家 Token 的最后一个隐状态

---

### 公式 9: [[Multi-Expert Training|Stage 2 总损失]]

$$
\mathcal{L}_{total} = w_{dyn} \cdot \mathcal{L}_{dyn} + w_{sem} \cdot \mathcal{L}_{sem} + w_{geo} \cdot \mathcal{L}_{geo} + w_{traj} \cdot \mathcal{L}_{traj}
$$

**含义**: 四路专家损失加权求和，权重设置体现动态和轨迹监督为主、语义和几何为辅

**符号说明**:
- $w_{dyn}=1.0$, $w_{traj}=1.0$: 主要监督信号权重
- $w_{sem}=0.1$, $w_{geo}=0.1$: 辅助对齐权重

---

### 公式 10: [[HMEF|HMEF 带噪动作构建]]

$$
A_\tau = (1 - \tau)\varepsilon + \tau A^{norm}, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 用 logit-normal 调度参数 $\tau$ 在纯噪声和归一化 GT 轨迹之间插值，构建训练时的带噪输入

**符号说明**:
- $\tau \in (0,1)$: 去噪进度（logit-normal 分布采样）
- $A^{norm}$: 归一化后的真值轨迹
- $\varepsilon$: 标准高斯噪声

---

### 公式 11: [[HMEF|HMEF 扩散损失]]

$$
\mathcal{L}_{diff} = \frac{1}{N_e} \sum_{e=1}^{N_e} \|\hat{A}_e - A^{norm}\|_2^2
$$

**含义**: 对所有 $N_e$ 路专家的去噪预测 $\hat{A}_e$ 分别计算 L2 损失后平均，鼓励每路专家独立具备轨迹预测能力

**符号说明**:
- $N_e = 4$: 专家数量
- $\hat{A}_e$: 第 $e$ 路专家去噪输出的轨迹

---

### 公式 12: [[HMEF|专家融合与动作损失]]

$$
\bar{A} = \sum_{e=1}^{N_e} \alpha_e \hat{A}_e, \quad \alpha_e = \text{softmax}(w_e)
$$

$$
\mathcal{L}_{act} = \mathcal{L}_{diff} + \lambda_{fusion} \|\bar{A} - A^{norm}\|_2^2
$$

**含义**: 可学习融合权重 $\alpha_e$（由 softmax 归一化）加权聚合各专家轨迹预测，附加融合一致性损失

**符号说明**:
- $w_e$: 可优化的专家权重参数
- $\lambda_{fusion}$: 融合一致性损失权重
- $\bar{A}$: 最终融合轨迹

---

## 关键图表

### Figure 1: 推理范式对比 / Reasoning Paradigm Comparison

![Figure 1](https://arxiv.org/html/2605.10426v3/figures/introduction.png)

**说明**: 四种自动驾驶推理范式对比。(a) 直接动作预测；(b) 文本 [[Chain-of-Thought Reasoning|CoT]] 推理；(c) 单世界模型潜在推理；(d) CoWorld-VLA 多专家潜在推理链。展示了从无结构推理到多专家结构化推理的演进。

---

### Figure 2: CoWorld-VLA 系统概览 / System Overview

![Figure 2](https://arxiv.org/html/2605.10426v3/overview_new.png)

**说明**: CoWorld-VLA 三阶段训练流水线。Stage 1 视频生成器预训练（[[Flow Matching]] + [[Wan]]）；Stage 2 多专家表示对齐（[[JEPA]]/[[VGGT]]/动态/轨迹四路监督）；Stage 3 [[HMEF|HMEF 扩散规划器]]融合多专家 Token 生成轨迹。

---

### Figure 3: 未来场景生成定性对比 / Future Scene Generation

![Figure 3](https://arxiv.org/html/2605.10426v3/figures/case1_main.png)

**说明**: Stage 1 与 Stage 2 的未来帧生成质量对比。Stage 2 在加入多专家表示监督后，行驶方向与车道级场景演化保真度显著优于 Stage 1，与真值更一致。

---

### Figure 4: 轨迹规划定性对比 / Trajectory Planning Comparison

![Figure 4a](https://arxiv.org/html/2605.10426v3/figures/traj_best.png)

![Figure 4b](https://arxiv.org/html/2605.10426v3/figures/traj_best3.png)

**说明**: 三个训练阶段的轨迹规划定性对比。Stage 2 轨迹方向合理但换道和转向有偏差；Stage 3 加入 [[HMEF|HMEF 规划器]]后轨迹与真值对齐更精准，尤其在复杂转弯和车道保持场景。

---

### Table 1: NAVSIM v1 性能对比

| Method | 发表 | Sensors | Frames | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|--------|------|---------|--------|-----|------|------|--------|-----|-------|
| UniAD | CVPR'23 | C | N | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Hydra-MDP | arXiv'24 | C+L | N | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | CVPR'25 | C+L | N | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| TrajDiff | arXiv'25 | C+L | N | 98.1 | 97.0 | 94.3 | 100 | 82.7 | 88.5 |
| LAW | ICLR'25 | C | N | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| FSDrive | NeurIPS'25 | C | N | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| Epona | ICCV'25 | C | N | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| Resim | NeurIPS'25 | C | N | – | – | – | – | – | 86.6 |
| PWM | NeurIPS'25 | C | N | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| WoTE | ICCV'25 | C+L | N | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| ResWorld | ICLR'26 | C+L | N | 98.9 | 96.5 | 95.6 | 100 | 83.1 | 89.0 |
| WorldDrive | arXiv'26 | C | N | 98.4 | 96.8 | 95.2 | 100 | 83.3 | 89.0 |
| DriveLaW | CVPR'26 | C | N | 99.0 | 97.1 | 96.7 | 100 | 81.3 | 89.1 |
| ReCogDrive† | ICLR'26 | C | 1 | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| DriveVLA-W0 | ICLR'26 | C | N | 98.4 | 95.3 | 95.4 | 100 | 80.9 | 87.2 |
| LaST-VLA† | arXiv'26 | C | 1 | 98.7 | 95.4 | 95.7 | 100 | 80.5 | 87.3 |
| SGDrive† | CVPR'26 | C | N | 98.6 | 95.1 | 95.4 | 100 | 81.2 | 87.4 |
| Uni-World VLA | arXiv'26 | C | N | 98.7 | 96.7 | 96.1 | 100 | 83.2 | 89.4 |
| CoWorld-VLA¶ | – | C | 1 | 98.5 | 96.9 | 95.4 | 100 | 83.2 | 89.1 |
| **CoWorld-VLA (ours)** | **–** | **C** | **1** | **99.1** | **97.0** | **96.5** | **100** | **84.0** | **90.0** |

**表格说明**: CoWorld-VLA 仅用单帧摄像头输入达到 90.0 PDMS，超越所有使用多帧或激光雷达的基线，NC 99.1% 为全表最高

---

### Table 2: NAVSIM v2 扩展 PDMS 对比

| Method | NC↑ | DAC↑ | DDC↑ | TL↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS*↑ | EPDMS↑ |
|--------|-----|------|------|-----|-----|------|-----|-----|-----|---------|--------|
| TransFuser | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 | – |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | – | 84.5 |
| DriveSuprim | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 | – |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 85.5 | 87.5 |
| WoTE | 98.5 | 96.8 | 98.8 | 99.8 | 86.1 | 97.9 | 95.5 | 98.3 | 82.9 | – | 87.7 |
| DreamerAD | 98.0 | 97.2 | 99.5 | 99.8 | 87.8 | 97.4 | 97.5 | 98.3 | 72.4 | – | 87.7 |
| PWM | 98.8 | 95.9 | 99.4 | 99.9 | 86.4 | 98.4 | 97.6 | 98.3 | 85.3 | – | 88.2 |
| DriveLaW | 98.7 | 96.9 | 99.6 | 99.8 | 87.5 | 98.3 | 97.6 | 98.4 | 77.4 | – | 88.6 |
| Drive-JEPA (R34) | 98.8 | 97.4 | 99.0 | 99.8 | 83.5 | 98.0 | 96.2 | 98.1 | 85.6 | 85.4 | – |
| Latent-WAM | 98.1 | 97.3 | 99.6 | 99.8 | 87.7 | 97.3 | 97.6 | 98.1 | 87.3 | – | 89.3 |
| WAM-Flow | 98.5 | 94.5 | 99.5 | 99.8 | 86.9 | 96.8 | 97.4 | 97.6 | 73.9 | 84.7 | – |
| ReCogDrive† | 98.3 | 95.2 | 98.3 | 99.8 | 87.1 | 97.5 | 96.6 | 99.5 | 86.5 | 83.6 | – |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | – | 86.1 |
| SGDrive† | 98.6 | 94.3 | 99.5 | 99.9 | 86.0 | 97.9 | 96.1 | 98.3 | 85.9 | – | 86.2 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | – | 86.8 |
| **CoWorld-VLA (ours)** | **99.1** | **97.0** | **99.6** | **99.9** | **87.8** | **98.5** | **97.7** | **98.2** | **86.2** | **86.2** | **90.0** |

**表格说明**: 在 NAVSIM v2 扩展 EPDMS 指标上同样达到 90.0，NC 和 TTC 均居全表前列

---

### Table 3: 视频生成 FVD 对比

| Method | SVD | GenAD | DrivingGPT | Epona | DriveLaW | **CoWorld-VLA** |
|--------|-----|-------|-----------|-------|----------|----------------|
| Dataset | NAVSIM | OpenDV | NAVSIM | NuPlan | NuPlan | NAVSIM |
| FVD↓ | 227.5 | 184.0 | 142.6 | 61.3 | 55.6 | **32.7** |

**表格说明**: CoWorld-VLA 视频生成质量 FVD=32.7，远优于 DriveLaW（55.6）和 Epona（61.3），说明动态专家监督显著提升了场景生成的保真度

---

### Table 4: 多专家设计消融

| EgoT. | Geo. | Sem. | Dyn. | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|-------|------|------|------|-----|------|------|--------|-----|-------|
| ✓ | | | | 97.7 | 92.7 | 92.7 | 100 | 78.5 | 83.7 |
| ✓ | ✓ | | | 97.7 | 93.9 | 92.7 | 100 | 80.3 | 85.1 |
| ✓ | | ✓ | | 98.1 | 93.7 | 93.8 | 100 | 79.0 | 85.2 |
| ✓ | ✓ | ✓ | | 98.3 | 95.4 | 95.1 | 100 | 81.1 | 87.3 |
| ✓ | ✓ | | ✓ | 98.4 | 95.6 | 95.0 | 100 | 81.9 | 87.7 |
| ✓ | ✓ | ✓ | ✓ | 98.4 | 96.5 | 95.3 | 100 | 82.3 | 88.7 |

**关键发现**: 四路专家相互互补，从仅轨迹（83.7）到全专家（88.7）提升 5.0 PDMS；几何和语义专家贡献相近（+1.4/+1.5），动态专家在几何+语义基础上再加 +1.4

---

### Table 5: 表示学习与规划模块消融

| 表示 | 规划器 | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|------|--------|-----|------|------|--------|-----|-------|
| EgoT. only | VLM only | 97.7 | 92.7 | 92.7 | 100 | 78.5 | 83.7 |
| Full Experts | VLM only | 98.4 | 96.5 | 95.3 | 100 | 82.3 | 88.7 |
| Full Experts | VLM+AE | 98.5 | 96.9 | 95.4 | 100 | 83.2 | 89.1 |
| Full Experts | VLM+HMEF | 98.3 | 96.8 | 95.0 | 100 | 83.2 | 88.9 |
| Full Experts | VLM+HMEF | **99.1** | **97.0** | **96.5** | **100** | **84.0** | **90.0** |

**关键发现**: [[HMEF|HMEF 规划器]]在全专家表示基础上带来最终 +1.3 PDMS 提升（88.7→90.0）；仅换规划器（VLM→HMEF）贡献约 0.2~1.3 PDMS

---

### Table 6: 扩散去噪步数影响

| Steps | NC↑ | DAC↑ | TTC↑ | Comf.↑ | EP↑ | PDMS↑ |
|-------|-----|------|------|--------|-----|-------|
| 5 | 99.0 | 96.8 | 95.9 | 100 | 83.6 | 89.5 |
| **10** | **99.1** | **97.0** | **96.5** | **100** | **84.0** | **90.0** |
| 20 | 99.1 | 96.7 | 96.4 | 100 | 83.5 | 89.7 |

**关键发现**: 10 步为最优，步数过多（20步）反而轻微下降，说明扩散规划不需要大量去噪步骤

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[NAVSIM]] v1 | navtest split | 非反应式闭环评估 | 主要评测 |
| [[NAVSIM]] v2 | navtest split | 扩展 EPDMS 指标 | 扩展评测 |
| NuPlan | - | 长尾驾驶场景 | 视频生成对比（FVD） |

### 实现细节

- **VLM Backbone**: [[Qwen3-VL]]（参数量未披露）
- **视频生成基座**: [[Wan]]（Stage 1 预训练）
- **扩散规划器步数**: 10 步推理
- **Stage 2 损失权重**: $w_{dyn}=1.0, w_{traj}=1.0, w_{sem}=0.1, w_{geo}=0.1$
- **输入**: 单帧图像（1 Frame），不依赖历史帧或激光雷达
- **训练**: 三阶段渐进训练，后续阶段冻结前阶段部分权重

### 可视化结果

Stage 2 相比 Stage 1 更好地保持行驶方向和车道级场景一致性，Stage 3 轨迹在换道、转弯等复杂场景中明显优于 Stage 2，与 GT 对齐更精准。

---

## 批判性思考

### 优点

1. **多源互补知识集成**: 同时引入语义（[[JEPA]]）、几何（[[VGGT]]）、动态（[[Wan]]）、轨迹四类监督，不同专家提供正交互补信息，消融实验清晰验证各专家贡献
2. **单帧优势显著**: 仅使用单帧摄像头达到 NAVSIM v1 90.0 PDMS，超越多帧和多传感器方法，计算成本更低
3. **推理时专家融合**: 专家 Token 在推理阶段通过扩散规划器主动参与轨迹生成，而非仅在训练阶段提供监督，确保了推理时的多专家信息整合

### 局限性

1. **三阶段训练计算开销大**: 需要同时维护 [[JEPA]]、[[VGGT]]、[[Wan]] 等多个教师模型，内存和计算要求高；作者承认但未量化
2. **仅支持单帧输入**: 当前框架不利用时序历史，对需要速度估计的场景可能存在局限
3. **机构不透明**: AFARI Research 信息不公开，代码尚未发布，复现困难

### 潜在改进方向

1. 探索 [[LoRA]] 参数高效微调替代全参数 Stage 3 训练
2. 引入特征缓存减少多专家教师的重复前向计算
3. 扩展到多摄像头配置以处理全环绕感知

### 可复现性评估

- [ ] 代码开源（承诺开源，暂未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（损失权重、去噪步数等关键超参已披露）
- [x] 数据集可获取（NAVSIM 公开可用）

---

## 关联笔记

### 基于

- [[Wan]]: Stage 1 视频生成基座，行为条件预测世界模型
- [[Qwen3-VL]]: VLM Backbone，多专家 Token 的载体
- [[JEPA]]: 语义专家监督信号来源
- [[VGGT]]: 几何专家监督信号来源

### 对比

- [[DriveLaW]]: 同为 NAVSIM CVPR'26 基线，单摄像头，PDMS 89.1
- [[PWM]]: NeurIPS'25 世界模型规划基线，多帧，PDMS 88.1
- [[Epona]]: ICCV'25 视频生成基线，FVD 61.3 vs 32.7
- [[ResWorld]]: ICLR'26 基线，使用激光雷达，PDMS 89.0

### 方法相关

- [[World Model]]: 核心范式，本文扩展为多专家世界模型
- [[Flow Matching]]: Stage 1 和动态专家的训练目标
- [[Chain-of-Thought Reasoning]]: 文本 CoT 的潜在版本（本文核心创新）
- [[Multi-Expert Training]]: Stage 2 多路专家联合优化

### 硬件/数据相关

- [[NAVSIM]]: 主要评测基准（v1 和 v2）

---

## 速查卡片

> [!summary] CoWorld-VLA (arXiv 2026)
> - **核心**: 四类专家 Token 构建潜在思维链，多专家扩散规划器融合轨迹
> - **方法**: Stage 1 视频预训练 → Stage 2 多专家对齐 → Stage 3 HMEF 扩散规划
> - **结果**: NAVSIM v1 PDMS 90.0（单帧摄像头，全表最优）；FVD 32.7（全表最优）
> - **代码**: https://github.com/AFARI-Research/CoWorld-VLA

---

*笔记创建时间: 2026-08-27*

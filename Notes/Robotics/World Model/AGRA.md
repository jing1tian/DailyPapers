---
title: "Making Foresight Actionable: Repurposing Representation Alignment in World Action Models"
method_name: "AGRA"
authors: [Lu Qiu, Yizhuo Li, Yi Chen, Yuying Ge, Yixiao Ge, Xihui Liu]
year: 2026
venue: arXiv
tags: [world-action-model, representation-alignment, robot-manipulation, video-diffusion, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.12217
created: 2026-06-12
---

# 论文笔记：Making Foresight Actionable — AGRA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | XPENG Robotics |
| 日期 | June 2026 |
| 项目主页 | [xpeng-robotics.github.io/agra](https://xpeng-robotics.github.io/agra) |
| 对比基线 | [[GR00T]]、[[Cosmos-Predict]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12217) |

---

## 一句话总结

> AGRA 通过将视频扩散模型的隐状态与冻结 DINOv2 编码器对齐，解决了 World Action Model 中"可见未来≠可控动作"的 action-grounding gap，大幅提升机器人操作成功率。

---

## 核心贡献

1. **问题诊断**: 发现并量化了 WAM 中的 action-grounding gap——视觉重建质量优秀但动作解码器关注错误区域（任务无关背景），而非手-物体交互区域
2. **AGRA 方法**: 设计 Action-Grounded Representation Alignment 损失，将视频扩散模型的隐状态对齐至冻结 [[DINOv2]] 编码器的语义特征，使表征组织方式转向动作控制所需的稀疏空间功能信息
3. **实验验证**: 在 XPENG IRON-R01-1.11 人形机器人上实现 80% in-distribution 成功率（vs 基线 34%），在 RoboCasa GR1 仿真平台上超越 GR00T-N1.6 基线 18.8 个百分点

---

## 问题背景

### 要解决的问题

[[World Action Model]]（WAM）将视频预测模型与动作解码器耦合：先用 [[Video Diffusion Model]] 预测未来视觉帧，再从视频 DiT 的隐状态解码机器人动作。然而，"生成合理的未来视觉预测并不总能保证提取出准确的动作控制信号"。

### 现有方法的局限

优化用于像素重建的隐状态会编码密集的外观细节，而动作控制需要的是**稀疏的、空间局部的功能性信息**（如手的位置、物体抓取点）。这导致：

- 动作解码器的 [[Cross-Attention]] 错误地集中于任务无关的背景区域
- 对任务无关的视觉扰动（背景、光照变化）高度敏感
- 泛化能力差：语义/实例/属性级别的 OOD 场景成功率大幅下降

### 本文的动机

冻结的 [[DINOv2]] 编码器具有"对象中心、空间连贯"的特性——语义和功能相似的区域被组织成空间一致的结构，正是动作控制所需要的表征形式。通过将视频 DiT 的内部表征**对齐**至 DINOv2 的语义空间，可以"重用"视频模型的时序动态建模能力，同时注入空间语义结构。

---

## 方法详解

### 模型架构

AGRA 基于 [[World Action Model]] 框架，在 [[Cosmos-Predict]] 视频 DiT 的基础上添加表征对齐目标：

- **视频骨干**: [[Cosmos-Predict-2.5]]-2B（28层），输入分辨率 192×336，时序压缩比 4（5个潜在帧）
- **动作头**: 8层 [[Transformer]] 动作解码器，500M 参数，通过 [[Cross-Attention]] 从视频 DiT 隐状态提取特征
- **语义教师**: 冻结 DINOv2 编码器（提供语义对齐目标）
- **输入**: 语言指令 $c$ + 当前观测 $o_0$ + 本体感知状态 $s_0$
- **输出**: [[Action Chunking|动作块]] $\hat{a}_{1:K} \in \mathbb{R}^{K \times d_a}$

### 核心模块

#### 模块 1: World-Action Interface（世界-动作接口）

**设计动机**: 利用 [[Cross-Attention]] 将视频 DiT 的时空特征注入动作解码器。

**具体实现**:

- 从视频 DiT 第 $\ell_j$ 层提取隐状态 $H^{\text{vid}}_{\ell_j}$
- 通过线性投影 $\text{Proj}_j$ 映射到动作特征空间
- 动作 token $X^{\text{act}}_j$ 以投影后的视频特征作为 Key/Value 进行 Cross-Attention

#### 模块 2: AGRA 对齐目标

**设计动机**: 用 [[DINOv2]] 的空间语义特征作为"锚点"，正则化视频特征的组织方式，使其对动作控制更友好。

**具体实现**:

- 冻结 DINOv2 提取参考帧 $\bar{o}_t$ 的 patch 级语义特征 $Y_t \in \mathbb{R}^{H_d \times W_d \times d_d}$
- 将 $Y_t$ 插值到视频 DiT 特征图分辨率得到对齐目标 $\tilde{Y}_t$
- 通过可学习投影头 $P_k$ 将视频隐状态映射到 DINOv2 特征维度
- 用 [[Cosine Similarity]] 损失最小化两者距离

**层选择策略**: AGRA 优先作用于网络深度约 1/3 处（第 8 层），保留更深层的运动动态建模能力。多层桥接时均匀采样：

$$
\ell_j = \left\lfloor \frac{j(M-1)}{N-1} \right\rceil
$$

其中 $M$ 为视频 DiT 总层数，$N$ 为桥接层数，$j \in \{0, \ldots, N-1\}$。

#### 模块 3: 训练-推理一致性

**设计动机**: 避免训练和推理时特征分布的不匹配。

**具体实现**: 推理时，视频 DiT 在固定高噪声水平 $\tau_v^{\text{cond}} = 1$ 下对未来潜在 token 做一次前向传播。高噪声表征保留全局任务动态，而非聚焦像素细节，与训练时 [[Noise Conditioning]] 分布保持一致。

---

## 关键公式

### 公式 1: [[Action Chunking|策略定义]]

$$
\hat{a}_{1:K} = \pi(o_0, s_0, c), \quad a_{1:K} \in \mathbb{R}^{K \times d_a}
$$

**含义**: WAM 策略从当前帧、本体状态和语言指令预测 $K$ 步动作块。

**符号说明**:
- $o_0$: 当前视觉观测
- $s_0$: 本体感知状态（关节位置等）
- $c$: 语言任务指令
- $K$: 动作预测时域长度
- $d_a$: 动作维度

### 公式 2: [[Cross-Attention|世界-动作接口]]

$$
G_j = \text{Proj}_j(H^{\text{vid}}_{\ell_j}) \in \mathbb{R}^{N_v \times d_{\text{act}}}
$$

$$
X^{\text{act}}_j = \text{CrossAttn}_j(Q = X^{\text{act}}_{j-1},\ K = G_j,\ V = G_j)
$$

**含义**: 视频 DiT 第 $\ell_j$ 层的隐状态经线性投影后，作为 Key/Value 注入动作解码器第 $j$ 层的 Cross-Attention。

**符号说明**:
- $H^{\text{vid}}_{\ell_j}$: 视频 DiT 第 $\ell_j$ 层的隐状态
- $G_j$: 投影后的视频特征（动作特征空间）
- $N_v$: 视频 token 数量
- $d_{\text{act}}$: 动作特征维度

### 公式 3: [[DINOv2|语义对齐目标构建]]

$$
Y_t = g_\psi(\bar{o}_t) \in \mathbb{R}^{H_d \times W_d \times d_d}
$$

$$
\tilde{Y}_t = \text{Interp}(Y_t;\ H_v, W_v) \in \mathbb{R}^{H_v \times W_v \times d_d}
$$

**含义**: 冻结 DINOv2 编码器 $g_\psi$ 从参考帧提取 patch 特征，再插值到视频 DiT 特征分辨率，作为对齐目标。

**符号说明**:
- $g_\psi$: 冻结 DINOv2 编码器
- $\bar{o}_t$: 第 $t$ 帧的参考图像（去噪后）
- $H_d, W_d$: DINOv2 输出特征图尺寸
- $H_v, W_v$: 视频 DiT 特征图尺寸
- $d_d$: DINOv2 特征维度

### 公式 4: [[Representation Alignment|AGRA 投影与特征提取]]

$$
H^{\text{vid}}_{\ell_k^{\text{agra}}} = f^{\text{vid}}_{\theta, \ell_k^{\text{agra}}}(z^{\tau_v}_{1:T_v}, o_0, c, \tau_v)
$$

$$
Z_k = P_k(H^{\text{vid}}_{\ell_k^{\text{agra}}}) \in \mathbb{R}^{T_v \times H_v \times W_v \times d_d}
$$

**含义**: 从视频 DiT 第 $\ell_k^{\text{agra}}$ 层提取隐状态，经可学习投影头 $P_k$ 映射到 DINOv2 特征空间。

**符号说明**:
- $f^{\text{vid}}_\theta$: 视频 DiT 前向函数
- $z^{\tau_v}_{1:T_v}$: 噪声水平 $\tau_v$ 的视频潜在 token
- $P_k$: 第 $k$ 个可学习线性投影头
- $T_v$: 视频时间帧数

### 公式 5: [[Cosine Similarity|AGRA 对齐损失]]

$$
\mathcal{L}_{\text{AGRA}} = -\frac{1}{K T_v H_v W_v} \sum_{k=1}^{K} \sum_{t=1}^{T_v} \sum_{u=1}^{H_v} \sum_{v=1}^{W_v} \cos(Z_{k,t,u,v},\ \tilde{Y}_{t,u,v})
$$

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2 + \varepsilon}
$$

**含义**: 对齐损失最大化视频 DiT 投影特征与 DINOv2 目标特征在所有时空位置的余弦相似度。

**符号说明**:
- $K$: AGRA 应用的层数（桥接层数）
- $Z_{k,t,u,v}$: 第 $k$ 个桥接层在 $(t, u, v)$ 位置的投影特征向量
- $\tilde{Y}_{t,u,v}$: DINOv2 在 $(t, u, v)$ 位置的插值目标特征向量
- $\varepsilon$: 数值稳定项

### 公式 6: [[Multi-task Learning|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{WAM}} + \lambda_{\text{agra}} \cdot \mathcal{L}_{\text{AGRA}}
$$

**含义**: 联合优化 WAM 原始损失（视频重建 + 动作预测）和 AGRA 表征对齐损失。

**符号说明**:
- $\mathcal{L}_{\text{WAM}}$: 原始 WAM 损失（视频扩散 + 动作解码）
- $\lambda_{\text{agra}}$: AGRA 损失权重超参数

---

## 关键图表

### Figure 1: 论文 Teaser — 问题诊断与方法概览

![Figure 1 Teaser](https://xpeng-robotics.github.io/agra/figures/teaser.png)

**说明**: 左侧展示基线 WAM 中正确的未来预测不能保证可靠控制（动作解码器关注背景），右侧展示 AGRA 如何通过与冻结基础编码器对齐来弥补 action-grounding gap。

### Figure 2: Action-Grounding Gap 诊断可视化

![Figure 2 Diagnosis](https://xpeng-robotics.github.io/agra/figures/diagnosis.png)

**说明**: 展示基线 WAM 动作解码器的 Cross-Attention Map（错误集中于背景）以及因果干预热图，揭示对任务无关区域的高敏感性。AGRA 修正后注意力集中于手-物体交互区域。

### Figure 3: DINOv2 vs Cosmos 特征 PCA 可视化

![Figure 3 PCA](https://xpeng-robotics.github.io/agra/figures/pca.png)

**说明**: 对比 DINOv2 和 Cosmos-Predict-2.5 表征的 PCA 主成分。DINOv2 呈现对象中心、空间连贯的结构（同类物体聚集），而 Cosmos 表征因优化像素重建而呈现弥散的外观细节结构。

### Figure 4: AGRA 模型架构

![Figure 4 Architecture](https://xpeng-robotics.github.io/agra/figures/architecture.png)

**说明**: 左为基线 WAM 架构（视频 DiT 隐状态直接注入动作解码器），右为 AGRA 在世界-动作接口添加 DINOv2 对齐分支，以及多层桥接 + 训练-推理一致性设计。

### Figure 5: Attention 与因果干预分析结果

![Figure 5 Exp Attention](https://xpeng-robotics.github.io/agra/figures/exp_attn.png)

**说明**: 定性展示 AGRA 如何将动作头注意力引导至任务关键的手-物体交互区域，基线则散布于无关背景。同时展示因果干预前后的敏感性热图对比。

### Table 1: 因果敏感性比率对比

| 干预类型 | WAM 基线 | AGRA |
|---------|---------|------|
| Mean 替换 | 8.41 | 10.31 |
| Zero 替换 | 1.95 | 2.82 |
| Shuffle 随机 | 0.99 | 1.01 |
| Swap 前景-背景 | 8.36 | 10.19 |

**关键发现**: AGRA 对前景（手/物体）区域的干预敏感性更高（Mean/Swap 比率提升），说明动作解码器更聚焦于任务关键区域；对背景 Shuffle 的低敏感性（~1.0）证明其对任务无关扰动的鲁棒性。

### Table 2: 消融实验（RoboCasa GR1 仿真，few-shot）

| 变体 | ID 成功率 | 未见外观 | 未见物体 | 未见组合 |
|-----|---------|---------|---------|---------|
| Freeze backbone | 52.89 | 49.33 | 43.87 | 51.42 |
| WAM（基线） | 58.41 | 53.77 | 43.18 | 59.57 |
| AGRA-SharedDenoisingPass | 58.24 | 57.77 | 41.87 | 58.57 |
| AGRA-VideoPass | 60.41 | 55.33 | 44.56 | 60.14 |
| **AGRA-ActionCondPass（完整版）** | **61.75** | **56.55** | **45.31** | **66.28** |

**关键发现**: 训练-推理一致性（ActionCondPass）至关重要；SharedDenoisingPass（推理时与训练时使用不同噪声通道）甚至不如基线 WAM，证明分布一致性的重要性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人数据（XPENG） | 未公开 | IRON-R01-1.11，双臂+手+腰 | 训练+测试 |
| RoboCasa GR1 仿真 | 24 个任务 | 人形机器人，多样化桌面操作 | 仿真验证+消融 |

### 任务设置

**真实机器人任务**:
- Pick-and-Place：根据语言指令抓取目标物体，放入容器
- Open-Steamer-Transfer-Bun：拾取物品、打开容器、转移内容物

**真实机器人评估场景**:
- In-Distribution（ID）：标准评估
- Semantic Generalization：同类别新物体
- Instance-Level Generalization：见过类别的不同实例
- Attribute Generalization：背景、光照等环境变化

### 实现细节

- **视频骨干**: Cosmos-Predict-2.5-2B（28层），输入分辨率 192×336
- **时序压缩**: 比例 4，5 个潜在视频帧
- **动作头**: 8层 Transformer，500M 参数
- **AGRA 对齐层**: Layer 8（约 1/3 深度），多层桥接时均匀采样
- **语义教师**: 冻结 DINOv2（优于 SigLIP，因空间连贯性更好）
- **仿真对比基线**: GR00T-N1.6、UWM、Diffusion Policy、FLARE、DiT4DiT、LDA-1B

### 主要结果

**真实机器人（IRON-R01-1.11）**:

| 场景 | WAM 基线 | AGRA | 提升 |
|-----|---------|------|-----|
| In-Distribution | 34% | 80% | +46% |
| Semantic OOD | 53% | 80% | +27% |
| Instance OOD | 48% | 80% | +32% |
| Attribute OOD | 48% | 80% | +32% |

**仿真（RoboCasa GR1，全数据集）**:

| 方法 | 总体成功率 |
|-----|---------|
| GR00T-N1.6（最强 VLA 基线） | 47.6% |
| **AGRA** | **66.4%** |
| 提升 | +18.8% |

### 关键消融发现

1. **DINOv2 vs SigLIP**: DINOv2 显著更优（对象中心、空间连贯），SigLIP 特征更弥散
2. **Layer 8 vs 其他层**: 第 8 层（~1/3 深度）最优；更深层（Layer 15+）干扰运动动态建模
3. **多层 vs 单层桥接**: 多层一致优于单层，验证层级信息融合的必要性
4. **训练-推理一致性**: $\tau_v^{\text{cond}} = 1$ 固定高噪声前向传播是关键设计

---

## 批判性思考

### 优点

1. **问题定位精准**: 通过 Attention 分析 + 因果干预两种独立手段诊断 action-grounding gap，证据充分
2. **方法简洁高效**: 仅添加一个对齐损失分支，无需修改 WAM 基础架构，可插拔性强
3. **实验结果显著**: 真实机器人 +46% ID 提升、仿真 +18.8% 超越强基线，说服力强
4. **洞察深刻**: 训练-推理一致性的设计（固定 $\tau_v^{\text{cond}}=1$）是非显而易见但关键的工程点

### 局限性

1. **任务多样性有限**: 真实机器人实验仅涉及 2 类操作任务，泛化性待验证
2. **video-action mismatch 未完全解决**: 论文自承仅部分缓解，视频预测精度和动作精度的耦合问题依然存在
3. **无开源代码**: 可复现性受限，XPENG 内部数据集也无法公开获取
4. **DINOv2 依赖**: 效果强依赖 DINOv2 的语义质量，对 DINOv2 表征不擅长的场景（如精细纹理辨别）可能失效

### 潜在改进方向

1. 探索更强的空间语义教师（如 SAM 的 mask encoder）以进一步提升 affordance 理解
2. 将 AGRA 扩展到在线 RL fine-tuning 场景，验证对齐在探索阶段的效果
3. 研究动态层选择策略（根据任务难度自适应选择对齐层）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（消融实验充分）
- [ ] 数据集可获取（RoboCasa 公开，真实数据未公开）

---

## 关联笔记

### 基于

- [[World Action Model]]: AGRA 建立在 WAM 框架之上
- [[Cosmos-Predict]]: 使用 Cosmos-Predict-2.5-2B 作为视频骨干
- [[DINOv2]]: 冻结 DINOv2 提供语义对齐目标

### 对比

- [[GR00T]]: 仿真实验中超越 GR00T-N1.6 18.8 个百分点
- [[Flash-WAM]]: 同为 WAM 优化方向，关注推理效率

### 方法相关

- [[Representation Alignment]]: 核心技术——跨模态表征对齐
- [[Video Diffusion Model]]: 视频预测骨干
- [[Cross-Attention]]: 世界-动作接口的核心机制
- [[Action Chunking]]: 动作预测的时域分块策略
- [[Cosine Similarity]]: AGRA 损失函数形式

### 硬件/数据相关

- [[IRON-R01]]: XPENG 人形机器人平台，双臂+手+腰控制
- [[RoboCasa]]: 仿真验证平台，24 个 GR1 人形机器人任务

---

## 速查卡片

> [!summary] AGRA — Making Foresight Actionable
> - **核心**: 视频预测≠可控动作，需用 DINOv2 语义对齐弥合 action-grounding gap
> - **方法**: AGRA 损失 = 余弦相似度对齐视频 DiT Layer 8 隐状态至冻结 DINOv2 特征
> - **结果**: 真实机器人 ID 成功率 34%→80%；仿真超 GR00T +18.8%
> - **代码**: 未开源（项目主页：xpeng-robotics.github.io/agra）

---

*笔记创建时间: 2026-06-12*

---
title: "Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models"
method_name: "Mind-VLA"
authors: [Xingyu Ding, Yuzhong Zhao, Yang Wu, Chaoyang Zhao, Chunhai Zhao, Yifan Zhang, Jian Cheng]
year: 2026
venue: arXiv
tags: [vla, 3d-spatial-representation, instruction-aware, robot-manipulation, diffusion-policy, auxiliary-supervision, occlusion-robustness]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.04633
created: 2026-08-07
---

# 论文笔记：Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未注明（作者来自多家机构） |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[SpatialVLA]] · [[GLaD]] · [[CogACT]] · [[Seer]] · [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.04633) / Code: — |

---

## 一句话总结

> Mind-VLA 将 3D 表示对齐从场景级提升为语言指令感知级，通过对目标物体三视图的 VAE 潜变量预测与 VGGT 多层特征对齐，用 345M 参数骨干在 LIBERO 和 CALVIN 上达到媲美 7B 级 VLA 的性能，并在遮挡场景下表现出显著鲁棒性。

---

## 核心贡献

1. **指令感知问题识别**: 明确指出现有 3D-aware VLA 方法（如 Spatial Forcing、GLaD）的"指令无关"本质——它们对整个场景均匀对齐，忽视语言指令所指定的目标物体几何信息。
2. **目标物体三视图对齐**: 提出以目标物体正交三视图（顶/正/侧）的 VAE 潜变量预测与 VGGT 多层余弦对齐取代全场景对齐，训练阶段引入、推理阶段移除辅助模块，零推理开销。
3. **遮挡鲁棒性验证**: 在真实机器人 25% 目标遮挡条件下，Mind-VLA 性能仅下降 13 pp（67→54%），而场景级 VGGT 对齐方案下降 29 pp，定量证明指令感知 3D 监督的鲁棒性收益。

---

## 问题背景

### 要解决的问题

现有 [[Vision-Language-Action|VLA]] 模型在执行细粒度操作任务时，难以精准感知语言指令所指目标物体的三维几何结构，尤其在目标被遮挡时成功率急剧下降。

### 现有方法的局限

已有的 3D-aware VLA 方法分两类：

- **显式 3D 输入**（如 3D Diff. Actor）：提供点云/深度图/正交视图作为额外输入，但需要深度传感器且对整体场景编码，不针对目标物体。
- **训练时 3D 监督**（如 [[Spatial Forcing]]、GLaD、QDepth-VLA）：将 VLA 特征与场景级 [[VGGT]] 特征对齐，避免推理时额外开销，但监督信号仍是**指令无关**的全场景几何。

两类方法共同缺陷：对齐目标不随语言指令变化，"treating all tasks uniformly"。

### 本文的动机

通过将对齐目标替换为语言指令指定的目标物体三视图，使 [[辅助监督|auxiliary supervision]] 成为指令条件化的——当指令变化、目标物体变化时，几何监督信号随之变化，从而让骨干网络真正学会"看指令、看目标"。

---

## 方法详解

### 模型架构

Mind-VLA 采用 **Transformer + 扩散动作头** 架构：

- **输入**: 语言指令 $\ell$（[[CLIP]] 文本编码）+ 观测 $o_t$（两路 RGB：主摄像头 + 腕部摄像头，冻结视觉编码器）+ 状态 $s_t$（8 维本体感知，线性投影）
- **Backbone**: 345M 参数因果 [[Transformer]]，$K$ 个时间步展开
- **核心模块**:
  - [[Scene Query|场景查询]] → 逐 patch RGB 重建 + 稠密 2D 运动预测
  - [[Object Query|目标物体查询]] → 三视图 [[VAE]] 潜变量预测
  - [[Action Query|动作查询]] → 条件化 [[Diffusion Transformer]] 预测未来动作
- **输出**: 未来 $T_a$ 步 7 维动作
- **总参数**: 345M（推理时辅助分支全部移除）

架构注意力掩码强制时序因果性，并将不同查询组隔离（见 Figure 2 右侧）。

### 核心模块

#### 模块 1: 目标物体三视图准备（Target-Object Tri-View）

**设计动机**: 利用 [[正交投影|Orthographic Projection]] 从固定视角捕获目标物体完整几何，避免因当前观测视角变化或遮挡导致几何信息缺失。

**具体实现**:
- **仿真环境**: 离线从物体 mesh 渲染顶视、正视、侧视三张正交图像
- **真实机器人**: 每个目标物体仅需一次手持相机拍摄，近似获取三个规范视角
- 三视图通过任务定义与语言指令绑定，推理时无论视角如何变化，监督目标始终对应正确物体

#### 模块 2: 三视图 VAE 潜变量预测（Tri-View Latent Prediction）

**设计动机**: 在紧凑的潜空间（原始 RGB 的 $\frac{1}{48}$）监督目标物体外观与几何，降低监督维度、加速收敛。

**具体实现**:
- 冻结 [[Stable Diffusion VAE]] 将三视图离线编码为堆叠潜变量 $\mathbf{Z}_{m(\ell)}$（$m(\ell)$ 表示指令 $\ell$ 对应的物体）
- 目标物体查询 token 经 MLP 解码为预测潜变量 $\hat{\mathbf{Z}}_t$
- MSE 损失在潜空间监督

#### 模块 3: 多层 VGGT 几何对齐（Multi-Level VGGT Alignment）

**设计动机**: 不仅在输出层监督潜变量，还在骨干中间层植入目标几何先验，使特征层次结构从底层就感知目标物体三维结构。

**具体实现**:
- 冻结 [[VGGT]] 提取三视图在 4 个中间层的几何特征 $\mathbf{g}_{m(\ell),j}$（对视角和空间位置做 mean-pool）
- 骨干对应层的图像 token 特征 $\bar{\mathbf{h}}_{t,j}$ 经投影头 $\phi_j$ 映射到 VGGT 空间
- 余弦距离损失在 4 层 × $K$ 时间步上求平均
- 与 Spatial Forcing 的本质区别：对齐目标是**目标物体三视图特征**，而非全场景图像特征

---

## 关键公式

### 公式 1: [[Diffusion Policy|动作扩散损失]]

$$
\mathcal{L}_{\mathrm{act}} = \mathbb{E}_{t, \tau, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}\!\left( \sqrt{\bar{\alpha}_{\tau}} \mathbf{A}_{t} + \sqrt{1 - \bar{\alpha}_{\tau}} \boldsymbol{\epsilon},\; \tau,\; \mathbf{H}_{t}^{a} \right) \right\|_{2}^{2} \right]
$$

**含义**: 扩散动作头的噪声预测损失，骨干输出动作查询特征 $\mathbf{H}_{t}^{a}$ 作为条件。

**符号说明**:
- $t$: 环境时间步
- $\tau$: 扩散噪声时间步（从 $\mathcal{U}(0,1)$ 均匀采样）
- $\boldsymbol{\epsilon}$: 标准高斯噪声
- $\bar{\alpha}_{\tau}$: 扩散调度的累积噪声比例
- $\mathbf{A}_{t}$: 未来 $T_a$ 步动作块
- $\boldsymbol{\epsilon}_{\theta}$: 可学习的噪声预测网络（[[Diffusion Transformer]]）
- $\mathbf{H}_{t}^{a}$: 骨干最终层动作查询输出（扩散条件）

---

### 公式 2: [[辅助监督|VLA 训练总目标]]

$$
\mathcal{L}_{\mathrm{VLA}} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{aux}} \mathcal{L}_{\mathrm{aux}}
$$

**含义**: 动作损失加权叠加辅助损失（图像重建 $\mathcal{L}_{\mathrm{obs}}$ + 稠密 2D 运动预测 $\mathcal{L}_{\mathrm{traj}}$），沿袭现有 3D-aware VLA 框架。

**符号说明**:
- $\lambda_{\mathrm{aux}}$: 辅助损失权重系数
- $\mathcal{L}_{\mathrm{aux}}$: 场景重建辅助损失（$\mathcal{L}_{\mathrm{obs}} + \mathcal{L}_{\mathrm{traj}}$）

---

### 公式 3: [[VAE|三视图潜变量预测损失]]

$$
\mathcal{L}_{\mathrm{tri}} = \frac{1}{K} \sum_{t=1}^{K} \left\| \hat{\mathbf{Z}}_{t} - \mathbf{Z}_{m(\ell)} \right\|_{2}^{2}
$$

**含义**: 目标物体查询预测的三视图 VAE 潜变量与离线编码的目标潜变量之间的 MSE，让模型学会将指令 $\ell$ 映射到对应物体的视觉-几何表征。

**符号说明**:
- $K$: 展开时间步数
- $\hat{\mathbf{Z}}_{t}$: 时间步 $t$ 的目标物体查询经 MLP 解码的预测潜变量
- $\mathbf{Z}_{m(\ell)}$: 指令 $\ell$ 对应目标物体 $m(\ell)$ 三视图的离线 VAE 编码（固定目标）

---

### 公式 4: [[VGGT|多层几何对齐损失]]

$$
\mathcal{L}_{\mathrm{geo}} = \frac{1}{4K} \sum_{t=1}^{K} \sum_{j=1}^{4} \left( 1 - \frac{\phi_{j}(\bar{\mathbf{h}}_{t,j})^{\top} \mathbf{g}_{m(\ell),j}}{\left\| \phi_{j}(\bar{\mathbf{h}}_{t,j}) \right\|_{2} \left\| \mathbf{g}_{m(\ell),j} \right\|_{2}} \right)
$$

**含义**: 骨干 4 个中间层的图像 token 特征经投影后与目标物体三视图的 VGGT 几何特征做余弦距离，在 4 层 × $K$ 时间步上均匀对齐，将目标物体几何先验植入骨干中间表示。

**符号说明**:
- $j \in \{1,2,3,4\}$: 骨干中间层索引
- $\bar{\mathbf{h}}_{t,j}$: 时间步 $t$、第 $j$ 层的图像 token 特征（空间维度 mean-pool）
- $\phi_{j}$: 第 $j$ 层的可学习线性投影头，将骨干特征映射到 VGGT 特征空间
- $\mathbf{g}_{m(\ell),j}$: 目标物体 $m(\ell)$ 三视图在 VGGT 第 $j$ 层的特征（对视角和空间维度 mean-pool，固定）

---

### 公式 5: [[指令感知 3D 对齐|Mind-VLA 最终训练目标]]

$$
\mathcal{L}_{\mathrm{MindVLA}} = \mathcal{L}_{\mathrm{VLA}} + \lambda_{\mathrm{tri}} \mathcal{L}_{\mathrm{tri}} + \lambda_{\mathrm{geo}} \mathcal{L}_{\mathrm{geo}}
$$

**含义**: 将动作损失、辅助重建损失、三视图潜变量预测损失、多层 VGGT 几何对齐损失统一为一个目标函数；$\lambda_{\mathrm{tri}}$ 和 $\lambda_{\mathrm{geo}}$ 平衡各项权重。推理时仅保留 $\mathcal{L}_{\mathrm{act}}$ 相关模块。

**符号说明**:
- $\lambda_{\mathrm{tri}}$: 三视图潜变量损失权重
- $\lambda_{\mathrm{geo}}$: 多层 VGGT 对齐损失权重

---

## 关键图表

### Figure 1: 3D-aware VLA 范式对比

![Figure 1](https://arxiv.org/html/2608.04633v1/x1.png)

**说明**: 对比三种范式。(a) 提供指令无关的 3D 特征作为额外模型输入；(b) 训练时将 VLA 表示与指令无关的 3D 特征对齐（如 [[Spatial Forcing]]、GLaD）；(c) Mind-VLA 将 VLA 表示与**指令感知**的目标物体 3D 特征对齐。核心区别：监督信号是否随语言指令变化。

---

### Figure 2: Mind-VLA 整体架构

![Figure 2](https://arxiv.org/html/2608.04633v1/x2.png)

**说明**: 语言指令、状态、观测 token 化后与三组可学习查询（Scene / Object / Action）一起经 Transformer 骨干处理。场景查询解码为逐 patch RGB 和稠密 2D 运动；目标物体查询解码为三视图 VAE 潜变量；骨干同时与从相同三视图提取的 VGGT 特征在 4 个中间层对齐；动作查询条件化 [[Diffusion Transformer]] 预测未来三步动作。右侧注意力掩码示意图展示时序因果性与查询组隔离。**辅助分支（蓝色）仅训练时使用**。

---

### Figure 3: LIBERO 与紧凑骨干基线对比

![Figure 3](https://arxiv.org/html/2608.04633v1/x3.png)

**说明**: Mind-VLA 在 LIBERO 四套任务上与紧凑骨干基线的逐套成功率对比。Mind-VLA 在 Spatial（98.0%）和 Object（98.0%）套上优势最明显，Long 套（87.5%）是唯一明显劣于部分基线的任务套。

---

### Figure 4: 真实机器人实验场景

![Figure 4](https://arxiv.org/html/2608.04633v1/x4.png)

**说明**: (a) 双臂 UFactory xArm 6 工作空间（单臂使用），配备固定摄像头和腕部摄像头 RealSense D455/D435；(b) 正常条件下三类任务：Pick（抓取指定物体）、Place（放置物体）、Drawer（接触丰富的抽屉操作）；(c) 遮挡条件（约 25% 目标被物理遮挡物覆盖，红色高亮标注）。

---

### Figure 5: 25% 遮挡下的真实机器人鲁棒性

![Figure 5](https://arxiv.org/html/2608.04633v1/x5.png)

**说明**: Mind-VLA 在 25% 目标遮挡时性能仅下降 13 pp，而场景图像 VGGT 消融版本下降 29 pp。直接证明指令感知的目标物体对齐（而非单纯目标聚焦）是遮挡鲁棒性的来源。

---

### Table 1a: LIBERO 基准——与 7B 级大型 VLA 对比（成功率 %）

| 方法 | 参数量 | Spatial | Object | Goal | Long | Avg. |
|------|--------|---------|--------|------|------|------|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TraceVLA | 7B | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| SpatialVLA | 4B | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 7B | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| CogACT | 7B | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 |
| GLaD | 7B | 95.0 | 97.4 | 94.4 | 89.4 | 94.1 |
| π₀ | 3B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| **Mind-VLA (Ours)** | **345M** | **98.0** | **98.0** | **92.0** | **87.5** | **93.9** |

**关键发现**: 345M 参数骨干以 20–99× 参数优势达到 93.9% 平均成功率，媲美 3B–7B 级 VLA；Spatial 和 Object 套甚至超越所有对比方法。

---

### Table 1b: CALVIN ABC-D——平均完成步骤数（1000 rollout）

| 方法 | T5 (%) | Avg. Len. |
|------|--------|-----------|
| 3D Diff. Actor | 41.2 | 3.27 |
| OpenVLA† | 43.5 | 3.27 |
| CLOVER | 45.4 | 3.53 |
| RoboDual | 54.4 | 3.66 |
| π₀† | 59.9 | 3.92 |
| Seer | 74.0 | 4.28 |
| VPP | 75.0 | 4.29 |
| **Mind-VLA (Ours)** | **79.4** | **4.47** |

**关键发现**: Mind-VLA 在 CALVIN ABC-D 上以 4.47 的平均完成步骤数达到 SOTA，T5（5 步连续完成率）同样最高（79.4%）。

---

### Table 2: 消融实验——逐步构建（LIBERO）

**渐进组件添加（基准→完整模型）**

| ID | 配置 | Spatial | Object | Goal | Long | Avg. |
|----|------|---------|--------|------|------|------|
| A0 | 基准（VLA + 图像重建） | 95.5 | 92.0 | 88.0 | 82.5 | 89.5 |
| A1 | + 场景级轨迹预测 | 97.0 | 93.5 | 89.5 | 85.0 | 91.3 |
| A2 | + 目标物体三视图潜变量预测 | 97.5 | 95.0 | 91.0 | 86.5 | 92.5 |

**多层几何对齐对比（在 A2 基础上）**

| ID | 配置 | Spatial | Object | Goal | Long | Avg. |
|----|------|---------|--------|------|------|------|
| A3a | + 场景图像 VGGT（指令无关） | 97.5 | 94.0 | 89.5 | 90.0 | 92.8 |
| A3b | + 目标物体三视图 VGGT（完整 Mind-VLA） | **98.0** | **98.0** | **92.0** | 87.5 | **93.9** |

**关键发现**: 指令感知的目标物体三视图对齐（A3b）在 Object 套上比指令无关的场景图像对齐（A3a）高 +4.0 pp；但 A3a 在 Long 套更优（90.0 vs 87.5），说明长时域任务更依赖全局场景上下文。

---

### Table 3: 真实机器人逐任务成功率（%）

| 方法 | Banana (Pick) | Potato (Pick) | Drawer | Banana (Place) | Potato (Place) |
|------|:---:|:---:|:---:|:---:|:---:|
| **正常条件** | | | | | |
| OpenVLA | 33 | 10 | 40 | 27 | 20 |
| Seer | 43 | 30 | 63 | 67 | 43 |
| Mind-VLA (scene-image VGGT) | 63 | 37 | 70 | 70 | 47 |
| **Mind-VLA (Ours)** | **70** | **57** | **73** | **80** | **63** |
| **遮挡条件（约 25%）** | | | | | |
| OpenVLA | 3 | 7 | 10 | — | — |
| Seer | 23 | 13 | 30 | — | — |
| Mind-VLA (scene-image VGGT) | 27 | 20 | 37 | — | — |
| **Mind-VLA (Ours)** | **57** | **43** | **63** | — | — |

**关键发现**: 遮挡条件下 Mind-VLA 成功率（约 54% 平均）比场景级消融（约 28%）高出一倍；Seer 在遮挡下从 45% 跌至 22%，OpenVLA 从 28% 跌至 7%。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套 × 10 任务，各 50 demo | 仿真操作，空间/物体/目标/长时域子套 | 基准评测（20 rollout/任务） |
| [[CALVIN]] ABC-D | 训练 ABC 场景，测试 D 场景 | 长时域链式操作，1000 rollout | 基准评测（平均完成步骤数） |
| [[DROID]] | 大规模真实机器人数据集 | 多样化操作，用于预训练 | 真实机器人预训练 |
| 任务特定真实数据 | 少量 demo | 单臂 xArm 6，Pick/Place/Drawer | 真实机器人微调 |

### 实现细节

- **Backbone**: 345M 参数因果 Transformer
- **视觉编码器**: 冻结（具体型号未注明）
- **文本编码器**: 冻结 [[CLIP]]
- **几何特征提取**: 冻结 [[VGGT]]（4 个中间层）
- **图像生成 VAE**: 冻结 [[Stable Diffusion]] VAE
- **优化器**: AdamW
- **硬件**: 8× NVIDIA H20 GPU
- **真实机器人**: UFactory xArm 6 + RealSense D455（固定） + RealSense D435（腕部）

---

## 批判性思考

### 优点

1. **参数效率突出**: 以 345M 参数骨干在 LIBERO（93.9%）和 CALVIN（4.47）上媲美 3B–7B 级 VLA，展现极高参数效率。
2. **零推理开销**: 所有辅助模块（VAE、VGGT、辅助解码器）仅训练时使用，推理时完全移除，不增加延迟。
3. **遮挡鲁棒性定量验证**: 通过严格对照实验证明，13 pp vs 29 pp 的遮挡性能差异直接来自指令感知对齐，而非简单目标聚焦。
4. **SAM 消融设计精巧**: 通过比较 SAM crop（指令感知但无规范视角先验）与三视图（指令感知+规范视角），分解"物体聚焦"和"规范视角先验"两个因素的独立贡献。

### 局限性

1. **闭集物体词汇**: 每个目标物体需提前准备三视图（仿真渲染或真实拍摄），无法零样本部署到未见物体。
2. **长时域任务劣势**: LIBERO-Long 87.5% 低于 DreamVLA 89.5%；单目标对焦策略可能丢失对长时域任务有益的全局场景上下文。
3. **单目标限制**: 每条指令仅映射一个目标物体；多目标/关系指令（"把 A 放到 B 上"中需同时感知 A 和 B）尚未解决。
4. **真实机器人评测规模有限**: 仅 3 类任务、2 种物体、30 次 trial，泛化性有待更广泛验证。

### 潜在改进方向

1. **开放词汇三视图生成**: 结合 3D 生成模型（如 Zero123、MVDiffusion）从单张图片生成三视图，打破闭集限制。
2. **多目标感知**: 扩展至多目标查询，或动态选择关注对象（如通过注意力权重）以支持关系型指令。
3. **Long-horizon 设计**: 引入任务规划模块，在关键子目标切换时动态更新目标物体查询。

### 可复现性评估

- [ ] 代码开源（未见发布）
- [ ] 预训练模型（未见发布）
- [x] 训练细节完整（损失函数、架构、硬件均有说明）
- [x] 数据集可获取（LIBERO、CALVIN、DROID 均为公开数据集）

---

## 关联笔记

### 基于

- [[Spatial Forcing]]: 提出场景级 VGGT 对齐的前驱方法，Mind-VLA 将其扩展为指令感知
- [[VGGT]]: 提供多层 3D 几何特征的视觉基础模型，Mind-VLA 将其作为冻结几何教师
- [[Diffusion Policy]]: 动作头采用扩散范式，沿袭自此
- [[Stable Diffusion VAE]]: 提供高压缩比的图像潜空间编码器

### 对比

- [[GLaD]]: 场景级几何蒸馏 VLA，7B 参数达 94.1%，Mind-VLA 以 345M 逼近
- [[CogACT]]: 7B VLA，LIBERO 93.6%，参数量为 Mind-VLA 的 20×
- [[OpenVLA]]: 7B 基线，LIBERO 76.5%，真实机器人对比基线
- [[Seer]]: 紧凑骨干 VLA，CALVIN 4.28，真实机器人对比基线

### 方法相关

- [[Vision-Language-Action]]: 核心研究范式
- [[辅助监督]]: 训练时辅助目标的设计思路
- [[正交投影]]: 三视图的几何基础
- [[CLIP]]: 文本编码器
- [[Diffusion Transformer]]: 动作解码模块

### 硬件/数据相关

- [[LIBERO]]: 主要仿真基准
- [[CALVIN]]: 长时域链式操作基准
- [[DROID]]: 真实机器人预训练数据集

---

## 速查卡片

> [!summary] Mind-VLA (2026)
> - **核心**: 将 3D 表示对齐从场景级提升为指令感知（目标物体三视图），填补现有方法"指令无关"缺陷
> - **方法**: 目标物体三视图 VAE 潜变量预测 + 4 层 VGGT 余弦对齐，辅助模块仅训练时使用
> - **结果**: LIBERO 93.9%、CALVIN 4.47，345M 参数；真实遮挡场景仅降 13 pp vs 基线降 29 pp
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-07*

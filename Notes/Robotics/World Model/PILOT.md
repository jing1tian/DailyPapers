---
title: "Decoupling Intention from Trajectory: A Representational Deduction Framework for World Action Models"
method_name: "PILOT"
authors: [Xiangkai Ma, Yue Ma, Junjie Wang, Sheng Xu, Mingyang Li, Han Zhang, Yuzheng Zhuang, Wenzhong Li, Zhihao Yuan]
year: 2026
venue: arXiv
tags: [world-action-model, motion-intention, representational-learning, flow-matching, robot-manipulation, humanoid]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.06994
created: 2026-08-11
---

# 论文笔记：PILOT — Decoupling Intention from Trajectory

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未知（arXiv 页面未列出） |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.06994) |

---

## 一句话总结

> PILOT 通过将高层运动意图（[[Motion-CoT]]）与低层轨迹生成解耦，并引入 [[Causal Dynamics Engine]] 在表征空间而非像素空间监督未来状态预测，将 WAM 的成功率与泛化能力大幅提升。

---

## 核心贡献

1. **[[Representational Deduction]] 框架**: 将策略分解为先生成运动语义潜变量、再条件生成动作轨迹的两阶段结构，消除 WAM 中高层语义与低层轨迹的表征纠缠
2. **[[Motion-CoT]] 机制**: 用 K=64 个可学习查询 token 从视觉-语言上下文中蒸馏出运动意图，并通过 [[Causally-Decoupled Attention]] 将其作为推理链引导动作生成，推理时无需生成像素级未来帧（推理延迟降低 90%）
3. **[[Causal Dynamics Engine]]（CDE）**: 以冻结的 [[V-JEPA2|VJEPA2-AC]] 编码器为基础，在表征空间（而非像素空间）预测未来状态，强化了 Motion-CoT 对物理状态演变的感知能力

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]]（WAM）存在**表征纠缠**（representational entanglement）：在同一模块中混合了"物理状态应如何演变"（高层意图）与"如何生成精细运动指令"（低层轨迹），导致预测性能受限，且在推理时需要昂贵的像素级未来帧生成。

### 现有方法的局限

- **静态视觉表征**无法反映动作交互下物理状态的动态演变过程
- **predict-then-act 范式**（先生成未来帧再生成动作）的推理延迟极高，且未来帧质量直接制约动作质量
- 现有 WAM 对高层语义与低层轨迹之间的信息流缺乏显式建模，导致 Motion-CoT 信号被扩散噪声污染

### 本文的动机

通过在**表征空间**（而非像素空间）建模物理状态的因果演变，PILOT 能以极低的推理开销提供可靠的运动意图信号——Motion-CoT token 在 t-SNE 分析中自然形成按动作类型聚类的结构，无需显式监督即能分离语义与背景信息。

---

## 方法详解

### 整体架构

PILOT 采用**因果分解**架构，将策略条件概率分解为：

$$
p_\Theta(\mathbf{a}_t | o_t, \ell, s_t) = \int p_\Theta(\mathbf{a}_t | \mathbf{m}, s_t) \cdot p_\Theta(\mathbf{m} | o_t, \ell, s_t) \, d\mathbf{m}
$$

- **输入**: 语言指令 $\ell$ + 视觉观测 $o_t$ + 本体状态 $s_t$
- **World Model（骨干网络）**: 编码 $(o_t, \ell)$ 为上下文序列 $\mathbf{c}$
- **核心模块**: [[Motion-CoT]] + [[Causally-Decoupled Attention]] + [[Causal Dynamics Engine]]
- **输出**: [[Action Chunking|动作块]] $\mathbf{a}_{t:t+H}$（H 步动作序列）
- **总参数**: 5.4B

### 核心模块

#### 模块1：World Model 上下文编码

**设计动机**: 利用预训练视觉-语言对齐能力，将 $(o_t, \ell)$ 压缩为紧凑语义上下文

**具体实现**:

$$
\mathbf{c} = \text{WM}_\theta(o_t, \ell) \in \mathbb{R}^{N_c \times d_c}
$$

- $N_c$: 上下文 token 数量
- $d_c$: 特征维度

辅助任务：World Model 还通过 [[Flow Matching]] 损失预测未来帧潜变量，保留指令接地的未来状态先验：

$$
\mathcal{L}_{wm} = \mathbb{E}_{(\tau, \epsilon)} \left\| \text{WM}_\theta(\mathbf{z}^\tau; o_t, \ell, \tau) - (\epsilon - \mathbf{z}^*) \right\|^2_{\odot \mathbf{M}_{\mathbf{z}^\tau}}
$$

#### 模块2：[[Motion-CoT]] 与 [[Causally-Decoupled Attention]]

**设计动机**: 将运动意图作为中间推理链（Chain-of-Thought），防止扩散噪声污染语义表征

**具体实现**:

Action Perceiver 输入由三类 token 组成：
- 非动作语义 token（Motion-CoT 查询）$\mathbf{Z} \in \mathbb{R}^{K \times d}$（K=64）
- 带噪动作 token $\mathbf{A}^\tau \in \mathbb{R}^{H \times d}$

[[Causally-Decoupled Attention]] 的非对称信息流：

**非动作 token**（仅见上下文，不见带噪动作）：

$$
\mathbf{Z} \leftarrow \mathbf{Z} + \text{Attn}(\mathbf{Z},\; [\mathbf{c};\mathbf{Z}])
$$

**动作 token**（额外访问运动语义）：

$$
\mathbf{A}^\tau \leftarrow \mathbf{A}^\tau + \text{Attn}(\mathbf{A}^\tau,\; [\mathbf{c};\mathbf{Z};\mathbf{A}^\tau])
$$

Perceiver 输出与 Motion-CoT 提取：

$$
\mathbf{O} = \text{Act.Per.}(\mathbf{Z}, \mathbf{A}^\tau, \mathbf{c};\tau) \in \mathbb{R}^{(1+K+H) \times d}
$$

$$
\mathbf{m} = \mathbf{O}[:, 1{:}K+1, :] \in \mathbb{R}^{K \times d}
$$

动作生成损失（[[Flow Matching|条件流匹配]]）：

$$
\mathcal{L}_{act} = \mathbb{E}_{(\tau,\epsilon)} \left\| \text{Proj}_{act}\!\left(\mathbf{O}[:, K+1{:}, :]\right) - (\mathbf{a}_t - \epsilon) \right\|^2
$$

#### 模块3：[[Causal Dynamics Engine]]（CDE）

**设计动机**: 在 JEPA 表征空间（而非像素空间）监督物理状态的因果演变，减少对无关背景的计算消耗

**具体实现**:

冻结 [[V-JEPA2|VJEPA2-AC]] 编码器 $E_\psi$ 提取当前与未来帧的 patch 级表征：

$$
\mathbf{r}_t = E_\psi(o_t), \quad \mathbf{r}_{t+\Delta} = E_\psi(o_{t+\Delta})
$$

线性投影将 Motion-CoT 与本体状态映射到 VJEPA2 潜空间：

$$
\tilde{\mathbf{m}} = \mathbf{W}_m \mathbf{m} \in \mathbb{R}^{K \times d_r}, \quad \tilde{\mathbf{s}} = \mathbf{W}_s s_t \in \mathbb{R}^{1 \times d_r}
$$

CDE 预测未来表征（256×1408 维）：

$$
\hat{\mathbf{r}}_{t+\Delta} = G_\xi(\mathbf{r}_t;\, \tilde{\mathbf{m}},\, \tilde{\mathbf{s}}) \in \mathbb{R}^{P \times d_r}
$$

[[Representational Deduction]] 损失（SmoothL1）：

$$
\mathcal{L}_{RD} = \text{SmoothL1}(\hat{\mathbf{r}}_{t+\Delta},\, \mathbf{r}_{t+\Delta})
$$

### 总训练目标

$$
\mathcal{L} = \mathcal{L}_{act} + \lambda_{wm}\,\mathcal{L}_{wm} + \lambda_{RD}\,\mathcal{L}_{RD}
$$

**符号说明**:
- $\mathcal{L}_{act}$: 流匹配动作损失
- $\mathcal{L}_{wm}$: 世界模型辅助损失（保留未来帧先验）
- $\mathcal{L}_{RD}$: 表征推断损失（CDE 监督）
- $\lambda_{wm},\, \lambda_{RD}$: 权重系数

---

## 关键图表

### Figure 1: PILOT 框架概览

![Figure 1](https://arxiv.org/html/2608.06994v1/x1.png)

**说明**: PILOT 的整体框架，展示高层运动意图（[[Motion-CoT]]）与低层轨迹生成解耦的核心思想。

### Figure 2: 表征可视化（t-SNE 与 PCA）

![Figure 2](https://arxiv.org/html/2608.06994v1/x2.png)

**说明**: (a) t-SNE 分析显示 PILOT 的隐表征自然按动作类型聚类，而 baseline 方法则高度纠缠；(b) PCA 叠加到未来帧上，表明 CDE 将计算资源集中于交互区域（操作物体）而非静态背景。

### Figure 3: PILOT 模块结构详解

![Figure 3](https://arxiv.org/html/2608.06994v1/x3.png)

**说明**: 完整的 PILOT 架构，包含 World Model、带 [[Causally-Decoupled Attention]] 的 Action Perceiver，以及 [[Causal Dynamics Engine]] 三个核心组件的交互关系。

### Figure 4: 推理延迟对比与 RoboCasa-GR1 成功率

![Figure 4](https://arxiv.org/html/2608.06994v1/x4.png)

**说明**: 在 H20 GPU 上测试，[[Motion-CoT]] 使推理延迟降低 90%（相比 predict-then-act 范式），同时在 RoboCasa-GR1 上达到 62.6% 成功率。

### Figure 5: Agibot-G1 真实机器人评测

![Figure 5](https://arxiv.org/html/2608.06994v1/x5.png)

**说明**: PILOT 在 Agibot-G1 人形机器人上的真实世界评测套件，涵盖 8 个标准任务（平均 83.1%）、3 个泛化任务（平均 68.3%）和 10% 数据少样本微调场景（62.4%）。

### Figure 6: CDE 预测的 PCA 伪彩色可视化

![Figure 6](https://arxiv.org/html/2608.06994v1/x6.png)

**说明**: [[Causal Dynamics Engine]] 预测的未来表征 PCA RGB 可视化，表明模型激活集中在物体交互区域而非背景，验证了表征空间监督比像素空间监督更有利于感知物理交互。

### Figure 7: Motion-CoT Token 的 t-SNE 可视化

![Figure 7](https://arxiv.org/html/2608.06994v1/x7.png)

**说明**: RoboCasa-GR1 任务中 [[Motion-CoT]] token 嵌入的 t-SNE 图（按任务类别着色），显示不同任务的运动意图形成清晰分离的簇，无需显式任务标签监督。

---

### Table 1: LIBERO 评测结果

| Method | Size | Spatial | Object | Goal | Long | Avg. |
|--------|------|---------|--------|------|------|------|
| π₀.₅ | 3.5B | 98.8 | 98.2 | 98.0 | 92.4 | 96.2 |
| GROOT-N1.6 | 3.3B | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| LAPA | 7B | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA | 7B | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Mantis | 5.8B | 98.8 | 99.2 | 94.4 | 94.2 | 96.7 |
| VLA-JEPA | 3B | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| F1 | 4B | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| [[Fast-WAM]] | 6B | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Motus | 8B | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **PILOT（Ours）** | 5.4B | 97.1 | 99.2 | 98.1 | 97.2 | **97.9** |

**关键发现**: PILOT 以 5.4B 参数超越所有方法，LIBERO-Long 尤其突出（97.2 vs 95.2），表明长时序任务中解耦意图具有显著优势。

### Table 2: LIBERO-Plus 鲁棒性评测

| Method | Size | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|--------|------|--------|-------|----------|-------|------------|-------|--------|------|
| ECoT | 3B | 0.3 | 26.8 | 40.2 | 42.6 | 16.4 | 10.2 | 36.9 | 24.3 |
| Spatial Forcing | 3.3B | 20.1 | 13.4 | 40.9 | 29.1 | 33.4 | 25.7 | 39.3 | 29.1 |
| π₀-FAST | 3.3B | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | 7B | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| UniVLA | 7B | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| PokeVLA | 7B | 84.7 | 46.1 | 84.8 | 94.6 | 82.6 | 89.8 | 77.2 | 79.3 |
| WorldVLA | 7B | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| [[Fast-WAM]] | 6B | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| **PILOT（Ours）** | 5.4B | **76.0** | **70.0** | **84.0** | **92.0** | **85.0** | **78.0** | **82.0** | **81.0** |

**关键发现**: PILOT 在所有 7 种扰动类型均领先，相机角度变化（76.0 vs 16.4 Fast-WAM）和机器人形态变化（70.0 vs 44.5 Fast-WAM）的提升尤为显著，说明运动意图解耦增强了视觉鲁棒性。

### Table 3: [[RoboCasa]]-GR1 跨任务评测

| 任务类别 | GR00T N1.5 | GR00T N1.6 | DiT4DiT | UWM-XL | StarVLA | LDA | VP-VLA | PhysBrain | LangForce | [[Fast-WAM]] | Motus | **PILOT** |
|---------|-----------|-----------|---------|--------|---------|-----|--------|-----------|-----------|---------|-------|--------|
| Pick & Place（6 tasks）| 45.3 | 24.2 | 50.3 | 30.7 | 50.3 | 56.3 | 54.3 | 52.7 | 55.7 | 52.7 | 49.3 | **59.7** |
| Articulated from Cuttingboard（5 tasks）| 46.4 | 56.9 | 57.6 | 17.4 | 52.8 | 64.2 | 60.8 | 52.0 | 53.2 | 60.6 | 56.6 | **61.2** |
| Articulated from Placemat（4 tasks）| 45.5 | 51.9 | 39.0 | 10.0 | 38.0 | 47.8 | 54.5 | 48.0 | 48.0 | 47.2 | 43.8 | **56.0** |
| Articulated from Tray（5 tasks）| 48.8 | 55.1 | 46.4 | 17.2 | 39.2 | 53.4 | 46.0 | 46.0 | 47.2 | 54.8 | 50.8 | **74.8** |
| Articulated from Plate（4 tasks）| 56.5 | 57.5 | 60.0 | 16.2 | 58.5 | 53.0 | 53.5 | 62.5 | 58.5 | 58.5 | 54.5 | **60.5** |
| **平均** | 48.2 | 47.6 | 50.8 | 19.2 | 47.8 | 55.4 | 53.8 | 50.0 | 52.6 | 56.7 | 53.1 | **62.6** |

**关键发现**: PILOT 在所有 5 个任务类别全面领先，"从托盘拿取关节物体"任务上取得突破性的 74.8%（比次优 LDA 高 21.4 个百分点），凸显意图解耦对复杂关节操作任务的价值。

### Table 4: 真实机器人（Agibot-G1）评测

| Method | Size | Pen | Eraser | Corr. Fluid | Charger | Pencil Case | Stapler | Book | Trash | Stand. Avg. | Flash+Cam.Offs. | Desk | Color | Gen. Avg. | Few-Shot (10%) |
|--------|------|-----|--------|-------------|---------|-------------|---------|------|-------|------------|-----------------|------|-------|----------|---------------|
| π₀ | 3.3B | 70 | 82 | 46 | 78 | 32 | 62 | 52 | 72 | 61.8 | 38 | 42 | 36 | 38.7 | 28.4 |
| OpenVLA | 7B | 64 | 76 | 40 | 72 | 26 | 56 | 46 | 66 | 55.8 | 32 | 36 | 30 | 32.7 | 24.6 |
| GR00T-N1 | 3.4B | 74 | 86 | 50 | 82 | 36 | 66 | 56 | 76 | 65.8 | 42 | 46 | 40 | 42.7 | 32.8 |
| π₀.₅ | 3.3B | 80 | 90 | 56 | 86 | 42 | 72 | 62 | 82 | 71.3 | 48 | 52 | 44 | 48.0 | 38.2 |
| Cosmos-Policy | 6B | 76 | 88 | 52 | 84 | 38 | 68 | 58 | 78 | 67.8 | 44 | 48 | 42 | 44.7 | 34.6 |
| [[Fast-WAM]] | 6B | 82 | 92 | 58 | 88 | 44 | 74 | 64 | 84 | 73.3 | 50 | 54 | 46 | 50.0 | 40.8 |
| **PILOT（Ours）** | 5.4B | **90** | **100** | **70** | **94** | **56** | **88** | **72** | **94** | **83.1** | **72** | **68** | **64** | **68.3** | **62.4** |

**关键发现**: 泛化场景（68.3% vs Fast-WAM 50.0%，提升 +18.3pp）和少样本微调（62.4% vs 40.8%，提升 +21.6pp）的突出表现，证明 [[Motion-CoT]] 大幅降低了样本需求。

### Table 5: 累积组件消融实验

| 未来预测 | Motion-CoT | RD | CDE | 解耦注意力 | LIBERO Avg. | RoboCasa Avg. |
|---------|-----------|----|----|-----------|-------------|---------------|
| ✗ | ✗ | — | — | — | 91.3 | 51.2 |
| ✓ | ✗ | — | — | — | 93.6 | 54.8 |
| ✗ | ✓ | ✗ | ✗ | ✗ | 94.5 | 56.7 |
| ✗ | ✓ | ✓ | ✓ | ✗ | 96.1 | 59.8 |
| ✗ | ✓ | ✓ | ✓ | ✓ | 96.9 | 61.3 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **97.9** | **62.6** |

**关键发现**: 每个组件都有正向贡献：
- [[Motion-CoT]] 单独引入：LIBERO +3.2pp，RoboCasa +5.5pp
- [[Causal Dynamics Engine]] + [[Representational Deduction]]：额外 +1.6pp / +3.1pp
- [[Causally-Decoupled Attention]]：额外 +0.8pp / +1.5pp
- 恢复未来帧预测辅助损失：额外 +1.0pp / +1.3pp

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 个子集（Spatial/Object/Goal/Long） | 标准桌面操作 benchmark | 仿真测试 |
| [[LIBERO]]-Plus | 7 种扰动类型 | 相机/机器人/光照/背景/噪声/布局变化 | 鲁棒性测试 |
| [[RoboCasa]]-GR1 | 24 个任务，5 类 | 关节物体操作 | 仿真测试 |
| Agibot-G1 Real | 8 标准 + 3 泛化 + 少样本 | 真实人形机器人 | 真实世界测试 |

### 实现细节

- **Backbone**: 预训练 Vision-Language [[World Action Model]]（未命名，6B 量级）
- **VJEPA2-AC 编码器**: 冻结参数，输出 256×1408 维 patch 表征
- **Motion-CoT 查询数 K**: 64
- **动作步数 H**: 未明确
- **扩散步数（推理）**: 4 步（[[Flow Matching]] 解码）
- **总参数**: 5.4B

### 可视化结果

- t-SNE 证明 Motion-CoT token **无监督聚类**按任务类型分离，说明意图解耦成功
- PCA 伪彩色图证明 CDE 激活集中于**物体交互区域**，具有良好的空间局部性
- 泛化场景中 PILOT 相机偏移成功率 72%，对比 Fast-WAM 的 50%，提升 22pp

---

## 批判性思考

### 优点

1. **效率-性能兼得**: 90% 推理延迟降低同时 LIBERO 达到最优，彻底解决了 WAM predict-then-act 范式的推理瓶颈
2. **鲁棒的泛化能力**: LIBERO-Plus 与少样本场景均大幅领先，运动意图解耦确实降低了背景依赖
3. **可解释性强**: t-SNE / PCA 可视化提供直觉证据，Motion-CoT 确实捕获了语义级运动模式

### 局限性

1. **VJEPA2-AC 依赖**: CDE 冻结编码器的质量上限受限于 VJEPA2-AC 预训练数据分布，跨域泛化未被评估
2. **真实机器人平台单一**: 仅在 Agibot-G1 上验证，在其他人形/操作臂平台的可迁移性未知
3. **K=64 的查询数量选取**: 消融中未分析 K 的敏感性，最优值可能依赖任务复杂度

### 潜在改进方向

1. 将 CDE 的监督信号从表征空间扩展到多模态（触觉、力觉）以支持接触丰富任务
2. 探索自适应 K（动态运动查询数量）以适应不同任务复杂度

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（ablation + 超参）
- [x] 数据集可获取（LIBERO / RoboCasa 均公开）

---

## 关联笔记

### 基于

- [[World Action Model|WAM]]: PILOT 是 WAM 范式的改进，解决其表征纠缠问题
- [[V-JEPA2]]: 冻结 VJEPA2-AC 编码器是 CDE 的基础
- [[Flow Matching]]: 动作解码器采用 4 步流匹配生成

### 对比

- [[Fast-WAM]]: 核心对比 baseline，PILOT 在所有场景全面超越
- PokeVLA: LIBERO-Plus 中的强竞争对手（79.3 vs 81.0）

### 方法相关

- [[Motion-CoT]]: 核心创新，运动语义的链式推理
- [[Causally-Decoupled Attention]]: 实现意图-轨迹信息流解耦
- [[Causal Dynamics Engine]]: 表征空间物理状态预测
- [[Representational Deduction]]: 总体方法论框架名称
- [[Action Chunking]]: 动作输出形式

### 数据集相关

- [[LIBERO]]: 标准仿真测试集
- [[RoboCasa]]: 多任务仿真测试集

---

## 速查卡片

> [!summary] PILOT
> - **核心**: 将高层运动意图（Motion-CoT）与低层动作轨迹解耦，消除 WAM 表征纠缠
> - **方法**: Motion-CoT 查询 + Causally-Decoupled Attention + Causal Dynamics Engine（表征空间未来预测）
> - **结果**: LIBERO 97.9%，RoboCasa-GR1 62.6%，Agibot-G1 真实 83.1%，推理延迟降低 90%
> - **代码**: 未开源（截至 2026-08-11）

---

*笔记创建时间: 2026-08-11*

---
title: "AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding"
method_name: "AffordanceVLA"
authors: [Qize Yu, Jiadi You, Yuran Wang, Jiaqi Liang, Bowen Ping, Yang Tian, Yue Chen, Minghong Cai, Zeying Gong, Ruihai Wu, Yinchuan Li, Junwei Liang, Yingcong Chen]
year: 2026
venue: arXiv
tags: [vla, affordance, robot-manipulation, mixture-of-transformers, imitation-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.06155
created: 2026-06-06
---

# 论文笔记：AffordanceVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University / HKUST(GZ) / CUHK / Knowin AI |
| 日期 | June 2026 |
| 项目主页 | [skywalker-yqz.github.io/AffordanceVLA](https://skywalker-yqz.github.io/AffordanceVLA/) |
| 对比基线 | [[Pi0]], [[GR-1]], [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06155) / [Code](https://github.com/Skywalker-yqz/AffordanceVLA/) |

---

## 一句话总结

> 提出 AffordanceVLA，通过结构化可供性预测（Which2Act/Where2Act/How2Act）作为中间表示，结合 [[Mixture-of-Transformers|MoT]] 架构和三阶段渐进式训练，弥合 VLM 语义空间与机器人控制策略之间的结构性鸿沟。

---

## 核心贡献

1. **结构化可供性框架**: 将 Which2Act（物体定位）、Where2Act（2D交互位置）、How2Act（3D几何推理）建模为任务导向的中间监督信号，实现精确的感知-动作映射
2. **MoT 架构 + 渐进式训练**: 设计含三个专家（理解、可供性生成、动作）的 [[Mixture-of-Transformers]] 架构，配合三阶段渐进式数据课程训练策略
3. **强竞争力的实验结果**: 在 [[LIBERO]] 上达到 95.8% 平均成功率，在 [[CALVIN]] ABC→D 上达到 4.33 平均长度，并通过了真实世界验证

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）|VLA]] 模型利用预训练 VLM 的丰富世界知识实现跟随指令的机器人操作，但 VLM 语义空间与具身控制策略之间存在结构性不匹配，阻碍了精确的感知-动作映射。

### 现有方法的局限

- 直接将 VLM 输出端到端映射到机器人动作，缺乏中间表示
- 语义层面的理解与低层动作执行之间缺乏显式的空间推理桥梁
- 简单堆叠数据量并不能充分挖掘数据集的内在价值（"盲目扩大数据量无法最大化数据集的内在能力"）

### 本文的动机

[[可供性（Affordance）]] 天然耦合了空间定位（视觉）、语义条件（语言）和执行引导（动作），可以作为将 VLM 高层语义与机器人低层控制连接起来的结构性桥梁。通过预测"操作哪个物体"、"在哪里交互"、"如何操作"三个层次的可供性，可以为动作生成提供丰富的几何和语义先验。

---

## 方法详解

### 模型架构

AffordanceVLA 采用 [[Mixture-of-Transformers]] ([[Mixture-of-Transformers|MoT]]) 架构，包含三个专用专家模块：

- **输入**: 语言指令 $l$ + 视觉观测 $O_t$
- **Backbone**: 基于 [[Pi0]] 的预训练 VLM 骨干
- **核心模块**: [[Mixture-of-Transformers|MoT]] 三专家系统，通过单向 UAA（Understanding-Affordance-Action）渐进注意力机制协调
- **输出**: [[Action Chunking|动作块]] $\hat{a}_{t:t+k}$ 以及结构化可供性预测 $\hat{A}_t$
- **中间表示**: Which2Act 视觉潜码 + Where2Act 2D可供性图 + How2Act 3D形状/布局

### 核心模块

#### 模块1: Understanding Expert（$\mathcal{M}_{und}$）

**设计动机**: 将视觉观测和语言指令融合为指令感知的统一表示，供下游两个专家使用

**具体实现**:
- 接收视觉观测 $O_t$ 和语言指令 $l$
- 通过 [[Transformer]] 编码生成指令感知表示 $h_t^{und}$
- 通过 UAA 注意力机制将 $h_t^{und}$ 单向传递给另外两个专家，避免表示坍塌

#### 模块2: Affordance Generation Expert（$\mathcal{M}_{gen}$）

**设计动机**: 利用 [[可供性（Affordance）]] 作为任务导向的中间表示，从语义层面到空间层面逐步解码操作信息

**具体实现**:
- 接收 $h_t^{und}$，通过三个并行子模块解码结构化可供性 token $\hat{A}_t$：
  - **Which2Act**: [[VQ-VAE]] 编码的视觉潜码，代表目标物体的紧凑外观特征（实现中替换为 Flux VAE 以提升质量）
  - **Where2Act**: 二值 2D 可供性图（Binary Affordance Map），指示交互区域
  - **How2Act**: 包含形状（3D点云，通过[[扩散模型]]生成）和布局（10维空间参数，含位置、朝向、包围盒）

#### 模块3: Action Expert（$\mathcal{M}_{act}$）

**设计动机**: 在可供性先验的引导下生成精确的机器人控制动作

**具体实现**:
- 同时接收 $h_t^{und}$ 和 $\hat{A}_t$ 作为条件
- 利用 [[Flow Matching]] 生成连续动作序列 $\hat{a}_{t:t+k}$
- 可供性提供了从语义到几何的丰富先验，减小了动作空间的不确定性

#### 模块4: UAA 渐进注意力机制

**设计动机**: 确保信息流的单向性（理解 → 可供性 → 动作），避免不同任务的表示相互干扰

**具体实现**:
- Understanding Expert 输出可被 Affordance 和 Action 专家使用
- Affordance Expert 输出仅被 Action Expert 使用
- 严格的单向信息流防止动作误差反向影响语义表示

---

## 关键公式

### 公式1: [[可供性（Affordance）|Which2Act 视觉潜码损失]]

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \|\hat{z}_{c,h,w} - z_{q,c,h,w}\|^2
$$

**含义**: 约束模型预测的视觉潜码 $\hat{z}$ 接近目标物体的 [[VQ-VAE]] 编码 $z_q$，实现物体视觉特征的精确匹配

**符号说明**:
- $\hat{z}_{c,h,w}$: 模型预测的视觉潜码，维度为 $C \times H \times W$
- $z_{q,c,h,w}$: VQ-VAE 量化后的目标物体视觉特征
- $C, H, W$: 通道数、高度、宽度

### 公式2: [[可供性（Affordance）|Where2Act 二元交叉熵损失]]

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1 - M_i) \log(1 - \sigma(\hat{y}_i)) \right]
$$

**含义**: 通过二元交叉熵约束 2D 可供性图的预测，使模型准确识别交互热区

**符号说明**:
- $M_i \in \{0,1\}$: 第 $i$ 个像素的二值可供性标签
- $\hat{y}_i$: 模型对第 $i$ 个像素的预测 logit
- $\sigma(\cdot)$: Sigmoid 激活函数
- $H_t, W_t$: 可供性图的高度和宽度

### 公式3: [[扩散模型|How2Act 形状扩散损失]]

$$
\mathcal{L}_{shape} = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, \varepsilon \sim \mathcal{N}(0,\mathbf{I})} \left[ \|\varepsilon - \hat{\varepsilon}_\theta(x_t, t, \bar{h}_{shape})\|^2 \right]
$$

**含义**: 使用扩散去噪目标训练 3D 点云生成网络，以 $\bar{h}_{shape}$ 为条件预测目标物体的 3D 形状

**符号说明**:
- $\varepsilon$: 从标准正态分布采样的噪声
- $\hat{\varepsilon}_\theta$: 去噪网络（以时间步 $t$ 和形状条件 $\bar{h}_{shape}$ 为条件）
- $x_t$: 扩散过程中第 $t$ 步的加噪点云
- $\bar{h}_{shape}$: 来自理解专家的形状条件表示

### 公式4: [[SmoothL1 Loss|How2Act 布局回归损失]]

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \text{SmoothL}_1\left(\hat{y}_{layout}^{(j)},\, y_{layout}^{(j)}\right)
$$

**含义**: 对 10 维空间布局参数（位置、朝向、包围盒）进行回归，提供鲁棒的 3D 空间定位

**符号说明**:
- $\hat{y}_{layout}^{(j)}$: 第 $j$ 维布局参数的预测值
- $y_{layout}^{(j)}$: 第 $j$ 维布局参数的真实值
- $\text{SmoothL}_1$: 平滑 L1 损失，对离群值更鲁棒

### 公式5: [[Curriculum Learning|第一阶段总损失]]

$$
\mathcal{L}_{Stage1} = \lambda_{which} \mathcal{L}_{which} + \lambda_{where} \mathcal{L}_{where} + \lambda_{shape} \mathcal{L}_{shape} + \lambda_{layout} \mathcal{L}_{layout}
$$

**含义**: 第一阶段在通用可供性数据上进行联合预训练，学习基础视觉-语言-空间对齐

**符号说明**:
- $\lambda_{which} = \lambda_{where} = \lambda_{shape} = 0.1$: 各可供性损失权重
- $\lambda_{layout} = 0.04$: 布局损失权重（较小，避免空间回归主导）

### 公式6: [[Curriculum Learning|第二/三阶段总损失]]

$$
\mathcal{L}_{Stage2} = \lambda_{act} \mathcal{L}_{act} + \lambda_{afd} \mathcal{L}_{afd}
$$

**含义**: 第二阶段在机器人数据上联合训练动作生成和可供性预测，建立感知-动作映射；第三阶段将 $\lambda_{afd}$ 退火至 0.15 以适应目标任务

**符号说明**:
- $\lambda_{act} = 1.0$: 动作损失权重（主任务）
- $\lambda_{afd} = 0.5$（Stage II）→ $0.15$（Stage III）: 可供性损失权重
- $\mathcal{L}_{act}$: [[Flow Matching]] 动作生成损失
- $\mathcal{L}_{afd}$: 可供性预测联合损失（$\mathcal{L}_{which} + \mathcal{L}_{where} + \mathcal{L}_{shape} + \mathcal{L}_{layout}$ 加权和）

---

## 关键图表

### Figure 1: AffordanceVLA Overview

![Figure 1 Overview](https://arxiv.org/html/2606.06155v1/x1.png)

**说明**: AffordanceVLA 全局概览。左下展示三专家 [[Mixture-of-Transformers|MoT]] 架构（Understanding、Affordance Generation、Action）及三类可供性中间表示（Which2Act/Where2Act/How2Act）；上方展示三阶段渐进式数据课程训练策略；右下展示在仿真和真实世界评估上的强大性能。

### Figure 2: Pipeline

![Figure 2 Pipeline](https://arxiv.org/html/2606.06155v1/x2.png)

**说明**: 详细的模型 pipeline。MoT 架构中三个专家通过单向 UAA（Understanding-Affordance-Action）渐进注意力机制协调：$\mathcal{M}_{und}$ 融合视觉和语言生成 $h_t^{und}$，$\mathcal{M}_{gen}$ 解码出 Which2Act/Where2Act/How2Act 三类可供性 token $\hat{A}_t$，$\mathcal{M}_{act}$ 基于 $h_t^{und}$ 和 $\hat{A}_t$ 生成[[Action Chunking|动作块]]。

### Figure 3: Data Efficiency Curve

![Figure 3 Data Efficiency](https://arxiv.org/html/2606.06155v1/x3.png)

**说明**: 数据效率对比曲线。在 10%–100% 微调数据量下，完整 AffordanceVLA 始终优于 Pi0、No-Afd（Pi0 架构无可供性）和 AffordanceVLA w/o stage II，表明可供性中间表示显著提升了数据利用效率，低数据量下优势尤为明显。

### Figure 4: Real-World Experiment Visualizations

![Figure 4 Real World](https://arxiv.org/html/2606.06155v1/x4.png)

**说明**: 真实世界实验可视化。上方展示基础任务（关闭、拾取等）的定性结果；左下展示 Drawer 和 Toaster 任务中 Where2Act token 的可视化热图，体现空间感知的精确性；右侧展示连续"拾取所有垃圾"任务的顺序执行过程，体现 How2Act 对精细操作的引导作用。

### Figure 5a: Subgoal Token Quantitative Evaluation

![Figure 5a Subgoal Tokens](https://arxiv.org/html/2606.06155v1/x5.png)

**说明**: 子目标 token 的综合定量评估（a 部分）。验证模型有效获取了任务相关语义，Which2Act/Where2Act/How2Act 三类 token 均展现出与任务相关的高质量表示，证明 MoT 架构的解耦设计成功分离了不同层次的信息。

### Figure 5b: How2Act Layout Token Evaluation

![Figure 5b How2Act Layout](https://arxiv.org/html/2606.06155v1/x6.png)

**说明**: How2Act 布局 token 的定量评估（b 部分）。展示模型在 3D 空间推理能力方面的评估，包括位置预测精度和包围盒估计质量，验证了 10 维布局参数回归的有效性。

### Figure 6: Backbone Representation Evaluation

![Figure 6 Backbone Evaluation](https://arxiv.org/html/2606.06155v1/x7.png)

**说明**: Backbone 表示质量评估。使用完整训练（100k 步）的 VLA backbone，配合不同训练步数（5k-100k 步）的解码器进行评估，验证骨干网络已学习到丰富且可解码的可供性表示，体现 backbone 与 decoder 解耦设计的有效性——即使解码器训练步数较少，性能也相对稳定。

### Figure 7: Qualitative Visualization of Where2Act Affordance Maps

![Figure 7 Where2Act Maps](https://arxiv.org/html/2606.06155v1/x8.png)

**说明**: Where2Act 可供性图的定性可视化。在相同视觉观测下，模型根据不同语言指令动态调整可供性预测热图，展示出鲁棒的视觉-语言对齐能力——例如"打开抽屉"指向把手，"关闭抽屉"时热图分布随交互目标改变，证明模型真正理解了操作语义而非记忆固定热区。

### Table 1: LIBERO Benchmark Results

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Pi0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| gr00t-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| AffordanceVLA (w/o stage II) | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | **89.8** | **95.8** |

**说明**: AffordanceVLA 在 [[LIBERO]] 四个子集上均达到最优，平均成功率 95.8% 超越 Pi0（94.4%）和 F1-VLA（95.7%），证明可供性中间表示对各类操作任务的普遍提升效果。

### Table 2: CALVIN ABC→D Benchmark Results

| Method | 1/5 | 3/5 | 5/5 | Avg. Len |
|--------|-----|-----|-----|----------|
| RoboFlamingo | 82.4 | 46.6 | 23.5 | 2.48 |
| SuSIE | 87.0 | 49.0 | 26.0 | 2.69 |
| GR-1 | 85.4 | 59.6 | 40.1 | 3.06 |
| OpenVLA | 91.3 | 62.0 | 43.5 | 3.27 |
| CLOVER | 96.0 | 70.8 | 45.4 | 3.53 |
| UniVLA | 95.5 | 75.4 | 56.5 | 3.80 |
| Pi0 | 93.8 | 76.7 | 60.1 | 3.84 |
| Seer | 94.4 | 79.9 | 64.3 | 3.98 |
| VPP | 95.3 | 80.3 | 64.5 | 4.01 |
| Seer-Large | 96.3 | 86.1 | 74.0 | 4.28 |
| AffordanceVLA (w/o stage II) | 93.4 | 75.4 | 58.9 | 3.81 |
| **AffordanceVLA (full)** | **96.8** | **87.5** | **75.9** | **4.33** |

**说明**: AffordanceVLA 在 [[CALVIN]] ABC→D 上以 4.33 平均任务链长度超越所有基线，包括更大模型 Seer-Large（4.28），体现结构化可供性对长序列任务规划的显著提升。5/5 成功率（75.9%）比 Seer-Large（74.0%）更高，说明在完整任务链执行能力上的优势更为突出。

### Table 3: Ablation Studies

| Method | Spatial | Object | Goal | Long | Avg (LIBERO) | 1/5 | 3/5 | 5/5 | Avg. Len (CALVIN) |
|--------|---------|--------|------|------|--------------|-----|-----|-----|-------------------|
| No-Afd (Pi0 Arch) | 96.0 | 95.4 | 92.4 | 85.8 | 92.4 | 94.5 | 78.0 | 62.8 | 3.93 |
| Frozen-Afd | 68.0 | 71.1 | 66.4 | 62.9 | 67.1 | 85.3 | 55.9 | 26.3 | 2.83 |
| w/o stage II | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 | 93.4 | 75.4 | 58.9 | 3.81 |
| w/o Which2Act | 97.5 | 97.6 | 95.0 | 88.1 | 94.6 | 96.7 | 83.3 | 72.1 | 4.20 |
| w/o Where2Act | 95.5 | 96.0 | 93.4 | 88.0 | 93.2 | 96.2 | 81.9 | 69.8 | 4.13 |
| w/o How2Act | 96.1 | 96.5 | 93.9 | 88.2 | 93.7 | 95.0 | 79.4 | 65.9 | 4.01 |
| Block-wise Tokens | 92.4 | 92.9 | 89.8 | 86.0 | 90.3 | 94.1 | 77.1 | 61.7 | 3.89 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | **89.8** | **95.8** | **96.8** | **87.5** | **75.9** | **4.33** |

**关键发现**: (1) Frozen-Afd 性能断崖（67.1%）证明可供性表示必须端到端训练，固定预训练特征无效；(2) Stage II 贡献巨大（86.2% → 95.8%），机器人数据的可供性联合训练不可或缺；(3) 三类可供性各有贡献，去除 How2Act 后 CALVIN 下降最多（-0.32），说明 3D 几何对长序列任务最关键；(4) 序列化 token 优于块状 token（Block-wise），有序排列有利于自回归生成

### Table 4: Real-World Task Results

| Task | Pi0 | AffordanceVLA |
|------|-----|----------------|
| Close | 86.7 | 93.3 |
| Pick up (Color) | 86.7 | 100.0 |
| Pick up (Shape) | 80.0 | 86.7 |
| Pick up (Microwave) | 80.0 | 80.0 |
| Safe | 26.7 | 86.7 |
| Red | 73.3 | 86.7 |
| Green | 53.3 | 80.0 |
| Duck | 80.0 | 93.3 |
| **Basic Average** | **70.8** | **88.3** |
| Drawer (pick) | 46.7 | 86.7 |
| Drawer (close) | 40.0 | 100.0 |
| Toaster (pick) | 46.7 | 80.0 |
| Toaster (toast) | 26.7 | 86.7 |
| Pick all rubbish (1st) | 93.3 | 100.0 |
| Pick all rubbish (2nd) | 53.3 | 80.0 |
| Pick all rubbish (3rd) | 6.7 | 46.7 |
| Empty (↓) | 33 | 11 |
| **Complex Average** | **44.8** | **82.9** |

**说明**: 真实世界实验中，AffordanceVLA 在基础任务上以 88.3% vs 70.8% 大幅超越 Pi0；在复杂任务上更是以 82.9% vs 44.8% 取得近乎翻倍的提升。Safe 任务（需精细开锁操作）从 26.7% 大幅提升至 86.7%，体现 3D 可供性引导对精细操作的决定性作用。Pick all rubbish (3rd) 仍较低（46.7%），反映在多步骤连续操作的末段误差积累仍是挑战。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| AGD20K | ~20K 图像 | 人-物交互可供性标注，含 affordance map | Stage I 预训练 |
| RefSpatial | VQA 格式 | 空间关系理解数据 | Stage I 预训练 |
| PRISM | 机器人操作数据 | 经 RexOmni 微调后用于可供性标注 | Stage I 预训练 |
| InternData-A1 | 大规模机器人数据 | 经自动化 pipeline 生成 10万+ 可供性标注 | Stage II 训练 |
| DROID | 机器人操作数据集 | 补充真实世界多样性 | Stage II 训练 |
| [[LIBERO]] | 4 个子集 | 仿真操作基准（Spatial/Object/Goal/Long） | 评估 |
| [[CALVIN]] ABC→D | 长链任务序列 | 跨域泛化基准 | 评估 |

### 实现细节

- **Backbone**: 基于 [[Pi0]] 预训练 VLM（PaliGemma + [[Flow Matching]] 动作头）
- **可供性标注 Pipeline**: 6步自动化流程：
  1. 微调 RexOmni 于 PRISM
  2. 规则化关键帧检测（Start/Pre-Action/Gripper/Stop/Apex/End）
  3. Claude Opus 4.5 指令分解
  4. Qwen3-VL 逐关键帧可供性标注
  5. 视觉定位和可供性生成（Which2Act/Where2Act/How2Act）
  6. 质量核查（几何一致性 + 100轮人工审核）
- **标注规模**: 从 InternData-A1 和 DROID 生成超过 10 万条可供性标注
- **Which2Act 编码器**: 实际使用中从 VQ-VAE 切换为 Flux VAE 以获得更高质量视觉特征
- **可供性 Token 格式**: 序列化（按类型顺序），消融实验证明优于块状（Block-wise）排列

### 可视化结果

Where2Act 热图（Figure 7）展示了显著的视觉-语言动态对齐：在相同场景下，"打开抽屉"和"关闭抽屉"的热图分布截然不同，证明模型并非记忆固定热区，而是真正理解了操作意图与空间位置的关联。Figure 4 中的 Drawer/Toaster 可视化进一步印证了 Where2Act 在真实场景中的精确定位能力。

---

## 批判性思考

### 优点

1. **中间表示设计合理**: Which2Act/Where2Act/How2Act 分别对应"操作什么"、"在哪里"、"怎么操作"，与人类操作的认知过程高度契合，具有良好的可解释性
2. **渐进式训练有效**: 三阶段训练从通用可供性知识到机器人数据再到目标任务，消融实验充分验证了每阶段的必要性，Stage II 贡献最为显著
3. **真实世界验证充分**: 复杂任务从 44.8% 提升至 82.9%，表明可供性中间表示对真实物理交互有实质性帮助，不只是仿真数据过拟合

### 局限性

1. **标注依赖外部模型**: 自动化标注 pipeline 依赖 RexOmni、Qwen3-VL 等外部模型，这些模型本身的错误可能引入噪声，且难以量化噪声对最终性能的影响
2. **3D 推理依赖深度精度**: How2Act 的 3D 形状和布局估计需要可靠的深度信息，在深度传感器精度不足的场景下可能退化
3. **评估场景局限**: 主要在桌面操作任务上验证，对移动操作、全身操作等更开放场景的泛化性未做充分验证

### 潜在改进方向

1. 探索可供性预测与[[强化学习]]的结合，在无标注环境中通过试错自动发现可供性
2. 扩展 How2Act 到更复杂的 6-DOF 抓取位姿估计，支持非平面物体操作
3. 将结构化可供性框架扩展到双臂协调、工具使用等更复杂的操作场景

### 可复现性评估

- [x] 代码开源（GitHub: Skywalker-yqz/AffordanceVLA）
- [ ] 预训练模型（待确认是否发布权重）
- [x] 训练细节完整（三阶段超参数均在论文中给出）
- [x] 数据集可获取（LIBERO/CALVIN 均为公开数据集；InternData-A1/DROID 需确认开放程度）

---

## 关联笔记

### 基于

- [[Pi0]]: 基础 VLM backbone 架构和 Flow Matching 动作生成范式
- [[Mixture-of-Transformers]]: 专家并行架构设计基础
- [[GR-1]]: CALVIN 基准上的重要比较对象，证明 VLA 可做长链任务

### 对比

- [[OpenVLA]]: 最基础的端到端 VLA 基线，AffordanceVLA 在所有指标上大幅超越
- [[UniVLA]]: CALVIN 上的强基线（Avg Len 3.80），被 AffordanceVLA（4.33）超越
- [[GR-1]]: CALVIN 上 Avg Len 3.06，早期经典方法

### 方法相关

- [[可供性（Affordance）]]: 核心中间表示概念
- [[Mixture-of-Transformers]]: 三专家 MoT 架构
- [[Flow Matching]]: 动作生成方式
- [[Action Chunking]]: 多步动作预测范式
- [[VQ-VAE]]: Which2Act 视觉编码基础（实现中替换为 Flux VAE）
- [[扩散模型]]: How2Act 形状生成
- [[Curriculum Learning]]: 三阶段渐进式训练策略

### 硬件/数据相关

- [[LIBERO]]: 主要仿真评估基准
- [[CALVIN]]: 主要仿真评估基准（长序列跨域泛化）

---

## 速查卡片

> [!summary] AffordanceVLA
> - **核心**: 用 Which2Act/Where2Act/How2Act 三类可供性作为 VLA 的中间表示，弥合语义与控制鸿沟
> - **方法**: MoT 三专家架构 + UAA 单向注意力 + 三阶段渐进式训练
> - **结果**: LIBERO 95.8% / CALVIN 4.33，真实世界复杂任务 82.9%（vs Pi0 的 44.8%）
> - **代码**: [github.com/Skywalker-yqz/AffordanceVLA](https://github.com/Skywalker-yqz/AffordanceVLA/)

---

*笔记创建时间: 2026-06-06*

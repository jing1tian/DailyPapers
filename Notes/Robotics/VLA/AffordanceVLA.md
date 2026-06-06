---
title: "AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding"
method_name: "AffordanceVLA"
authors: [Qize Yu, Jiadi You, Yuran Wang, Jiaqi Liang, Bowen Ping, Yang Tian, Yue Chen, Minghong Cai, Zeying Gong, Ruihai Wu, Yinchuan Li, Junwei Liang, Yingcong Chen]
year: 2026
venue: arXiv
tags: [vla, affordance, mixture-of-experts, robot-manipulation, intermediate-representation, 3d-reasoning, visuomotor-policy]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.06155
created: 2026-06-06
---

# 论文笔记：AffordanceVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST, 中山大学, 华为诺亚方舟实验室等 |
| 日期 | June 2026 |
| 项目主页 | [https://skywalker-yqz.github.io/AffordanceVLA/](https://skywalker-yqz.github.io/AffordanceVLA/) |
| 对比基线 | [[Pi0]], [[OpenVLA]], [[F1-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.06155) / [Code](https://github.com/Skywalker-yqz/AffordanceVLA/) |

---

## 一句话总结

> AffordanceVLA 将可供性预测（Which/Where/How2Act）作为结构化中间监督，通过[[混合专家架构|Mixture-of-Transformer]]桥接VLM语义空间与机器人物理控制，无需推理时解码即可大幅提升操作成功率。

---

## 核心贡献

1. **结构化可供性分解**: 将[[可供性|Affordance]]拆解为 Which2Act（物体定位）、Where2Act（2D交互定位）、How2Act（3D几何推理）三个互补组件，提供精确的中间监督信号。
2. **Mixture-of-Transformer架构**: 设计三专家[[MoT架构]]（Understanding、Affordance Generation、Action），通过单向因果注意力机制防止表示坍塌。
3. **三阶段渐进式训练策略**: 通用可供性预训练 → 可供性增强机器人数据协训 → 目标任务微调，配套自动化数据标注流水线解决可供性标签稀缺问题。
4. **强实验验证**: LIBERO 95.8%、CALVIN 4.33平均链长、真实环境88.3%，全面超越[[Pi0]]基线。

---

## 问题背景

### 要解决的问题

现有[[视觉语言动作模型|VLA]]将观测直接端到端映射到动作序列，缺乏对任务相关场景结构的显式推理，导致在需要精确空间理解的复杂操作任务中泛化能力不足。

### 现有方法的局限

- **直接端到端映射**（[[OpenVLA]]等）：无中间表示，视觉语言对齐能力在纯动作优化下逐渐侵蚀
- **视频预测/世界模型**（[[视觉预测|Visual Foresight]]等）：生成冗余像素信息，计算开销大，与动作控制耦合松散
- **结构化线索**（CoT-VLA、ThinkAct等）：使用语言比率或关键姿态，缺乏精细的空间几何信息
- **独立可供性模块**：仅表示稀疏2D/3D接触点，与外部抓取生成器形成脆弱的开环流水线

### 本文的动机

可供性（Affordances）天然编码"对哪个物体操作、在哪里交互、如何交互"，是连接VLM语义空间与3D物理控制的理想中间预测目标。关键设计：将可供性**内化**于VLA主干中联合优化，保留骨干网络的视觉-语言对齐能力，而非在控制优化中侵蚀它。

---

## 方法详解

### 模型架构

![Figure 1: AffordanceVLA Overview](https://arxiv.org/html/2606.06155v1/x1.png)

**Figure 1**: AffordanceVLA整体框架概览。输入语言指令 $l$ 和视觉观测 $O_t$，经过三个专家逐级处理，最终输出动作序列。

AffordanceVLA 采用 **[[MoT架构|Mixture-of-Transformer]] (MoT)** 架构，三个专家通过**单向因果注意力**（Understanding → Affordance → Action）顺序处理信息：

- **输入**: 语言指令 $l$ + 视觉观测 $O_t$
- **Backbone**: 预训练[[视觉语言模型|VLM]]骨干
- **核心模块**: [[可供性预测|Affordance Prediction]] 作为中间监督，协调三专家协同工作
- **输出**: 动作序列 $a_{t:t+k}$（[[Action Chunking|动作块]]）
- **推理时**: Affordance Generation Expert 仅参与训练，部署时不解码可供性

### 核心模块

![Figure 2: MoT Pipeline](https://arxiv.org/html/2606.06155v1/x2.png)

**Figure 2**: Mixture-of-Transformer架构流程图，展示三专家协同工作与UAA注意力机制。

#### Understanding Expert ($\mathcal{M}_{und}$)

**设计动机**: 利用预训练[[视觉语言模型|VLM]]先验，建立细粒度视觉-语言对齐。

**具体实现**:
- 融合视觉观测 $O_t$ 和语言指令 $l$，生成指令感知语义表示 $h_t^{und}$
- 保持[[视觉编码器|Vision Encoder]]预训练能力，提供后续专家所需的语义基础
- 输出 $h_t^{und}$ 同时供 Affordance Generation 和 Action Expert 使用

#### Affordance Generation Expert ($\mathcal{M}_{gen}$)

**设计动机**: 将高层语义对齐锚定为可操作的几何线索，过滤任务无关噪声。

**具体实现**:
- 以 $h_t^{und}$ 为条件，并行生成三类结构化可供性先验 $A_t$：
  - **Which2Act**: 物体中心视觉定位 — 预测目标物体的视觉潜码
  - **Where2Act**: 2D精细交互定位 — 预测可供性热图
  - **How2Act**: 3D空间几何推理 — 预测体素形状 + 10-DoF位姿
- 注意力机制：仅从 Understanding Expert 获取信息，**严格阻止** Action 信息流入预测阶段

#### Action Expert ($\mathcal{M}_{act}$)

**设计动机**: 接收已定位的可供性信息，专注精确的物理执行，从复杂视觉推理中解放出来。

**具体实现**:
- 以 $h_t^{und}$ 和 $A_t$ 为双重条件，合成动作序列 $a_{t:t+k}$
- 通过[[扩散策略|Diffusion Policy]]生成连续动作
- 注意力机制：同时关注 Understanding 和 Affordance Generation 的输出

### Which2Act：物体定位

**目标**: 物体中心的视觉定位，将目标物体从场景中"抠出"

**实现**: 基于目标[[边界框|Bounding Box]]裁剪观测，使用冻结的 [[Flux VAE]] 编码器重建连续视觉潜码 $z_q$，通过[[均方误差|MSE]]损失优化：

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \|\hat{z}_{c,h,w} - z_{q,c,h,w}\|^2
$$

这种"自编码"形式将空间定位转化为潜码重建，迫使模型孤立交互实体。

### Where2Act：2D交互定位

**目标**: 细粒度交互区域预测，将语义意图转化为可操作的视觉可供性

**实现**: [[Transformer解码器]]以空间位置嵌入为交叉注意力查询，将1D查询token转化为2D空间分布，通过[[二值交叉熵|BCE]]损失逐像素优化：

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1 - M_i) \log(1 - \sigma(\hat{y}_i)) \right]
$$

其中 $M_i$ 为可供性热图的真实标签，$\hat{y}_i$ 为预测logit。

### How2Act：3D几何推理

**目标**: 全面刻画目标物体的3D形状与空间位姿

**实现**: 两个协同分支：

**形状生成分支** — 条件[[扩散模型|Diffusion Model]]预测3D体素潜码：

$$
\mathcal{L}_{shape} = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, \varepsilon \sim \mathcal{N}(0, I)} \left[ \|\varepsilon - \hat{\varepsilon}_\theta(x_t, t, \bar{h}_{shape})\|^2 \right]
$$

**位姿回归分支** — MLP回归10-DoF空间位姿（旋转+缩放+平移），通过[[Smooth L1损失]]优化：

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \text{SmoothL1}\left(\hat{y}_{layout}^{(j)},\, y_{layout}^{(j)}\right)
$$

---

## 关键公式

### 公式1: [[MSE损失|Which2Act 损失]]

$$
\mathcal{L}_{which} = \frac{1}{C \cdot H \cdot W} \sum_{c,h,w} \|\hat{z}_{c,h,w} - z_{q,c,h,w}\|^2
$$

**含义**: 目标物体视觉潜码的均方误差重建损失，驱动模型精确定位并重建目标物体特征。

**符号说明**:
- $\hat{z}_{c,h,w}$: 预测的视觉潜码，通道 $c$、位置 $(h,w)$
- $z_{q,c,h,w}$: 由冻结 Flux VAE 编码的真实视觉潜码
- $C, H, W$: 潜码的通道数、高度、宽度

### 公式2: [[二值交叉熵|Where2Act 损失]]

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1 - M_i) \log(1 - \sigma(\hat{y}_i)) \right]
$$

**含义**: 2D可供性热图的逐像素二值交叉熵损失，督促模型预测精确的交互区域分布。

**符号说明**:
- $M_i \in \{0, 1\}$: 位置 $i$ 的可供性热图真实标签
- $\hat{y}_i$: 模型输出的未归一化预测logit
- $\sigma(\cdot)$: Sigmoid激活函数
- $H_t, W_t$: 目标热图的高度和宽度

### 公式3: [[扩散模型|How2Act 形状损失]]

$$
\mathcal{L}_{shape} = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, \varepsilon \sim \mathcal{N}(0, I)} \left[ \|\varepsilon - \hat{\varepsilon}_\theta(x_t, t, \bar{h}_{shape})\|^2 \right]
$$

**含义**: 条件扩散过程的噪声预测损失，用于生成目标物体的3D体素形状潜码。

**符号说明**:
- $\varepsilon \sim \mathcal{N}(0, I)$: 添加的高斯噪声
- $\hat{\varepsilon}_\theta$: 噪声预测网络（以可供性条件 $\bar{h}_{shape}$ 为输入）
- $x_t$: 时间步 $t$ 的加噪潜码
- $t \sim \mathcal{U}(0, T)$: 均匀采样的扩散时间步

### 公式4: [[Smooth L1损失|How2Act 位姿损失]]

$$
\mathcal{L}_{layout} = \frac{1}{10} \sum_{j=1}^{10} \text{SmoothL1}\left(\hat{y}_{layout}^{(j)},\, y_{layout}^{(j)}\right)
$$

**含义**: 10自由度物体空间位姿的鲁棒回归损失（对异常值不敏感）。

**符号说明**:
- $\hat{y}_{layout}^{(j)}$: 第 $j$ 个自由度的预测值（旋转/缩放/平移）
- $y_{layout}^{(j)}$: 对应的真实位姿标签
- 10-DoF 包含：旋转（3/4维）、缩放（3维）、平移（3维）

### 公式5: [[多任务学习|Stage I 总损失]]

$$
\mathcal{L}_{Stage1} = \lambda_{which} \mathcal{L}_{which} + \lambda_{where} \mathcal{L}_{where} + \lambda_{shape} \mathcal{L}_{shape} + \lambda_{layout} \mathcal{L}_{layout}
$$

**含义**: Stage I 通用可供性预训练的多任务联合损失。

**符号说明**:
- $\lambda_{which} = \lambda_{where} = \lambda_{shape} = 0.1$: Which/Where/How2Act 权重（形状部分）
- $\lambda_{layout} = 0.04$: How2Act 位姿部分权重（较小，避免过拟合）

### 公式6: [[联合训练|Stage II 总损失]]

$$
\mathcal{L}_{Stage2} = \lambda_{act} \mathcal{L}_{act} + \lambda_{afd} \mathcal{L}_{afd}
$$

**含义**: Stage II 可供性增强机器人协训的联合损失，同时优化动作控制和可供性预测。

**符号说明**:
- $\lambda_{act} = 1.0$: 动作损失权重（主要目标）
- $\lambda_{afd} = 0.5$: 可供性损失权重（Stage III 降至 0.15）

---

## 关键图表

### Figure 1: 系统概览

![Figure 1: AffordanceVLA Overview](https://arxiv.org/html/2606.06155v1/x1.png)

**说明**: AffordanceVLA 整体架构。输入语言指令和视觉观测，经三专家 MoT 处理，Affordance Expert 生成 Which/Where/How2Act 三类可供性先验，最终 Action Expert 输出动作序列。

### Figure 2: MoT架构流水线

![Figure 2: Pipeline / MoT Architecture](https://arxiv.org/html/2606.06155v1/x2.png)

**说明**: 展示[[MoT架构]]中三专家的详细交互结构。单向因果注意力（UAA）确保 Affordance Generation 仅从 Understanding 获取信息，Action 同时关注两个前序专家，严格阻止信息反向流动。

### Figure 3: 数据效率曲线

![Figure 3: Data Efficiency Curve](https://arxiv.org/html/2606.06155v1/x3.png)

**说明**: AffordanceVLA 在仅使用 **40% 下游数据**时即超越 [[Pi0]] 全量训练性能，体现了可供性中间监督对数据效率的显著提升。

### Figure 4: 真实环境实验可视化

![Figure 4: Real-World Experiment Visualizations](https://arxiv.org/html/2606.06155v1/x4.png)

**说明**: 真实机器人操作任务的可视化结果，包括基础任务（关门、颜色/形状区分抓取）和复杂任务（抽屉操作、烤面包机操作）。

### Figure 5: 子目标Token定量评估（Part 1）

![Figure 5a: Subgoal Token Quantitative Evaluation Part 1](https://arxiv.org/html/2606.06155v1/x5.png)

**说明**: 不同子目标token配置下的性能对比，验证 AffordanceVLA（N_gen=388）相对于精简版（fast，N_gen=64）的优势。

### Figure 5: 子目标Token定量评估（Part 2）

![Figure 5b: Subgoal Token Quantitative Evaluation Part 2](https://arxiv.org/html/2606.06155v1/x6.png)

**说明**: 子目标token分配策略的消融，展示 Which/Where/How2Act 各组件token数量对最终性能的影响。

### Figure 6: 骨干表示能力评估

![Figure 6: Backbone Representation Evaluation](https://arxiv.org/html/2606.06155v1/x7.png)

**说明**: 通过特征可视化验证 AffordanceVLA 的 Understanding Expert 保留了预训练[[视觉语言模型|VLM]]的视觉-语言对齐能力，而纯动作训练（No-Afd）会造成表示退化。

### Figure 7: Where2Act 可供性热图可视化

![Figure 7: Where2Act Affordance Map Qualitative Visualization](https://arxiv.org/html/2606.06155v1/x8.png)

**说明**: Where2Act 生成的2D交互定位热图的定性展示，模型能准确聚焦于任务相关的交互区域（如把手、按钮等），对视觉混淆场景（颜色/形状区分）同样鲁棒。

### Table 1: LIBERO Benchmark 结果

| 方法 | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Pi0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| gr00t-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| AffordanceVLA (w/o stage II) | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | **89.8** | **95.8** |

**说明**: AffordanceVLA 以 95.8% 平均成功率位列第一，在 Spatial 和 Object 子任务上取得最优，Long 任务因长程依赖略低于 F1-VLA。

### Table 2: CALVIN ABC→D Benchmark

| 方法 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | 平均链长 |
|------|-----|-----|-----|-----|-----|---------|
| RoboFlamingo | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.48 |
| SuSIE | 87.0 | 69.0 | 49.0 | 38.0 | 26.0 | 2.69 |
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| CLOVER | 96.0 | 83.5 | 70.8 | 57.5 | 45.4 | 3.53 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| Pi0 | 93.8 | 85.0 | 76.7 | 68.6 | 60.1 | 3.84 |
| Seer | 94.4 | 87.2 | 79.9 | 72.2 | 64.3 | 3.98 |
| VPP | 95.3 | 88.2 | 80.3 | 72.9 | 64.5 | 4.01 |
| Seer-Large | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| AffordanceVLA (w/o stage II) | 93.4 | 84.7 | 75.4 | 68.1 | 58.9 | 3.81 |
| **AffordanceVLA (full)** | **96.8** | **92.0** | **87.5** | **80.8** | **75.9** | **4.33** |

**说明**: AffordanceVLA 以 4.33 平均链长超越 Seer-Large（4.28），在 CALVIN 跨域泛化测试中展现强 OOD 泛化能力。

### Table 3: 消融实验（LIBERO & CALVIN）

| 配置 | Spatial | Object | Goal | Long | Avg. | 1/5 | 3/5 | 5/5 | 平均链长 |
|------|---------|--------|------|------|------|-----|-----|-----|---------|
| **架构与训练策略** |
| No-Afd (Pi0 Arch) | 96.0 | 95.4 | 92.4 | 85.8 | 92.4 | 94.5 | 78.0 | 62.8 | 3.93 |
| Frozen-Afd | 68.0 | 71.1 | 66.4 | 62.9 | 67.1 | 85.3 | 55.9 | 26.3 | 2.83 |
| w/o stage II | 88.5 | 91.7 | 91.3 | 73.3 | 86.2 | 93.4 | 75.4 | 58.9 | 3.81 |
| **可供性表示消融** |
| w/o Which2Act | 97.5 | 97.6 | 95.0 | 88.1 | 94.6 | 96.7 | 83.3 | 72.1 | 4.20 |
| w/o Where2Act | 95.5 | 96.0 | 93.4 | 88.0 | 93.2 | 96.2 | 81.9 | 69.8 | 4.13 |
| w/o How2Act | 96.1 | 96.5 | 93.9 | 88.2 | 93.7 | 95.0 | 79.4 | 65.9 | 4.01 |
| **注意力机制** |
| Block-wise Tokens | 92.4 | 92.9 | 89.8 | 86.0 | 90.3 | 94.1 | 77.1 | 61.7 | 3.89 |
| **AffordanceVLA (full)** | **98.6** | **98.4** | **96.2** | **89.8** | **95.8** | **96.8** | **87.5** | **75.9** | **4.33** |

**关键发现**:
- Frozen-Afd（冻结可供性）导致严重崩塌（67.1%），证明可供性必须与动作联合优化
- No-Afd 仅轻微提升（92.4% vs 94.4%），证明结构化可供性表示而非数据本身是关键
- 三个可供性组件均有贡献，How2Act 影响最大（去掉后降幅最显著）

### Table 4: 数据效率训练配置

| 方法 | Stage I | Stage II | Stage III |
|------|---------|----------|-----------|
| Pi0 | × | × | 分阶段（5k~Full）|
| No-Afd (Pi0 Arch) | × | 仅动作损失 | 分阶段（5k~Full）|
| AffordanceVLA (w/o stage II) | VQA可供性 | × | 分阶段（5k~Full）|
| AffordanceVLA (full) | VQA可供性 | 动作损失 + 可供性损失 | 分阶段（5k~Full）|

### Table 5: 真实环境任务结果

| 方法 | Close | Pick (Color) | Pick (Shape) | Pick Microwave | Pick Safe | Avg. Basic | Drawer Pick | Drawer Close | Toaster Pick | Toaster Toast | Rubbish 1st | Rubbish 2nd | Rubbish 3rd | Empty (↓) | Avg. Complex |
|------|-------|---------|---------|---------|---------|-----------|---------|---------|---------|---------|---------|---------|---------|--------|------------|
| Pi0 | 86.7 | 86.7 | 80.0 | 80.0 | 26.7 | 73.3 | 53.3 | 80.0 | 70.8 | 46.7 | 40.0 | 46.7 | 26.7 | 93.3 | 44.8 |
| **AffordanceVLA** | **93.3** | **100.0** | **86.7** | **80.0** | **86.7** | **88.3** | **86.7** | **100.0** | **80.0** | **46.7** | **86.7** | **100.0** | **80.0** | **11** | **82.9** |

**说明**: 真实环境中 AffordanceVLA 在基础任务（88.3% vs 73.3%）和复杂任务（82.9% vs 44.8%）上均大幅超越 Pi0，尤其在需要细粒度指令理解的颜色/形状区分任务中优势明显。

### Table 6: 模型变体对比

| 变体 | Which2Act tokens | Where2Act tokens | How2Act (Shape/Layout) | N_gen | 编码器 |
|------|---------|---------|---------------------|-------|--------|
| Prev | 64 | 16 | 30/2 | 112 | VQ-VAE |
| AffordanceVLA-fast | 16 | 16 | 30/2 | 64 | Flux VAE |
| AffordanceVLA | 64 | 64 | 256/4 | 388 | Flux VAE |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途（阶段）|
|--------|------|------|-----------|
| AGD20K | 20K | 可供性热图VQA | Stage I 预训练 |
| RefSpatial | 250 | 参考空间可供性VQA | Stage I 预训练 |
| PRISM | 412K | 具身VQA（点+bbox）| Stage I 预训练 |
| InternData-A1 | 100K+ | 机器人轨迹（自动标注可供性）| Stage II 协训 |
| LIBERO | 4套任务 | 仿真操作基准 | Stage III 微调+评测 |
| CALVIN | ABC→D | 长程跨域泛化 | Stage III 微调+评测 |
| DROID | 精选子集 | 真实机器人轨迹 | Stage III 微调+评测 |

### 实现细节

- **Backbone**: 预训练 VLM（基于 [[Pi0]] 架构，3B 参数级别）
- **可供性编码器**: Flux VAE（替换原 VQ-VAE）
- **Stage I 学习率**: 视觉编码器冻结，仅优化 Affordance Generation Expert
- **Stage II**: 解冻 Understanding 和 Action Expert，视觉编码器以低学习率微调
- **Stage III**: 与 Stage II 相同可训练参数，$\lambda_{afd}$ 降至 0.15
- **自动标注流水线**: 规则关键帧检测 → LLM指令分解 → VLM可供性标注 → SAM/SAM-3D视觉定位

### 可视化结果

- Where2Act 热图准确聚焦于任务相关交互区域（把手、按钮、目标物体），对视觉混淆场景鲁棒
- 骨干表示分析表明 AffordanceVLA 保留了 VLM 的视觉语言对齐能力
- 真实环境操作流畅，对指令中颜色/形状细节的响应明显优于 Pi0

---

## 批判性思考

### 优点
1. **优雅的中间监督设计**: 可供性比视频预测更轻量、比语言比率更精确，同时对齐语义与几何两个维度
2. **架构创新**: MoT 单向因果注意力有效防止表示坍塌，且 Affordance Expert 推理时不解码，无额外延迟
3. **数据效率**: 40% 数据即超越全量训练基线，实用价值高
4. **强烈的可供性动机**: 消融实验设计严谨，Frozen-Afd vs No-Afd 的对比清晰验证了"联合优化"的必要性

### 局限性
1. **标注依赖**: Stage II 的自动标注流水线质量难以保证，错误标注可能引入噪声，且规则化关键帧检测可能遗漏关键状态
2. **How2Act 复杂度**: 3D 体素 + 10-DoF 回归的组合对数据质量和计算资源要求较高，实际场景中 3D 真实标签获取困难
3. **泛化边界未充分探索**: 仅在 LIBERO/CALVIN/DROID 评测，对开放世界、非结构化环境的性能未知
4. **可供性监督在部署时丢弃**: 推理时 Affordance Expert 仅提供隐式调节，可解释性有限

### 潜在改进方向
1. 将 How2Act 的 3D 推理替换为[[NeRF]]或[[3D Gaussian Splatting]]等更高效的3D表示
2. 探索在线可供性标注（从部署中持续学习），减少对离线标注数据的依赖
3. 将可供性预测与[[强化学习|RL]]结合，利用环境反馈动态调整可供性先验

### 可复现性评估
- [x] 代码开源（GitHub: Skywalker-yqz/AffordanceVLA）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、数据集均有描述）
- [x] 数据集可获取（AGD20K、LIBERO、CALVIN 均公开）

---

## 关联笔记

### 基于
- [[Pi0]]: 采用 Pi0 的扩散动作头架构，在此基础上增加 MoT 和可供性监督
- [[OpenVLA]]: 早期 VLA 基线，AffordanceVLA 超越其约 19 个百分点
- [[Flux VAE]]: 用于 Which2Act 的视觉潜码编码

### 对比
- [[F1-VLA]]: 当前 LIBERO SOTA，AffordanceVLA 以 95.8% 略超其 95.7%
- [[Seer-Large]]: CALVIN 强基线，AffordanceVLA 以 4.33 超越其 4.28
- [[Pi0]]: 最主要对比基线，真实环境中 AffordanceVLA 大幅领先

### 方法相关
- [[可供性|Affordance]]: 论文的核心概念
- [[混合专家架构|Mixture of Experts]]: MoT 的基础架构思想
- [[扩散策略|Diffusion Policy]]: How2Act 形状生成和动作生成的基础
- [[中间表示|Intermediate Representation]]: 设计哲学的核心
- [[Action Chunking]]: 动作输出形式

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评测基准
- [[CALVIN]]: 跨域长程泛化基准
- [[DROID]]: 真实机器人数据集

---

## 速查卡片

> [!summary] AffordanceVLA (arXiv 2606.06155)
> - **核心**: 以结构化可供性（Which/Where/How2Act）为中间监督，桥接 VLM 语义与机器人物理控制
> - **方法**: Mixture-of-Transformer + 三专家单向因果注意力 + 三阶段渐进训练
> - **结果**: LIBERO 95.8%、CALVIN 4.33、真实环境 88.3%（vs Pi0 73.3%）
> - **代码**: [GitHub](https://github.com/Skywalker-yqz/AffordanceVLA/)

---

*笔记创建时间: 2026-06-06*

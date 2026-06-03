---
title: "SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models"
method_name: "SKIP"
authors: [Ziheng He, Yixiang Chen, Ning Yang, Zhanqian Wu, Qisen Ma, Yuan Xu, Jiabing Yang, Peiyan Li, Xiangnan Wu, Xiaofeng Wang, Zheng Zhu, Jing Liu, Nianfeng Liu, Yan Huang]
year: 2026
venue: arXiv
tags: [world-model, keyframe-selection, video-diffusion, embodied-ai, robot-manipulation, interpolation, policy-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.00664
created: 2026-06-03
---

# 论文笔记：SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UCAS, CASIA, NJU, GigaAI, THU, FiveAges |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[KeyWorld]], [[RoboEnvision]], [[IRASim]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.00664) |

---

## 一句话总结

> SKIP 通过稀疏关键帧生成替代密集帧逐帧视频合成，在 4.16× 加速的同时将 FVD 降低 89%，生成视频可直接替代真实演示用于策略训练，成功率仅下降 1.3 个百分点。

---

## 核心贡献

1. **事件保留稀疏生成 (Event-Preserving Sparse Generation)**: 证明离散操作事件关键帧可以替代密集帧合成，同时支持策略训练和行为验证，避免长程漂移问题
2. **三模块架构 SKIP**: SKIP-Selector（多模态关键帧选择）+ SKIP-Generator（稀疏关键帧扩散生成）+ SKIP-Reconstructor（动作条件密集恢复），形成完整的稀疏到密集视频生成流水线
3. **AC-FILM 动作条件插值**: 基于 Flow 的插值器，使用逐层 [[FiLM]] 调制将动作上下文转换为调制参数，并通过门控抑制静态阶段的插值误差

---

## 问题背景

### 要解决的问题

[[Embodied World Model|具身世界模型]] 通常需要生成密集的视频序列（逐帧或按块递归），计算代价极高。尤其对于机器人操作任务，视频包含大量静态背景帧，而仅有少数关键操作事件帧（如抓取、放置）是对策略有意义的。

### 现有方法的局限

- **密集递归生成**（如 [[IRASim]]）：逐块滚动推理，长程积累漂移误差，推理速度慢
- **基于语言子任务的关键帧方法**（如 [[RoboEnvision]]）：依赖语言标注，需要额外标注成本
- **几何关键帧方法**（如 [[KeyWorld]]，使用 [[Ramer-Douglas-Peucker|RDP]] 算法）：仅考虑轨迹几何形状，忽略操作语义事件，难以捕获夹爪关闭/开启等关键时刻

### 本文的动机

机器人操作任务天然是**事件驱动的稀疏结构**：轨迹中大多数帧变化微小，只有接触/抓取/放置等离散事件帧包含关键语义信息。若能准确识别并仅生成这些事件关键帧，再通过学习插值恢复密集视频，则可在维持策略训练质量的同时大幅降低生成计算量。

---

## 方法详解

### 模型架构

SKIP 采用**三阶段串联**架构，将密集视频生成解耦为关键帧选择、关键帧合成、密集恢复三步：

- **输入**: 初始观测 $o_0$ + 语言指令 $l$ + （可选）动作序列 $\{a_t\}$
- **SKIP-Selector**: 多模态特征融合 → [[Kernel Temporal Segmentation|核时序分割]] → 事件关键帧集合 $\mathcal{K}$
- **SKIP-Generator**: 微调 [[Wan2.2-TI2V]] 视频扩散模型 → 合成稀疏关键帧
- **SKIP-Reconstructor**: 间隔预测器 + [[AC-FILM]] 插值 → 密集视频帧序列
- **总关键帧数**: $K_{kf} = 41$（通过收敛分析确定）

### 核心模块

#### 模块1: SKIP-Selector — 事件保留关键帧选择

**设计动机**: 利用 [[多模态特征融合]] 捕获视觉、语义和本体感知三种互补信息，保证关键操作事件帧不遗漏。

**具体实现**:
- **视觉流 (V)**: [[光流|Optical Flow]] 特征，捕获机械臂运动信号
- **语义流 (S)**: 冻结 [[DINO]] 特征，检测场景语义变化
- **本体感知流 (P)**: 接触事件信号，捕获视觉不可见的夹爪状态变化

三路特征分别构建 [[RBF 核相似矩阵]]，经过 center + trace-normalize 后等权平均，得到融合相似矩阵 $\hat{G}$。

然后用 [[核时序分割 (KTS)]] 动态规划最优分割任务阶段，并在分割后强制执行 **夹爪事件后处理**（gripper-event post-processing），确保夹爪开合时刻 ±2 帧必须被覆盖。

#### 模块2: SKIP-Generator — 稀疏关键帧扩散合成

**设计动机**: 利用大规模预训练的 [[视频扩散模型]] 对关键帧分布进行高质量建模，仅从初始帧和语言指令生成有序关键帧，无需访问未来动作。

**具体实现**:
- 基于 [[Wan2.2-TI2V-5B]] 微调（仅 [[DiT]] backbone 可训练，VAE/文本编码器冻结）
- 训练数据：SKIP-Selector 选出的关键帧子集
- 推理时直接生成 $K_{kf}$ 个有序关键帧，无需递归滚动

**训练配置**:
- Batch Size: 1，Epochs: 30
- 分辨率: 256×256
- 优化器: AdamW，学习率: 1e-5

#### 模块3: SKIP-Reconstructor — 动作条件密集恢复

**设计动机**: 利用机器人动作序列作为显式条件，准确恢复关键帧之间的中间帧，同时避免静态阶段的过度插值。

**两阶段恢复**:
1. **间隔预测器 (Gap Predictor)**: 用冻结图像编码器拼接相邻关键帧特征，回归两帧之间的中间帧数 $\Delta_t$
2. **动作条件插值 [[AC-FILM]]**: 基于 [[Flow-based Interpolation]] 的插值网络，通过逐层 [[FiLM]] 调制将动作上下文注入特征图；门控机制 $s = \sigma(w_s \bar{m} + b_s)$ 在静态阶段抑制插值残差

**训练配置**:
- Batch Size: 4，Epochs: 20（1 轮冻结预热 + 19 轮联合训练）
- 损失: $\ell_1$ + [[VGG 感知损失]] + [[Gram 风格损失]]
- Zero-init 保证训练初始为恒等变换

---

## 关键公式

### 公式1: [[RBF 核相似矩阵|RBF 核相似度]]

$$
\hat{G}^{(r)}_{ij} = \exp\!\left(-\frac{\|f^{(r)}_i - f^{(r)}_j\|^2}{2\sigma_r^2}\right)
$$

**含义**: 计算第 $r$ 个模态流中帧 $i$ 和帧 $j$ 特征之间的高斯核相似度，作为时序分割的输入。

**符号说明**:
- $f^{(r)}_i$: 第 $r$ 模态（V/S/P）在帧 $i$ 处的特征向量
- $\sigma_r$: 第 $r$ 模态的带宽超参数
- $\hat{G}^{(r)}_{ij} \in [0,1]$: 归一化后的帧间相似度

### 公式2: [[核时序分割 (KTS)|KTS 分割目标]]

$$
\min_{\{c_k\}} \sum_{k=1}^{L} \text{Var}(\mathcal{S}_k) + \lambda L
$$

**含义**: 最小化各时段内帧特征方差之和，同时对分段数 $L$ 施加惩罚，通过动态规划求最优分段边界。

**符号说明**:
- $\mathcal{S}_k$: 第 $k$ 个时段内的帧集合
- $\text{Var}(\mathcal{S}_k)$: 时段内特征方差（within-segment variance）
- $\lambda$: 分段数惩罚系数
- $L$: 总分段数

### 公式3: [[FiLM|AC-FILM 特征调制]]

$$
F'_\ell = \gamma_\ell \odot F_\ell + \beta_\ell
$$

**含义**: 将动作上下文通过 FiLM 层逐级调制插值网络的特征图，使插值运动与真实动作对齐。

**符号说明**:
- $F_\ell$: 插值网络第 $\ell$ 层特征图
- $\gamma_\ell, \beta_\ell$: 由动作序列 MLP 预测的逐通道缩放和偏移参数
- $\odot$: 逐元素乘法

### 公式4: [[门控机制|AC-FILM 静态门控]]

$$
s = \sigma(w_s \bar{m} + b_s), \quad F'_\ell \leftarrow s \cdot F'_\ell
$$

**含义**: 根据动作幅度均值 $\bar{m}$ 预测一个标量门控值 $s$，在静态阶段（动作幅度小）自动抑制插值残差，避免引入人工伪影。

**符号说明**:
- $\sigma$: Sigmoid 激活函数
- $\bar{m}$: 当前插值窗口内动作向量幅度的均值
- $w_s, b_s$: 可学习的门控参数

---

## 关键图表

### Figure 1: SKIP 系统概览

![Figure 1](https://arxiv.org/html/2606.00664/2606.00664v1/x1.png)

**说明**: SKIP 整体框架。从初始观测和语言指令出发，SKIP-Selector 提取事件保留关键帧，SKIP-Generator 合成稀疏关键帧视频，SKIP-Reconstructor 通过学习间隔预测和动作条件插值恢复密集视频。

### Figure 2: SKIP 详细架构

![Figure 2](https://arxiv.org/html/2606.00664/2606.00664v1/x2.png)

**说明**: 三模块详细架构图。SKIP-Selector 融合视觉流（光流）、语义流（DINO）和本体感知流（接触），经 KTS 分割和夹爪事件后处理得到关键帧索引；SKIP-Generator 基于微调扩散模型合成关键帧；SKIP-Reconstructor 先预测帧间隔再用 AC-FILM 插值填充。

### Figure 3: 策略训练成功率对比

![Figure 3](https://arxiv.org/html/2606.00664/2606.00664v1/x3.png)

**说明**: π₀.₅ 在五种训练数据混合方案下的成功率对比。SKIP-Mix70（70% 合成）在仿真达到 94.9%（全真实 95.8%，仅差 0.9pp）；Dense-Mix100 基线崩溃至 37.5%，验证了稀疏生成避免长程漂移的优势。

### Figure 4: 真实机器人任务概览

![Figure 4](https://arxiv.org/html/2606.00664/2606.00664v1/x4.png)

**说明**: Franka Panda 机械臂的四个真实任务（T1-T4）演示序列，用于真实机器人策略评估。

### Figure 5: 按事件数分组的关键帧质量分析

![Figure 5](https://arxiv.org/html/2606.00664/2606.00664v1/x5.png)

**说明**: 随操作复杂度（夹爪事件数）增加，SKIP-Selector 相对于 Uniform、RDP、TriPSS 的优势扩大，在 5+ 事件桶中提升最显著，证明多模态融合在复杂任务上的必要性。

### Figure 6: 单条轨迹事件覆盖可视化

![Figure 6](https://arxiv.org/html/2606.00664/2606.00664v1/x6.png)

**说明**: 一条含四个夹爪事件的 LIBERO 轨迹上各方法的关键帧选择对比。SKIP-Selector 精确命中全部四个事件，而 Uniform、RDP、TriPSS 均有遗漏。

### Figure 7: 关键帧预算收敛分析

![Figure 7](https://arxiv.org/html/2606.00664/2606.00664v1/x7.png)

**说明**: 关键帧数量 $K_{kf}$ 从 10 增加到 80 时，事件覆盖率和视频质量在 $K_{kf} \approx 41$ 处收敛，超出后边际收益递减，据此确定最终关键帧预算为 41。

### Figure 8: 关键帧布局对密集恢复的定性影响

![Figure 8](https://arxiv.org/html/2606.00664/2606.00664v1/x8.png)

**说明**: 在长序列双物体操作任务上，事件感知关键帧能保留"物体在夹爪中"的状态，而均匀或几何关键帧会错过该状态，导致密集恢复出现物体凭空消失/出现的伪影。

### Figure 9: 附加可视化结果

![Figure 9](https://arxiv.org/html/2606.00664/2606.00664v1/x9.png)

**说明**: 论文附加定性可视化结果。

### Table 1: 关键帧选择质量指标对比

| 方法 | GripCov ↑ | MEC ↑ | MaxSemDist ↑ | P95SemDist ↑ | OAS ↑ |
|------|-----------|-------|--------------|--------------|-------|
| Uniform | 0.801 | — | — | — | 0.762 |
| RDP | 0.864 | — | — | — | 0.848 |
| TriPSS | — | — | — | — | — |
| **SKIP-Selector** | **0.999** | — | — | — | **0.911** |

**说明**: SKIP-Selector 在夹爪事件覆盖率（GripCov=0.999，近乎完美）和综合操作感知分（OAS=0.911）上显著优于所有基线，尤其相比几何方法 RDP 提升明显。

### Table 2: 视频生成质量对比（LIBERO 基准）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | 速度 |
|------|--------|--------|---------|-------|------|
| 密集递归生成 | 21.423 | — | — | 4177 | 1× |
| **SKIP** | **21.635** | — | — | **458** | **4.16×** |

**说明**: SKIP 在 PSNR 上微超密集基线（+0.2dB），FVD 降低 **89%**（4177→458），同时实现 **4.16×** 推理加速，证明稀疏生成在质量和效率上的双重优势。

### Table 3: π₀.₅ 策略训练成功率对比

| 训练数据 | LIBERO 仿真 ↑ | Franka 真实 ↑ |
|----------|--------------|--------------|
| 全真实演示 | 95.8% | 80.0% |
| Dense-Mix100 | 37.5% | 25.0% |
| SKIP-Mix70 (70% 合成) | 94.9% | 73.3% |
| SKIP-Mix100 (100% 合成) | 93.9% | 66.7% |

**说明**: SKIP-Mix70 仅比全真实演示低 0.9pp（仿真）和 6.7pp（真实），而密集生成基线崩溃至 37.5%/25.0%，验证稀疏关键帧生成方案的有效性。

### Table 4: 多模态融合消融实验

| 配置 | LIBERO 仿真成功率 ↑ |
|------|-------------------|
| 仅视觉流 (V) | 71.5% |
| V + 语义流 (S) | 81.2% |
| V + S + 本体感知 (P) | 91.4% |
| **完整 (V+S+P+夹爪后处理)** | **94.9%** |

**关键发现**: 每个模态都贡献显著提升（+9.7pp, +10.2pp, +3.5pp），夹爪事件后处理最终保障极端场景覆盖，三者缺一不可。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 50 任务，各约 50 条演示 | 桌面操作，含多物体、长序列任务 | 关键帧选择评估 + 策略训练 |
| Franka Panda 真实数据 | 4 任务 (T1–T4)，各 15 条 | 真实世界抓取/放置任务 | 真实机器人策略验证 |

### 实现细节

- **策略骨干**: [[π₀.₅ (Pi0.5)]]，使用 30,000 步 cosine 调度微调，峰值学习率 5e-5，Batch Size 32
- **SKIP-Generator**: AdamW，LR 1e-5，Batch 1，30 epochs，分辨率 256×256，仅 DiT 可训练
- **AC-FILM**: Batch 4，20 epochs（1 冻结 + 19 联合），零初始化，$\ell_1$ + VGG 感知 + Gram 风格损失
- **时序分割**: KTS 动态规划，优于谱聚类/凝聚聚类/贝叶斯变点等替代方案（+0.02~0.09 OAS）
- **关键帧预算**: $K_{kf} = 41$，收敛分析确定

### 可视化结果

- 事件感知关键帧在复杂多步任务上显著减少"物体凭空消失"伪影
- SKIP-Generator 生成的视频视觉一致性高，LIBERO FVD 由 4177 降至 458
- AC-FILM 在静态阶段准确抑制插值残差，在运动阶段精确对齐动作轨迹

---

## 批判性思考

### 优点
1. **问题定义清晰**: 将操作任务的稀疏事件结构显式建模，理论动机充分
2. **三模态融合合理**: 视觉/语义/本体感知互补，夹爪后处理保障极端情况，消融验证充分
3. **实际可部署性强**: 4.16× 加速 + 仅 1.3pp 策略降级，具有实际工程价值

### 局限性
1. **场景限制强**: 假设固定第三人称视角 + 单臂操作，难以直接推广到多视角/双臂/移动操作
2. **评估数据集偏窄**: 仅在 LIBERO + 4 个真实任务上验证，任务多样性有限，泛化性存疑
3. **长复合任务误差积累**: 多阶段任务中出现外观漂移，重建误差仍是开放问题
4. **依赖动作标注**: AC-FILM 插值需要完整动作序列，在无动作标注场景下退化

### 潜在改进方向
1. 扩展到多视角和双臂操作，验证多相机场景下的鲁棒性
2. 直接在稀疏关键帧上训练 VLA 模型，探索是否减少计算同时维持性能
3. 引入在线关键帧预算自适应调整，根据任务复杂度动态确定 $K_{kf}$

### 可复现性评估
- [ ] 代码开源（论文中未提及）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（详尽的超参数表格）
- [x] 数据集可获取（LIBERO 公开可用）

---

## 关联笔记

### 基于
- [[Wan2.2-TI2V]]: SKIP-Generator 的基础视频扩散模型
- [[π₀.₅ (Pi0.5)]]: 下游策略评估使用的 VLA 模型
- [[核时序分割 (KTS)]]: 关键帧分割的核心算法

### 对比
- [[KeyWorld]]: 使用 RDP 几何关键帧，无法覆盖语义事件
- [[RoboEnvision]]: 依赖语言子任务标注，SKIP 无需额外标注
- [[IRASim]]: 密集递归生成基线，FVD 4177 vs SKIP 的 458

### 方法相关
- [[FiLM]]: AC-FILM 插值的核心调制机制
- [[DINO]]: 语义流特征提取器
- [[光流 (Optical Flow)]]: 视觉流特征来源
- [[视频扩散模型]]: SKIP-Generator 的生成范式

### 硬件/数据相关
- [[Franka Panda]]: 真实机器人实验平台
- [[LIBERO]]: 仿真评估 benchmark

---

## 速查卡片

> [!summary] SKIP: Sparse Keyframe Interpolation Paradigm
> - **核心**: 稀疏关键帧生成 + 动作条件密集插值，替代密集递归视频生成
> - **方法**: SKIP-Selector（多模态KTS关键帧）→ SKIP-Generator（Wan2.2扩散）→ SKIP-Reconstructor（AC-FILM插值）
> - **结果**: 4.16× 加速，FVD -89%，策略成功率仅降 1.3pp（仿真）/ 6.7pp（真实）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-03*

---
title: "PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation"
method_name: "PiL-World"
authors: [Chong Ma, Taiyi Su, Jian Zhu, Jianjun Zhang, Zitai Huang, Yi Xu, Hanli Wang]
year: 2026
venue: arXiv
tags: [world-model, vla-evaluation, closed-loop, robot-manipulation, video-generation]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.05773
created: 2026-06-06
---

# 论文笔记：PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tongji University; Midea Group AIRC |
| 日期 | June 2026 |
| 项目主页 | - |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05773) / Code: - |

---

## 一句话总结

> PiL-World 是首个支持 VLA 策略闭环评估的分块世界模型，通过动作视觉控制投影与潜在历史记忆，将仿真与真实成功率的差距从 63.2% 压缩到 12.0%。

---

## 核心贡献

1. **Policy-in-the-Loop 评估范式**: 将世界模型生成的观测反馈给 [[VLA]] 策略，实现真正意义上的闭环评估，而非沿预收集轨迹的开环预测
2. **Action-to-Control 投影模块**: 利用确定性机器人运动学与相机投影，将关节空间动作转化为头部视角视觉控制帧（gripper marker），精准条件化世界模型生成
3. **成功失败混合训练策略**: 引入失败演示轨迹参与微调，使世界模型分布更贴近真实策略执行分布，显著提升评估可信度
4. **Hallucination-Free Ratio（HFR）指标**: 提出量化 rollout 可信度持续时间的新评估指标，补充成功率之外的质量维度

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 策略评估高度依赖真实机器人执行，成本高且效率低。已有世界模型评估方法多为**开环**预测：沿预收集动作序列逐步预测观测，无法反映策略在生成观测上的实际决策行为。

### 现有方法的局限

- **开环预测**（如 [[Ctrl-World]]）不将生成观测反馈给策略，导致仿真与真实成功率差距高达 63.2%
- 预收集轨迹分布与真实策略执行分布存在显著偏差
- 现有指标（成功率）无法量化生成质量随时间的退化程度（hallucination 问题）

### 本文的动机

若能让世界模型生成的观测**实时驱动** VLA 策略产生下一个动作块，则评估过程可真实模拟闭环部署，从而大幅降低机器人评估成本，同时提升评估与真实结果的一致性。

---

## 方法详解

### 模型架构

PiL-World 采用**分块自回归视频生成**架构，交替运行 [[VLA]] 推理与[[世界模型|World Model]] 预测：

- **输入**: 任务指令 $g$ + 当前多视角观测 $x_t$ + [[潜在历史记忆|Latent History Memory]] $\mathcal{H}_t$
- **条件信号**: [[Action-to-Control 投影|Action-to-Control Projection]] $\Gamma(A_t^{\Delta,K})$
- **核心模块**: [[潜在历史记忆]] 编码任务上下文 + [[视觉控制帧]] 提供动作条件
- **输出**: K 帧步幅对齐的未来多视角观测 $\hat{x}_{t+\Delta:t+K\Delta:\Delta}$
- **预训练数据**: RealSource World（1100 万+ 帧，35 个操作任务）

### 核心模块

#### 模块 1: Action-to-Control 投影（Γ）

**设计动机**: 让[[世界模型]]理解动作与视觉变化之间的精确对应关系，而不是仅靠端到端隐式学习。

**具体实现**:
- 利用确定性[[正向运动学|Forward Kinematics]]将关节空间动作 $a_t$ 转换为三维空间中 gripper 位置
- 通过相机投影矩阵映射到头部视角图像平面
- 以 marker 位置编码 gripper 三维位姿，marker 大小编码 gripper 开合状态
- 输出控制视频 $C_t = \{c_{t+\Delta}, c_{t+2\Delta}, \ldots, c_{t+K\Delta}\}$，作为世界模型生成条件

#### 模块 2: 潜在历史记忆（$\mathcal{H}_t$）

**设计动机**: 多轮预测中防止[[视频生成漂移|Generation Drift]]，保留任务上下文信息。

**具体实现**:
- [[VAE]] 编码器分别处理当前帧、历史序列帧（最近 $H_h=5$ 帧）和目标未来帧
- 三类 latent 分离编码，各司其职：
  - $Z_t^{v,0} = E_\phi(x_t^v)$：当前帧 latent
  - $Z_t^{v,h} = E_\phi(\{x_\tau^v\}_{\tau \in \mathcal{I}_t^h})$：历史帧 latent
  - $Z_t^{v,f} = E_\phi(\{x_\tau^v\}_{\tau=t+\Delta:t+K\Delta:\Delta})$：目标未来帧 latent
- 历史 latent 作为 conditioning 防止遗忘，抑制长程 hallucination

#### 模块 3: 分块预测与闭环 Rollout

**设计动机**: 将 [[Action Chunking]] 与世界模型分块预测对齐，使策略与世界模型能够交替运行。

**具体实现**:
- 每个预测步生成 $K=15$ 帧，步幅 $\Delta=3$（对应原始 45 个时间步）
- 最终生成帧 $\hat{x}_{t+K\Delta}$ 作为终止观测，反馈给 VLA 策略进行下一轮推理
- 最多 $R=5$ 轮闭环 rollout，评估跨轮次的策略一致性

---

## 关键公式

### 公式 1: [[VLA Policy|VLA 动作块预测]]

$$
A_t = \{a_{t+1}, a_{t+2}, \ldots, a_{t+H_\pi}\} = \pi(x_t, s_t, g)
$$

**含义**: VLA 策略 $\pi$ 基于当前观测 $x_t$、状态 $s_t$ 和任务目标 $g$，一次性预测 $H_\pi=50$ 步动作块。

**符号说明**:
- $A_t$：在时刻 $t$ 预测的动作块
- $H_\pi = 50$：动作块长度
- $\pi$：VLA 策略网络
- $x_t$：多视角观测
- $s_t$：机器人状态
- $g$：任务指令

### 公式 2: [[步幅对齐动作采样|步幅对齐动作子集]]

$$
A_t^{\Delta,K} = \{a_{t+\Delta}, a_{t+2\Delta}, \ldots, a_{t+K\Delta}\}
$$

**含义**: 从完整动作块中以步幅 $\Delta$ 采样 $K$ 个关键帧动作，对齐世界模型的分块预测频率。

**符号说明**:
- $\Delta = 3$：采样步幅（每隔 3 步取一个动作）
- $K = 15$：采样数量（对应 $K \times \Delta = 45$ 个原始时间步）

### 公式 3: [[World Model Prediction|世界模型分块预测]]

$$
\hat{x}_{t+\Delta:t+K\Delta:\Delta} \sim W_\theta(\cdot \mid x_t,\, \mathcal{H}_t,\, \Gamma(A_t^{\Delta,K}),\, g)
$$

**含义**: 世界模型 $W_\theta$ 以当前观测、历史记忆、动作控制信号和任务指令为条件，预测未来 $K$ 帧多视角观测序列。

**符号说明**:
- $W_\theta$：PiL-World 世界模型（参数 $\theta$）
- $\mathcal{H}_t$：潜在历史记忆
- $\Gamma(\cdot)$：Action-to-Control 投影函数
- $\hat{x}_{t+\Delta:t+K\Delta:\Delta}$：预测的步幅对齐观测序列

### 公式 4: [[VAE 潜在编码|三类潜在编码]]

$$
\begin{aligned}
Z_t^{v,0} &= E_\phi(x_t^v) \\
Z_t^{v,h} &= E_\phi\!\left(\{x_\tau^v\}_{\tau \in \mathcal{I}_t^h}\right) \\
Z_t^{v,f} &= E_\phi\!\left(\{x_\tau^v\}_{\tau=t+\Delta:t+K\Delta:\Delta}\right)
\end{aligned}
$$

**含义**: VAE 编码器 $E_\phi$ 对当前帧、历史帧和目标未来帧分别编码，三类 latent 联合条件化扩散模型生成过程。

**符号说明**:
- $E_\phi$：VAE 编码器
- $v$：视角索引（头部视角 / 左腕视角 / 右腕视角）
- $\mathcal{I}_t^h$：历史帧时间索引集合（最近 $H_h=5$ 帧）

### 公式 5: [[扩散生成损失|生成训练损失]]

$$
\mathcal{L}_{\text{gen}} = \mathbb{E}_{t,\lambda,\varepsilon}\!\left[\left\|\Psi_\theta\!\left(\tilde{Z}_{t,\lambda}^f,\, \lambda,\, C_t\right) - u_{t,\lambda}\right\|_2^2\right]
$$

**含义**: 标准流匹配（Flow Matching）损失，训练去噪网络 $\Psi_\theta$ 预测从噪声 latent 到目标 latent 的速度场。

**符号说明**:
- $\Psi_\theta$：去噪/速度场网络（参数 $\theta$）
- $\tilde{Z}_{t,\lambda}^f$：噪声化后的目标 latent
- $\lambda$：噪声水平（时间步）
- $C_t = \Gamma(A_t^{\Delta,K})$：动作控制视频
- $u_{t,\lambda}$：目标速度场

---

## 关键图表

### Figure 1: 方法动机与评估范式对比

![Figure 1](https://arxiv.org/html/2606.05773v1/Figures/Figure1.png)

**说明**: (a) Policy-in-the-Loop 闭环流程：世界模型预测观测 → 反馈给 VLA 策略 → 策略输出下一个动作块。(b) 开环 vs 闭环：开环沿固定动作序列预测，闭环中生成观测实时更新策略输入。(c) 一致 vs 不一致 rollout：有效 rollout 应与真实执行过程保持一致。

### Figure 2: PiL-World 整体架构

![Figure 2](https://arxiv.org/html/2606.05773v1/Figures/Figure2.jpg)

**说明**: (a) 闭环 rollout 流程：[[Action Chunking]] 动作块经 Action-to-Control 投影生成控制视频，世界模型预测未来多视角观测，终止帧反馈给策略进行下轮推理。(b) Action-to-Control 投影细节：关节动作 → 正向运动学 → 图像平面 → gripper marker。(c) 成功/失败混合微调数据。(d) 潜在历史记忆：近期多视角帧编码为历史 latent，条件化未来生成。

### Figure 3: 跨 Checkpoint 真实与想象成功率一致性

![Figure 3](https://arxiv.org/html/2606.05773v1/Figures/rollout_success_trend.png)

**说明**: 每个点代表一个任务-checkpoint 组合，形状表示任务，颜色表示 checkpoint。PiL-World（Pearson 相关系数 0.94）与真实成功率高度一致，而 Ctrl-World 偏差显著。

### Figure 4: 单步 LPIPS 增益（Ground-Truth 动作条件）

![Figure 4](https://arxiv.org/html/2606.05773v1/Figures/lpips_gain_pilworld_square_legend_v3.png)

**说明**: 计算 Ctrl-World LPIPS 减 PiL-World LPIPS 的差值，正值表示 PiL-World 生成质量更优。Sort Cubes 改善 33.7%，Stack Bowls 19.5%，Stack Blocks 5.4%。

### Figure 5a: Sort Cubes Rollout 示例

![Figure 5a](https://arxiv.org/html/2606.05773v1/Figures/sortcubes_example.png)

**说明**: Sort Cubes 任务上的完整闭环 rollout 可视化示例，展示多视角生成质量与真实执行的对齐程度。

### Figure 5b: Stack Bowls Rollout 示例

![Figure 5b](https://arxiv.org/html/2606.05773v1/Figures/stackbowls_example.png)

**说明**: Stack Bowls 任务上的闭环 rollout 示例，展示叠碗任务中 gripper marker 控制与生成观测的一致性。

### Figure A.1: 目标任务示例与子任务分割

![Figure A.1](https://arxiv.org/html/2606.05773v1/Figures/task_introduce.png)

**说明**: Sort Cubes、Stack Bowls、Stack Blocks 三个任务的典型场景图及子任务分割方式，所有任务均为双臂操作。

### Figure A.2: Action-to-Control 投影流程

![Figure A.2](https://arxiv.org/html/2606.05773v1/Figures/action_projection.png)

**说明**: 详细展示从关节空间动作到视觉 gripper marker 的完整投影流程：关节动作 → 正向运动学 → 三维 gripper 位置 → 头部视角投影 → marker 编码。

### Figure A.3: RealSource World 单步预测定性对比

![Figure A.3](https://arxiv.org/html/2606.05773v1/Figures/realsource_qualitative_grid.png)

**说明**: 多组 ground-truth vs 预测对比，涵盖头部视角、左腕视角和右腕视角，展示预训练世界模型在 35 个操作任务上的多视角生成质量。

### Table 1: 真实对齐的闭环 Rollout 评估（40k 步 checkpoint）

| 任务 | 方法 | SR_real | SR_imag | \|ΔSR\| | HFR |
|------|------|---------|---------|---------|-----|
| Sort Cubes | Ctrl-World | 83.3% | 11.5% | 71.8% | 39.5% |
| Sort Cubes | **PiL-World** | 83.3% | **68.3%** | **15.0%** | **83.3%** |
| Stack Bowls | Ctrl-World | 96.7% | 24.1% | 72.6% | 47.4% |
| Stack Bowls | **PiL-World** | 96.7% | **92.5%** | **4.2%** | **83.9%** |
| Stack Blocks | Ctrl-World | 50.0% | 4.9% | 45.1% | 37.7% |
| Stack Blocks | **PiL-World** | 50.0% | **33.3%** | **16.7%** | **43.0%** |

**关键发现**: PiL-World 将三任务平均 \|ΔSR\| 从 63.2% 降至 12.0%，HFR 从 41.5% 提升至 70.1%。

### Table 2: 单步视觉预测 LPIPS（Ground-Truth 动作条件）

| 任务 | 方法 | Overall LPIPS ↓ | Head LPIPS ↓ | Wrist Avg. LPIPS ↓ |
|------|------|----------------|-------------|-------------------|
| Sort Cubes | Ctrl-World | 0.1454 | 0.1030 | 0.1666 |
| Sort Cubes | **PiL-World** | **0.0965** | **0.0597** | **0.1148** |
| Stack Bowls | Ctrl-World | 0.1366 | 0.0959 | 0.1569 |
| Stack Bowls | **PiL-World** | **0.1100** | **0.0597** | **0.1351** |
| Stack Blocks | Ctrl-World | 0.1277 | 0.0885 | 0.1474 |
| Stack Blocks | **PiL-World** | **0.1208** | **0.0617** | **0.1503** |

**关键发现**: PiL-World 在头部视角 LPIPS 上改善最为显著（受益于 Action-to-Control 投影），腕部视角改善相对较小。

### Table A.1: 目标任务数据统计

| 任务 | 训练轨迹数 | 训练 Clips | 测试轨迹数 | 测试子任务数 | 测试 Clips |
|------|-----------|-----------|-----------|------------|-----------|
| Sort Cubes | 100 | 8,707 | 20 | 60 | 1,830 |
| Stack Bowls | 100 | 7,342 | 18 | 54 | 1,301 |
| Stack Blocks | 200 | 15,626 | 20 | 40 | 820 |

### Table A.2: 潜在历史记忆消融实验

| 任务 | 方法 | Overall LPIPS ↓ | Head LPIPS ↓ | Wrist Avg. LPIPS ↓ |
|------|------|----------------|-------------|-------------------|
| Sort Cubes | **PiL-World** | **0.0965** | **0.0597** | **0.1148** |
| Sort Cubes | w/o history | 0.3146 | 0.3176 | 0.3131 |
| Stack Bowls | **PiL-World** | **0.1100** | **0.0597** | **0.1351** |
| Stack Bowls | w/o history | 0.2759 | 0.3333 | 0.2472 |
| Stack Blocks | **PiL-World** | **0.1208** | **0.0617** | **0.1503** |
| Stack Blocks | w/o history | 0.3403 | 0.3473 | 0.3367 |

**关键发现**: 移除潜在历史记忆后 LPIPS 大幅劣化（3x+），证明历史记忆对长程预测质量至关重要。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RealSource World | 11,428 episodes，1400 万+ 帧，35 个任务 | 多视角双臂操作，多样化任务 | 世界模型预训练 |
| Sort Cubes（目标任务） | 100 训练轨迹，8,707 clips | 包含成功与失败演示 | 微调 + 评估 |
| Stack Bowls（目标任务） | 100 训练轨迹，7,342 clips | 包含成功与失败演示 | 微调 + 评估 |
| Stack Blocks（目标任务） | 200 训练轨迹，15,626 clips | 包含成功与失败演示 | 微调 + 评估 |

### 实现细节

- **基础架构**: 基于 [[Ctrl-World]] 架构扩展，使用 [[LoRA]] 进行高效参数微调
- **输入/输出分辨率**: 224×224
- **预测参数**: $K=15$ 帧，步幅 $\Delta=3$，历史帧 $H_h=5$，最大 rollout 轮次 $R=5$
- **VLA 动作块长度**: $H_\pi = 50$
- **预训练**: 22 epochs，64 块 H20 GPU
- **微调**: 20 epochs，8 块 H20 GPU
- **人工标注**: 3 名标注员（M=3）评估成功率和 hallucination 起始点

### 可视化结果

Sort Cubes 和 Stack Bowls 的 rollout 示例（Figure 5）表明：PiL-World 生成的多视角观测与真实执行高度一致，gripper marker 准确追踪真实 gripper 运动轨迹；而 Ctrl-World 在早期轮次即出现明显 hallucination。Stack Blocks 因接触丰富（contact-rich）而挑战更大，改善幅度相对有限。

---

## 批判性思考

### 优点

1. **评估范式创新**: Policy-in-the-Loop 闭环评估是对开环评估的根本性升级，更接近真实部署场景
2. **工程严谨**: Action-to-Control 投影利用确定性运动学（而非端到端学习），避免了投影过程中的歧义
3. **失败数据利用**: 将失败轨迹纳入微调是一个务实且有效的设计，贴近真实策略执行分布
4. **量化指标 HFR**: 新指标能够量化 rollout 质量的时间维度，而不仅仅是终态成功率

### 局限性

1. **任务规模有限**: 仅在 3 个双臂操作任务上验证，泛化性待确认
2. **接触丰富任务表现较弱**: Stack Blocks（含大量接触）的改善幅度（5.4% LPIPS）远小于其他任务
3. **HFR 依赖人工标注**: 自动化困难，规模化评估成本高
4. **头部视角投影局限**: 在严重遮挡条件下，基于单视角的 marker 投影可能失效

### 潜在改进方向

1. 扩展到更多元化任务（单臂、灵巧手、移动操作）以验证泛化性
2. 开发自动化 hallucination 检测替代人工标注 HFR
3. 将腕部视角也纳入 Action-to-Control 投影以改善腕部视角预测质量
4. 探索与[[仿真器]]结合的混合评估策略（Sim+WM）

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（暂无）
- [x] 训练细节完整（论文和附录中有较完整描述）
- [ ] 数据集可获取（RealSource World 未公开）

---

## 关联笔记

### 基于
- [[Ctrl-World]]: 直接对比的开环世界模型 baseline，PiL-World 在其架构基础上扩展
- [[Action Chunking]]: VLA 策略输出动作块的核心范式，PiL-World 与之对齐设计

### 对比
- [[Ctrl-World]]: 开环 vs 闭环评估的核心对比对象

### 方法相关
- [[世界模型]]: PiL-World 的核心技术范畴
- [[VLA]]: 被评估的策略类型
- [[扩散模型]]: 世界模型视频生成基础架构
- [[VAE]]: 潜在空间编码模块
- [[LoRA]]: 高效微调方法
- [[Flow Matching]]: 训练损失所基于的生成建模框架
- [[正向运动学]]: Action-to-Control 投影的核心算法

### 硬件/数据相关
- [[双臂机器人]]: 评估平台
- [[RealSource World]]: 预训练数据集

---

## 速查卡片

> [!summary] PiL-World (2026)
> - **核心**: 首个 VLA Policy-in-the-Loop 闭环评估世界模型
> - **方法**: Action-to-Control 投影 + 潜在历史记忆 + 成功/失败混合训练
> - **结果**: \|ΔSR\| 从 63.2% → 12.0%，HFR 从 41.5% → 70.1%，Pearson 相关 0.94
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-06-06*

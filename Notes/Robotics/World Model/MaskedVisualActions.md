---
title: "Masked Visual Actions for Unified World Modeling"
method_name: "MaskedVisualActions"
authors: [Hadi Alzayer, Wenlong Huang, Haonan Chen, Christopher Luey, Lvmin Zhang, Maneesh Agrawala, Gordon Wetzstein, Li Fei-Fei, Yilun Du, Jiajun Wu, Jia-Bin Huang]
year: 2026
venue: arXiv
tags: [world-model, video-generation, robot-manipulation, forward-dynamics, inverse-dynamics, test-time-scaling, policy-evaluation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.19343
created: 2026-07-23
---

# 论文笔记：Masked Visual Actions for Unified World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Stanford University, University of Maryland College Park, Harvard University |
| 日期 | July 2026 |
| 项目主页 | [masked-visual-actions.github.io](https://masked-visual-actions.github.io) |
| 对比基线 | [[Ctrl-World]], [[Wan-Move]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.19343) |

---

## 一句话总结

> 用像素空间中的"部分遮罩轨迹"作为统一控制接口，让同一个视频生成模型既能做前向动力学预测，又能做逆向动作提取，无需重新训练即可跨机器人本体迁移。

---

## 核心贡献

1. **统一控制接口**: 提出 [[视觉掩码动作|Masked Visual Actions]]，在像素空间用灰色遮罩表达机器人动作，揭示机器人轨迹 → 前向模型，揭示目标物体轨迹 → 逆向模型，同一个检查点完成两种推理
2. **跨本体泛化**: 像素对齐的密集条件化比稀疏骨架/末端执行器表示具有更强的跨机器人泛化能力，对未见的双臂机器人 R1-Pro 仍有效
3. **极简微调**: 仅用 15 小时真实（[[DROID]]）与仿真（[[RoboCasa]]）视频的 [[LoRA]] 微调即可赋予 [[Wan 2.2 A14B]] 世界模型能力，支持规划、策略评估、动作提取三类应用

---

## 问题背景

### 要解决的问题

机器人学习需要一个通用的[[世界模型|World Model]]：既能前向预测执行某动作后的场景变化（[[前向动力学模型|Forward Dynamics Model]]），又能从期望的物体运动反推机器人应该做什么动作（[[逆向动力学模型|Inverse Dynamics Model]]）。现有方法将二者分开训练，且动作表示都是低维向量，无法跨机器人本体迁移。

### 现有方法的局限

- **低维动作向量**（UVA, UWM, AIM, X-WAM）：掩码的是模态通道而非空间区域，动作仍是本体相关的向量，无法跨本体迁移
- **稀疏视觉条件化**（骨架可视化、末端执行器轨迹、深度点图）：遇到非标准夹爪、未见本体时质量急剧下降，产生幻觉
- **文本/关键点控制**：与视频模型预训练的视觉先验不对齐，泛化能力有限

### 本文的动机

视频生成模型在像素空间学到了丰富的物理先验。如果把机器人动作也表达在像素空间（以灰色遮罩轨迹的形式），就能直接复用这些先验，同时遮罩不携带本体特异的坐标，自然实现跨本体迁移。

---

## 方法详解

### 模型架构

**Masked Visual Actions** 采用**条件视频生成**架构：

- **输入**: 初始帧 $I_0$ + 遮罩视频（灰色背景 + 已揭示的实体轨迹）
- **Backbone**: [[Wan 2.2 A14B]]（14B 参数视频生成模型）
- **核心模块**: [[视觉掩码动作|Masked Visual Actions]] + [[LoRA]]（rank 256）微调
- **输出**: 完整的未来视频帧序列（包含被遮罩实体的预测运动）
- **总参数**: 14B（仅 LoRA 权重被更新）

### 实体世界模型形式化

将场景视频 $V$ 建模为多个**实体轨迹**的联合分布：

$$
p(V) = p(e_1, e_2, \ldots, e_n)
$$

其中 $e_i$ 是第 $i$ 个实体（机器人手臂、操作物体等）随时间的轨迹。给定子集 $\mathcal{S}$ 中的实体条件，预测其余实体：

$$
p\!\left(\{e_i\}_{i \notin \mathcal{S}} \mid \{e_j\}_{j \in \mathcal{S}},\, I_0\right)
$$

**遮罩定义**：将已知实体集合 $\mathcal{S}$ 的像素轨迹并集作为条件掩码：

$$
M(\mathcal{S}) = \bigcup_{i \in \mathcal{S}} e_i
$$

### 核心模块

#### 模块 1：前向动力学模型（Forward Dynamics Model）

**设计动机**：利用[[视觉掩码动作|机器人轨迹掩码]]作为主动实体（Active），预测被动实体（Passive，即操作物体）的响应。

$$
p\!\left(\{e_i\}_{i \in \mathcal{P}} \mid \{e_j\}_{j \in \mathcal{A}},\, I_0\right)
$$

其中 $\mathcal{A}$ = 主动实体集合（机器人），$\mathcal{P}$ = 被动实体集合（物体）

**具体实现**:
- 使用 [[SAM]] 3 从 [[DROID]] 真实演示中分割机器人掩码（约 1,000 个演示）
- 使用渲染方法从 [[RoboCasa]] 仿真中生成机器人可视化（约 4,000 个样本）
- 揭示机器人轨迹区域，其余区域用均匀灰色遮罩

#### 模块 2：逆向动力学模型（Inverse Dynamics Model）

**设计动机**：通过条件交换，用期望的物体运动轨迹反推机器人动作，无需单独训练。

$$
p\!\left(\{e_i\}_{i \in \mathcal{A}} \mid \{e_j\}_{j \in \mathcal{P}},\, I_0\right)
$$

**具体实现**:
- 揭示目标物体轨迹，遮罩机器人区域 → 模型自动综合出能产生该物体运动的机器人动作
- 生成的机器人轨迹再通过[[逆向动力学模型|学习到的逆向动力学模块]]转换为关节空间动作

#### 模块 3：数据集构建（两路 Pipeline）

**分割路（Segmentation-based）**：
- 输入：[[DROID]] 真实操作视频（~1,000 次演示）
- 使用 [[SAM]] 3 提取机器人掩码
- 优势：真实外观、真实物理

**渲染路（Rendering-based）**：
- 输入：[[RoboCasa]] 仿真录制数据（~4,000 个样本）
- 从记录的关节状态渲染机器人可视化
- 优势：精确掩码、多样本体

---

## 关键公式

### 公式 1：[[世界模型|实体联合分布]]

$$
p(V) = p(e_1, e_2, \ldots, e_n)
$$

**含义**：将视频帧序列 $V$ 分解为场景中所有可辨识实体轨迹的联合概率，世界模型即这一分布的估计器。

**符号说明**：
- $V$：完整视频（帧序列）
- $e_i$：第 $i$ 个实体随时间的完整轨迹（像素级别）
- $n$：场景中实体总数

### 公式 2：[[视觉掩码动作|条件预测分布]]

$$
p\!\left(\{e_i\}_{i \notin \mathcal{S}} \mid \{e_j\}_{j \in \mathcal{S}},\, I_0\right)
$$

**含义**：给定已知实体子集 $\mathcal{S}$ 的轨迹和初始帧，预测其余实体的未来运动。调换 $\mathcal{S}$ 的内容即可在前向/逆向模型之间切换。

**符号说明**：
- $\mathcal{S}$：已揭示的实体索引集合（条件部分）
- $I_0$：初始场景帧（上下文）
- $\{e_i\}_{i \notin \mathcal{S}}$：待预测的实体轨迹

### 公式 3：[[视觉掩码动作|像素遮罩定义]]

$$
M(\mathcal{S}) = \bigcup_{i \in \mathcal{S}} e_i
$$

**含义**：将所有已知实体的像素轨迹取并集，得到输入给视频模型的条件掩码；未被覆盖的像素区域用均匀灰色填充。

**符号说明**：
- $M(\mathcal{S})$：合并后的像素掩码
- $e_i$：第 $i$ 个实体在所有帧中占据的像素集合

### 公式 4：[[前向动力学模型|前向模型推理]]

$$
p\!\left(\{e_i\}_{i \in \mathcal{P}} \mid \{e_j\}_{j \in \mathcal{A}},\, I_0\right)
$$

**含义**：揭示主动实体（机器人）轨迹，预测被动实体（物体）的响应，即前向动力学预测。

**符号说明**：
- $\mathcal{A}$：主动实体集合（机器人手臂）
- $\mathcal{P}$：被动实体集合（操作目标物体）

### 公式 5：[[逆向动力学模型|逆向模型推理]]

$$
p\!\left(\{e_i\}_{i \in \mathcal{A}} \mid \{e_j\}_{j \in \mathcal{P}},\, I_0\right)
$$

**含义**：揭示被动实体（物体）的期望运动，反向综合出能产生该运动的机器人轨迹，即逆向动力学推理。

**符号说明**：
- $\mathcal{A}$：需要综合出来的机器人轨迹（预测目标）
- $\mathcal{P}$：已知的目标物体运动（条件）

---

## 关键图表

### Figure 1：Masked Visual Actions 系统概览

![Figure 1 - Overview](https://arxiv.org/html/2607.19343v1/x1.png)

**说明**：Masked Visual Actions 整体框架。微调后的视频模型接受灰色遮罩的实体轨迹作为条件，揭示机器人轨迹时做前向模型（预测物体运动），揭示物体轨迹时做逆向模型（综合机器人动作）。

### Figure 2：动作表示方式对比

![Figure 2 - Action Representations](https://arxiv.org/html/2607.19343v1/x2.png)

**说明**：对比低维动作向量（关节角度）与视觉条件化方法（末端执行器轨迹、骨架可视化、掩码轨迹）的表示空间。Masked Visual Actions 直接在像素空间操作，与视频模型预训练域对齐。

### Figure 3：三类应用场景

![Figure 3 - Applications](https://arxiv.org/html/2607.19343v1/x3.png)

**说明**：前向模型支持：(a) **规划**——对 N 个候选轨迹生成预测视频后用 [[VLM]] 评分选优；(b) **策略评估**——模拟策略展开用于无需真实环境的成功率估计。逆向模型支持：(c) **动作提取**——从目标物体运动中提取可执行的机器人动作。

### Figure 4：数据集构建流程

![Figure 4 - Dataset Construction](https://arxiv.org/html/2607.19343v1/x4.png)

**说明**：左路（分割路）：[[SAM]] 3 从 [[DROID]] 真实视频提取机器人掩码；右路（渲染路）：从 [[RoboCasa]] 仿真关节状态渲染机器人可视化。两路合并提供约 5,000 个带标注的掩码演示。

### Figure 5：未见本体的泛化能力

![Figure 5 - Unseen Embodiment](https://arxiv.org/html/2607.19343v1/x5.png)

**说明**：对比 Masked Actions 与原始关节状态条件化方法在双臂 R1-Pro 机器人（训练集外本体）上的表现。关节状态方法产生严重幻觉伪影；Masked Actions 能维持视觉保真度，体现出密集像素条件的跨本体鲁棒性。

### Figure 6：DROID 数据集上的基线对比

![Figure 6 - DROID Baseline Comparison](https://arxiv.org/html/2607.19343v1/x6.png)

**说明**：在 DROID 测试集上，Masked Visual Actions 对比 Ctrl-World 和轨迹条件化 Wan-Move 的定性生成对比。本方法在已见本体上保持性能的同时，对未见本体展现了更强的泛化能力。

### Figure 7：不同动作条件化信号对比

![Figure 7 - Action Conditioning Signals](https://arxiv.org/html/2607.19343v1/x7.png)

**说明**：跨训练域和本体对比末端执行器轨迹、骨架可视化和像素掩码三种条件化方式。在分布内（same robot, standard gripper）三者表现相近；遭遇分布偏移（自定义夹爪、未见本体）时，稀疏表示急剧退化，像素掩码维持性能。

### Figure 8：规划应用——VLM 评分选优

![Figure 8 - Planning](https://arxiv.org/html/2607.19343v1/x8.png)

**说明**：[[测试时缩放|测试时计算扩展]]流程：[[Diffusion Policy]] 采样 $N=10$ 条候选轨迹 → 各轨迹送入世界模型生成预测视频 → [[VLM]]（Gemini）对每个预测视频评分 → 选择得分最高的轨迹执行。在多个 RoboCasa 任务上提升成功率。

### Figure 9：RoboCasa 策略评估相关性

![Figure 9 - Policy Evaluation Correlation (Sim)](https://arxiv.org/html/2607.19343v1/x9.png)

**说明**：仿真环境中，视频世界模型估计的成功率与真实环境成功率的散点图，相关系数 $r = 0.982$，证明模型可替代真实环境进行策略评估。

### Figure 10：真实世界策略评估

![Figure 10 - Policy Evaluation (Real World)](https://arxiv.org/html/2607.19343v1/x10.png)

**说明**：真实机器人演示中，视频世界模型对任务进度的跟踪与真实标注之间的对齐情况。模型存在轻微正向偏差（倾向于高估任务进度），但总体趋势一致。

### Figure 11：动作提取结果

![Figure 11 - Action Extraction](https://arxiv.org/html/2607.19343v1/x11.png)

**说明**：逆向模型动作提取流程演示：给定目标物体（杯子）运动轨迹，模型综合出与目标物体运动一致的机器人轨迹，再经学习到的[[逆向动力学模型]]转换为关节动作，CoffeeServeMug 任务成功率 90%，与 [[Diffusion Policy]]、[[ACT]]、[[SmolVLA]] 在 100 次演示训练下持平。

### Table 1：视频生成质量定量评估

| 数据集 | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|--------|---------|--------|--------|
| DROID | 0.095 | 0.887 | 23.74 |
| BEHAVIOR | 0.123 | 0.843 | 22.90 |
| Real-World | 0.148 | 0.864 | 22.79 |

**说明**：在三个测试域上，Masked Visual Actions 均优于 Ctrl-World 基线和轨迹条件化 Wan-Move。DROID（训练域）最优，BEHAVIOR 和 Real-World（分布外）亦保持竞争力，体现了跨域泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | ~1,000 次演示 | 真实操作视频，多场景 | 训练（分割路）+ 测试 |
| [[RoboCasa]] | ~4,000 个样本 | 仿真渲染，精确掩码 | 训练（渲染路）|
| BEHAVIOR | 未注明 | 分布外真实视频 | 测试 |
| Real-World | 未注明 | 自定义夹爪真实场景 | 测试 |

### 实现细节

- **Backbone**: [[Wan 2.2 A14B]]（14B 参数，预训练视频生成模型）
- **微调方式**: [[LoRA]]（rank 256）
- **训练步数**: 10,000 steps（约 4 天）
- **Batch Size**: 4
- **硬件**: 8 × NVIDIA H200 GPU
- **条件格式**: 灰色背景 + 已揭示区域，通道拼接输入

### 三类应用实验

**1. 规划（Planning）**
- 基础策略：[[Diffusion Policy]]
- 候选数量：$N = 10$
- 评分器：Gemini [[VLM]]
- 结论：在多个 [[RoboCasa]] 任务上通过[[测试时缩放]]提升成功率

**2. 策略评估（Policy Evaluation）**
- 仿真相关系数：$r = 0.982$（世界模型估计 vs. 真实环境成功率）
- 真实世界：与真实标注趋势一致，轻微正向偏差
- 结论：可替代真实环境进行策略筛选，节省部署成本

**3. 逆向建模 / 动作提取（Inverse Modeling）**
- 任务：CoffeeServeMug
- 本方法（逆向模型）: **90% 成功率**
- [[Diffusion Policy]]（100 次演示）: ~90%
- [[ACT]]（100 次演示）: ~85%
- [[SmolVLA]]（100 次演示）: ~88%
- 结论：无需任务特定训练，与监督策略持平

---

## 批判性思考

### 优点
1. **接口统一性强**: 前向/逆向模型共享同一检查点，无需维护两套参数
2. **本体无关泛化**: 像素遮罩不携带关节角度等本体特异信息，自然迁移到未见机器人
3. **数据效率高**: 仅 5,000 个带遮罩演示 + LoRA 微调，利用大型视频模型的预训练先验

### 局限性
1. **相关性非因果性**: 模型学到的是视频统计相关，而非真正的物理因果关系，在需要精确细微交互的场景中失效
2. **速度瓶颈**: 受限于基础视频模型的生成速度，无法实时用于闭环控制，目前主要适用于离线规划/评估
3. **分辨率/训练数据量限制**: 基础视频模型的能力上限约束了世界模型的表达能力
4. **遮罩噪声**: SAM 3 分割的掩码质量影响训练数据质量，在复杂遮挡场景下可能引入标注错误

### 潜在改进方向
1. **加速推理**: 引入视频扩散蒸馏（如 [[Diffusion Policy]] 的一步推理方案）以支持实时闭环控制
2. **主动实体自动检测**: 目前需要手动指定哪些实体是"主动的"，可以联合学习实体分割与世界模型

### 可复现性评估
- [x] 代码开源（承诺发布）
- [x] 预训练模型（承诺发布）
- [x] 训练细节完整（LoRA rank、训练步数、GPU 配置均有说明）
- [x] 数据集可获取（DROID 和 RoboCasa 均为公开数据集）

---

## 关联笔记

### 基于
- [[Wan 2.2 A14B]]: 基础视频生成模型，提供视觉先验
- [[SAM]]: 用于从真实视频提取机器人分割掩码（SAM 3）
- [[LoRA]]: 参数高效微调方法，rank 256

### 对比
- [[Ctrl-World]]: 最直接的基线，在已见本体上持平，未见本体上本方法显著更优
- [[Diffusion Policy]]: 用于规划应用的基础策略；动作提取的比较基线
- [[ACT]]: 动作提取实验的对比基线
- [[SmolVLA]]: 动作提取实验的对比基线

### 方法相关
- [[视觉掩码动作|Masked Visual Actions]]: 本文提出的核心接口方法
- [[前向动力学模型]]: 条件在主动实体上预测被动实体运动
- [[逆向动力学模型]]: 条件在被动实体运动上推断主动实体动作
- [[测试时缩放]]: 规划应用中通过多候选生成+VLM评分实现的推理时计算扩展
- [[世界模型]]: 本文所构建的世界模型类型

### 硬件/数据相关
- [[DROID]]: 真实操作数据集，提供约 1,000 次演示
- [[RoboCasa]]: 仿真环境，提供约 4,000 个渲染样本

---

## 速查卡片

> [!summary] Masked Visual Actions for Unified World Modeling
> - **核心**: 用像素空间的灰色遮罩轨迹统一前向/逆向世界模型，同一检查点处理两类推理
> - **方法**: Wan 2.2 14B + LoRA (rank 256)，5,000 个带遮罩演示，分割路 + 渲染路数据构建
> - **结果**: 策略评估相关系数 r=0.982；CoffeeServeMug 动作提取 90% 成功率；跨本体泛化优于稀疏条件化
> - **代码**: https://masked-visual-actions.github.io

---

*笔记创建时间: 2026-07-23*

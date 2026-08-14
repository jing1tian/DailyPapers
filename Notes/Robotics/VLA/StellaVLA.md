---
title: "StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models"
method_name: "StellaVLA"
authors: [Siyu Xu, Yunke Wang, Zijian Wang, Dihao Zhu, Chenghao Xia, Chengbin Du, Daochang Liu, Tao Huang, Chang Xu]
year: 2026
venue: arXiv
tags: [vla, in-context-learning, structured-demonstration, cross-embodiment, test-time-adaptation, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.11671v1
created: 2026-08-14
---

# 论文笔记：StellaVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Sydney 等 |
| 日期 | August 2026 |
| 项目主页 | [stelledge.com/blog/stellavla](https://stelledge.com/blog/stellavla) |
| 对比基线 | [[StarVLA]]、[[π0.5]]、LingBot-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2608.11671) |

---

## 一句话总结

> StellaVLA 将原始专家轨迹自动转化为"结构化演示"（任务计划 + 子目标描述 + 言语化 3D 动作），作为上下文注入 VLA 推理，使模型在分布外场景与跨体态数据中都能大幅泛化。

---

## 核心贡献

1. **结构化演示提取流水线**: 零人工标注，使用 VLM 自动将原始轨迹分解为子任务段、语义理由（子目标描述）与运动理由（3D/2D 位移）
2. **并行双专家训练范式**: 共享主干下同时训练[[Action Chunking|动作预测头]]与空间-语言专家头，推理时仅保留动作头，无自回归解码开销
3. **KV 缓存演示前缀**: 检索到的结构化演示在首帧计算一次 [[KV Cache]]，后续时间步只推理当前观测后缀，几乎零边际延迟

---

## 问题背景

### 要解决的问题

[[Vision-Language-Action Model|VLA 模型]]在训练分布以外的场景（不同视角、物体、背景）中性能急剧下降。现有[[In-Context Imitation Learning|上下文模仿学习（ICIL）]]方法以原始视频帧为演示，模型只能"模仿专家做了什么"，缺乏对"为什么这样做"的理解。

### 现有方法的局限

- 原始轨迹直接作为视觉 token，信息密度低、泛化能力弱
- 跨体态演示（人手、XR 重定向）因动作空间不一致难以直接利用
- 测试时自适应方法（如 TTT）需要更新参数，引入额外延迟

### 本文的动机

基础模型在**中间推理被显式化**时泛化更好（CoT 类工作的核心结论）。StellaVLA 将专家的决策逻辑以结构化语言形式呈现给 VLA，让模型推理"为什么"而非仅模仿"什么"。

---

## 方法详解

### 模型架构

StellaVLA 采用**检索增强 + 双专家**架构：

- **输入**: 结构化演示前缀 $\mathcal{P}_{\text{demo}}$ + 任务指令 $\mathcal{I}$ + 当前观测 $o_t$（第三视角 RGB + 腕部 RGB）
- **Backbone**: [[Qwen3-VL]]（4B 参数，完全微调）
- **核心模块**: [[In-Context Policy Adaptation|检索增强演示]] → 共享潜表示 $h_t$ → 动作专家 / 空间-语言专家
- **输出（推理）**: 连续[[Action Chunking|动作块]] $\hat{A}_t$（仿真 8 步，实机 16 步）
- **总参数**: ~4B（Qwen3-VL-4B-Instruct 主干 + 轻量 MLP 动作头）

### 核心模块

#### 模块 1：离线结构化上下文提取（Offline Structured Context Extraction）

**设计动机**: 利用 [[VLM]] 的语义理解能力在零人工标注下将原始轨迹"理性化"

**具体实现**:

1. **语义分割（Semantic Segmentation via Causal Deduction）**: 用 Qwen3-VL 将连续轨迹分解为 $K$ 个语义片段，识别每段实现的高层子目标
2. **运动言语化（Kinematic Verbalization）**: 对每个片段生成两层理由：
   - **语义理由**（Semantic Rationale / Sub-goal Description）: 高层目标描述，如 "Reach for the handle of the blue mug"
   - **运动理由**（Kinematic Rationale / Movement Description）: 由确定性 verbalizer $\Phi$ 生成，包含 3D 工作空间位移（$\Delta x, \Delta y, \Delta z$，夹爪状态）和 2D 相机平面投影轨迹

**输出**: 理由增强轨迹 $\tau_{\text{rat}} = \{(o_t, a_t, s_t, l_k)\}_{t=1}^T$ 构成演示池 $\mathcal{D}_{\text{pool}}$

#### 模块 2：并行双专家训练（Parallel Dual-Training Paradigm）

**设计动机**: 使 VLA 主干通过语言监督内化空间推理，而无需推理时运行语言生成

**具体实现**:

- **检索**: 基于任务指令语言嵌入的余弦相似度，从 $\mathcal{D}_{\text{pool}}$ 中 leave-one-out 检索 Top-1 演示；训练后部署时按任务匹配
- **共享表示**: 主干 $f_\theta$ 将 $x_t = [\mathcal{P}_{\text{demo}} \oplus \mathcal{I} \oplus o_t]$ 编码为 $h_t$
- **动作专家**: 2-block 残差 MLP 头，输入最后一层 action-placeholder token 隐状态，输出连续动作块
- **空间-语言专家（仅训练期）**: 自回归 LM 头，输出当前子任务标签 $\hat{s}_t$ + 言语化 3D/2D 运动 $\hat{m}_t$
- **关键约束**: 动作预测不依赖语言生成结果，两个专家严格并行共享 $h_t$

#### 模块 3：非对称推理 + 演示前缀缓存

**设计动机**: 消除部署时的自回归解码开销，同时保留训练期学到的空间推理能力

**具体实现**:

- 部署时完全丢弃空间-语言专家，单次前向通过主干 + MLP 动作头
- 结构化演示前缀 $\mathcal{P}_{\text{demo}}$ 在每个 episode 首帧计算一次 [[KV Cache]]，后续时间步只推理当前观测后缀

---

## 关键公式

### 公式 1：[[模仿学习|原始轨迹]]表示

$$
\tau = \{(o_t, a_t)\}_{t=1}^T
$$

**含义**: 原始演示轨迹由 $T$ 步感知-动作对组成

**符号说明**:
- $o_t$: 第 $t$ 步感知输入（多视角 RGB + 本体感知）
- $a_t \in \mathbb{R}^{d_a}$: 第 $t$ 步连续控制指令
- $T$: 轨迹总长

---

### 公式 2：[[模仿学习|理由增强轨迹]]

$$
\tau_{\text{rat}} = \{(o_t,\, a_t,\, s_t,\, l_k)\}_{t=1}^T
$$

**含义**: 为每步观测-动作对附加语义标签和片段级结构化理由

**符号说明**:
- $s_t$: 第 $t$ 步所属子任务标签
- $l_k$: 第 $k$ 片段的结构化理由（子目标描述 + 运动描述）

---

### 公式 3：检索增强输入拼接

$$
x_t = [\mathcal{P}_{\text{demo}} \oplus \mathcal{I} \oplus o_t]
$$

$$
h_t = f_\theta(x_t)
$$

**含义**: 将演示前缀、任务指令和当前观测拼接为 VLM 输入，主干编码为共享潜表示

**符号说明**:
- $\mathcal{P}_{\text{demo}}$: 包含结构化理由的检索演示前缀
- $\mathcal{I}$: 自然语言任务指令
- $o_t$: 当前帧多视角观测
- $f_\theta$: VLM 主干（Qwen3-VL-4B）
- $h_t$: 共享潜表示，同时送入两个专家头

---

### 公式 4：[[辅助监督|空间-语言监督目标]]

$$
c_t = (s_t,\ \Phi(A_t))
$$

$$
A_t = (a_t, \ldots, a_{t+H-1})
$$

**含义**: 空间-语言专家的监督信号由子任务标签和对真实动作块的言语化组成

**符号说明**:
- $c_t$: 监督目标（子任务 + 言语化运动）
- $s_t$: 第 $t$ 步子任务标签（语义监督）
- $\Phi(\cdot)$: 确定性 verbalizer，将动作块映射为 3D 位移描述 + 2D 相机投影路径
- $A_t$: 真实动作块（$H$ 步）

---

### 公式 5：[[辅助监督|双理由联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{act}}(\hat{A}_t,\, A_t) + \lambda \cdot \mathcal{L}_{\text{lang}}(\hat{c}_t,\, c_t)
$$

**含义**: 动作回归损失与语言生成损失的加权和，两个专家通过共享 $h_t$ 互相强化

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 动作块 L1 回归损失
- $\mathcal{L}_{\text{lang}}$: 理由文本自回归交叉熵损失
- $\hat{A}_t$: 动作专家预测的动作块
- $\hat{c}_t = (\hat{s}_t, \hat{m}_t)$: 空间-语言专家预测（子任务 + 言语化运动）
- $\lambda = 0.3$: 语言损失权重（消融最优值）

---

## 关键图表

### Figure 1: Teaser / 系统概览

![Figure 1: StellaVLA Teaser](https://arxiv.org/html/2608.11671v1/teaser.png)

**说明**: 系统全貌。左侧展示多源演示（机器人遥操作、人手 XR 重定向），被自动转化为包含子任务计划、子目标描述、3D 运动理由的结构化上下文；右侧展示推理时检索演示注入 VLA 的流程，实现在不同体态和分布外场景下的泛化。

---

### Figure 2: Framework / 框架详情

![Figure 2: StellaVLA Framework](https://arxiv.org/html/2608.11671v1/framework.png)

**说明**: 左上：检索到的演示以任务计划 + 子目标关键帧序列 + 机器人状态 + 2D 轨迹 + 3D 运动呈现给 VLA。右侧：离线自动标注流水线——VLM 分析轨迹视频，输出子任务分割与结构化理由，无需人工。左下：VLM 主干共同编码演示前缀与当前观测/指令，产生共享 $h_t$，驱动动作专家（训练+推理）和空间-语言专家（仅训练）。

---

### Figure 3: Simulation Benchmarks / 仿真测试台

![Figure 3: Simulation Benchmarks](https://arxiv.org/html/2608.11671v1/simulation_benchmarks.png)

**说明**: 三个仿真基准可视化。上：[[LIBERO]] 四个标准子集（Spatial / Object / Goal / Long）。中：[[LIBERO-Plus]] 七个扰动轴（相机视角、传感器噪声、机器人初始状态、语言描述、背景纹理等）。下：[[VLA-Arena]] 三难度级别（L0-L2）和 11 个任务套件（安全约束、干扰物、外推、长时程）。

---

### Figure 4: Real-Robot Evaluation / 实机评测

![Figure 4: Real-Robot Evaluation](https://arxiv.org/html/2608.11671v1/real_scene_indist_ood.png)

**说明**: 在 [[AgileX]] Piper 机械臂上的评测场景与结果。上方：4 个分布内任务（笔插杯、胡萝卜入碗、积木入抽屉并关闭、三碗叠放），StellaVLA 平均成功率 85%。下方：OOD-L1（物体属性扰动，75% vs 基线 60%）和 OOD-L2（未见抽屉，任务进度 1.9/4 vs 基线 1.1/4）。

---

### Table 1: LIBERO 分布内结果

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| StarVLA-OFT | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| **StellaVLA** | **99.6** | **99.0** | **99.6** | **96.8** | **98.8** |

**说明**: StellaVLA 全面优于匹配控制组 StarVLA-OFT，Goal（+3.4）和 Long（+3.0）提升最大——恰是仅凭当前观测难以消歧目标意图的场景，演示理由效果最显著。

---

### Table 2: VLA-Arena 分级泛化结果

| 方法 | Overall Score |
|------|---------------|
| LingBot-VLA | 0.22 |
| π₀.₅ | 0.44 |
| **StellaVLA** | **0.63** |

| 难度 | L0（分布内）| L1（中等）| L2（最难）|
|------|------------|-----------|-----------|
| StellaVLA | 0.84 | 0.62 | 0.43 |

**说明**: StellaVLA 以 0.63 总分位列 VLA-Arena 排行榜第一，大幅超越 π₀.₅（0.44）。Task Workflows 子集中 L2（0.84）反超 L0（0.64），说明任务偏离训练时，结构化演示更有价值。

---

### Table 3: LIBERO-Plus 零样本鲁棒性

| 扰动类型 | StarVLA-OFT | StellaVLA | 增益 |
|----------|-------------|-----------|------|
| Camera Viewpoint | 47.0 | 70.5 | +23.5 |
| Sensor Noise | 73.1 | 92.8 | +19.7 |
| Robot Initial State | 60.1 | 74.8 | +14.7 |
| Language | 87.0 | 95.3 | +8.3 |
| **平均** | 75.0 | **85.1** | +10.1 |

**说明**: 无需重训练，在所有扰动轴上均有显著提升。视角变化（+23.5pp）和传感器噪声（+19.7pp）改善最大，印证了结构化演示在观测分布偏移下的优势。

---

### Table 4: 演示条件消融（LIBERO Goal 套件尤为关键）

| Demo 条件 | Spatial | Object | Goal | Long | Avg |
|-----------|---------|--------|------|------|-----|
| 正确演示 | 99.6 | 99.0 | 99.6 | 96.8 | 98.8 |
| 无演示 | 72.2 | 94.4 | 24.8 | 58.2 | 62.4 |
| 错误任务演示 | 72.6 | 52.0 | 0.0 | 55.0 | 44.9 |

**说明**: Goal 套件中无演示 24.8%、错误演示 0.0%，证明模型在**主动利用**演示内容而非忽略；错误演示比无演示更差，说明模型确实在推理演示语义。

---

### Table 5: 演示模态分析

| 模态（评估时）| Avg | LP |
|---------------|-----|----|
| Image + Text | 98.8 | 85.1 |
| Text-only | 98.8 | 84.4 |
| Image-only | 92.9 | 75.7 |

**说明**: 纯文本演示在 LIBERO 分布内与完整模态持平，在 LIBERO-Plus 分布外仅差 0.7pp；纯图像演示泛化明显更差——结构化语言比原始视觉帧携带更可迁移的任务信息。

---

### Table 6: 空间-语言组件消融

| 配置 | AVG |
|------|-----|
| 完整模型（3D + 2D）| 98.8 |
| 去除 3D 运动 | 97.8 |
| 去除 2D 夹爪路径 | 97.3 |

**说明**: 2D 路径贡献（−1.5pp）略大于 3D（−1.0pp），因为 2D 轨迹直接锚定当前图像中的运动；3D 提供工作空间运动学信息。两者均有贡献，缺一不可。

---

### Table 7: λ（语言损失权重）消融

| λ | AVG | LP |
|---|-----|----|
| 0.0 | 97.6 | 86.9 |
| **0.3** | **98.8** | 85.1 |
| 0.6 | 97.5 | 84.0 |
| 1.0 | 97.2 | 81.9 |

**说明**: $\lambda = 0.3$ 为最优点；LP 随权重增大单调下降，过大的语言权重会过拟合离线标注 schema，损害分布外泛化。

---

### Table 8: 推理延迟对比

| 配置 | KV Cache | 延迟（ms）|
|------|----------|-----------|
| 无演示 | — | 64 |
| Image+Text 演示 | 无 | 183 |
| Image+Text 演示 | 有 | 91 |
| Text-only 演示 | — | 109 |
| 动作+语言头同时运行 | 有 | 3177 |
| **最终部署（仅动作头 + KV Cache）** | 有 | **88** |

**说明**: 使用演示前缀 KV Cache + 仅动作头，延迟仅为 88ms，接近无演示的 64ms，远低于实机 205ms 动作块控制周期，满足实时控制要求。

---

### Table 9: 跨体态动作一致性（实机）

| 演示来源对比 | 分布内 L1 差异 | OOD L1 差异 |
|--------------|--------------|------------|
| Robot vs. None | 0.0041 | 0.0045 |
| Human-hand vs. None | 0.0041 | 0.0044 |
| XR vs. None | 0.0041 | 0.0044 |
| Human-hand vs. Robot | 0.0015 | 0.0014 |
| XR vs. Robot | 0.0015 | 0.0016 |
| Human-hand vs. XR | 0.0014 | 0.0014 |

**说明**: 三种体态来源（机器人遥操作、人手 XR）引导下的预测动作差异极小（< 0.05mm 夹爪宽度），证明结构化语言表示成功抹平了体态间动作空间差距。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件，各 10 任务 | 标准机器人操作仿真基准 | 分布内评测 |
| [[LIBERO-Plus]] | 7 扰动轴 | 无需重训练的零样本鲁棒性测试 | OOD 评测 |
| [[VLA-Arena]] | L0/L1/L2 三难度，11 套件 | 任务级泛化，含安全/干扰/外推/长时程 | 泛化评测 |
| 实机自采集 | 125 机器人 + 26 人手 XR episodes | [[Cross-Embodiment]] 多源真实数据 | 实机训练与评测 |

### 实现细节

- **Backbone**: [[Qwen3-VL]]-4B-Instruct（完全微调）
- **动作头**: [[OpenVLA-OFT]] 风格 2-block 残差 MLP
- **动作块长度**: 8 步（仿真），16 步（实机，6 关节位置增量 + 夹爪宽度，q99 归一化）
- **优化器**: AdamW（$\beta_1=0.9, \beta_2=0.95$，weight decay $10^{-8}$）
- **学习率**: 主干 $5 \times 10^{-6}$，动作头 $10^{-4}$，余弦衰减 + 500 步热身
- **Batch Size**: 128（global），梯度裁剪 1.0
- **训练步数**: 30k 步
- **精度**: bf16，DeepSpeed ZeRO-2
- **2D 夹爪路径 dropout**: 0.5（保留 3D 监督在视觉缺失时的有效性）
- **演示 dropout**: 0.0（同任务演示可靠检索时）/ 0.5（否则）

---

## 批判性思考

### 优点

1. **零标注成本**: 离线结构化提取完全自动，不依赖人工理由标注，大幅降低使用门槛
2. **推理零延迟**: 非对称推理设计使部署时完全避免自回归开销，实机 88ms 延迟实用性强
3. **跨体态统一**: 结构化语言作为共同表示，成功弥合机器人/人手/XR 的体态差距，实验数据有力支持
4. **消融实验充分**: 演示条件、模态、组件、超参均有系统消融，结论可信度高

### 局限性

1. **检索依赖闭集任务**: 仿真中基于精确任务匹配检索，开放世界场景下纯语言相似度检索效果未充分验证
2. **语言 dropout 扩展性**: LIBERO-Plus 下较大的 $\lambda$ 会损害泛化，提示语言监督与泛化能力存在 tension，如何平衡需进一步研究
3. **实机数据规模较小**: 仅 125 个机器人 episode，任务多样性有限（4 个任务）；大规模场景扩展性未知
4. **Qwen3-VL 依赖**: 离线提取和主干均使用 Qwen3-VL，不同规模/版本的替代性未探索

### 潜在改进方向

1. 结合视觉相似度进行演示检索，而非仅依赖语言嵌入
2. 探索轻量化主干（如 2B 量级）以进一步降低实机推理延迟
3. 将结构化演示扩展到多步长时程任务中的子任务级检索

### 可复现性评估

- [ ] 代码开源（未在 arXiv 提供 GitHub 链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（附录提供完整超参数）
- [x] 数据集可获取（LIBERO 公开；实机数据自采集）

---

## 关联笔记

### 基于

- [[OpenVLA-OFT]]: 动作头设计基础（OFT 风格 MLP）
- [[Qwen3-VL]]: VLM 主干
- [[Action Chunking]]: 动作块预测范式

### 对比

- [[StarVLA]]: 匹配控制组（相同架构，无检索演示和语言监督）
- [[π0.5]]: VLA-Arena 对比基线（SOTA 之一）
- [[In-Context Policy Adaptation]]: 本文属于该范式的进一步发展

### 方法相关

- [[KV Cache]]: 演示前缀缓存加速推理的核心技术
- [[Cross-Embodiment]]: 跨体态统一的关键动机
- [[In-Context Imitation Learning]]: 本文改进的基础范式
- [[辅助监督]]: 空间-语言专家训练的监督形式

### 数据集相关

- [[LIBERO]]: 主仿真评测数据集
- [[LIBERO-Plus]]: 零样本鲁棒性评测
- [[AgileX]]: 实机平台

---

## 速查卡片

> [!summary] StellaVLA
> - **核心**: 将原始轨迹自动转化为"子目标+3D运动"结构化演示，作为上下文注入 VLA，实现分布外泛化
> - **方法**: 离线 VLM 提取理由 → 检索增强 + 双专家训练（动作头 + 语言头）→ 推理时仅用动作头 + KV Cache
> - **结果**: VLA-Arena 第一（0.63），LIBERO 98.8%，LIBERO-Plus 85.1%，实机 OOD 仅降 10pp（vs 基线 25pp）
> - **代码**: 未公开（博客: stelledge.com/blog/stellavla）

---

*笔记创建时间: 2026-08-14*

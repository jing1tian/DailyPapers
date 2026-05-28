---
title: "FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies"
method_name: "FineVLA"
authors: [Xintong Hu, Xuhong Huang, Jinyu Zhang, Yutong Yao, Yuchong Sun, Qiuyue Wang, Mingsheng Li, Sicheng Xie, Yitao Liu, Junhao Chen, Yixuan Chen, Yingming Zheng, Shuai Bai, Tao Yu]
year: 2026
venue: arXiv
tags: [vla, fine-grained-instruction, robot-manipulation, imitation-learning, data-curation, benchmark, dual-arm]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.27284
created: 2026-05-28
---

# 论文笔记：FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | XLANG Lab, University of Hong Kong; Qwen Team, Alibaba Inc. |
| 日期 | May 2026 |
| 项目主页 | https://finevla.xlang.ai/ |
| 对比基线 | [[视觉语言动作模型]] (Raw goal-level VLA baseline) |
| 链接 | [arXiv](https://arxiv.org/abs/2605.27284) / Code: TBA |

---

## 一句话总结

> FineVLA 通过构建 47,159 条细粒度轨迹数据集与混合监督策略，让 VLA 模型不仅知道"做什么"，还能遵循"如何做"的执行级指令，在仿真和真实双臂平台上大幅提升可操控性。

---

## 核心贡献

1. **FineVLA-Data 数据集**: 从 972,247 条已有轨迹中，经四阶段流水线（格式统一、规范化清洗、DTW 聚类去重、细粒度标注）蒸馏出 47,159 条涵盖 10 维执行属性的高质量轨迹，指令密度提升 10.4×。
2. **RoboFine-Bench 评测基准**: 包含 500 段视频、10,816 条原子事实、1,030 道 VQA 题的细粒度机器人视频理解基准，支持 VQA 和 Caption 双轨评测。
3. **FineVLA-Policy 可操控策略**: 通过细粒度与原始目标级指令互补混合（FG:Raw ≈ 1:1）训练，在仿真达到 86.8% 成功率，真实双臂平台比纯目标级基线提升 +12.8 分。

---

## 问题背景

### 要解决的问题

现有 [[视觉语言动作模型|VLA]] 只能接受目标级语言指令（"把苹果放进碗里"），无法根据执行级指令（"用右臂从侧面靠近，顺时针旋转放置"）精确控制执行过程。这限制了机器人在需要特定操作方式场景下的适用性。

### 现有方法的局限

- 现有机器人数据集（BridgeV2、RT-1、DROID 等）仅提供目标级任务描述，缺乏"主动手臂、接触区域、接近方向"等执行层面的标注。
- 现有 VLM 评测基准（VideoMME 等）面向通用视频理解，不覆盖机器人操作的专属细节（操作序列、失败恢复等）。
- [[行为克隆]] 方法直接在稀疏标注数据上训练，无法赋予策略执行属性层面的可操控性。

### 本文的动机

执行级属性（如用哪只手、从哪个方向接近、接触哪个部位）直接决定任务成功与否，且无法从目标级语言中推断。通过自动化标注工具生成并人工验证的细粒度指令，可以让策略在**不牺牲目标完成率的前提下**获得对执行属性的精细控制能力。

---

## 方法详解

### 整体框架

FineVLA 由四个组件构成，形成从数据到策略的完整流水线：

- **FineVLA-Tool**: 数据处理流水线工具箱
- **FineVLA-Data**: 细粒度标注数据集
- **RoboFine-VLM**: 基于 [[Qwen]] 微调的自动标注模型（可扩展到新轨迹）
- **FineVLA-Policy**: 混合监督训练的可操控 [[视觉语言动作模型|VLA]] 策略

![Figure 1: FineVLA 整体框架概览](https://arxiv.org/html/2605.27284v1/x2.png)

**说明**: FineVLA 框架连接数据构建、机器人视频理解、可扩展标注和可操控 VLA 策略学习四个模块。

---

### 核心模块

#### 模块 1: FineVLA-Tool — 四阶段数据流水线

![Figure 2: FineVLA-Tool 四阶段数据流水线](https://arxiv.org/html/2605.27284v1/x3.png)

**说明**: 展示从 10 个开源数据集聚合到 47,159 条细粒度轨迹的完整流水线。

**Stage 1 — 格式转换**

从 10 个开源数据集统一转换为 LeRobot 格式：
- BridgeData-V2, BC-Z, RT-1, Galaxea, RoboMIND-V1/V2, RoboCOIN, RH20T, RDT, DROID
- 总计输入：972,247 条轨迹

**Stage 2 — 规范化与清洗**

- 将动作/状态表示归一化为绝对坐标 + 四元数旋转
- 通过 [[DTW 距离]] 阈值剔除不一致轨迹

**Stage 3 — DTW 聚类去冗余**

使用 [[动态时间规整|DTW]] 在任务内识别冗余演示并移除，保留操作多样性。最终从 972,247 条压缩至 47,159 条代表性样本。

**Stage 4 — 细粒度标注**

沿十个维度对轨迹进行标注：

| 维度 | 内容 |
|------|------|
| 1. 动作序列 | 时序操作步骤 |
| 2. 主动执行体 | 哪个夹爪/肢体 |
| 3. 目标物体 | 操作对象 |
| 4. 初始配置 | 场景初始状态 |
| 5. 最终配置 | 结束状态 |
| 6. 接触与接近 | 抓握模式、接近角度 |
| 7. 轨迹与朝向 | 运动路径、旋转 |
| 8. 物体交互 | 操作原语 |
| 9. 失败与恢复 | 错误处理 |
| 10. 体态运动 | 手臂运动模式 |

标注后指令密度：从 9.3 词/轨迹 → 96.8 词/轨迹（**10.4× 提升**），覆盖 47 个独特动作动词。

---

#### 模块 2: RoboFine-Bench — 细粒度机器人视频理解基准

![Figure 3: RoboFine-Bench 概览](https://arxiv.org/html/2605.27284v1/x4.png)

**说明**: 基准统计、VQA 轨道和 Caption 轨道示意图。

**规模**:
- 500 段视频，32 种机器人本体，多样相机视角
- 10,816 条原子事实（跨越 10 个标注维度分解）
- 1,030 道 VQA 问题（三类推理轴）

**三类推理轴**:
1. 实体与场景基础理解（Entity and Scene Grounding）
2. 动作与运动理解（Action and Motion Understanding）
3. 交互与状态推理（Interaction and State Reasoning）

**双轨评测**:

| 轨道 | 设置 | 评估指标 |
|------|------|----------|
| VQA 轨道 | 判别式问答，确定性匹配 | 准确率 |
| Caption 轨道（Easy） | 生成式，提供任务指令 | Consistency / Coverage / Anti-Hallucination |
| Caption 轨道（Hard） | 生成式，仅视觉输入 | Consistency / Coverage / Anti-Hallucination |

**人类相关性**: Pearson 相关系数 0.98（Easy 模式）、0.97（Hard 模式），验证基准有效性。

---

#### 模块 3: RoboFine-VLM — 可扩展标注模型

- 基于 [[Qwen]] Qwen3.5-397B-A17B 在 FineVLA-Data 上微调
- 在 RoboFine-Bench 上 VQA 准确率 71.0%（vs. Gemini-3.1-Pro 62.1%）
- Caption Easy: 85.2%，Caption Hard: 83.6%
- 作为未来轨迹的自动标注器，大幅降低人工成本

---

#### 模块 4: FineVLA-Policy — 混合监督可操控策略

**基础架构（双系统）**:

在两种动作解码架构上验证：

| 架构 | System 1 | System 2 | 动作预测 |
|------|----------|----------|----------|
| **StarVLA-OFT** | — | VL Backbone | MLP head + L1 loss，预测动作块 |
| **StarVLA-GR00T** | DiT [[Flow Matching]] | VL Backbone | 双系统，System 1 生成连续动作 |

**混合监督策略**:

训练数据由细粒度（FG）指令数据和原始目标级（Raw）数据以比例混合：

$$
\mathcal{D}_{train} = r \cdot \mathcal{D}_{FG} + (1-r) \cdot \mathcal{D}_{Raw}
$$

**含义**: 按比例 $r$ 混合细粒度数据与原始目标级数据构成训练集。

**符号说明**:
- $\mathcal{D}_{FG}$: 细粒度指令数据（含执行级属性）
- $\mathcal{D}_{Raw}$: 原始目标级数据（全量源轨迹）
- $r$: FG 数据占比，实验中测试 7 个配置（0, 1:4, 1:2, 1:1, 2:1, 4:1, 1.0）

实验最优：$r \approx 0.5$（FG:Raw = 1:1）

---

## 关键公式

### 公式 1: [[行为克隆|BC 动作预测损失（OFT）]]

$$
\mathcal{L}_{OFT} = \mathbb{E}_{(o, l, a) \sim \mathcal{D}} \left[ \| \pi_\theta(o, l) - a \|_1 \right]
$$

**含义**: StarVLA-OFT 使用 L1 损失最小化预测动作块与真实动作块之间的差异。

**符号说明**:
- $o$: 观测（图像 + 状态）
- $l$: 语言指令（细粒度或原始目标级）
- $a$: 真实动作块（action chunk）
- $\pi_\theta$: 参数化策略网络

---

### 公式 2: [[Flow Matching|Flow Matching 训练目标（GR00T）]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, a_0 \sim \mathcal{D},\, \epsilon \sim \mathcal{N}(0,I)} \left[ \| v_\phi(a_\tau, \tau, c) - (a_0 - \epsilon) \|^2 \right]
$$

**含义**: StarVLA-GR00T 的 System 1（[[Diffusion Transformer (DiT)|DiT]]）使用流匹配学习从噪声到动作的速度场，条件 $c$ 来自 System 2（VL backbone）。

**符号说明**:
- $\tau$: 时间步，均匀采样自 $[0, 1]$
- $a_\tau = \tau a_0 + (1-\tau)\epsilon$: 插值噪声动作
- $v_\phi$: 速度场网络（DiT）
- $c$: 来自 VL backbone 的条件嵌入
- $a_0$: 真实动作，$\epsilon$: 标准高斯噪声

---

## 关键图表

### Figure 1: FineVLA 整体框架概览

![Figure 1](https://arxiv.org/html/2605.27284v1/x2.png)

**说明**: FineVLA 连接四个核心模块：FineVLA-Tool 数据构建、RoboFine-Bench 视频理解评测、RoboFine-VLM 可扩展标注、FineVLA-Policy 可操控策略学习。

---

### Figure 2: FineVLA-Tool 数据流水线

![Figure 2](https://arxiv.org/html/2605.27284v1/x3.png)

**说明**: 四阶段处理：格式转换 → 规范化清洗 → [[动态时间规整|DTW]] 聚类 → 细粒度标注，将 972K 条轨迹压缩为 47K 条高质量样本。

---

### Figure 3: RoboFine-Bench 概览

![Figure 3](https://arxiv.org/html/2605.27284v1/x4.png)

**说明**: 包含 VQA 和 Caption 双轨评测，三类推理轴覆盖机器人操作全场景。

---

### Figure 4a/4b: 基准评分与人类排名相关性

![Figure 4a - Easy Mode Correlation](https://arxiv.org/html/2605.27284v1/x5.png)

![Figure 4b - Hard Mode Correlation](https://arxiv.org/html/2605.27284v1/x6.png)

**说明**: Caption 轨道分数与人类排名高度相关（Pearson 0.98 / 0.97），证明 RoboFine-Bench 评估的有效性。

---

### Figure 5: 真实世界配对评测场景

![Figure 5](https://arxiv.org/html/2605.27284v1/x7.png)

**说明**: 在相同视觉场景下，通过不同语言变体（右臂 vs. 左臂 / 顺时针 vs. 逆时针 / 侧面 vs. 上方接近）测试策略的执行属性可操控性。

---

### Figure 6: RoboTwin 混合比例消融曲线

![Figure 6](https://arxiv.org/html/2605.27284v1/x8.png)

**说明**: 成功率随 FG 比例增加呈倒 U 形曲线，峰值约在 FG:Raw = 1:1 至 1:2，混合监督优于任一单独使用。

---

### Table 1: RoboFine-Bench VLM 评测结果

| 模型 | VQA (%) | Caption Easy (%) | Caption Hard (%) |
|------|---------|-----------------|-----------------|
| GPT-4o | 60.8 | 79.4 | 76.2 |
| Gemini-3.1-Pro | 62.1 | 82.1 | 80.3 |
| Qwen-VL-Max | 57.3 | 75.8 | 72.9 |
| **RoboFine-VLM (Ours)** | **71.0** | **85.2** | **83.6** |

**说明**: RoboFine-VLM 在所有三个指标上均超越商业通用 VLM，验证领域专属微调的优势。

---

### Table 2: RoboTwin 仿真混合比例实验（Table 4）

| FG:Raw 比例 | RDT-OFT Easy | RDT-OFT Hard | RDT-GR00T Easy | RDT-GR00T Hard | AlohaMix-OFT Easy | AlohaMix-OFT Hard |
|-------------|-------------|-------------|----------------|----------------|-------------------|-------------------|
| Raw-only (0:1) | 61.5 | 60.0 | 55.1 | 53.4 | 71.8 | 71.4 |
| 1:4 | 68.2 | 66.5 | 58.9 | 57.8 | 79.4 | 75.2 |
| **1:2** | **74.1** | **72.1** | 61.7 | 60.9 | 82.8 | 78.6 |
| **1:1** | 73.9 | 72.4 | **69.4** | **68.2** | **86.8** | **82.5** |
| 2:1 | 70.2 | 68.9 | 65.8 | 64.1 | 84.1 | 80.3 |
| 4:1 | 66.7 | 65.4 | 64.0 | 63.2 | 81.2 | 77.8 |
| FG-only (1:0) | 62.9 | 62.0 | 62.1 | 61.5 | 78.3 | 76.1 |

**关键发现**:
- FG-only 相比 Raw-only 提升 +1.4 至 +8.1 个点
- 混合监督（1:1 至 1:2）优于任一纯方式
- 最优配置（AlohaMix-OFT, 1:1）达到 86.8%/82.5%，比基线提升 **+15.0/+11.1**

---

### Table 3: 真实世界可操控性评测（Table 5）

| 监督方式 | 颜色 | 姿态 | 接近方向 | 旋转方向 | 主动手臂 | **平均 ID** |
|----------|------|------|----------|----------|----------|------------|
| Raw-only | 22 | 24 | 60 | 76 | 60 | 49.9 |
| **FG:Raw = 1:1** | **40** | **47** | **78** | **86** | **64** | **62.7** |
| 提升量 | +18 | +23 | +18 | +10 | +4 | **+12.8** |

**关键发现**: 对原始指令"不可见"的执行属性（姿态、颜色、接近方向）获益最大；组合泛化（OOD）仍困难（仅 10/100）。

---

## 实验

### 数据集与环境

| 数据集/环境 | 规模 | 特点 | 用途 |
|-------------|------|------|------|
| BridgeData-V2 | ~60K | 多任务桌面操作 | 训练数据来源 |
| RT-1 | ~130K | Google 机器人数据 | 训练数据来源 |
| DROID | ~75K | 多样化真实数据 | 训练数据来源 |
| RDT | ~15K | 双臂任务 | FG 微调子集 |
| AlohaMix | ~200K | 双臂混合 | FG 微调子集（更大规模） |
| RoboTwin | — | 仿真评测平台 | 主要仿真实验 |
| Cobot Magic | 600 demos | 真实双臂平台 | 真实世界实验 |

### 实现细节

- **Backbone**: StarVLA（VL backbone）
- **预训练**: 100K steps，64× A100 GPUs，batch size 512
- **微调（RoboTwin）**: 100K steps，8× A100 GPUs，batch size 128
- **真实世界适应**: 600 条演示，12 个桌面任务
- **混合比例搜索**: 7 个 FG:Raw 配置（0:1, 1:4, 1:2, 1:1, 2:1, 4:1, 1:0）

### 架构效应分析

FG 监督缩小了 OFT 与 GR00T 解码器之间的差距：
- Raw-only 差距：6.4/6.6 分（Easy/Hard）
- FG-only 差距：0.8/0.5 分

数据规模效应：AlohaMix（13× 更大）上 FG 收益从 +1.4/+2.0（RDT）增长至 +6.5/+4.7。

---

## 批判性思考

### 优点

1. **完整的系统级工作**: 从数据工具链、基准评测到策略训练形成闭环，可复现性强。
2. **令人信服的互补性发现**: 细粒度与原始指令互补的倒 U 形规律在多个 setting 下稳定复现，科学可信。
3. **实用的自动化标注器**: RoboFine-VLM 超越商业 VLM，为未来数据扩展提供可行路径。

### 局限性

1. **标注验证成本**: RoboFine-VLM 减少但未消除人工验证需求。
2. **真实验证范围窄**: 仅在桌面双臂平台测试，移动操作、灵巧手等场景未覆盖。
3. **组合泛化能力弱**: OOD 演员-目标绑定测试仅 10/100，说明策略尚不能真正理解属性组合。
4. **执行指令的安全性**: 精细执行指令在部署中可能引入不安全行为，需可行性检查机制。
5. **跨具身泛化未验证**: 是否可迁移到未见过的机器人形态不明确。

### 潜在改进方向

1. 引入在线 RL 信号（如接触力传感器反馈）强化执行属性学习。
2. 利用 [[World Model]] 在仿真中自动生成细粒度标注，减少人工成本。
3. 结合 [[可供性]] 预测预先过滤不可行的执行指令，提升安全性。

### 可复现性评估

- [x] 代码开源（FineVLA-Tool + 训练代码）
- [x] 预训练模型（FineVLA-Policy checkpoints）
- [x] 训练细节完整（GPU 数、batch size、训练步数均报告）
- [x] 数据集可获取（FineVLA-Data 公开，CC BY-SA 4.0）

---

## 关联笔记

### 基于

- [[视觉语言动作模型]]: FineVLA 在 VLA 框架基础上引入执行级指令对齐
- [[行为克隆]]: StarVLA-OFT 使用 BC + L1 损失
- [[扩散策略]]: StarVLA-GR00T 中 System 1 基于 Flow Matching / DiT

### 对比

- [[TraceVLA]]: 同属操作轨迹增强 VLA，但 TraceVLA 关注视觉轨迹线索而非语言执行属性
- [[动作分块]]: FineVLA-Policy 输出连续动作块，是主流 VLA 动作表示

### 方法相关

- [[动态时间规整]]: Stage 3 DTW 聚类用于去除冗余演示
- [[Diffusion Transformer (DiT)]]: StarVLA-GR00T System 1 动作生成模块
- [[Flow Matching]]: GR00T 架构的动作生成方法
- [[模仿学习]]: FineVLA-Policy 整体训练范式

### 硬件/数据相关

- [[Cobot Magic]]: 真实世界双臂评测平台

---

## 速查卡片

> [!summary] FineVLA
> - **核心**: 细粒度执行级指令对齐让 VLA 从"做什么"升级到"怎么做"
> - **方法**: 四阶段数据流水线（47K 轨迹）+ 混合监督（FG:Raw = 1:1）+ RoboFine-Bench 评测
> - **结果**: 仿真 86.8% 成功率，真实双臂 +12.8 分（vs. 纯目标级基线 49.9 → 62.7）
> - **代码**: https://finevla.xlang.ai/

---

*笔记创建时间: 2026-05-28*

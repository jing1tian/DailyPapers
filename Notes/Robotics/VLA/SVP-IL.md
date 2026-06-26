---
title: "Decoupling Semantics and Geometric Grounding: Spatial Visual Prompts for Language-Conditioned Imitation Learning"
method_name: "SVP-IL"
authors: [Yanzhe Tang, Xinyu Shao, Yuxuan Hu, Siyu Chen, Bowen Yang, Yajun Gao, Tongtong Cao, Xiu Li, Long Zeng]
year: 2026
venue: arXiv
tags: [vla, imitation-learning, spatial-grounding, diffusion-policy, segmentation, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.25360
created: 2026-06-26
---

# 论文笔记：Decoupling Semantics and Geometric Grounding: Spatial Visual Prompts for Language-Conditioned Imitation Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University（清华大学）、Huawei Technologies、National University of Singapore |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[Diffusion Policy]]、[[ACT]]、[[Pi0|π₀]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25360) / Code: 未公开 |

---

## 一句话总结

> SVP-IL 把"语言→空间定位"这件事从端到端 VLA 的隐式对齐中剥离出来，用冻结的 LLM+SAM 3 生成显式几何掩码，再以特征级融合注入扩散策略，在数据稀缺、多物体歧义场景下大幅提升成功率。

---

## 核心贡献

1. **解耦架构 (Decoupled Perception-Control)**: 将语义推理（目标消歧）从动作生成闭环中显式剥离，交给冻结的基础模型完成，训练策略只需专注几何控制
2. **空间视觉提示 (Spatial Visual Prompt, SVP)**: 用两阶段基础模型流水线（LLM 解析指令类别 → [[SAM3|SAM 3]] 生成二值掩码）把自然语言指令转化为精确的几何提示
3. **轻量特征级融合**: 用侧支路 CNN 处理掩码特征，与主干 RGB 特征在多个层级做逐元素相加融合，避免像素级叠加或早期拼接带来的信息损失
4. **数据高效与强鲁棒性**: 仅用 50~100 条示范即可在歧义任务上把成功率从 24.0%（π₀ 基线）提升到 39.5%，且在视觉分布偏移下退化幅度远小于基线

---

## 问题背景

### 要解决的问题

端到端 [[Vision-Language-Action Model|VLA]] 模型把语义推理、空间定位、动作生成耦合进同一个网络，依赖隐式的跨模态对齐（latent vector）把语言指令转化为精确空间控制信号。在模仿学习典型的小数据场景下，这种隐式对齐效率低下，导致面对"拿起特定那个杯子"这类需要**目标消歧**的多物体场景时，模型容易出现空间定位不准、抓取失败、近距离误抓等问题。

### 现有方法的局限

- **端到端 VLA**（[[RT-2]]、[[OpenVLA]]、[[Pi0|π₀]] 等）：将语言 token 转换为精确空间控制本身就"效率低下"（notoriously inefficient），在数据受限的模仿学习设置下尤为明显；端到端特性天然稀释了模型对细粒度语义定位的建模能力
- **早期空间定位方法**：依赖稠密点云、深度图等表征，计算和标注成本高
- **近期空间提示方法**（[[VIMA]] 的多模态提示、MOKA 的标记式提示、ReKep 的关键点、RoboMAP 的可操作性热力图）：多数仍需要针对任务定制提示形式，或未充分利用现代分割基础模型的零样本能力

### 本文的动机

如果用现代视觉语言基础模型（LLM + [[SAM3|SAM 3]]）把"语言指令 → 目标物体的几何掩码"这一步完全解耦出来，并通过轻量的特征级融合注入到现有视觉运动策略（如 [[Diffusion Policy]]）中，就能让策略网络只需学习"看到掩码后如何精确控制"，而不必同时学习语义消歧，从而在小数据场景下显著提升精度和泛化能力。

---

## 方法详解

### 模型架构

SVP-IL 采用**解耦感知-控制 (Decoupled Perception-Control)** 架构，整体是对标准 [[Diffusion Policy]] 的扩展：

- **输入**: 语言指令 $l$ + 历史观测 $o_{t-k:t}$（含 RGB 图像 $I_t$）+ 机器人本体状态
- **语义解析**: 冻结 LLM 将指令 $l$ 解析为目标物体类别 $c$
- **空间定位**: 冻结的 [[SAM3|SAM 3]] 接收图像 $I_t$ 和类别 $c$，输出二值几何掩码 $M_t \in \{0,1\}^{H\times W}$（即 **空间视觉提示**）
- **主干编码**: $\Phi_{rgb}$ —— 预训练 ImageNet 特征骨干，处理原始 RGB 图像，权重保持不变
- **提示编码**: $\Phi_{prompt}$ —— 轻量侧支路 CNN，专门处理掩码 $M_t$
- **融合方式**: 在多个网络层级上做 [[特征级融合|逐元素相加]]，而非图像级拼接或像素叠加
- **动作生成**: 融合后的视觉特征 + 本体状态 + 文本嵌入 一起送入 1D U-Net 噪声预测器（沿用 [[Diffusion Policy]] 的扩散去噪机制）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$
- **系统频率**: 策略以 10 Hz 同步执行，底层控制 300 Hz；LLM 每个任务只调用一次（结果缓存），SAM 3 逐步执行（降采样后）

### 核心模块

#### 模块 1：空间视觉提示提取 (Spatial Visual Prompt Extraction)

**设计动机**: 把"指令理解"和"空间定位"这两件强语义性的工作，完全交给现成的、冻结的基础模型完成，训练阶段策略网络无需再学习这部分能力。

**具体实现**:
- 用 [[LLM]] 把自然语言指令 $l$ 解析为目标物体的类别描述 $c$（一次解析、全程缓存，不随每步重新调用）
- 用 [[SAM3|SAM 3]] 接收当前帧图像 $I_t$ 和类别 $c$，做开放词汇分割，输出二值掩码 $M_t$
- 整个流程不需要任何任务特定的训练或微调，完全是零样本调用

#### 模块 2：特征级融合 (Feature-Level Fusion)

**设计动机**: 直接把掩码图像和 RGB 图像在像素级拼接（pixel-overlay）或者在网络浅层做拼接（early-concat）都会破坏预训练 RGB 骨干已经学到的特征分布；用独立的侧支路处理掩码、再在深层特征空间相加，可以同时保留预训练知识和引入空间先验。

**具体实现**:
- 主干 $\Phi_{rgb}$ 保持 ImageNet 预训练权重不变，处理原始图像 $I_t$
- 侧支路 $\Phi_{prompt}$ 是一个轻量 CNN，参数 $\theta_{prompt}$ 单独训练，处理掩码 $M_t$
- 在网络的第 $i$ 层，将两路特征逐元素相加：$F^{(i)} = \Phi_{rgb}^{(i)}(I_t) + \Phi_{prompt}^{(i)}(M_t;\theta_{prompt})$

#### 模块 3：动作生成 (Action Generation)

**设计动机**: 复用成熟的 [[Diffusion Policy]] 范式来建模多模态动作分布，融合后的特征只是替换了原本单一的 RGB 特征输入。

**具体实现**:
- 融合特征 $F^{(i)}$、机器人本体状态、语言文本嵌入一并输入 1D U-Net
- U-Net 按标准扩散去噪流程预测动作块 $a_{t:t+k}$ 上的噪声，逐步反推出干净动作序列

---

## 关键公式

### 公式 1：[[SAM3|空间提示生成]]（总览形式）

$$
c = \Gamma(l), \qquad M_t = \Psi(I_t, c)
$$

**含义**: 用语言解析函数 $\Gamma$ 把指令转为目标类别，再用视觉定位函数 $\Psi$ 生成对应的几何掩码，这是整套空间视觉提示提取流程的抽象表达。

**符号说明**:
- $l$: 自然语言指令
- $\Gamma$: 语义解析函数（由 LLM 实现）
- $c$: 解析出的目标物体类别
- $\Psi$: 视觉定位函数（由 SAM 3 实现）
- $I_t$: 时刻 $t$ 的图像观测
- $M_t$: 输出的二值几何掩码

### 公式 2：重定义的控制策略

$$
a_t \sim \pi_\phi(\,\cdot \mid o_{t-k:t},\, M_{t-k:t},\, l\,)
$$

**含义**: 策略网络的条件输入从"纯观测 + 语言"扩展为"观测 + 几何掩码序列 + 语言"，显式地把空间提示纳入动作生成的条件分布。

**符号说明**:
- $\pi_\phi$: 参数为 $\phi$ 的策略网络（基于扩散模型）
- $o_{t-k:t}$: 从 $t-k$ 到 $t$ 的历史观测序列
- $M_{t-k:t}$: 对应时间窗口的几何掩码序列
- $a_t$: 时刻 $t$ 采样得到的动作

### 公式 3：LLM 语义解析

$$
c = \mathrm{LLM}(l)
$$

**含义**: 用大语言模型把自然语言指令直接映射为目标物体的类别标签，是空间提示提取流程的第一阶段。

**符号说明**:
- $\mathrm{LLM}(\cdot)$: 冻结的大语言模型解析函数
- $l$: 输入语言指令
- $c$: 输出目标类别

### 公式 4：[[SAM3|SAM 3]] 几何定位

$$
M_t = \mathrm{SAM3}(I_t, c)
$$

**含义**: 用 SAM 3 以类别 $c$ 作为开放词汇提示，在图像 $I_t$ 上做零样本分割，输出对应目标的二值掩码，是空间提示提取流程的第二阶段。

**符号说明**:
- $\mathrm{SAM3}(\cdot,\cdot)$: 冻结的 SAM 3 分割模型
- $I_t$: 时刻 $t$ 的 RGB 图像
- $c$: 目标类别提示
- $M_t \in \{0,1\}^{H\times W}$: 输出二值掩码

### 公式 5：提示特征提取

$$
F_{prompt}^{(i)} = \Phi_{prompt}^{(i)}(M_t;\, \theta_{prompt})
$$

**含义**: 侧支路 CNN 在第 $i$ 层对掩码进行特征编码，生成可与主干特征相加融合的提示特征。

**符号说明**:
- $\Phi_{prompt}^{(i)}$: 侧支路 CNN 第 $i$ 层的特征映射函数
- $\theta_{prompt}$: 侧支路 CNN 的可训练参数
- $M_t$: 输入的几何掩码
- $F_{prompt}^{(i)}$: 第 $i$ 层输出的提示特征

### 公式 6：特征级融合

$$
F^{(i)} = \Phi_{rgb}^{(i)}(I_t) + F_{prompt}^{(i)}
$$

**含义**: 在网络第 $i$ 层，把冻结的 RGB 主干特征与可训练的提示特征逐元素相加，得到融合后的视觉特征，作为后续动作生成的输入。

**符号说明**:
- $\Phi_{rgb}^{(i)}$: 预训练 RGB 主干第 $i$ 层的特征映射函数（权重不更新或与策略一起微调）
- $I_t$: 原始 RGB 图像
- $F_{prompt}^{(i)}$: 公式 5 中得到的提示特征
- $F^{(i)}$: 第 $i$ 层融合后的特征，逐元素加和（element-wise addition）

---

## 关键图表

### Figure 1: 端到端 VLA 与 SVP-IL 的架构对比

![Figure 1](https://arxiv.org/html/2606.25360v1/x1.png)

**说明**: (a) 标准端到端 VLA 模型联合训练视觉语言骨干和动作专家，二者之间只通过隐式的 latent vector 传递信息，语义理解和空间控制被耦合在一起；(b) SVP-IL 用冻结的视觉语言模型从指令中显式抽取几何掩码，把语义推理和空间控制彻底解耦。

### Figure 2: SVP-IL 整体框架总览

![Figure 2](https://arxiv.org/html/2606.25360v1/x2.png)

**说明**: 语义推理被转移到两阶段基础模型流水线（LLM 解析器 → [[SAM3|SAM 3]]），从原始语言生成精确的空间视觉提示；注入模块以轻量侧支路 CNN 的形式运作，通过逐元素相加把几何特征直接融合进深层视觉编码器。

### Figure 3: 仿真任务与环境分布可视化

![Figure 3](https://arxiv.org/html/2606.25360v1/x3.png)

**说明**: 左列展示 Clean 设置（简洁背景），右侧多列展示 Randomized 设置（包含复杂纹理、动态光照、密集干扰物等严重分布偏移），用于检验策略在视觉分布外场景下的鲁棒性。

### Figure 4: 真实世界双臂评测场景

![Figure 4](https://arxiv.org/html/2606.25360v1/x4.png)

**说明**: 在 [[ALOHA|Aloha-AgileX]] 双臂平台上的真实世界评测设置，头部 RGB 摄像头提供 2 帧历史观测；展示了摆桌（Table Setting）、桌面清理（Desktop Cleaning）、双臂分拣（Bimanual Sorting）三类任务，工作空间中含有密集的未见干扰物，插图展示了显式几何掩码的效果。

### Table I: RoboTwin 2.0 与自定义任务成功率（%）

| Task | DP | ACT | π₀ | SVP-IL |
|------|----|----|-----|--------|
| Adjust Bottle | 97.0 | 97.0 | 97.0 | 96.0 |
| Click Bell | 54.0 | 58.0 | 54.0 | 83.0 |
| Hanging Mug | 8.0 | 7.0 | 11.0 | 24.0 |
| Lift Pot | 39.0 | 88.0 | 84.0 | 90.0 |
| Place Can Basket | 18.0 | 1.0 | 41.0 | 42.0 |
| Place Cans Plasticbox | 40.0 | 16.0 | 34.0 | 88.0 |
| Place Empty Cup | 37.0 | 61.0 | 37.0 | 49.0 |
| Place Shoe | 23.0 | 5.0 | 28.0 | 45.0 |
| Shake Bottle | 65.0 | 74.0 | 97.0 | 93.0 |
| **RoboTwin Avg** | 42.3 | 45.2 | 53.7 | **67.8** |
| Move Single Object to Basket | 27.0 | 4.0 | 42.0 | 67.0 |
| Move Single Object to Stand | 32.0 | 0.0 | 23.0 | 54.0 |
| Move Single Object to Scale | 19.0 | 0.0 | 27.0 | 38.0 |
| Pick Specific Object to Basket | 9.0 | 5.0 | 7.0 | 25.0 |
| Pick Specific Object to Stand | 17.0 | 1.0 | 19.0 | 34.0 |
| Pick Specific Object to Scale | 12.0 | 7.0 | 26.0 | 19.0 |
| **Custom Avg** | 19.3 | 2.8 | 24.0 | **39.5** |

**说明**: 在 [[RoboTwin 2.0]] 标准 benchmark 上，SVP-IL 平均成功率 67.8%，超过 π₀（53.7%）、ACT（45.2%）、DP（42.3%）；在需要目标消歧的自定义多物体任务上优势更明显（39.5% vs π₀ 24.0%），ACT 和 DP 在歧义场景下大幅崩溃（分别只有 2.8% 和 19.3%）。

### Table II: 视觉分布偏移影响（%）

| Task | DP Clean | DP Random | SVP-IL Clean | SVP-IL Random |
|------|----------|-----------|--------------|---------------|
| Adjust Bottle | 97.0 | 66.0 | 96.0 | 76.0 |
| Click Bell | 54.0 | 4.0 | 83.0 | 34.0 |
| Hanging Mug | 8.0 | 6.0 | 24.0 | 9.0 |
| Lift Pot | 39.0 | 9.0 | 90.0 | 13.0 |
| Place Can Basket | 18.0 | 8.0 | 42.0 | 15.0 |
| Place Cans Plasticbox | 40.0 | 4.0 | 88.0 | 4.0 |
| Place Empty Cup | 37.0 | 2.0 | 49.0 | 4.0 |
| Place Shoe | 23.0 | 7.0 | 45.0 | 22.0 |
| Shake Bottle | 65.0 | 28.0 | 93.0 | 39.0 |
| **RoboTwin Avg** | 42.3 | 14.9 | 67.8 | 24.0 |
| Move to Basket | 27.0 | 10.0 | 67.0 | 10.0 |
| Move to Stand | 32.0 | 8.0 | 54.0 | 14.0 |
| Move to Scale | 19.0 | 5.0 | 38.0 | 13.0 |
| Pick to Basket | 9.0 | 7.0 | 25.0 | 10.0 |
| Pick to Stand | 17.0 | 9.0 | 34.0 | 17.0 |
| Pick to Scale | 12.0 | 2.0 | 19.0 | 9.0 |
| **Custom Avg** | 19.3 | 6.8 | 39.5 | 12.2 |

**说明**: 在 Randomized 环境下训练时，DP 的性能严重退化（RoboTwin 均值 42.3%→14.9%，跌幅 65%），而 SVP-IL 相对更稳定（67.8%→24.0%，跌幅约 65% 但绝对值仍更高），表明显式空间先验比单纯的域随机化更能抵御视觉分布偏移。

### Table III: 消融实验 — 融合策略对比（%）

| Task | DP | Pixel-Overlay | Early-Concat | SVP-IL |
|------|----|----|-------|--------|
| Adjust Bottle | 97.0 | 90.0 | 76.0 | 96.0 |
| Click Bell | 54.0 | 90.0 | 71.0 | 83.0 |
| Hanging Mug | 8.0 | 6.0 | 6.0 | 24.0 |
| Lift Pot | 39.0 | 59.0 | 42.0 | 90.0 |
| Place Can Basket | 18.0 | 26.0 | 29.0 | 42.0 |
| Place Cans Plasticbox | 40.0 | 16.0 | 42.0 | 88.0 |
| Place Empty Cup | 37.0 | 33.0 | 28.0 | 49.0 |
| Place Shoe | 23.0 | 38.0 | 24.0 | 45.0 |
| Shake Bottle | 65.0 | 93.0 | 73.0 | 93.0 |
| **RoboTwin Avg** | 42.3 | 50.1 | 43.4 | 67.8 |
| Move to Basket | 27.0 | 29.0 | 28.0 | 67.0 |
| Move to Stand | 32.0 | 31.0 | 36.0 | 54.0 |
| Move to Scale | 19.0 | 26.0 | 24.0 | 38.0 |
| Pick to Basket | 9.0 | 7.0 | 6.0 | 25.0 |
| Pick to Stand | 17.0 | 15.0 | 21.0 | 34.0 |
| Pick to Scale | 12.0 | 12.0 | 18.0 | 19.0 |
| **Custom Avg** | 19.3 | 20.0 | 22.2 | 39.5 |

**关键发现**: 像素级叠加（Pixel-Overlay）和网络浅层拼接（Early-Concat）都明显弱于本文提出的深层特征级相加融合；尤其在需要消歧的 Custom 任务上，Pixel-Overlay（20.0%）和 Early-Concat（22.2%）远低于 SVP-IL 的特征级融合方案（39.5%），说明融合方式本身、而不仅仅是"有没有掩码"，对最终性能至关重要。

### Table IV: 真实世界物理评测（%）

| Task | DP | π₀ | SVP-IL |
|------|----|----|--------|
| Table Setting | 20 | 30 | 75 |
| Desktop Cleaning | 35 | 20 | 50 |
| Bimanual Sorting | 30 | 45 | 55 |
| **Average** | 28.3 | 31.7 | **60.0** |

**说明**: 在 [[ALOHA|Aloha-AgileX]] 真实双臂平台上，SVP-IL 在三类含密集干扰物的真实任务上平均成功率达到 60.0%，远超 π₀（31.7%）和 DP（28.3%），验证了仿真中观察到的优势可以迁移到真实机器人。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 9 个代表性任务，每任务 50 条示范 | 标准操作 benchmark，提供 Clean / Randomized 两种环境协议 | 训练与测试 |
| 自定义多物体任务 | 6 个任务，每任务 100 条示范 | 语言条件下的多物体目标消歧场景（如"把指定物体放入篮子"） | 训练与测试 |
| 真实世界任务 | 3 类任务（摆桌、桌面清理、双臂分拣） | [[ALOHA|Aloha-AgileX]] 双臂平台，密集未见干扰物 | 真机评测 |

### 实现细节

- **Backbone**: 预训练 ImageNet 视觉骨干（处理 RGB），轻量侧支路 CNN（处理掩码）
- **空间定位模型**: 冻结 LLM（指令解析）+ 冻结 [[SAM3|SAM 3]]（掩码生成）
- **基线**: [[ACT]]、[[Diffusion Policy]]（DP）、[[Pi0|π₀]]，训练配置与 SVP-IL 保持一致；π₀ 按官方配置做全参数微调
- **执行频率**: 策略 10 Hz，底层控制 300 Hz
- **硬件**: [[ALOHA|Aloha-AgileX]] 双臂机器人（真实世界评测）

### 可视化结果

论文在 Figure 3、Figure 4 中给出了仿真 Clean/Randomized 环境对比和真实世界任务场景的可视化，并在真实任务插图中展示了 SAM 3 生成的几何掩码效果，直观体现了空间视觉提示在密集干扰物场景下对目标的精确锁定能力。

---

## 批判性思考

### 优点

1. **解耦思路简单有效**: 不需要重新设计端到端 VLA 的对齐机制，只需把现成基础模型的零样本能力接到现有视觉运动策略上，工程实现成本低
2. **数据效率显著**: 仅用 50~100 条示范就能在多物体歧义任务上大幅超越同等数据量下训练的 π₀、ACT、DP，对真实机器人数据采集成本敏感的场景很有价值
3. **真实世界验证完整**: 不仅有仿真 benchmark，还在真实双臂平台上完成了三类含干扰物任务的物理评测，证明了 sim-to-real 的可迁移性
4. **消融充分**: 通过 Pixel-Overlay 和 Early-Concat 两种替代融合方式的对比，证明了"深层特征级相加"这一设计选择本身的必要性，而非只是"加了掩码就变好"

### 局限性

1. **性能受限于上游视觉语言模型**: 论文自述在透明、高反光表面等场景下 SAM 3 容易生成噪声掩码，整体性能存在天花板
2. **依赖显式、以物体为中心的指令**: 无法处理"把桌子收拾干净"这类隐含意图的指令，语义解析阶段仍是 bottleneck
3. **SAM 3 计算开销大**: 论文承认 SAM 3 带来了显著的额外计算成本，限制了实时性，作者将"轻量分割替代方案"列为未来工作
4. **代码未公开**: 截至本文撰写时未提供 GitHub 仓库链接，复现难度较高
5. **对比基线相对有限**: 主要与 DP、ACT、π₀ 比较，未与 VIMA、MOKA、ReKep 等同样采用空间提示思路的方法做直接的数值对比

### 潜在改进方向

1. 用更轻量的开放词汇分割模型（蒸馏版 SAM 或专用小模型）替代 SAM 3，降低推理开销
2. 探索把掩码序列的时序一致性（如对相邻帧掩码做平滑约束）纳入训练目标，提升真实环境下的鲁棒性
3. 扩展语义解析模块以支持隐式指令（需要常识推理的任务描述）

### 可复现性评估

- [ ] 代码开源（论文未提供仓库链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（示范数量、基线配置、硬件平台描述清晰）
- [x] 数据集可获取（RoboTwin 2.0 公开；自定义任务和真实数据未明确说明是否会发布）

---

## 关联笔记

### 基于

- [[Diffusion Policy]]: SVP-IL 的动作生成模块直接沿用 DP 的扩散去噪机制，仅替换视觉特征输入
- [[SAM3|SAM 3]]: 空间视觉提示生成的核心视觉定位模型

### 对比

- [[ACT]]: 主要基线之一，在歧义任务上性能崩溃明显（2.8%），凸显端到端方法在目标消歧上的弱点
- [[Pi0|π₀]]: 端到端 VLA 代表性基线，对比体现"隐式语义-空间耦合"与"显式解耦"的差异
- [[VIMA]]: 同样采用多模态/视觉提示思路的相关工作，但提示形式和注入方式不同

### 方法相关

- [[SAM3|SAM 3]]: 核心空间定位组件
- [[Action Chunking]]: 动作输出形式，沿用 DP/ACT 的动作块范式

### 硬件/数据相关

- [[ALOHA|Aloha-AgileX]]: 真实世界评测的双臂机器人平台
- [[RoboTwin 2.0]]: 仿真评测的标准 benchmark

---

## 速查卡片

> [!summary] Decoupling Semantics and Geometric Grounding: Spatial Visual Prompts for Language-Conditioned Imitation Learning
> - **核心**: 把语言指令的语义消歧从端到端 VLA 中剥离，用冻结 LLM+SAM 3 生成几何掩码，再特征级融合进扩散策略
> - **方法**: 两阶段空间视觉提示提取（LLM 解析类别 → SAM 3 分割）+ 侧支路 CNN 深层特征相加融合 + Diffusion Policy 动作生成
> - **结果**: RoboTwin 2.0 平均 67.8%（vs π₀ 53.7%），自定义歧义任务 39.5%（vs π₀ 24.0%），真实世界平均 60.0%（vs π₀ 31.7%）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-26*

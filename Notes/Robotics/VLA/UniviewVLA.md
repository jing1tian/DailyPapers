---
title: "UniviewVLA: A Unified Multiview Vision-Language-Action Model with World Modeling"
method_name: "UniviewVLA"
authors: [Tao Xu, Runhao Zhang, Zhijian Huang, Jiayi Guan, Jiaxin Wang, Yifan Ding, Yong-Lu Li, Long Chen, Guang Chen, Jinghui Lu]
year: 2026
venue: arXiv
tags: [vla, world-model, multiview, occlusion, action-entropy, token-compression, manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.21501v1
created: 2026-06-24
---

# 论文笔记：UniviewVLA: A Unified Multiview Vision-Language-Action Model with World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tongji University, Shanghai Innovation Institute, Xiaomi EV, Shanghai Jiao Tong University |
| 日期 | June 2026 |
| 项目主页 | [sii-quantum.github.io/MultiviewVLA.github.io](https://sii-quantum.github.io/MultiviewVLA.github.io/) |
| 对比基线 | [[UniVLA]], [[OpenVLA]], [[OpenVLA-OFT]], [[π₀.₅\|π₀-FAST]], [[GR00T-N1.7\|GR00T N1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.21501) / Code: 未提供（项目主页有演示视频） |

---

## 一句话总结

> UniviewVLA 用世界模型从两路标准相机观测生成"看不见的"未来多视角，配合 token 压缩和熵驱动视角选择，零额外硬件解决遮挡操作问题。

---

## 核心贡献

1. **生成式多视角世界模型**: 无需额外物理相机或显式三维重建，仅凭标准的 agent-view + wrist-view 两路观测，post-train 一个世界模型来预测未来辅助视角，捕捉被遮挡的场景演化信息。
2. **Motion-Informative Token Compression（运动信息 Token 压缩）**: 把每个生成视角从 625 个 token 压缩到 16 个，单视角推理延迟从 6–7 秒降到 0.2–0.3 秒，使生成视角真正可用于实时动作预测。
3. **Action-Entropy View Selection（动作熵视角选择）**: 训练免费（training-free）的动态视角选择机制，在推理时用动作分布的 Shannon 熵衡量每个候选视角的"动作信息量"，自动挑选遮挡最少、最利于决策的视角。

---

## 问题背景

### 要解决的问题

机器人操作中，动作关键线索（如被柜门、容器边缘遮挡的物体位置）高度依赖观察视角，但实际部署中机器人通常只有 1–2 路标准相机（agent-view + wrist-view），这些线索往往在标准视角下被遮挡，导致 [[VLA|视觉-语言-动作模型]] 难以做出正确决策。

### 现有方法的局限

- **增加物理相机**：需要在训练和推理阶段严格保持相机数量与视角一致（camera parity），灵活性差，且增加硬件与标定成本。
- **显式三维重建**：从两路观测重建三维场景以推断被遮挡区域，计算成本高，难以满足实时控制的需求。
- **直接生成完整视角**：若不压缩，每个生成视角需要 625 个 token，6–7 秒的生成延迟远超机器人闭环控制对实时性的要求。

### 本文的动机

作者认为，[[World Model|世界模型]] 已被证明能从历史观测中学习场景动力学；如果让世界模型直接"想象"出标准相机看不到的辅助视角（而不是重建完整三维场景），就能以更低成本获得遮挡区域的空间证据。同时，真正用于动作预测的只是视角中"运动相关"的少量信息，因此可以大幅压缩生成视角的 token 数，并用动作分布的不确定性（熵）自动判断哪个虚拟视角更有用，从而省去人工指定视角或额外训练视角选择器的成本。

---

## 方法详解

### 模型架构

UniviewVLA 沿用 [[UniVLA]] 的离散 token [[Autoregressive Policy|自回归]]建模范式（基于 [[Transformer]] 架构，初始化自 Emu3 类视觉-语言自回归骨干），将语言、图像、动作统一编码为离散 token 序列：

- **输入**: 语言指令 $L$ + 标准 agent-view $\mathbf{x}_a$ + wrist-view $\mathbf{x}_w$ + 观测历史 $o_{t-1:t} = \{\mathbf{x}_{a,t-1:t}, \mathbf{x}_{w,t-1:t}\}$
- **Backbone**: 基于 Emu3 的纯自回归 [[Transformer]]，图像用 [[VQ-VAE|VQ tokenizer]] 离散化
- **核心模块**: [[Motion-Informative Token Compression]] 压缩生成视角，[[Action-Entropy View Selection]] 动态挑选最优虚拟视角
- **输出**: 压缩后的辅助视角 token + [[FAST]] 风格动作 token $a_{t,1:N_a}$
- **训练**: 两阶段——Stage 1 世界模型 post-training（30K steps），Stage 2 动作微调（LIBERO/CALVIN 24K steps，真机 30K steps）
- **硬件**: 8× NVIDIA H 系列 GPU，bf16 + DeepSpeed ZeRO-3

### 核心模块

#### 模块1: Generative Multiview World Model Post-training（生成式多视角世界模型后训练）

**设计动机**: 利用 [[World Model]] 从两帧标准观测中预测未来辅助视角，为模型注入跨视角场景演化知识，而无需额外相机。

**具体实现**:
- 候选辅助视角集合 $\mathcal{V}_{\text{sel}}$（LIBERO 上为 60°/120°/240°/300°/俯视，CALVIN 上为 30°/216°/288°/俯视）
- 每个视角的下一帧表示为 $N_{\text{tok}}=625$（$25\times25$）个 [[VQ-VAE|VQ token]]
- 通过自回归似然最大化训练模型，让它学会"想象"被遮挡视角中的场景演化

#### 模块2: Motion-Informative Token Compression（运动信息 Token 压缩）

**设计动机**: 完整生成一个视角需要 625 个 token、6–7 秒延迟，远不能满足实时控制；但动作预测真正需要的只是"发生了运动/变化"的局部区域。

**具体实现**:
- 用 [[余弦相似度]]衡量同一空间位置 token 在 $t$ 与 $t+1$ 时刻 embedding 的差异，得到逐 token 的运动相关性分数
- 取 Top $K=16$ 的高分 token（而非全部 625 个），将单视角 token 数从 625 压缩到 16（多视角候选总数从 3125 压缩到 80）
- 压缩后单视角推理延迟从 6–7 秒降至 0.2–0.3 秒

#### 模块3: Dynamic Test-Time View Selection via Action Entropy（动作熵驱动的动态视角选择）

**设计动机**: 不同候选辅助视角在不同时刻提供的信息量不同，人工固定视角或训练专门的视角选择器成本高；用动作分布的不确定性作为代理指标可以免训练地评估视角质量。

**具体实现**:
- 对每个候选视角 $v$，计算策略在该视角条件下输出动作分布的 [[Shannon Entropy|Shannon 熵]] $H_v$
- 选择熵最低（即模型对动作最有信心）的视角 $v^*$ 作为当前决策依据
- 每 30 个策略步动态刷新一次所选视角，无需额外训练或视角选择标签

---

## 关键公式

### 公式1: [[World Model|世界模型]]后训练目标

$$
\mathcal{L}_{\text{world}} = -\sum_{v \in \mathcal{V}_{\text{sel}}} \sum_{i=1}^{N_{\text{tok}}} \log p_\theta\left(c_{t+1,i}^v \mid c_{t+1,<i}^v,\ o_{t-1:t},\ L\right)
$$

**含义**: 对所有候选辅助视角 $v$，自回归地最大化下一帧 VQ token 序列的似然，使世界模型学会从两帧标准观测中预测多个视角的未来场景演化。

**符号说明**:
- $\mathbf{c}_{t+1}^v = (c_{t+1,1}^v, \ldots, c_{t+1,N_{\text{tok}}}^v)$: 视角 $v$ 下一帧的 VQ token 序列，$N_{\text{tok}}=625$
- $o_{t-1:t}$: 标准 agent-view 与 wrist-view 的两帧观测历史
- $\mathcal{V}_{\text{sel}}$: 候选辅助视角集合

### 公式2: [[Motion-Informative Token Compression]] 运动相关性打分

$$
\delta_{t,j}^v = 1 - \cos\left(E(c_{t,j}^v),\ E(c_{t+1,j}^v)\right), \quad j = 1, \ldots, N_{\text{tok}}
$$

**含义**: 用 token 嵌入在前后两帧的余弦距离衡量该位置的"运动/变化"程度，分数越高代表该 token 包含越多动作关键的视觉变化信息，据此挑选 Top-$K$ token 压缩表示。

**符号说明**:
- $E(\cdot)$: VQ 嵌入表（embedding table）
- $c_{t,j}^v, c_{t+1,j}^v$: 视角 $v$ 中位置 $j$ 在 $t$、$t+1$ 时刻的 token
- $\delta_{t,j}^v$: 位置 $j$ 的运动相关性分数，取 Top $K=16$ 的位置保留

### 公式3: 动作微调目标

$$
\mathcal{L}_{\text{act}} = -\sum_{v \in \mathcal{V}_{\text{sel}}} \sum_{m=1}^{K+N_a} \log p_\theta\left(s_{t,m}^v \mid s_{t,<m}^v,\ o_{t-1:t},\ L\right)
$$

**含义**: 将压缩后的辅助视角 token 与 [[FAST]] 动作 token 拼接为统一序列，按视角分别构造前缀，自回归地联合训练动作预测与压缩视角生成。

**符号说明**:
- $\mathbf{s}_t^v = (\hat{o}_{t+1,1}^{\prime v}, \ldots, \hat{o}_{t+1,K}^{\prime v}, a_{t,1}, \ldots, a_{t,N_a})$: 视角 $v$ 的压缩表示 + 动作 token 拼接序列
- $K=16$: 压缩后单视角保留的 token 数
- $N_a$: [[FAST]] 动作 token 数量

### 公式4: [[Action-Entropy View Selection]] 动作熵计算

$$
H_v = \frac{1}{N_a} \sum_{n=1}^{N_a} \mathcal{H}\left[p_\theta\left(a_{t,n} \mid a_{t,<n},\ \hat{o}_{t+1}^{\prime v},\ o_{t-1:t},\ L\right)\right]
$$

**含义**: 对每个候选视角 $v$，计算策略在该视角条件下逐个动作 token 的 Shannon 熵并取平均，得到该视角整体的"动作不确定性"评分，熵越低说明该视角下模型对动作越有信心。

**符号说明**:
- $\mathcal{H}[p] = -\sum_a p(a) \log p(a)$: Shannon 熵算子
- $\hat{o}_{t+1}^{\prime v}$: 视角 $v$ 的压缩辅助视角表示（16 个 token）
- $H_v$: 视角 $v$ 的平均动作熵

### 公式5: 视角选择规则

$$
v^* = \arg\min_{v \in \mathcal{V}_{\text{sel}}} H_v, \qquad \hat{o}_{t+1}^{\prime *} = \hat{o}_{t+1}^{\prime v^*}
$$

**含义**: 选取动作熵最低的视角作为当前决策所用的辅助视角，整个过程无需额外训练或人工标注的视角选择标签，每 30 个策略步重新评估一次。

**符号说明**:
- $v^*$: 被选中的最优辅助视角
- $\hat{o}_{t+1}^{\prime *}$: 最终送入动作头的压缩视角表示

---

## 关键图表

### Figure 1: Multiview Observations under Occlusion / 遮挡下的多视角观测

![Figure 1](https://arxiv.org/html/2606.21501v1/x1.png)

**说明**: 绿框标记动作熵最低（被 [[Action-Entropy View Selection]] 选中）的最优视角，红框标记熵较高的视角；柱状图展示各视角的动作熵，熵越低代表该视角越具"动作信息量"，直观体现遮挡场景下不同视角提供的决策价值差异。

### Figure 2: Motion-Informative Token Compression / 运动信息 Token 压缩

![Figure 2](https://arxiv.org/html/2606.21501v1/x2.png)

**说明**: 展示如何用 token 嵌入的余弦距离评分筛选 Top-16 运动相关 token，将每个生成视角从 625 个 token 压缩到 16 个，单视角延迟从 6–7 秒降至 0.2–0.3 秒。

### Figure 3: UniviewVLA Pipeline / 整体流程

![Figure 3](https://arxiv.org/html/2606.21501v1/x3.png)

**说明**: UniviewVLA 用统一的离散 token [[Transformer]] 自回归建模语言指令、多视角观测和动作；分两阶段训练（[[World Model]] 后训练 + 动作微调）并支持动态推理时视角选择。

### Figure 4: Full Future Auxiliary-View Token Generation / 完整未来辅助视角生成

![Figure 4](https://arxiv.org/html/2606.21501v1/x4.png)

**说明**: 前两列蓝框为标准物理相机视角（agent-view、wrist-view）输入，绿框为模型从这些输入生成的未来辅助视角，验证了世界模型可以在无额外相机的情况下"推断"被遮挡的场景演化。

### Figure 5: Real-Robot Occlusion Tasks / 真机遮挡任务

![Figure 5](https://arxiv.org/html/2606.21501v1/x5.png)

**说明**: 目标盘子或被操作物体在默认 agent-view 下部分被遮挡，需要额外的空间证据才能可靠完成抓取，体现真实部署中遮挡问题的现实性。

### Figure 6: Real-Robot Performance over 15 Trials / 真机 15 次试验表现

![Figure 6](https://arxiv.org/html/2606.21501v1/x6.png)

**说明**: 两个真机任务（Oreo-to-Plate、Occluded-Doll Move）各 15 次试验下，UniviewVLA 相比双相机和三相机基线的成功率对比。

### Figure 7: LIBERO Multiview / LIBERO 多视角设置

![Figure 7](https://arxiv.org/html/2606.21501v1/x7.png)

**说明**: LIBERO 仿真环境中用于训练多视角监督的 5 个相机角度（60°、120°、240°、300°、俯视）布局。

### Figure 8: CALVIN Multiview / CALVIN 多视角设置

![Figure 8](https://arxiv.org/html/2606.21501v1/x8.png)

**说明**: 由于 CALVIN 场景布局存在视角冗余，选用 4 个互补角度（30°、216°、288°、俯视）作为训练监督。

### Figure 9: Six Customized Occlusion-Focused Tasks / 六个自制遮挡任务

![Figure 9](https://arxiv.org/html/2606.21501v1/x9.png)

**说明**: 每个任务都在保持与标准两路物理观测不变的前提下，刻意把动作关键状态线索隐藏在默认 agent-view 视角之外（如柜内物体、瓶身背面等），用于专门评估遮挡场景下的策略表现。

### Figure 10: Real-Robot Multiview / 真机多视角设置

![Figure 10](https://arxiv.org/html/2606.21501v1/x10.png)

**说明**: Mobile [[ALOHA]] 平台上复用左腕相机作为侧视角，构成真机实验的多视角观测配置。

### Table 1: Long-Horizon Manipulation on CALVIN ABCD→D

| Method | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Avg. |
|--------|---------|---------|---------|---------|---------|------|
| RT-1 | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| Robo-Flamingo | 0.964 | 0.896 | 0.824 | 0.740 | 0.660 | 4.09 |
| GR-1 | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21 |
| MDT | 0.986 | 0.958 | 0.916 | 0.862 | 0.801 | 4.52 |
| RoboVLMs | 0.967 | 0.930 | 0.899 | 0.865 | 0.826 | 4.49 |
| Fast-dVLA | 0.984 | 0.952 | 0.922 | 0.870 | 0.812 | 4.54 |
| MINT | 0.974 | 0.942 | 0.917 | 0.882 | 0.861 | 4.57 |
| **UniviewVLA** | **0.983** | **0.958** | **0.928** | **0.893** | **0.838** | **4.60** |

**说明**: UniviewVLA 在 5 步任务链的平均完成数（Avg.）上达到 4.60，超过所有列出的 baseline，包括同为离散 token 自回归范式的 MINT（4.57）。

### Table 2: LIBERO Benchmark（成功率 %）

| Method | Long | Goal | Spatial | Object | Avg. |
|--------|------|------|---------|--------|------|
| OpenVLA | 53.7 | 79.2 | 84.9 | 88.4 | 76.6 |
| π₀-FAST | 60.2 | 88.6 | 96.4 | 96.8 | 85.5 |
| CoT-VLA | 69.0 | 87.6 | 87.5 | 91.6 | 83.9 |
| VLA-0 | 87.6 | 96.2 | 97.0 | 97.8 | 94.7 |
| GR00T N1 | 90.6 | 93.0 | 94.4 | 97.6 | 93.9 |
| UniVLA | 91.4 | 93.2 | 96.0 | 99.2 | 95.0 |
| OpenVLA-OFT | 90.7 | 96.2 | 96.2 | 98.3 | 95.4 |
| **UniviewVLA** | **94.2** | **93.4** | **96.4** | **99.2** | **95.8** |

**说明**: UniviewVLA 在 LIBERO 四个子集平均达到 95.8%，尤其在 Long-Horizon 子集（94.2%）上领先所有 baseline，说明多视角辅助信息对长时序任务的累积误差有缓解作用。

### Table 3: Camera-Configuration Evaluation（占用真实相机数量对比）

| Method | Cameras | LIBERO Long | LIBERO Goal | LIBERO Spatial | LIBERO Object | LIBERO Avg. | CALVIN Avg. |
|--------|---------|-------------|-------------|----------------|---------------|-------------|-------------|
| Two-Camera Policy | A+W | 90.2 | 93.4 | 92.8 | 95.6 | 93.0 | 4.42 |
| Three-Camera Policy | A+W+S | 94.2 | 95.4 | 93.6 | 97.4 | 95.2 | 4.54 |
| **UniviewVLA** | **A+W** | **94.2** | **93.4** | **96.4** | **99.2** | **95.8** | **4.60** |

**说明**: UniviewVLA 仅用两路物理相机（A=agent-view, W=wrist-view），生成的虚拟辅助视角效果即可匹配甚至超过真实部署第三路物理相机（S=side-view）的三相机策略，证明世界模型生成视角可以替代额外硬件。

### Table 4: Occlusion-Focused Evaluation（六个遮挡任务成功率 %）

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Avg. |
|--------|--------|--------|--------|--------|--------|--------|------|
| Two-Camera Policy | 24 | 68 | 34 | 26 | 84 | 4 | 40.0 |
| Three-Camera Policy | 52 | 74 | 68 | 88 | 100 | 16 | 66.3 |
| UniviewVLA (Manual) | 48 | 78 | 66 | 86 | 96 | 28 | 67.0 |
| **UniviewVLA (Entropy)** | **58** | **88** | **72** | **88** | **98** | **36** | **73.3** |

**关键发现**: 在专门设计的遮挡任务上，仅双相机基线只有 40.0% 成功率；引入生成式辅助视角后即可大幅提升（Manual 固定视角 67.0%），而用 [[Action-Entropy View Selection|动作熵动态选择视角]] 后进一步提升至 73.3%，比双相机基线高 +33.3 个百分点，且超过物理三相机策略（66.3%）。

### Table 5: 六个自制遮挡任务的语言指令

| ID | 任务 | 语言指令 |
|----|------|----------|
| 1 | Scene1 Bowl | "open the bottom drawer of the cabinet and put the bowl in it" |
| 2 | Scene2 Bowl | "put the black bowl on the left plate" |
| 3 | Scene4 Wine | "pick up the wine bottle at the back and put it on the wine rack" |
| 4 | Scene4 Drawer | "put the black bowl at the left in the bottom drawer of the cabinet and close it" |
| 5 | Scene7 Mug | "put the yellow and white mug on the plate" |
| 6 | Scene8 Moka | "turn off the stove and put the moka pot on top of the cabinet" |

**说明**: 六个任务均基于 LIBERO 风格场景改造，刻意让关键物体（碗、酒瓶、马克杯、摩卡壶等）的位置或状态在默认视角下部分或完全不可见，强制策略依赖额外视角信息。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 子集，每子集 500 episode 评估 | Long/Goal/Spatial/Object 四类泛化任务 | 标准 benchmark 评估 |
| [[CALVIN]] ABCD→D | 长时序多任务链 | 跨场景泛化，5 步任务链评估 | 标准 benchmark 评估 |
| 自制遮挡任务 | 6 个任务，每任务 60 条演示 | LIBERO 风格，3D-SpaceMouse 遥操作采集，关键线索被遮挡 | 遮挡场景专项评估 |
| 真机任务 | 2 个任务，每任务 100 条演示，各 15 次试验 | Mobile [[ALOHA]] 平台单臂操作（Oreo-to-Plate、Occluded-Doll Move） | 真实机器人部署评估 |

### 实现细节

- **Backbone**: 基于 Emu3 的离散 token 自回归 [[Transformer]]，沿用 [[UniVLA]] 范式
- **视觉 tokenizer**: [[VQ-VAE|VQ tokenizer]]，每视角 $25\times25=625$ token
- **动作 tokenizer**: [[FAST]]（Frequency-space Action Sequence Tokenizer）
- **优化器/学习率**: Stage 1 世界模型后训练用 cosine decay，从 $8\times10^{-5}$ 降至 $5\times10^{-6}$
- **Batch Size**: Stage 1 全局批大小 8
- **训练步数**: Stage 1 30K steps；Stage 2 LIBERO/CALVIN 24K steps，真机任务 30K steps
- **硬件**: 8× NVIDIA H 系列 GPU，bf16 + DeepSpeed ZeRO-3
- **推理时视角刷新频率**: 每 30 个策略步重新评估一次候选视角的动作熵

### 可视化结果

定性结果（Figure 4）显示，模型仅凭标准两路相机输入即可生成与真实场景演化高度一致的辅助视角；在真机遮挡任务（Figure 5、Figure 6）中，UniviewVLA 在 Oreo-to-Plate 任务成功率达 53.3%（双相机基线仅 13.3%），在 Occluded-Doll Move 任务达 46.7%（双相机基线 20.0%），两任务平均提升 +33.4 个百分点。

---

## 批判性思考

### 优点

1. **零额外硬件成本**: 不需要在部署端增加物理相机或保持训练/推理相机数量严格一致，工程落地友好。
2. **延迟工程做得扎实**: 通过 [[Motion-Informative Token Compression]] 把生成延迟从 6–7 秒压到 0.2–0.3 秒，是少有的真正考虑了"生成式辅助视角"实时性约束的工作。
3. **视角选择免训练**: [[Action-Entropy View Selection]] 用模型自身的动作分布熵作为信号，不需要额外的视角选择器训练或标签，设计简洁且可解释。
4. **在标准 benchmark 和专门遮挡任务上双重验证**: 既不牺牲 LIBERO/CALVIN 上的常规性能（95.8%、4.60，均超过对应 baseline），又在遮挡专项任务和真机部署上展现明确增益。

### 局限性

1. **生成视角仍有不可忽视的推理开销**: 论文自述"auxiliary-view generation adds modest inference overhead"，0.2–0.3 秒/视角在多视角候选场景下仍会累积。
2. **依赖多视角训练数据覆盖度**: 世界模型生成质量取决于训练时是否见过相近的视角和场景分布，跨域泛化能力未知。
3. **评测场景仍偏桌面操作**: 主要在 LIBERO/CALVIN 风格的桌面操作和小规模真机任务上验证，未覆盖移动视角、更长时序或非桌面场景。
4. **真机任务规模有限**: 仅 2 个真机任务、每任务 15 次试验，统计显著性有限，且其中一个任务（Occluded-Doll Move, 46.7%）实际低于三相机基线（53.3%），说明生成视角并非总能完全替代真实相机。

### 潜在改进方向

1. 将动作熵选择机制扩展到更多候选视角或连续视角空间，而非固定的离散角度集合。
2. 探索更轻量的世界模型架构以进一步降低单视角生成延迟，支持更高频的视角刷新。
3. 在移动操作、双臂协同等更复杂具身场景中验证生成式多视角范式的有效性。

### 可复现性评估
- [ ] 代码开源（项目主页未提供 GitHub 链接，仅有演示视频）
- [ ] 预训练模型
- [x] 训练细节完整（两阶段步数、学习率、batch size、硬件均有说明）
- [x] 数据集可获取（LIBERO、CALVIN 公开；自制遮挡任务和真机数据未明确是否开源）

---

## 关联笔记

### 基于
- [[UniVLA]]: UniviewVLA 沿用其离散 token 自回归建模范式作为基础架构
- [[World Model]]: 核心依赖的生成式世界模型，用于预测未来辅助视角
- [[FAST]]: 动作 tokenizer，用于将连续动作序列离散化

### 对比
- [[OpenVLA]] / [[OpenVLA-OFT]]: 标准 LIBERO/CALVIN 强 baseline
- [[π₀.₅|π₀-FAST]]: 同样使用 FAST 动作 tokenizer 的对比方法
- [[GR00T-N1.7|GR00T N1]]: LIBERO 上的对比 baseline
- Three-Camera Policy: 物理三相机策略，UniviewVLA 用生成视角达到甚至超越其效果

### 方法相关
- [[Motion-Informative Token Compression]]: 核心 token 压缩技术，支撑实时辅助视角生成
- [[Action-Entropy View Selection]]: 核心动态视角选择技术
- [[VQ-VAE]]: 图像离散化所用的视觉 tokenizer
- [[Action Chunking]]: 动作预测的基础范式（FAST token 序列即对应动作块）

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评估 benchmark
- [[CALVIN]]: 长时序多任务仿真评估 benchmark
- [[ALOHA]]: 真机实验所用的 Mobile ALOHA 双臂遥操作平台

---

## 速查卡片

> [!summary] UniviewVLA
> - **核心**: 用世界模型从两路标准相机观测生成被遮挡的"虚拟"辅助视角，零额外硬件解决机器人操作中的视角遮挡问题
> - **方法**: Motion-Informative Token Compression（625→16 token，延迟 6-7s→0.2-0.3s）+ training-free Action-Entropy View Selection（按动作熵动态选视角）
> - **结果**: LIBERO 95.8%、CALVIN 4.60（均超 baseline）；遮挡任务 40.0%→73.3%；真机平均提升 +33.4 个百分点
> - **代码**: 未开源（仅项目主页演示视频）

---

*笔记创建时间: 2026-06-24*

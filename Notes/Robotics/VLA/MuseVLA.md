---
title: "MuseVLA: An Adaptive Multimodal Sensing Vision-Language-Action Model for Robotic Manipulation"
method_name: "MuseVLA"
authors: [Xingyuming Liu, Ruichun Ma, Heyu Guo, Qixiu Li, Qingwen Yang, Lin Luo, Shiqi Jiang, Chenren Xu, Jiaolong Yang, Baining Guo]
year: 2026
venue: arXiv
tags: [multimodal-sensing, vla, dexterous-manipulation, sensor-fusion, diffusion-policy, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.17598
created: 2026-08-14
---

# 论文笔记：MuseVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft Research / Peking University |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[π0]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17598) |

---

## 一句话总结

> MuseVLA 将多模态传感器（热成像、声学、毫米波雷达）作为"按需调用的工具"，通过可学习的 Sensor Token 自适应选择传感模态并转化为统一的 Grounded Sensor Image，在灵巧手操作任务上实现了 80.6% 的平均成功率。

---

## 核心贡献

1. **自适应 Sensor Token 机制**: 引入可学习的传感器令牌（`<None>`、`<Thermal>`、`<Acoustic>`、`<mmWave>`），由 VLM 根据任务指令动态选择最相关的感知模态，避免了固定传感器融合的噪声累积
2. **Grounded Sensor Image 统一表示**: 用[[语义分割]]定位目标物体区域，将异构传感器热力图叠加到分割掩码上，转化为统一 RGB 表示送入 VLA，实现零额外模型改动的多模态融合
3. **数据合成流水线**: 通过在已有 RGB 数据集指令中注入物理属性关键词、在分割区域叠加颜色编码掩码，从 MolmoAct、AgiBotWorld-Alpha、VITRA 合成 9.6K 集数据，显著提升对未见传感器引导任务的泛化能力

---

## 问题背景

### 要解决的问题

现有 [[VLA|Vision-Language-Action 模型]] 仅依赖 RGB 视觉输入，无法感知温度、声音、毫米波反射等物理属性，导致机器人在需要辨别材质温度（热饮 vs 冷饮）、声学属性（震动物体）或雷达不可见遮挡物的任务中完全失效。

### 现有方法的局限

- **RGB-only VLA**（π₀、π₀.₅）：无传感器感知能力，无法处理热觉/声学/雷达引导任务
- **固定多传感器融合**：假设所有传感器始终可用且同等重要，引入无关传感器噪声，泛化性差
- **传感器专用架构**：如触觉、深度单模态扩展，难以统一多种异构传感器

### 本文的动机

将传感器类比为"按需调用的工具"：只有当任务需要特定物理属性时才激活对应传感器，并将异构传感数据统一转化为 RGB 热力图叠加表示，使现有 VLA 框架无需架构改动即可处理多模态感知。

---

## 方法详解

### 模型架构

MuseVLA 采用**解耦感知-行动**架构：

- **输入**: 语言指令 $l$ + RGB 观测 $o_t$
- **VLM Backbone**: [[PaliGemma-2]] — 生成 Sensor Token 和目标描述 $l_d$
- **分割模块**: [[SAM2|SAM3]] — 根据 $l_d$ 定位目标物体区域
- **传感器处理**: [[Digital Beamforming]]（雷达）/ 直接热力图（热成像、声学）→ [[Grounded Sensor Image]]
- **动作专家**: [[Diffusion Transformer (DiT)]] — 以 Grounded Sensor Image 为条件生成机械手动作
- **输出**: 12-DoF 灵巧手动作序列

### 核心模块

#### 模块一：Sensor Token 自适应选择

**设计动机**: 利用 VLM 的指令理解能力，根据任务语义动态决定激活哪种传感模态。

**具体实现**:
- 在词表中扩充四个特殊令牌：`<None>`、`<Thermal>`、`<Acoustic>`、`<mmWave>`
- VLM 在自回归解码时，首先生成 Sensor Token（多分类），再生成目标描述字符串 $l_d$
- 训练时只需监督 Sensor Token 的交叉熵损失 $\mathcal{L}_{\text{sensor}}$ 和目标描述的生成损失 $\mathcal{L}_{\text{target}}$

#### 模块二：Grounded Sensor Image 生成

**设计动机**: 将异构传感器读数（热力图、声强图、雷达反射图）转化为 [[语义分割]] 区域叠加的统一 RGB 表示，避免引入新的编码器。

**具体实现**:
1. 用 [[SAM2|SAM3]] 对目标描述 $l_d$ 做开放词汇语义分割，得到二值掩码 $M$
2. 将传感器热力图 $s_{i,t}$（已通过 [[Homography|单应变换]] 对齐到 RGB 视角）叠加到掩码区域
3. 非掩码区域保持原始 RGB，保留空间上下文

#### 模块三：毫米波雷达 Digital Beamforming

**设计动机**: 毫米波雷达输出稀疏天线阵列信号，需要[[Digital Beamforming|数字波束成形]]将其转化为 2D 空间热力图。

**具体实现**:
- 使用 Calterah 4T4R 60GHz 雷达，通过相位加权求和生成角度-强度热力图
- 声学阵列同样通过波束成形计算各方向声强，生成声学空间图

---

## 关键公式

### 公式一：[[VLA|多模态策略定义]]

$$
\pi: (l,\; o_t,\; s_{1,t}, \ldots, s_{N,t}) \rightarrow A
$$

**含义**: 策略 $\pi$ 以语言指令 $l$、RGB 观测 $o_t$ 和 $N$ 路传感器读数 $s_{i,t}$ 为输入，输出动作 $A$。

**符号说明**:
- $l$: 自然语言任务指令
- $o_t$: 时刻 $t$ 的 RGB 图像观测
- $s_{i,t}$: 第 $i$ 个传感器在时刻 $t$ 的读数（热力图/声强图/雷达图）
- $A$: 机械手动作序列

### 公式二：[[Grounded Sensor Image|解耦策略执行流]]

$$
\begin{aligned}
\pi_1 &: (l,\; o_t) \rightarrow (l_s,\; l_d) \\
G &: (o_t,\; s_{i,t},\; l_d) \rightarrow m_{i,t} \\
\pi_2 &: (l,\; o_t,\; l_s,\; l_d,\; m_{i,t}) \rightarrow A
\end{aligned}
$$

**含义**: 策略拆分为三步——VLM 生成 Sensor Token $l_s$ 和目标描述 $l_d$，Grounded Sensor Image 生成函数 $G$ 产生叠加图 $m_{i,t}$，最终策略以所有信息为条件输出动作。

**符号说明**:
- $l_s$: Sensor Token（`<None>`/`<Thermal>`/`<Acoustic>`/`<mmWave>`）
- $l_d$: 目标描述字符串（如 "the hot drink"）
- $m_{i,t}$: Grounded Sensor Image（传感器热力图叠加后的 RGB 图）

### 公式三：[[Grounded Sensor Image|传感器图像叠加]]

$$
M = f_{\text{seg}}(o_{\text{RGB}},\; l_d)
$$

$$
m = M \odot s + (1 - M) \odot o_{\text{RGB}}
$$

**含义**: 先对 RGB 图像做语义分割得到二值掩码 $M$，再按掩码将传感器热力图 $s$ 叠加到目标区域，非目标区域保持原始 RGB。

**符号说明**:
- $f_{\text{seg}}$: [[SAM2|SAM3]] 开放词汇分割函数
- $M \in \{0,1\}^{H \times W}$: 二值分割掩码
- $s$: 对齐到 RGB 视角的传感器热力图
- $\odot$: 逐像素乘法
- $m$: 最终 Grounded Sensor Image

### 公式四：[[Diffusion Transformer (DiT)|VLM 训练损失]]

$$
\mathcal{L}_{\text{VLM}} = \mathcal{L}_{\text{sensor}} + \mathcal{L}_{\text{target}}
$$

**含义**: VLM 端的训练损失由 Sensor Token 分类损失和目标描述生成损失两部分构成。

**符号说明**:
- $\mathcal{L}_{\text{sensor}}$: Sensor Token（4 类）的交叉熵分类损失
- $\mathcal{L}_{\text{target}}$: 目标描述字符串的自回归生成损失

### 公式五：[[Diffusion Transformer (DiT)|联合训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{VLM}} + \lambda\; \mathbb{E}_{\tau, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(a^\tau, \tau, c)\|_2^2\right]
$$

**含义**: 整体训练损失由 VLM 损失和扩散动作专家的去噪损失加权求和，$\lambda$ 控制两部分的平衡。

**符号说明**:
- $\lambda$: 损失平衡系数（实验中取 $10^{-2}$）
- $\tau$: 扩散时间步，$\tau \sim \mathcal{U}(0, 1)$
- $\varepsilon$: 添加的高斯噪声
- $\varepsilon_\theta$: 去噪网络
- $a^\tau$: 时间步 $\tau$ 的加噪动作
- $c$: 条件信息（RGB 图、Grounded Sensor Image、语言指令）

### 公式六：[[Digital Beamforming|毫米波雷达数字波束成形]]

$$
P(\theta, \phi) = 20 \log_{10} \left\| \sum_{k=1}^{K} a_k e^{j\phi_k} e^{-j\Delta_k(\theta, \phi)} \right\|
$$

$$
\Delta_k(\theta, \phi) = \frac{2\pi}{\lambda}\!\left( d_k^x \cos\phi \sin\theta + d_k^y \sin\phi \right)
$$

**含义**: 对 $K$ 个天线单元的接收信号做相位加权求和，计算各方向 $(\theta, \phi)$ 上的功率，生成角度域热力图。

**符号说明**:
- $P(\theta, \phi)$: 方位角 $\theta$、仰角 $\phi$ 方向的波束功率（dB）
- $a_k, \phi_k$: 第 $k$ 个天线的幅度和初相
- $\Delta_k(\theta, \phi)$: 第 $k$ 天线相对参考点的相位延迟
- $d_k^x, d_k^y$: 第 $k$ 天线的阵列坐标
- $\lambda$: 信号波长

### 公式七：[[Homography|传感器视角对齐（单应变换）]]

$$
[u,\; v,\; 1]^T \sim H_i\, [u_i,\; v_i,\; 1]^T
$$

**含义**: 通过单应矩阵 $H_i$ 将第 $i$ 个传感器坐标系下的像素坐标对齐到 RGB 图像坐标系。

**符号说明**:
- $(u_i, v_i)$: 传感器热力图中的像素坐标
- $(u, v)$: 对应的 RGB 图像像素坐标
- $H_i$: 传感器 $i$ 到 RGB 相机的 $3 \times 3$ 单应矩阵

### 公式八：[[Grounded Sensor Image|掩码坐标系变换]]

$$
M_i(u_i, v_i) = M\!\left(\pi\!\left(H_i\, [u_i,\; v_i,\; 1]^T\right)\right)
$$

**含义**: 将 RGB 空间的分割掩码 $M$ 通过逆单应变换投影回传感器坐标系，用于在传感器热力图上定位目标区域。

**符号说明**:
- $M_i$: 传感器坐标系下的掩码
- $\pi(\cdot)$: 从齐次坐标到像素坐标的投影函数

---

## 关键图表

### Figure 1: Adaptive Multisensory Overview / 自适应多感知机器人操作

![Figure 1](https://arxiv.org/html/2606.17598v2/intro-overview.png)

**说明**: MuseVLA 系统整体展示。面对"拿起热饮"等需要温度感知的任务，模型自适应激活热成像传感器；面对"找到振动物体"则激活声学传感器；面对雷达引导的遮挡取物任务则激活 mmWave。对比 RGB-only VLA 完全无法完成这些任务。

### Figure 2: Model Overview / 模型总览

![Figure 2](https://arxiv.org/html/2606.17598v2/model-overview.png)

**说明**: MuseVLA 完整架构。[[PaliGemma-2]] VLM 接收 RGB 图像和任务指令，输出 [[Grounded Sensor Image|Sensor Token]] 和目标描述；[[SAM2|SAM3]] 执行目标分割；传感器热力图与 RGB 融合形成 [[Grounded Sensor Image]]；[[Diffusion Transformer (DiT)]] 动作专家以 Grounded Sensor Image 为条件生成最终动作。

### Figure 3: Grounded Sensor Image Processing / 传感器图像处理流程

![Figure 3](https://arxiv.org/html/2606.17598v2/sensor-image.png)

**说明**: [[Grounded Sensor Image]] 生成的三步流程：(1) SAM3 用目标描述分割 RGB 图得到掩码 $M$；(2) 传感器热力图通过 [[Homography|单应变换]] 对齐到 RGB 视角；(3) 热力图叠加到掩码区域，非目标区域保留原始 RGB。

### Figure 4: Data Synthesis Pipeline / 数据合成流水线

![Figure 4](https://arxiv.org/html/2606.17598v2/data-synthesis.png)

**说明**: 从已有 RGB 机器人数据集合成多感知数据的流水线。向已有任务指令注入物理属性关键词（如"the warmest cup"），对目标物体做语义分割并叠加颜色编码掩码（红=热、绿=冷），模拟真实传感器观测。从 MolmoAct、AgiBotWorld-Alpha、VITRA 共生成 9.6K 集数据。

### Figure 5a: Robot Hardware Setup / 机器人硬件平台

![Figure 5a](https://arxiv.org/html/2606.17598v2/eval-hardware-setup.png)

**说明**: 实验平台配置。机械臂搭载 12-DoF 灵巧手，传感器模块包括 RGB-D 相机（Intel RealSense）、双 [[Thermal Camera|热成像相机]]（infiRay T2S）、[[mmWave Radar|毫米波雷达]]（Calterah 4T4R 60GHz）、麦克风阵列（Sipeed 6+1Mic）。

### Figure 5b: Task Objects / 任务物体

![Figure 5b](https://arxiv.org/html/2606.17598v2/eval-task-objects.png)

**说明**: 评测使用的 7 类物体，涵盖不同温度（热饮/冷饮）、声学属性（可振动/不可振动）和雷达反射属性（遮挡物/目标物）。

### Figure 6: Unseen Task Execution Trajectories / 未见任务执行轨迹

![Figure 6 - Thermal](https://arxiv.org/html/2606.17598v2/unseen_thermal.png)
![Figure 6 - Acoustic](https://arxiv.org/html/2606.17598v2/unseen_acoustic.png)
![Figure 6 - Radar](https://arxiv.org/html/2606.17598v2/unseen_radar.png)

**说明**: MuseVLA 在未见任务（unseen tasks）上的执行轨迹，分别对应热觉引导、声学引导、雷达引导三种模态。预训练后泛化成功率为 66.7%。

### Figure 7: Training Task Execution Trajectories / 训练任务执行轨迹

![Figure 7 - Open Box](https://arxiv.org/html/2606.17598v2/seen_open_box.png)
![Figure 7 - Pick Clothes](https://arxiv.org/html/2606.17598v2/seen_pick_clothes.png)
![Figure 7 - Pick Towels](https://arxiv.org/html/2606.17598v2/seen_pick_towels.png)
![Figure 7 - Pick Hot Drink](https://arxiv.org/html/2606.17598v2/seen_pick_hot_drink.png)
![Figure 7 - Pick Cold Drink](https://arxiv.org/html/2606.17598v2/seen_pick_cold_drink.png)

**说明**: MuseVLA 在训练任务变体上的执行轨迹，涵盖开箱、拣选衣物、拣选毛巾、拣选热饮、拣选冷饮等子任务。

### Figure 8: Multi-Stage Task Execution / 多阶段任务执行轨迹

![Figure 8 - Thermal Stage 1](https://arxiv.org/html/2606.17598v2/multi_stage_thermal_1.png)
![Figure 8 - Thermal Stage 2](https://arxiv.org/html/2606.17598v2/multi_stage_thermal_2.png)
![Figure 8 - RGB Stage 1](https://arxiv.org/html/2606.17598v2/multi_stage_rgb_1.png)
![Figure 8 - RGB Stage 2](https://arxiv.org/html/2606.17598v2/multi_stage_rgb_2.png)

**说明**: 多阶段任务中，机器人先用热成像识别目标物体，再执行精确抓取放置。展示了 Sensor Token 在多步骤任务中的正确切换。

### Figure 9: Zero-Shot Unseen Task Execution / 零样本未见任务执行轨迹

![Figure 9 - Acoustic Cloth Bag](https://arxiv.org/html/2606.17598v2/unseen_acoustic_cloth_bag.png)
![Figure 9 - Acoustic Plush Toy](https://arxiv.org/html/2606.17598v2/unseen_acoustic_plush_toy.png)
![Figure 9 - Radar Cloth Bag](https://arxiv.org/html/2606.17598v2/unseen_radar_cloth_bag.png)
![Figure 9 - Radar Clothes](https://arxiv.org/html/2606.17598v2/unseen_radar_clothes.png)
![Figure 9 - Thermal Cold Drink](https://arxiv.org/html/2606.17598v2/unseen_thermal_cold_drink.png)
![Figure 9 - Thermal Plastic Bag](https://arxiv.org/html/2606.17598v2/unseen_thermal_plastic_bag.png)

**说明**: 零样本评估中，MuseVLA（预训练版本）在从未见过的物体+传感器组合上成功执行任务，展示合成数据预训练带来的跨物体泛化能力。

---

### Table 1: 任务成功率对比（训练任务）

| 方法 | Thermal | Acoustic | mmWave | 平均 | Sensing 分 | Manipulation 分 | 综合分 |
|------|---------|----------|--------|------|-----------|----------------|--------|
| π₀-RGB | 33.3% | 25.0% | 4.2% | 20.8% | 48.6% | 43.1% | 0.458 |
| π₀.₅-RGB | 16.7% | 33.3% | 8.3% | 19.4% | 41.7% | 33.3% | 0.375 |
| π₀-Raw | 16.7% | 41.7% | 25.0% | 27.8% | 86.1% | 27.8% | 0.569 |
| π₀.₅-Raw | 16.7% | 33.3% | 20.8% | 23.6% | 83.3% | 29.2% | 0.563 |
| MuseVLA-RGB | 12.5% | 33.3% | 20.8% | 22.2% | 41.7% | 43.1% | 0.424 |
| MuseVLA-Raw | 41.7% | 25.0% | 33.3% | 33.3% | 91.7% | 33.3% | 0.625 |
| MuseVLA-RawAdapt | 70.8% | 41.7% | 66.7% | 59.7% | 93.1% | 59.7% | 0.764 |
| **MuseVLA（完整）** | **83.3%** | **58.3%** | **87.5%** | **76.4%** | **95.8%** | **77.8%** | **0.868** |

**表格说明**: Sensing 分衡量传感器正确选择+目标定位；Manipulation 分衡量物理操作成功率。MuseVLA 相比 RGB-only 基线提升 58%，相比 Raw 传感器基线提升 47%。MuseVLA-RawAdapt（无 Grounded Sensor Image）表明自适应选择本身就带来 +26.4pp 的提升。

---

### Table 2: 自适应传感器选择准确率

| 方法 | 训练任务-Sensor 准确率 | 训练任务-Target 准确率 | 未见任务-Sensor 准确率 | 未见任务-Target 准确率 |
|------|----------------------|----------------------|----------------------|----------------------|
| PaliGemma-2（零样本） | 0% | 13.0% | 0% | 9.5% |
| MuseVLA（无预训练） | 100% | 100% | 85% | 40.5% |
| **MuseVLA（预训练）** | **100%** | **93.5%** | **100%** | **82.0%** |

**表格说明**: 预训练使未见任务的 Target 描述准确率从 40.5% 提升至 82.0%，说明合成数据有效改善了目标描述的语言泛化。

---

### Table 3: 预训练对合成数据的影响（seen/unseen 任务分析）

| 方法 | 已见-Thermal | 已见-Acoustic | 已见-mmWave | 已见均值 | 未见-Thermal | 未见-Acoustic | 未见-mmWave | 未见均值 |
|------|------------|--------------|------------|---------|------------|--------------|------------|---------|
| MuseVLA-Raw | 41.7% | 25.0% | 33.3% | 33.3% | 31.3% | 25.0% | 18.8% | 25.0% |
| MuseVLA（无预训练） | 83.3% | 58.3% | 87.5% | 76.4% | 25.0% | 31.3% | 25.0% | 27.1% |
| **MuseVLA（预训练）** | **87.5%** | **70.8%** | **83.3%** | **80.6%** | **75.0%** | **56.3%** | **68.8%** | **66.7%** |

**关键发现**: 不加预训练时未见任务成功率与 Raw 基线相当（~25%）；加入合成数据预训练后，未见任务成功率跃升至 66.7%，说明合成数据是跨物体泛化的关键。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人数据 | 720 集 | 10 种子任务、7 类物体、3 种传感模态 | 微调训练 |
| 合成多感知数据 | 9.6K 集（1.05M 帧） | 从 MolmoAct、AgiBotWorld-Alpha、VITRA 合成 | 预训练 |

### 实现细节

- **VLM Backbone**: [[PaliGemma-2]]（3B 参数）
- **权重初始化**: VITRA 的 VLM + 动作专家权重
- **优化器**: AdamW，学习率 $1 \times 10^{-5}$
- **Batch Size**: 512
- **训练步数**: 20,000 步（约 20 小时）
- **硬件**: 64 × A100 (40GB)
- **损失权重**: $\lambda = 10^{-2}$

### 可视化结果

- MuseVLA 能在多阶段任务中正确切换传感模态（热成像→抓取→放置）
- 零样本泛化到未见物体（布袋、毛绒玩具等）时仍能正确识别传感属性
- mmWave 引导的遮挡取物任务中，雷达热力图清晰定位被遮挡目标

---

## 批判性思考

### 优点

1. **架构轻量**: Sensor Token + Grounded Sensor Image 的设计无需修改 VLA 主体架构，只扩充词表和图像预处理步骤
2. **数据高效**: 合成数据流水线从已有 RGB 数据集自动生成，无需真实多传感器采集
3. **扩展性好**: 新增传感器只需扩充词表 Token 和对应热力图处理模块

### 局限性

1. **传感器物理约束**: 热力图分辨率较低（infiRay T2S 为低分辨率热成像），在精密操作中定位精度受限
2. **单目标假设**: Grounded Sensor Image 每次只叠加一个目标区域的传感数据，多目标场景（如多个热饮同时存在）需要额外设计
3. **硬件依赖**: 需要专业多传感器硬件平台，部署门槛高于纯 RGB VLA

### 潜在改进方向

1. 将 Sensor Token 扩展为多个 Token，支持同时激活多种传感器的多目标感知
2. 探索传感器模态的软注意力加权，替代当前的硬性单模态选择
3. 利用更高分辨率传感器（如高精度热成像）提升精密操作能力

### 可复现性评估

- [x] 代码开源（论文中提及）
- [x] 预训练模型（论文中提及）
- [x] 训练细节完整
- [ ] 数据集可获取（真实机器人数据依赖特定硬件）

---

## 关联笔记

### 基于

- [[PaliGemma-2]]: VLM backbone
- [[Diffusion Transformer (DiT)]]: 动作专家架构
- [[SAM2]]: 目标分割（论文用 SAM3）

### 对比

- [[π0]]: RGB-only 基线，同为灵巧手操作 VLA
- [[π0.5]]: RGB-only 基线

### 方法相关

- [[Grounded Sensor Image]]: 核心统一表示
- [[Sensor Token]]: 自适应选择机制
- [[Digital Beamforming]]: 雷达信号处理
- [[Homography]]: 多传感器视角对齐
- [[语义分割]]: 目标区域定位

### 硬件/数据相关

- [[mmWave Radar]]: Calterah 4T4R 60GHz 毫米波雷达
- [[Thermal Camera]]: infiRay T2S 热成像相机

---

## 速查卡片

> [!summary] MuseVLA
> - **核心**: 将传感器作为按需工具，通过 Sensor Token 自适应选择，Grounded Sensor Image 统一表示
> - **方法**: PaliGemma-2 生成 Sensor Token + SAM3 分割 + 传感器热力图叠加 + DiT 动作生成
> - **结果**: 已见任务 80.6% 成功率，未见任务 66.7%，超 RGB-only 基线 58%
> - **代码**: 见 arXiv 项目页

---

*笔记创建时间: 2026-08-14*

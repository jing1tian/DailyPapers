---
title: "Foresight Without Seeing: Latent Futures for World Action Models"
method_name: "ForeWAM"
authors: [Jiakai Huang, Zhongbo Wu, Zheng Zhang, Zihan Wang, Shan You, Tao Huang]
year: 2026
venue: arXiv
tags: [world-action-model, latent-future, kv-cache, flow-matching, diffusion-policy, robot-manipulation, knowledge-distillation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.11605v1
created: 2026-08-14
---

# 论文笔记：Foresight Without Seeing: Latent Futures for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未列出（arXiv 预印本） |
| 日期 | August 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Fast-WAM]], [[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.11605) / 代码暂未开放 |

---

## 一句话总结

> ForeWAM 通过在噪声初始化的未来槽上执行一次视频 KV 预填充，将预测性动力学以隐式缓存状态注入 [[ActionDiT]]，使 2B 直接策略 WAM 在无任何具身预训练的情况下，于 LIBERO 上达到 96.9%，并在 LIBERO-Plus 鲁棒性基准上以 +10.1 分大幅超越 6B Fast-WAM。

---

## 核心贡献

1. **Future-KV Interface**: 对当前帧潜变量与噪声初始化的未来槽执行单次 [[KV Cache|KV 预填充]]，缓存逐层 Key-Value 状态以供 [[ActionDiT]] 动作去噪使用，推理时无需显式生成任何未来帧。
2. **Latent-Action-Supervised Dynamics Registers**: 冻结 [[LaWAM|LaWM]] 教师编码视觉转换为潜在动作目标，监督紧凑 [[Register Token|动力学寄存器]] 捕获交互引发的转换（物体运动、接触变化、任务进度）。
3. **Training-Inference Asymmetry（训练-推理非对称性）**: 真实未来帧和教师模型仅在训练时使用，推理时无需任何未来观测或视频生成，动作生成延迟降至 568ms（相比 [[Fast-WAM]] 的 667ms），Flash 变体达 220ms。

---

## 问题背景

### 要解决的问题

直接策略 [[Fast-WAM|World Action Model（WAM）]] 跳过推理时的未来视频生成以提高效率，但这消除了预测性动力学到达动作预测系统的显式路径。核心问题：**如何让直接策略 WAM 的 [[ActionDiT]] 在无需显式生成未来观测的情况下访问预测性动力学？**

### 现有方法的局限

| 范式 | 描述 | 局限 |
|------|------|------|
| 级联 WAM（Cascaded WAM） | 先生成未来帧再预测动作 | 推理时计算代价极高 |
| 联合 WAM（Joint WAM） | 在统一生成过程中联合生成视频和动作 | 仍需多步去噪，延迟高 |
| 直接策略 WAM | [[Fast-WAM]]（6B 参数）跳过未来生成 | 完全丢失了预测性动力学上下文 |

### 本文的动机

预测性动力学可以在不被显式实例化为未来观测的情况下惠及直接动作策略。通过 [[Flow-Matching|Flow Matching]] 视频 DiT 预填充生成的 [[KV Cache|KV 状态]] 隐含了世界的预测性动力学，这些状态可在不迭代生成视频的情况下被 [[ActionDiT]] 动作去噪器直接利用。

---

## 方法详解

### 模型架构

ForeWAM 采用**双 DiT 共享主干**架构：

- **输入**: 语言指令 $l$ + 当前帧 $o$（两视角 224×224 拼接为 224×448）+ 本体感觉状态 $p$（8维）
- **Video Backbone**: [[Wan2.1]] T2V-1.3B（预训练初始化，30个 Transformer 块，$d_v = 1536$）
- **核心模块**: [[Future-KV|Future-KV Interface]] 用于缓存预测性上下文；[[Dynamics Registers|Latent-Action-Supervised Dynamics Registers]] 用于捕获交互动力学
- **Action DiT**: 30个 Transformer 块，$d_a = 1024$（插值检查点初始化）
- **输出**: 7维动作块 $a_{1:H}$（6-DoF 末端执行器位姿 + 夹爪控制），动作水平线 $H=32$
- **总参数**: ~2B

### 四类 Token 组与结构化注意力

模型处理四类 token 组，通过[[Isolated Attention Mask|结构化掩码]]控制注意力路由：

- **C（Current-frame tokens）**: 当前帧编码
- **D（Dynamics registers）**: 动力学寄存器（$N_D = 16$ 个 [[Register Token|寄存器]]）
- **F（Future-slot tokens）**: 未来槽 token（噪声初始化）
- **A（Action tokens）**: 动作序列 token

注意力路由规则：
- 动作路径读取完整视频序列（C + D）
- 未来槽整合当前帧和动力学寄存器（C + D）
- A token 通过 [[KV Cache|KV 缓存]] 读取 F 的预测性上下文（即 Future-KV）

### 核心模块

#### 模块1: Future-KV Interface

**设计动机**: 利用 [[Flow-Matching|视频 Flow Matching]] 的预测能力，通过单次 [[KV Cache|KV 预填充]] 将未来世界状态的隐式表示传递给 [[ActionDiT]] 动作去噪器，无需迭代生成视频帧。

**具体实现**:

1. 构建随机未来底物（Stochastic Future Substrate）：将当前帧潜变量与噪声初始化的未来槽拼接
2. 对底物执行单次视频 DiT 前向传播（KV 预填充），获取逐层 KV 状态缓存
3. 动作去噪阶段全程复用缓存的 KV 状态

**训练时**: 有监督真实未来帧参与视频 [[Flow-Matching|Flow Matching]] 目标训练，形成预测能力。

**推理时**: 无真实未来帧，Future-KV 从噪声初始化的未来槽生成，仍提供预测性上下文。

#### 模块2: Latent-Action-Supervised Dynamics Registers

**设计动机**: 通过 [[Knowledge Distillation|知识蒸馏]] 让紧凑的 [[Register Token|寄存器]] token 捕获交互引发的转换（物体运动、接触变化、任务进度），弥补直接策略 WAM 缺乏对世界状态变化理解的缺陷。

**具体实现**:

1. 使用冻结的 [[LaWAM|LaWM]] 教师编码视觉转换 $(o_t, o_{t+1})$ 为潜在动作目标 $z_{LA}$（32维）
2. 训练投影头 $g_\psi$，将 $N_D = 16$ 个动力学寄存器均值映射到潜在动作空间
3. 使用 stop-gradient 算子 $\operatorname{sg}(\cdot)$ 防止梯度回流到教师模型
4. 推理时无需教师模型，寄存器自主捕获动力学信息

---

## 关键公式

### 公式1: [[Latent-Action|直接策略 WAM 分布]]

$$
p_\theta(a_{1:H} \mid o, l, p)
$$

**含义**: 直接策略 WAM 的基本形式，动作分布直接条件于当前观测、语言指令和本体感觉，不经过任何显式未来视频生成。

**符号说明**:
- $a_{1:H}$: 长度为 $H=32$ 的动作序列
- $o$: 当前帧观测（两视角拼接）
- $l$: 语言任务指令
- $p$: 本体感觉状态（8维）

### 公式2: [[Latent-Action|显式未来 WAM 因式分解]]

$$
p(a_{1:H} \mid o, l, p) = \int p_\phi(u_{1:T} \mid o, l, p)\, p_\theta(a_{1:H} \mid o, l, p, u_{1:T})\, du_{1:T}
$$

**含义**: 级联/联合 WAM 的完整因式分解，动作分布边缘化掉显式生成的未来观测序列 $u_{1:T}$，计算代价高。

**符号说明**:
- $u_{1:T}$: 未来 $T$ 步的显式观测序列
- $p_\phi$: 视频生成模型（生成未来观测）
- $p_\theta$: 动作预测模型（条件于未来观测）

### 公式3: [[Future-KV|随机未来底物（Stochastic Future Substrate）]]

$$
\tilde{z}^{F_{\text{sub}}}_{1:T} = \operatorname{concat}\bigl(z_{\text{cur}}(o),\ \varepsilon_F\bigr), \quad \varepsilon_F \sim \mathcal{N}(0, I)
$$

**含义**: 将当前帧编码的潜变量与随机噪声初始化的未来槽拼接，构成视频 KV 预填充的输入底物。

**符号说明**:
- $z_{\text{cur}}(o)$: 当前帧 $o$ 经 VAE 编码的潜变量
- $\varepsilon_F$: 噪声初始化的未来槽，服从标准正态分布
- $\tilde{z}^{F_{\text{sub}}}_{1:T}$: 随机未来底物（包含干净当前帧 + 噪声未来槽）

### 公式4: [[KV Cache|KV 预填充（KV Prefill）]]

$$
(D_\theta,\; H_{KV}) = \operatorname{KVPrefill}_\phi\!\bigl(\tilde{z}^{F_{\text{sub}}}_{1:T},\, l,\, p\bigr)
$$

**含义**: 对随机未来底物执行单次视频 DiT 前向传播，同时产生动力学寄存器激活 $D_\theta$ 和逐层 KV 缓存 $H_{KV}$，供后续动作去噪复用。

**符号说明**:
- $D_\theta$: 动力学寄存器的激活状态（捕获隐式预测性动力学）
- $H_{KV}$: 视频 DiT 各层的 Key-Value 状态缓存
- $\phi$: 视频 DiT 模型参数

### 公式5: [[Dynamics Registers|推理时策略分布]]

$$
p_\theta\!\bigl(a_{1:H} \mid o, l, p,\; D_\theta(o, l, p, \varepsilon_F),\; H_{KV}(o, l, p, \varepsilon_F)\bigr)
$$

**含义**: ForeWAM 在推理时的完整策略分布，以动力学寄存器激活和 KV 缓存为条件，无需任何真实未来观测，体现训练-推理非对称性。

### 公式6: [[Flow-Matching|Flow Matching 线性插值]]

$$
y_t = (1-t)\,y + t\,\varepsilon
$$

**含义**: Flow Matching 训练中在干净样本 $y$ 和噪声 $\varepsilon$ 之间进行线性插值，$t \in [0,1]$ 为插值系数。

**符号说明**:
- $y$: 干净样本（视频潜变量序列或动作序列）
- $\varepsilon$: 随机采样的高斯噪声
- $t$: 时间步（插值系数，从 0 到 1）

### 公式7: [[Flow-Matching|Flow Matching 损失]]

$$
\mathcal{L}_{FM}(y) = \mathbb{E}_{y,\varepsilon,t}\Bigl[\bigl\|f_\theta(y_t, t, o, l, p) - (\varepsilon - y)\bigr\|_2^2\Bigr]
$$

**含义**: Flow Matching 训练目标，最小化网络预测的速度场与真实插值方向之间的均方误差。

**符号说明**:
- $f_\theta$: 神经网络预测的速度场（从噪声到干净样本的方向）
- $y_t$: 在时间步 $t$ 的插值样本
- $(\varepsilon - y)$: 真实的速度方向（从干净样本指向噪声）

### 公式8: 视频与动作的 Flow Matching 损失

$$
\mathcal{L}_{video} = \mathcal{L}_{FM}(z_{1:T}), \quad \mathcal{L}_{action} = \mathcal{L}_{FM}(a_{1:H})
$$

**含义**: 分别对视频潜变量序列（监督 Future-KV 的预测能力）和动作序列（监督可执行动作生成）应用 [[Flow-Matching|Flow Matching]] 损失。

**符号说明**:
- $z_{1:T}$: 未来 $T$ 帧的视频潜变量序列（训练时来自真实未来帧）
- $a_{1:H}$: 长度为 $H=32$ 的动作序列

### 公式9: [[Dynamics Registers|Latent-Action 对齐蒸馏损失]]

$$
\mathcal{L}_{LA} = \Bigl\|g_\psi\!\Bigl(\frac{1}{N_D}\sum_{i=1}^{N_D} D_i\Bigr) - \operatorname{sg}(z_{LA})\Bigr\|_2^2
$$

**含义**: 将 $N_D$ 个动力学寄存器的均值通过投影头映射到潜在动作空间，最小化与冻结 [[LaWAM|LaWM]] 教师输出之间的均方误差，使寄存器捕获交互动力学。

**符号说明**:
- $g_\psi$: 可训练投影头（将寄存器激活映射到 32维潜在动作空间）
- $N_D = 16$: 动力学寄存器数量
- $D_i$: 第 $i$ 个动力学寄存器的激活向量
- $\operatorname{sg}(\cdot)$: stop-gradient 算子（阻止梯度回流到教师）
- $z_{LA}$: 冻结 LaWM 教师编码的潜在动作目标（32维）

### 公式10: 总训练目标

$$
\mathcal{L} = \mathcal{L}_{video} + \mathcal{L}_{action} + \lambda_{LA} \cdot \mathcal{L}_{LA}
$$

**含义**: 三个互补损失的加权组合，共同优化视频未来预测、可执行动作生成和动力学寄存器对齐三个目标。

**符号说明**:
- $\mathcal{L}_{video}$: 视频潜变量 Flow Matching 损失（监督未来预测能力，仅训练时使用）
- $\mathcal{L}_{action}$: 动作 Flow Matching 损失（监督可执行动作生成）
- $\lambda_{LA}$: Latent-Action 损失权重超参数

---

## 关键图表

### Figure 1: World Action Model 范式对比

![Figure 1: WAM Paradigm Comparison](https://arxiv.org/html/2608.11605v1/paradigm.png)

**说明**: 四种 WAM 范式对比：(a) 级联 WAM 先生成未来观测再预测动作；(b) 联合 WAM 在统一生成过程中同时生成视频和动作；(c) 直接策略 WAM（如 [[Fast-WAM]]）跳过未来生成，仅使用当前观测的潜在世界表示；(d) ForeWAM 保留直接动作预测，同时通过隐式 [[Future-KV|Future-KV 状态]] 和 [[Dynamics Registers|动力学寄存器]] 暴露预测性动力学。斜线填充的 token 表示噪声变量；未来槽为随机内部状态，而非已观测到的未来帧。

### Figure 2: Dynamics-conditioned Action DiT 架构

![Figure 2: ForeWAM Architecture](https://arxiv.org/html/2608.11605v1/architecture_1.png)

**说明**: ForeWAM 的核心训练-推理非对称架构。**训练时**：真实未来帧监督视频 [[Flow-Matching|Flow Matching]] 目标，冻结 [[LaWAM|LaWM]] 教师提供潜在动作监督目标；**推理时**：保留当前帧潜变量，未来槽用噪声 $\varepsilon_F$ 初始化，一次视频 DiT 前向传播生成供 [[ActionDiT]] 动作去噪全程读取的逐层 [[KV Cache|KV 缓存]]。

### Figure 3: 结构化注意力掩码

**说明**: 展示 ForeWAM 中四类 token 组（C、D、F、A）的[[Isolated Attention Mask|结构化注意力路由]]。动作 token（A）通过 [[Future-KV|Future-KV]] 读取未来槽（F）的 KV 缓存，同时能访问当前帧（C）和 [[Dynamics Registers|动力学寄存器]]（D）。该路由定义架构连接，并不直接证明 token 语义的解耦。（注：图片暂无外链，可运行图片检查脚本本地化。）

### Table 1: LIBERO 标准测试集成功率（%，50 次评估/任务）

| Method | Params | 具身预训练 | Spatial | Object | Goal | Long | Overall |
|--------|--------|-----------|---------|--------|------|------|---------|
| [[OpenVLA]] | 7B | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| [[pi0\|π₀]] | 3.3B | Yes | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| [[Pi0.5\|π₀.₅]] | 3.3B | Yes | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| [[Pi0-FAST\|π₀-Fast]] | 3.3B | Yes | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| [[UniVLA]] | 7B | Yes | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| [[WorldVLA]] | 7B | Yes | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| [[Fast-WAM]] | 6B | No | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **ForeWAM** | **2B** | **No** | **97.0** | **99.6** | **97.2** | **92.8** | **96.7** |
| **ForeWAM-Flash** | **2B** | **No** | **97.8** | **99.2** | **97.4** | **93.0** | **96.9** |

**表格说明**: ForeWAM 以 2B 参数（仅 Fast-WAM 1/3）达到与 6B [[Fast-WAM]] 相当的平均成功率（96.7% vs 97.6%），且无需任何具身预训练数据。ForeWAM-Flash 达 96.9%，与 [[Pi0.5|π₀.₅]] 持平。

### Table 2: LIBERO-Plus 扰动鲁棒性测试成功率（%）

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|--------|--------|-------|----------|-------|------------|-------|--------|---------|
| [[OpenVLA]] | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| [[pi0\|π₀]] | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| [[Pi0.5\|π₀.₅]] | 75.4 | 77.5 | 85.6 | 96.9 | 94.6 | 89.7 | 85.7 | 85.7 |
| [[Pi0-FAST\|π₀-Fast]] | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| [[UniVLA]] | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| [[WorldVLA]] | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| [[Fast-WAM]] | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| **ForeWAM** | **62.5** | 37.4 | **73.0** | 74.1 | 55.1 | **58.8** | **70.4** | **61.6** |
| **ForeWAM-Flash** | 57.9 | **40.4** | 67.2 | 71.0 | 53.0 | 53.7 | 65.3 | 58.2 |

**表格说明**: ForeWAM 在 [[LIBERO-Plus]] 鲁棒性基准上以 61.6% 整体成功率超越 [[Fast-WAM]] +10.1 分。最大增益来自**摄像头视角扰动**（+46.1 分，16.4% → 62.5%）和**传感器噪声**（+21.1 分，37.7% → 58.8%）。仅机器人形态扰动类别（37.4%）仍低于 Fast-WAM（44.5%）。

### Table 3: 动作生成推理延迟（ms）

| Method | 推理延迟 (ms) |
|--------|-------------|
| [[Fast-WAM]] | 667 |
| ForeWAM | 568 |
| ForeWAM-Flash | 220 |

**表格说明**: ForeWAM 通过单次 [[KV Cache|KV 预填充]] 复用机制将推理延迟从 667ms 降至 568ms（-15%）；ForeWAM-Flash（仅 2步去噪）进一步降至 220ms（-67%），具备实时控制潜力。

### Table 4: LIBERO-Plus 消融实验（整体成功率 %）

| 配置 | Overall (%) | 说明 |
|------|------------|------|
| Base policy（基础策略） | 53.6 | 无 Future-KV，无 LA 监督 |
| Future-KV only | 58.5 | 仅加入 [[Future-KV|Future-KV Interface]] |
| LA supervision only | 58.0 | 仅加入 [[Dynamics Registers|Latent-Action 监督]] |
| **Ours（两者均有）** | **61.6** | 完整 ForeWAM |

**关键发现**: [[Future-KV]] 和 [[Dynamics Registers|Latent-Action 监督]] 各自独立带来约 +5 分提升，两者结合后提升 +8 分，呈现互补性（作者谨慎指出此消融无法证明因果必要性，需更具针对性的干预实验）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 130个任务，4个子套件（Spatial/Object/Goal/Long） | 标准机器人操作 benchmark | 训练 + 测试（in-distribution） |
| [[LIBERO-Plus]] | 7个扰动类别（摄像头/机器人/语言/灯光/背景/噪声/布局） | 系统性 OOD 扰动测试 | 测试（out-of-distribution 鲁棒性） |

### 实现细节

- **Video Backbone**: [[Wan2.1]] T2V-1.3B，30个 Transformer 块，$d_v = 1536$
- **Action DiT**: 30个 Transformer 块，$d_a = 1024$（插值检查点初始化）
- **动力学寄存器**: $N_D = 16$，潜在动作目标维度 32
- **优化器**: [[AdamW]]，学习率 $1 \times 10^{-4}$，权重衰减 0.01
- **学习率调度**: 余弦退火（Cosine Annealing）
- **梯度裁剪**: 1.0
- **动作去噪步数**: 10步（标准版），2步（Flash 变体）
- **Flow Matching 参数**: 1000 时间步，shift = 5.0
- **训练序列**: 33帧观测，时间比例 4:1（动作步数:视频帧数）
- **相机输入**: 两视角 224×224 拼接为 224×448
- **本体感觉**: 8维；**动作输出**: 7维（6-DoF 末端执行器 + 夹爪）
- **硬件**: NVIDIA A800 80GB（推理延迟测量）
- **预训练策略**: 无具身机器人数据预训练，仅在 LIBERO 上训练

---

## 批判性思考

### 优点

1. **参数效率极高**: 2B 模型在标准 LIBERO 上与 6B [[Fast-WAM]] 持平（96.7% vs 97.6%），参数量仅 1/3
2. **鲁棒性显著提升**: LIBERO-Plus 上 +10.1 分，尤其摄像头视角扰动 +46.1 分，表明隐式预测上下文显著增强了场景理解能力
3. **推理轻量化**: 单次预填充后复用 [[KV Cache|KV 缓存]]，Flash 变体延迟仅 220ms
4. **训练-推理解耦优雅**: 教师模型和真实未来帧仅在训练使用，部署时零额外依赖
5. **科学态度严谨**: 作者明确说明消融实验的局限性，未过度主张因果关系

### 局限性

1. **评估范围受限**: 仅在 LIBERO 系列 benchmark 上验证，未测试真实机器人或跨形态泛化
2. **机器人形态鲁棒性弱于 Fast-WAM**: LIBERO-Plus 机器人扰动类别 37.4%，低于 [[Fast-WAM]] 的 44.5%
3. **无代码/模型开放**: 可复现性受限，无法独立验证
4. **消融实验局限**: 仅展示了互补性，未能通过控制变量验证每个组件的因果必要性

### 潜在改进方向

1. 在真实机器人系统（特别是不同形态）上验证，解决跨形态泛化问题
2. 探索多步迭代 Future-KV 更新（当前为单次预填充）以进一步提升预测质量
3. 将 LaWM 教师替换为更强的潜在动作模型，或尝试无监督 dynamics register 训练

### 可复现性评估

- [ ] 代码开源（暂未开放）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（论文中有充分描述）
- [x] 数据集可获取（LIBERO 为公开数据集）

---

## 关联笔记

### 基于

- [[Wan2.1]]: 视频生成主干网络，提供视频 DiT 预训练权重
- [[Flow-Matching|Flow Matching]]: 用于视频和动作生成的训练目标框架
- [[LaWAM]]: 提供潜在动作目标的冻结 LaWM 教师模型

### 对比

- [[Fast-WAM]]: 6B 参数直接策略 WAM，ForeWAM 主要对比基线
- [[Flash-WAM]]: 基于一致性蒸馏的推理加速 WAM，另一种效率优化方向
- [[pi0]]: 具身预训练 VLA，LIBERO 对比基线
- [[Pi0.5]]: 强 VLA 基线，ForeWAM-Flash 在 LIBERO 上与之持平
- [[UniVLA]]: 7B 具身预训练 VLA
- [[WorldVLA]]: 7B 世界模型 VLA

### 方法相关

- [[Future-KV]]: 本文核心机制，通过视频 DiT 单次预填充缓存预测性 KV 状态
- [[Dynamics Registers]]: Latent-Action 监督的紧凑动力学寄存器
- [[KV Cache]]: Future-KV Interface 的底层机制
- [[Register Token]]: 动力学寄存器的设计灵感来源
- [[ActionDiT]]: 动作去噪网络的架构类型
- [[Latent-Foresight]]: 相关但不同的潜在预见方法（ForeWAM 更隐式，无需显式预测未来状态）
- [[Knowledge Distillation]]: Latent-Action 监督的蒸馏机制
- [[Isolated Attention Mask]]: 四类 token 结构化注意力路由

### 数据集相关

- [[LIBERO]]: 主要训练和测试 benchmark（130任务）
- [[LIBERO-Plus]]: OOD 鲁棒性测试 benchmark，ForeWAM 关键优势所在

---

## 速查卡片

> [!summary] Foresight Without Seeing: ForeWAM
> - **核心**: 单次视频 KV 预填充将隐式未来动力学注入直接策略 WAM，无需显式视频生成
> - **方法**: Future-KV Interface（KV 预填充缓存）+ Latent-Action-Supervised Dynamics Registers（知识蒸馏）
> - **结果**: LIBERO 96.9%（2B，无具身预训练）；LIBERO-Plus 61.6%（vs Fast-WAM 51.5%，+10.1）；Flash 变体 220ms 延迟
> - **代码**: 暂未开放

---

*笔记创建时间: 2026-08-14*

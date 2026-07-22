---
title: "FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation"
method_name: "FM-VLA"
authors: [Ruicheng Li, Qixiu Li, Ruichun Ma, Yu Deng, Lin Luo, Zhiying Du, Jianfeng Xiang, Huizhi Liang, Ruicheng Wang, Jiaolong Yang, Baining Guo]
year: 2026
venue: arXiv
tags: [vla, force-sensing, contact-rich-manipulation, memory-augmented, flow-matching, proprioception]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.18231
created: 2026-07-22
---

# 论文笔记：FM-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft Research Asia（推测，含 Baining Guo / Jiaolong Yang） |
| 日期 | July 2026 |
| 项目主页 | [FM-VLA-Page](https://qft-333.github.io/FM-VLA-Page/) |
| 对比基线 | [[Pi0.5]] / [[ForceVLA]] / TA-VLA / π-MEM |
| 链接 | [arXiv](https://arxiv.org/abs/2607.18231) |

---

## 一句话总结

> FM-VLA 将 6 轴力/力矩历史通过预训练 [[VAE]] 压缩为 K=8 个力觉记忆 token，作为 [[Pi0.5]] 流匹配动作专家的条件输入，使机器人在 contact-rich 非马尔可夫操作任务上成功率突破 83%，推理开销仅增加 3.3ms。

---

## 核心贡献

1. **[[Force Memory Token|力觉记忆 Token]]**: 用预训练 [[VAE]]（[[Perceiver-IO]] 编/解码器）将全集 F/T 历史 $\{f_\tau\}_{\tau=1}^t$ 压缩为 $K=8$ 个紧凑隐变量 token，携带长时间接触事件的宏观时序结构
2. **双流记忆架构**: 长时力觉记忆（全局接触历史）+ 短时本体记忆（最近 $W=0.9\text{s}$ 关节位置窗口）并行注入 [[Rectified Flow]] 动作专家，互补接触感知与运动上下文
3. **零额外推理开销设计**: 力编码器冻结推理，仅增加 3.3ms（vs. 视觉记忆方案 +39~129ms），[[VAE]] 解耦预训练策略保证泛化

---

## 问题背景

### 要解决的问题

接触丰富的机器人操作任务（如"按按钮 N 次"、"擦碗 N 圈"）具有 **非马尔可夫性**（Non-Markovian）：当前成功状态在视觉上与失败状态无法区分，机器人必须记忆已发生的接触事件才能正确行动。

### 现有方法的局限

- **视觉记忆 VLA**（MemoryVLA、π-MEM）：依赖视觉 token 历史，在接触模糊的场景下（如相同外观的按钮被按过/未按过）失效
- **短时力觉条件化**（[[ForceVLA]]、TA-VLA）：仅捕捉即时接触力，不能累积时间序列的接触事件计数信息
- **序列模型**（GRU、Transformer）：长时力序列存在梯度消失（GRU）或瞬时峰值过拟合（Q-Former）问题

### 本文的动机

力/力矩（wrench）信号在接触事件发生时产生明确峰值，天然是接触计数的可靠来源。将全局 wrench 历史通过 [[VAE]] 压缩为固定数量的隐向量 token，可以：1）捕获宏观时序结构而非逐帧细节；2）与 [[Pi0.5]] 基座的 token 接口无缝兼容；3）冻结编码器后推理代价极低。

---

## 方法详解

### 模型架构

FM-VLA 采用 **双阶段训练 + 双流记忆** 架构，以 [[Pi0.5]]（[[PaliGemma 2]] VLM + [[Rectified Flow]] 动作专家）为基座：

- **输入**: 语言指令 $l$ + 三路 RGB 图像 $o_t$（头部 + 双腕）+ 6 轴 F/T 历史 $\{\hat{f}_\tau\}_{\tau=1}^t$ + 短窗口关节状态 $\{s_\tau\}_{\tau=t-W+1}^{t}$
- **Backbone**: [[PaliGemma 2]]（[[SigLIP]] 视觉编码器）+ 流匹配动作网络
- **核心模块**: [[Force Memory Token]] 编码器（冻结 [[VAE]]）+ 短时状态 MLP 投影
- **输出**: 动作块 $a_{t:t+k}$（7-DoF 双臂关节位置）
- **总参数**: ~3B（视觉语言主干）+ ~12M（力觉 VAE 编码器）

### 核心模块

#### 模块 1: 力觉记忆编码（Force VAE）

**设计动机**: 利用 [[Perceiver-IO]] 的 cross-attention 机制将不定长 F/T 时间序列映射到固定尺寸隐向量集合，使用 [[Masked ELBO]] + [[Free-Bits Regularization]] 避免后验坍塌

**具体实现**:
- **输入预处理**: 6 轴 wrench 信号经 [[EMA Smoothing|一阶指数平滑]]（$\alpha=0.3$）降噪，再经 [[Quantile Normalization]] 消除量纲差异，加入 [[Fourier Positional Encoding]] 编码时间位置
- **编码器**: [[Perceiver-IO]] 2 层 cross-attention + 10 层 self-attention，隐藏维度 384，输出 $K \times 96$ 隐向量（$K=8$）
- **解码器**: 对称 [[Perceiver-IO]] 结构，用于训练阶段重建 wrench 序列（推理时弃用）
- **训练正则**: Per-dimension free-bits，阈值 $C=1.0$，防止每个隐维度 KL 过低

#### 模块 2: 短时本体记忆（State Memory）

**设计动机**: 力觉记忆捕获接触计数等宏观历史，而运动执行需要近期关节轨迹信息（如运动方向、速度），两者互补

**具体实现**:
- 取最近 $W=0.9\text{s}$ 的关节位置序列 $\{s_\tau\}_{\tau=t-W+1}^{t}$，经线性投影为单个状态记忆 token
- 与 $K=8$ 个力觉 token 拼接，共 $K+1=9$ 个额外条件 token 注入动作专家
- 注入层使用 **零初始化线性投影**（Zero-init projection），保证训练初期不破坏基座分布

#### 模块 3: 随机噪声预填充（Temporal Anti-Shortcut）

**设计动机**: 避免模型从 F/T 序列长度推断当前时刻（利用"前 N 步为零"这一捷径），而非真正理解接触历史

**具体实现**: 训练时在 wrench 历史前随机预填充最多 10s 的零噪声段，迫使模型从信号内容而非时间位置判断状态

---

## 关键公式

### 公式 1: [[EMA Smoothing|一阶指数平滑]]

$$
\hat{f}_t = \alpha \cdot f_t + (1 - \alpha) \cdot \hat{f}_{t-1}
$$

**含义**: 对原始 6 轴 wrench 信号 $f_t$ 进行低通滤波，平滑传感器噪声

**符号说明**:
- $f_t \in \mathbb{R}^6$: 第 $t$ 步原始 wrench 读数（力 $F_{x,y,z}$ + 力矩 $M_{x,y,z}$）
- $\hat{f}_t$: 平滑后的 wrench 值
- $\alpha = 0.3$: EMA 平滑系数（越小越平滑）

### 公式 2: [[Masked ELBO|带掩码的 ELBO 目标]]

$$
\begin{aligned}
\mathcal{L}_{VAE} = &-\mathbb{E}_{q_\phi(\mathbf{z}|\hat{\mathbf{f}}_{1:T})} \left[\sum_{t \in \mathcal{M}} \log p_\theta(\hat{f}_t | \mathbf{z})\right] \\
&+ \sum_{d=1}^{D} \max\left(C,\ D_{\mathrm{KL}}\bigl(q_\phi(z_d|\hat{\mathbf{f}}_{1:T}) \,\|\, \mathcal{N}(0, 1)\bigr)\right)
\end{aligned}
$$

**含义**: Force-VAE 的训练目标——重建损失仅计算被掩码子集 $\mathcal{M}$ 的帧，KL 正则化使用 per-dimension free-bits 防止后验坍塌

**符号说明**:
- $\hat{\mathbf{f}}_{1:T} \in \mathbb{R}^{T \times 6}$: 完整平滑 wrench 历史序列（全集）
- $\mathbf{z} \in \mathbb{R}^{K \times 96}$: 编码器输出的 $K=8$ 个隐变量向量
- $q_\phi$: Perceiver-IO 编码器（变分后验）
- $p_\theta$: Perceiver-IO 解码器（条件似然）
- $\mathcal{M}$: 随机掩码帧子集（类似 MAE 掩码策略）
- $C = 1.0$: free-bits 阈值（nat），确保每维 KL $\geq C$
- $D$: 隐变量总维度（$K \times 96 = 768$）

### 公式 3: [[Force Memory Token|力觉 Token 生成]]

$$
\mathbf{T}_f = W_{\text{proj}} \cdot q_\phi(\hat{\mathbf{f}}_{1:t}) \in \mathbb{R}^{K \times d_{\text{model}}}
$$

**含义**: 将 VAE 编码器输出的隐向量通过零初始化 MLP 线性投影到动作专家的 token 空间维度

**符号说明**:
- $q_\phi(\hat{\mathbf{f}}_{1:t}) \in \mathbb{R}^{K \times 96}$: VAE 后验均值（推理时取均值，不采样）
- $W_{\text{proj}}$: 零初始化投影矩阵（$96 \to d_{\text{model}}$），保护基座预训练分布
- $K = 8$: 力觉 token 数量（由 token 数消融实验确定）

### 公式 4: [[Rectified Flow|整流流匹配]]训练目标（FM-VLA 微调阶段）

$$
\mathcal{L}_{flow} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, \mathbf{a}_0 \sim \mathcal{N}(0,I),\, \mathbf{a}_1} \left[\bigl\|v_\theta(\mathbf{a}_\tau,\, \tau,\, \mathbf{c}) - (\mathbf{a}_1 - \mathbf{a}_0)\bigr\|^2\right]
$$

其中插值轨迹：

$$
\mathbf{a}_\tau = (1 - \tau)\,\mathbf{a}_0 + \tau\,\mathbf{a}_1
$$

**含义**: 学习速度场 $v_\theta$ 使动作噪声 $\mathbf{a}_0$ 沿直线流到目标动作 $\mathbf{a}_1$，条件 $\mathbf{c}$ 包含视觉 token、语言 token 和新增的力觉记忆 token $\mathbf{T}_f$

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流时间步（连续）
- $\mathbf{a}_0 \sim \mathcal{N}(0, I)$: 噪声动作（流的起点）
- $\mathbf{a}_1$: 示教动作（流的终点）
- $\mathbf{c}$: 条件 token 集合，新增 $\mathbf{T}_f$（$K$ 个力觉 token）+ 状态记忆 token
- $v_\theta$: 动作专家速度场网络（[[Pi0.5]] 架构，参数微调）

### 公式 5: 短时状态记忆 Token

$$
\mathbf{t}_s = W_s \cdot \text{concat}(s_{t-W+1}, \ldots, s_t) \in \mathbb{R}^{d_{\text{model}}}
$$

**含义**: 将最近 $W$ 步关节状态拼接后线性投影为单个状态记忆 token

**符号说明**:
- $s_\tau \in \mathbb{R}^{d_s}$: 第 $\tau$ 步关节位置向量（7-DoF × 2 臂）
- $W$: 短时窗口长度（对应 0.9s 历史）
- $W_s$: 线性投影矩阵（零初始化）

---

## 关键图表

### Figure 1: Motivation / 视觉记忆 vs 力觉记忆对比

![Figure 1](https://arxiv.org/html/2607.18231v1/x1.png)

**说明**: 左侧展示纯视觉记忆 VLA 在"按按钮 N 次"任务中失败的根本原因——视觉无法区分"已按 1 次"和"已按 2 次"；右侧展示 FM-VLA 通过力觉历史 token（接触峰值序列）正确计数并完成任务。接触事件在力信号上产生明确峰值，而视觉外观完全相同。

### Figure 2: Architecture Overview / FM-VLA 系统架构

![Figure 2](https://arxiv.org/html/2607.18231v1/x2.png)

**说明**: 完整系统管线。左：[[VAE]] 编码器将全局 wrench 历史 $\{f_\tau\}_{\tau=1}^t$ 压缩为 $K=8$ 个力觉 token；右：短时状态历史 $\{s_\tau\}$ 投影为 1 个状态 token。两类 token 与视觉/语言 token 拼接后共同条件化 [[Rectified Flow]] 动作专家，生成动作块。注意 VAE 编码器在推理时完全冻结。

### Figure 3: Token Count Ablation / Token 数量消融

![Figure 3](https://arxiv.org/html/2607.18231v1/x3.png)

**说明**: $K=4$ 时出现信息瓶颈（平均成功率下降 ~15%）；$K=8$ 时达到性能峰值；$K=16,32$ 时引入分布偏移（超出 [[Pi0.5]] 基座见过的 token 范围），性能反而下降。

### Figure 4: Force Signal Visualization / 轨迹与力信号可视化

![Figure 4](https://arxiv.org/html/2607.18231v1/x4.png)

**说明**: 三个任务的推理轨迹与选定力通道的时序曲线，每帧时刻标注在力图上。FM-VLA 能正确识别力峰值事件（每次接触产生明确的力信号峰）并在记忆中累积，从而在第 N 次接触后停止，完成准确的计数操作。

---

## 实验

### 数据集

| 任务 | 演示数量 | 描述 | 记忆类型 |
|------|---------|------|---------|
| Find Block Under Cups | 200 | 依次揭开两个相同杯子找隐藏积木 | 空间记忆（已搜索过哪个位置）|
| Press Button N Times | 350 | 按按钮 N∈{1,2,3} 次后停止 | 接触计数记忆 |
| Wipe Bowl N Rounds | 200 | 用海绵擦碗 N∈{1,2,3} 圈 | 接触轨迹记忆 |

所有数据使用人机遥操作采集，部署在 AgiBot G1 双臂人形机器人上。

### 实现细节

- **Backbone**: [[Pi0.5]]（[[PaliGemma 2]] + [[SigLIP]]，3B 参数）
- **力觉 VAE 预训练**: 100,000 步，batch size 512，峰值 LR $3\times10^{-4}$，余弦衰减
- **VLA 微调（Stage 2）**: 50,000 步，batch size 32，峰值 LR $5\times10^{-5}$，VAE 编码器冻结
- **优化器**: AdamW（两阶段）
- **硬件**: 8× A100 (40GB)，bfloat16 精度，梯度裁剪 1.0
- **传感器采样率**: F/T 传感器 100Hz，三路 RGB 相机（头部 + 双腕）

### 主要结果

### Table 1: 方法对比 — 各任务成功率（%）

| Method | Task 1: Cups | Task 2: Buttons | Task 3: Wipe | **Average** |
|--------|:------------:|:---------------:|:------------:|:-----------:|
| [[Pi0.5]]（baseline）| 72.2 | 11.1 | 0.0 | 27.8 |
| TA-VLA | 50.0 | 11.1 | 5.6 | 22.2 |
| π-MEM（视觉记忆 K=5）| 77.8 | 33.3 | 50.0 | 53.7 |
| **FM-VLA（VAE，ours）** | **100.0** | **72.2** | **77.8** | **83.3** |

**关键发现**: FM-VLA 在所有任务上均显著超越 baseline，平均提升 55.5pp（vs. π0.5）和 29.6pp（vs. 最佳对比视觉记忆方法 π-MEM）。Task 1 实现 100% 成功率，Task 3（擦碗）相较 π0.5 从 0% 提升至 77.8%。

### Table 2: 消融实验 — 模态分析

| 配置 | Task 1 | Task 2 | Task 3 | Average |
|------|:------:|:------:|:------:|:-------:|
| Force-only（无 State 记忆）| 38.9 | 33.3 | 5.6 | 25.9 |
| State-only（无 Force 记忆）| 55.6 | 38.9 | 27.8 | 40.7 |
| **Force + State（FM-VLA）** | **100.0** | **72.2** | **77.8** | **83.3** |

**关键发现**: Force 和 State 两类记忆互补，缺一不可。Force-only 在 Task 3（需要精细运动控制）上失败；State-only 在 Task 2/3（需要接触计数）上失败；两者结合才达到最优。

### Table 3: 消融实验 — 力编码器架构对比

| 编码器架构 | Task 1 | Task 2 | Task 3 | Average |
|---------|:------:|:------:|:------:|:-------:|
| GRU | 55.6 | 22.2 | 22.2 | 33.3 |
| Q-Former | 72.2 | 55.6 | 44.4 | 57.4 |
| **VAE（ours）** | **100.0** | **72.2** | **77.8** | **83.3** |

**关键发现**: GRU 在长时序列上梯度消失；Q-Former 过拟合于瞬时接触峰值而非宏观时序结构；VAE 通过压缩瓶颈自然学习宏观接触模式，表现最优。

### Table 4: 推理效率对比

| Method | 推理延迟（ms）| 额外开销（ms）|
|--------|:-----------:|:-----------:|
| [[Pi0.5]] baseline | 60.7 | — |
| π-MEM（K=5，视觉）| 99.8 | +39.1 |
| π-MEM（K=16，视觉）| 190.0 | +129.3 |
| **FM-VLA（VAE）** | **64.0** | **+3.3** |

**关键发现**: 力觉 token 编码开销仅 3.3ms（VAE 编码器极轻量），远低于视觉记忆方案（+39~129ms），可满足实时控制需求。

---

## 批判性思考

### 优点

1. **工程优雅**: VAE 预训练解耦 + 编码器冻结，最小侵入性改造 [[Pi0.5]] 基座，几乎不增加推理开销
2. **信号互补性利用**: 力觉记忆（长时接触事件）+ 短时状态记忆（运动上下文）的双流设计有很强的设计动机，消融实验也很好地验证了
3. **任务覆盖有代表性**: 三个任务分别考察空间记忆、计数记忆、轨迹记忆，全面评估了非马尔可夫记忆能力

### 局限性

1. **固定 token 瓶颈**: $K=8$ 固定容量在需要记忆数百次接触事件的超长时域任务（如"按 10 次"）中可能不够，但此类任务未在实验中出现
2. **单平台验证**: 仅在 AgiBot G1 上评测，灵巧手（如 DEXHAND）等不同末端执行器、不同 F/T 传感器配置的泛化性未验证
3. **VAE 领域泛化**: VAE 在演示数据集上预训练，新任务需要重新采集演示才能重训 VAE，增加了部署门槛
4. **任务规模较小**: 每个任务 200~350 次演示，数据效率验证不够充分；任务种类只有 3 个，实验规模偏小

### 潜在改进方向

1. 更大规模的跨机器人 wrench 预训练（类似视觉基座的预训练范式），提升 VAE 泛化能力
2. 动态 token 数量机制，根据任务复杂度自适应分配力觉记忆容量
3. 结合触觉阵列传感器（tactile array）扩展到更细粒度的接触空间感知

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数充分）
- [ ] 数据集可获取（私有遥操作数据）

---

## 关联笔记

### 基于

- [[Pi0.5]]: 基座 VLA 模型（PaliGemma 2 + 流匹配动作专家）
- [[VAE]]: 力觉序列压缩的核心生成模型
- [[Rectified Flow]]: 动作生成的流匹配范式
- [[Perceiver-IO]]: 处理不定长 F/T 时间序列的跨注意力架构

### 对比

- [[ForceVLA]]: 同样使用力觉信号，但仅作短时条件化，无长时记忆
- π-MEM（视觉记忆）: 视觉历史 token 方案，推理开销大且接触感知弱

### 方法相关

- [[Force Memory Token]]: 本文提出的核心概念
- [[Masked ELBO]]: VAE 训练中的掩码重建目标
- [[Free-Bits Regularization]]: 防止 VAE 后验坍塌的 per-dimension KL 下界
- [[EMA Smoothing]]: 力信号预处理
- [[Quantile Normalization]]: 跨维度幅值归一化

### 硬件/数据相关

- AgiBot G1: 双臂人形机器人平台，7-DoF × 2 + 6 轴 F/T 腕部传感器（100Hz）

---

## 速查卡片

> [!summary] FM-VLA (2026)
> - **核心**: 用预训练 VAE 将全局 wrench 历史压缩为 K=8 力觉记忆 token，注入 π0.5 流匹配动作专家
> - **方法**: Force-VAE（Perceiver-IO，Masked ELBO + Free-Bits）+ 短时状态 token，双流记忆，Stage-2 冻结编码器微调
> - **结果**: 接触丰富操作任务平均 83.3% 成功率（vs. π0.5 的 27.8%），推理仅增加 3.3ms
> - **代码**: 未开源（项目主页：https://qft-333.github.io/FM-VLA-Page/）

---

*笔记创建时间: 2026-07-22*

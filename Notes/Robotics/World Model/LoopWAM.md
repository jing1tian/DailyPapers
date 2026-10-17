---
title: "LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts"
method_name: "LoopWAM"
authors: [Wenxuan Guo, Yingping Liang, Zhaoguo Wang, Ziyuan Li, Angyuan Ma, Zhenyu Wu, Chengkun Wang, Bingyao Yu, Ze Chen, Borui Zhang, Xiuwei Xu, Meng Zhang, Erjin Zhou, Jiwen Lu]
year: 2026
venue: arXiv
tags: [world-action-model, closed-loop-learning, flow-matching, imitation-learning, generalist-robot-policy, quality-weighted-learning, diffusion-transformer]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.18873v2
created: 2026-10-17
---

# 论文笔记：LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, Dexmal, Loopeva |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[ABot-M0.6]], [[LingBot-VLA-2.0]], [[pi0.5\|π0.5]], [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.18873) |

---

## 一句话总结

> LoopWAM 是一个闭环 [[World Action Model|世界动作模型]]，通过质量感知的噪声分离策略，从部署滚出的成功、次优和失败轨迹中均衡学习，持续自我迭代提升。

---

## 核心贡献

1. **闭环部署-训练循环**: 将机器人部署的所有质量级别轨迹（成功、次优、失败）纳入训练，实现持续自我改进，在 RoboChallenge Table30-V2 真实机器人评测中经三轮迭代从 23.67% 提升至 43.33%。
2. **源相关噪声采样（Source-Dependent Noise Sampling）**: 对专家段和 rollout 段施加不对称的 [[Beta Distribution|Beta 分布]]噪声——专家段视频低噪（Beta(7,1)）、动作高噪，rollout 段视频高噪、动作低噪——使模型既能从专家示范学习动作，又能从低质量轨迹的视频结果中学到因果关系，而不模仿其动作。
3. **质量引导推理（Quality Steering）**: 将质量分数 $q$ 作为文本后缀附加到指令中，推理时强制使用专家质量分 $q^\star$，驱动模型生成专家级别行为。
4. **质量加权动作监督**: 动作损失按质量分数 $\lambda(q)$ 加权，失败轨迹 $\lambda=0$ 从而零动作梯度，视频损失仍对所有轨迹计算，保留因果世界知识。

---

## 问题背景

### 要解决的问题

现有机器人泛化策略在部署时会产生大量混合质量的 rollout 数据（成功、次优、失败），如何从中学习而不损害策略质量？

### 现有方法的局限

1. **直接模仿所有 rollout**：会强化次优或错误行为，降低策略质量。
2. **丢弃失败 rollout**：浪费了包含物理因果关系（如为何失败、错误的物理后果）的宝贵信息。
3. **仅从专家示范学习**：数据量有限，无法利用自部署产生的大量交互经验。

### 本文的动机

[[World Action Model|世界动作模型]] 同时建模视频（世界状态）和动作的联合分布，天然区分"看到了什么"和"做了什么"。通过在视频和动作两路上施加不同噪声分布，可以让低质量轨迹的视频路提供世界后果监督，同时避免动作路学习错误行为。

---

## 方法详解

### 模型架构

LoopWAM 采用 **[[Mixture-of-Transformers|MoT]] [[Diffusion Transformer|双流扩散 Transformer]]** 架构，从 [[Wan2.2|Wan2.2-TI2V-5B]] 初始化：

- **输入**: 语言指令 $\ell$ + 当前观测图像 $o_t$（多摄像头拼合）+ 机器人状态 $s_t$ + 质量分数 $q$
- **Video Backbone**: 来自 Wan2.2-TI2V-5B 的视频 DiT 和 VAE
- **双流结构**: 视频流（$V$）和动作流（$A$）独立参数、独立噪声级别，通过联合注意力交互
- **核心模块**: [[Flow Matching|流匹配]] 联合去噪未来视频 latent $Z_t$ 和动作块 $A_t$
- **输出**: [[Action Chunking|动作块]] $A_t$（64步动作序列）+ 预测未来视频帧 $Z_t$
- **总参数**: ~5B（继承自 Wan2.2）

### 核心模块

#### 模块1：联合世界-动作建模（Joint World-Action Modeling）

**设计动机**: 利用 [[Video Prediction|视频预测]] 提供的物理因果知识辅助动作策略学习，通过[[Bidirectional Attention|双向注意力]]让预测的未来视频状态反哺动作生成。

**注意力机制**（三块：观测 $O$，视频 $V$，动作 $A$）:
- 观测 tokens 仅在初始帧内自注意
- 视频和动作 tokens 互相全注意（$V \leftrightarrow A$ 双向）
- 对应注意力掩码矩阵：

$$
M_{(O,V,A)} = \begin{pmatrix}1&0&0\\1&1&1\\1&1&1\end{pmatrix}
$$

**含义**: 行/列对应 O、V、A 三组 token 之间的注意力是否开启；V 和 A 互相全连接，O 只接收自身信息。

#### 模块2：源相关噪声采样（Source-Dependent Noise Sampling）

**设计动机**: 让低质量 rollout 数据的视频路有低噪声（易学习视频后果），动作路有高噪声（抑制对低质量动作的模仿）。

**具体实现**：
- 每段数据有来源标签 $d \in \{E(\text{专家}), R(\text{rollout})\}$
- 专家段（$d=E$）：$\sigma_v \sim \operatorname{Beta}(7,1)$，$\sigma_a \sim \mathcal{P}_0$
- Rollout 段（$d=R$）：$\sigma_v \sim \mathcal{P}_0$，$\sigma_a \sim \operatorname{Beta}(1,7)$
- 基础分布 $\mathcal{P}_0$ 使用 $\kappa=5$ 的翘曲函数

直觉：Beta(7,1) 集中在接近 1（高噪声），Beta(1,7) 集中在接近 0（低噪声），$\mathcal{P}_0$ 为均匀基础分布的翘曲版。

#### 模块3：质量引导推理（Quality Steering）

**设计动机**: 通过文本条件控制生成质量，无需改变模型结构即可在推理时强制专家级行为。

**具体实现**:
- 训练：指令 $\ell$ 后附加质量分 $q$ 的文本描述
- 推理：固定使用专家质量分 $q^\star$（如 coarse-level 中 $q^\star=5$）
- 质量分定义（coarse-level 1-5 scale）：
  - $q^\star=5$：专家示范
  - $q=4$：LoopWAM 自生成 rollout
  - $q=3$：π0.5 / X-VLA rollout
  - $q=2$：RLDX-1 rollout（RoboCasa365）
  - $q=1$：失败轨迹

---

## 关键公式

### 公式1：[[Flow Matching|流匹配]] 联合分布建模

$$
p_{\theta}(A_t, Z_t \mid o_t, s_t, \ell, q)
$$

**含义**: LoopWAM 对动作块 $A_t$ 和未来视频 latent $Z_t$ 的联合条件分布进行流匹配建模，条件为当前观测 $o_t$、状态 $s_t$、指令 $\ell$ 和质量分 $q$。

**符号说明**:
- $A_t$: 动作块（64步动作序列）
- $Z_t$: 未来视频帧的 VAE latent
- $o_t$: 当前 RGB 观测（多摄像头拼合）
- $s_t$: 机器人关节状态
- $\ell$: 语言指令（含质量分后缀）
- $q$: 质量分数（1–5 coarse 或 1–12 episode 级别）

---

### 公式2：[[Flow Matching|流场]] 前向加噪过程

$$
X_m^{\sigma_m} = (1-\sigma_m)X_m + \sigma_m \epsilon_m, \quad \epsilon_m \sim \mathcal{N}(0,I), \quad m \in \{v, a\}
$$

**含义**: 对视频流 $v$ 和动作流 $a$ 分别进行独立加噪，噪声级别 $\sigma_m$ 由各自的源相关分布采样决定。

**符号说明**:
- $X_m$: 干净数据（$m=v$ 时为视频 latent，$m=a$ 时为动作序列）
- $\sigma_m \in [0,1]$: 对应流 $m$ 的噪声级别
- $\epsilon_m \sim \mathcal{N}(0,I)$: 标准高斯噪声
- $X_m^{\sigma_m}$: 加噪后的混合样本

---

### 公式3：基础噪声分布翘曲函数

$$
\sigma = \phi_\kappa(u) = \frac{\kappa u}{1+(\kappa-1)u}, \quad u \sim \mathcal{U}(0,1)
$$

**含义**: 将均匀分布 $u$ 翘曲为基础噪声分布 $\mathcal{P}_0$，$\kappa=5$ 时向较大 $\sigma$ 值偏移，强化中高噪声区域的训练。

**符号说明**:
- $\kappa$: 翘曲强度参数（论文使用 $\kappa=5$）
- $u \sim \mathcal{U}(0,1)$: 均匀分布采样
- $\phi_\kappa$: 单调递增翘曲函数

---

### 公式4：源相关噪声采样策略

$$
\begin{aligned}
d=E:&\quad \sigma_v \sim \operatorname{Beta}(7,1),\quad \sigma_a \sim \mathcal{P}_0 \\
d=R:&\quad \sigma_v \sim \mathcal{P}_0,\quad \sigma_a \sim \operatorname{Beta}(1,7)
\end{aligned}
$$

**含义**: 专家段（$E$）视频高噪/动作基础噪，rollout 段（$R$）视频基础噪/动作高噪，实现来源相关的差异化噪声分布。

**符号说明**:
- $d$: 数据来源标签，$E$ 为专家示范，$R$ 为部署 rollout
- $\operatorname{Beta}(7,1)$: 集中在接近 1 处（高噪声）
- $\operatorname{Beta}(1,7)$: 集中在接近 0 处（低噪声）
- $\mathcal{P}_0$: 由 $\phi_5$ 翘曲后的基础噪声分布

---

### 公式5：时间步加权函数

$$
g(\sigma) = \exp\!\left[-2\!\left(\sigma-\tfrac{1}{2}\right)^2\right] - e^{-1/2}
$$

$$
C = \frac{1}{T}\sum_{j=0}^{T-1} g\!\left(\phi_5\!\left(1-\frac{j}{T}\right)\right), \quad T=1000
$$

$$
\omega(\sigma) = \frac{g(\sigma)}{C + 10^{-10}}
$$

**含义**: $\omega(\sigma)$ 为对中间噪声级别（$\sigma \approx 0.5$）赋予最高权重的训练加权函数，$g(\sigma)$ 是以 0.5 为中心的高斯形状函数，$C$ 为归一化常数。

**符号说明**:
- $g(\sigma)$: 以 $\sigma=0.5$ 为峰值的钟形加权函数
- $C$: 在基础分布 $\mathcal{P}_0$ 上的归一化常数（$T=1000$ 点估计）
- $\omega(\sigma)$: 归一化后的训练权重

---

### 公式6：单样本流匹配损失

$$
\ell_{m,i} = \omega(\sigma_{m,i})\,\left\|v_{\theta,i}^m - (\epsilon_{m,i} - X_{m,i})\right\|_{\mathrm{valid}}^2
$$

**含义**: 对每个样本 $i$ 的流 $m$，用时间步加权 $\omega$ 监督模型预测速度场 $v_\theta^m$ 接近真实速度 $(\epsilon - X)$，仅在有效（非 padding）位置计算。

**符号说明**:
- $v_{\theta,i}^m$: 模型预测的速度场（去噪方向）
- $\epsilon_{m,i} - X_{m,i}$: 真实速度（从干净数据到噪声方向）
- $\|\cdot\|_{\mathrm{valid}}^2$: 对非 padding token 的均方误差
- $\omega(\sigma_{m,i})$: 时间步加权

---

### 公式7：LoopWAM 总训练损失

$$
\mathcal{L}_{\mathrm{LoopWAM}} = \frac{1}{B}\sum_{i=1}^{B}\ell_{v,i} + \frac{\sum_{i=1}^{B}\lambda(q_i)\,\ell_{a,i}}{\max\!\left(1,\sum_{i=1}^{B}\lambda(q_i)\right)}
$$

**含义**: 视频流损失对所有 $B$ 个样本均等计算（学习世界后果），动作流损失按质量权重 $\lambda(q)$ 加权归一化（避免学习低质量动作）。

**符号说明**:
- $B$: batch 大小
- $\ell_{v,i}$: 第 $i$ 样本的视频流损失
- $\ell_{a,i}$: 第 $i$ 样本的动作流损失
- $\lambda(q_i) \in [0,1]$: 质量加权函数，失败轨迹 $\lambda=0$，专家 $\lambda=1$
- $\max(1, \cdot)$: 防止除零的分母截断

---

### 公式8：推理时的策略边际分布

$$
\pi_\theta(A_t \mid o_t, s_t, \ell;\, q^\star) = \int p_\theta(A_t, Z_t \mid o_t, s_t, \ell, q^\star)\,\mathrm{d}Z_t
$$

**含义**: 推理时固定质量分为专家级 $q^\star$，对未来视频 latent $Z_t$ 边际化，得到纯动作策略。

**符号说明**:
- $q^\star$: 推理时使用的专家质量分（coarse-level 中 $q^\star=5$）
- $Z_t$: 被边际化掉的未来视频 latent（推理时不需要输出）

---

### 公式9：推理时的 Euler 步迭代

$$
X_m^{k+1} = X_m^k + (\sigma_{k+1} - \sigma_k)\,v_\theta^m, \quad \sigma_k = \phi_5\!\left(1-\frac{k}{K}\right)
$$

**含义**: 从纯噪声 $X_m^0 = \epsilon$ 出发，沿模型预测速度场做 $K$ 步 Euler 积分还原干净数据（动作 + 视频）。

**符号说明**:
- $k$: 推理步序号（$k=0,\ldots,K-1$）
- $K$: 总推理步数
- $\sigma_k$: 第 $k$ 步对应的噪声级别（由翘曲函数调度）
- $v_\theta^m$: 模型预测的速度场

---

## 关键图表

### Figure 1: Teaser / 核心动机与迭代改进

![Figure 1](https://arxiv.org/html/2610.18873v2/teaser.png)

**说明**: 左侧展示 LoopWAM 从混合质量部署 rollout 中学习的核心思路；右侧显示在 RoboChallenge Table30-V2 上三轮迭代的真实机器人成功率曲线，从初始 23.67% 持续提升至 43.33%。

---

### Figure 2: 方法总览 / LoopWAM 架构与训练流程

![Figure 2](https://arxiv.org/html/2610.18873v2/LoopWAM_method0926.png)

**说明**: 展示 [[Mixture-of-Transformers|MoT]] 双流 Transformer 架构，以及训练数据流（专家示范 + 部署 rollout）如何通过源相关噪声采样和质量加权损失整合。视频流和动作流通过联合注意力交互，共享 Wan2.2 初始化的视频骨干。

---

### Figure 3: 真实机器人闭环改进曲线

![Figure 3](https://arxiv.org/html/2610.18873v2/closed_loop_improvement.png)

**说明**: RoboChallenge Table30-V2 上跨三轮"部署→收集 rollout→再训练"循环的各子任务成功率变化，每轮均有明显提升，验证闭环学习的持续有效性。

---

### Figure 4: 子任务分割示例——按键（ARX5）

| 阶段 | 图片 |
|------|------|
| 初始状态 | ![press_init](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/press_the_button/init.png) |
| 按粉色按钮 | ![press_01](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/press_the_button/01_press_2.png) |
| 按蓝色按钮 | ![press_02](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/press_the_button/02_press_3.png) |
| 按绿色/黄色按钮 | ![press_03](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/press_the_button/03_press_4.png) |

**说明**: 使用 Qwen2.5-VL-7B 动态切换阶段特定提示，将长任务分解为 4 个子任务阶段分别执行。

---

### Figure 5: 子任务分割示例——浇花（ARX5）

| 阶段 | 图片 |
|------|------|
| 初始状态 | ![water_init](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/water_the_flowers/init.png) |
| 浇水后放回 | ![water_01](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/water_the_flowers/01_place.png) |

**说明**: 两阶段子任务：先取水壶浇水，再归位放下。

---

### Figure 6: 子任务分割示例——碎纸（UR5）

| 阶段 | 图片 |
|------|------|
| 初始状态 | ![shred_init](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/shred_paper/init.png) |
| 插入后收手 | ![shred_01](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/shred_paper/01_retract.png) |

**说明**: 两阶段：将纸插入碎纸机，再缩回夹爪避免被夹住。

---

### Figure 7: 子任务分割示例——盖章（ALOHA）

| 阶段 | 图片 |
|------|------|
| 初始状态 | ![stamp_init](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/stamp_positioning/init.png) |
| 蘸墨水 | ![stamp_01](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/stamp_positioning/01_dip.png) |
| 盖章并归位 | ![stamp_02](https://arxiv.org/html/2610.18873v2/figures/table30_subtask_keyframes_train/stamp_positioning/02_stamp.png) |

**说明**: 三阶段：取印章、蘸墨水、盖章归位，展示 ALOHA 双臂协作能力。

---

### Figure 8: RoboChallenge Table30-V2 排行榜

![Figure 8](https://arxiv.org/html/2610.18873v2/figures/table30v2-leaderboard.png)

**说明**: LoopWAM 以 56.80 总分和 43.33% 成功率位列排行榜第一，明显超出第二名 VLA-DM0.5（54.42 分、40.67%）。

---

### Figure 9: 失败恢复示例

![Figure 9](https://arxiv.org/html/2610.18873v2/retry-sucess.png)

**说明**: 展示 LoopWAM 在首次尝试失败后，通过迭代执行和子任务重试成功完成任务的定性案例，体现闭环学习赋予策略的恢复能力。

---

### Figure 10: 视频预测可视化

![Figure 10a](https://arxiv.org/html/2610.18873v2/video-pred-1.png)

![Figure 10b](https://arxiv.org/html/2610.18873v2/video-pred-2.png)

**说明**: 对比模型预测的未来视频帧与真实观测帧，预测能正确捕获场景演化趋势（物体运动、机器人位姿变化），但在细粒度视觉细节上存在差异。

---

### Figure 11: ARX5 代表性任务

![Figure 11a](https://arxiv.org/html/2610.18873v2/figures/lyp/arx-01.png)
![Figure 11b](https://arxiv.org/html/2610.18873v2/figures/lyp/arx-02.png)

**说明**: ARX5 单臂平台上的代表性操作任务定性结果展示。

---

### Figure 12: ALOHA 代表性任务

![Figure 12a](https://arxiv.org/html/2610.18873v2/figures/lyp/aloha-01.png)
![Figure 12b](https://arxiv.org/html/2610.18873v2/figures/lyp/aloha-02.png)
![Figure 12c](https://arxiv.org/html/2610.18873v2/figures/lyp/aloha-03.png)
![Figure 12d](https://arxiv.org/html/2610.18873v2/figures/lyp/aloha-04.png)

**说明**: ALOHA 双臂平台代表性任务，包括精细操作和双臂协作场景。

---

### Figure 13: UR5 代表性任务

![Figure 13a](https://arxiv.org/html/2610.18873v2/figures/lyp/ur5-01.png)
![Figure 13b](https://arxiv.org/html/2610.18873v2/figures/lyp/ur5-02.png)

**说明**: UR5 平台代表性操作任务定性结果。

---

### Figure 14: W1 长视程任务

![Figure 14a](https://arxiv.org/html/2610.18873v2/figures/lyp/w1-01.png)
![Figure 14b](https://arxiv.org/html/2610.18873v2/figures/lyp/w1-02.png)
![Figure 14c](https://arxiv.org/html/2610.18873v2/figures/lyp/w1-03.png)
![Figure 14d](https://arxiv.org/html/2610.18873v2/figures/lyp/w1-04.png)

**说明**: W1 平台上的长视程任务，需要多步操作和长期规划，展示 LoopWAM 对复杂序列任务的泛化能力。

---

### Table 1: RoboChallenge Table30-V2 排行榜（整体指标）

| Method | Overall Score | Overall SR (%) |
|--------|---------------|----------------|
| debuggers_vla | 18.72 | 11.00 |
| W0 | 21.85 | 12.00 |
| test_01 | 22.73 | 12.33 |
| holobrain | 27.93 | 14.33 |
| Baseline Model | 31.48 | 14.33 |
| MagicBot | 20.82 | 14.67 |
| mc | 23.63 | 15.33 |
| JV0 | 27.58 | 16.00 |
| my16 | 41.27 | 30.67 |
| VLA-DM0.5 | 54.42 | 40.67 |
| LoopWAM (expert-trained only) | 36.10 | 23.67 |
| **LoopWAM (after 3 rounds)** | **56.80** | **43.33** |

**表格说明**: LoopWAM 在加入闭环训练后大幅超越仅用专家数据的基础版本（23.67% → 43.33%），并以 56.80 分夺得排行榜第一。

---

### Table 2: RoboTwin 2.0（50 双臂任务，成功率 %）

| Method | Embodied PT. | Clean | Rand. | Average |
|--------|-------------|-------|-------|---------|
| π0 | ✓ | 65.9 | 58.4 | 62.2 |
| X-VLA | ✓ | 72.9 | 72.8 | 72.9 |
| π0.5 | ✓ | 82.7 | 76.8 | 79.8 |
| ABot-M0 | ✓ | 81.2 | 80.4 | 80.8 |
| Motus (from Wan2.2) | ✗ | 77.6 | 77.0 | 77.3 |
| Motus | ✓ | 88.7 | 87.0 | 87.8 |
| LingBot-VA | ✓ | 92.9 | 91.5 | 92.2 |
| Fast-WAM | ✗ | 91.9 | 91.8 | 91.8 |
| AHA-WAM | ✗ | 93.4 | 92.2 | 92.8 |
| LingBot-VA 2.0 | ✓ | 93.8 | 93.4 | 93.6 |
| **LoopWAM** | **✗** | **94.6** | **94.4** | **94.5** |

**表格说明**: LoopWAM 在**无需额外体现预训练（Embodied PT.）**的情况下达到 94.5% 平均成功率，超越所有需要体现预训练的方法，包括 [[LingBot-VLA-2.0]]（93.6%）。

---

### Table 3: RoboCasa365（50 任务，成功率 %）

| Method | Embodied PT. | Atomic Seen | Composite Seen | Composite Unseen | Overall |
|--------|-------------|-------------|----------------|------------------|---------|
| Diffusion Policy | ✗ | 15.7 | 0.2 | 1.3 | 6.1 |
| π0 | ✓ | 34.6 | 6.1 | 1.1 | 14.8 |
| π0.5 | ✓ | 39.6 | 7.1 | 1.2 | 16.9 |
| GigaWorld-Policy 0.1 | ✓ | 44.4 | 11.8 | 2.9 | 20.7 |
| GR00T N1.6 | ✓ | 51.1 | 9.4 | 1.7 | 21.9 |
| GR00T N1.5 | ✓ | 50.7 | 14.8 | 2.7 | 23.9 |
| RLDX-1 | ✓ | 67.6 | 27.9 | 8.5 | 36.0 |
| PRTS | ✓ | 66.3 | 30.3 | 18.8 | 39.6 |
| ABot-M0.5 | ✓ | 75.6 | 37.7 | 3.3 | 40.3 |
| ABot-M0.6 | ✓ | 79.4 | 48.3 | 7.9 | 46.6 |
| **LoopWAM** | **✗** | **79.7** | **50.3** | **14.9** | **49.5%** |

**表格说明**: LoopWAM 在无体现预训练情况下以 49.5% 整体成功率超越所有基线，尤其在 Composite Unseen（泛化）任务上以 14.9% 大幅超越 ABot-M0.6（7.9%）。

---

### Table 4: RoboTwin 2.0 消融实验（成功率 %）

| Quality Steering | Quality-Weighted Action | Clean | Rand. | Average |
|-----------------|------------------------|-------|-------|---------|
| 无 | ✗ | 73.5 | 71.1 | 72.3 |
| Extra-coarse (2分制) | ✗ | 92.2 | 93.3 | 92.8 |
| Extra-coarse (2分制) | ✓ | 93.9 | 93.3 | 93.6 |
| Coarse (5分制) | ✗ | 93.2 | 92.2 | 92.7 |
| **Coarse (5分制, ours)** | **✓** | **94.6** | **94.4** | **94.5** |

**关键发现**: 
1. 质量引导（Quality Steering）是最关键模块：移除后从 94.5% 骤降至 72.3%（-22.2%）。
2. 细粒度质量标签（coarse 5分制）优于粗粒度（extra-coarse 2分制）：92.8% → 94.5%。
3. 质量加权动作监督在 coarse 设置下带来 +1.8% 提升（92.7% → 94.5%），但在 extra-coarse 下效果有限（+0.8%）。

---

## 实验

### 数据集与评测平台

| 平台/数据集 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| RoboChallenge Table30-V2 | 30 任务，4 种机器人 | 真实机器人多任务评测，ARX5/ALOHA/UR5/W1 | 闭环改进主评测 |
| RoboTwin 2.0 | 50 双臂任务 | 仿真，Clean+Random 配置 | 泛化性评测 |
| RoboCasa365 | 50 任务 | 仿真，Atomic+Composite 任务类型 | 组合泛化评测 |
| 专家示范 | — | 多机器人操作演示 | 初始训练数据 |
| 自部署 rollout | 3 轮迭代 | 混合质量轨迹 | 闭环增量训练数据 |

### 实现细节

- **基础模型**: Wan2.2-TI2V-5B（视频流） + 动作 DiT（隐藏维度 1024）
- **动作表示**: 相对关节目标（相对于块起始状态）+ 绝对夹爪指令
- **动作视程**: 64 步动作块
- **视频段长度**: 每段 9 帧 RGB，每 8 个控制步采样 1 帧
- **状态归一化**: 以专家数据统计量归一化动作和状态
- **子任务切换**: Qwen2.5-VL-7B 驱动，每个任务配备阶段特定提示
- **质量标注**: Coarse 级别（1-5）；成功 rollout 按策略整体成功率及 0.7% 手动抽检打分

### 闭环训练细节（RoboChallenge Table30-V2）

- **Round 0**: 仅专家示范训练 → 23.67% SR
- **Round 1**: + 第1轮 rollout（含失败轨迹） → 提升
- **Round 2**: + 第2轮 rollout → 进一步提升
- **Round 3**: 最终 → 43.33% SR，总分 56.80（排行榜第一）

---

## 批判性思考

### 优点

1. **数据利用率高**: 将废弃的失败轨迹变为视频监督信号，显著提高数据效率
2. **无体现预训练竞争力强**: 在 RoboTwin 2.0 和 RoboCasa365 上无需昂贵的机器人特化预训练即超越所有基线
3. **架构简洁**: 质量分通过文本后缀注入，无需修改模型结构，工程实现简单
4. **跨机器人泛化**: 单模型覆盖 ARX5、ALOHA、UR5、W1 四种形体，验证了 WAM 框架的泛化性

### 局限性

1. **计算成本高**: 5B 参数模型 + 视频生成的推理延迟未在论文中详细报告，实时部署可能面临挑战
2. **质量标注依赖人工**: 当前 coarse 级别质量标注需手动抽检 0.7% 轨迹，规模化时成本上升
3. **无显式限制章节**: 论文未设专门的 Limitations 小节，潜在局限（如对新任务的冷启动问题）未被充分讨论
4. **真实机器人仅 30 任务**: RoboChallenge Table30-V2 的 30 任务评测规模相对有限，难以完全代表真实世界多样性

### 潜在改进方向

1. 自动化质量标注（使用 VLM 替代人工评分）
2. 蒸馏小模型以降低推理延迟
3. 探索在线（on-policy）连续学习而非离线迭代

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文描述较充分）
- [ ] 数据集可获取（依赖 Wan2.2 商业版权）

---

## 关联笔记

### 基于

- [[Wan2.2]]: 视频 backbone 初始化来源（Wan2.2-TI2V-5B）
- [[Flash-WAM]]: 同类 WAM 框架
- [[Fast-WAM]]: 对比基线，同为无体现预训练 WAM

### 对比

- [[LingBot-VLA-2.0]]: RoboTwin 2.0 上的最强有体现预训练对比方法
- [[ABot-M0.6]]: RoboCasa365 上的强基线
- [[pi0.5\|π0.5]]: 广泛使用的泛化 VLA 基线

### 方法相关

- [[World Action Model]]: 核心框架
- [[Flow Matching]]: 核心生成方法
- [[Mixture-of-Transformers]]: 双流架构设计
- [[Action Chunking]]: 动作表示
- [[Beta Distribution]]: 源相关噪声采样工具
- [[Closed-Loop Policy Learning]]: 核心学习范式

### 数据相关

- [[RoboTwin 2.0]]: 仿真评测 benchmark
- [[RoboCasa365]]: 组合泛化仿真 benchmark

---

## 速查卡片

> [!summary] LoopWAM
> - **核心**: 闭环 WAM，从混合质量 rollout 持续自我提升
> - **方法**: 源相关噪声采样 + 质量引导推理 + 质量加权动作损失
> - **结果**: Table30-V2 真实机器人 43.33%（+19.66% over 3 rounds），RoboTwin 2.0 94.5%，RoboCasa365 49.5%
> - **代码**: 未开源（截至 2026-10-17）

---

*笔记创建时间: 2026-10-17*

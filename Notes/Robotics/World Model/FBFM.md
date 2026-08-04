---
title: "FBFM: A Training-Free Asynchronous Feedback Mechanism for Flow-Matching in World-Action Models Execution"
method_name: "FBFM"
authors: [Peize Li, Ruimeng Zhang, Ru Zhang, Cong Huang, Kai Chen, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, flow-matching, inference-time-guidance, closed-loop-control, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.29235
created: 2026-08-04
---

# 论文笔记：FBFM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | (未明确列出) |
| 日期 | July 2026 |
| 项目主页 | 无 |
| 对比基线 | [[Real-Time Chunking]]、[[WAM]] Base |
| 链接 | [arXiv](https://arxiv.org/abs/2607.29235) |

---

## 一句话总结

> FBFM 是一种无需训练的推理时反馈机制，通过带掩码的伪逆引导将真实环境观测注入 [[Flow-Matching]] 的速度场，实现 [[WAM]] 在执行块内的异步闭环修正。

---

## 核心贡献

1. **块内反馈问题形式化**: 将 [[WAM]] 执行中的块内反馈建模为"动态部分可观测生成问题"，保留已返回状态观测的值和时间位置
2. **FBFM 机制**: 提出带掩码伪逆引导方法，结合动态潜状态反馈与固定跨块动作一致性约束，保持 WAM 冻结不训练
3. **双架构验证**: 分别在阶段式生成（LingBot-VA）和联合生成（DreamZero）两类 WAM 架构上实例化 FBFM，证明方法的通用性

---

## 问题背景

### 要解决的问题

[[WAM|世界-动作模型]] 在执行过程中以"块"（chunk）为单位生成预测，然后逐步执行。现有方法仅在相邻块边界处更新历史（KV cache 刷新），无法在**块内部**进行步级别的误差修正。长时序任务中，预测偏差会随执行积累，导致可靠性下降。

### 现有方法的局限

- **[[Real-Time Chunking]] (RTC)**: 只在块边界用前一块的执行结果约束后续生成的重叠动作，无法在当前生成块中途注入反馈
- **开环生成**: 整个块的状态/动作预测完全基于历史，不受执行过程中新观测影响
- **粗粒度反馈**: 现有异步方法仍以"块"为最小单位，而非单时间步

### 本文的动机

[[Flow-Matching]] 推理过程本身是连续迭代的 ODE 求解过程，天然存在多个 solver step。可以在每个 solver step 处注入来自环境的观测约束，通过向量-雅可比积（[[JVP|VJP]]）计算端点预测关于当前迭代点的梯度，进而修正速度场，从而实现无需重训练的块内闭环控制。

---

## 方法详解

### 问题设定

环境建模为受控马尔可夫过程 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P)$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P$ 为转移核。[[WAM]] 通过感知编码将物理状态映射为潜状态，并预测未来状态-动作块：

- 潜状态块 $\mathbf{Z}_t = [z_{t+1}^\top, \ldots, z_{t+H}^\top]^\top \in \mathbb{R}^{D_Z}$
- 动作块 $\mathbf{A}_t = [a_t^\top, \ldots, a_{t+H-1}^\top]^\top \in \mathbb{R}^{D_A}$
- 联合表示 $\mathbf{X}_t = [\mathbf{Z}_t; \mathbf{A}_t] \in \mathbb{R}^{D_X}$（$D_X = D_Z + D_A$）

### 核心机制：带掩码伪逆引导

FBFM 在 [[Flow-Matching]] 的每个 solver step $k$ 处修正速度场。设 $Q \in \{Z, A, X\}$ 为模态索引，对应潜状态、动作或联合表示流。

**核心公式**（统一引导场更新）将实时观测 $\mathbf{Y}_{t,k}^Q$ 注入到当前迭代点，通过端点估计的雅可比转置传播修正量。

#### 三要素

**动作重叠掩码 $\mathbf{W}_t^A$**（固定）: 将当前已执行动作的坐标置为 1，约束新生成的块与前一块的重叠段保持一致：

$$
\mathbf{W}_t^A = \operatorname{Diag}(\mathbf{1}[0 \in \mathcal{I}_t^A], \ldots, \mathbf{1}[H-1 \in \mathcal{I}_t^A]) \otimes \mathbf{I}_{d_a}
$$

**动态状态掩码 $\mathbf{W}_{t,k}^Z$**（随观测到达动态更新）: 哪些时刻的状态已有实测结果就激活对应条目：

$$
\mathbf{W}_{t,k}^Z = \operatorname{Diag}(w_{t,1}^{Z,k}, \ldots, w_{t,H}^{Z,k}) \otimes \mathbf{I}_{d_z}, \quad w_{t,i}^{Z,k} > 0 \iff (i, z_{t+i}) \in \mathcal{F}_{t,k}
$$

**模态预调节器 $\mathbf{P}^Q$**: 对不同模态（状态 vs. 动作）的残差进行缩放，处理量纲差异。

### 模型架构

[[FBFM]] 可以装载到两类 WAM 架构上，无需修改模型权重：

#### 阶段式 WAM（LingBot-VA）

$$
p_\theta(\mathbf{Z}_t, \mathbf{A}_t | \mathcal{H}_t) = p_{\theta_Z}(\mathbf{Z}_t | \mathcal{H}_t) \cdot p_{\theta_A}(\mathbf{A}_t | \mathbf{Z}_t, \mathcal{H}_t)
$$

- 先运行状态流 $p_{\theta_Z}$：用动态状态掩码注入实测观测，随着前一块执行逐步约束状态预测
- 后运行动作流 $p_{\theta_A}$：将修正后的状态上下文 $\mathbf{Z}_{t,k}$ 作为条件，同时固定重叠动作坐标
- 状态上下文刷新：

$$
\check{\mathbf{Z}}_{t,k} = (\mathbf{I}_{D_Z} - \mathbf{W}_{t,k}^Z)\hat{\mathbf{Z}}_t + \mathbf{W}_{t,k}^Z \mathbf{Y}_{t,k}^Z, \quad \mathcal{C}_{t,k}^Z = \Phi_Z(\check{\mathbf{Z}}_{t,k}, \mathcal{H}_t)
$$

#### 联合生成 WAM（DreamZero）

$$
p_\theta(\mathbf{X}_t | \mathcal{H}_t), \quad \mathbf{X}_t = [\mathbf{Z}_t; \mathbf{A}_t]
$$

- 联合块对角掩码：

$$
\mathbf{W}_{t,k}^X = \begin{bmatrix} \mathbf{W}_{t,k}^Z & \mathbf{0} \\ \mathbf{0} & \mathbf{W}_t^A \end{bmatrix}
$$

- 通过**跨模态雅可比块** $\mathbf{J}_{ZA}^\top = (\partial \hat{\mathbf{Z}}_{t,k}^1 / \partial \mathbf{A}_t^{\tau_k^X})^\top$，状态残差可直接修正动作坐标：

$$
\mathbf{g}_{t,k}^A = \mathbf{J}_{ZA}^\top \mathbf{e}_{t,k}^Z + \mathbf{J}_{AA}^\top \mathbf{e}_{t,k}^A, \quad \mathbf{e}_{t,k}^A = \mathbf{0} \implies \mathbf{g}_{t,k}^A = \Big(\frac{\partial \hat{\mathbf{Z}}_{t,k}^1}{\partial \mathbf{A}_t^{\tau_k^X}}\Big)^\top \mathbf{e}_{t,k}^Z
$$

---

## 关键公式

### 公式1: [[Controlled Markov Process|受控马尔可夫过程]]

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P)
$$

**含义**: 定义机器人操作环境的基本结构，$s_0 \sim \rho_T$，$s_{t+1} \sim P(\cdot | s_t, a_t)$

**符号说明**:
- $\mathcal{S}$: 物理状态空间
- $\mathcal{A}$: 动作空间
- $P$: 转移核

---

### 公式2: [[WAM|WAM 联合预测分布]]

$$
(\hat{\mathbf{Z}}_{t,H}, \hat{\mathbf{A}}_{t,H}) \sim p_\theta(\cdot | \mathcal{H}_t)
$$

**含义**: WAM 以交互历史 $\mathcal{H}_t$ 为条件，生成 horizon $H$ 内的状态和动作块

**符号说明**:
- $\hat{\mathbf{Z}}_{t,H} = (\hat{z}_{t+1}, \ldots, \hat{z}_{t+H})$: 预测状态序列
- $\hat{\mathbf{A}}_{t,H} = (\hat{a}_t, \ldots, \hat{a}_{t+H-1})$: 预测动作序列
- $\mathcal{H}_t = (c_T, z_0, a_0, \ldots, a_{t-1}, z_t)$: 完整交互历史

---

### 公式3: [[Flow-Matching|Flow Matching 训练目标]]

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}\Big[\big\|v_\theta(\mathbf{X}_t^\tau, \tau; \mathcal{H}_t) - (\mathbf{X}_t - \varepsilon)\big\|_2^2\Big]
$$

**含义**: 训练神经网络速度场 $v_\theta$，使其预测从噪声 $\varepsilon$ 到目标 $\mathbf{X}_t$ 的条件流

**符号说明**:
- $\mathbf{X}_t^\tau = (1-\tau)\varepsilon + \tau\mathbf{X}_t$: 线性插值路径（$\tau \in [0,1]$）
- $\varepsilon \sim \mathcal{N}(0, \mathbf{I})$: 源分布噪声
- $v_\theta$: 参数化速度场（条件于历史 $\mathcal{H}_t$）

---

### 公式4: [[Flow-Matching|Flow Matching ODE 积分]]

$$
\frac{d\mathbf{X}_t^\tau}{d\tau} = v_\theta(\mathbf{X}_t^\tau, \tau; \mathcal{H}_t), \quad \hat{\mathbf{X}}_t = \mathbf{X}_t^0 + \int_0^1 v_\theta(\mathbf{X}_t^\tau, \tau; \mathcal{H}_t)\,d\tau
$$

**含义**: 推理时通过 ODE 积分从纯噪声 $\mathbf{X}_t^0 = \varepsilon$ 求解到预测结果 $\hat{\mathbf{X}}_t$

**符号说明**:
- Forward Euler 离散化: $\mathbf{X}_t^{\tau_{k+1}} = \mathbf{X}_t^{\tau_k} + \Delta\tau_k \cdot v_\theta(\mathbf{X}_t^{\tau_k}, \tau_k; \mathcal{H}_t)$

---

### 公式5: [[Flow-Matching|噪声参数化端点估计]]

$$
\hat{\mathbf{X}}_t^1 = \mathbf{X}_t^\sigma - \sigma \cdot \tilde{v}_\theta(\mathbf{X}_t^\sigma, \sigma; \mathcal{H}_t)
$$

**含义**: 当采用噪声参数化时，从当前迭代点 $\mathbf{X}_t^\sigma$ 直接估计干净端点 $\hat{\mathbf{X}}_t^1$

**符号说明**:
- $\sigma$: 当前噪声水平
- $\tilde{v}_\theta$: 噪声预测网络（与速度场 $v_\theta$ 等价参数化）

---

### 公式6: [[Pseudoinverse Guidance|FBFM 核心引导场更新]]

$$
\begin{aligned}
v_{\text{FBFM}}^Q &= \bar{v}_{t,k}^Q + \lambda_{\tau_k^Q}^Q (\mathbf{J}_{t,k}^Q)^\top \mathbf{P}^Q \mathbf{W}_{t,k}^Q (\mathbf{Y}_{t,k}^Q - \hat{\mathbf{Q}}_{t,k}^1) \\
\hat{\mathbf{Q}}_{t,k}^1 &= \mathbf{Q}_t^{\tau_k^Q} + (1-\tau_k^Q)\bar{v}_{t,k}^Q \\
\mathbf{J}_{t,k}^Q &= \frac{\partial \hat{\mathbf{Q}}_{t,k}^1}{\partial \mathbf{Q}_t^{\tau_k^Q}}
\end{aligned}
$$

**含义**: 在每个 solver step $k$，用观测残差 $(\mathbf{Y} - \hat{\mathbf{Q}}^1)$ 通过端点雅可比 $\mathbf{J}^\top$ 修正速度场

**符号说明**:
- $Q \in \{Z, A, X\}$: 模态索引（状态/动作/联合）
- $\bar{v}_{t,k}^Q$: 原始（未引导）速度场
- $\mathbf{J}_{t,k}^Q$: 端点估计关于当前迭代点的雅可比矩阵
- $\mathbf{P}^Q$: 模态预调节器（量纲缩放）
- $\mathbf{W}_{t,k}^Q$: 支撑掩码（动态状态 + 固定动作重叠）
- $\mathbf{Y}_{t,k}^Q$: 观测目标（实测状态 + 前一块已执行动作）
- $\lambda_{\tau_k^Q}^Q$: 步长增益（自适应）

---

### 公式7: [[Pseudoinverse Guidance|引导增益的自适应步长]]

$$
\lambda_\tau = \min\!\Big(\beta,\; \frac{1-\tau}{\tau \cdot r_\tau^2}\Big), \quad r_\tau^2 = \frac{(1-\tau)^2}{\tau^2 + (1-\tau)^2}
$$

**含义**: 在小 $\tau$（接近噪声端）时限制增益防止过修正；$\beta = 10$ 为硬截断

**符号说明**:
- $\tau$: 当前 flow time
- $r_\tau^2$: 端点估计的相对不确定度
- $\beta$: 引导截断上限

---

### 公式8: [[JVP|DreamZero 联合引导 (Appendix C.3)]]

$$
\begin{aligned}
\hat{\mathbf{X}}_j^1 &= \mathbf{X}_j^{\sigma_j} - \sigma_j \mathbf{v}_k \\
\mathbf{e}_j &= \mathbf{P}\mathbf{W}_j(\mathbf{Y}_j - \hat{\mathbf{X}}_j^1) \\
\mathbf{g}_j &= \mathbf{J}_k^\top \mathbf{e}_j \\
\tilde{v}_j &= \mathbf{v}_k - \lambda(\sigma_j)\mathbf{g}_j
\end{aligned}
$$

**含义**: DreamZero 采用 UniPC 加速求解器，引导通过缓存速度 $\mathbf{v}_k$ 和雅可比 $\mathbf{J}_k$ 跨步复用

**符号说明**:
- $\mathbf{v}_k$: DiT 刷新点的缓存速度
- $\mathbf{J}_k$: 缓存的端点雅可比
- $\sigma_j$: 第 $j$ 个 UniPC 子步的噪声水平

---

### 公式9: [[Pseudoinverse Guidance|广义伪逆一致性]]

$$
\mathbf{H}^\dagger = \mathbf{H}^\top(\mathbf{H}\mathbf{H}^\top)^{-1}, \quad h \circ h^\dagger \circ h = h, \quad h^\dagger \circ h \circ h^\dagger = h^\dagger
$$

**含义**: Moore-Penrose 伪逆满足一致性条件，使得掩码修正在可观测坐标上精确成立

**符号说明**:
- $\mathbf{H}$: 线性测量矩阵
- $\mathbf{H}^\dagger$: Moore-Penrose 伪逆

---

### 公式10: [[Pseudoinverse Guidance|对齐坐标近似误差分析]]

$$
\begin{aligned}
\mathbf{e}_{\text{exact}} &= \mathbf{W}_t[h^\dagger(\mathbf{Y}_t) - h^\dagger(h(\hat{\mathbf{X}}_t^1))] \\
\mathbf{e}_{\text{approx}} &= \mathbf{W}_t[h^\dagger(\mathbf{Y}_t) - \hat{\mathbf{X}}_t^1] \\
\|\mathbf{e}_{\text{exact}} - \mathbf{e}_{\text{approx}}\|_2 &= \|\delta_t\|_2, \quad \delta_t = \mathbf{W}_t[h^\dagger(h(\hat{\mathbf{X}}_t^1)) - \hat{\mathbf{X}}_t^1]
\end{aligned}
$$

**含义**: 当编码器-解码器对满足 $h^\dagger(h(\mathbf{X})) \approx \mathbf{X}$ 时（对齐坐标假设），近似误差 $\delta_t \approx 0$

---

## 关键图表

### Figure 1: 阶段式 WAM 的 FBFM 工作流

![Figure 1 - Stage-wise WAM FBFM](https://arxiv.org/html/2607.29235v1/material/serial.png)

**说明**: 在前一块执行期间，真实观测逐步约束状态流（动态掩码激活）。最新修正的状态上下文再条件化动作流，动作流的重叠段同时被前一块已执行动作约束（固定动作掩码）。两个流顺序运行，各自独立的端点雅可比分别引导状态和动作修正。

---

### Figure 2: 联合生成 WAM 的 FBFM 工作流

![Figure 2 - Joint-generation WAM FBFM](https://arxiv.org/html/2607.29235v1/material/parallel.png)

**说明**: 前一块执行期间观测到的编码转移和跨块重叠动作共同约束**单一状态-动作流**。通过端点雅可比的**跨模态块**（$\mathbf{J}_{ZA}^\top$），状态反馈可直接修正动作坐标，无需状态和动作流顺序解耦。

---

### Figure 3: 阶段式与联合生成 WAM 的掩码结构对比

**阶段式 WAM（左）— 状态流掩码（step 0 → step H）:**

![Figure 3a - Serial State Mask step 0](https://arxiv.org/html/2607.29235v1/material/mask/serial_0_cropped.png)

![Figure 3b - Serial Action Mask step 1](https://arxiv.org/html/2607.29235v1/material/mask/serial_1_cropped.png)

**联合生成 WAM（右）— 块对角状态-动作联合掩码:**

![Figure 3c - Parallel Joint Mask step 0](https://arxiv.org/html/2607.29235v1/material/mask/parallel_0_cropped.png)

![Figure 3d - Parallel Joint Mask step 1](https://arxiv.org/html/2607.29235v1/material/mask/parallel_1_cropped.png)

**说明**: 阶段式 WAM 对状态流和动作流使用独立的端点雅可比和掩码；状态流的动态掩码条目随真实观测到达而激活，动作重叠掩码固定。联合生成 WAM 形成块对角状态-动作联合掩码，端点雅可比转置将修正同时传播到状态和动作坐标。

---

### Figure 4: 真实机器人手臂停球任务的观测预测对比

![Figure 4 - Real-world robot observation prediction](https://arxiv.org/html/2607.29235v1/material/wan2.2/robot_arm_ball_stop_keyframes_2row.jpg)

**说明**: 使用 RealSense D435i 从物理机器人任务录制的 RGB 序列。上下各展示 4 个关键帧（0/0.25/0.5/1 s 和 2/3/4/5 s）。每组内三行分别为：参考录像、Wan2.2 Base 无反馈、FBFM（使用全部 30 个测量潜变量槽）。FBFM 预测 MAE 从 9.63 降至 9.27，PSNR 从 20.06 dB 提升至 23.10 dB。

---

### Figure 5: LingBot-VA 机制诊断实验

![Figure 5 - LingBot-VA mechanism diagnostic](https://arxiv.org/html/2607.29235v1/material/lbva_aux_mechanism_main.jpg)

**说明**: 三个子图验证 FBFM 的作用机制：
- **(a)** 波段-0 next-state 潜变量 MSE：FBFM 改善 1.13%，确认状态反馈确实修正了潜状态预测
- **(b)** 动作 RMS 对比（相同缓存重复 vs. RTC-FBFM 缓存切换）：切换时 Fresh-action RMS 变化量为 0.00890，证明状态反馈通过速度场传播到动作坐标
- **(c)** 缓存切换和重复基准线的速度场 RMS：引导后速度场 RMS 保持在基线噪音层的 2.52× 以上，确认状态→动作传播路径

---

### Table 1: LingBot-VA 在 RoboTwin2.0 上的成功率（%）

| Method | Clean | Rand. | Overall |
|--------|-------|-------|---------|
| Base | 80.5 | 79.8 | 80.1 |
| **FBFM** | **83.3** | **82.9** | **83.1** |

**说明**: 跨 42 个任务配置单元的等权宏平均。FBFM 在 Clean 和 Randomized 配置下分别提升 2.86 和 3.10 个百分点，42 个 Clean 单元中 19 个改善，42 个 Randomized 单元中 15 个改善。整体提升 2.98 个百分点。

---

### Table 2: DreamZero 在 LIBERO 四套件上的成功率（%）

| Method | Spatial | Object | Goal | LIBERO-10 | Total |
|--------|---------|--------|------|-----------|-------|
| Base | 78.5 | 73.0 | 68.5 | 60.5 | 70.1 |
| **FBFM** | 77.0 | 72.0 | **71.0** | **63.0** | **70.8** |

**说明**: 每套件 10 个任务×20 轮次 = 200 episodes，4 套件合计 800 episodes。FBFM 在 Goal（+2.5）和 LIBERO-10（+2.5）显著提升，但在 Spatial（-1.5）和 Object（-1.0）略有下降，汇聚增益 +0.625 个百分点。差异归因于 DreamZero 的加速推理对缓存速度/雅可比的复用削弱了即时反馈效果。

---

### Table 3 & 4: 核心符号对照表

| 符号 | 含义 |
|------|------|
| $\mathcal{T}, \mathcal{S}, \mathcal{A}, P$ | 任务族、状态空间、动作空间、转移核 |
| $t, s_t, \mathcal{O}, E$ | 环境时间、物理状态、感知映射、编码器 |
| $\mathcal{H}_t$ | 交互历史 $(c_T, z_0, a_0, \ldots, z_t)$ |
| $H, d_a, d_z, D_A, D_Z, D_X$ | Horizon、维度（单步）、维度（块） |
| $\tau, k, \sigma, \varepsilon$ | flow time、solver step、噪声水平、源噪声 |
| $v_\theta, \mathbf{u}$ | 学习速度场、条件目标速度 |
| $\mathcal{I}_t^A, \mathcal{F}_{t,k}$ | 动作重叠槽集合、动态反馈集合 |
| $Q, \mathbf{W}^Q, \mathbf{P}^Q, \mathbf{J}^Q$ | 模态索引、掩码、预调节器、雅可比 |
| $h_Q, h_Q^\dagger$ | 模态编码器、近似伪逆 |
| $\mathbf{Y}^Q, \mathbf{e}^Q, \mathbf{g}^Q, \lambda$ | 观测目标、残差、引导梯度、增益 |

---

### Table 5: 两架构的运行时配置对比（Appendix C.1）

| 配置项 | LingBot-VA | DreamZero |
|--------|-----------|-----------|
| 生成方式 | Stage-wise | Joint |
| $(H, d, s)$ | $(32, 16, 16)$ | $(16, 8, 8)$ |
| 预测状态槽数 | 2 | 2 |
| 引导/JJ 刷新次数 | 25_Z / 50_A | 16 UniPC / 8 DiT-JJ |
| 伪时钟释放点 | 26 calls / 16 actions | 8 blocks / 8 actions |
| 状态目标刷新 | 每个潜变量 4 次观测 | 每 3 个动作 |
| 状态预调节器 $P_Z$ | 1 | 56/9600 |
| 比例状态增益 $k_p$ | n/a | 0.0486968 |
| 引导截断 $\beta$ | 10 | 10 |
| 精度 | BF16 | BF16 |

---

## 实验

### 数据集与 Benchmark

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 42 任务×2 配置 | Clean + Randomized 场景 | LingBot-VA 测试 |
| [[LIBERO]] | 4 套件×10 任务×20 轮 | Spatial/Object/Goal/LIBERO-10 | DreamZero 测试 |
| 物理机器人（停球） | 单场景 RGB 序列 | RealSense D435i、Wan2.2 | 真实世界验证 |

### 实现细节

- **WAM 骨干**: [[LingBot-VA]]（阶段式生成）、[[DreamZero]]（联合生成）
- **反馈注入**: 在 Flow Matching ODE 的每个 solver step 计算端点雅可比 [[JVP|VJP]]
- **无需重训练**: WAM 参数完全冻结，仅在推理时修改速度场
- **引导上下文**: `BF16` 精度、引导截断 $\beta = 10$
- **DreamZero 优化**: UniPC 加速求解器，跨步复用缓存速度和雅可比

### 真实世界验证

[[Real-Time Chunking]] 框架内测试，在机器人手臂停球任务上验证：

| 指标 | Base | FBFM | 改善 |
|------|------|------|------|
| MAE | 9.63 | 9.27 | ↓3.7% |
| PSNR (dB) | 20.06 | 23.10 | ↑3.04 dB |

---

## 批判性思考

### 优点

1. **无需训练**: 完全推理时方法，可插拔到任意 [[Flow-Matching]] 驱动的 WAM，无需数据收集或微调
2. **理论清晰**: 通过 Moore-Penrose 伪逆和 Jacobian-vector product 提供严格数学推导，并给出近似误差的分析界
3. **通用架构支持**: 同时覆盖阶段式（LingBot-VA）和联合生成（DreamZero）两类主流 WAM 架构，证明方法的普适性

### 局限性

1. **DreamZero 增益受限**: 加速推理器（UniPC）跨步复用缓存速度和雅可比，削弱即时反馈效果，导致部分套件性能下降
2. **对齐坐标近似**: $h^\dagger(h(\hat{\mathbf{X}})) \approx \hat{\mathbf{X}}$ 对任意非线性编码器-解码器对不保证，潜在误差在复杂编码空间中可能累积
3. **推理时延**: 每个 solver step 计算 Jacobian-vector products 引入额外计算开销，实时部署需进一步工程优化

### 潜在改进方向

1. **PID 风格反馈控制器**: 在 flow matching 过程中引入积分和微分项，形成更丰富的反馈控制策略
2. **加速求解器适配**: 为 UniPC、DDIM 等加速求解器设计专用的引导方案，避免缓存复用带来的即时性损失
3. **编码器感知引导**: 修正对齐坐标近似，在非线性编码空间中精确计算伪逆

### 可复现性评估

- [ ] 代码开源（论文中提及具体分支和 commit 哈希，但未公开仓库）
- [ ] 预训练模型（依赖 LingBot-VA 和 DreamZero 原始权重）
- [x] 训练细节完整（推理配置见 Table 5，运行时参数详尽）
- [x] 数据集可获取（RoboTwin2.0、LIBERO 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[Real-Time Chunking]]: FBFM 的直接前驱，RTC 在块边界施加动作重叠约束；FBFM 将此扩展至块内每个 solver step
- [[Flow-Matching]]: FBFM 的核心生成框架，通过 ODE 积分实现分布拟合
- [[JVP]]: FBFM 的关键计算原语，端点雅可比通过 vector-Jacobian product 高效计算
- [[LingBot-VA]]: 阶段式 WAM 的实例化目标
- [[DreamZero]]: 联合生成 WAM 的实例化目标

### 对比

- [[WAM]]: FBFM 在多种 WAM 架构上的统一框架
- [[RoboTwin 2.0]]: 主要评测 benchmark（阶段式 WAM）
- [[LIBERO]]: 主要评测 benchmark（联合生成 WAM）

### 方法相关

- [[Pseudoinverse Guidance]]: FBFM 引导的数学基础，掩码伪逆修正残差
- [[Action Chunking]]: WAM 的基本执行单元
- [[Flow-Matching]]: 生成框架核心

### 硬件/数据相关

- [[LIBERO]]: 联合生成 WAM 测试套件
- [[RoboTwin 2.0]]: 阶段式 WAM 测试环境

---

## 速查卡片

> [!summary] FBFM (2026)
> - **核心**: 无需训练的推理时 Flow Matching 反馈机制，将真实观测注入 WAM 执行块内部
> - **方法**: 带掩码的伪逆引导（动态状态掩码 + 固定动作重叠掩码）通过端点雅可比修正速度场
> - **结果**: LingBot-VA 在 RoboTwin2.0 整体 +2.98pp（80.1%→83.1%），DreamZero 在 LIBERO 汇聚 +0.625pp
> - **代码**: 未公开（论文引用内部 commit: e482dcc / cb08c9e）

---

*笔记创建时间: 2026-08-04*

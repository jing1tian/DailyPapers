---
title: "DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation"
method_name: "DECOWAM"
authors: [Siyuan Ma, Boshi Zhang, Yutian Zhang, Qinglian Wu, Jiaqi Zhai, Dong Wei, Qiaojun Yu]
year: 2026
venue: arXiv
tags: [world-action-model, legged-mobile-manipulation, parameter-efficient-adaptation, disentanglement, whole-body-control]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.20114v1
created: 2026-08-22
---

# 论文笔记：DECOWAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University; Shanghai AI Lab; Harbin Institute of Technology; DEEP Robotics |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[FastWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.20114) / Code（计划随 ARMDOG 数据集一同发布） |

---

## 一句话总结

> DECOWAM 通过三组解耦条件接口（未来信息瓶颈、底座-手臂因子化、自运动速度条件），用仅 25.95M 可训练参数适配冻结的 [[FastWAM]] 骨干，实现四足移动操作的全身协调预测，Action MSE 降低 21.7%。

---

## 核心贡献

1. **解耦条件接口设计**: 将 *底座运动*、*手臂动作*、*摄像头自运动* 作为独立因子显式建模，替代端到端混合建模
2. **参数高效分阶段适配**: Stage-1 全量微调对齐领域，Stage-2 冻结骨干只训 25.95M 参数（232× 压缩），推理时序严格因果
3. **ARMDOG 数据集**: 首个同步 RGB-D、本体感知、IMU、全身状态指令与语言标注的四足移动操作数据集，1,487 段 / 343,550 帧

---

## 问题背景

### 要解决的问题

四足移动操作（Legged Mobile Manipulation）的关键挑战：底座运动、手臂控制和摄像头视角变化相互耦合，导致现有 [[世界模型|World-Action Model]] 在预测未来视频时无法区分"底座移动造成的视角漂移"与"场景本身变化"，同时造成动作空间内底座与手臂的量纲不匹配。

### 现有方法的局限

- **[[FastWAM]]**（前身）：全量微调参数量超 6B，底座与手臂动作混合建模，在静止操作时会产生非预期底座运动
- **[[Motus]]**、X-WAM：在 ARMDOG 域外训练，未针对全身协调显式解耦
- **[[Action Chunking|VLA 类方法]]**（[[π0.5]]、X-VLA、GR00T）：不预测未来视频，缺少世界模型的预见性

### 本文的动机

冻结预训练视频-动作骨干的 *视频先验*，以极少参数注入机器人专有信息：**何处移动（底座）/ 如何操作（手臂）/ 摄像头如何运动（自运动）** 三者分离，可在保持视频质量的同时提升动作精度，且部署时移除特权未来信息，保持因果推理。

---

## 方法详解

### 模型架构

DECOWAM 采用 **冻结骨干 + 四组可训练模块** 架构：

- **输入**: 语言指令 $\ell$ + 当前 RGB 帧 $x_0$ + 全身状态 $s_0$（含底座速度 $v_0 \in \mathbb{R}^3$）
- **Backbone**: 冻结 [[FastWAM]]（WAN 视频生成骨干），通过 [[残差适配器|Residual Adapter]] 每层注入
- **核心模块**:
  - [[残差适配器|Residual Adapter]] $\phi_{adp}$：参数高效地向冻结层注入领域知识
  - [[动作等价未来瓶颈|Action-Equivalent Future Bottleneck]] $\phi_q$：Teacher-Student 蒸馏未来信息
  - [[底座-手臂因子化|Base-Arm Factorization]] $\phi_{ba}$：[[梯度反转层|GRL]] 对抗解耦
  - 自运动速度条件 $\phi_{ego}$：底座速度 $v_0$ 直接作用于视频 token
- **输出**: 未来 8 帧 RGB 预测 + 48-step、14-D [[Action Chunking|动作块]]（3D 底座 + 7D 手臂 + 1D 夹爪 + 3D padding）
- **总参数**: 6,751.38M（仅 25.95M 可训练）

### 核心模块

#### 模块 1：残差适配器（Residual Adapter）

**设计动机**: 用 [[LoRA]] 式低秩瓶颈保留预训练视频先验，同时注入机器人领域知识

**具体实现**:

$$
h_l^+ = h_l + \alpha_l \, W_{\text{up}}^{(l)} \, \sigma\!\left(W_{\text{down}}^{(l)} \, \text{LN}(h_l)\right)
$$

每个 Transformer 块后插入，$\alpha_l$ 为块级缩放，$W_{\text{down}} \in \mathbb{R}^{r \times d}$，$W_{\text{up}} \in \mathbb{R}^{d \times r}$ 低秩瓶颈，$\sigma$ 为 SiLU。

#### 模块 2：动作等价未来瓶颈（Action-Equivalent Future Bottleneck）

**设计动机**: 训练时可利用未来帧信息辅助动作预测（特权信息），部署时移除，借助 [[知识蒸馏|Knowledge Distillation]] 让学生端在无未来帧时保持性能

**具体实现**:
- 计算视觉摘要 $c = \rho(e_0)$，$f = \rho(e_{1:T})$，其中 $\rho(e) = [\text{mean}(e),\; \text{std}(e)]$
- Teacher 嵌入观察当前和未来：$z_t = q_t([c, f, s_0])$
- Student 嵌入仅观察当前：$z_s = q_s([c, s_0])$，$z_t, z_s \in \mathbb{R}^{d_q}$
- Student 通过低秩投影调制动作专家：$\tilde{u}^a = u^a + \eta_q B_q z_s$
- 三项联合损失约束（见公式区）

#### 模块 3：底座-手臂因子化（Base-Arm Factorization）

**设计动机**: 底座导航尺度（0.x m/s）与手臂操作尺度（mm 级）量纲差距大，混合建模导致预测偏差；[[梯度反转层|GRL]] 对抗训练强制解耦

**具体实现**:

$$
z_{\text{base}} = b_\phi(u^a), \quad z_{\text{arm}} = m_\phi(u^a), \quad z_{\text{base}}, z_{\text{arm}} \in \mathbb{R}^{16}
$$

再通过对抗解纠缠损失（GRL）使 $z_{\text{base}}$ 无法预测手臂动作，$z_{\text{arm}}$ 无法预测底座动作（见公式区）。调制动作专家：$\bar{u}^a = \tilde{u}^a + \eta_{ba} B_{ba}[z_{\text{base}}, z_{\text{arm}}]$

#### 模块 4：自运动感知视频条件（Ego-Motion-Aware Video Conditioning）

**设计动机**: 底座移动造成的摄像头视角平移/旋转需从场景动态中剥离，避免模型把"摄像头移了"和"物体动了"混淆

**具体实现**:

$$
v_0 = \Pi_{\text{base}}(s_0) = (v_x, v_y, \omega_z) \in \mathbb{R}^3
$$

$$
\tilde{h}^v_i = h^v_i + \beta B_v v_0, \quad i = 1, \ldots, N_v
$$

所有视频 token 加入底座速度偏置，低成本显式解释自运动。

---

## 关键公式

### 公式 1：[[联合世界-动作模型|联合生成目标]]

$$
p_\theta(x_{1:T},\; a_{1:K} \mid x_0, s_0, \ell)
$$

**含义**: 给定初始帧 $x_0$、全身状态 $s_0$、语言指令 $\ell$，联合生成未来帧序列与动作块。

**符号说明**:
- $x_{1:T}$：未来 $T$ 帧 RGB
- $a_{1:K}$：$K$ 步动作块
- $\ell$：语言任务指令

---

### 公式 2：[[Action Chunking|动作分解]]

$$
a_k = [a^{\text{arm}}_k,\; a^{\text{grip}}_k,\; a^{\text{base}}_k,\; a^{\text{pad}}_k], \quad a^{\text{base}}_k \in \mathbb{R}^3
$$

**含义**: 14-D 动作向量由 7D 手臂关节、1D 夹爪、3D 底座速度、3D padding 拼接。

---

### 公式 3：分阶段训练目标（Stage-1）

$$
\Theta^{(1)} = \arg\min_\Theta\; \mathbb{E}_{\mathcal{D}}\!\left[\mathcal{L}_{\text{video}}(\Theta) + \mathcal{L}_{\text{action}}(\Theta)\right]
$$

**含义**: Stage-1 全参数优化视频重建与动作回归损失之和，完成领域对齐。

---

### 公式 4：Stage-2 可训练参数组

$$
\Phi = \{\phi_{\text{adp}},\; \phi_q,\; \phi_{ba},\; \phi_{\text{ego}}\}
$$

$$
\Phi^\star = \arg\min_\Phi\; \mathcal{L}(\Theta^{(1)}, \Phi)
$$

**含义**: Stage-2 冻结 $\Theta^{(1)}$，只优化四组解耦适配模块 $\Phi$（共 25.95M 参数）。

---

### 公式 5：[[残差适配器|Residual Adapter]]

$$
h_l^+ = h_l + \alpha_l \, W_{\text{up}}^{(l)} \, \sigma\!\left(W_{\text{down}}^{(l)} \, \text{LN}(h_l)\right)
$$

**含义**: 每个冻结 Transformer 块之后通过低秩瓶颈投影注入可训练偏置。

**符号说明**:
- $\alpha_l$：块级缩放系数
- $W_{\text{down}}^{(l)} \in \mathbb{R}^{r \times d}$，$W_{\text{up}}^{(l)} \in \mathbb{R}^{d \times r}$：低秩上/下投影
- $\sigma$：SiLU 激活

---

### 公式 6：视觉摘要计算

$$
c = \rho(e_0), \quad f = \rho(e_{1:T}), \quad \rho(e) = [\text{mean}(e),\; \text{std}(e)]
$$

**含义**: 将 VAE 编码的时序特征压缩为均值+标准差统计量，作为 Teacher/Student 的紧凑输入。

---

### 公式 7：Teacher / Student 嵌入

$$
z_t = q_t([c, f, s_0]), \quad z_s = q_s([c, s_0]), \quad z_t, z_s \in \mathbb{R}^{d_q}
$$

**含义**: Teacher 可见未来摘要 $f$，Student 只可见当前摘要 $c$；两者均可见状态 $s_0$。

---

### 公式 8：Student 调制动作专家

$$
\tilde{u}^a = u^a + \eta_q B_q z_s
$$

**含义**: 用 Student 嵌入 $z_s$ 低秩调制动作专家特征 $u^a$；部署时 Teacher 路径移除，仅保留此式。

---

### 公式 9：[[动作等价未来瓶颈|瓶颈联合损失]]

$$
\mathcal{L}_{\text{rec}}^q = \|r_s(z_s) - a_{1:K}\|_2^2 + \|r_t(z_t) - a_{1:K}\|_2^2
$$

$$
\mathcal{L}_q = \lambda_{\text{act}}^q \mathcal{L}_{\text{rec}}^q + \lambda_{\text{dist}}^q \|z_s - \text{sg}(z_t)\|_2^2 + \lambda_{\text{geom}}^q \mathcal{L}_{\text{geom}}
$$

**含义**: 三项约束：动作重建（Teacher 和 Student 各自回归动作）、嵌入蒸馏（Student 靠近停梯度 Teacher）、几何保持（见公式 10-11）。

**符号说明**:
- $r_s, r_t$：轻量动作回归头
- $\text{sg}(\cdot)$：停止梯度
- $\lambda^q_{\text{act}}=1.0$，$\lambda^q_{\text{dist}}$、$\lambda^q_{\text{geom}}$：权重超参

---

### 公式 10：几何距离定义

$$
d^{ij}_z = \frac{\|z_t^i - z_t^j\|_2}{\tau_z}, \quad d^{ij}_a = \frac{\|a_{1:K}^i - a_{1:K}^j\|_2}{\tau_a}
$$

**含义**: 在潜在空间和动作空间分别计算 batch 内样本对的归一化距离。

---

### 公式 11：几何保持损失

$$
\mathcal{L}_{\text{geom}} = \frac{1}{B(B-1)} \sum_{i \neq j} \text{SL1}(d^{ij}_z,\; d^{ij}_a)
$$

**含义**: 要求 Teacher 嵌入空间的样本对距离与动作空间距离一致（Smooth L1），使嵌入保留动作结构信息。

---

### 公式 12：底座-手臂因子化

$$
z_{\text{base}} = b_\phi(u^a), \quad z_{\text{arm}} = m_\phi(u^a), \quad z_{\text{base}}, z_{\text{arm}} \in \mathbb{R}^{16}
$$

**含义**: 从动作专家特征 $u^a$ 中分别提取 16-D 底座潜变量和手臂潜变量。

---

### 公式 13：底座-手臂联合调制

$$
\bar{u}^a = \tilde{u}^a + \eta_{ba} B_{ba}[z_{\text{base}}, z_{\text{arm}}]
$$

**含义**: 将解耦后的底座和手臂潜变量拼接后低秩注入动作专家特征。

---

### 公式 14：[[梯度反转层|对抗解纠缠损失]]

$$
\mathcal{L}_{\text{disent}} = \|g_b(z_{\text{base}}) - a^b_{1:K}\|_2^2 + \|g_m(z_{\text{arm}}) - a^m_{1:K}\|_2^2
$$

$$
\quad + \|\tilde{g}_b(\text{GRL}(z_{\text{arm}})) - a^b_{1:K}\|_2^2 + \|\tilde{g}_m(\text{GRL}(z_{\text{base}})) - a^m_{1:K}\|_2^2
$$

**含义**: 前两项令底座/手臂潜变量各自预测对应动作；后两项通过 [[梯度反转层|GRL]] 使 $z_{\text{arm}}$ 无法预测底座动作（反之亦然），实现对抗解耦。

**符号说明**:
- $g_b, g_m$：底座/手臂动作回归头
- $\tilde{g}_b, \tilde{g}_m$：对抗回归头
- GRL：梯度反转层，前向传播为恒等，反向传播取负梯度

---

### 公式 15：归一化底座速度

$$
v_0 = \Pi_{\text{base}}(s_0) = (v_x, v_y, \omega_z) \in \mathbb{R}^3
$$

**含义**: 从全身状态 $s_0$ 中提取底座线速度和偏航角速度，作为自运动先验。

---

### 公式 16：自运动视频 Token 条件化

$$
\tilde{h}^v_i = h^v_i + \beta B_v v_0, \quad i = 1, \ldots, N_v
$$

**含义**: 所有 $N_v$ 个视频 token 加入底座速度偏置，显式告知模型摄像头如何移动。

---

### 公式 17：[[Flow Matching|条件流匹配]]采样路径

$$
y_\tau = (1 - \tau)\varepsilon + \tau y, \quad v^\star(y_\tau, \tau) = y - \varepsilon
$$

**含义**: 在时间步 $\tau \in [0,1]$ 的线性插值路径上定义目标速度场 $v^\star$。

**符号说明**:
- $\varepsilon \sim \mathcal{N}(0, I)$：噪声
- $y$：干净样本（帧或动作）
- $\tau$：流匹配时间步

---

### 公式 18：[[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{FM}}(F_\theta;\; y, c) = \mathbb{E}_{\tau, \varepsilon}\!\left[\|F_\theta(y_\tau, \tau, c) - v^\star(y_\tau, \tau)\|_2^2\right]
$$

**含义**: 训练网络 $F_\theta$ 拟合条件速度场，同时用于视频帧（$\mathcal{L}_{\text{video}}$）和动作（$\mathcal{L}_{\text{action}}$）。

---

### 公式 19：Stage-2 完整优化目标

$$
\mathcal{L} = \lambda_v \mathcal{L}_{\text{video}} + \lambda_a \mathcal{L}_{\text{action}} + \gamma_q \lambda_q \mathcal{L}_q + \gamma_{ba} \lambda_{ba} \mathcal{L}_{\text{disent}}
$$

**含义**: 视频生成、动作预测、瓶颈蒸馏、对抗解耦四项损失的加权总和。

**符号说明**:
- $\lambda_v = \lambda_a = 1.0$：主任务权重
- $\lambda_q = 0.2$，$\lambda_{ba} = 0.1$：辅助损失权重
- $\gamma_q, \gamma_{ba}$：Stage-2 阶段门控系数

---

## 关键图表

### Figure 1：DECOWAM 整体架构与训练/部署流程

![Figure 1](https://arxiv.org/html/2608.20114v1/figures/intro_overview.png)

**说明**: (A) 训练时冻结 WAN 骨干，插入可训练残差适配器、Teacher-Student 瓶颈、底座/手臂解耦潜变量和自运动速度条件；[[ActionDiT]] 生成未来 8 帧 RGB 和 48-step 14-D 动作块。(B) 部署时移除未来帧输入和 Teacher 路径，保持严格因果推理。(C) 分阶段训练：Stage-1 全量对齐，Stage-2 冻结骨干只训解耦模块。

---

### Figure 2：动作等价未来瓶颈

![Figure 2](https://arxiv.org/html/2608.20114v1/figures/quotient_bottleneck.png)

**说明**: 特权 Teacher 同时观察当前与未来视觉摘要，因果 Student 仅观察当前摘要和机器人状态。部署时仅保留 Student 路径，通过 [[知识蒸馏|Knowledge Distillation]] 将未来信息压缩进 $z_s \in \mathbb{R}^{d_q}$。

---

### Figure 3：ARMDOG 数据集构成

![Figure 3](https://arxiv.org/html/2608.20114v1/figures/dataset_overview.png)

**说明**: (a) 任务分布：Bottle Pick&Place 56%、Place Block 39%、Object Pick&Place 4%、Climb Slope 1%。(b) 数据规模：质量过滤后 1,487 段 / 343,550 帧 / 321.3 min / 15 Hz。(c) 模型接口：文本 + 当前 RGB + 机器人状态为输入，未来 RGB 和 14-D 动作为预测目标。

---

### Figure 4：WAM 参考模型定性对比

![Figure 4](https://arxiv.org/html/2608.20114v1/figures/wam_rollout.png)

**说明**: 固定 ARMDOG replay 样本上的定性对比。红色虚线标出摄像头视野边界。DECOWAM 更好地保持工作空间几何和物体布局，对比模型出现模糊或视角漂移。

---

### Figure 5：实机全身协调运动对比

![Figure 5](https://arxiv.org/html/2608.20114v1/figures/real_robot_coordination.png)

**说明**: 三列帧从左到右依次展示任务进展。X-VLA（上）和原始 FastWAM（中）在需要静止时产生非预期底座运动并未能完成操作，DECOWAM（下）成功完成全身协调操作。

---

### TABLE I：ARMDOG 23 段验证集 Replay 诊断

| 模型 | F-MSE ↓ | PSNR ↑ | A-MSE ↓ |
|------|---------|--------|---------|
| FastWAM (40k) | 1.616e-3 | 30.468 | 1.154e-4 |
| FastWAM (50k) | 1.032e-3 | 31.441 | 6.87e-5 |
| FastWAM (80k) | 2.136e-3 | 28.544 | 4.77e-4 |
| **DECOWAM (50k)** | **8.77e-4** | **31.663** | **5.38e-5** |

**说明**: DECOWAM 以 1/232 的可训练参数超越全量 FastWAM，F-MSE 降低 15.0%，A-MSE 降低 21.7%。

---

### TABLE II：解耦模块消融实验

| 变体 | F-MSE ↓ | PSNR ↑ | SSIM ↑ | A-MSE ↓ | A-MAE ↓ |
|------|---------|--------|--------|---------|---------|
| 完整冻结解耦模型 | 9.35e-4 | 31.378 | 9.9241e-1 | 8.09e-5 | 4.310e-3 |
| w/o quotient 瓶颈 | 1.02e-3 | 31.228 | 9.9170e-1 | 8.40e-5 | 4.316e-3 |
| w/o base-vel 条件 | 1.01e-3 | 31.221 | 9.9178e-1 | 8.80e-5 | 4.342e-3 |
| w/o 解耦模块 | 9.80e-4 | 31.105 | 9.9138e-1 | 9.60e-5 | 5.135e-3 |

**关键发现**: 移除任一解耦模块均导致性能下降；底座速度条件对 A-MSE 影响最大（+8.8%），说明自运动建模对动作精度至关重要。

---

### TABLE III：与 VLA 参考模型的动作对比

| 模型 | 类别 | 使用 RGB | A-MSE ↓ | A-MAE ↓ | A-L2 ↓ |
|------|------|---------|---------|---------|--------|
| π0.5 | VLA | 否 | 1.79e-4 | 3.60e-3 | 2.83e-2 |
| X-VLA | VLA | 否 | 2.11e-5 | 1.82e-3 | 1.00e-2 |
| GR00T | VLA | 否 | 4.18e-3 | 2.51e-2 | 1.83e-1 |
| **DECOWAM** | **WAM** | **是，8帧** | **5.38e-5** | **4.01e-3** | **2.24e-2** |

**说明**: DECOWAM 在同时预测未来视频的情况下，A-MSE 超越 π0.5，接近 X-VLA（X-VLA 不预测视频）。

---

### TABLE IV：世界-动作模型参考横向对比

| 模型 | 可训练参数 | RGB 帧数 | F-MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | A-MSE ↓ | A-MAE ↓ | A-L2 ↓ |
|------|-----------|---------|---------|--------|--------|---------|---------|---------|--------|
| **DECOWAM** | **25.95M** | 8f 384p | **8.77e-4** | **31.66** | **9.93e-1** | **2.95e-2** | **5.38e-5** | **4.01e-3** | **2.24e-2** |
| FastWAM | 6020.75M | 8f 384p | 1.03e-3 | 31.44 | 9.92e-1 | 3.03e-2 | 6.87e-5 | 4.08e-3 | 2.35e-2 |
| Motus | 5894.81M | 8f 384p | 5.19e-3 | 23.41 | 9.57e-1 | 1.01e-1 | 5.05e-4 | 9.97e-3 | 5.99e-2 |
| Cosmos 2.5 | — | 8f 384p | 4.28e-2 | 14.38 | 6.63e-1 | 2.71e-1 | — | — | — |
| X-WAM | 5037.75M | 4f 160p | 2.62e-3 | 26.57 | 9.81e-1 | 3.50e-2 | 6.31e-4 | 1.11e-2 | 2.48e-2 |
| UVA | 261.62M | 4f 128p | 1.79e-2 | 19.32 | 8.40e-1 | 2.00e-1 | 9.76e-4 | 1.31e-2 | 9.81e-2 |

**关键发现**: DECOWAM 以最少可训练参数（比 FastWAM 少 232×）在所有 WAM 中取得最佳视频和动作指标。

---

### TABLE V(a)：实机实验任务进展与完成效率（79 次试验）

| 模型 | 平均完成时间 ↓ | 到达 ↑ | 抓取 ↑ | 运输 ↑ | 放置 ↑ | 成功率 ↑ |
|------|-------------|--------|--------|--------|--------|---------|
| GR00T | 57s | 73.4% | 13.9% | 11.4% | 8.9% | 8.9% |
| π0.5 | 50s | 89.9% | 62.0% | 53.2% | 49.4% | 49.4% |
| FastWAM | 65s | 91.1% | 63.3% | 59.5% | 57.0% | 57.0% |
| X-WAM | 82s | 77.2% | 26.6% | 19.0% | 15.2% | 15.2% |
| **DECOWAM** | **49s** | **92.4%** | **69.6%** | **67.1%** | **58.2%** | **58.2%** |

### TABLE V(b)：实机全身鲁棒性（79 次试验）

| 模型 | BD-SR ↑ | WBCM-SR ↑ | BDP-SR ↑ | AR-SR ↑ |
|------|--------|----------|---------|--------|
| GR00T | 70.9% | 16.5% | 1.3% | 11.4% |
| π0.5 | 87.3% | 36.7% | 11.4% | 25.3% |
| FastWAM | 83.5% | 34.2% | 12.7% | 27.8% |
| X-WAM | 69.6% | 27.8% | 5.1% | 12.7% |
| **DECOWAM** | **87.3%** | **44.3%** | **30.4%** | **32.9%** |

**关键发现**: DECOWAM WBCM-SR（全身协调成功率）44.3% 为最高，BDP-SR（底座位移鲁棒性）30.4% 是 FastWAM 的 2.4×，证明解耦设计显著改善全身协调。

---

### TABLE VI：部署诊断（参数与推理效率）

| 模型 | 总参数 | 可训练参数 | 推理延迟 | F-MSE ↓ | A-MSE ↓ |
|------|--------|----------|---------|---------|---------|
| FastWAM | 6725.44M | 6020.75M | 1196.6 ms | 1.032e-3 | 6.87e-5 |
| **DECOWAM** | **6751.38M** | **25.95M** | **1333.2 ms** | **8.77e-4** | **5.38e-5** |

**说明**: DECOWAM 推理延迟增加约 11.4%（因增加了解耦模块），但在更少可训练参数下取得更优的视频和动作质量。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[ARMDOG]] | 1,487 段 / 343,550 帧 / 321.3 min | 同步 RGB-D + 本体感知 + IMU + 全身状态/指令 + 语言，15 Hz | 训练 + 测试 |
| ARMDOG 验证切片 | 23 段（固定） | 8 种任务变体，用于跨模型一致性对比 | Replay 测试 |

### 实现细节

- **Backbone**: 冻结 FastWAM (WAN-based)，Stage-1 全量微调 50k 步
- **Stage-2 可训参数**: 25.95M（残差适配器 + 瓶颈 + 因子化 + 自运动条件）
- **优化器**: AdamW，Stage-2 仅优化 $\Phi$
- **动作块长度**: $K=48$ 步，14-D 归一化动作向量
- **视频分辨率**: 384p，8 帧预测
- **硬件**: 四足机器人（DEEP Robotics 平台）+ 手臂末端执行器
- **推理延迟**: 1333.2 ms/步

### 可视化结果

Figure 4 展示定性对比：在 WAM Replay 上，DECOWAM 的未来帧预测明显更锐利，工作空间物体位置更准确，而对比模型（尤其 Motus、Cosmos 2.5）出现模糊和视角漂移。Figure 5 展示实机部署中 DECOWAM 成功完成全身协调操作，X-VLA 和 FastWAM 则在需要静止时仍产生底座漂移。

---

## 批判性思考

### 优点

1. **解耦思路清晰，消融实验充分**: 三种解耦接口各有独立消融，贡献分析清晰
2. **参数效率极高**: 232× 参数压缩，推理延迟仅增加 11%，工程部署友好
3. **ARMDOG 数据集贡献**: 首个同步全身状态+语言的四足操作数据集，具有独立价值
4. **实机验证扎实**: 79 次真实机器人试验，WBCM-SR 等全身协调专属指标设计合理

### 局限性

1. **任务分布单调**: ARMDOG 56% 为 Bottle Pick&Place，真实泛化能力存疑
2. **推理延迟增加**: 1333 ms 对高频控制（>10 Hz）存在挑战
3. **无项目主页/代码**: 论文提交时代码未开源，复现成本高
4. **X-VLA 动作指标更优**: 不预测视频的 VLA 在 A-MSE 上优于 DECOWAM，说明视频预测与动作精度之间存在 trade-off

### 潜在改进方向

1. 扩展 ARMDOG 任务多样性（双臂操作、动态障碍等）
2. 结合在线 RL 进一步优化 BDP-SR（底座位移鲁棒性）
3. 探索 Flow Consistency Distillation 降低推理步数以减少延迟

### 可复现性评估

- [ ] 代码开源（计划随数据集发布）
- [ ] 预训练模型（待发布）
- [x] 训练细节完整（损失权重、超参数均在论文中说明）
- [ ] 数据集可获取（ARMDOG 计划开源 HDF5）

---

## 关联笔记

### 基于

- [[FastWAM]]: WAN-based World-Action Model 骨干，DECOWAM 的 Stage-1 起点
- [[ActionDiT]]: 动作-视频联合 DiT 架构，DECOWAM 沿用其 ActionDiT 结构
- [[Flow Matching]]: 视频和动作生成均采用条件流匹配

### 对比

- [[Motus]]: 另一个大参数 WAM，在 ARMDOG 上 F-MSE 差 5.9×
- [[π0.5]]: 强 VLA baseline，实机成功率 49.4% vs DECOWAM 58.2%
- [[知识蒸馏|Knowledge Distillation]]: Teacher-Student 瓶颈的理论基础

### 方法相关

- [[残差适配器|Residual Adapter]]: 参数高效适配核心机制
- [[梯度反转层|Gradient Reversal Layer]]: 对抗解纠缠的关键组件
- [[Action Chunking]]: 14-D 动作块预测方式
- [[LoRA]]: 与残差适配器结构相近的低秩适配思路

### 硬件/数据相关

- [[ARMDOG]]: 本文贡献的四足移动操作数据集

---

## 速查卡片

> [!summary] DECOWAM
> - **核心**: 冻结 FastWAM 骨干，三组解耦接口（未来瓶颈/底座-手臂因子化/自运动速度条件）
> - **方法**: Stage-2 仅训 25.95M 参数（232× 压缩），Teacher-Student + GRL 对抗解耦
> - **结果**: A-MSE -21.7%，实机成功率 58.2%（最高），WBCM-SR 44.3%（最高）
> - **代码**: 计划随 ARMDOG 数据集一同开源

---

*笔记创建时间: 2026-08-22*

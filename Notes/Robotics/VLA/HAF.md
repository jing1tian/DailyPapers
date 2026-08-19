---
title: "HAF: Adapting Generalist VLAs to Humanoid Whole-Body Loco-manipulation via Hierarchical Action Flow and Spectral Latent RL"
method_name: "HAF"
authors: [Langzhe Gu, Chengkai Hou, Meng Li, Xinhua Wang, Jiaming Liu, Xinyuan Lv, Bowei Zhang, Shuanghao Bai, Guangrun Li, Jingyang He, Gaole Dai, Ziluo Ding, Zhiyuan Xu, Kuan Cheng, Jian Tang, Zhengping Che, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [vla, humanoid-robot, loco-manipulation, flow-matching, reinforcement-learning, whole-body-control, hierarchical-policy]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.16837
created: 2026-08-19
---

# 论文笔记：HAF

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多家中国研究机构（含北京大学等） |
| 日期 | August 2026 |
| 项目主页 | [grange007.github.io/HAF](https://grange007.github.io/HAF) |
| 对比基线 | [[π₀.₅]], [[GR00T]], [[DSRL]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.16837) / Code: N/A |

---

## 一句话总结

> HAF 通过分层动作流生成（HAF-VLA）和谱空间潜变量强化学习（HAF-Steer）两个互补组件，让通用 [[Flow-Matching]] VLA 高效适应全身型人形机器人的行走-操作任务。

---

## 核心贡献

1. **HAF-VLA（分层动作流生成器）**: 将全身动作去噪拆分为三个顺序阶段（行走/头部→腰部→双臂操作），利用跨阶段 [[KV Cache]] 条件化，保持运动学一致性，同时显著提升协调效果。
2. **HAF-Steer（谱空间潜变量 RL）**: 利用流匹配可逆性恢复专家演示的噪声，通过 [[DCT]] 压缩到紧凑频谱子空间，在冻结主干网络的前提下以 [[SAC]] 进行离线到在线 RL 微调，实现安全高效的策略提升。
3. **真实环境七任务验证**: 在 TienKung 人形机器人上测试七项需导航+全身姿态调整+双臂操作的任务，平均成功率 70.5%，较 π₀.₅（53.3%）提升 17.2 个百分点。

---

## 问题背景

### 要解决的问题

如何将通用 [[Flow-Matching]] VLA 模型（如 π₀.₅）迁移到人形机器人的[[Whole-Body Control|全身行走-操作任务]]（[[Loco-Manipulation]]）。这类任务同时涉及腿部运动、腰部姿态、双臂操控三类高度耦合的关节自由度，单阶段动作去噪难以有效协调。

### 现有方法的局限

- **单阶段 VLA**（如 π₀.₅）：对全身 30+ 自由度同时去噪，运动学连贯性差，操作精度低。
- **大型人形基础模型**（如 GR00T N1.7）：在专用数据少、场景多样性高时表现不佳（38.1%）。
- **世界模型方法**（如 Cosmos Policy）：平均仅 27.6%，对长时程任务鲁棒性弱。
- **现有潜变量 RL**（如 [[DSRL]]）：单一重复噪声向量导致时序抖动，且真实机器人上探索不安全（导致提前终止）。

### 本文的动机

行走-操作任务中，足部locomotion 在动力学上先于腰部稳定，腰部稳定又先于臂部精操作，存在天然的时序依赖关系。HAF 利用这一先验设计三阶段生成顺序，并通过频谱子空间的维度压缩使在线 RL 探索既高效又安全。

---

## 方法详解

### 模型架构概览

HAF 由两个可独立使用、也可组合的模块构成：

- **输入**: 语言指令 $l$ + 头部 RGB 图像观测 $o_t$ + 关节状态 $s_t$
- **Backbone**: 预训练 [[Flow-Matching]] VLA（π₀.₅ 结构）
- **HAF-VLA**: 三阶段 [[Hierarchical Chunking|分层动作块]] 去噪，跨阶段 [[KV Cache]] 条件化
- **HAF-Steer**: 频谱潜变量演员 $\pi_\psi$，冻结 VLA 主干，仅更新小型 RL 模块
- **输出**: [[Action Chunking|动作块]] $A_t \in \mathbb{R}^{H \times D}$，执行 $H=40$ 步后重新推理
- **总参数**: 主干冻结；HAF-Steer 仅训练演员、双 Critic 和温度参数 $\alpha$

### 核心模块

#### 模块 1：HAF-VLA — 分层全身动作生成

**设计动机**: 利用人形机器人的运动学依赖（locomotion → waist → manipulation）以及 [[KV Cache]] 机制，使早期阶段的干净动作为后续阶段提供全局上下文，避免高维度全联合去噪中的运动不一致。

**动作索引集嵌套定义**（Eq. 1）：

$$
A_1 = A^{\text{move}} \cup A^{\text{head}}, \quad
A_2 = A_1 \cup A^{\text{waist}}, \quad
A_3 = A_2 \cup A^{\text{manip}}
$$

**三阶段生成过程**：

- **Stage 1**：对 $A_1$（腿+头，低维）独立采样噪声 $\epsilon_1 \sim \mathcal{N}(0,I)$，去噪 10 步得 $\hat{A}_1$；
- **Stage 2**：将 $\hat{A}_1$ 重编码为键值对 $C_t^1$ 注入注意力，对 $A_2 \setminus A_1$（腰部关节）去噪 10 步得 $\hat{A}_2$；
- **Stage 3**：以 $C_t^1, C_t^2$ 双重条件，对 $A_3 \setminus A_2$（双臂）去噪 10 步得 $\hat{A}_3$；最终 $A_t = [\hat{A}_1; \hat{A}_2; \hat{A}_3]$。

训练采用 [[teacher forcing]]：缓存计算时用真值 mask，防止误差累积。推理时仅 Stage 3 输出驱动机器人（Receding-Horizon Control）。

**单次推理耗时约 0.12 s（RTX 5090）**，每次推理执行 $H=40$ 步。

#### 模块 2：HAF-Steer — 谱空间潜变量离线到在线 RL

**设计动机**: 流匹配 VLA 的初始噪声 $\epsilon$ 完整决定最终动作；通过 [[DCT]] 对 $\epsilon$ 的时域系数进行截断，可以在一个低维、平滑、有边界的频谱子空间中安全探索，同时保留专家轨迹的低频模态。

**流程概览**:

1. **流逆向恢复**（Eq. 5a）：对专家演示动作 $A^*$ 数值反向积分 $\mathcal{F}^{-1}_\theta$，恢复初始噪声 $\epsilon^*$；
2. **DCT 压缩**（Eq. 5b-5c）：取前 $K=8$ 个时间系数 $c^* = \text{DCT}_K(\epsilon^*)$，标准化为 $z^* = (c^* - \mu_c)/(\sigma_c + \delta)$；
3. **行为克隆初始化**（Eq. 7）：以 $z^*$ 为监督训练频谱演员 $\pi_\psi$；
4. **混合离线-在线 SAC**（Eq. 8）：离线 mini-batch 加 BC 正则 $\lambda_{\text{BC}}$，在线样本纯 SAC 梯度；训练过程中逐步减少离线采样比；
5. **推理解码**（Eq. 9）：采样 $z \sim \pi_\psi(\cdot|x)$ → 反标准化 → IDCT 零填充还原 $\epsilon$ → 冻结 VLA 解码得 $A$。

---

## 关键公式

### 公式 1：[[Hierarchical Chunking|嵌套动作索引集]]

$$
A_1 = A^{\text{move}} \cup A^{\text{head}},\quad A_2 = A_1 \cup A^{\text{waist}},\quad A_3 = A_2 \cup A^{\text{manip}}
$$

**含义**: 三个阶段的动作空间逐步包含，确保阶段 $s$ 的生成仅在 $A_s$ 范围内激活，其余维度用 mask 屏蔽。

**符号说明**:
- $A^{\text{move}}$: 腿部运动关节索引集
- $A^{\text{head}}$: 头部关节索引集
- $A^{\text{waist}}$: 腰部关节索引集
- $A^{\text{manip}}$: 双臂（双手）操作关节索引集

### 公式 2：[[Flow-Matching|掩码噪声输入与目标]]

$$
X_t = \bigl[(1-\tau)\,\epsilon_t + \tau\,A_t\bigr] \odot m_s,\qquad
u_t = (A_t - \epsilon_t) \odot m_s
$$

**含义**: 在 $[0,1]$ 均匀时间步 $\tau$ 下，对当前阶段 $s$ 的动作子集施加线性内插噪声，形成流匹配训练对。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间步
- $\epsilon_t \sim \mathcal{N}(0,I)$: 独立高斯噪声（每阶段独立采样）
- $m_s$: 阶段 $s$ 的 0/1 掩码（仅对 $A_s$ 中的维度为 1）
- $X_t$: 掩码后的噪声输入
- $u_t$: 流匹配回归目标（速度场方向）

### 公式 3：[[Flow-Matching|HAF 流匹配训练损失]]

$$
\mathcal{L}_{\text{HAF}} = \mathbb{E}\!\left[\sum_{s=1}^{3} \frac{1}{HD}\,\bigl\|v_\theta(X, \tau; h) - u\bigr\|_F^2\right]
$$

**含义**: 三个阶段的流匹配损失等权求和，共享动作专家网络 $v_\theta$，由语言条件 $h$ 驱动。

**符号说明**:
- $H$: 动作块长度（时间步数）
- $D$: 各阶段动作维度
- $v_\theta$: 共享动作专家网络（Flow-Matching velocity field）
- $h$: 语言和视觉条件嵌入

### 公式 4：[[Spectral Latent Space|频谱潜变量构建]]

$$
\epsilon^* = \mathcal{F}^{-1}_\theta(A^* \mid \zeta^*),\quad
c^* = \text{DCT}_K(\epsilon^*),\quad
z^* = \frac{c^* - \mu_c}{\sigma_c + \delta}
$$

**含义**: 将专家动作通过流逆向映射至初始噪声，再经 [[DCT]] 截断为前 $K=8$ 系数的低维频谱表示，并标准化。

**符号说明**:
- $\mathcal{F}^{-1}_\theta$: 冻结流模型的数值逆向积分（ODE 反向求解）
- $\zeta^*$: 推理时的状态-语言条件
- $\text{DCT}_K$: 离散余弦变换并截留前 $K$ 个时间频率系数
- $\mu_c, \sigma_c$: 数据集系数均值与标准差；$\delta$: 数值稳定常数

### 公式 5：[[BC|行为克隆初始化损失]]

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}\!\left[\bigl\|\mu_\psi(x) - z^*\bigr\|_2^2\right]
$$

**含义**: 频谱演员 $\pi_\psi$ 以恢复的专家频谱潜变量 $z^*$ 为监督，进行行为克隆预热训练。

**符号说明**:
- $\mu_\psi(x)$: 频谱演员对当前观测 $x$ 的确定性输出（均值）
- $z^*$: 专家演示对应的频谱潜变量目标

### 公式 6：[[SAC|混合离线-在线 SAC 目标]]

$$
\mathcal{L}_\pi = \mathcal{L}_{\text{SAC}}(\mathcal{D}_{\text{on}}) + \mathcal{L}_{\text{SAC}}(\mathcal{D}_{\text{off}}) + \lambda_{\text{BC}} \cdot \mathcal{L}_{\text{BC}}(\mathcal{D}_{\text{off}})
$$

**含义**: 在线数据用纯 SAC 梯度更新策略，离线数据同时施加 BC 正则防止偏离专家分布，实现安全的离线到在线过渡。

**符号说明**:
- $\mathcal{D}_{\text{on}}$: 在线采集的 replay buffer
- $\mathcal{D}_{\text{off}}$: 离线演示的频谱 replay buffer
- $\lambda_{\text{BC}}$: BC 正则化权重（仅作用于离线数据）
- $\mathcal{L}_{\text{SAC}}$: 标准 SAC 熵正则化策略梯度损失

### 公式 7：[[SAC|推理解码流程]]

$$
z \sim \pi_\psi(\cdot \mid x),\quad
c = \mu_c + (\sigma_c + \delta) \odot z,\quad
\epsilon = \text{IDCT}_K(c),\quad
A = \mathcal{F}_\theta(\epsilon \mid \zeta)
$$

**含义**: 从频谱演员采样→反标准化→逆 DCT（零填充截断系数）→经冻结流模型正向积分得到最终动作轨迹。

**符号说明**:
- $\text{IDCT}_K$: 逆离散余弦变换（截断系数补零到 $H$ 维后变换）
- $\mathcal{F}_\theta$: 冻结流模型正向积分（ODE 求解）

---

## 关键图表

### Figure 1：Overview — HAF 任务与系统概览

![Figure 1](https://grange007.github.io/HAF/static/images/paper/haf-overview.png)

**说明**: 展示 HAF 在七项真实人形机器人任务上的部署概况，以及 HAF-VLA 与 HAF-Steer 两组件的定位——前者负责分层整体动作生成，后者负责在线策略精化。

### Figure 2：HAF-VLA Pipeline — 分层动作流架构

![Figure 2](https://grange007.github.io/HAF/static/images/paper/haf-vla.png)

**说明**: 共享动作专家网络在三个顺序阶段中逐步扩展激活动作空间（locomotion/head → waist → bimanual manipulation）；Stage 1/2 的干净去噪动作被重编码为 [[KV Cache|键值对]] $C_t^1, C_t^2$，为后续阶段提供全局条件。

### Figure 3：HAF-Steer Pipeline — 频谱潜变量 RL

![Figure 3](https://grange007.github.io/HAF/static/images/paper/haf-steer.png)

**说明**: 展示流逆向恢复初始噪声 → [[DCT]] 截断为频谱潜变量 $z^*$ → 行为克隆初始化 → 混合离线/在线 [[SAC]] 训练 → 逆 DCT + 冻结 VLA 解码的完整流水线。

### Figure 4：数据采集遥操作设置

![Figure 4](https://arxiv.org/html/2608.16837v1/fig_teleoperation_setup.png)

**说明**: 同构遥操作系统：动捕主臂控制双臂（Bimanual），IMU 控制头部朝向，操纵杆控制腿部行走和腰部姿态。每任务采集约 120 条轨迹。

### Figure 5：七项真实世界任务序列

![Figure 5](https://arxiv.org/html/2608.16837v1/fig_task_progression.png)

**说明**: Laundry Loading、Clothes Retrieval、Table Tidy、Basket Transfer、Toy Storage、Ball Tossing、Box Transfer；每行展示一项任务的代表性执行步骤，均需导航+全身姿态调整+物体交互。

### Figure 6：泛化性实验

![Figure 6](https://grange007.github.io/HAF/static/images/paper/generalization.png)

**说明**: 左侧：Laundry Loading 中放置干扰椅的视觉干扰测试（40% vs π₀.₅ 26.7%）；右侧：Clothes Retrieval 中起始位置向后偏移 20cm 的位置干扰测试（43.3% vs 36.7%）。

### Figure 7：HAF-Steer 分布内/分布外评估设置

![Figure 7](https://arxiv.org/html/2608.16837v1/fig_haf_steer_id_ood_setups.png)

**说明**: 蓝色标记为演示数据覆盖的目标桌位置（分布内），红色标记为偏移 30cm 的未见位置（分布外），验证 HAF-Steer 的泛化能力。

---

### Table 1：主实验结果（7 任务，各 10 次 rollout）

| 任务 | HAF-VLA | π₀.₅ | GR00T N1.7 | Cosmos | ACT |
|------|---------|------|-----------|--------|-----|
| Laundry Loading | **66.7%** | 53.3% | 40.0% | 0.0% | 10.0% |
| Clothes Retrieval | **53.3%** | 53.3% | 33.3% | 26.7% | 23.3% |
| Table Tidy | **80.0%** | 70.0% | 40.0% | 16.7% | 23.3% |
| Basket Transfer | **63.3%** | 50.0% | 43.3% | 33.3% | 16.7% |
| Toy Storage | **80.0%** | 53.3% | 30.0% | 40.0% | 23.3% |
| Ball Tossing | **56.7%** | 33.3% | 36.7% | 3.3% | 30.0% |
| Box Transfer | **93.3%** | 60.0% | 43.3% | 73.3% | 50.0% |
| **平均** | **70.5%** | 53.3% | 38.1% | 27.6% | 25.2% |

**关键发现**: HAF-VLA 在所有任务均领先，平均超越 π₀.₅ 17.2pp，超越 GR00T 32.4pp；Box Transfer（需远距离导航后精密操作）达 93.3%。

### Table 2：消融实验（Laundry Loading，10 次 rollout）

| 方法 | 阶段设计 | 总去噪步数 | 成功率 |
|------|---------|---------|------|
| **HAF-VLA** | Locomotion/Head → Waist → Manip. | 30 | **66.7%** |
| All-Joint Denoising | 所有关节每阶段同时去噪 | 30 | 53.3% |
| Arm-First Hierarchy | Manip. → Waist → Locomotion/Head | 30 | 50.0% |
| π₀.₅（30步） | 全身单阶段动作 | 30 | 20.0% |

**关键发现**: 阶段顺序至关重要——反转层级（Arm-First）降至 50%，全关节同时去噪 53.3%，说明运动学依赖顺序是核心设计决策。

### Table 3：泛化干扰测试（10 次 rollout）

| 任务 | 干扰类型 | HAF-VLA | π₀.₅ |
|------|---------|---------|------|
| Laundry Loading | 视野内新增干扰椅 | **40.0%** | 26.7% |
| Clothes Retrieval | 起始位置向后偏 20cm | **43.3%** | 36.7% |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实遥操作数据 | ~120 条/任务 | 同构主臂+操纵杆+IMU | HAF-VLA 训练 & HAF-Steer 离线数据 |
| 在线探索数据 | 动态增长 | 真实机器人稀疏奖励 | HAF-Steer 在线训练 |

### 实现细节

- **机器人平台**: TienKung 2.0 / TienKung 3.0 人形机器人，配备头部自我中心 RGB 相机
- **推理硬件**: RTX 5090 GPU，单次三阶段推理约 0.12 s
- **动作块**: $H=100$（总视野），$H_{\text{exec}}=40$（每次推理执行步数）
- **去噪步数**: 每阶段 10 步，共 30 步（与 π₀.₅ 计算量相当）
- **DCT 系数**: $K=8$，将 $H \times D$ 压缩至 $K \times D$
- **RL 算法**: SAC + 双 Critic + 熵正则 + BC 正则
- **奖励**: 稀疏终端奖励（成功=1，失败=0）+ 里程碑归一化评分
- **Backbone**: 预训练 π₀.₅（冻结），仅 HAF-Steer 演员/Critic 参与训练

### 实验结果

- HAF-VLA 在 7 项任务全面领先（平均 70.5%），尤其在需要复杂行走+操作协调的任务（Box Transfer: 93.3%）优势最显著
- HAF-Steer 在分布内和分布外设置均进一步提升 π₀.₅ 和 HAF-VLA，而 [[DSRL]] 基线在真实机器人上因不安全探索被强制终止
- 泛化测试中 HAF-VLA 在视觉干扰和位置偏移下均保持领先优势

---

## 批判性思考

### 优点

1. **运动学先验的明确利用**: 三阶段顺序与人形机器人动力学依赖自然对齐，设计简洁而有效
2. **频谱子空间的安全探索**: DCT 截断将探索限制在低频平滑模态，避免高频抖动和实际危险动作
3. **无需微调主干**: HAF-Steer 冻结大模型，仅训练小型 RL 头部，参数效率高且计算代价低
4. **真实机器人验证充分**: 7 项任务、多种干扰测试、消融实验完整，说服力强

### 局限性

1. **推理延迟增加**: 三阶段串行去噪使单次推理时间较 π₀.₅ 增加约 3×（虽已在 RTX 5090 上压缩至 0.12s）
2. **依赖 Base VLA 先验**: 若 Base VLA 对极端未见场景存在偏见，HAF-Steer 的频谱子空间探索也难以纠正
3. **任务覆盖局限**: 仅测试七项家庭任务，户外/工业场景泛化性未验证
4. **奖励稀疏**: 纯稀疏终端奖励可能导致样本效率较低，信用分配困难

### 潜在改进方向

1. 引入高效去噪方案（如 DDIM/一致性模型）减少三阶段延迟
2. 设计更密集的过程奖励（milestones-based shaping）提升 RL 样本效率
3. 扩展到更多自由度（灵巧手、指尖触觉）和室外 [[Loco-Manipulation]] 场景

### 可复现性评估

- [ ] 代码开源（暂未发布）
- [ ] 预训练模型（未提供）
- [ ] 训练细节完整（较完整：数据量/阶段步数/DCT 参数均有描述）
- [ ] 数据集可获取（私有遥操作数据）

---

## 关联笔记

### 基于

- [[π₀.₅]]: HAF 的 Base VLA 主干，HAF-VLA 和 HAF-Steer 均在其上构建
- [[Flow-Matching]]: HAF 的动作生成核心机制，Eq.2-3 直接基于此
- [[SAC]]: HAF-Steer 的在线 RL 算法基础
- [[BC]]: HAF-Steer 的行为克隆初始化与 BC 正则

### 对比

- [[GR00T]]: 大型人形机器人基础模型（38.1% vs HAF 70.5%）
- [[DSRL]]: 同类潜变量 RL 基线，单重复噪声向量导致真实机器人不安全
- [[π₀.₅]]: 核心对比方法，HAF 在同计算量下提升 17.2pp

### 方法相关

- [[Loco-Manipulation]]: HAF 的目标任务类型
- [[Whole-Body Control]]: HAF-VLA 的核心设计动机
- [[KV Cache]]: 跨阶段条件化机制
- [[DCT]]: 频谱潜变量构建的关键变换
- [[Hierarchical Chunking]]: HAF-VLA 分层动作生成框架
- [[teacher forcing]]: HAF-VLA 训练时的缓存策略

### 硬件/数据相关

- [[TienKung]]: 部署平台（TienKung 2.0 & 3.0 人形机器人）

---

## 速查卡片

> [!summary] HAF (2026)
> - **核心**: 分层动作流（HAF-VLA）+ 频谱潜变量 RL（HAF-Steer）适配人形全身行走-操作
> - **方法**: 三阶段顺序去噪（locomotion→waist→manip）+ KV-Cache 跨阶段条件；DCT 频谱子空间 SAC 在线微调
> - **结果**: 7 任务平均 70.5%，超越 π₀.₅（53.3%）和 GR00T（38.1%）
> - **项目**: [grange007.github.io/HAF](https://grange007.github.io/HAF)

---

*笔记创建时间: 2026-08-19*

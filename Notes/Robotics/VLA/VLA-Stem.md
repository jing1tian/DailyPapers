---
title: "VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning"
method_name: "VLA-Stem"
authors: [Bojun Zou, Wenhao Yu, Jiaohao Li, Jiaxuan Lin, Guanting Ye, Haohan Geng, Weihao Yu, Yanyong Zhang, Jianmin Ji]
year: 2026
venue: arXiv
tags: [vla, safety, collision-avoidance, reinforcement-learning, teacher-student-distillation, curriculum-learning, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.37378v1
created: 2026-10-29
---

# 论文笔记：VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Science and Technology of China (USTC) |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[VLSA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.37378) / Code: — |

---

## 一句话总结

> VLA-Stem 是一个轻量级 RL 安全层，冻结 VLA 策略不变，通过独立的 Stem 策略将 [[Action Chunking|动作块]] 转化为运动参考并实时规避障碍，在六个受限仿真场景中碰撞规避成功率平均提升至 52.8%（最强基线 23.7%）。

---

## 核心贡献

1. **VLA 条件化执行（VLA-Conditioned Execution）**: 冻结 VLA 策略，将其输出的 [[Action Chunking|动作块]] 转换为相对末端执行器参考轨迹，驱动独立的 Stem 策略，无需微调 VLA 本身。
2. **轨迹级引导奖励（Trajectory-Level Guidance）**: 提出前向走廊匹配机制，用软空间约束（而非严格逐步对齐）衡量执行轨迹与参考轨迹的偏差，同时奖励轨迹进度，兼顾安全性与任务灵活性。
3. **教师-学生蒸馏（Teacher-Student Distillation）**: 教师策略以特权目标信息辅助训练，学生策略通过 [[PPO]] + KL 蒸馏目标在无特权信息下学习部署，显著提升泛化能力。

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）|VLA 模型]] 在无障碍环境中训练，部署时面对静态或动态障碍物时碰撞率极高（基础 VLA 如 [[π0]] / [[π0.5]] 在受限场景中碰撞率可达 85–97%）。如何在不修改 VLA 权重的前提下为其增加障碍规避能力？

### 现有方法的局限

- **VLA 微调**（VLA-finetune）：对特定障碍分布过拟合，任务泛化差（StackCube 静态场景成功率仅 0.62%）。
- **[[CBF]]-QP 安全滤波器**（[[VLSA]]）：基于控制障碍函数实时修正末端速度，但依赖精确几何建模，动态场景适应性弱，且与末端执行器控制接口耦合。
- **规划方法**（CHOMP、TrajOpt 等）：需要全局地图，难以处理实时动态障碍。

### 本文的动机

受脑干（Brainstem）调节高层运动指令同时保障基础生存安全的生物学启发：VLA 对应大脑高层运动皮层，Stem 策略对应脑干调节层，以低开销、即插即用的方式附加安全保障，同时保留 VLA 的任务执行能力。

---

## 方法详解

### 模型架构

[[VLA-Stem]] 采用**双策略解耦**架构：

- **输入（Stem 策略）**:
  - [[Proprioceptive State|本体感知状态]] $\mathbf{x}_t$（16 维：9 关节角 + 7 维末端执行器位姿）
  - 障碍物信息 $\mathbf{o}_t^{\mathrm{obs}}$（33 维：3 个障碍槽 × 11 特征/槽，含中心、尺寸、方向、存在标志）
  - VLA 参考轨迹 $\tau_c^{\mathrm{rel}}$（$h=8$ 帧相对末端位姿序列）
- **Backbone（Stem 策略）**: MLP 网络，无视觉输入
- **冻结部分**: VLA 策略（[[π0.5]] 或 [[π0]]）完全不参与训练
- **输出**: 7 维归一化末端执行器增量动作（经 tanh 映射到 $[-1,1]$）
- **训练方式**: [[课程学习]] 三阶段 [[PPO]] + [[On-Policy Distillation|KL 蒸馏]]

### 核心模块

#### 模块 1：VLA 动作块转参考轨迹

**设计动机**: 直接利用 VLA 的空间意图作为 Stem 的软约束，使安全规避不脱离任务目标。

**具体实现**:
- 每隔 $h=8$ 步，冻结 VLA 被调用一次，生成 [[Action Chunking|动作块]] $\mathcal{A}_c^{\mathrm{VLA}}$
- 将其转换为以区间起点为参考系的相对末端位姿序列 $\tau_c^{\mathrm{rel}} = \{\Delta T_{c,k}\}_{k=1}^{h}$
- Stem 策略在每步 $t$ 接收最新参考轨迹并输出修正动作

#### 模块 2：前向走廊匹配（Trajectory-Level Guidance）

**设计动机**: 逐步对齐（step-wise alignment）强制时序同步，灵活性差；轨迹级引导仅要求空间邻近，允许时间偏移，兼顾绕障与任务完成。

**具体实现**:
- 参考轨迹的第 $i$ 段由相邻参考点 $A_i, B_i$ 构成
- 计算当前末端位置到各段的投影距离（位置偏差 $d_i^{\mathrm{pos}}$ + 旋转偏差 $d_i^{\mathrm{rot}}$）
- 在前向走廊 $\mathcal{I}_t^+$（最多向前看 $K=6$ 段）内匹配最近段，记为 $i_t^\ast$
- 最佳匹配偏差 $d_t^\ast$ 用于计算跟随奖励；进度索引 $m_t = \max(m_{t-1}, i_t^\ast)$ 单调递增，防止倒退

#### 模块 3：教师-学生蒸馏

**设计动机**: 教师策略在训练时能看到目标物体位置（特权信息），大幅提升轨迹跟随质量；部署时学生策略无特权信息但通过 KL 蒸馏吸收教师的隐式知识。

**具体实现**:
- 教师输入：23 维（16 维 [[Proprioceptive State|本体感知]] + 7 维目标物体特权信息）
- 学生训练目标（Eq. 15–16）：$L_{\mathrm{PPO}} + \lambda L_{\mathrm{distill}}$，其中 $L_{\mathrm{distill}} = D_{\mathrm{KL}}(\pi_{\mathrm{stu}} \| \pi_{\mathrm{tea}})$，$\lambda=0.1$
- 消融实验（Table IV）显示去掉蒸馏后正常场景成功率从 70.6% 跌至 29.0%

#### 模块 4：三阶段[[课程学习]]

- **阶段 1**: 仅轨迹跟随，无障碍物，学习基本任务执行能力
- **阶段 2**: 引入静态障碍，学习规避
- **阶段 3**: 引入动态障碍，学习等待与时机判断

---

## 关键公式

### 公式 1：[[Action Chunking|VLA 动作块生成]]

$$
\mathcal{A}_c^{\mathrm{VLA}} = \pi_{\mathrm{VLA}}\left(\mathbf{o}_{t_c}^{\mathrm{VLA}}\right)
$$

**含义**: 冻结的 VLA 策略每 $h$ 步被调用一次，生成一段完整的动作块

**符号说明**:
- $\mathcal{A}_c^{\mathrm{VLA}}$: 第 $c$ 个区间的 VLA 动作块（$h$ 步）
- $t_c$: 第 $c$ 个区间的起始时间步
- $\mathbf{o}_{t_c}^{\mathrm{VLA}}$: VLA 的观测输入（视觉 + 语言）

### 公式 2：相对末端位姿参考轨迹

$$
\Delta T_{c,k} = \left(T_{t_c}^{w}\right)^{-1} T_{t_c+k}^{w}, \qquad \tau_c^{\mathrm{rel}} = \{\Delta T_{c,k}\}_{k=1}^{h}
$$

**含义**: 将 VLA 动作块转换为以区间起点为参考系的相对末端位姿序列，作为 Stem 策略的引导信号

**符号说明**:
- $T_{t}^{w}$: 世界坐标系下 $t$ 时刻的末端执行器齐次变换矩阵
- $\Delta T_{c,k}$: 相对于区间起点的第 $k$ 步相对位姿变换
- $h=8$: 动作块长度

### 公式 3：Stem 策略执行

$$
a_t^{\mathrm{exec}} = \pi_{\mathrm{stem}}\left(\mathbf{x}_t, \mathbf{o}_t^{\mathrm{obs}}, \tau_c^{\mathrm{rel}}\right), \qquad t_c \le t < t_c + h
$$

**含义**: Stem 策略在每步根据本体感知、障碍信息和参考轨迹输出执行动作

**符号说明**:
- $\mathbf{x}_t$: 16 维本体感知（关节角 + 末端位姿）
- $\mathbf{o}_t^{\mathrm{obs}}$: 33 维障碍物信息（3 槽 × 11 维）
- $\tau_c^{\mathrm{rel}}$: 当前区间 VLA 参考轨迹

### 公式 4：前向走廊定义

$$
\mathcal{I}_t^{+} = \{m_{t-1}, \ldots, \min(m_{t-1}+K, h-1)\}
$$

**含义**: 在单调进度索引前最多向前看 $K$ 段的搜索窗口，防止回退同时保留时序灵活性

**符号说明**:
- $m_{t-1}$: 上一步的进度索引（单调不减）
- $K=h-2=6$: 前向走廊宽度

### 公式 5：最优匹配与进度更新

$$
i_t^{\ast} = \arg\min_{i \in \mathcal{I}_t^{+}} d_i, \qquad d_t^{\ast} = d_{i_t^{\ast}}, \qquad m_t = \max(m_{t-1}, i_t^{\ast})
$$

**含义**: 在走廊内找到最近段，取其偏差为当前最优偏差，并更新单调进度

**符号说明**:
- $d_i = d_i^{\mathrm{pos}} + \alpha d_i^{\mathrm{rot}}$: 综合位置与旋转偏差（$\alpha = 4/7$）

### 公式 6：位置偏差（投影到线段）

$$
\lambda_i = \mathrm{clip}\!\left(\frac{(\mathbf{p}_t - A_i)^\top (B_i - A_i)}{\|B_i - A_i\|^2 + \epsilon},\ 0,\ 1\right), \quad \hat{\mathbf{p}}_i = A_i + \lambda_i (B_i - A_i)
$$

$$
d_i^{\mathrm{pos}} = \|\mathbf{p}_t - \hat{\mathbf{p}}_i\|
$$

**含义**: 将当前末端位置投影到参考线段上，计算点到线段的最短距离

**符号说明**:
- $A_i = \mathbf{p}_i^{\mathrm{ref}}, B_i = \mathbf{p}_{i+1}^{\mathrm{ref}}$: 参考线段两端点
- $\hat{\mathbf{p}}_i$: 当前位置在线段上的投影点

### 公式 7：旋转偏差（四元数插值）

$$
s_i = \mathrm{sign}\!\left(\langle q_i^{\mathrm{ref}},\, q_{i+1}^{\mathrm{ref}} \rangle\right), \quad \hat{q}_i = \frac{(1-\lambda_i) q_i^{\mathrm{ref}} + \lambda_i s_i q_{i+1}^{\mathrm{ref}}}{\|(1-\lambda_i) q_i^{\mathrm{ref}} + \lambda_i s_i q_{i+1}^{\mathrm{ref}}\|}
$$

$$
d_i^{\mathrm{rot}} = 2 \arccos\!\left(\mathrm{clip}\!\left(|\langle q_t,\, \hat{q}_i \rangle|,\ 0,\ 1\right)\right)
$$

**含义**: 对参考四元数进行 SLERP 近似插值后计算姿态角距离，处理四元数符号翻转问题

**符号说明**:
- $s_i$: 四元数符号修正系数（保证插值方向正确）
- $q_t$: 当前末端执行器四元数

### 公式 8：轨迹跟随奖励

$$
r_t^{\mathrm{follow}} = \exp(-10 \cdot \tilde{d}_t^{\ast}) = \exp(-7 d_t^{\ast})
$$

$$
r_t^{\mathrm{step}} = \frac{\max(0,\ m_t - m_{t-1})}{h-1}
$$

**含义**: 跟随奖励鼓励靠近参考轨迹；步进奖励额外奖励轨迹进度推进，防止原地停留

**符号说明**:
- $\tilde{d}_t^\ast = 0.7 d_t^\ast$: 加权后的综合偏差（$0.7 d^{\mathrm{pos}} + 0.4 d^{\mathrm{rot}}$）

### 公式 9：总奖励

$$
r_t = w_{\mathrm{task}} r_t^{\mathrm{task}} + w_{\mathrm{coll}} r_t^{\mathrm{coll}} + w_{\mathrm{guidance}} r_t^{\mathrm{guidance}}
$$

$$
r_t^{\mathrm{task}} = 0.4\, r_t^{\mathrm{grasp}} + r_t^{\mathrm{succ}}, \qquad r_t^{\mathrm{coll}} = -r_t^{\mathrm{viol}}
$$

**含义**: 三类奖励的加权组合：任务成功奖励、碰撞惩罚、引导奖励（跟随 + 步进），权重按课程阶段动态调整

**符号说明**:
- $r_t^{\mathrm{guidance}} = r_t^{\mathrm{follow}} + r_t^{\mathrm{step}}$: 引导奖励的两个子项

### 公式 10：教师-学生蒸馏目标

$$
L(\pi_{\mathrm{stu}}) = L_{\mathrm{PPO}}(\pi_{\mathrm{stu}};\, \mathcal{D}) + \lambda \cdot D_{\mathrm{KL}}(\pi_{\mathrm{stu}}(\cdot \mid o) \,\|\, \pi_{\mathrm{tea}}(\cdot \mid o, o_{\mathrm{extra}})), \quad \lambda=0.1
$$

**含义**: 在 [[PPO]] 强化学习目标基础上，加入 KL 散度约束使学生策略动作分布向教师靠拢

**符号说明**:
- $o_{\mathrm{extra}}$: 教师专有的特权信息（目标物体位置等）
- $\lambda=0.1$: KL 蒸馏权重

---

## 关键图表

### Figure 1：生物学动机与系统定位

![Figure 1](https://arxiv.org/html/2610.37378v1/first.png)

**说明**: 仿脑干（Brainstem）调节机制：[[VLA（视觉-语言-动作模型）|VLA 策略]] 类比大脑高层运动皮层负责任务规划，[[VLA-Stem]] 类比脑干实时调节运动指令，在遇到障碍时修正执行路径而不改变任务目标。

### Figure 2：VLA-Stem 整体流水线

![Figure 2](https://arxiv.org/html/2610.37378v1/overview.png)

**说明**: 左侧为三阶段[[课程学习]]流程（轨迹跟随 → 静态障碍 → 动态障碍）训练教师策略，右侧为蒸馏后的学生策略（部署时无特权信息）。VLA 策略完全冻结，Stem 策略接收本体感知、障碍信息和 VLA 参考轨迹作为输入。

### Figure 3：动态障碍规避（仿真演示）

![Figure 3](https://arxiv.org/html/2610.37378v1/figure/sm_sim.png)

**说明**: 对比 [[π0.5]] 直接执行（碰撞障碍物）与 VLA-Stem 的表现：VLA-Stem 识别到动态障碍物挡路后主动等待其通过，随后继续执行任务，体现了时序灵活性。

### Figure 4：参考轨迹引导可视化

![Figure 4](https://arxiv.org/html/2610.37378v1/figure/trajectory_visual.png)

**说明**: (a) StackCube 任务的参考轨迹可视化，Stem 执行轨迹在障碍规避后回归参考轨迹；(b) 参考轨迹干预实验定量结果（30 对试验/方向），VLA-Stem 在 90/90 次试验中成功跟随被扰动参考，验证了参考可控性。

### Figure 5：真实机器人实验（UR7e）

![Figure 5](https://arxiv.org/html/2610.37378v1/real_stem.png)

**说明**: (a) 真实场景示例（Object Stowing / Stacking + 障碍物）；(b)(c) 定量结果（每设置 15 次试验）：碰撞率从 73–100% 降至 20–40%，成功率从 0–20% 升至 47–60%，Sim-to-Real 零样本迁移有效。

### Table I：主实验结果（仿真，6 个受限场景）

| 方法 | PickCube Normal SUCC | PickCube Static SUCC↑/COLL↓ | PickCube Dynamic SUCC↑/COLL↓ | Object Stowing Static SUCC↑/COLL↓ | Object Stowing Dynamic SUCC↑/COLL↓ | StackCube Static SUCC↑/COLL↓ | StackCube Dynamic SUCC↑/COLL↓ |
|------|----------------------|------------------------------|-------------------------------|------------------------------------|------------------------------------|-------------------------------|-------------------------------|
| [[π0]] | 0.6884 | 0.0106 / 0.9422 | 0.0044 / 0.9703 | 0.0006 / 0.5363 | 0.0559 / 0.6809 | 0.1316 / 0.6478 | 0.1453 / 0.5484 |
| [[π0.5]] | 0.7166 | 0.0325 / 0.8781 | 0.0131 / 0.9328 | 0.0000 / 0.5897 | 0.0256 / 0.6156 | 0.1725 / 0.7319 | 0.1247 / 0.7012 |
| VLA-finetune | 0.7166 | 0.2847 / 0.4413 | 0.1869 / 0.5156 | 0.0072 / 0.2159 | 0.4784 / 0.2331 | 0.0062 / 0.7469 | 0.0641 / 0.6072 |
| [[VLSA]]-EE | 0.7166 | 0.1619 / 0.2303 | 0.0959 / 0.2847 | 0.3503 / 0.2609 | 0.1678 / 0.2663 | 0.2372 / 0.4509 | 0.4078 / 0.2244 |
| [[VLSA]]-ARM | 0.7166 | 0.1516 / 0.2253 | 0.0856 / 0.2903 | 0.3506 / 0.2644 | 0.1716 / 0.2578 | 0.2334 / 0.3753 | 0.4038 / 0.1628 |
| **VLA-Stem** | **0.7325** | **0.5219 / 0.2316** | **0.4153 / 0.3122** | **0.5784 / 0.1756** | **0.5950 / 0.1203** | **0.5141 / 0.3353** | **0.5419 / 0.0784** |

**关键发现**: VLA-Stem 在所有受限场景中 SUCC 显著领先（静态均值 52.8% vs [[VLSA]] 的 25.5%），正常场景略优于 [[π0.5]] base，证明安全层不损害原始任务能力。

### Table II：VLA 参考消融（PickCube）

| 配置 | Normal SUCC | Static SUCC / COLL | Dynamic SUCC / COLL |
|------|-------------|--------------------|---------------------|
| VLA-Stem w/o 参考 | 0.0000 | 0.0000 / 0.1300 | 0.0000 / 0.0156 |
| **VLA-Stem** | **0.7325** | **0.5219 / 0.2316** | **0.4153 / 0.3122** |

**关键发现**: 去掉 VLA 参考与引导奖励后，学生策略完全无法完成任务（0%），证明 VLA 引导是任务成功的必要条件。

### Table III：轨迹级 vs 逐步引导消融（Object Stowing）

| 配置 | Normal SUCC | Static SUCC / COLL | Dynamic SUCC / COLL |
|------|-------------|--------------------|---------------------|
| Step-wise 对齐 | 0.2888 | 0.4778 / 0.0366 | 0.1453 / 0.2194 |
| **轨迹级引导** | **0.7061** | **0.5784 / 0.1756** | **0.5950 / 0.1203** |

**关键发现**: 逐步对齐强制时序同步，灵活性差，正常场景成功率仅 28.9%；轨迹级引导通过放松时序约束实现 70.6%，提升 2.4×。

### Table IV：教师-学生蒸馏消融（Object Stowing）

| 配置 | Normal SUCC | Static SUCC / COLL | Dynamic SUCC / COLL |
|------|-------------|--------------------|---------------------|
| VLA-Stem w/o 蒸馏 | 0.2903 | 0.0525 / 0.3066 | 0.0900 / 0.4419 |
| **VLA-Stem** | **0.7061** | **0.5784 / 0.1756** | **0.5950 / 0.1203** |

**关键发现**: 去掉教师蒸馏后性能大幅下降（静态 5.2% vs 57.8%），且碰撞率反而升高，说明教师特权信息对学生安全策略的形成至关重要。

---

## 实验

### 数据集与环境

| 环境 | 场景 | 说明 |
|------|------|------|
| [[ManiSkill3]] PickCube | Normal / Static / Dynamic | 抓取方块，每设置 3,200 Episode |
| [[ManiSkill3]] Object Stowing | Normal / Static / Dynamic | 将物体放入容器 |
| [[ManiSkill3]] StackCube | Normal / Static / Dynamic | 叠放方块 |
| 真实 UR7e 机械臂 | Object Stowing / Stacking | 每设置 15 次试验 |

### 实现细节

- **VLA 基础模型**: [[π0.5]]（默认）、[[π0]]（对比）
- **Stem 策略网络**: MLP，输入 16+33+64（参考轨迹编码）维
- **动作块长度**: $h=8$
- **优化器**: [[PPO]]，$\lambda_{\mathrm{KL}}=0.1$
- **前向走廊宽度**: $K=6$
- **障碍物槽数**: $M=3$（最多同时跟踪 3 个障碍）
- **课程阶段**: 跟随 → 静态障碍 → 动态障碍

### 可视化结果

- 动态场景中 VLA-Stem 能主动等待移动障碍物通过，再继续执行任务（类似人类反应）
- 参考轨迹干预实验（90/90 = 100% 跟随成功率）验证了 Stem 的可控性，为上层安全监督打开了接口
- 真实机器人结果展示 Sim-to-Real 零样本迁移有效性

---

## 批判性思考

### 优点

1. **即插即用**: 冻结 VLA 不修改权重，适配任意 VLA 架构
2. **训练高效**: Stem 策略仅有低维状态输入，[[课程学习]] + [[PPO]] 收敛快
3. **轨迹可控性**: 参考轨迹干预实验证明 Stem 的执行轨迹可被外部引导，为更高层安全监督提供接口

### 局限性

1. **障碍表示受限**: 当前仅使用低维几何特征（33 维），无法处理复杂形状或密集障碍场景
2. **任务范围有限**: 仅评估结构化桌面操作，长程任务、接触力安全未覆盖
3. **视觉感知缺失**: Stem 策略无视觉输入，障碍检测依赖外部感知模块（论文未深入讨论）

### 潜在改进方向

1. 引入接触力/扭矩安全约束，扩展到力控任务
2. 用视觉特征替代几何障碍表示，提升真实场景适应性
3. 探索多 VLA 组合下的 Stem 策略共享与迁移

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（附录详细描述了奖励函数、网络架构、超参数）
- [x] 数据集可获取（[[ManiSkill3]] 开源仿真器）

---

## 关联笔记

### 基于

- [[π0.5]]: 主要 VLA 基础模型（被 Stem 策略包装）
- [[π0]]: 另一 VLA 基础模型（消融对比）
- [[PPO]]: Stem 策略的强化学习训练算法
- [[课程学习]]: 三阶段训练策略

### 对比

- [[VLSA]]: 最强基线，基于 [[CBF]]-QP 实时安全滤波器
- VLA-finetune: 直接微调 VLA 策略以适应有障碍场景

### 方法相关

- [[Action Chunking]]: VLA 输出的动作块，VLA-Stem 的核心输入形式
- [[CBF]]: VLSA 基线使用的控制障碍函数方法
- [[On-Policy Distillation]]: 教师-学生蒸馏的实现框架
- [[Proprioceptive State]]: Stem 策略的输入之一

### 硬件/数据相关

- [[ManiSkill3]]: 所有仿真实验的平台

---

## 速查卡片

> [!summary] VLA-Stem (arXiv 2610.37378)
> - **核心**: 冻结 VLA + 轻量级 RL Stem 策略实现即插即用安全规避
> - **方法**: VLA 动作块 → 相对参考轨迹 → Stem 策略（PPO + 课程学习 + KL 蒸馏）
> - **结果**: 受限场景平均 SUCC 52.8% vs 最强基线 23.7%；真实机器人碰撞率 20–40%（基线 73–100%）
> - **代码**: 未开源

---

*笔记创建时间: 2026-10-29*

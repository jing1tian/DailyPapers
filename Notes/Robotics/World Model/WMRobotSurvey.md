---
title: "World Model for Robot Learning: A Comprehensive Survey"
method_name: "WMRobotSurvey"
authors: [Bohan Hou, Gen Li, Jindou Jia, Tuo An, Xinying Guo, Sicong Leng, Haoran Geng, Yanjie Ze, Tatsuya Harada, Philip Torr, Oier Mees, Marc Pollefeys, Zhuang Liu, Jiajun Wu, Pieter Abbeel, Jitendra Malik, Yilun Du, Jianfei Yang]
year: 2026
venue: arXiv
tags: [world-model, survey, robot-learning, vla, video-generation, reinforcement-learning, embodied-ai, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.00080
created: 2026-05-18
---

# 论文笔记：World Model for Robot Learning: A Comprehensive Survey

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Nanyang Technological University, UC Berkeley, Stanford, University of Tokyo, Oxford, Microsoft, ETH Zurich, Princeton, Harvard |
| 日期 | April 2026 |
| 项目主页 | [https://ntumars.github.io/wm-robot-survey/](https://ntumars.github.io/wm-robot-survey/) |
| 对比基线 | [[WAMSurvey]], [[World Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.00080) / [Code](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy) |

---

## 一句话总结

> 从策略集成、仿真器使用、视频生成三大视角系统梳理机器人学习中的世界模型，提出统一的概率框架揭示各范式的内在联系，覆盖 43 页、100+ 代表性方法。

---

## 核心贡献

1. **统一概率框架**: 将策略模型、被动世界模型、可控世界模型、逆动力学模型统一为同一联合分布的不同边缘或条件形式，揭示各范式深层联系
2. **细粒度架构分类法**: 区分 IDM 解耦式、单骨干统一式、MoE/MoT 专家式、Unified VLA 式、隐空间 WM 式五大策略范式，并独立梳理世界模型作为仿真器和视频生成器两条主线
3. **综合资源库**: 配套 GitHub 持续维护新兴工作，涵盖数据集、基准、评估协议的完整综述

---

## 问题背景

### 要解决的问题

[[World Model|世界模型]] 在机器人学习中如何发挥作用？现有文献碎片化，缺乏统一框架理解世界模型与 [[VLA|Vision-Language-Action Policy]] 的关系，以及世界模型在策略学习、强化学习仿真器、视频生成三大应用场景中的角色。

### 现有方法的局限

- **概念碎片化**: 策略模型、视频生成模型、逆动力学模型被视为独立方法，忽视其统一的概率根源
- **综述缺位**: 缺乏从策略-仿真器-视频生成三维度系统梳理世界模型的综述
- **评估标准不统一**: 生成质量指标与下游任务成功率之间存在根本性解耦

### 本文的动机

世界模型的复兴——从认知科学、控制论到现代机器学习——为具身 AI 的核心三大能力提供基础：**前瞻性推理**（foresight）、**想象驱动规划**（imagination-driven planning）、**数据扩增**（data amplification）。

---

## 方法详解

### 统一概率框架

[[World Model|世界模型综述]] 的核心洞见：将机器人学习中的各类模型统一为同一联合分布的不同查询方式：

$$
p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l)
$$

由此得到四种相关形式：

- **策略模型**: $p(a_{t+1:t+k} | o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \, do$
- **被动世界模型**: $p(o_{t+1:t+k} | o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, l) \, da$
- **可控世界模型**: $p(o_{t+1:t+k} | o_t, a_{t+1:t+k})$
- **[[Inverse Dynamics Model|逆动力学模型]]**: $p(a_{t+1:t+k} | o_{t:t+k})$

> "策略模型、被动世界模型、可控世界模型与逆动力学模型并非完全独立的抽象；它们对应于对同一理想联合分布的不同查询或分解方式。"

### 视频生成世界模型

[[Action-Conditioned World Model|动作条件化视频生成模型]] 的一般形式：

$$
p(v_{t+1:t+H} | o_t, a_{t:t+H-1}, l)
$$

其中 $v_{t+1:t+H}$ 为未来视频帧，$o_t$ 为当前观测，$a_{t:t+H-1}$ 为动作序列，$l$ 为语言指令。

机器人视频生成有别于通用视频合成：生成的未来必须满足**时序一致性**、**动作忠实性**、**物理可信度**和**下游决策可用性**四重要求。

---

## 关键公式

### 公式1: [[World Model|机器人学习联合分布（世界模型统一框架）]]

$$
p(o_{t+1:t+H} | x_t, a_{t:t+H-1}, l)
$$

**含义**: 世界模型的功能性定义——给定当前状态 $x_t$、动作序列和语言指令，预测未来 H 步的观测分布

**符号说明**:
- $x_t$: 环境状态（在视觉观测空间中为 $o_t$）
- $a_{t:t+H-1}$: 动作序列
- $l$: 语言指令
- $H$: 预测时域

---

### 公式2: [[Inverse Dynamics Model|IDM 解耦策略（两阶段流水线）]]

**阶段一——未来预测:**

$$
\hat{\mathbf{o}}_{t+1:t+H} = \mathcal{W}(o_t, l)
$$

或潜在空间形式:

$$
\hat{\mathbf{z}}_{t+1:t+H} = \mathcal{W}(E_{\text{img}}(o_t),\, E_{\text{text}}(l))
$$

**阶段二——动作恢复:**

$$
\pi(a_{t+1:t+H'} | o_t, l) = P\!\left(a_t \mid E_{\text{img}}(o_t),\, E_{\text{text}}(l),\, \Phi(\hat{\mathbf{o}}_{t+1:t+H})\right)
$$

**含义**: 先让世界模型生成未来观测（或潜在表示），再由逆动力学模型从中恢复动作

**符号说明**:
- $\mathcal{W}$: 世界模型
- $E_{\text{img}}, E_{\text{text}}$: 图像/文本编码器
- $\Phi$: 未来观测的聚合函数

---

### 公式3: [[Video Policy|单骨干统一策略目标（联合去噪）]]

$$
\hat{y} = f_\theta(\tilde{\mathbf{x}}_\tau,\, o_t,\, l,\, \tau), \quad \mathbf{x} = [z^v;\, z^a]
$$

$$
\mathcal{L}_{\text{unified}} = \mathbb{E}\left[\ell(\hat{y},\, y)\right]
$$

**含义**: 统一骨干同时对视觉 token $z^v$ 和动作 token $z^a$ 进行联合去噪，世界模型不再是独立上游模块

**符号说明**:
- $\tilde{\mathbf{x}}_\tau$: 第 $\tau$ 去噪步的噪声输入
- $z^v, z^a$: 视觉和动作的潜在表示
- $\tau$: 去噪时间步

---

### 公式4: [[MoE|MoE/MoT 专家耦合（专家交互）]]

$$
(\mathbf{h}^v_{\ell+1},\, \mathbf{h}^a_{\ell+1}) = \mathcal{F}^{\text{mix}}_\ell(\mathbf{h}^v_\ell,\, \mathbf{h}^a_\ell;\, o_t,\, l)
$$

**含义**: 视频专家和动作专家在第 $\ell$ 层通过混合注意力函数交换信息，保留模态专用结构同时实现联合优化

**符号说明**:
- $\mathbf{h}^v_\ell, \mathbf{h}^a_\ell$: 第 $\ell$ 层视频/动作专家隐状态
- $\mathcal{F}^{\text{mix}}_\ell$: 跨专家混合注意力模块

---

### 公式5: [[Reinforcement Learning|RL 仿真器（世界模型转换采样）]]

$$
(\hat{o}_{t+1},\, \hat{r}_t,\, \hat{d}_t) \sim p_\phi(\cdot \mid o_{\leq t},\, a_{\leq t},\, l)
$$

$$
J(\theta) = \mathbb{E}_{\hat{\tau} \sim (\pi_\theta, p_\phi)}\!\left[\sum_t \gamma^t \hat{r}_t\right]
$$

**含义**: 世界模型 $p_\phi$ 作为学习型仿真器，生成转换样本供策略 $\pi_\theta$ 在想象中优化

**符号说明**:
- $\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t$: 想象的下一观测、奖励、终止信号
- $p_\phi$: 参数为 $\phi$ 的世界模型
- $\gamma$: 折扣因子

---

### 公式6: [[Reinforcement Learning|GRPO 式策略损失]]

$$
\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\hat{A}_t\right)\right]
$$

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

**含义**: GRPO/PPO 风格的裁剪代理目标，用于在世界模型仿真器中稳定训练策略

**符号说明**:
- $r_t(\theta)$: 新旧策略的概率比
- $\hat{A}_t$: 优势估计
- $\varepsilon$: 裁剪阈值

---

### 公式7: [[World Model|策略-世界模型协同进化]]

$$
\phi^{k+1} \leftarrow \text{UpdateWM}\!\left(\phi^k,\; D_{\text{real}} \cup D_{\text{policy}}(\pi_{\theta^k})\right)
$$

$$
\theta^{k+1} \leftarrow \text{UpdatePolicy}\!\left(\theta^k,\; \hat{D}(\phi^{k+1})\right)
$$

**含义**: 策略与世界模型交替更新——策略新探索的数据用于改进世界模型，改进后的世界模型再用于提升策略

---

## 关键图表

### Figure 1: 综述组织结构概览

![Figure 1](https://arxiv.org/html/2605.00080v1/x2.png)

**说明**: 三大主线视角——策略集成（第 3 节）、仿真器应用（第 4 节）、视频生成（第 5 节）——的组织结构图。

---

### Figure 2: 时序演化双轨

![Figure 2](https://arxiv.org/html/2605.00080v1/x3.png)

**说明**: （上轨）从解耦的视频生成+IDM 向统一架构演化；（下轨）从验证用途向学习型 RL 与协同进化演化。

---

### Figure 3: 三大架构范式对比

![Figure 3](https://arxiv.org/html/2605.00080v1/x4.png)

**说明**:
- (a) **IDM 式**: 独立世界模型 → 逆动力学，两阶段解耦
- (b) **单骨干式**: 视觉 token 与动作 token 联合去噪，完全耦合
- (c) **MoT 式**: 专用专家通过跨注意力交互，保留模态结构

---

### Figure 4: MLLM 内化路线

![Figure 4](https://arxiv.org/html/2605.00080v1/x5.png)

**说明**:
- (a) **显式未来图像预测**: 在统一 VLA 内预测未来帧（GR-1、WorldVLA 等）
- (b) **隐空间世界建模**: 无需像素解码，直接预测潜在目标（FLARE、VLA-JEPA 等）

---

### Figure 5: 世界模型双重用途

![Figure 5](https://arxiv.org/html/2605.00080v1/x6.png)

**说明**:
- (a) **RL 仿真器**: 生成想象转换 $(\hat{o}_{t+1}, \hat{r}_t)$，供策略在想象中训练
- (b) **评估器**: 对候选动作进行预测后排序/筛选，支持决策时规划

---

### Figure 6: 机器人视频世界模型统一进化视图

![Figure 6](https://arxiv.org/html/2605.00080v1/x7.png)

**说明**: 四阶段能力演化：
- **Stage 1 - 想象驱动**: 预测未来用于监督（UniPi、Dreamitate）
- **Stage 2 - 动作可控**: 严格的动作条件化（IRASim、EnerVerse-AC）
- **Stage 3 - 结构感知**: 引入物理/交互结构（TesserAct、RoboVIP）
- **Stage 4 - 基础模型规模**: 大规模视频骨干（Cosmos Predict 2.5、GigaWorld-0）

---

### Table 1: 世界模型策略范式综合对比

| 范式类别 | 代表方法 | 推理时生成未来 | 骨干类型 | 耦合方式 |
|---------|---------|------------|---------|---------|
| **IDM 式** | UniPi, VidMan, Vidar, TC-IDM | 显式视频/潜在 | VGM | 解耦 |
| **单骨干** | UVA, UWA, VideoVLA, Cosmos Policy | 联合潜在/视频 | VGM | 共享骨干 |
| **MoE/MoT** | GE-Act, Motus, LingBot-VA, DiT4DiT, LDA-1B | 潜在/专家回滚 | VGM | 专家融合 |
| **Unified VLA** | GR-1, UP-VLA, WorldVLA, DreamVLA, F1, HALO | 显式/隐式未来 | MLLM | MLLM 内化 |
| **隐空间 WM** | FLARE, VLA-JEPA, WoG, DIAL | 潜在对齐（无像素） | 多样 | 特征对齐 |

---

### Table 2: 机器人视频生成方法对比

| 方法 | 任务条件 | 动作条件 | 结构感知 | 基础模型规模 | 主要用途 |
|------|---------|---------|---------|------------|---------|
| UniPi | ✓ | - | - | ✓ | 规划 |
| IRASim | - | ✓ | - | - | 规划 |
| RoboMaster | - | ✓ | ✓ | - | 数据 |
| Ctrl-World | - | ✓ | ✓ | - | 评估 |
| TesserAct | ✓ | - | ✓ | - | 监督 |
| Mask2IV | ✓ | - | ✓ | - | 数据 |
| Vid2World | - | ✓ | - | ✓ | 仿真 |
| Genie Envisioner | ✓ | ✓ | - | ✓ | 仿真 |
| Cosmos Predict 2.5 | ✓ | - | - | ✓ | 仿真 |
| GigaWorld-0 | - | - | ✓ | ✓ | 数据 |
| ABot-PhysWorld | - | ✓ | - | ✓ | 仿真 |

---

## 实验

### 评估体系：三维世界建模能力

| 维度 | 指标 | 说明 |
|------|------|------|
| **视觉保真度** | [[PSNR]], [[SSIM]], [[LPIPS]], [[DreamSim]] | 像素级→感知级→语义级 |
| | [[FVD]] | 分布级视频逼真度 |
| **物理常识** | VideoPhy, PhyGenBench, VBench-2.0 | 物理交互一致性 |
| | WorldModelBench, Physics-IQ | 物理定律遵守 |
| | EWMBench (HSD/nDTW/DYN) | 运动轨迹正确性 |
| **动作可信度** | WorldSimBench | 隐式操控评估 |
| | Wow, wo, val! | IDM 图灵测试 + 真实执行成功率 |

### 主流动作策略评估基准

| 基准 | 年份 | 任务数 | 仿真平台 | 评估重点 |
|------|------|--------|----------|----------|
| MetaWorld | 2019 | 50 | MuJoCo | 多任务 |
| RLBench | 2020 | 100 | CoppeliaSim | 泛化 |
| LIBERO | 2023 | 130 | MuJoCo | 语言条件 |
| RoboCasa | 2024 | 100 | MuJoCo | 数据规模/多任务 |
| SimplerEnv | 2024 | — | SAPIEN | 仿真-真实迁移 |
| RoboTwin | 2025 | 40 | SAPIEN/ManiSkill3 | 长时/双臂/灵巧 |
| ManiSkill3 | 2025 | 62 | SAPIEN | 数据扩展 |
| RoboVerse | 2025 | 276 | MetaSim | 仿真到真实 |

### 机器人遥操作训练数据集

| 名称 | 年份 | 规模 | 机器人数 |
|------|------|------|---------|
| OXE | 2023 | 1M+ 轨迹 | 22 |
| DROID | 2024 | 76K 轨迹 | Franka |
| ARIO | 2024 | 3M+ 轨迹 | 35 |
| AgiBot World | 2025 | — | AgiBot |

---

## Section 8: 挑战与未来方向

### 8.1 因果条件差距

当前 VLA 框架中，世界模型的未来预测往往更依赖历史上下文和任务意图，而非具体的待执行动作。预测的未来可能在语义上合理，但并不忠实于候选动作的物理后果，限制了闭环控制中精确动作条件化的能力。

**解决方向**: WorldVLA 的隐式统一训练策略将未来状态预测与动作生成耦合，鼓励策略对齐的预测动力学。

### 8.2 效率瓶颈

世界模型策略的计算开销远高于普通 VLA，尤其是训练和推理阶段：
- 扩散视频预测的迭代去噪导致高延迟
- 大模型参数量和复杂环境动力学使自适应代价高

**解决方向**:
- MimicVideo、LingBot-VA：部分去噪（仅捕捉运动动力学，跳过细节）
- LeWorldModel：聚焦预测表示而非全维度生成的[[World Model|潜空间模型]]
- Fast-WAM：训练时用世界模型增强表示，推理时完全省略

### 8.3 多模态感知瓶颈

当前世界模型擅长视觉合成，但与真实物理交互脱节：
- 仅视觉+本体感知无法捕捉摩擦力、刚度、接触稳定性等不可观测属性
- 触觉高频瞬态信号与低维特征易被高维视觉淹没

**解决方向**: 视觉-触觉联合潜表示（Higuera et al., 2026; Zheng et al., 2026）；谨慎平衡异步异频信号的融合。

### 8.4 经典控制集成

世界模型可作为 [[Model Predictive Control|MPC]] 的前向动力学，但代价巨大（需对每步动作优化进行迭代 rollout），限制高容量模型的实时部署。

**解决方向**: 将神经网络动力学与 Lyapunov 稳定性等形式化控制保证结合，实现自适应开放世界系统。

### 8.5 符号结构集成

像素 rollout 的长时误差累积问题可通过符号/结构化世界模型缓解：
- 操作对象、关系、谓词、可供性等结构化状态可更稳定地建模离散/规则转换
- 但需要合适的抽象层和感知接地，高维观测难以干净映射到预定义符号

**解决方向**: 混合世界模型——结合学习的感知表示与符号约束生成模型（Liang et al., 2026）。

### 8.6 评估指标开放挑战

缺乏广泛认可的评估指标：视觉保真度高不等于控制效用高；指标碎片化导致跨基准比较困难。

**解决方向**: 面向功能的评估框架，联合评估预测真实性、动作敏感性、长时一致性和控制效用，形成紧凑的标准化指标集（任务成功率、策略排名保真度、可执行性诊断）。

---

### LIBERO 基准代表性结果

| 组别 | 方法 | Spatial | Object | Goal | Long | **Avg** |
|------|------|---------|--------|------|------|---------|
| 解耦 | MimicVideo | 94.2 | 96.8 | 90.6 | 94.0 | 93.9 |
| | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | **98.1** |
| 单骨干 | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | **98.5** |
| | UD-VLA | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| MoE/MoT | Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| 统一VLA | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| | CoWVLA | 97.2 | 97.8 | 94.6 | 92.8 | 95.6 |
| | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| | TriVLA | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |
| 隐空间WM | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

**关键发现**: 各范式均能实现竞争力性能；Long 套件是最大分水岭（TriVLA 仅 73.2%）。

### RoboTwin / CALVIN / SIMPLER 结果

| 组别 | 方法 | RT-A | RT-B | C-A | C-D | S-G | S-W |
|------|------|------|------|-----|-----|-----|-----|
| 解耦 | Vidar | 65.8 | 17.5 | – | – | – | – |
| | Video2Act | 54.6 | 54.1 | – | – | – | – |
| 单骨干 | VideoVLA | – | – | – | – | 73.1 | 53.1 |
| MoE/MoT | Motus | 88.7 | 87.0 | – | – | – | – |
| | LingBot-VA | **92.9** | **91.6** | – | – | – | – |
| | BagelVLA | 75.3 | 20.9 | 4.41 | – | – | – |
| 统一VLA | GR-1 | – | – | 3.06 | 4.21 | – | – |
| | DreamVLA | – | – | 4.44 | – | – | – |
| | CoWVLA | – | – | 4.21 | 4.47 | 60.9 | 76.0 |
| | InternVLA-A1 | 89.4 | 87.0 | – | – | – | – |
| 隐空间WM | VLA-JEPA | – | – | – | – | 65.2 | 57.3 |
| | WoG | – | – | – | – | 69.4 | 63.5 |

*RT-A/B: RoboTwin 简单/随机；C-A/C-D: CALVIN ABCD/ABCDD；S-G/S-W: SIMPLER Google/WidowX*

---

## 批判性思考

### 优点
1. **统一概率视角**: 通过联合分布统一策略/世界模型/IDM，为领域提供清晰的概念框架
2. **策略-仿真器-视频三维综述**: 三条主线独立深入且相互关联，覆盖面远超既有综述
3. **四阶段视频演化图**: Figure 6 的能力阶段划分简洁有力，有利于定位具体工作

### 局限性
1. **定量对比缺位**: Table 5/6 中大量 "–" 说明各方法评估协议不统一，限制直接比较
2. **WAM 综述视角差异**: 与同期 WAMSurvey（arXiv:2605.12090）存在范围重叠，术语体系不同（本文不采用"WAM"命名），读者需对照阅读
3. **应用综述分量不均**: 导航（6.1）和自动驾驶（6.2）相对简略，仅几页

### 潜在改进方向
1. 建立联合评估指标（Counterfactual Consistency、Foresight-Conditioned Success）
2. 多模态物理状态预测（视觉+触觉+力）的统一世界模型
3. 在等价数据和计算量条件下系统比较 Cascaded vs. Joint vs. 隐空间 WM 范式

### 可复现性评估
- [x] GitHub 仓库持续维护（https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy）
- [ ] 预训练模型（综述论文，无模型）
- [x] 训练细节完整（引用各方法原文）
- [x] 数据集可获取（引用公开数据集）

---

## 关联笔记

### 基于
- [[World Model]]: 综述的核心主题
- [[VLA|Vision-Language-Action Model]]: 与世界模型集成的策略骨干
- [[Inverse Dynamics Model]]: IDM 式解耦策略的核心组件
- [[Action-Conditioned World Model]]: 动作可控视频生成的核心范式

### 对比
- [[WAMSurvey]]: Fudan 团队同期综述，采用 WAM 命名，更侧重 Cascaded/Joint 分类，两篇互补
- [[World Model]]: 通用世界模型概念笔记

### 方法相关
- [[Video Policy]]: IDM 式策略的视频生成骨干
- [[JEPA|Joint-Embedding Predictive Architecture]]: 隐空间 WM（VLA-JEPA、FLARE）的理论基础
- [[Diffusion Model|扩散模型]]: 单骨干统一策略和 MoT 式策略的主流骨干
- [[MoE|Mixture of Experts]]: MoE/MoT 专家耦合策略的架构基础
- [[Reinforcement Learning]]: 世界模型作为 RL 仿真器的应用主线

### 硬件/数据相关
- [[LIBERO]]: 最常用的世界模型策略评估基准
- [[Action Chunking]]: 策略输出的标准动作表示

---

## 速查卡片

> [!summary] World Model for Robot Learning Survey (NTU/Berkeley, 2026)
> - **核心**: 以统一概率框架 $p(o_{t+1:t+k}, a_{t+1:t+k}|o_t,l)$ 串联所有世界模型范式
> - **三轴**: 策略集成（IDM/单骨干/MoT/Unified VLA/隐空间）、RL 仿真器（单级/协同进化）、视频生成（想象→可控→结构感知→基础模型）
> - **评估**: 视觉保真度/物理常识/动作可信度三维框架
> - **GitHub**: https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy

---

*笔记创建时间: 2026-05-18*

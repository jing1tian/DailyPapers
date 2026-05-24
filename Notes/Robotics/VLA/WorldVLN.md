---
title: "WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation"
method_name: "WorldVLN"
authors: [Baining Zhao, Jiacheng Xu, Weicheng Feng, Xin Zhang, Zhaolu Wang, Haoyang Wang, Shilong Ji, Ziyou Wang, Jianjie Fang, Zhiheng Zheng, Weichen Zhang, Yu Shang, Wei Wu, Chen Gao, Xinlei Chen, Yong Li]
year: 2026
venue: arXiv
tags: [aerial-vln, world-action-model, autoregressive, uav-navigation, reinforcement-learning, vision-language-action, grpo]
zotero_collection: Robotics/VLA
image_source: mixed
arxiv_html: https://arxiv.org/html/2605.15964
created: 2026-05-18
---

# 论文笔记：WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, EmbodiedCity |
| 日期 | May 2026 |
| 项目主页 | [embodiedcity.github.io/WorldVLN](https://embodiedcity.github.io/WorldVLN/) |
| 对比基线 | [[OpenVLA]], [[π₀]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15964) / [Project](https://embodiedcity.github.io/WorldVLN/) |

---

## 一句话总结

> WorldVLN 将空中视觉-语言导航重新定义为"预测驱动的世界-动作问题"，通过自回归潜在世界模型预测短期状态转变并解码为可执行航点动作，在两个无人机导航 benchmark 上实现超过 12% 成功率提升，且零样本迁移到真实无人机。

---

## 核心贡献

1. **自回归世界-动作模型（Autoregressive WAM）**: 将视频生成的 [[World Action Model|WAM]] 范式引入空中 VLN，用潜在自回归 Transformer 预测短时域状态转变并直接解码航点动作，而非端到端观测到动作映射。
2. **Action-aware GRPO**: 首个专为自回归 WAM 设计的强化学习方法，以带时间衰减权重的三组合奖励（轨迹奖励 + 任务奖励 + 参考奖励）优化策略，在监督训练平台期后带来 10+ 个点的额外提升。
3. **零样本真实无人机迁移**: 模型仅在仿真中训练，零样本部署到配备 Jetson Orin NX 的真实四旋翼无人机，在室内和室外场景均验证成功。

---

## 问题背景

### 要解决的问题

[[空中视觉语言导航|Aerial Vision-Language Navigation (VLN)]] 要求无人机根据自然语言指令在开放环境中导航到目标位置。现有方法直接将观测和指令映射为动作，缺乏对世界动态的显式建模。

### 现有方法的局限

- **[[Vision-Language-Action Model|VLA]] 方法**：继承了语言和视觉理解能力，但未建模"动作条件化的世界动态"，对空间推理和精细运动控制能力有限。
- **[[World Action Model|WAM]] 方法**：视频生成基础模型提供了强时空先验，但聚焦于"视觉逼真的未来合成"而非目标导向的动作生成。
- **结构不匹配**：通用视频模型双向生成，而导航需要"因果式观测-行动-更新循环"。
- **目标不匹配**：视频生成优化视觉可信度，而 VLN 需要编码几何一致性和任务效益的"动作感知后果建模"。

### 本文的动机

将空中 VLN 重新表述为"预测驱动的世界-动作问题"：智能体应预期潜在的世界演化，并根据预测后果行动。通过预测短时域潜在转变（而非完整序列）并直接解码为动作，既保留时空先验，又对齐导航目标。

---

## 方法详解

### 模型架构

WorldVLN 采用**潜在自回归 Transformer + 动作解码器**的两阶段架构：

- **输入**: 语言指令 $\ell$ + 自中心观测历史 $o_{\leq t}$ + 历史潜在表示 $z_{\leq t}$
- **Backbone**: [[InfinityStar]]-8B 潜在自回归 Transformer（世界骨干）
- **视频编码器**: [[WAN VAE]] — 多尺度残差量化视频 Tokenizer
- **核心模块**: [[World Action Model|WAM]] 预测潜在转变 + [[Action Decoder]] 解码航点动作
- **输出**: 4-DOF 航点动作 $a_t = (\Delta x_t, \Delta y_t, \Delta z_t, \Delta\psi_t) \in \mathbb{R}^4$
- **训练框架**: 8 × NVIDIA A800 80GB GPU；仿真在 RTX 4090 工作站运行

### 核心模块

#### 模块1：潜在自回归世界骨干

**设计动机**: 利用 [[InfinityStar]] 的粗到细尺度预测和片段序自回归能力，对指令条件化的导航动力学建模。

**具体实现**:
- 以 [[WAN VAE]] 对视频帧进行时域压缩比 4 的编码，得到潜在表示 $z_t$
- 给定指令编码 $e_\ell$ 和历史潜在上下文 $z_{\leq t}$，预测未来 $K=16$ 帧的潜在片段
- **闭环感知**: 执行动作后用真实观测更新自回归上下文，而非依赖幻觉帧

#### 模块2：动作解码器（Action Decoder）

**设计动机**: 将世界骨干输出的预测潜在片段 $\hat{z}_{t+1:t+K}$ 直接解码为可执行航点动作序列。

**具体实现**:
1. **视觉嵌入模块**: 将潜在片段通过特征重塑、卷积映射、上采样和投影，转为时空嵌入 token
2. **[[Spatiotemporal Transformer]]**: 因子化时间注意力（捕捉跨帧运动演化）+ 空间注意力（建模帧内几何结构）
3. **动作回归头**: MLP 将聚合表示映射为连续动作向量 $(\Delta x, \Delta y, \Delta z, \Delta\psi)$

#### 模块3：Action-aware GRPO

**设计动机**: 监督训练在有限演示分布上会遇到协变量偏移；[[GRPO]] 可直接优化任务奖励，但需适配自回归 WAM 的片段级采样结构。

**具体实现**:
- 每个训练样例采样 $G=4$ 个完整轨迹 rollout
- 对每个 rollout 的每个片段 $j$ 计算组合奖励（详见公式部分）
- 时间衰减加权 $\gamma^{j-1}$（$\gamma=0.9$）强调早期决策的影响
- 冻结前五个模型 chunk，仅微调后续部分

---

## 关键公式

### 公式1: [[Vision-Language-Action Model|导航策略定义]]

$$
a_t \sim \pi_\theta(\cdot \mid o_{\leq t},\, a_{<t},\, \ell), \quad a_t = (\Delta x_t, \Delta y_t, \Delta z_t, \Delta\psi_t) \in \mathbb{R}^4
$$

**含义**: 策略 $\pi_\theta$ 根据观测历史、历史动作和语言指令自回归生成当前航点动作。

**符号说明**:
- $a_t$: 时刻 $t$ 的航点动作，4-DOF（三维位移 + 偏航角变化）
- $o_{\leq t}$: 历史自中心观测序列
- $\ell$: 自然语言导航指令
- $\pi_\theta$: 参数化策略

---

### 公式2: [[World Action Model|潜在世界预测]]

$$
\hat{z}_{t+1:t+K} \sim p_\theta(\cdot \mid e_\ell,\, z_{\leq t})
$$

**含义**: 世界骨干根据指令编码和历史潜在上下文，预测未来 $K$ 帧的潜在片段。

**符号说明**:
- $\hat{z}_{t+1:t+K}$: 预测的未来潜在片段（$K=16$ 帧）
- $e_\ell$: 指令编码向量
- $z_{\leq t}$: 历史潜在表示序列（用真实观测更新）

---

### 公式3: [[Action Decoder|动作解码]]

$$
a_{t:t+K-1} = D_\phi(\hat{z}_{t+1:t+K})
$$

**含义**: 动作解码器将预测的潜在片段直接转化为可执行的航点动作序列。

**符号说明**:
- $D_\phi$: 动作解码器（参数 $\phi$）
- $\hat{z}_{t+1:t+K}$: 世界骨干输出的预测潜在片段

---

### 公式4: [[Closed-Loop Navigation|闭环自回归导航]]

$$
(e_\ell, z_0) \to \hat{z}_{1:K} \to a_{0:K-1} \to o_{1:K} \to z_{1:K} \to \hat{z}_{K+1:2K} \to \cdots
$$

**含义**: 完整的闭环导航循环：预测潜在转变 → 解码动作 → 执行并获取真实观测 → 更新潜在上下文 → 下一次预测。

**符号说明**:
- 每轮预测 $K=16$ 帧，时间压缩比 4，对应 4 步执行动作
- 真实观测 $o_{1:K}$ 通过 [[WAN VAE]] 编码为 $z_{1:K}$ 更新上下文（而非用幻觉帧）

---

### 公式5: [[World Action Model|世界模型监督损失]]

$$
\mathcal{L}_{wm} = -\sum_t \log p_\theta(z_{t+1:t+K} \mid e_\ell,\, z_{\leq t})
$$

**含义**: 监督潜在自回归骨干预测真实未来潜在片段的负对数似然损失。

**符号说明**:
- $z_{t+1:t+K}$: 通过 $E_{vid}(o_{t+1:t+K})$ 编码的真实未来潜在片段
- $e_\ell$: 指令编码

---

### 公式6: [[Action Decoder|动作解码器监督损失]]

$$
\mathcal{L}_{act} = \sum_t \| D_\phi(E_{vid}(o_{t+1:t+K})) - a^*_{t:t+K-1} \|
$$

**含义**: 训练解码器从真实视频编码潜变量中恢复专家航点动作，以 L1 距离为损失。

**符号说明**:
- $a^*_{t:t+K-1}$: 专家示范动作序列
- $E_{vid}$: WAN VAE 视频编码器

---

### 公式7: [[GRPO|片段级组合奖励]]

$$
r_j^{(i)} = \gamma^{j-1}\!\left(\lambda_{traj}\, r_{traj,j}^{(i)} + \lambda_{task}\, r_{task,j}^{(i)} + \lambda_{ref}\, r_{ref,j}^{(i)}\right)
$$

**含义**: 第 $i$ 条轨迹第 $j$ 个片段的组合奖励，以时间衰减因子 $\gamma$ 强调早期决策。

**符号说明**:
- $\gamma = 0.9$: 时间衰减因子
- $\lambda_{traj} = 0.2,\; \lambda_{task} = 0.7,\; \lambda_{ref} = 0.1$: 奖励权重系数
- $r_{traj}$: 轨迹奖励（局部几何监督）
- $r_{task}$: 任务奖励（全局终点成功）
- $r_{ref}$: 参考奖励（防止策略漂移）

---

### 公式8: [[GRPO|轨迹奖励]]

$$
r_{traj,j}^{(i)} = \frac{1}{1 + \| a^{(i)}_{(j-1)K:jK-1} - a^{*\,(i)}_{(j-1)K:jK-1} \|}
$$

**含义**: 度量预测动作序列与专家动作的局部几何距离，以反比形式映射为 $(0,1]$ 区间奖励。

**符号说明**:
- $a^{(i)}_{(j-1)K:jK-1}$: 第 $i$ 条轨迹第 $j$ 片段的采样动作
- 分母中的距离 $d_{traj} = 0.45 \cdot MSE_{xyz} + 0.45 \cdot MSE_{yaw} + 0.1 \cdot MSE_{all}$（加权组合）

---

### 公式9: [[GRPO|任务奖励]]

$$
r_{task,j}^{(i)} = \frac{1}{1 + \|(x_T^{(i)}, y_T^{(i)}, z_T^{(i)}) - (x^*, y^*, z^*)\|_2}
$$

**含义**: 评估最终位置与目标位置的欧氏距离，以全局终点成功为优化目标。

**符号说明**:
- $(x_T, y_T, z_T)$: 轨迹最终位置
- $(x^*, y^*, z^*)$: 目标位置

---

### 公式10: [[GRPO|参考奖励]]

$$
r_{ref,j}^{(i)} = \log \pi_{ref}\!\left(a^{(i)}_{(j-1)K:jK-1} \mid o_{\leq t},\, a^{(i)}_{<(j-1)K},\, \ell\right)
$$

**含义**: 参考策略的对数似然，作为正则项防止策略过度偏离初始行为分布。

**符号说明**:
- $\pi_{ref}$: 监督训练后的参考（初始）策略
- 该项相当于对动作分布施加隐式 KL 约束

---

### 公式11: [[GRPO|GRPO 目标函数]]

$$
J_{GRPO} = \mathbb{E}_{i,j}\!\left[\min\!\left(\rho_j^{(i)} A_j^{(i)},\; \text{clip}\!\left(\rho_j^{(i)}, 1-\varepsilon_{clip}, 1+\varepsilon_{clip}\right) A_j^{(i)}\right)\right]
$$

**含义**: PPO 风格裁剪目标，以优势估计 $A_j^{(i)}$ 加权概率比，防止策略更新步幅过大。

**符号说明**:
- $\rho_j^{(i)} = \pi_\theta / \pi_{old}$: 新旧策略的概率比
- $A_j^{(i)}$: 片段级优势（组内归一化奖励 $r_j^{(i)}$）
- $\varepsilon_{clip} = 0.02$: 裁剪阈值

---

## 关键图表

### Figure 1: WorldVLN 整体架构

![[WorldVLN_fig1_architecture.png]]

**说明**: WorldVLN 的整体架构。输入语言指令和观测历史，通过 [[World Action Model|WAM]] 自回归骨干预测短时域潜在世界转变，动作解码器将预测潜变量解码为航点动作，执行后用真实观测更新自回归上下文（闭环感知）。

---

### Figure 2: 两阶段训练框架

![[WorldVLN_fig2_training.png]]

**说明**: **Stage 1** 用指令-视频对监督潜在自回归骨干，用视频-轨迹对训练动作解码器。**Stage 2** 采样多条 rollout，以片段级三组合奖励（轨迹准确性 + 任务进展 + 参考策略正则化）+ 时间衰减加权，通过 Action-aware [[GRPO]] 更新 WorldVLN。

---

### Figure 3: 定性案例分析

![[WorldVLN_fig3_case_analysis.png]]

**说明**: WorldVLN 与 VLA 基线对比。在室外以物体为中心的机动（如"向左飞越障碍物"）和室内地标导航中，WorldVLN 显示更强的空间定位能力和更精确的航点动作。

---

### Figure 4: 消融实验

![[WorldVLN_fig4_ablation.png]]

**说明**: 四个子图展示：训练动态曲线、量化消融结果、潜在预测探针分析、Action-aware GRPO 提升效果。自回归建模较全序列预测提升 5.7+ 个点；GRPO 在监督训练平台期后额外带来 10+ 个点提升。

---

### Figure 5: 真实无人机部署

![[WorldVLN_figurenew.png]]

**说明**: WorldVLN 仅在仿真中训练，零样本迁移到真实四旋翼无人机。室内（10m×15m×3m 动捕场地）和室外（GPS+LiDAR 高度估计）均验证成功。

---

### Figure 6: 潜在时空自回归世界骨干架构

![[WorldVLN_fig6_backbone_arch.png]]

**说明**: [[InfinityStar]] 骨干的视觉金字塔条件和未来目标片段金字塔的详细架构，展示粗到细尺度预测和片段序自回归机制。

---

### Figure 7: 动作解码器架构

![[WorldVLN_fig7_action_decoder.png]]

**说明**: 动作解码器将世界骨干输出的潜变量转化为时空嵌入，再通过 [[Spatiotemporal Transformer]]（因子化时间/空间注意力）和 MLP 回归头输出 4-DOF 导航动作。

---

### Figure 8: UAV-Flow 定性示例

![[WorldVLN_fig8_outdoor_cases.png]]

**说明**: UAV-Flow benchmark 上覆盖多种细粒度无人机飞行动作的定性示例，包括 Approach、Land、Surround、Rotate 等动作类型。

---

### Figure 9: IndoorUAV-VLA 定性示例

![[WorldVLN_figure1.png]]

**说明**: IndoorUAV-VLA benchmark 上 Easy/Medium/Hard 三个复杂度级别的定性示例，Hard 级别要求组合执行三种低级动作类型。

---

### Figure 10: 真实无人机平台与系统架构

![[WorldVLN_training.png]]

**说明**: 250mm 轴距四旋翼平台（Logi C270 RGB 摄像头 + Jetson Orin NX 16GB + CUAV PX4 飞控），高层推理在地面站服务器运行，低层控制由 PX4 处理。

---

### Table 1: UAV-Flow-Sim 结果

| 模型 | 指令类型 | Approach | Retreat | Pass | Land | Turn | Move | Shift | Rotate | Surround | A/D | 平均 SR |
|------|----------|----------|---------|------|------|------|------|-------|--------|----------|-----|---------|
| Seq2Seq-UAV | Fixed | 0.00 | 0.00 | 15.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.50 |
| CMA-UAV | Fixed | 0.00 | 91.67 | 25.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 11.67 |
| Travel-UAV | Fixed | 35.71 | 100.00 | 25.00 | 53.70 | 20.00 | 13.33 | 85.71 | 46.67 | 58.33 | 84.21 | 52.27 |
| Travel-UAV | Open | 42.86 | 100.00 | 32.50 | 40.74 | 6.67 | 6.67 | 75.51 | 40.00 | 100.00 | 78.95 | 52.39 |
| OpenVLA-UAV | Fixed | 45.24 | 100.00 | 37.50 | 46.30 | 73.33 | 66.67 | 77.55 | 20.00 | 100.00 | 89.47 | 65.61 |
| OpenVLA-UAV | Open | 47.62 | 100.00 | 55.00 | 40.74 | 73.33 | 60.00 | 85.71 | 6.67 | 100.00 | 84.21 | 65.33 |
| π₀-0-UAV | Fixed | 59.52 | 100.00 | 50.00 | 66.67 | 33.33 | 60.00 | 18.37 | 46.67 | 58.33 | 26.32 | 51.92 |
| π₀-0-UAV | Open | 71.43 | 100.00 | 62.50 | 72.22 | 80.00 | 80.00 | 40.82 | 60.00 | 75.00 | 15.79 | 65.78 |
| **WorldVLN (Ours)** | **Fixed** | **97.62** | **91.67** | **40.00** | **92.59** | **60.00** | **100.00** | **85.71** | **46.67** | **58.33** | **94.74** | **79.12** |
| **WorldVLN (Ours)** | **Open** | **95.24** | **91.67** | **40.00** | **98.15** | **53.33** | **100.00** | **85.71** | **33.33** | **41.67** | **94.74** | **78.02** |

**说明**: WorldVLN 在 UAV-Flow-Sim 上平均成功率达 79.12%（Fixed）/ 78.02%（Open），较最强基线提升 13.51+ 个百分点。在 Approach、Move 等需要精细空间控制的动作类型上优势尤为显著。

---

### Table 2: IndoorUAV-VLA 结果

| 模型 | Easy SR | Easy NDTW | Medium SR | Medium NDTW | Hard SR | Hard NDTW | 平均 SR | 平均 NDTW |
|------|---------|-----------|-----------|-------------|---------|-----------|---------|-----------|
| GPT-4o | 30.30 | 12.30 | 4.00 | 6.57 | 1.96 | 4.84 | 11.69 | 9.20 |
| Seq2Seq | 1.60 | 2.63 | 1.20 | 2.74 | 1.03 | 3.03 | 1.33 | 2.74 |
| CMA | 1.28 | 1.75 | 0.75 | 1.94 | 1.03 | 2.01 | 0.99 | 1.88 |
| OpenVLA | 22.52 | 2.89 | 1.19 | 1.12 | 0.00 | 0.12 | 7.81 | 2.42 |
| π₀-FAST | 18.09 | 8.83 | 5.26 | 2.93 | 1.14 | 2.68 | 8.62 | 4.71 |
| NaVid | 25.31 | 13.10 | 18.21 | 3.21 | 2.31 | 1.72 | 15.82 | 5.28 |
| π₀ | 46.58 | 14.52 | 21.64 | 7.64 | 7.55 | 4.27 | 27.16 | 9.44 |
| **WorldVLN (Ours)** | **49.43** | **13.04** | **37.72** | **14.52** | **41.19** | **12.80** | **41.76** | **13.48** |

**关键发现**: WorldVLN 在 Hard 难度上成功率达 41.19%，较次优基线 π₀（7.55%）提升 33.64 个点，表明 WAM 对复杂多步动作组合的鲁棒性远超 VLA 基线。Medium 难度提升 16.08 点，说明挑战性越高优势越明显。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| UAV-Flow-Sim | 10 类飞行动作 | 固定/开放词汇指令，室外无人机细粒度动作 | 测试 |
| IndoorUAV-VLA | Easy/Medium/Hard 三级 | 基于动作组合复杂度分级，1-3 类低级动作组合 | 测试 |
| 仿真导航视频 | 配对视频-指令-轨迹 | 用于 Stage 1 监督训练 | 训练 |

### 实现细节

- **Backbone**: [[InfinityStar]]-8B 潜在自回归 Transformer
- **视频编码器**: [[WAN VAE]]（时域压缩比 4）
- **预测窗口**: $K=16$ 帧/片段
- **Stage 1 配置**: 49 帧（1 初始帧 + 3×16 片段）
- **Stage 2 配置**: 组大小 $G=4$，最大迭代 3000 步，学习率 $8\times10^{-7}$
- **奖励权重**: $\lambda_{traj}=0.2, \lambda_{task}=0.7, \lambda_{ref}=0.1$，衰减 $\gamma=0.9$
- **PPO 裁剪**: $\varepsilon_{clip}=0.02$，KL 系数 0.9
- **硬件**: 训练 8×A800 80GB；仿真 RTX 4090

### 可视化结果

定性分析显示，WorldVLN 在室外以物体为中心的机动（如"绕障碍物飞行"）中展现更强空间定位，在室内地标导航中生成更精准的航点序列。VLA 基线常在方向判断和距离估计上出错，而 WorldVLN 通过潜在世界预测能隐式编码空间几何关系。

---

## 批判性思考

### 优点

1. **范式创新**: 将 VLN 从观测-动作映射重新定义为预测驱动的世界-动作问题，在概念上更合理，且与视频生成预训练对齐。
2. **显著性能提升**: 在两个 benchmark 上均实现 12%+ 成功率提升，尤其在 Hard 难度下提升 33.64 点，优势来自对复杂多步组合导航的建模能力。
3. **零样本真实迁移**: 仿真到真实世界的零样本迁移是实用性的重要验证，尽管硬件依赖地面站服务器进行推理。
4. **Action-aware GRPO**: 针对自回归 WAM 特性设计的 RL 方法，时间衰减加权在概念上合理，且带来了监督训练后显著的额外提升。

### 局限性

1. **计算开销**: 推理需要地面站服务器（非全板载），限制了在无基础设施场景下的实际部署能力。
2. **短时域**: 仅预测 $K=16$ 帧（时间压缩后约 4 个动作步），长距离导航可能累积误差。
3. **仿真数据依赖**: Stage 1 依赖仿真生成的视频-指令-轨迹对，真实世界数据收集成本较高。
4. **基线局限**: 未与近期专门针对空中导航优化的 VLA 变体比较，如专门针对 UAV 域微调的模型。

### 潜在改进方向

1. **长时域 VLN**: 通过分层规划或更长 rollout 窗口支持长距离导航任务
2. **模型压缩与全板载部署**: 将推理压缩到边缘设备（如 Jetson Orin NX 本地推理）
3. **主动不确定性估计**: 在预测不确定时触发重新规划，提升鲁棒性

### 可复现性评估

- [ ] 代码开源（项目页面列出，具体发布状态待确认）
- [ ] 预训练模型（待确认）
- [x] 训练细节完整（附录中提供详尽超参数和硬件配置）
- [x] 数据集可获取（使用公开仿真环境生成）

---

## 关联笔记

### 基于

- [[World Action Model|WAM]]: 世界动作模型的核心范式
- [[InfinityStar]]: 潜在自回归 Transformer 骨干
- [[WAN VAE]]: 视频编码器/Tokenizer
- [[GRPO]]: 强化学习训练方法基础

### 对比

- [[OpenVLA]]: VLA 基线，直接观测-动作映射
- [[Pi05|π₀]]: 扩散策略 VLA 基线
- [[WAM-Survey]]: WAM 相关综述

### 方法相关

- [[World Action Model|WAM]]: 核心架构范式
- [[Spatiotemporal Transformer]]: 动作解码器核心组件
- [[GRPO]]: RL 训练方法
- [[空中视觉语言导航|Aerial VLN]]: 任务定义

### 硬件/数据相关

- [[PX4]]: 无人机飞控系统

---

## 速查卡片

> [!summary] WorldVLN
> - **核心**: 将空中 VLN 定义为预测驱动的世界-动作问题，自回归预测潜在世界状态并解码为航点动作
> - **方法**: InfinityStar-8B 骨干 + Action Decoder + Action-aware GRPO（两阶段训练）
> - **结果**: UAV-Flow 79.12% SR（+13.51pp），IndoorUAV-VLA 41.76% SR（+14.60pp），零样本真实 UAV 迁移
> - **代码**: [embodiedcity.github.io/WorldVLN](https://embodiedcity.github.io/WorldVLN/)

---

*笔记创建时间: 2026-05-18*

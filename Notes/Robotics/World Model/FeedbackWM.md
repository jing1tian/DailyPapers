---
title: "Feedback World Model Enables Precise Guidance of Diffusion Policy"
method_name: "FeedbackWM"
authors: [Tuo An, Jindou Jia, Gen Li, Jingliang Li, Chuhao Zhou, Pengfei Liu, Bofan Lyu, Jiaqi Bai, Xinying Guo, Geng Li, Jianfei Yang]
year: 2026
venue: arXiv
tags: [world-model, diffusion-policy, robot-manipulation, inference-time-guidance, ood-generalization]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.15705
created: 2026-05-18
---

# 论文笔记：Feedback World Model Enables Precise Guidance of Diffusion Policy

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MARS Lab, Nanyang Technological University (NTU) |
| 日期 | May 2026 |
| 项目主页 | [Feedback World Model](https://lorenzo-0-0.github.io/Feedback_World_Model/) |
| 对比基线 | [[Diffusion Policy]], [[Action-Conditioned World Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15705) / Code N/A |

---

## 一句话总结

> 提出反馈世界模型，通过在推理时以实时观测闭环校正潜在状态预测，并结合行动感知引导，显著提升扩散策略在分布外场景下的成功率。

---

## 核心贡献

1. **反馈世界模型（Feedback World Model）**: 维护轻量级反馈状态，利用实时观测迭代纠正 [[Action-Conditioned World Model|行动条件世界模型]] 的预测漂移，无需额外训练数据或参数更新
2. **行动感知引导（Action-Aware Guidance）**: 按行动可控性对潜在维度加权，强调任务相关的状态变化，将纠正后的预测更精准地转化为控制信号
3. **实验验证**: 在 LIBERO-Plus、Robomimic 和真实机器人操作任务上，OOD 成功率平均从 40% 提升至 52%（相对提升 30%），预测误差最高降低 76.4%

---

## 问题背景

### 要解决的问题

[[Diffusion Policy|扩散策略]] 在分布内（in-distribution）场景表现良好，但遇到分布外（OOD）的初始状态时性能显著下降。已有方法通过[[Action-Conditioned World Model|行动条件世界模型]]引导扩散策略，但世界模型在 OOD 场景下会产生累积预测误差（prediction drift），导致引导信号失效。

### 现有方法的局限

- **Standard Guidance**：依赖世界模型在 OOD 区域的准确性，而世界模型仅在专家演示数据上训练，泛化能力有限
- **Rollout-Enhanced Guidance**：用策略 rollout 数据扩充训练集，但需要额外数据收集，成本高且不总是可行
- **开环预测**：世界模型进行开环预测，无法利用执行过程中的真实观测进行修正，误差随时间积累

### 本文的动机

在控制理论中，[[闭环控制]] 通过实时反馈修正误差是提升鲁棒性的经典思路。本文将这一思想引入世界模型引导：每次执行后，用真实观测更新反馈状态，实时纠正后续预测的基准，从而补偿模型误差，无需任何额外训练。

---

## 方法详解

### 模型架构

FeedbackWM 基于 [[Diffusion Policy]] 框架，在推理时引入反馈闭环和行动感知引导：

- **输入**: 观测 $O_t$（图像序列）+ 语言指令 $\ell$
- **Backbone**: [[Vision Transformer]]（图像编码）+ [[Diffusion Policy]]（动作生成）
- **核心模块**: [[反馈世界模型]] 用于闭环预测校正 + [[行动感知引导]] 用于加权能量函数
- **输出**: [[Action Chunking|动作块]] $A_t$
- **额外参数**: 仅反馈状态 $\bar{z}_t$（与潜在空间同维度），无新增可训练参数

整体推理流程：

- **外层闭环**: 每步执行后，真实观测 $O_{t+1}$ 更新反馈状态 $\bar{z}_{t+1}$，修正后续预测基准
- **内层去噪**: 每次去噪步，用反馈校正后的预测潜在 $z_{t+1}^{fb}$ 计算行动感知能量，引导动作分布

### 核心模块

#### 模块 1: 反馈世界模型（Feedback World Model）

**设计动机**: 利用[[闭环控制]]原理，通过实时观测纠正[[Action-Conditioned World Model|行动条件世界模型]]的开环预测漂移

**具体实现**:

1. 编码当前真实观测：$z_t = \psi(O_t)$（[[Vision Transformer]] 编码器 $\psi$）
2. 计算反馈误差：$e_t = z_t - \bar{z}_t$（真实潜在与预测潜在之差）
3. 修正预测速度：$\hat{v}_t = v_t(A_t^{(\tau)}) + Le_t$（$L$ 为反馈增益矩阵）
4. 反馈预测下一状态：$z_{t+1}^{fb} = z_t + \delta t \cdot \hat{v}_t$
5. 更新反馈状态（执行后）：$\bar{z}_{t+1} = \bar{z}_t + \delta t \cdot \hat{v}_t(A_t^{(0)})$

在理论上（Proposition 1），系统是 Lyapunov 稳定的，误差上界为 $\gamma / \lambda_{min}(L)$。

#### 模块 2: 行动感知引导（Action-Aware Guidance）

**设计动机**: 并非所有潜在维度都同等重要，某些维度对动作变化敏感（action-controllable），对这些维度加权可使引导更精准地关注任务相关状态变化

**具体实现**:

1. 离线计算可控性权重：对演示数据中每个观测 $z_i$，对不同动作 $a \sim \mathcal{A}_{demo}$ 计算世界模型输出的方差
2. 归一化权重：$\bar{w}_j$ 为归一化后权重，$w_j^{(\beta)} = 1 + \beta(\bar{w}_j - 1)$ 为温度调节后权重
3. 加权能量函数：$E_{ctrl}(A_t^{(\tau)})$ 按权重衡量反馈预测潜在与最近专家潜在的距离
4. 加权梯度引导：$\tilde{s} = s_\phi - \lambda \nabla_{A_t^{(\tau)}} E_{ctrl}$

---

## 关键公式

### 公式 1: [[Vision Transformer|观测编码]]

$$
z_t = \psi(O_t)
$$

**含义**: 将当前观测 $O_t$ 编码为潜在表示 $z_t$

**符号说明**:
- $\psi$: 预训练图像编码器（[[Vision Transformer]]）
- $O_t$: 时刻 $t$ 的观测（图像序列）
- $z_t$: 潜在状态向量

### 公式 2: [[Action-Conditioned World Model|行动条件世界模型]]预测

$$
\hat{z}_{t+1} = f_\theta(z_t, A_t)
$$

**含义**: 世界模型 $f_\theta$ 预测在当前状态 $z_t$ 下执行动作 $A_t$ 后的下一潜在状态

**符号说明**:
- $f_\theta$: 行动条件世界模型（预训练，推理时冻结）
- $A_t^{(\tau)}$: 去噪步 $\tau$ 时的动作序列
- $\hat{z}_{t+1}$: 预测的下一潜在状态

### 公式 3: [[Diffusion Policy|扩散策略]]得分函数

$$
s_\phi(A_t^{(\tau)}, O_t, \ell, \tau) \approx \nabla_{A_t^{(\tau)}} \log p_\tau(A_t^{(\tau)} \mid O_t, \ell)
$$

**含义**: 扩散策略网络 $s_\phi$ 估计动作分布的得分（即对数概率梯度）

**符号说明**:
- $s_\phi$: [[Diffusion Policy|扩散策略]]网络
- $\tau$: 去噪时间步（$\tau \in [0, 1]$）
- $\ell$: 语言指令

### 公式 4: [[Model Predictive Control|引导扩散得分]]

$$
\tilde{s}(A_t^{(\tau)}, O_t, \ell, \tau) = s_\phi(A_t^{(\tau)}, O_t, \ell, \tau) + \lambda g(A_t^{(\tau)}, O_t)
$$

**含义**: 将世界模型引导梯度 $g$ 叠加到扩散策略得分上，引导动作分布朝向专家状态

**符号说明**:
- $\lambda$: 引导强度超参数
- $g$: 能量函数关于动作的负梯度

### 公式 5: 最近邻专家状态检索

$$
i^* = \arg\min_i \|\hat{z}_{t+1}(A_t^{(\tau)}) - z_i^E\|_2^2
$$

**含义**: 在专家演示轨迹潜在库 $\{z_i^E\}$ 中，找到与当前预测下一状态最近邻的专家潜在状态

**符号说明**:
- $z_i^E$: 第 $i$ 个专家演示步的潜在状态
- $i^*$: 最近邻专家状态索引

### 公式 6: 基础能量函数

$$
E(A_t^{(\tau)}) = \|\hat{z}_{t+1}(A_t^{(\tau)}) - z_{i^*}^E\|_2^2
$$

**含义**: 度量预测下一状态与最近专家状态的 L2 距离，作为引导能量

### 公式 7: 标准引导梯度

$$
\tilde{s} = s_\phi - \lambda \nabla_{A_t^{(\tau)}} E(A_t^{(\tau)})
$$

**含义**: 用能量函数梯度引导扩散策略，使动作朝向减小与专家状态距离的方向更新

### 公式 8: [[Action-Conditioned World Model|世界模型速度预测]]

$$
v_t(A_t^{(\tau)}) \triangleq \frac{f_\theta(z_t, A_t^{(\tau)}) - z_t}{\delta t} = \frac{\hat{z}_{t+1} - z_t}{\delta t}
$$

**含义**: 将世界模型预测的状态变化解释为潜在空间速度，便于后续控制器设计

**符号说明**:
- $\delta t$: 时间步长
- $v_t$: 预测的潜在状态速度

### 公式 9: 反馈误差计算

$$
e_t = z_t - \bar{z}_t
$$

**含义**: 当前真实潜在状态与反馈预测状态之差，即反馈误差信号

**符号说明**:
- $\bar{z}_t$: 反馈状态（模型对当前状态的预测，基于历史反馈更新）
- $e_t$: 反馈误差

### 公式 10: 反馈校正速度与反馈预测状态

$$
\begin{aligned}
\hat{v}_t(A_t^{(\tau)}) &= v_t(A_t^{(\tau)}) + Le_t \\
z_{t+1}^{fb}(A_t^{(\tau)}) &= z_t + \delta t \cdot \hat{v}_t(A_t^{(\tau)})
\end{aligned}
$$

**含义**: 用反馈误差 $Le_t$ 修正开环预测速度，得到更准确的下一状态预测

**符号说明**:
- $L$: 反馈增益矩阵（控制反馈强度）
- $\hat{v}_t$: 反馈校正后的速度
- $z_{t+1}^{fb}$: 反馈校正后的预测下一潜在状态

### 公式 11: 反馈状态更新（执行后外层闭环）

$$
\bar{z}_{t+1} = \bar{z}_t + \delta t \cdot \hat{v}_t(A_t^{(0)})
$$

**含义**: 每次执行动作 $A_t^{(0)}$（去噪完成后的最终动作）后，更新反馈状态，为下一步预测提供校正基准

### 公式 12: 下一步反馈误差更新

$$
e_{t+1} = z_{t+1} - \bar{z}_{t+1}
$$

**含义**: 执行后获得真实观测，更新反馈误差用于下一时间步的闭环校正

### 公式 13: 误差有界性（Lyapunov 稳定性）

$$
\lim_{t \to \infty} \|z_t - \bar{z}_t\| \leq \frac{\gamma}{\lambda_{min}(L)}
$$

**含义**: 在适当增益 $L$ 下，反馈误差有界，系统 Lyapunov 稳定。$\gamma$ 为世界模型误差上界，$\lambda_{min}(L)$ 为增益矩阵最小特征值

**符号说明**:
- $\gamma$: 世界模型单步误差的上界
- $\lambda_{min}(L)$: 增益矩阵 $L$ 的最小特征值

### 公式 14: 行动可控性权重计算

$$
w_j = \frac{1}{M} \sum_{i=1}^M \mathrm{Var}_{a \sim \mathcal{A}_{demo}}([f_\theta(z_i, a)]_j)
$$

**含义**: 对每个潜在维度 $j$，计算世界模型输出关于动作变化的方差，作为该维度对动作的响应程度（可控性）

**符号说明**:
- $M$: 演示轨迹数量
- $j$: 潜在空间维度索引（$j = 1, \ldots, D$）
- $\mathcal{A}_{demo}$: 演示动作分布
- $w_j$: 第 $j$ 维的可控性权重

### 公式 15: 权重归一化与温度调节

$$
\bar{w}_j = \frac{w_j}{\frac{1}{D}\sum_{k=1}^D w_k + \epsilon}, \quad w_j^{(\beta)} = 1 + \beta(\bar{w}_j - 1)
$$

**含义**: 先将权重归一化为均值为 1 的相对权重，再用温度参数 $\beta$ 调节对比度；$\beta=0$ 退化为均匀权重（无加权）

**符号说明**:
- $\bar{w}_j$: 归一化可控性权重
- $\beta$: 温度参数，控制权重对比度
- $\epsilon$: 数值稳定性小量
- $D$: 潜在空间总维度数

### 公式 16: 行动感知能量函数

$$
E_{ctrl}(A_t^{(\tau)}) = \sum_{j=1}^D w_j^{(\beta)}\bigl(z_{t+1,j}^{fb}(A_t^{(\tau)}) - \hat{z}_{i^*,j}^E\bigr)^2
$$

**含义**: 以可控性权重对每个潜在维度加权，计算反馈校正后的预测状态与最近专家状态的加权 L2 距离，作为行动感知引导能量

**符号说明**:
- $z_{t+1,j}^{fb}$: 反馈预测状态的第 $j$ 维
- $\hat{z}_{i^*,j}^E$: 最近邻专家状态的第 $j$ 维（注：使用 $\hat{z}^E$ 表示从专家轨迹中检索的潜在状态）
- $w_j^{(\beta)}$: 温度调节后的可控性权重

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2605.15705/2605.15705v1/x1.png)

**说明**: FeedbackWM 的整体架构。在去噪过程中，反馈世界模型预测当前动作轨迹的未来潜在结果，行动感知能量引导策略朝向专家状态。每次执行后，新观测更新反馈状态，纠正后续预测并形成外层闭环以减小推理时的预测漂移。

### Figure 2: Simulated Tasks / 仿真任务

![Figure 2](https://arxiv.org/html/2605.15705/2605.15705v1/x2.png)

**说明**: 仿真任务套件。LIBERO-Plus 包含 Kitchen 和 Living Room 场景的四个单任务设置；Robomimic 包含 Transport、Square 和 Tool-Hang 三个代表性任务。

### Figure 3: Latent Prediction MSE / 潜在预测 MSE

![Figure 3](https://arxiv.org/html/2605.15705/2605.15705v1/x3.png)

**说明**: 在仿真和真实 OOD 任务上的潜在预测 MSE 对比。Base WM 仅在专家演示上训练；Rollout-Enhanced WM 额外用策略 rollout 数据训练；Feedback WM 在 Base WM 基础上加入在线反馈校正。FeedbackWM 在所有任务上均显著降低预测误差，Drawer-Open 上降低 76.4%。

### Figure 4: Real-World Tasks and Results / 真实世界任务与结果

![Figure 4](https://arxiv.org/html/2605.15705/2605.15705v1/x4.png)

**说明**: 真实机器人在 Peach-P&P 和 Drawer-Open 任务上的分布内和分布外评估结果。每个设置执行 20 次，报告成功率。OOD 场景：Peach-P&P 从 40% 提升至 80%，Drawer-Open 从 20% 提升至 70%。

### Figure 5: Latent-Space Rollout Trajectories / 潜在空间轨迹

![Figure 5](https://arxiv.org/html/2605.15705/2605.15705v1/x5.png)

**说明**: 三个案例的潜在空间 rollout 轨迹对比。将 Base WM 和 Feedback WM 预测的观测潜在编码并投影到三维空间（PC1, PC2, 时间步）。Feedback WM 的预测轨迹始终紧贴真实观测流形，而 Base WM 随时间漂移偏离。

### Figure 6: Action Controllability / 行动可控性

![Figure 6](https://arxiv.org/html/2605.15705/2605.15705v1/x5.png)

**说明**: 世界模型潜在空间的行动可控性可视化。(a) 三个 Robomimic 任务上可控性权重 $w_j$ 的排序分布（对数坐标），显示维度间可控性差异悬殊；(b) Transport 任务上的反事实动作方差热图，持续的垂直条纹表明某些维度在不同观测下始终响应于动作变化，支持离线冻结权重 $W$ 在推理时使用。

### Figure 7: Square OOD Initial States

![Figure 7](https://arxiv.org/html/2605.15705/2605.15705v1/x7.png)

**说明**: Square 任务的 OOD 初始状态。最左列为分布内演示起始状态，其余列为通过扰动机械臂关节生成的 OOD 起始状态。上行：agentview 摄像头；下行：front-view 摄像头。

### Figure 8: Transport OOD Initial States

![Figure 8](https://arxiv.org/html/2605.15705/2605.15705v1/x8.png)

**说明**: Transport 双臂任务的 OOD 初始状态。对两个机械臂独立施加关节扰动。上行：shouldercamera0；下行：robot-0 wrist 摄像头。

### Figure 9: ToolHang OOD Initial States

![Figure 9](https://arxiv.org/html/2605.15705/2605.15705v1/x9.png)

**说明**: ToolHang 任务的 OOD 初始状态。上行：side view；下行：front view。

### Table 1: 仿真 OOD 操作任务成功率

| Benchmark | Task | Base | Mixed BC | Filtered BC | Std. Guid. | Rollout Guid. | **Ours** |
|-----------|------|------|----------|------------|-----------|---------------|------|
| Robomimic | Square | 0.36 | 0.08 | 0.12 | 0.40 | 0.44 | **0.46** |
| Robomimic | Tool-Hang | 0.27 | 0.22 | 0.22 | 0.22 | 0.31 | **0.36** |
| Robomimic | Transport | 0.34 | 0.26 | 0.36 | 0.46 | 0.52 | **0.60** |
| LIBERO-Plus | Kitchen-S3 | 0.55 | 0.57 | **0.70** | 0.61 | 0.50 | 0.68 |
| LIBERO-Plus | Kitchen-S6 | 0.45 | 0.48 | 0.54 | 0.48 | 0.50 | **0.55** |
| LIBERO-Plus | Kitchen-S8 | 0.59 | 0.42 | 0.35 | 0.62 | 0.62 | **0.65** |
| LIBERO-Plus | LR-S6 | 0.26 | 0.24 | 0.24 | 0.21 | 0.24 | **0.36** |
| **Average** | | 0.40 | 0.32 | 0.36 | 0.43 | 0.45 | **0.52** |

**说明**: FeedbackWM（Ours）在 7 个任务中 6 个取得最优，平均成功率 0.52 超越所有基线，相对 Base 提升 30%。Mixed BC 和 Filtered BC（数据增强方法）在部分任务上反而降低性能，说明简单增加数据不如改善推理机制。

### Table 2: 消融实验

| Method | Square | Transport | Tool-Hang |
|--------|--------|-----------|-----------|
| Base | 0.36 | 0.34 | 0.27 |
| + Feedback WM | 0.40 | 0.54 | 0.32 |
| + Action-Aware | **0.46** | **0.60** | **0.36** |

**关键发现**: 两个组件均有独立贡献。仅加入反馈世界模型已显著提升（Transport: 0.34→0.54），再加行动感知引导进一步提升（Transport: 0.54→0.60）。

### Table 3: LIBERO-Plus 单任务设置

| Alias | Scene | Language instruction | # variants |
|-------|-------|---------------------|-----------|
| Kitchen-S3 | KITCHEN_SCENE3 | turn on the stove and put the moka pot on it | 44 |
| Kitchen-S6 | KITCHEN_SCENE6 | put the yellow and white mug in the microwave and close it | 42 |
| Kitchen-S8 | KITCHEN_SCENE8 | put both moka pots on the stove | 34 |
| LR-S6 | LIVING_ROOM_SCENE6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 42 |

**说明**: LIBERO-Plus 评估场景覆盖厨房和客厅场景，每个任务有 34-44 种初始状态变体，通过扰动机器人关节生成 OOD 初始状态。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Robomimic Square | 200 demos (20%) | 精密组装任务 | 训练/OOD测试 |
| Robomimic Transport | 200 demos (20%) | 双臂传递任务 | 训练/OOD测试 |
| Robomimic Tool-Hang | 200 demos (20%) | 工具悬挂任务 | 训练/OOD测试 |
| LIBERO-Plus Kitchen-S3/S6/S8 | 34-44 variants | 厨房场景操作 | OOD测试 |
| LIBERO-Plus LR-S6 | 42 variants | 客厅场景操作 | OOD测试 |
| Peach P&P（真实） | 50 轨迹 | 桃子抓放任务 | 训练/真实OOD测试 |
| Drawer-Open（真实） | 65 轨迹 | 抽屉开启任务 | 训练/真实OOD测试 |

### 实现细节

- **Backbone**: [[Diffusion Policy]]（DDIM 去噪）+ 预训练 [[Vision Transformer]] 图像编码器
- **OOD 评估协议**: 扰动机器人初始关节姿态生成 OOD 起始状态
- **反馈增益 $L$**: 单位矩阵（实践中简单有效）
- **温度参数 $\beta$**: 控制行动感知权重对比度
- **引导强度 $\lambda$**: 超参数
- **每个设置**: 20 次 rollout，报告成功率

### 可视化结果

潜在空间轨迹分析（Figure 5）清晰显示：Base WM 在 OOD 场景下预测轨迹偏离真实观测流形，而 Feedback WM 的轨迹始终贴近真实流形，验证了闭环校正的有效性。真实机器人实验（Figure 4）表明，在分布内场景两种方法表现相近，在 OOD 场景 FeedbackWM 优势显著（Peach-P&P: 40%→80%，Drawer-Open: 20%→70%）。

---

## 批判性思考

### 优点

1. **零额外训练成本**: 仅在推理时增加反馈状态，无需重新训练或额外数据
2. **理论保证**: 通过 Lyapunov 稳定性分析证明误差有界
3. **模块化设计**: 反馈世界模型和行动感知引导可独立使用，消融实验验证了各组件的独立贡献
4. **强基线对比**: 包含了数据增强（Mixed BC、Filtered BC）和其他引导方法作为基线，对比充分

### 局限性

1. **推理延迟增加**: 去噪过程中额外的世界模型前向传播增加了推理时间，论文承认"比先前引导方法有更高的推理延迟"
2. **依赖世界模型质量**: 若世界模型对某方向完全无法预测（如极端 OOD），反馈校正可能不足
3. **超参数敏感性**: $L$、$\beta$、$\lambda$ 等超参数的选择未详细分析

### 潜在改进方向

1. 与更高效的引导框架结合（如论文提到的未来工作方向）
2. 自适应学习反馈增益矩阵 $L$ 而非使用固定单位矩阵
3. 将反馈机制扩展至多步预测（multi-horizon feedback）

### 可复现性评估

- [ ] 代码开源（项目主页存在但未见代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（实验细节较完整，附录有额外说明）
- [x] 数据集可获取（Robomimic 和 LIBERO-Plus 均为公开数据集）

---

## 关联笔记

### 基于

- [[Diffusion Policy]]: 基础策略框架，本文在推理时为其增加引导
- [[Action-Conditioned World Model]]: 用于预测执行动作后的下一状态潜在表示
- [[World Model]]: 世界模型的更广泛框架

### 对比

- [[DINO-WM]]: 同样使用 DINO 特征的世界模型引导方法（Standard Guidance baseline）
- [[Dreamer]]: 基于世界模型的强化学习，同样利用潜在空间预测

### 方法相关

- [[Model Predictive Control]]: 反馈控制思想的来源
- [[Action Chunking]]: 扩散策略输出的动作表示形式
- [[Vision Transformer]]: 图像编码主干

### 硬件/数据相关

- [[Robomimic]]: 仿真评估基准
- [[LIBERO]]: 另一仿真评估基准

---

## 速查卡片

> [!summary] Feedback World Model Enables Precise Guidance of Diffusion Policy
> - **核心**: 在推理时通过实时观测闭环校正世界模型预测漂移，提升扩散策略 OOD 鲁棒性
> - **方法**: 反馈世界模型（Lyapunov 稳定闭环）+ 行动感知加权能量函数引导
> - **结果**: OOD 成功率平均从 40% 提升至 52%（+30%），预测误差最高降低 76.4%
> - **代码**: [项目主页](https://lorenzo-0-0.github.io/Feedback_World_Model/)（代码暂未开源）

---

*笔记创建时间: 2026-05-18*

---
title: "BICPO-VLA: Behavior-Identified Continuation Preference Optimization for Smooth Asynchronous Vision-Language-Action Control"
method_name: "BICPO-VLA"
authors: [Ming Shang, Yuchen Huang, Jiaoyang Chen, Haoyuan Hu, Han Yu, Liping Song, Luyun Feng, Shuo Bao, Wei Dong, Xinzhou Wang, Fuchun Sun]
year: 2026
venue: arXiv
tags: [vision-language-action, asynchronous-control, action-chunking, preference-optimization, flow-matching, imitation-learning, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.13924v1
created: 2026-08-18
---

# 论文笔记：BICPO-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学（Fuchun Sun 等） |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[pi0\|π₀]], [[pi0.5\|π₀.₅]], [[B-VLA]], [[Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13924) / Code: — |

---

## 一句话总结

> 通过将行为语义识别、Haar 动作坐标分解和接管偏好优化三步分离，BICPO-VLA 解决了异步 VLA 执行时新旧动作块衔接不连贯的问题。

---

## 核心贡献

1. **行为条件化动作纤维形式化**：引入"[[行为条件化动作纤维]]"概念，将行为语义（指令 + 任务进度）与具体轨迹实现解耦，把异步接管问题重新定义为在同一语义纤维内选择与 handoff 状态最兼容的实现。
2. **指令感知 Haar 结构化生成器**：[[Haar动作分解|Haar 子空间分解]]将动作块正交分解为 scaffold（逐对均值）和 residual（逐对差值），两阶段有序生成可精确重建；[[Mamba]] 因果历史编码器在融合历史前先将视觉 token 通过语言过滤。
3. **行为识别续接偏好优化（BICPO）**：将已知 outgoing 动作前滚至 handoff 状态，在同一语义纤维的候选实现中用参考相对 [[Flow-DPO]] 排名，DPO 目标可迁移到 [[pi0.5\|π₀.₅]]、Legato、RTC 等外部策略无需修改其网络结构。

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）|VLA]] 模型采用[[Action Chunking|动作分块]]执行：机器人在执行当前动作块（outgoing chunk）的同时推理下一块。由于推理延迟 $k$ 个控制周期，新旧动作块的切换点（handoff）往往存在位置跳变和速度方向不连续，即"request-to-handoff gap"。

### 现有方法的局限

- **RTC / Legato / POTR**：通过前缀条件、补全或边界正则化缓解接管不连续，但未从语义层面识别行为身份，平滑目标可能退化为静止。
- **π₀ / [[Diffusion Policy]]**：迭代去噪或 [[Flow Matching]] 生成动作，但保留原始动作坐标，无结构化 handoff 适配接口。
- **直接对 SFT 偏好候选**：会导致保守运动或进度抑制，降低任务成功率（π₀.₅ 从 96.9% 降至 49.1%）。

### 本文的动机

核心观察："**一个行为指定一族动作实现，而非唯一轨迹。**"同一语义行为对应多种具体轨迹（行为条件化动作纤维），它们共享任务含义但在局部运动和 handoff 兼容性上不同。因此可以在不改变行为语义的前提下，通过偏好优化选择最顺畅的实现。

---

## 方法详解

### 模型架构

BICPO-VLA 采用**三阶段分离架构**：

- **输入**：视觉观测 $o_r$、语言指令 $\ell$、执行历史 $\mathcal{H}_r$
- **行为识别**：[[Mamba]] 因果历史编码器 + 指令过滤注意力 → 行为状态 $z_r$ + 记忆 $m_r$ + 先验 $\mu_r$
- **Haar 生成**：固定正交 [[Haar动作分解|Haar 变换]] 分解动作块为 scaffold $C$ 和 residual $D$，共享专家网络 $F_\theta$ 有序生成
- **BICPO 续接**：冻结动作-状态模型前滚 outgoing 动作 → handoff 条件 $g_{r,k}$ → [[Flow-DPO]] 排名同纤维候选
- **输出**：重建动作块 $\hat{A}_{r+k} = W_H^T [\hat{C}, \hat{D}]$

### 核心模块

#### 模块 1：指令感知行为识别

**设计动机**：指令相关的物体和接触点因任务不同而不同，需在时序聚合前先用语言过滤视觉证据。

**具体实现**：
- 将视觉 token $V_t \in \mathbb{R}^{N_v \times d}$ 通过指令特征 $L_b$ 做[[Cross-Attention|交叉注意力]]路由（Eq. 2）
- [[Mamba]] 因果块聚合路由后观测和已执行指令 → 行为状态 $z_r$（Eq. 3）
- 相位条件解码器结合长时记忆 $m_r$ → 先验动作 $\mu_r$（Eq. 4）
- 完整行为条件 $b_r \equiv (\ell, z_r, m_r, \mu_r)$ 定义动作纤维 $\mathcal{F}_r$

#### 模块 2：Handoff 状态前滚

**设计动机**：[[Handoff状态滚动|Handoff 上下文]]通过已知 outgoing 命令确定性前滚，而非预测未来场景，消解 handoff 歧义。

**具体实现**：
- 冻结动作-历史模型 $\mathcal{R}_{act}$ 逐步前滚 $k$ 步（Eq. 5）
- 取绝对状态 $\bar{h}_{r,k}$ 和差值状态 $\Delta h_{r,k}$ 双路编码（Eq. 6-7）
- 投影为 handoff 条件 $g_{r,k}$：绝对描述接管运动上下文，差值揭示推理期间的执行推进

#### 模块 3：Haar 有序生成

**设计动机**：正交 [[Haar动作分解|Haar 变换]]将动作块分解为互补坐标，精确可逆，scaffold 编码逐对运动水平，residual 恢复对内变化，两阶段均响应 handoff 条件。

**具体实现**：
- 共享流专家 $F_\theta$ 加阶段嵌入 $e_C, e_D$ 和掩码 $M_C, M_D$ 区分两阶段（Eq. 11-16）
- Scaffold 阶段：从噪声 $\varepsilon_C$ 生成 $\hat{C}$
- Residual 阶段：以 $\text{sg}(\hat{C})$ 为条件从 $\varepsilon_D$ 生成 $\hat{D}$（stop-gradient 保持阶段分工）
- 固定逆变换重建：$\hat{A}_{r+k} = W_H^T [\hat{C}, \hat{D}]$（无重建误差）

#### 模块 4：BICPO 续接偏好优化

**设计动机**：偏好优化基于条件排名而非全局平滑追求，防止通过静止或相位切换实现平滑假象。

**具体实现**：
- 在共享条件 $\chi_{r,k} = \{\Phi_r, z_r, m_r, \mu_r, g_k\}$ 下用不同流噪声采样候选对（Eq. 19）
- 用零阶跳变代价 $J_\text{jump}$ 和一阶趋势失配代价 $J_\text{trend}$ 确定偏好（Eq. 20-22）
- 参考相对 [[Flow-DPO]] 损失：仅更新 $P_\psi, W_g$ 及部分专家参数，语义通道和 Haar 变换冻结

---

## 关键公式

### 公式 1：[[Cross-Attention|指令路由注意力]]

$$
\tilde{V}_t = V_t + \text{Attn}(\text{LN}(V_t),\; L_b,\; L_b;\; M_l)
$$

**含义**：将视觉 token 通过语言特征做交叉注意力，过滤指令无关的视觉证据，再进行时序聚合。

**符号说明**：
- $V_t \in \mathbb{R}^{N_v \times d}$：时刻 $t$ 的视觉 token
- $L_b = \text{LN}(LW_l)$：线性投影后归一化的语言特征
- $M_l$：注意力掩码

### 公式 2：[[Mamba|Mamba 行为状态编码]]

$$
z_r = \mathcal{B}_\phi\bigl(\{\tilde{V}_t\}_{t \le r},\; \{u_t\}_{t < r}\bigr)
$$

**含义**：Mamba 因果历史编码器对指令路由后的观测序列和已执行命令做时序聚合，得到行为状态。

**符号说明**：
- $\tilde{V}_t$：指令路由后的视觉 token
- $u_t$：时刻 $t$ 的执行命令
- $z_r$：请求时刻 $r$ 的行为状态（编码指令相对进度）

### 公式 3：相位条件先验解码

$$
\mu_r = \mathcal{D}_\omega(z_r,\; m_r), \quad \mu_r \in \mathbb{R}^{H \times d_a}
$$

**含义**：解码器将行为状态 $z_r$ 和长时记忆 $m_r$ 融合生成先验动作序列，锚定行为语义。

**符号说明**：
- $m_r$：检索到的长时记忆
- $H$：动作块长度；$d_a$：动作维度

### 公式 4：Handoff 状态前滚

$$
h_{j+1} = \mathcal{R}_{act}(h_j,\; u_j), \quad j = r, \ldots, r+k-1
$$

$$
\bar{h}_{r,k} = \text{sg}(h_{r+k}), \quad \Delta h_{r,k} = \bar{h}_{r,k} - \text{sg}(h_r)
$$

$$
g_{r,k} = P_\psi\bigl([\text{LN}(\bar{h}_{r,k});\; \text{LN}(\Delta h_{r,k})]\bigr)
$$

**含义**：用冻结动作-历史模型确定性前滚 outgoing 命令，得到绝对 handoff 状态和推理期间的位移，投影为 handoff 条件。

**符号说明**：
- $h_j$：动作-历史模型隐状态
- $\mathcal{R}_{act}$：冻结的单步转移模型
- $\text{sg}(\cdot)$：stop-gradient
- $P_\psi$：可学习投影层

### 公式 5：[[Haar动作分解|一级 Haar 变换]]

$$
c_i = \frac{a_{2i} + a_{2i+1}}{\sqrt{2}}, \quad d_i = \frac{a_{2i} - a_{2i+1}}{\sqrt{2}}, \quad i = 0, \ldots, \tfrac{H}{2}-1
$$

等价矩阵形式：

$$
\begin{bmatrix} C \\ D \end{bmatrix} = W_H A, \quad A = W_H^T \begin{bmatrix} C \\ D \end{bmatrix}
$$

**含义**：正交 Haar 变换将动作块分解为 scaffold（逐对均值）和 residual（逐对差值），精确可逆，$\|W_H\|=1$ 保持能量不变。

**符号说明**：
- $C \in \mathbb{R}^{H/2 \times d_a}$：scaffold 系数，编码逐对运动水平
- $D \in \mathbb{R}^{H/2 \times d_a}$：residual 系数，恢复对内变化
- $W_H$：正交 Haar 矩阵，无可学习参数

### 公式 6：[[Flow Matching|线性流路径]]

$$
x_\tau = (1-\tau)y + \tau\varepsilon, \quad \frac{\partial x_\tau}{\partial \tau} = v^* = \varepsilon - y, \quad \tau \in [0,1]
$$

**含义**：Flow Matching 的线性插值路径，$\tau=0$ 为数据，$\tau=1$ 为高斯噪声，速度场目标为 $\varepsilon-y$。

**符号说明**：
- $y$：目标系数（$C^*$ 或 $D^*$）
- $\varepsilon \sim \mathcal{N}(0,I)$：高斯噪声
- $\tau$：流时间步

### 公式 7：Scaffold 阶段生成

$$
r_C = e([\varepsilon_C, \mathbf{0}]) + W_\mu[\mu_r^C, \mathbf{0}] + W_g g_k + e_C
$$

$$
\hat{v}_C = M_C \odot F_\theta(r_C,\; 1,\; \Phi_r)
$$

$$
\hat{C} = \varepsilon_C - \hat{v}_C
$$

**含义**：从噪声 $\varepsilon_C$ 出发，以先验 scaffold $\mu_r^C$ 和 handoff 条件 $g_k$ 为条件，由共享专家 $F_\theta$ 预测速度场，生成 scaffold 系数。

**符号说明**：
- $e_C$：scaffold 阶段嵌入
- $M_C$：scaffold 掩码（遮蔽残差位置）
- $W_g$：handoff 条件投影（零初始化）
- $\Phi_r$：多模态上下文（视觉+语言）

### 公式 8：Residual 阶段生成（条件于 scaffold）

$$
r_D = e([\text{sg}(\hat{C}),\; \varepsilon_D]) + W_\mu[\mu_r^C,\; \mu_r^D] + W_g g_k + e_D
$$

$$
\hat{v}_D = M_D \odot F_\theta(r_D,\; 1,\; \Phi_r)
$$

$$
\hat{D} = \varepsilon_D - \hat{v}_D
$$

**含义**：以 stop-gradient 的已生成 scaffold $\hat{C}$ 为条件，生成 residual 系数；stop-gradient 防止残差损失改写 scaffold 的梯度。

**符号说明**：
- $\text{sg}(\hat{C})$：已生成 scaffold，stop-gradient 截断
- $e_D$：residual 阶段嵌入
- $M_D$：residual 掩码

### 公式 9：动作块重建

$$
\hat{A}_{r+k} = W_H^T [\hat{C},\; \hat{D}]
$$

**含义**：通过固定正交逆 Haar 变换从两阶段系数精确重建完整动作块，无重建误差。

### 公式 10：监督训练损失

$$
\mathcal{L}_\text{struct} = \frac{1}{2}\Bigl(\|\hat{v}_C - (\varepsilon_C - C^*)\|_2^2 + \|\hat{v}_D - (\varepsilon_D - D^*)\|_2^2\Bigr)
$$

**含义**：对 scaffold 和 residual 速度场分别计算 Flow Matching 损失并平均，与地面真值系数对齐。

**符号说明**：
- $C^*, D^*$：地面真值 Haar 系数
- $\hat{v}_C, \hat{v}_D$：两阶段预测速度场

### 公式 11：续接代价函数

$$
J_\text{jump}(A) = \|a_0 - u_{r+k-1}\|_W^2
$$

$$
J_\text{trend}(A) = \|(a_0 - u_{r+k-1}) - (u_{r+k-1} - u_{r+k-2})\|_W^2
$$

$$
J_\text{cont}(A) = \lambda_0 J_\text{jump}(A) + \lambda_1 J_\text{trend}(A)
$$

**含义**：零阶 handoff 跳变代价衡量新块起点与最后执行命令之间的位置不连续，一阶趋势失配代价衡量加速度层面的不连续，两者加权为总续接代价。

**符号说明**：
- $a_0$：候选块的第一个命令
- $u_{r+k-1}, u_{r+k-2}$：最后两个已执行命令
- $W \succeq 0$：通道权重对角矩阵
- $\lambda_0, \lambda_1$：权重系数

### 公式 12：分阶段 Flow 能量

$$
\mathcal{E}_\theta^C(C|\chi) = \mathbb{E}_{\tau,\varepsilon_C}\|v_\theta^C(x_\tau^C, \tau, \chi) - (\varepsilon_C - C)\|_2^2
$$

$$
\mathcal{E}_\theta^D(D|C,\chi) = \mathbb{E}_{\tau,\varepsilon_D}\|v_\theta^D(x_\tau^D, \tau; C, \chi) - (\varepsilon_D - D)\|_2^2
$$

$$
\mathcal{E}_\theta(A|\chi) = \frac{1}{2}\bigl[\mathcal{E}_\theta^C(C|\chi) + \mathcal{E}_\theta^D(D|C,\chi)\bigr]
$$

**含义**：将候选动作块 $A$ 的两阶段 Flow 期望误差之和定义为整体候选兼容性能量，能量越低兼容性越高。

**符号说明**：
- $\chi_{r,k}$：共享生成条件 $\{\Phi_r, z_r, m_r, \mu_r, g_k\}$
- $\mathcal{E}_\theta^C, \mathcal{E}_\theta^D$：scaffold 和 residual 阶段能量

### 公式 13：[[Flow-DPO|参考相对 Flow-DPO]] 损失

$$
\Delta_\theta = \bigl[\mathcal{E}_\theta(A^-) - \mathcal{E}_\theta(A^+)\bigr] - \bigl[\mathcal{E}_\text{ref}(A^-) - \mathcal{E}_\text{ref}(A^+)\bigr]
$$

$$
\mathcal{L}_\text{pref} = -q \log \sigma(\beta \Delta_\theta)
$$

$$
\mathcal{L} = \mathcal{L}_\text{pref} + \lambda_+ \mathcal{E}_\theta(A^+) + \lambda_\text{rep} \mathcal{L}_\text{replay}
$$

**含义**：以冻结参考策略为基准计算偏好变化，防止策略偏离太远；偏好候选能量锚定 $A^+$ 支持；重放损失 $\mathcal{L}_\text{replay}$ 保持原始策略。

**符号说明**：
- $A^+, A^-$：偏好（更连续）和非偏好候选
- $\mathcal{E}_\text{ref}$：冻结参考策略能量
- $q$：跟随排名 margin 的置信权重
- $\beta$：DPO 温度；$\sigma$：sigmoid

---

## 关键图表

### Figure 1：动机与动作纤维视图

![Figure 1](https://arxiv.org/html/2608.13924v1/fig1_motivation.png)

**说明**：左：现有 VLA 范式（改进记忆、减少生成步骤、通用平滑正则化）仍保持上下文到动作块的直接映射。中：BICPO-VLA 显式提取行为条件。右：指令确定行为纤维，Haar 坐标组织实现，前滚 handoff 状态加 BICPO 选择最顺畅实现。下：仿真和真实场景均有提升。

### Figure 2：BICPO-VLA 整体架构

![Figure 2](https://arxiv.org/html/2608.13924v1/fig2_framework.png)

**说明**：左侧为行为识别模块（指令感知视觉-动作编码 + 行为记忆）；中间为 Haar 结构化生成器（scaffold → residual 有序生成，精确重建）；右侧为 BICPO 续接优化（前滚 handoff 状态，同纤维候选对比）。

### Figure 3：仿真定性对比

![Figure 3](https://arxiv.org/html/2608.13924v1/fig3_simulation.png)

**说明**：BICPO-VLA（上行）vs [[pi0.5|π₀.₅]]（下行）在两个多阶段任务上的轨迹。BICPO-VLA 在抓取→放置→按钮激活的相位过渡中保持预期行为；圆圈标注最终结果，π₀.₅ 多个关键点失败。

### Figure 4：真实场景评估

![Figure 4](https://arxiv.org/html/2608.13924v1/fig4_realworld.png)

**说明**：上：BICPO-VLA 和 π₀.₅ 在牛奶放置、纸球处理任务上的轨迹展示；下：六个任务成功率对比，BICPO-VLA 全面领先，领先最强基线 7-11 个点。

### Table 1：CALVIN ABC→D 长序列结果

| 方法 | 1 | 2 | 3 | 4 | 5 | Avg. |
|------|------|------|------|------|------|------|
| RoboVLM | 98.0 | 93.6 | 85.4 | 77.8 | 70.4 | 4.25 |
| ReconVLA | 95.6 | 87.6 | 76.9 | 69.3 | 64.1 | 3.95 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| UP-VLA | 92.8 | 86.5 | 81.5 | 76.9 | 69.9 | 4.08 |
| RoboDual | 94.4 | 82.7 | 72.1 | 62.4 | 54.4 | 3.66 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| π₀ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| π₀.₅ | 94.8 | 88.9 | 84.1 | 79.7 | 72.1 | 4.21 |
| B-VLA | 96.0 | 92.0 | 87.3 | 82.9 | 77.3 | 4.36 |
| **BICPO-VLA（Ours）** | **98.9** | **95.4** | **91.0** | **86.0** | **80.7** | **4.52** |

**关键发现**：BICPO-VLA 在 11 个基线中全面领先，平均序列长度 4.52（前者最强 4.36），5/5 子任务完成率 80.7%（前者 77.3%）；提升在 1→5 子任务均持续，而非随链增长消失，说明有效减少跨阶段过渡的复合误差。

### Table 2：RoboTwin 2.0 Hard 双臂操作成功率（%）

| 方法 | AB | CA | CB | DB | GR | MP | P2R | PB | BF | CP | Avg.I | Overall |
|------|----|----|----|----|----|----|----|----|----|----|----|---------|
| RDT | 75 | 12 | 9 | 32 | 43 | 11 | 1 | 2 | 27 | 17 | 34.2 | 22.9 |
| ACT | 23 | 4 | 3 | 1 | 25 | 0 | 0 | 0 | 0 | 1 | 11.2 | 5.7 |
| DP | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0.5 |
| DP3 | 3 | 14 | 3 | 53 | 2 | 3 | 0 | 1 | 18 | 1 | 15.0 | 9.8 |
| π₀ | 56 | 11 | 6 | 24 | 80 | 22 | 6 | 4 | 4 | 45 | 35.4 | 25.8 |
| π₀.₅ | 75 | 44 | 64 | 69 | 82 | 32 | 19 | 28 | 46 | 55 | 66.8 | 51.4 |
| B-VLA | 83 | 52 | 77 | 77 | 90 | 41 | 25 | 36 | 61 | 62 | 75.8 | 60.4 |
| **BICPO-VLA** | **88** | **56** | **82** | **83** | **94** | **49** | **30** | **40** | **67** | **69** | **80.6** | **65.8** |

**关键发现**：全部 10 个任务均提升 4-8 点，overall 从 60.4% 提升至 65.8%；在难度最大的 P2R（place-to-rack）和 PB 任务上也有明显改善。

### Table 3：LIBERO 续接质量与目标迁移性

| 方法 | SR (%) ↑ | $J_\text{jump}$ ↓ | $J_\text{trend}$ ↓ |
|------|----------|-----------|-----------|
| **BICPO-VLA** | **99.1** | **2.37** | **6.40** |
| BICPO-VLA w/o DPO | 98.8 | 3.01 | 8.07 |
| BICPO-VLA with chosen-only SFT | 97.9 | 2.85 | 6.80 |
| π₀.₅-FM | 96.9 | 4.22 | 8.50 |
| π₀.₅-FM + chosen-only SFT | 49.1 | 3.13 | 6.90 |
| π₀.₅-FM + continuity DPO | 97.1 | 2.52 | 6.70 |
| Legato | 97.5 | 2.98 | 7.60 |
| Legato + continuity DPO | 97.6 | 2.51 | 6.60 |
| RTC | 97.3 | 3.07 | 7.80 |
| RTC + continuity DPO | 97.4 | 2.58 | 6.90 |

**关键发现**：DPO 使 BICPO-VLA 跳变代价降低 21.3%、趋势代价降低 20.7%；迁移到 π₀.₅/Legato/RTC 后续接代价降低 15.8-40.3%（跳变）和 11.5-21.2%（趋势），SR 仅微升 0.1-0.2 点，说明续接改善主要体现在平滑性而非成功率上；直接 SFT 导致 π₀.₅ SR 从 96.9% 骤降至 49.1%，验证偏好排名优于无条件平滑回归。

### Table 4：LIBERO 推理延迟鲁棒性

| 指标 | k=3 | k=4 | k=5 | Random |
|------|-----|-----|-----|--------|
| SR (%) ↑ | 99.1 | 98.9 | 98.8 | 98.8 |
| $J_\text{jump}$ ↓ | 2.37 | 2.47 | 2.50 | 2.46 |
| $J_\text{trend}$ ↓ | 6.40 | 6.21 | 6.23 | 6.17 |

**关键发现**：固定延迟 k=3-5 或随机延迟下，SR 仅变化 0.3 点，handoff 代价稳定，前滚条件具有良好泛化性，无需为每个延迟值单独设计续接规则。

### Table 5：组件消融（CALVIN ABC→D）

| 变体 | 1 | 2 | 3 | 4 | 5 | Avg. |
|------|------|------|------|------|------|------|
| Full | 98.9 | 95.4 | 91.0 | 86.0 | 80.7 | 4.520 |
| w/o Haar | 98.2 | 94.8 | 90.7 | 85.2 | 79.6 | 4.490 |
| w/o BICPO | 97.8 | 95.0 | 90.5 | 85.4 | 78.9 | 4.480 |
| w/o Instr. | 97.0 | 92.5 | 87.9 | 84.0 | 78.2 | 4.396 |

**关键发现**：去除指令感知影响最大（4.52→4.40），三个组件各有互补贡献；不同组件影响不同子任务层数，支持行为识别、实现结构、handoff 选择的三阶段分工设计。

### Table 6：指令介入位置消融（CALVIN ABC→D）

| 变体 | 1 | 2 | 3 | 4 | 5 | Avg. |
|------|------|------|------|------|------|------|
| 视觉路由（本文） | 98.9 | 95.4 | 91.0 | 86.0 | 80.7 | 4.520 |
| Query 融合 | 98.6 | 94.4 | 89.2 | 85.1 | 80.3 | 4.476 |
| Memory 融合 | 97.5 | 93.1 | 88.3 | 84.2 | 78.6 | 4.417 |

**关键发现**：在感知层（历史聚合前）对视觉 token 做指令路由，效果优于在 Query 或 Memory 阶段融合，分别领先 0.044 和 0.103 Avg. Len；指令应尽早介入感知而非在高层融合。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[CALVIN]] ABC→D | 跨环境长序列（5子任务链） | 长时组合评测 |
| [[RoboTwin 2.0]] Hard | 10 个双臂操作任务 | 双臂灵巧操作 |
| [[LIBERO]] | 近饱和成功率，重点评 handoff | 续接质量评测 + 目标迁移 |
| 6 个真实任务 | 少数据真实场景 | 真实部署验证 |

### 实现细节

- **行为编码器**：Mamba 因果块 + 倒反视觉运动校准（reciprocal visuomotor calibration）
- **生成器**：共享流专家 $F_\theta$，阶段嵌入区分两阶段，掩码分离 scaffold/residual
- **训练阶段**：模仿学习 → 续接预热 $\mathcal{L}_\text{struct}$ → 离线 [[Flow-DPO|DPO]] $\mathcal{L}$
- **冻结组件**：语义通路（$z_r, m_r, \mu_r$）、Haar 变换、参考策略
- **更新组件**：$P_\psi, W_g$（零初始化），部分专家参数

### 真实场景

6 个有限数据真实任务平均成功率 69.3%（B-VLA 60.2%、π₀.₅ 47.3%、OpenVLA-OFT 33.3%），每个任务领先最强基线 7-11 点；关键 handoff 处 BICPO-VLA 成功保持子目标意图。

---

## 批判性思考

### 优点

1. **优雅的问题分解**：行为语义-Haar 结构-handoff 适配三层分离有清晰的数学保证，每层有独立不变量，设计极具说服力。
2. **DPO 目标高度可迁移**：只需将偏好 objective 植入 π₀.₅/Legato/RTC 原有 Flow 能量，无需修改架构，工程实用性强。
3. **Haar 变换零损失**：正交变换无重建误差，信息无损，且无额外参数，简洁优雅。

### 局限性

1. **假设 outgoing 命令已知**：前滚需要完整 outgoing 动作块，若执行中途被中断或有外部扰动则失效；论文也明确说明这是未来工作。
2. **延迟假设**：要求延迟 $k$ 小于剩余块长度 $H-k$，长推理延迟或大延迟变化时适用性受限。
3. **Haar 局限于偶数块长度**：当 $H$ 为奇数时需要特殊处理（论文未提及细节）。
4. **评测基准偏仿真**：LIBERO 近饱和使主要收益落在 handoff 代价而非成功率，真实场景数量（6 个任务）较少。

### 潜在改进方向

1. 处理外部扰动 handoff（perception-conditioned rollout）
2. 多级 Haar 分解（2 级+）以捕获更丰富的频率结构
3. 在线偏好更新（active preference learning from robot feedback）

### 可复现性评估

- [ ] 代码开源（未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节基本完整（损失函数、超参数命名）
- [x] 数据集可获取（CALVIN/LIBERO/RoboTwin 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[pi0.5\|π₀.₅]]：宿主 Flow Matching 策略，BICPO DPO 迁移目标之一
- [[pi0\|π₀]]：Flow Matching 动作生成奠基工作
- [[Diffusion Policy]]：扩散策略基线，动作块生成的前置工作
- [[DPO]]：直接偏好优化原始目标，BICPO 的理论基础
- [[Flow Matching]]：线性路径 Flow 生成框架，BICPO-VLA 生成器所用

### 对比

- [[B-VLA]]：最强直接基线，BICPO-VLA 在全部 benchmark 上超越
- [[Asynchronous Inference\|RTC]] / Legato：异步执行方向对比方法，DPO 目标可迁移至此

### 方法相关

- [[Haar动作分解]]：本文核心技术，正交 Haar 子空间分解
- [[行为条件化动作纤维]]：本文核心形式化概念
- [[Flow-DPO]]：参考相对续接偏好优化
- [[Handoff状态滚动]]：outgoing 动作前滚机制
- [[Mamba]]：因果历史编码器架构
- [[Action Chunking]]：VLA 动作块生成范式

### 数据集相关

- [[CALVIN]]：长序列多子任务 benchmark
- [[LIBERO]]：续接质量评测基准
- [[RoboTwin 2.0]]：双臂操作 Hard 任务集

---

## 速查卡片

> [!summary] BICPO-VLA
> - **核心**：将行为语义、Haar 动作结构、handoff 接管适配三层分离，解决异步 VLA 的 request-to-handoff gap
> - **方法**：指令感知 Mamba 编码 + Haar scaffold/residual 两阶段生成 + 参考相对 Flow-DPO 续接偏好优化
> - **结果**：CALVIN 4.52 Avg.（SOTA）；RoboTwin Overall 65.8%；LIBERO handoff 代价降低 20%+；DPO 可迁移到 π₀.₅/Legato/RTC
> - **代码**：未开源

---

*笔记创建时间: 2026-08-18*

---
title: "LAW: Unifying Language, Action, and World Modeling for Agentic Physical Intelligence"
method_name: "LAW"
authors: [Jiajun Cao, Kai Tang, Bohua Zhang, Zezhong Qian, Ying Li, Xiaozhu Ju]
year: 2026
venue: arXiv
tags: [vla, world-model, robot-manipulation, agentic-ai, value-learning, mixture-of-experts, subtask-decomposition, long-horizon]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.23334
created: 2026-10-20
---

# 论文笔记：LAW: Unifying Language, Action, and World Modeling for Agentic Physical Intelligence

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University (State Key Laboratory of Multimedia Information Processing); Beijing Innovation Center of Humanoid Robotics |
| 日期 | October 2026 |
| 项目主页 | [https://hey-cjj.github.io/LAW-project/](https://hey-cjj.github.io/LAW-project/) |
| 对比基线 | [[π0.5]], [[Fast-WAM]], [[LingBot-VLA-2.0]], [[Qwen-RobotManip]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.23334) / [Code](https://github.com/hey-cjj/LAW) |

---

## 一句话总结

> LAW 通过语义子任务进度统一语言推理、动作生成与世界建模，在 50 个 RoboTwin 任务中达到 94.21% 成功率，真实机器人长时程任务上以 65% 超越 π0.5 的 28.33%。

---

## 核心贡献

1. **LAW 闭环智能体框架**: 以[[语义子任务分解|语义子任务进度]]作为统一信号，将[[VLA|视觉-语言-动作模型]]与[[World Action Model|世界动作模型 (WAM)]]协调起来，实现规划-执行-状态感知的闭环。
2. **混合专家 VLM**: 基于 Qwen3-VL-2B-Instruct，冻结主干网络并为规划器、状态专家、价值专家分别挂载 [[LoRA]] 适配器，三个专家共享特征提取。
3. **MoTE（Mixture-of-Transformer Experts）**: 在 WAM 的混合注意力之后插入[[MoTE|混合 Transformer 专家]]层，每个模态拥有 2 个私有专家 + 1 个共享专家，top-1 路由，残差缩放 α=0.1，同时建模视频与动作。
4. **双流层次记忆**: 为 VLM 和 WAM 分别维护三类视觉帧：最近 5 帧（步幅 4）、子任务边界关键帧（全分辨率）、空间池化的长程历史帧，解决长时程观测遗忘问题。
5. **子任务局部价值学习**: 在每个完成的子任务末尾给予奖励 1，回报在子任务边界截断，使得失败轨迹的已完成前缀仍可提供监督；用 [[HL-Gauss]] 软标签 + 交叉熵损失训练价值头；动作融合采用 [[Markley 四元数均值]]。

---

## 问题背景

### 要解决的问题

现有机器人策略缺乏高层任务推理与低层动作生成之间的紧密协同：VLA 缺少自检机制（不知道子任务何时完成），WAM 缺少语义引导（无法自主规划和切换子任务），导致长时程操作成功率低。

### 现有方法的局限

- **纯 VLA**（如 [[π0.5]]）：语言推理能力强，但没有进度感知和世界建模，长时程任务频繁失败。
- **纯 WAM**（如 [[Fast-WAM]]）：生成质量高，但缺乏语义层规划，无法自主处理多阶段任务。
- **两阶段流水线**：规划与执行分离，状态反馈延迟，错误无法实时修正。

### 本文的动机

以**语义子任务进度**作为通用接口：VLM 预测剩余子任务序列，状态专家异步判断当前子任务是否完成，完成即切换；同时用价值专家从 WAM 生成的多个动作候选中选出最优并融合。三者通过统一信号耦合，形成闭环。

---

## 方法详解

### 模型架构

LAW 采用**双塔异步闭环**架构：

- **高频分支（WAM 侧）**：WAM 每步对当前观测和子任务 prompt 采样 N 个动作候选（窗口长 32），价值专家打分后 top-K 融合执行
- **低频分支（VLM 侧）**：Qwen3-VL-2B 驱动的混合专家 VLM 异步处理：规划器维护子任务序列，状态专家检测完成，完成时更新子任务 prompt
- **输入**：语言目标 $g$ + 当前观测 $o_t$ + 子任务 prompt $c_t$ + [[双流层次记忆|视觉记忆]] $\mathcal{M}_t^{\mathrm{V}}, \mathcal{M}_t^{\mathrm{W}}$
- **Backbone（VLM）**：Qwen3-VL-2B-Instruct（冻结） + 三路 [[LoRA]]
- **Backbone（WAM）**：Fast-WAM / OpenWAM-α + [[MoTE]] 层
- **输出**：融合后的[[Action Chunking|动作块]] $\hat{A}_t$（长度 32）
- **总参数**：VLM ~2B；WAM 规模同 Fast-WAM / OpenWAM-α

### 核心模块

#### 模块 1：LAW 智能体循环（Sec. 3.1）

**设计动机**：利用[[语义子任务分解]]将长时程任务分解为可验证的短时段，实现进度感知的闭环控制。

**具体实现**：
- 规划器用 [[自回归生成]] 预测剩余子任务列表 $\mathcal{S}_t$
- WAM 为当前子任务采样 N 个动作候选，价值专家对每个候选打分
- 取 top-K 候选按[[Markley 四元数均值]]加权融合
- 状态专家 **异步** 预测当前子任务完成度 $\hat{d}_t$，超过 0.5 触发 Advance 事件，切换到下一子任务

#### 模块 2：混合专家 VLM（Sec. 3.2）

**设计动机**：三类任务（规划/状态/价值）的输入格式相近，共享视觉编码可降低参数量，但各任务语义差异较大，用独立 [[LoRA]] 解耦。

**具体实现**：
- 规划器 LoRA：以 cross-entropy 训练子任务序列生成（语言模型损失 $\mathcal{L}_{\mathrm{plan}}$）
- 状态专家 LoRA：二值分类头，同时预测 **完成度** $\hat{d}_t$ 和 **失败标志** $\hat{f}_t$，用 BCE 损失
- 价值专家 LoRA：101 个分类 bin 的离散价值头（[[HL-Gauss]] 分布）

#### 模块 3：MoTE（Sec. 3.3）

**设计动机**：视频 token 与动作 token 的特征分布差异大，私有专家处理模态专有信息，共享专家捕获跨模态公共特征，同时降低 expert collapse 风险。

**具体实现**：
- 在 WAM Transformer 的混合注意力之后插入 MoTE 层
- 每个模态 2 个私有专家 + 1 个全局共享专家
- [[Top-K 路由]]（K=1）：路由权重通过 $W_r \cdot \mathrm{LN}(\mathbf{H})$ 计算后 softmax
- 残差缩放系数 $\alpha = 0.1$
- 辅助平衡损失 $\mathcal{L}_{\mathrm{balance}}$ 防止路由崩溃

#### 模块 4：双流层次记忆（Sec. 3.4）

**设计动机**：单帧输入会遗忘长时程上下文，但全部历史帧太贵；不同时间跨度的信息有不同的精度需求。

**具体实现**：
- **近期帧**：最近 5 帧，步幅 4（高频局部细节）
- **子任务边界帧**：每次子任务切换时存入，保留全分辨率（中期语义锚点）
- **长程帧**：历史帧的空间池化（低分辨率全局上下文）
- VLM 和 WAM 各维护一套独立记忆流（$\mathcal{M}_t^{\mathrm{V}}$ 和 $\mathcal{M}_t^{\mathrm{W}}$）

#### 模块 5：子任务局部价值学习（Sec. 3.5）

**设计动机**：长时程任务中 episode 级奖励稀疏；失败轨迹中的已完成子任务段存在大量有效监督，不应丢弃。

**具体实现**：
- 仅在每个子任务的末尾动作处给奖励 1，其余为 0
- 回报在子任务边界截断（不跨子任务传播），已完成前缀独立训练
- 价值目标用 [[HL-Gauss]] 转为 101 bin 软标签分布
- 执行时：WAM 生成 N 个候选，按价值加权取 top-K，再用[[Markley 四元数均值]]融合四元数部分

---

## 关键公式

### 公式 1：[[语义子任务分解|规划器输出]]

$$
\mathcal{S}_t = \mathrm{VLM}_{\mathrm{plan}}(g, o_t, \mathcal{M}_t^{\mathrm{V}}) = (s_{t,1}, \ldots, s_{t,L_t})
$$

**含义**：规划器在时刻 $t$ 给出剩余子任务的有序列表。

**符号说明**：
- $g$：语言目标
- $o_t$：当前观测
- $\mathcal{M}_t^{\mathrm{V}}$：VLM 侧视觉记忆
- $L_t$：剩余子任务数量

### 公式 2：[[World Action Model|WAM 动作采样]]

$$
\{A_t^{(n)}\}_{n=1}^{N} = \mathrm{WAM}_{\mathrm{act}}(o_t, c_t, \mathcal{M}_t^{\mathrm{W}}; \epsilon_n)
$$

**含义**：WAM 以当前子任务 prompt 和随机噪声为条件，生成 N 个候选动作块。

**符号说明**：
- $c_t$：当前子任务 prompt
- $\mathcal{M}_t^{\mathrm{W}}$：WAM 侧视觉记忆
- $\epsilon_n$：第 n 次采样的随机噪声
- $A_t^{(n)}$：长度 32 的动作块候选

### 公式 3：[[Value Function|价值函数]] 评分与动作融合

$$
x_t = (g, o_t, c_t, \mathcal{M}_t^{\mathrm{V}})
$$

$$
q_t^{(n)} = Q_\theta(x_t, A_t^{(n)}), \quad \hat{A}_t = \operatorname{Fuse}_{K,T}\big(\{A_t^{(n)}, q_t^{(n)}\}_{n=1}^{N}\big)
$$

**含义**：价值专家对每个候选评分，取 top-K 按温度加权融合。

**符号说明**：
- $Q_\theta$：价值专家（101 bin 离散价值头）
- $K=8$：保留的候选数
- $T$：融合温度参数
- $\hat{A}_t$：最终执行的融合动作块

### 公式 4：[[语义子任务分解|状态专家]] 完成度判断

$$
\hat{d}_t = \mathrm{VLM}_{\mathrm{state}}(g, o_t, c_t, \mathcal{M}_t^{\mathrm{V}})
$$

$$
e_t = \begin{cases} \text{Advance}, & \hat{d}_t \geq 0.5 \\ \text{Continue}, & \text{otherwise} \end{cases}
$$

**含义**：状态专家异步预测当前子任务完成度，超过阈值则触发子任务切换。

### 公式 5：[[混合专家|规划器损失]]

$$
\mathcal{L}_{\mathrm{plan}} = -\sum_{i=1}^{|\mathcal{S}|} \log p_\theta(s_i \mid g, o_t, \mathcal{M}_t^{\mathrm{V}}, s_{<i})
$$

**含义**：自回归语言模型损失，监督剩余子任务序列的逐词预测。

### 公式 6：[[状态预测|状态专家损失]]

$$
\mathcal{L}_{\mathrm{state}} = \lambda_d \cdot \mathrm{BCE}(\hat{d}_t, d_t) + \lambda_f \cdot \mathrm{BCE}(\hat{f}_t, f_t)
$$

**含义**：同时监督完成度和失败标志的二值交叉熵损失。

**符号说明**：
- $d_t$：完成度真值（0 或 1）
- $f_t$：失败标志真值
- $\lambda_d, \lambda_f$：权重系数

### 公式 7：[[HL-Gauss|离散价值头输出]]

$$
Q_\theta(x_t, A_t) = \sum_{b=1}^{101} z_b \cdot p_\theta(b \mid x_t, A_t)
$$

**含义**：101 个 bin 的期望值作为标量 Q 值输出（HL-Gauss 表示）。

**符号说明**：
- $z_b$：第 $b$ 个 bin 的中心值
- $p_\theta(b|\cdot)$：模型输出的 bin 概率分布

### 公式 8：[[MoTE|混合 Transformer 专家路由]]

$$
\mathbf{w} = \operatorname{TopKSoftmax}(W_r \operatorname{LN}(\mathbf{H}))
$$

$$
\tilde{\mathbf{H}} = \mathbf{H} + \alpha \sum_{e \in \mathcal{E}} w_e E_e(\operatorname{LN}(\mathbf{H}))
$$

**含义**：对隐状态做 top-1 路由，经专家变换后残差相加（缩放系数 α=0.1）。

**符号说明**：
- $W_r$：路由权重矩阵
- $\mathcal{E}$：激活专家集合（私有专家 + 共享专家）
- $\alpha = 0.1$：残差缩放系数
- $E_e$：第 $e$ 个专家的 FFN 变换

### 公式 9：[[World Action Model|WAM 总损失]]

$$
\mathcal{L}_{\mathrm{WAM}} = \lambda_v \mathcal{L}_{\mathrm{video}} + \lambda_a \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{bal}} \mathcal{L}_{\mathrm{balance}} + \lambda_z \mathcal{L}_z
$$

**含义**：WAM 联合训练视频预测、动作预测、MoE 路由平衡、辅助正则化。

**符号说明**：
- $\mathcal{L}_{\mathrm{video}}$：视频帧预测损失
- $\mathcal{L}_{\mathrm{action}}$：动作序列预测损失
- $\mathcal{L}_{\mathrm{balance}}$：MoE 路由负载均衡损失
- $\mathcal{L}_z$：辅助正则项

### 公式 10：[[双流层次记忆|层次视觉记忆]]

$$
\mathcal{M}_t^{X} = \Big[\{\bar{\mathbf{x}}_{c_i}^{X}\}_{i=1}^{m};\ \{\mathbf{x}_{\beta_i}^{X}\}_{i=1}^{p};\ \{\mathbf{x}_{t-4j}^{X}\}_{j=1}^{5}\Big], \quad X \in \{\mathrm{V}, \mathrm{W}\}
$$

**含义**：三类帧拼接构成记忆：空间池化长程帧、子任务边界全分辨率帧、最近 5 帧（步幅 4）。

**符号说明**：
- $\bar{\mathbf{x}}_{c_i}^{X}$：空间池化的长程历史帧（$m$ 帧）
- $\mathbf{x}_{\beta_i}^{X}$：子任务边界关键帧（$p$ 帧，全分辨率）
- $\mathbf{x}_{t-4j}^{X}$：最近帧（步幅 4，5 帧）

### 公式 11：[[子任务局部价值学习|价值回报目标]]

$$
y_t = \sum_{i=0}^{h_t-1} \gamma^i r_{t+i} + b_t \cdot \gamma^{h_t} Q_{\bar{\theta}}\big(x_{t+h_t}, A^{\mathrm{data}}_{t+h_t}\big)
$$

**含义**：在子任务边界截断的 n 步回报，$b_t=0$ 表示子任务已结束（已完成或已失败）。

**符号说明**：
- $h_t$：从当前时刻到下一子任务边界的步数
- $b_t \in \{0, 1\}$：是否跨子任务传播价值
- $Q_{\bar{\theta}}$：目标价值网络（EMA 参数）

### 公式 12：[[HL-Gauss|价值软标签分布]]

$$
\rho_{t,b} \propto \exp\!\left[-\frac{(z_b - \tilde{y}_t)^2}{2\sigma^2}\right], \quad \mathcal{L}_{\mathrm{value}} = -\mathbb{E}_{\mathcal{D}}\!\left[\sum_{b=1}^{101} \rho_{t,b} \log p_\theta(b \mid x_t, A_t)\right]
$$

**含义**：将标量回报 $\tilde{y}_t$ 转为 Gauss 软分布，再用交叉熵训练价值头。

**符号说明**：
- $z_b$：第 $b$ 个 bin 的中心值
- $\sigma$：Gauss 扩散宽度
- $\tilde{y}_t$：归一化后的回报目标

### 公式 13：[[Markley 四元数均值|加权动作融合]]

$$
w_n = \frac{\exp(Q_\theta(x_t, A_t^{(n)}) / T)}{\sum_{k \in \mathcal{E}_t} \exp(Q_\theta(x_t, A_t^{(k)}) / T)}, \quad n \in \mathcal{E}_t
$$

$$
M = \sum_{n \in \mathcal{E}_t} w_n q_n q_n^{\top}, \quad \hat{q} \in \operatorname*{arg\,max}_{\|q\|_2=1} q^{\top} M q
$$

**含义**：按价值加权融合 top-K 候选的平移部分（加权均值），四元数部分用 Markley 方法（最大特征向量）融合。

**符号说明**：
- $\mathcal{E}_t$：top-K 候选集合（$K=8$）
- $T$：softmax 温度
- $q_n$：第 $n$ 个候选的旋转四元数
- $M$：四元数外积加权和矩阵
- $\hat{q}$：融合后的四元数（$M$ 的最大特征向量）

---

## 关键图表

### Figure 1：Overview / 系统概览

![Figure 1 - LAW Overview](https://arxiv.org/html/2610.23334v1/figures/Teaser.png)

**说明**：LAW 智能体的整体概览。展示了 VLM 侧的规划-状态感知循环与 WAM 侧的多候选采样-价值融合执行循环如何以[[语义子任务分解|语义子任务进度]]为接口耦合成闭环系统。

### Figure 2：LAW 架构与循环流程

![Figure 2 - LAW Architecture](https://arxiv.org/html/2610.23334v1/figures/LAW_main.png)

**说明**：详细展示 LAW 的端到端架构。包括混合专家 VLM（规划器/状态专家/价值专家三路 [[LoRA]]）与[[MoTE]]增强的 WAM，以及两者之间的异步通信机制。

### Figure 3：语义子任务标注可视化

![Figure 3 - Subtask Visualization](https://arxiv.org/html/2610.23334v1/figures/Subtask.png)

**说明**：在 RoboTwin 任务（Handover Block、Put Bottles Dustbin、Stack Bowls Three、Blocks Ranking RGB）上可视化语义子任务的时间分割边界和失败案例。说明子任务粒度的选择对模型性能的影响。

### Figure 4：混合频率协调控制

**说明**（无图像 URL）：展示 LAW 的混合频率协调机制，VLM 以低频更新子任务 prompt，WAM 以高频执行动作，二者通过状态专家的异步信号同步，实现进度感知控制。

### Figure 5：MoTE 架构

![Figure 5 - MoTE Architecture](https://arxiv.org/html/2610.23334v1/moe.png)

**说明**：[[MoTE]] 的详细结构。插入混合注意力之后，每个 token 根据路由权重被分配到视频私有专家、动作私有专家或共享专家，残差相加后输出。

### Figure 6：双流层次记忆与注意力模式

![Figure 6 - Hierarchical Memory](https://arxiv.org/html/2610.23334v1/figures/memory.png)

**说明**：[[双流层次记忆]]的结构图。展示三类帧（近期/边界/长程池化）在 WAM 中的注意力模式，以及 VLM 和 WAM 各自独立维护记忆流的双流设计。

### Figure 7：真实机器人实验结果

![Figure 7 - Real World Results](https://arxiv.org/html/2610.23334v1/figures/real_world_results_final.png)

**说明**：在三个真实长时程操作任务上的结果对比（Puzzle 3 阶段、Domino 5 阶段、Organize 9 阶段）。LAW 平均 65% 成功率（任务分数 84.06）显著超越 [[π0.5]] 的 28.33%（55.50）和 [[Fast-WAM]] 的 25%（51.87）。

### Figure 8：状态预测消融累积图

![Figure 8 - State Prediction Ablation](https://arxiv.org/html/2610.23334v1/figures/state_confusion_ablation.png)

**说明**：逐步添加混合专家、失败数据、记忆后，子任务完成预测准确率的累积提升曲线，从 85.78% 升至 99.02%。

### Figure 9：不同 MoE 配置的动作损失曲线

**说明**（无图像 URL）：对比 Attn-MoE、FFN-MoE、无共享专家 MoTE、完整 MoTE 的训练动作损失曲线，说明 MoTE（混合注意力后 + 共享专家）的收敛优势。

### Figure 10：真实机器人平台

![Figure 10 - AgileX Cobot Magic Platform](https://arxiv.org/html/2610.23334v1/figures/agilex_cobot_magic.jpg)

**说明**：AgileX Cobot Magic 双臂平台，配备 2 个从动臂（执行）、2 个主动臂（示教采集）、1 个场景摄像头和 2 个腕部摄像头。

### Figure 11：RoboTwin 子任务分割示例（1）

![Figure 11 - RoboTwin Subtask Examples 1](https://arxiv.org/html/2610.23334v1/figures/subtask_robotwin_1.png)

**说明**：Handover Block、Put Bottles Dustbin、Stack Bowls Three、Blocks Ranking RGB 四个 RoboTwin 任务上的语义子任务分割可视化。

### Figure 12：RoboTwin 子任务分割示例（2）

![Figure 12 - RoboTwin Subtask Examples 2](https://arxiv.org/html/2610.23334v1/figures/subtask_robotwin_2.png)

**说明**：Blocks Ranking Size、Handover Mic、Place Dual Shoes、Shake Bottle 四个 RoboTwin 任务上的子任务分割。

### Figure 13：RoboDojo 子任务分割示例（1）

![Figure 13 - RoboDojo Subtask Examples 1](https://arxiv.org/html/2610.23334v1/figures/subtask_robodojo_1.png)

**说明**：Cover Blocks、Swap Blocks、Classify Objects、Stack Blocks 四个 RoboDojo 任务上的子任务分割。

### Figure 14：RoboDojo 子任务分割示例（2）

![Figure 14 - RoboDojo Subtask Examples 2](https://arxiv.org/html/2610.23334v1/figures/subtask_robodojo_2.png)

**说明**：Sweep Blocks、Organize Table、Play Stacking Toy、Fasten Screws 四个 RoboDojo 任务上的子任务分割。

### Table 1：RoboTwin 50 任务结果对比

| 方法 | Clean (%) | Randomized (%) |
|------|-----------|----------------|
| X-VLA | ~72.9 | — |
| [[π0.5]] | 82.74 | 76.76 |
| Motus | ~87.9 | — |
| [[Fast-WAM]] | 91.88 | 91.78 |
| LingBot-VA | 92.90 | 91.50 |
| [[Qwen-RobotManip]] | ~93.0 | — |
| [[LingBot-VLA-2.0]] | ~93.6 | — |
| **LAW + Fast-WAM** | **94.22** | **94.20** |
| **LAW + OpenWAM-α** | **95.60** | **95.52** |

**关键发现**：LAW 框架在不同 WAM 骨干上均带来一致提升（Fast-WAM +2.38%，OpenWAM-α +1.96%），说明 LAW 是与 WAM 骨干正交的增益。

### Table 2：RoboDojo 结果（无机器人领域预训练组最佳）

| 方法 | Score | Success (%) |
|------|-------|-------------|
| 基线方法 | — | — |
| **LAW** | **6.47** | **3.82** |

**关键发现**：在 RoboDojo 跨场景泛化 benchmark 中，LAW 在无机器人领域预训练的组别中排名第一。

### Table 3：真实机器人 3 个长时程任务结果

| 方法 | 平均成功率 | 任务分数 |
|------|-----------|---------|
| [[Fast-WAM]] | 25.00% | 51.87 |
| [[π0.5]] | 28.33% | 55.50 |
| **LAW** | **65.00%** | **84.06** |

**关键发现**：LAW 在最长 9 个子任务的 Organize 任务上表现出色，说明层次记忆和子任务切换对长时程任务至关重要。

### Table 4：VLM 规划器与状态专家消融（10 个 RoboTwin 任务）

| 配置 | 状态预测准确率 | 规划器精确匹配 |
|------|------------|------------|
| 文本输入 only | — | 70.10% |
| 基础混合专家 | 85.78% | — |
| + 失败数据 | — | — |
| + 记忆 | 99.02% | — |
| 完整 LAW VLM | 99.02% | 99.64% |

**关键发现**：混合专家架构 + 失败数据 + 层次记忆的累积贡献使状态预测从 85.78% 提升至 99.02%，规划器精确匹配从 70.10% 提升至 99.64%。

### Table 5：WAM 组件消融（50 个 RoboTwin 任务）

| 配置 | 平均成功率 |
|------|---------|
| Fast-WAM（基线） | 91.83% |
| + MoTE | 92.67% |
| + subtask prompt | 92.68% |
| + MoTE + subtask | 93.16% |
| + MoTE + subtask + 记忆 | 93.80% |
| + MoTE + subtask + 记忆 + 价值融合（完整 LAW） | **94.21%** |

**关键发现**：每个组件均带来独立增益，价值融合是单步最大收益（+0.41%），各模块协同效果显著。

### Table 6：不同 WAM 骨干上的 LAW 效果

| 配置 | 成功率 |
|------|-------|
| Fast-WAM | 91.83% |
| LAW + Fast-WAM | 94.21% |
| OpenWAM-α | 93.60% |
| **LAW + OpenWAM-α** | **95.56%** |

**关键发现**：LAW 对两种 WAM 骨干均带来约 2% 的一致提升，验证框架的通用性。

### Table 7：VLM 选型对比（去除价值融合）

| VLM | 成功率 |
|-----|-------|
| GPT-5.5 High | 88.7% |
| Embodied-R1.5 | 88.8% |
| **LAW VLM（Qwen3-VL-2B）** | **93.80%** |

**关键发现**：专门针对机器人任务微调的 2B 模型超越大型通用模型，说明任务专用适配的重要性。

### Table 8：MoE 放置位置消融

| 配置 | 成功率 |
|------|-------|
| 无 MoE（Fast-WAM） | 91.83% |
| Attn-MoE（注意力中加 MoE） | 91.85% |
| FFN-MoE（FFN 替换为 MoE） | 91.94% |
| MoTE 无共享专家 | 91.93% |
| **MoTE（混合注意力后）** | **92.67%** |

**关键发现**：MoTE 放置在混合注意力之后（而非替换注意力或 FFN）效果最好，共享专家是关键设计。

### Table 9：子任务粒度消融（10 个任务）

| 子任务类型 | 成功率 |
|---------|-------|
| 无子任务 | 92.4% |
| 低层次（关节动作级） | 89.1% |
| **语义级（本文）** | **95.8%** |

**关键发现**：语义级子任务（人类可理解的操作步骤）效果最佳，低层次子任务反而有害（切换过于频繁）。

### Table 10：价值监督粒度消融（5 个任务）

| 监督方式 | 成功率 |
|---------|-------|
| 无价值引导 | 88.20% |
| Episode 级奖励 | 87.90% |
| **子任务局部奖励（本文）** | **90.80%** |

**关键发现**：Episode 级奖励甚至不如无价值引导（信号过于稀疏），子任务局部奖励提供密集且准确的监督信号，提升显著。

### Table 11：记忆消融（RoboDojo）

| 记忆配置 | Score |
|---------|-------|
| 无记忆 | 4.06 |
| 仅近期帧 | — |
| 仅边界帧 | — |
| **完整层次记忆** | **7.46** |

**关键发现**：层次记忆在需要长程上下文的 RoboDojo 跨场景任务上带来 7.46 vs 4.06 的巨大增益（+84%），说明长程记忆对复杂任务至关重要。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin | 50 任务 | 标准化仿真，干净 + 随机化双评估 | 主要 benchmark |
| RoboDojo | 35 任务 | 跨场景泛化，无机器人领域预训练要求 | 泛化测试 |
| 真实机器人（AgileX Cobot Magic） | 3 任务（20 次/任务） | Puzzle/Domino/Organize，3-9 个子任务 | 真实验证 |

### 实现细节

- **VLM Backbone**: Qwen3-VL-2B-Instruct（冻结主干 + 3 路任务 LoRA）
- **WAM Backbone**: Fast-WAM / OpenWAM-α + MoTE 层（每层 2 私有 + 1 共享专家，K=1）
- **动作候选数 N**: 多次采样（具体值未披露）
- **融合候选数 K**: 8
- **动作窗口长度**: 32
- **价值头**: 101 个 bin，HL-Gauss 软标签
- **MoTE 残差系数**: α = 0.1
- **硬件**: 真实机器人用 AgileX Cobot Magic 双臂平台，场景摄像头 + 2 腕部摄像头

### 可视化结果

三个真实任务上，LAW 在长链任务（9 个子任务的 Organize）中表现出最大优势：精确的子任务切换时机和层次记忆使机器人能连贯完成"开抽屉-放积木-关抽屉"的多次重复操作，而 π0.5 和 Fast-WAM 在 4-5 个子任务后即开始出现定位偏差和遗忘。

---

## 批判性思考

### 优点

1. **闭环设计清晰**：语义子任务进度作为统一接口，VLM 和 WAM 职责明确，异步协调避免相互阻塞。
2. **与 WAM 骨干正交**：LAW 框架在 Fast-WAM 和 OpenWAM-α 上均有一致约 2% 提升，说明通用性好，未来可直接受益于更强的 WAM。
3. **子任务局部价值学习创新**：将价值回报在子任务边界截断，有效利用失败轨迹的已完成前缀，大幅提升数据利用率。
4. **消融实验充分**：对子任务粒度、MoE 放置、记忆配置、价值监督粒度等关键设计均有定量验证。
5. **真实机器人验证**：在 3 个长时程真实任务上的 65% vs 28.33% 对比，说服力强。

### 局限性

1. **体系复杂度高**：同时训练三个 LoRA 专家 + MoTE + 记忆管理，工程实现难度大，超参（α、K、T、子任务粒度）较多。
2. **子任务标注依赖人工**：语义子任务标注需要领域知识，自动化程度有限；论文中未给出标注成本的定量分析。
3. **跨机型迁移未验证**：论文主要在单一双臂平台上验证，跨机型（单臂、四足等）的泛化能力未知。
4. **RoboDojo 成功率低**：虽然在无预训练组排名最高，但 3.82% 的绝对成功率说明跨场景泛化仍是瓶颈。
5. **推理速度未讨论**：N 个候选 + 价值评分 + 融合的推理开销未给出实测延迟数据，部署实用性存疑。

### 潜在改进方向

1. **自适应子任务分解**：让规划器动态调整子任务粒度，无需人工预定义语义层级。
2. **多机型迁移**：将 LAW 框架迁移至单臂、四足等平台，验证通用性。
3. **在线价值更新**：在机器人部署时用少量真实反馈更新价值专家，提高对域漂移的鲁棒性。
4. **推理加速**：减少候选数 N 或用蒸馏缩小价值专家，降低在线推理延迟。

### 可复现性评估

- [x] 代码开源（https://github.com/hey-cjj/LAW）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（各关键超参在论文中列出）
- [x] 数据集可获取（RoboTwin/RoboDojo 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: WAM 骨干，LAW 默认集成并在其上提升 2.38%
- [[Qwen3-VL]]: VLM 基础模型（Qwen3-VL-2B-Instruct）
- [[HL-Gauss]]: 离散价值头的分布表示方法
- [[LoRA]]: VLM 三个专家的参数高效微调方法

### 对比

- [[π0.5]]: 长时程真实任务上 65% vs 28.33%
- [[LingBot-VLA-2.0]]: RoboTwin 上 94.22% vs 93.6%
- [[Qwen-RobotManip]]: RoboTwin 上超越 93.0%

### 方法相关

- [[MoTE]]: 核心架构创新，视频-动作联合专家层
- [[语义子任务分解]]: 核心设计理念
- [[双流层次记忆]]: 长时程观测管理方法
- [[子任务局部价值学习]]: 稀疏奖励下的有效价值监督
- [[Markley 四元数均值]]: 候选动作四元数融合方法
- [[Value Function]]: 价值专家的理论基础
- [[Mixture of Experts]]: MoTE 的通用框架

### 硬件/数据相关

- [[AgileX Cobot Magic]]: 真实实验机器人平台
- [[RoboTwin]]: 主要评估 benchmark（50 任务）
- [[RoboDojo]]: 跨场景泛化 benchmark（35 任务）

---

## 速查卡片

> [!summary] LAW: Unifying Language, Action, and World Modeling
> - **核心**: 以语义子任务进度统一 VLM 规划与 WAM 执行的闭环智能体框架
> - **方法**: 混合专家 VLM（规划/状态/价值三路 LoRA）+ MoTE-WAM + 双流层次记忆 + 子任务局部价值学习
> - **结果**: RoboTwin 94.21%（+2.38% vs Fast-WAM）；真实机器人 65%（vs π0.5 28.33%）
> - **代码**: https://github.com/hey-cjj/LAW

---

*笔记创建时间: 2026-10-20*

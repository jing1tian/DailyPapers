---
title: "DWM: Separating World Effects from Actions in Latent World Models"
method_name: "DWM"
authors: [Yi-Ge Zhang, Tianqi Du, Qi Zhang, Yisen Wang]
year: 2026
venue: arXiv
tags: [world-model, disentanglement, contrastive-learning, model-based-rl, planning, latent-space]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.18715v1
created: 2026-07-23
---

# 论文笔记：DWM — Separating World Effects from Actions in Latent World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 待确认 |
| 日期 | July 2026 |
| 项目主页 | — |
| 对比基线 | [[LeWM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.18715) / [HTML](https://arxiv.org/html/2607.18715v1) |

---

## 一句话总结

> 在 [[LeWM]] 基线之上，通过辅助 World Head + [[InfoNCE|对比损失]] + [[正交约束]]，将 [[潜在世界模型]] 的状态转换分解为动作驱动分量与[[动作不变表示|动作不变的世界效应]]，使 CEM 规划成功率平均提升 **13.1 pp**。

---

## 核心贡献

1. **问题诊断**：指出现有 [[Action-Conditioned World Model|动作条件世界模型]] 将 [[世界效应]] 与动作效应混在同一预测目标中，导致在具有持续性环境动力学（重力、漂移）的任务上规划质量下降。
2. **DWM 分解框架**：提出辅助 World Head $h_w$，通过 [[InfoNCE|世界对比损失]] $\mathcal{L}_{wc}$ 强制其输出对当前动作不变，并用[[正交约束]] $\mathcal{L}_{orth}$ 保证动作分量 $\hat{z}^a$ 与世界分量 $\hat{z}^w$ 方向独立。
3. **推理零开销**：World Head 仅在训练时使用，推理阶段丢弃，完全等价于原 [[LeWM]] 流水线。
4. **新 Benchmark**：构建 PushT-W、Reacher-W、TwoRoom-W 三个含持续性[[世界效应]]的任务变体，供后续研究使用。

---

## 问题背景

### 要解决的问题

[[Action-Conditioned World Model|动作条件潜在世界模型]] 的下一步 latent 预测目标 $\hat{z}_{t+1}$ 融合了两类动力学来源：
- **动作效应**：由 agent 选择的 $a_t$ 直接驱动的变化；
- **[[世界效应]]（World Effects）**：与动作无关、持续存在的环境动力学（重力滑动、惯性漂移、弹簧回复力等）。

单一预测目标无法区分二者，导致在有显著[[世界效应]]的任务中[[Model Predictive Control|MPC]] 规划失效。

### 现有方法的局限

- [[LeWM]] 等 [[JEPA]] 世界模型以单个 prediction head 预测下一个 latent，预测目标同时包含动作效应和世界效应，模型无法将二者解耦。
- 向训练数据加入零动作轨迹也不能解决此问题：单目标监督信号从结构上阻止了自然解耦。
- 基线 LeWM 在 PushT-W（重力驱动滑块）上规划成功率仅 32%，远低于无世界效应版本的 94%。

### 本文的动机

若能在训练阶段专门施加监督信号，让辅助 head 只捕捉"换动作也不变"的部分，则残差天然成为纯动作驱动分量，两者组合仍还原完整预测，不改变推理行为。

---

## 方法详解

### 模型架构

DWM 以 [[LeWM]] 为骨干，在训练阶段增加一条 **橙色辅助分支**：

- **输入**：观测 $o_t$（RGB 像素）+ 历史 $(o_{t-h}, \ldots, o_{t-1})$ + 当前动作 $a_t$
- **编码器 $\varphi$**：[[ViT|ViT-tiny]]（patch=14, depth=12, hidden=192），把 $o_t$ 映射到 $z_t$
- **预测器 $g$**：6 层 [[Transformer]]（16 heads, hidden=64, MLP=2048），注入 $a_t$，输出 rollout 状态 $r_t$
- **Pred Head $h_p$**（原始）：2 层 MLP（192→2048→192），$\hat{z} := h_p(r_t)$，仅用于推理
- **World Head $h_w$**（新增，训练专用）：相同 2 层 MLP 结构，$\hat{z}^w := h_w(r_t)$
- **动作分量**：残差计算 $\hat{z}^a := \hat{z} - \hat{z}^w$
- **输出**：推理时仅用 $\hat{z}$ 送入 [[Cross-Entropy Method|CEM]] 规划器，World Head 丢弃

### 核心模块

#### 模块 1: World Head（世界头）

**设计动机**：从同一 rollout 状态 $r_t$ 提取一个对动作不变的分量，捕捉纯环境动力学。

**具体实现**：
- 与 Pred Head 并联，共享 $r_t$ 输入；
- 由 [[InfoNCE|世界对比损失]] $\mathcal{L}_{wc}$ 驱动：对同一 $(z_t, \text{history})$ 但不同 $a_t$ 的 rollout，强制 $\hat{z}^w$ 相互一致；
- 训练结束后整体丢弃，推理阶段零开销。

#### 模块 2: 世界对比损失（$\mathcal{L}_{wc}$）

**设计动机**：用[[对比学习]]使 World Head 输出对当前动作不变，即"动作无关性"的监督信号。

**具体实现**：
- 对一个 batch，固定 $(z_t, \text{history})$ 而对不同 $a_t$ 施扰；
- 以 [[InfoNCE]] 形式（温度 $\tau=0.07$）拉近同状态不同动作下的 $\hat{z}^w$，推远不同状态下的 $\hat{z}^w$；
- 本质：$h_w$ 应该忽略 $a_t$ 的影响。

#### 模块 3: 正交约束（$\mathcal{L}_{orth}$）

**设计动机**：仅靠对比损失无法阻止 $\hat{z}^w$ 与 $\hat{z}^a$ 共享方向，需要额外约束确保解耦干净。

**具体实现**：
- 强制 $\hat{z}^w \perp \hat{z}^a$（即 $\hat{z}^w \cdot \hat{z}^a \approx 0$）；
- 实现为余弦相似度惩罚，权重 $\lambda_{orth}$；
- 与 $\mathcal{L}_{wc}$ 缺一不可（见消融实验）。

---

## 关键公式

### 公式 1: [[DWM|DWM 动作分量分解]]

$$
\hat{z}^w := h_w(r_t), \quad \hat{z}^a := \hat{z} - \hat{z}^w, \quad \hat{z} := h_p(r_t)
$$

**含义**：World Head 提取动作不变的世界分量，残差即为动作驱动分量，二者合并还原完整预测。

**符号说明**：
- $r_t$：预测器 $g$ 对 $(z_t, a_t)$ 的输出 rollout 状态
- $\hat{z}$：完整的下一步 latent 预测（Pred Head 输出）
- $\hat{z}^w$：世界效应分量（World Head 输出）
- $\hat{z}^a$：动作效应分量（残差）

### 公式 2: [[InfoNCE|世界对比损失 $\mathcal{L}_{wc}$]]

$$
\mathcal{L}_{wc} = -\log \frac{\exp(\hat{z}^w_i \cdot \hat{z}^{w+}_i / \tau)}{\sum_{j} \exp(\hat{z}^w_i \cdot \hat{z}^w_j / \tau)}
$$

**含义**：[[InfoNCE]] 损失，把同一状态下不同动作的 $\hat{z}^w$（正样本对）拉近，把不同状态的 $\hat{z}^w$ 推远，迫使 World Head 忽略动作信息。

**符号说明**：
- $\hat{z}^{w+}_i$：与第 $i$ 个样本同状态、不同动作的正样本 World Head 输出
- $\tau = 0.07$：InfoNCE 温度超参数
- 分母：batch 内所有负样本之和

### 公式 3: [[正交约束|正交约束损失 $\mathcal{L}_{orth}$]]

$$
\mathcal{L}_{orth} = \left\| \frac{\hat{z}^w}{\|\hat{z}^w\|} \cdot \frac{\hat{z}^a}{\|\hat{z}^a\|} \right\|^2
$$

**含义**：强制世界分量与动作分量方向正交，确保二者在 latent 空间中张成独立子空间。

**符号说明**：
- $\hat{z}^w / \|\hat{z}^w\|$：归一化世界分量
- $\hat{z}^a / \|\hat{z}^a\|$：归一化动作分量
- 损失为余弦相似度的平方

### 公式 4: [[DWM|总训练目标]]

$$
\mathcal{L} = \mathcal{L}_{WM} + \lambda_{wc} \cdot \mathcal{L}_{wc} + \lambda_{orth} \cdot \mathcal{L}_{orth}
$$

**含义**：在原有 [[LeWM]] 世界模型损失基础上，叠加两项辅助解耦损失（最优超参：$\lambda_{wc}=0.3, \lambda_{orth}=0.5$）。

**符号说明**：
- $\mathcal{L}_{WM}$：原 LeWM 的 latent MSE 预测损失
- $\lambda_{wc} = 0.3$：世界对比损失权重（最优）
- $\lambda_{orth} = 0.5$：正交约束权重（最优）

---

## 关键图表

### Figure 1: PushT-W 中世界效应与动作效应的纠缠

![Figure 1](https://arxiv.org/html/2607.18715v1/Figs/fig1.png)

**说明**：三行对比 PushT-W 中的规划行为。**上行**：零动作 rollout，灰色 T 形块在重力 $g$ 方向持续滑动（纯世界效应）。**中行**：[[LeWM]] 基线将滑动与推动融合在单一预测目标中，预测轨迹偏离目标（绿色 T）。**下行**：DWM 成功分离两类效应，灰色 T 既跟随重力滑动又被 pusher 推向目标。

### Figure 2: DWM 架构图

![Figure 2](https://arxiv.org/html/2607.18715v1/Figs/framework.png)

**说明**：上方虚线框为不变的 [[LeWM]] 基线：$o_t \xrightarrow{\varphi} z_t \xrightarrow{g+a_t} r_t \xrightarrow{h_p} \hat{z}$，下一帧 $o_{t+1}$ 编码为 $z_{t+1}$ 提供监督 $\mathcal{L}_{WM}$。**橙色分支**（仅训练）：同一 $r_t$ 送入 World Head $h_w$ 得 $\hat{z}^w$，残差得 $\hat{z}^a$，两项辅助损失施加解耦约束。推理时橙色分支丢弃，退化为标准 LeWM。

### Figure 3: 规划成功率对比（CEM）

![Figure 3](https://arxiv.org/html/2607.18715v1/x1.png)

**说明**：左图（W-variant 任务）DWM 在三个世界效应任务上全面超越 LeWM 基线；右图（Flat 对照）两者性能相当，确认增益不以牺牲原始任务为代价。平均改善 **+13.1 pp**。

### Figure 4: OOD 重力泛化（PushT-W）

![Figure 4](https://arxiv.org/html/2607.18715v1/x2.png)

**说明**：在训练重力 $(45, 25)$ 下学习，测试在 OOD 重力 $(90, 0)$ 与 $(0, 45)$。DWM 动作分量余弦相似度 0.9991（高度稳定），OOD 合并成功率 44% vs LeWM 30%（+14 pp，$p \approx 0.040$），体现分解表示的鲁棒性。

---

### Table 1: 表示诊断——世界分量方差比

| 任务 | World/Pred 方差比 | World/Pred Spread 比 |
|------|-------------------|----------------------|
| PushT-W | 0.0019 | 0.0424 |
| TwoRoom-W | 0.0045 | 0.0783 |
| Reacher-W | 0.0107 | 0.0909 |

**说明**：$h_w$ 输出的方差比 $h_p$ 低 2~3 个数量级，验证 World Head 确实学到了动作不变的低方差特征，而非与 Pred Head 输出相同的内容。

### Table 2: 多步预测质量（MSE & 关节位置误差）

| 任务 | 指标 | LeWM | DWM | 改善 |
|------|------|------|-----|------|
| PushT-W | Offline MSE | 0.372 | 0.038 | **-89.8%** |
| PushT-W | Rollout@20 MSE | 1.034 | 0.344 | **-66.7%** |
| TwoRoom-W | Offline MSE | 0.317 | 0.217 | -31.6% |
| TwoRoom-W | Rollout@20 MSE | 0.382 | 0.261 | -31.7% |
| Reacher-W | Offline qpos error | 0.046 | 0.033 | -29.8% |
| Reacher-W | Rollout@20 qpos error | 0.299 | 0.221 | -26.1% |

**说明**：DWM 在 offline（单步）和长程 rollout（20 步）上均大幅降低预测误差，PushT-W 改善最显著（因重力效应最强），说明分解有助于提升世界模型的多步预测精度。

### Table 3: 消融实验（PushT-W CEM 成功率）

| $\lambda_{wc}$ | $\lambda_{orth}$ | CEM 成功率 |
|----------------|-----------------|----------|
| 0.0 | 0.0 | 32.0%（基线） |
| 0.1 | 0.5 | 36.0% |
| 0.2 | 0.5 | 38.0% |
| 0.3 | 0.0 | 40.0% |
| 0.3 | 0.1 | 42.0% |
| **0.3** | **0.5** | **44.0%（最优）** |

**关键发现**：两项损失均不可缺——仅 $\mathcal{L}_{wc}$ 得 40%，仅 $\mathcal{L}_{orth}$ 无法驱动（需对比信号），完整组合达到最优 44%。

---

## 实验

### 数据集 / 环境

| 环境 | 世界效应 | 设定 | 用途 |
|------|---------|------|------|
| **PushT-W** | 重力驱动滑块 | 重力方向 $(45,25)$，阻尼 0.92 | 主评估 |
| **Reacher-W** | 垂直面重力 | $g = -9.81\sin(60°)$ 沿 z 轴 | 主评估 |
| **TwoRoom-W** | 常数漂移 | $\Delta p = a \cdot s + b$，$b=(4,2)$ px/step | 主评估 |
| PushT / Reacher / TwoRoom | 无 | Flat 对照 | 验证不退步 |
| **Ball-in-Cup** | 摆动动力学（状态相关） | 真实场景 | 泛化验证 |

### 实现细节

- **编码器**：[[ViT|ViT-tiny]]（hidden=192, depth=12, patch=14）
- **预测器**：6 层 [[Transformer]]（16 heads, hidden=64, MLP=2048）
- **两个 Head**：均为 2 层 MLP（192→2048→192）
- **上下文长度**：$h=3$，帧步长：5
- **优化器**：AdamW（lr=$5\times10^{-5}$，weight decay=$10^{-3}$）
- **Batch Size**：128，精度：bfloat16
- **训练轮数**：5~100（任务相关）
- **InfoNCE 温度**：$\tau=0.07$
- **规划器**：[[Cross-Entropy Method|CEM]]，3 个随机种子，每种子 50 个 start-goal 对

### 可视化结果

- PushT-W 上零动作 rollout 预测：DWM 能复现重力滑动轨迹，LeWM 预测直接偏离；
- OOD 重力下动作分量方向几乎不变（余弦相似度 0.9991），印证动作表示与环境动力学解耦成功。

---

## 批判性思考

### 优点
1. **轻量且无推理代价**：仅增加训练阶段辅助损失，推理路径完全不变，部署无额外开销。
2. **理论动机清晰**：明确区分"动作效应"与"[[世界效应]]"，有助于理解世界模型在哪类任务上会失效。
3. **泛化性好**：在重力滑动（PushT-W）、常数漂移（TwoRoom-W）和状态相关摆动（Ball-in-Cup）三种不同类型的世界效应上均有改善，说明方法不依赖特定物理形式。
4. **新 Benchmark 有价值**：W-variant 系列任务为评估世界模型在持续性动力学下的表现提供了标准化平台。

### 局限性
1. **仅改善世界效应场景**：Flat 对照任务上无增益，对无持续性环境动力学的任务贡献有限。
2. **残差分解的信息泄漏风险**：$\hat{z}^a = \hat{z} - \hat{z}^w$ 的残差设计未必严格保证动作信息不渗入 $\hat{z}^w$，仅靠正交约束可能不够充分。
3. **规模有限**：实验仅在低维 2D/3D 控制任务上验证，未覆盖复杂长任务或真实机械臂操作。
4. **超参敏感性**：两个损失权重 $\lambda_{wc}, \lambda_{orth}$ 需要网格搜索，虽然消融给出了较好的参考值。

### 潜在改进方向
1. 将分解思路推广到生成式世界模型（如 [[Video Diffusion Model|视频扩散]]），在像素空间也做世界/动作解耦。
2. 探索在 $\hat{z}^w$ 上额外施加物理一致性约束（如能量守恒），使其更好捕捉有物理意义的世界效应。
3. 将 World Head 输出用于异常检测（类似 [[Violation-of-Expectation]] 框架）：当观测与预期的世界效应不符时报警。

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文内含架构与超参）
- [x] 数据集可获取（PushT、Reacher、TwoRoom 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[LeWM]]：直接基线，DWM 在其训练流水线上增加辅助分支
- [[JEPA]]：LeWM 所属的联合嵌入预测架构范式
- [[InfoNCE]]：世界对比损失的实现形式

### 对比
- [[LeWM]]：单头预测，无解耦能力，W-variant 任务性能较差
- [[DINO-WM]]：基于预训练 DINO 特征，未显式分离世界效应
- [[DreamerV3]]：像素重建型世界模型，未解决世界/动作解耦问题

### 方法相关
- [[世界效应]]：本文核心概念——动作不变的环境动力学
- [[动作不变表示]]：World Head 学到的表示类型
- [[正交约束]]：解耦的关键数学工具
- [[对比学习]]：世界对比损失的方法论基础
- [[Cross-Entropy Method]]：规划求解器
- [[Model Predictive Control]]：闭环执行框架
- [[ViT]]：编码器骨干
- [[Transformer]]：预测器骨干

### 硬件/数据相关
- [[Push-T]]：核心评估环境（PushT-W 变体）

---

## 速查卡片

> [!summary] DWM
> - **核心**: 在 [[LeWM]] 训练中增加辅助 World Head，分离状态转换中的世界效应与动作效应
> - **方法**: [[InfoNCE|世界对比损失]] + [[正交约束]]，训练专用分支，推理无开销
> - **结果**: W-variant 任务 CEM 规划成功率平均 +13.1 pp；多步预测 MSE 最高降低 89.8%
> - **代码**: 未开源（截至 2026-07-23）

---

*笔记创建时间: 2026-07-23*

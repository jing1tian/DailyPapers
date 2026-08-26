---
title: "WAM-OPD: On-Policy Distillation for World Action Models"
method_name: "WAM-OPD"
authors: [Liuhaichen Yang, Zhuang Jiang, Chenchao Sheng, Zezhi Tang]
year: 2026
venue: arXiv
tags: [world-action-model, on-policy-distillation, knowledge-distillation, flow-matching, LoRA, robot-manipulation, inference-efficiency, video-prediction]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.22364
created: 2026-08-26
---

# 论文笔记：WAM-OPD: On-Policy Distillation for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开 |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.22364) / Code: N/A |

---

## 一句话总结

> WAM-OPD 通过让学生模型在环境中自主行动来确定历史分布，由冻结的教师模型对学生访问的状态提供监督，从而解决视频优先世界动作模型加速时的条件分布失配问题。

---

## 核心贡献

1. **识别视频优先WAM蒸馏的条件失配问题**: 当加速学生模型部署时消费自身生成的视频计划（而非教师视频计划），产生环境级和模型级双重分布偏移，现有离线蒸馏方法无法解决。
2. **部署一致的联合视频-动作损失框架**: 学生动作分支在推理时完全复用自身生成的视频计划（通过 stop-gradient），训练与部署行为精确一致；教师提供相干的视频与动作标签。
3. **参数高效的 JointLoRA 适配**: 在所有30个共享 Transformer block 上插入 rank-8 [[LoRA]] 适配器，仅用120步、批量大小4即可在两个仿真任务上取得显著提升。

---

## 问题背景

### 要解决的问题

[[世界动作模型（WAM）]] 将视频预测与机器人动作生成联合建模，形成"视频优先"策略。为了加速推理，通常对原始（教师）模型进行蒸馏，得到更快的学生模型（如 [[Flash-WAM]]）。然而，部署时学生模型消费**自身生成的视频计划**，而非教师的视频计划，这导致：

- **条件失配（Conditioning Mismatch）**: 训练时见过的是教师视频计划，部署时用的是学生视频计划，两者存在系统性差异。
- **历史分布偏移（History Distribution Shift）**: 学生动作改变了环境状态轨迹，所遇到的状态与教师训练时使用的离线数据分布不同。

### 现有方法的局限

标准离线知识蒸馏（offline distillation）将固定数据集中的教师输出作为监督信号，无法覆盖学生在实际部署时访问的状态空间。当学生模型质量低于教师时，其行动轨迹会系统地偏离训练分布，导致任务成功率崩溃（如 Flash-WAM 在 Handover Mic 任务上成功率降至 0%）。

### 本文的动机

借鉴强化学习中[[On-Policy Distillation]]的思路：让学生自身在环境中采集数据，由此确定训练时的历史分布，使训练分布与部署分布相匹配。同时利用冻结教师模型对这些学生访问的状态提供高质量的视频和动作标签，避免纯强化学习的奖励稀疏问题。

---

## 方法详解

### 模型架构

WAM-OPD 在已发布的学生模型（Flash-WAM）上进行后训练（post-training），采用 **参数高效微调 + 在线数据收集** 架构：

- **输入**: 历史观测 $h_t$（包含视频帧序列、机器人状态、语言指令）
- **Backbone**: 学生WAM的30个共享 [[Transformer]] block（全部冻结原始权重）
- **适配模块**: Rank-8 [[LoRA|JointLoRA]] 适配器，插入全部30个共享块
- **监督来源**: 冻结教师WAM对学生历史 $h_S$ 的标注输出 $(z_T, a_T)$
- **输出**: 视频计划 $z_S$ + 动作块 $a_S$（条件化于 $\text{sg}(z_S)$）

### 核心模块

#### 模块1：部署一致的WAM蒸馏（Deployment-Consistent WAM Distillation）

**设计动机**: 消除训练与部署之间的条件分布不一致，使学生动作分支在训练时与部署时行为完全相同。

**具体实现**:
- 学生先生成视频计划 $z_S = f^{\text{vid}}_\theta(h_S)$
- 动作分支条件化于 **stop-gradient** 的学生视频计划：$a_S = f^{\text{act}}_\theta(h_S, \text{sg}(z_S))$
- 这精确匹配部署时动作分支消费学生视频的行为

#### 模块2：在线数据收集协议（On-Policy Data Collection）

**设计动机**: 利用[[On-Policy Distillation]]思路，让学生在环境中实际行动，决定训练时的历史分布 $d^{\pi_S}$。

**具体实现**:
- 从每个任务收集 8 条学生轨迹（共160个标注上下文）
- 对每个学生历史 $h_S$，冻结教师模型前向推理得到标签 $(z_T, a_T)$
- 固定轨迹包（trajectory package）在初始收集器上是在策略的（on-policy），此后固定不变

#### 模块3：联合目标函数（Joint Objective）

**设计动机**: 同时对视频预测和动作生成两个分支进行监督，辅以动作流匹配正则化保证动作分布的多样性。

**具体实现**:
- 视频端点对齐（Pseudo-Huber 惩罚）
- 动作端点对齐（Pseudo-Huber 惩罚）
- 动作[[Flow-Matching|流匹配]]辅助正则化项

#### 模块4：参数高效更新（Parameter-Efficient Update）

**设计动机**: 在极少数据（160个上下文，120步优化）条件下防止过拟合。

**具体实现**:
- 在学生WAM的所有30个共享 Transformer block 上插入 Rank-8 [[LoRA|JointLoRA]] 适配器
- 冻结所有原始（已发布）权重，仅训练 LoRA 参数

---

## 关键公式

### 公式1：WAM 策略分解

[[世界动作模型（WAM）]] 将策略分解为视频预测与条件动作生成：

$$
p_\theta(z_t, a_t \mid h_t) = p_\theta(z_t \mid h_t) \cdot p_\theta(a_t \mid h_t, z_t)
$$

**含义**: WAM 先预测未来视频计划 $z_t$，再条件于视频计划生成动作 $a_t$。

**符号说明**:
- $h_t$: 时刻 $t$ 的历史（视频帧 + 机器人状态 + 语言指令）
- $z_t$: 预测的未来视频计划（latent 或像素空间）
- $a_t$: 机器人动作块

### 公式2：[[Flow-Matching|条件流匹配]] 轨迹

$$
\frac{d}{d\sigma} \phi_\sigma(x) = v_\sigma(\phi_\sigma(x))
$$

$$
x_\sigma = (1 - \sigma) x_0 + \sigma \varepsilon, \quad u_\sigma(x_\sigma \mid x_0) = \varepsilon - x_0
$$

**含义**: 流匹配通过速度场 $v_\sigma$ 在噪声 $\varepsilon$ 与数据 $x_0$ 之间构建条件最优传输轨迹。

**符号说明**:
- $\sigma \in [0, 1]$: 流时间步，$\sigma=0$ 对应数据，$\sigma=1$ 对应噪声
- $x_0$: 目标数据（视频计划或动作）
- $\varepsilon \sim \mathcal{N}(0, I)$: 噪声
- $u_\sigma$: 条件向量场（条件流匹配目标）

### 公式3：[[Flow-Matching|条件流匹配]]损失

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{\sigma, x_0, \varepsilon} \left[ \| v_\theta(x_\sigma, \sigma, c) - (\varepsilon - x_0) \|^2_2 \right]
$$

**含义**: 训练速度场网络 $v_\theta$ 预测从数据到噪声的方向，条件于上下文 $c$。

### 公式4：[[On-Policy Distillation|在策略蒸馏]]目标

$$
\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{h \sim d^{\pi_S}} \left[ D(q_\theta(\cdot \mid h), \, q_T(\cdot \mid h)) \right]
$$

**含义**: 在学生策略 $\pi_S$ 诱导的历史分布 $d^{\pi_S}$ 上最小化学生分布与教师分布的散度，使训练分布与部署分布一致。

**符号说明**:
- $d^{\pi_S}$: 学生策略在环境中滚动产生的历史分布
- $q_\theta$: 学生模型的输出分布
- $q_T$: 冻结教师模型的输出分布
- $D(\cdot, \cdot)$: 分布差异度量（本文用端点回归近似）

### 公式5：部署一致的视频-动作生成

$$
z_S = f^{\text{vid}}_\theta(h_S), \quad a_S = f^{\text{act}}_\theta(h_S, \, \text{sg}(z_S))
$$

**含义**: 学生先生成视频计划 $z_S$，动作分支通过 stop-gradient 操作符 $\text{sg}(\cdot)$ 条件化于 $z_S$，与部署时完全一致。

**符号说明**:
- $f^{\text{vid}}_\theta$: 视频预测分支
- $f^{\text{act}}_\theta$: 动作生成分支
- $\text{sg}(\cdot)$: stop-gradient 操作符，阻止视频梯度流向动作损失

### 公式6：联合训练目标（核心损失函数）

$$
\mathcal{L}(\theta) = \lambda_z \, \rho_\delta(z_S - z_T) + \lambda_a \, \rho_\delta(a_S - a_T) + \lambda_{\text{FM}} \, \left\| v^{\text{act}}_\theta\!\left(x^\alpha_{\sigma=1}, 1, c\right) - (\varepsilon_a - a_T) \right\|^2_2
$$

其中 Pseudo-Huber 惩罚项为：

$$
\rho_\delta(e) = \delta^2 \left( \sqrt{1 + (e/\delta)^2} - 1 \right)
$$

**含义**: 联合优化视频端点对齐、动作端点对齐与动作流匹配辅助正则化三个目标。

**符号说明**:
- $\lambda_z, \lambda_a, \lambda_{\text{FM}}$: 权重系数（本文设为 $1, 1, 0.2$）
- $\rho_\delta(\cdot)$: Pseudo-Huber 惩罚，对大误差鲁棒
- $z_T, a_T$: 冻结教师模型对学生历史的标注输出
- $v^{\text{act}}_\theta$: 动作流匹配速度网络
- $x^\alpha_{\sigma=1}$: 噪声化的动作端点（$\sigma=1$）
- $\varepsilon_a$: 动作域噪声样本

---

## 关键图表

### Figure 1：方法概览

![[WAM-OPD_fig1_overview.png]]

**说明**: WAM-OPD 整体流程。学生模型独立在环境中行动，决定历史分布 $d^{\pi_S}$；冻结教师模型对每个学生历史标注相干的视频与动作目标 $(z_T, a_T)$；学生动作分支通过 stop-gradient 消费自身视频计划 $\text{sg}(z_S)$，与部署完全一致。

### Table 1：精确配对的保留集评估结果

| 任务 | Released (Flash-WAM) | WAM-OPD | 提升 |
|------|---------------------|---------|------|
| Handover Mic（话筒交接） | 0.0% | 58.3% | **+58.3 pp** |
| Put Object Cabinet（物体放入柜子） | 16.7% | 33.3% | **+16.7 pp** |

**说明**: 在两个 RoboTwin 2.0 仿真任务上，WAM-OPD 相较于已发布的 Flash-WAM 学生模型均取得显著提升。Handover Mic 任务提升尤为突出，从完全失败（0%）提升至58.3%。评估使用固定6个保留场景种子和固定噪声库，确保公平对比。

### Table 2（计划中）：广泛评估矩阵

| 评估维度 | 状态 |
|---------|------|
| 多任务平均成功率 | TBD |
| 真实机器人迁移 | TBD |
| 不同收集预算的数据消融 | TBD |
| 视频损失权重 $\lambda_z$ 消融 | TBD |
| 动作损失权重 $\lambda_a$ 消融 | TBD |
| LoRA rank 消融 | TBD |

**说明**: 当前实验为概念验证性质，广泛评估（多任务、真实机器人、超参消融）列为后续工作。

---

## 实验

### 数据集与仿真环境

| 数据集/环境 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| [[RoboTwin 2.0]] | 8 条学生轨迹（160个标注上下文）| 双手机器人操作仿真，干净无扰动 | 在策略数据收集 |
| Handover Mic 任务 | 6个保留场景种子 | 话筒在两手间传递 | 评估 |
| Put Object Cabinet 任务 | 6个保留场景种子 | 物体放入柜中抽屉 | 评估 |

### 实现细节

- **基础模型**: Flash-WAM（已发布的加速学生模型）
- **适配器**: Rank-8 [[LoRA|JointLoRA]]，覆盖共享 Transformer blocks 0–29（全部30个）
- **优化器**: AdamW，学习率 $2 \times 10^{-5}$
- **Batch Size**: 4
- **训练步数**: 120 步（约3个 epoch）
- **损失权重**: $(\lambda_z, \lambda_a, \lambda_{\text{FM}}) = (1, 1, 0.2)$
- **校准轨迹**: 4 条（用于检查点选择）
- **评估**: RoboTwin 原生 "latched eval_success" 指标

### 数据收集协议

1. 运行已发布的 Flash-WAM 学生模型在环境中采集 **8 条轨迹**（每任务）
2. 对每条轨迹的每个时间步历史 $h_S$，用冻结教师 WAM 前向推理标注 $(z_T, a_T)$，共 **160 个上下文**
3. 另收集 **4 条校准轨迹** 用于检查点选择
4. 使用**两个固定噪声库**确保评估确定性

### 关键发现

- Handover Mic：Flash-WAM 完全失败（0%），WAM-OPD 恢复至 **58.3%**，验证了方法核心机制的有效性
- Put Object Cabinet：从16.7%提升至 **33.3%**，提升绝对值16.7个百分点
- 仅需 **120步** 优化（批量4，三个epoch），极低计算成本
- 作者明确指出：当前结果是"任务特定"的，尚不能代表广泛泛化能力

---

## 批判性思考

### 优点

1. **问题识别清晰**: 精确识别了视频优先WAM部署时的"条件失配"双重分布偏移，比一般蒸馏文献更细致。
2. **方法优雅简洁**: stop-gradient 技巧直接消除训练-部署不一致，无需复杂架构改动。
3. **参数效率高**: rank-8 LoRA + 120步训练，数据需求极低（仅8条轨迹），工程实用性强。
4. **透明诚实**: 作者主动指出局限性，承认结果为任务特定、尚无多任务和真实机器人验证。

### 局限性

1. **实验规模有限**: 仅在两个清洁仿真任务上验证，保留集仅6个场景种子，统计可信度不足。
2. **在策略性有限**: 固定的轨迹包仅在初始收集器上是在策略的，后续迭代仍为离线蒸馏。
3. **JointLoRA梯度干扰未测量**: 视频损失和动作损失共享同一套 LoRA 参数，两者的梯度交互（可能有益或有害）未被分析。
4. **无真实机器人验证**: 所有结果均来自仿真，Sim2Real迁移能力未知。
5. **超参敏感性未分析**: $(\lambda_z, \lambda_a, \lambda_{\text{FM}})$ 的选择依赖先验经验，消融实验列为后续工作。

### 潜在改进方向

1. **迭代在策略收集**: 多轮交替执行"学生收集 → 教师标注 → LoRA训练"，逐步扩展在策略分布覆盖。
2. **分离视频与动作LoRA**: 使用独立的 LoRA 适配器分别更新视频分支和动作分支，隔离梯度干扰。
3. **广泛多任务验证**: 扩展到 RoboTwin 2.0 的更多任务以及真实机器人平台。
4. **数据效率分析**: 系统研究轨迹数量对性能的影响，确定最低有效数据量。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型（学生 Flash-WAM 模型可能需要单独获取）
- [x] 训练细节完整（优化器、学习率、步数、批量大小均有明确说明）
- [x] 数据集可获取（RoboTwin 2.0 公开仿真环境）

---

## 关联笔记

### 基于

- [[Flash-WAM]]: WAM-OPD 的优化目标，即已发布的加速学生模型（被WAM-OPD后训练提升）
- [[Flow-Matching]]: WAM 中视频和动作生成的核心建模框架
- [[On-Policy Distillation]]: 方法的核心思想来源，让学生在环境中行动决定训练分布

### 对比

- [[Flash-WAM]]: 直接的基线，WAM-OPD 在其基础上进行后训练以恢复任务能力

### 方法相关

- [[世界动作模型（WAM）]]: 本文方法的应用目标模型类型
- [[LoRA]]: 参数高效适配的核心技术（Rank-8 JointLoRA）
- [[Flow-Matching]]: 视频和动作生成以及辅助损失的建模基础

### 硬件/数据相关

- [[RoboTwin 2.0]]: 实验所用仿真环境，提供 Handover Mic 和 Put Object Cabinet 两个任务

---

## 速查卡片

> [!summary] WAM-OPD
> - **核心**: 视频优先WAM加速模型存在"条件失配"，学生在部署时消费自身视频计划，标准离线蒸馏无法解决
> - **方法**: 学生在线采集轨迹（在策略），冻结教师标注，stop-gradient保证训练-部署一致，rank-8 JointLoRA参数高效更新
> - **结果**: Handover Mic 0%→58.3%，Put Object Cabinet 16.7%→33.3%（仅2个仿真任务，120步训练）
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-26*

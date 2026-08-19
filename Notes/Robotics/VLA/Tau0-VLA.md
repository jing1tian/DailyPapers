---
title: "τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation"
method_name: "Tau0-VLA"
authors: [Xiaowei Cai, Yunuo Cai, Bingao Chen, Jingxiao Chen, Zhi Chen, Siyuan Feng, Tengyu Hou, Jingshun Huang, Han Jiang, Runkun Ju, Dong Li, Mingxiang Li, Shaowei Li, Xinchen Li, Yifan Li, Yi Liu, Zhongyuan Liu, Jianlan Luo, Junwen Miao, Ruiqi Ni, Buqing Nie, Mingjie Pan, Xinlin Ren, Jianheng Song, Jiaxu Wang, Peiqi Wang, Sen Wang, Xiaoyan Wang, Dafeng Wei, Dongming Wu, Pengwei Xie, Pu Yang, Hangjian Ye, Xiangyu Yue, Jinyu Zhang, Qinglin Zhang, Xueyong Zhao, Pengfei Zhou, Yue Zhou]
year: 2026
venue: arXiv cs.RO
tags: [vla, hierarchical-policy, world-model, test-time-computation, flow-matching, robot-manipulation, beam-search]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.16885v1
created: 2026-08-19
---

# 论文笔记：τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Agibot、Shanghai Innovation Institute、CUHK |
| 日期 | August 2026 |
| 项目主页 | [tau0-vla.github.io](https://tau0-vla.github.io/) |
| 对比基线 | [[GR00T N1.7]]、[[pi0.5]]、[[LingBot-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.16885) / Code via project page |

---

## 一句话总结

> τ₀-VLA 用「世界模型引导的测试时 beam search」生成子任务，再由低层 VLA 执行，使机器人首次在 25 步长达 88 分钟的家务任务中取得 45% 完整成功率。

---

## 核心贡献

1. **层次化 VLA 框架**: 将长时序操纵分解为高层子任务规划（[[Hierarchical Policy|层次策略]]）与低层 [[Vision-Language-Action Model|VLA]] 动作执行，两者通过 [[Execution Memory|执行记忆]] $\mathcal{M}_t$ 解耦。
2. **世界模型引导的测试时计算**: 高层策略在推理时借助 [[World Model|世界模型]] 预测候选子任务的视觉结果，用价值模型打分，通过 [[Beam Search|束搜索]] 选出最优子任务——无需额外训练即可通过扩大计算预算提升性能。
3. **多体态统一低层策略**: 低层策略采用 [[Mixture-of-Transformers|混合 Transformer 动作专家]]（MoT）+ [[Masked Flow Matching|掩码流匹配]]，在 40 维统一动作空间中同时支持固定基座、移动和双臂三类体态，训练于 40,115 小时真实数据。

---

## 问题背景

### 要解决的问题

长时序机器人操纵（如"炒西红柿鸡蛋"共 22 步、"整理房间"共 25 步）要求机器人**既能可靠执行单个技能，又能跨越长时间连贯排序**。现有 end-to-end VLA 难以在数十分钟的任务中维持一致性。

### 现有方法的局限

- **端到端 VLA**（RT-2、π₀）：长时序推理能力弱，遇到早期错误无法恢复。
- **语言层次策略**（RT-H、Hi Robot）：子任务生成依赖于单次前向推理，无法在推理时利用更多计算资源改善决策质量。
- **基于搜索的规划**（RoboMonkey、VLA-Reasoner）：往往依赖动作级别的采样验证，在真实机器人上代价高昂。

### 本文的动机

将子任务生成建模为**计算可扩展的推理问题**：固定低层执行器不变，仅在高层 token 级别 beam search，代价远低于动作级别采样。世界模型提供免费的"脑内预演"，价值模型提供评分，两者合力引导 beam search。

---

## 方法详解

### 模型架构

τ₀-VLA 采用**两级层次化**架构：

- **输入（高层）**: 任务指令 $\ell$ + 历史执行记忆 $\mathcal{M}_{t-1}$ + 上一子任务 $z_{t-1}^*$ + 当前观测 $o_t$
- **高层 Backbone**: Qwen3.5-9B（提案模型、世界模型、价值模型、反思模型）
- **低层 Backbone**: Qwen3.5-2B 视觉语言主干 + [[Mixture-of-Transformers|MoT]] 动作专家
- **输入（低层）**: 观测 $o_t$、本体状态 $s_t$、子任务 $z_t^*$、体态标识 $\eta$
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H-1}$（40 维统一空间）
- **世界模型初始化**: Step1X-Edit

### 核心模块

#### 模块 1：提案模型（Proposal Model）

**设计动机**: 用 [[Large Language Model|大语言模型]] 的文本推理能力生成候选子任务，同时更新执行记忆。

**具体实现**:
- 以高层决策上下文 $h_t$ 为输入，输出 **直接提案** $z_t^{\text{dir}}$ 和更新后的记忆 $\mathcal{M}_t$
- 训练时使用已有的任务、阶段、可执行子任务标注自动监督
- 训练数据包含**扰动记忆**（记忆超前/滞后/错误），使模型学会自我纠错

#### 模块 2：世界模型 + 价值模型（World & Value Model）

**设计动机**: 在不执行动作的情况下预测子任务后的场景状态，并评估该状态是否符合任务目标（[[Test-Time Computation|测试时计算]]的关键）。

**具体实现**:
- **世界模型** $\mathcal{W}$: 接收当前头部相机图像 $\tilde{o}$ 和候选子任务 $z$，预测执行该子任务后的终态图像 $\hat{o}$
- **价值模型** $V$: 接收任务指令 $\ell$、候选子任务 $z$、预测终态图像 $\hat{o}$，输出得分 $v$
- 世界模型由 Step1X-Edit 初始化（图像编辑模型），使视觉预测更真实

#### 模块 3：测试时 Beam Search（TTC）

**设计动机**: 通过扩大推理计算预算系统性提升子任务规划质量，无需重训练。

**具体实现**:
- 参数：分支因子 $N=3$（每个 beam 采样 $N$ 个候选）、束宽 $B=2$（保留前 $B$ 个 beam）、搜索深度 $D$
- 自适应路由：提案模型计算 token 置信度，路由器 $g_t \in \{0, 1\}$ 决定是否启用 TTC
- 仅在必要时触发 TTC，节省推理资源

#### 模块 4：反思模型（Reflective Model）

**设计动机**: 在 beam search 结果的基础上生成最终子任务，而非直接从候选集中选取，允许更灵活的后处理推理。

**具体实现**:
- 以观测对齐的上下文 $\bar{h}_t$ 和 beam 搜索摘要 $\mathcal{C}_t$ 为条件
- 输出的 $z_t^*$ **可以与任何保留的候选重合，但不限于候选集合**

#### 模块 5：低层 MoT 动作专家 + 掩码流匹配

**设计动机**: 统一支持异构体态，避免为每个机器人平台分别训练。

**具体实现**:
- 40 维统一动作空间（末端执行器、关节、夹爪、腰部、底盘），用**每样本掩码**选择有效通道
- [[Mixture-of-Transformers]] 动作专家：多个专家 Transformer，根据体态/任务类型路由
- [[Conditional Flow Matching|条件流匹配]] 训练，掩码同时作用于流路径和训练目标

---

## 关键公式

### 公式 1: [[Vision-Language-Action Model|低层 VLA 策略]] 映射

$$
a_{t:t+H-1} = \pi_\theta(o_t, s_t, c_t, \eta)
$$

**含义**: 低层策略 $\pi_\theta$ 将观测、本体状态、语言条件和体态标识映射到 $H$ 步动作块。

**符号说明**:
- $o_t$: 当前观测（多视角 RGB 图像）
- $s_t$: 本体感知状态（关节角度、末端位姿等）
- $c_t$: 语言/子任务条件
- $\eta$: 体态标识符
- $a_{t:t+H-1}$: H 步 [[Action Chunking|动作块]]

### 公式 2: [[Hierarchical Policy|高层决策上下文]] 定义

$$
h_t = (\ell,\, \mathcal{M}_{t-1},\, z_{t-1}^*,\, o_t)
$$

**含义**: 高层策略的输入上下文由任务指令、历史执行记忆、上一子任务和当前观测四元组构成。

**符号说明**:
- $\ell$: 自然语言任务指令
- $\mathcal{M}_{t-1}$: 前一步 [[Execution Memory|执行记忆]]
- $z_{t-1}^*$: 前一步最终子任务决策
- $o_t$: 当前观测

### 公式 3: 层次化策略执行

$$
(\mathcal{M}_t,\, z_t^*) = \mu(h_t)
$$

$$
a_{t:t+H-1} = \pi_\theta(o_t,\, s_t,\, z_t^*,\, \eta)
$$

**含义**: 高层策略 $\mu$ 输出更新后的执行记忆和当前子任务；低层策略以子任务为条件生成动作。

### 公式 4: 提案模型（Eq. 5）

$$
(z_t^{\text{dir}},\, \mathcal{M}_t) = P(h_t)
$$

**含义**: 提案模型 $P$ 由高层上下文直接生成候选子任务（Fast Path）并更新执行记忆。

### 公式 5: 世界模型与价值模型接口（Eq. 6）

$$
\hat{o} = \mathcal{W}(\tilde{o},\, z), \quad v = V(\ell,\, z,\, \hat{o})
$$

**含义**: 世界模型 $\mathcal{W}$ 由当前图像和候选子任务预测终态图像；价值模型 $V$ 对该结果打分。

**符号说明**:
- $\tilde{o}$: 当前头部相机图像
- $z$: 候选子任务
- $\hat{o}$: 预测的终态图像
- $v$: 价值分数

### 公式 6: 束搜索（Eq. 7）

$$
\mathcal{C}_t = \operatorname{Search}(h_t,\, P,\, \mathcal{W},\, V,\, N,\, B,\, D)
$$

**含义**: Beam Search 以高层上下文和四个模型为输入，输出保留分支摘要集合 $\mathcal{C}_t$。

**符号说明**:
- $N$: 每个 beam 的分支因子（$N=3$）
- $B$: 保留 beam 宽度（$B=2$）
- $D$: 搜索深度

### 公式 7: Beam Search 内部 —— 候选采样（Eq. 8）

$$
(z_{t,d}^{b,i},\, \mathcal{M}_{t,d}^{b,i}) \sim P(\cdot \mid h(b)),\quad i = 1, \ldots, N
$$

**含义**: 对保留的每个 beam $b$，提案模型采样 $N$ 个候选子任务。

### 公式 8: Beam Search 内部 —— 世界预测与累积打分（Eq. 9–10）

$$
\hat{o}(b \oplus z_{t,d}^{b,i}) = \mathcal{W}(\tilde{o}(b),\, z_{t,d}^{b,i})
$$

$$
v(b \oplus z_{t,d}^{b,i}) = V(\ell,\, z_{t,d}^{b,i},\, \hat{o}(b \oplus z_{t,d}^{b,i}))
$$

$$
S(b \oplus z_{t,d}^{b,i}) = S(b) + v(b \oplus z_{t,d}^{b,i})
$$

**含义**: 对每个扩展分支 $b \oplus z^{b,i}$，用世界模型预测终态图，价值模型打分，累积加入父分支得分。

### 公式 9: Beam 保留（Eq. 11）

$$
A_{t,d} = \bigl(b \oplus z_{t,d}^{b,i}\bigr)_{b \in Q_{t,d-1},\; i=1,\ldots,N}
$$

$$
Q_{t,d} = \operatorname{Top}_B(A_{t,d},\, S)
$$

**含义**: 将所有扩展分支汇总，全局按累积得分选出前 $B$ 个保留。

### 公式 10: 最终 Beam 摘要（Eq. 12）

$$
\mathcal{C}_t = \bigl((\rho(b),\, \hat{o}(b),\, S(b))\bigr)_{b \in Q_{t,D}}
$$

**含义**: 深度 $D$ 处保留的每个 beam 以（子任务序列、预测终态图、累积分数）三元组形式汇总为反思模型的输入。

### 公式 11: 反思模型（Eq. 13）

$$
z_t^* = F(\bar{h}_t,\, \mathcal{C}_t)
$$

**含义**: 反思模型 $F$ 综合观测对齐上下文 $\bar{h}_t$ 和 beam 摘要 $\mathcal{C}_t$，生成最终子任务决策。

### 公式 12: 掩码流匹配（Eq. 14）— [[Masked Flow Matching|掩码流路径与速度目标]]

$$
a_{t+j}^{\tau} = \tau \mathbf{M} \varepsilon_{t+j} + (1 - \tau) \mathbf{M} a_{t+j}
$$

$$
u_{t+j} = \mathbf{M}(\varepsilon_{t+j} - a_{t+j})
$$

**含义**: 在流时间 $\tau \in [0,1]$ 上对有效动作通道插值；掩码矩阵 $\mathbf{M}$ 确保仅在当前体态的有效维度上监督。

**符号说明**:
- $\tau$: 流时间，$\tau \in [0, 1]$
- $\mathbf{M}$: 对角掩码矩阵，选择当前体态有效动作通道
- $\varepsilon_{t+j} \sim \mathcal{N}(0, I_{d_a})$: 高斯噪声
- $a_{t+j}$: 目标动作（第 $j$ 步）
- $u_{t+j}$: 速度场目标

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1 - Overview](https://arxiv.org/html/2608.16885v1/teaser_v8.png)

**说明**: τ₀-VLA 整体架构。高层策略通过 [[World Model|世界模型]] 引导的 [[Beam Search|束搜索]] 预演候选子任务并打分，选出最优子任务传递给低层 VLA 执行，支持三类异构机器人体态。

### Figure 2: Framework / 模型框架详解

![Figure 2 - Framework](https://arxiv.org/html/2608.16885v1/framework_0726.png)

**说明**: (a) 高层推理流程：提案模型更新执行记忆并生成候选，自适应路由决定是否触发 TTC，反思模型生成最终子任务。(b) 低层策略：Qwen3.5-2B 视觉语言主干 + [[Mixture-of-Transformers|MoT]] 动作专家，[[Masked Flow Matching|掩码流匹配]] 生成 40 维动作块。(c) 测试时计算路径：完整 beam search 流程。

### Figure 3: Evaluation Tasks / 评估任务展示

![Figure 3 - Tasks](https://arxiv.org/html/2608.16885v1/demo_new.png)

**说明**: 六类真实机器人评估任务（跨三个平台）：整理房间（25 步，~88 分钟）、备菜（14 步，~44 分钟）、炒西红柿鸡蛋（22 步，~10 分钟）、制作奶茶（13 步，~3 分钟）、收取衣物（5 步）、整理化妆台（8 步/3 组）。

### Figure 4: TTC Prediction Accuracy / 测试时计算预测精度对比

![Figure 4 - TTC Accuracy](https://tau0-vla.github.io/media/ttc-accuracy.png)

**说明**: 跨四个场景比较「一次性规划（Plan Once）」、「Best-of-N」和「TTC（Ours）」的下一步子任务预测精度。TTC 在所有场景中均显著领先，尤其在分布偏移（OOD）场景下提升最大。

### Figure 5: Compute vs. Accuracy / 计算代价与精度权衡

![Figure 5 - TTC Scaling](https://tau0-vla.github.io/media/ttc-scaling.png)

**说明**: 随计算预算增加，子任务预测精度呈现**饱和曲线**，边际收益递减——表明存在最优计算分配点，过度搜索收益有限。

---

### Table 1: 长时序任务成功率对比（每场景各 10 次试验）

| 方法 | Clean Room | Prepare Ingredients | Tomato & Egg | Make Milk Tea | **平均 SR** |
|------|-----------|--------------------|--------------|--------------|-----------:|
| GR00T N1.7 | 0/10 (59.80%) | 1/10 (68.57%) | 0/10 (24.32%) | 0/10 (28.46%) | **2.50%** |
| LingBot-VLA | 0/10 (66.60%) | 0/10 (35.00%) | 0/10 (12.27%) | 0/10 (63.85%) | **0.00%** |
| π₀.₅ | 4/10 (86.20%) | 2/10 (73.93%) | 0/10 (49.77%) | 3/10 (82.31%) | **22.50%** |
| τ₀-VLA (direct) | 4/10 (92.80%) | 2/10 (66.43%) | 0/10 (65.00%) | 5/10 (96.15%) | **27.50%** |
| **τ₀-VLA (hierarchical)** | **5/10 (94.80%)** | **4/10 (82.86%)** | **4/10 (81.82%)** | **5/10 (91.92%)** | **45.00%** |

**表格说明**: 层次化 τ₀-VLA 以 45% 成功率大幅领先 π₀.₅（22.5%）。最难任务「炒西红柿鸡蛋」中，其余所有基线均为 0，τ₀-VLA (hierarchical) 达到 4/10（81.82% 步骤完成率）。

### Table 2: 跨体态直接执行性能（各 10 次试验）

| 方法 | Collect Laundry | Cotton Pad | Eyelash Curler | Makeup Puff |
|------|-----------------|-----------|----------------|-------------|
| GR00T N1.7 | 4/10 (76.00%) | 10/10 (87.50%) | 8/10 (77.50%) | 7/10 (52.50%) |
| LingBot-VLA | 2/10 (35.00%) | 9/10 (67.50%) | 3/10 (22.50%) | 3/10 (33.75%) |
| π₀.₅ | 9/10 (88.00%) | 9/10 (85.00%) | 8/10 (85.00%) | 7/10 (73.75%) |
| **τ₀-VLA** | **10/10 (97.00%)** | **10/10 (95.00%)** | **9/10 (92.50%)** | **10/10 (95.00%)** |

**表格说明**: 低层策略单独使用时，τ₀-VLA 跨所有四个体态/任务场景全面超越所有基线，收取衣物和 Makeup Puff 达到满分 10/10，展示强大的多体态泛化能力。

### Table 3: 闭环测试时计算性能（各 10 次试验）

| 方法 | Make Milk Tea | Book Organization | Clean Room |
|------|---------------|-------------------|-----------|
| Plan Once | 5/10 (91.92%) | 6/10 (66.67%) | 5/10 (94.80%) |
| **TTC (Ours)** | **7/10 (95.38%)** | **9/10 (93.33%)** | **7/10 (97.60%)** |

**表格说明**: 闭环测试中 TTC 在所有三个场景均超越一次性规划，Book Organization（含 OOD 变体）从 6/10 提升至 9/10，提升最显著。

---

## 实验

### 数据集

| 数据集/平台 | 规模 | 特点 | 用途 |
|-------------|------|------|------|
| 真实世界异构数据 | 40,115 小时 | 覆盖固定基座、移动、双臂三类体态 | 低层策略训练 |
| 多模态协同训练数据 | — | 指令跟随、空间推理、感知等 | 低层策略知识隔离阶段 |
| 任务演示标注 | — | 含任务、阶段、可执行子任务标注 | 高层策略监督 |

### 评估平台

- **AGIBOT G1**: 轮式人形机器人，7-DoF 双臂，全向底盘
- **ARX AC One**: 双臂 6-DoF 固定基座
- **Franka Research 3**: 双臂 7-DoF 力矩控制

### 实现细节

- **低层 Backbone**: Qwen3.5-2B 视觉语言模型
- **高层 Backbone**: Qwen3.5-9B（提案/价值/反思模型）
- **世界模型初始化**: Step1X-Edit
- **动作空间**: 40 维统一空间
- **低层训练三阶段**: 知识隔离协同训练 → 端到端协同训练 → 任务特定微调
- **束搜索参数**: $N=3$, $B=2$, 深度 $D$ 依任务而定

---

## 批判性思考

### 优点
1. **测试时可扩展**: TTC 无需重训练即可通过增加计算提升性能，类似 LLM 领域的 reasoning-time compute scaling。
2. **真实长时序评估**: 25 步 88 分钟任务是 VLA 领域最难的公开评估之一，45% SR 有实际意义。
3. **多体态统一**: MoT + 掩码流匹配的统一设计有效避免了多平台碎片化训练问题。

### 局限性
1. **世界模型质量瓶颈**: 图像预测的准确性直接影响 beam search 质量；世界模型幻觉可能误导价值模型打分。
2. **计算代价饱和**: Figure 5 显示精度随计算增加而饱和，极限性能受制于模型本身容量而非搜索深度。
3. **数据规模壁垒**: 40,115 小时数据是绝大多数实验室无法复现的条件。

### 潜在改进方向
1. 引入**不确定性感知 beam search**（对世界模型预测置信度低的候选赋予惩罚权重）
2. 将 TTC 与**在线学习**结合，利用执行反馈实时更新价值模型

### 可复现性评估
- [ ] 代码开源（项目页面暂未挂出代码链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（论文中描述较充分）
- [ ] 数据集可获取（40,115 小时私有数据）

---

## 关联笔记

### 基于
- [[GR00T N1.7]]: Nvidia 通用机器人基础模型，作为强基线对比
- [[pi0.5]]: Physical Intelligence 层次化 VLA，主要竞品
- [[Step1X-Edit]]: 世界模型初始化来源
- [[Qwen]]: 语言模型 backbone

### 对比
- [[LingBot-VLA]]: 同类层次化 VLA，对比长时序性能
- [[RoboMonkey]]: 动作级别 TTC 的代表，对比子任务级别 TTC 的优势

### 方法相关
- [[Hierarchical Policy]]: 核心层次化策略框架
- [[World Model]]: 视觉世界模型用于规划
- [[Beam Search]]: 测试时搜索算法
- [[Mixture-of-Transformers]]: 低层动作专家架构
- [[Masked Flow Matching]]: 统一体态动作生成
- [[Execution Memory]]: 记忆机制维持长时序状态
- [[Test-Time Computation]]: 推理时计算扩展

### 硬件/数据相关
- [[AGIBOT G1]]: 主要评估机器人平台

---

## 速查卡片

> [!summary] τ₀-VLA (2026)
> - **核心**: 层次化 VLA + 世界模型引导束搜索，实现长时序机器人操纵
> - **方法**: Qwen3.5-9B 高层规划（提案→世界模型预测→价值打分→束搜索→反思模型） + Qwen3.5-2B + MoT 低层执行
> - **结果**: 45% 长时序任务 SR（vs. π₀.₅ 22.5%），低层跨体态性能全面领先
> - **代码**: https://tau0-vla.github.io/

---

*笔记创建时间: 2026-08-19*

---
title: "World In-Context Learning for Agentic Robot Manipulation"
method_name: "RoboScript"
authors: [Heng Fang, Zehui Guo, Yuehan Zhang, Mingyang Du, Dingkang Liang, Xiang Bai]
year: 2026
venue: arXiv
tags: [in-context-learning, agentic-manipulation, foundation-model, sim-to-real, manipulation, world-model]
zotero_collection: Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2611.12885
created: 2026-11-13
---

# 论文笔记：World In-Context Learning for Agentic Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 华中科技大学（HUST）、ArcheOne AI |
| 日期 | November 2026 |
| 项目主页 | [ArcheOne-AI.github.io/RoboScript](https://ArcheOne-AI.github.io/RoboScript) |
| 对比基线 | [[FastWAM]], [[4D-WAM]], [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.12885) / [Code](https://github.com/ArcheOne-AI/RoboScript) |

---

## 一句话总结

> RoboScript 提出"世界上下文学习（WICL）"范式，让冻结权重的基础模型通过在仿真器中排练、将反馈写入 context 而非权重，以 1 个演示在 50 个双臂任务上达到 83.4% 成功率，超越最强训练策略 20+ 点。

---

## 核心贡献

1. **WICL 范式（World In-Context Learning）**: 将 [[In-Context Learning]] 与世界反馈结合——冻结权重的模型从演示出发、在仿真器中排练、将经验蒸馏进 context，而非梯度更新权重，实现跨场景泛化。
2. **RoboScript 模块化基础设施**: 严格数据隔离（部署层 D vs 特权层 P）保证学习到的 [[Task Memory]] 可直接在真实机器人上运行；方法、基准、机体均可插件化替换。
3. **递归自我改进（RSI）**: 冻结权重下，反思 agent 读取部署记录、写入 Guidance，三轮净提升 +18/+8/+4，无需梯度更新或重新排练。

---

## 问题背景

### 要解决的问题

训练型动作策略（[[VLA]]、[[WAM]]）将操作技能写入网络权重，导致"绑定到训练分布"——FastWAM 在干净场景 77.8%、随机化场景仅 1.9%。每个新任务/分布都需要新数据和新训练。

### 现有方法的局限

[[In-Context Learning]] 避免了重训耦合，但缺乏来自世界的反馈：演示来自另一场景，无法告诉机器人当前面对的桌面会发生什么。Code as Policies 等方法无世界反馈时误差累积；Frontier 模型在提示中给演示也不改善（GPT-6 Astra 实验 [92]）。

### 本文的动机

关键观察：**Agent 在 context 中学到的是语义的**——任务步骤与对象角色，而非视觉外观。在一个世界（仿真器）中排练获得的经验，只要语义一致就可迁移到另一个世界（真实机器人）。这与 Coding Agent 的逻辑相同：terminal 是软件世界，仿真器是机器人世界。

---

## 方法详解

### 整体范式：WICL 与三大学习范式的关系

| 范式 | 学习来源 | 知识存储 |
|------|----------|----------|
| [[Imitation Learning\|模仿学习]] | 演示 | 权重 |
| [[Reinforcement Learning\|强化学习]] | 演示 + 世界反馈 | 权重 |
| [[In-Context Learning]] | 演示 | Context |
| **WICL（本文）** | **演示 + 世界反馈** | **Context** |

### RoboScript 四阶段流水线

RoboScript 将 WICL 实现为冻结权重下的完整生命周期：

- **技能构建（Skill Building）**: 对所有 50 个任务的演示分段，自动识别现有入口无法解释的 gap，调用 [[Coding Agent]] 编写环境专属技能，驱动器验证后冻结。
- **任务学习（S1 + S2）**: S1 模仿从演示初始化 [[Task Memory]]；S2 在新训练实例上排练，仅第一次尝试计入统计，失败后 reviewer 编辑缓存副本再重试，合并条件是重试成功。
- **部署（Deployment）**: 以冻结 [[Task Memory]] 对新场景执行，含场景预判（scene anticipation）和无重置故障恢复（failure recovery）。
- **反思（Reflection）**: 读取 episode 记录，识别失败模式，写入 Guidance 文档，实现 [[Recursive Self-Improvement]] 而无需梯度。

### 感知与动作接口

**几何优先感知**（Geometric-first perception）：
- 深度通道生成高度图 → [[Connected Component|连通域提议]] → 3D 边界体 + 属性卡（中心、范围、朝向）
- 对象身份：Set-of-Mark 标注、[[Grounding Model]] 和 [[VLM]] 三方投票
- 识别后通过夹持关系追踪，不在每步重检测

**带回执的动作接口（Action Interface with Receipts）**：
- 三层 API：共享入口（抓取、放置、协调双臂等）→ 环境绑定（坐标系、关节限位等）→ 环境特定技能
- 每个动作返回 receipt：指令位姿、实际到达位姿、夹持器开合量、是否持有对象
- Receipts 来自本体感知和运动规划器，不依赖仿真器状态，真实机器人上也可用

### Task Memory 结构

[[Task Memory]] 是结构化 Markdown 文档，包含五个部分：
- **指令与对象角色**：任务目标 + 每个对象的角色与识别方式
- **工作流（Workflow）**：按顺序的操作阶段，每阶段含目标、使用手臂、API 调用和完成条件
- **参数表**：~21 个类型化规格（GraspSpec、PlaceSpec 及标量阈值），通过排练习得
- **Lessons**：排练中 reviewer 识别的失败模式
- 所有规格均用对象角色和相对几何表达，**从不包含绝对坐标**（linter 强制）

### 数据隔离（Information Boundary）

| 层级 | 包含数据 | 可访问角色 |
|------|----------|-----------|
| 可部署层（D） | RGB-D 图像、相机参数、本体感知、运动规划器反馈 | 所有 actor |
| 特权层（P） | 真实物体位姿、资产网格、接触、GT 分割、成功信号 | 仅控制平面（S1/S2 reviewer 诊断） |

**关键**：acting agent 在排练和评估中只看 D 层数据，确保学到的 [[Task Memory]] 可直接部署在真实机器人上。

---

## 关键公式

### 公式 1：[[In-Context Learning|WICL 初始化]]

$$
C_0 = I_\theta(d)
$$

**含义**：从演示 $d$ 初始化 context，$I_\theta$ 是 reader role 对演示的理解过程。

**符号说明**：
- $C_0$：初始 context（Task Memory 草稿）
- $I_\theta$：冻结模型的初始化函数
- $d$：单个专家演示轨迹

### 公式 2：[[World In-Context Learning|WICL 学习更新]]

$$
(\tau_k, r_k) \sim W^{\pi_\theta(\cdot \mid o, C_k)},\ e_k
$$

$$
C_{k+1} = U_\theta(C_k,\ \tau_k,\ r_k)
$$

**含义**：agent 以当前 context $C_k$ 在世界 $W$ 的第 $k$ 个排练实例 $e_k$ 上执行，收集轨迹 $\tau_k$ 和反馈 $r_k$，然后 reviewer role 更新 context（所有权重冻结）。

**符号说明**：
- $\tau_k$：排练轨迹（动作序列）
- $r_k$：世界反馈（包含 receipts 和成功信号）
- $U_\theta$：context 更新函数（reviewer role）
- $e_k \in E_{reh}$：排练实例集（与评估集不重叠）

### 公式 3：[[In-Context Learning|WICL 部署]]

$$
a_t \sim \pi_\theta(\cdot \mid o_t,\ C^*),\quad e \in E_{test}
$$

**含义**：以冻结的 $C^*$ 在未见实例上执行，不从世界接收反馈，确保测量的是泛化而非记忆。

**符号说明**：
- $C^*$：排练收敛后冻结的 Task Memory
- $E_{test}$：与 $E_{reh}$ 不重叠的评估实例集

### 公式 4：[[Recursive Self-Improvement|递归自我改进]]

$$
G_{j+1} = V_\theta(G_j,\ D_j)
$$

**含义**：在第 $j$ 个评估块上，反思 agent 读取 episode 记录 $D_j$，诊断失败模式，更新 Guidance $G_{j+1}$，权重全程冻结。

**符号说明**：
- $G_j$：第 $j$ 轮的 Guidance 文档（$G_0 = \emptyset$）
- $D_j$：第 $j$ 个评估块的 episode 记录（仅含可部署数据 + 成功与否）
- $V_\theta$：反思函数（frozen model）
- $B_0, B_1, \ldots$：不重叠的评估实例块

### 公式 5：[[Transfer Across Worlds|跨世界迁移条件]]

$$
\sigma(W) = \sigma(W') \implies C^* \text{ 在 } W' \text{ 上有效}
$$

**含义**：若世界 $W'$ 与排练世界 $W$ 在语义内容（对象、角色、指令、相对几何）上一致，则 $C^*$ 可直接迁移。权重学习则会吸收 $W$ 中所有帮助拟合训练数据的规律，包括 $W'$ 不共享的部分。

**符号说明**：
- $\sigma(e)$：实例的语义内容（对象、角色、指令、相对几何）
- $\sigma(W)$：世界 $W$ 中实例的语义集合

---

## 关键图表

### Figure 1：三种学习范式对比与主结果

![Figure 1 — 范式对比与主结果](https://archeone-ai.github.io/RoboScript/static/images/1-intro.webp)

**说明**：(a) 三种范式——权重学习（绑定训练分布）、ICL（绑定示范场景）、**WICL**（绑定语义，可跨世界迁移）。(b) RoboScript 将 coding agent 栈映射到机器人操作，仿真器替代 terminal。(c) 50 个 RoboTwin 2.0 任务 1 演示无梯度主结果：RoboScript(Sol) 83.4%，远超最强训练策略 4D-WAM 61.7%。

### Figure 2：RoboScript 基础设施架构

![Figure 2 — Infrastructure](https://archeone-ai.github.io/RoboScript/static/images/5-infra-overview.webp)

**说明**：每个组件作为独立服务运行，模型网关统一管理所有 LLM 调用。Agent plane 只通过可部署接口（D）访问世界；仅 Control plane 可访问特权数据（P）。方法、基准、机体均为可替换插件。

### Figure 3：WICL 管道与递归自我改进

![Figure 3 — WICL pipeline & RSI](https://archeone-ai.github.io/RoboScript/static/images/2-WICL.webp)

**说明**：(a) 从 1 个演示出发，在世界中排练，冻结 context，部署到未见实例。(b) RSI——在不重叠评估块上迭代写 Guidance，全程冻结权重。(c) Context 知识引用共享语义 $\sigma$，可跨世界迁移；权重则不能。

### Figure 4：RoboScript 四个阶段

![Figure 4 — RoboScript stages](https://archeone-ai.github.io/RoboScript/static/images/3-RoboScript.webp)

**说明**：技能构建 → 任务学习（S1 模仿 + S2 排练）→ 部署（含预判与恢复）→ 反思（RSI）。绿色边框表示可部署层数据；橙色边框表示特权层数据（仅限排练）。

### Figure 5 & 7：随机化场景下的成功率保持

![Figure 5 & 7 — Randomization retention](https://archeone-ai.github.io/RoboScript/static/images/plot-random.webp)

**说明**：(a) 每任务干净 vs 随机化成功率散点图。π0.5（训练策略）大多任务点在对角线下方，每类别下降 18.7–31.8 点；RoboScript 三个模型散布对角线两侧，均值变化 ≤ 8 点。(b) GPT-6.1-Sol 500 失败 episode 的失败原因分布，知识失败在干净/随机化场景几乎不变（37 vs 38）。

### Figure 6：失败案例分析

![Figure 6 — Failure cases](https://archeone-ai.github.io/RoboScript/static/images/plot-failure.webp)

**说明**：GPT-6.1-Sol 四类失败各一例。(a) 感知——杯子未被识别；(b) 知识——存储放置使手机倾斜 58°；(c) 执行——入座时马克杯掉落；(d) 假接受——扇子偏转 34° 通过自检但失败官方检测。

### Figure 8：冻结 Task Memory 的部署能力

![Figure 8 — Deployment qualitative](https://archeone-ai.github.io/RoboScript/static/images/plot-qual.webp)

**说明**：三个部署能力示例：(a) 镜像演示到另一只手臂；(b) 验证失败后回溯到跳过的阶段；(c) 重新抓取掉落的瓶子（重感知 → 新抓取规格 → 成功）。

### Figure 9：递归自我改进过程

![Figure 9 — RSI](https://archeone-ai.github.io/RoboScript/static/images/plot-rsi.webp)

**说明**：GPT-6.1-Sol 在 17 个弱任务（成功率 ≤ 8/10）上三轮 RSI，净提升 +18/+8/+4 个 episode；Guidance 条目从 74 增长到 112；单条 Guidance 可将某任务从 5/10 提升到 9/10。

### Figure 10：真实机器人设置

![Figure 10 — Real robot setup](https://archeone-ai.github.io/RoboScript/static/images/real-robo.webp)

**说明**：双臂 PiPER 机器人（6-DoF × 2，平行夹爪）+ RGB-D 相机，与仿真 ALOHA 运动学结构不同。Memory 通过轻量适配器（坐标系 + 运动学 + 夹爪参数重映射）迁移，无需重录演示或重排练。

### Figure 13：学习与部署成本

![Figure 13 — Cost analysis](https://archeone-ai.github.io/RoboScript/static/images/plot-cost.webp)

**说明**：(a) 分阶段成本——评估（P3）占大头。(b) S1/S2 迭代次数。(c) 每次调用的输出 token——DeepSeek-V4.1-Flash 约为 GPT-6.1-Sol 的 19 倍。(d) 每任务 token 用量与成功率成负相关（难任务消耗更多 token）。

### Table 1：RoboTwin 2.0 主结果

| 方法 | 演示数/任务 | 训练 | Agent | 干净(%) | 随机化(%) | 均值(%) | 保持率(%) |
|------|-----------|------|-------|---------|-----------|---------|----------|
| StarVLA | 50 | ✓ | ✗ | 46.5 | 3.2 | 24.8 | 6.8 |
| GR00T-N1.7 | 50 | ✓ | ✗ | 43.6 | 20.7 | 32.2 | 47.5 |
| GalaxeaVLA | 50 | ✓ | ✗ | 62.7 | 12.7 | 37.7 | 20.3 |
| Xiaomi-R0 | 50 | ✓ | ✗ | 62.9 | 18.2 | 40.6 | 28.9 |
| Abot-M0 | 50 | ✓ | ✗ | 57.4 | 30.4 | 43.9 | 52.9 |
| X-VLA | 50 | ✓ | ✗ | 68.0 | 20.9 | 44.5 | 30.7 |
| Spatial Forcing | 50 | ✓ | ✗ | 77.2 | 26.7 | 52.0 | 34.6 |
| π0.5 | 50 | ✓ | ✗ | 70.7 | 46.0 | 58.4 | 65.1 |
| AHA-WAM | 50 | ✓ | ✗ | 64.3 | 3.2 | 33.8 | 5.0 |
| FastWAM | 50 | ✓ | ✗ | 77.8 | 1.9 | 39.9 | 2.4 |
| X-WAM | 50 | ✓ | ✗ | 70.0 | 25.8 | 47.9 | 36.9 |
| LingBot-VA | 50 | ✓ | ✗ | 80.7 | 34.6 | 57.7 | 42.9 |
| **4D-WAM** | 50 | ✓ | ✗ | 81.5 | 41.8 | **61.7** | 51.3 |
| HarnessVLA (Claude Code) | N+1 | ✓ | ✓ | — | 58.4 | — | — |
| **RoboScript (DeepSeek-V4.1-Flash)** | **1** | ✗ | ✓ | 77.8 | 78.4 | 78.1 | 100.8 |
| **RoboScript (GPT-5.6-Luna)** | **1** | ✗ | ✓ | 79.8 | 76.0 | 77.9 | 95.2 |
| **RoboScript (GPT-6.1-Sol)** | **1** | ✗ | ✓ | **83.6** | **83.2** | **83.4** | **99.5** |

**关键发现**：1 个演示 + 排练超越 50 个演示训练的最强策略 21.7 点；保持率 95–101%（训练策略最高 65.1%）。

### Table 2：消融实验（GPT-6.1-Sol，10 任务）

| 变体 | 干净(%) | 随机化(%) | 均值(%) |
|------|---------|-----------|---------|
| **RoboScript（完整）** | **77.0** | **80.0** | **78.5** |
| w/o 手臂检查 | 78.0 | 72.0 | 75.0 |
| w/o 完成检查 | 72.0 | 82.0 | 77.0 |
| w/o 恢复 | 68.0 | 70.0 | 69.0 |
| 演示→直接写 Memory（无世界） | 69.0 | 69.0 | 69.0 |
| 演示→部署 context（无世界） | 69.0 | 66.0 | 67.5 |
| 仅指令（无演示、无世界） | 68.0 | 68.0 | 68.0 |
| **RoboScript（重学习）** | 87.0 | 85.0 | 86.0 |
| w/o 演示 | 58.0 | 55.0 | 56.5 |
| w/o 特权监督 | 87.0 | 86.0 | 86.5 |

**关键发现**：(1) 演示 + 世界缺一不可（无世界 −9.5pp，p=0.003）；(2) 无演示是最大 gap（−29.5pp，p=0.004）；(3) 恢复机制影响最大（−9.5pp）；(4) 特权监督不是性能来源（w/o 特权监督 86.5% ≈ 完整 86.0%）。

### Table 3：基础模型对比与成本

| 模型 | 均值(%) | S1 通过 | S2 实例数/任务 | 评估调用次数 | 评估 token(M) | 评估步数 | 总成本(USD) |
|------|---------|---------|--------------|------------|--------------|---------|------------|
| DeepSeek-V4.1-Flash | 78.1 | 50/50 | 11.1 | 53.1 | 3.89 | 183 | 739 |
| GPT-5.6-Luna | 77.9 | 47/50 | 14.5 | 62.2 | 2.61 | 190 | 308 |
| GPT-6.1-Sol | 83.4 | 49/50 | 12.5 | 54.7 | 2.33 | 149 | 1,911 |

---

## 实验

### 基准与协议

**RoboTwin 2.0** [11]：50 个双臂任务，干净设置（固定背景）+ 随机化设置（纹理/光照/桌高/干扰物随机化）。每任务 10 个 episode × 2 设置 = 1,000 个 episode 总评估。使用官方 server + 未见指令 + 官方步数限制。

### 实现细节

- **基础模型**：GPT-6.1-Sol / GPT-5.6-Luna / DeepSeek-V4.1-Flash（相同代码、提示、协议）
- **排练规模**：每任务平均 11–15 个排练实例，S1 总计 301–518 次尝试
- **硬件**：16× NVIDIA RTX 4090（2 台服务器），基础模型通过 provider API 访问
- **部署延迟**：GPT-6.1-Sol 中位 6.4 分钟/episode（约 80% 时间等待 LLM）
- **超参数**：S2 早停：近 5 次计入实例中有 3 次首试成功；步长 $\alpha_k = \max(0.2, 10/(9+k))$；编辑预算随 $k$ 递减

### 真实机器人结果

| 任务 | 成功率 |
|------|--------|
| Click Bell | 10/10 |
| Grab Roller | 7/10 |
| Move Playingcard Away | 7/10 |
| Stack Bowls Two | 6/10 |

Memory 在仿真 ALOHA 上排练，通过轻量适配器（坐标约定、手臂运动学、夹爪参数重映射）迁移到 PiPER，工作流和相对规格不变，无需重新录制演示或重排练。失败主要源于夹持器未对准和真实传感器的深度估计误差。

---

## 批判性思考

### 优点

1. **颠覆性范式**：WICL 将"技能在权重 vs 在 context"的问题清晰化，并实验证明 context 路径不仅可行而且在分布外场景有优势。
2. **严格数据隔离**：明确区分 D/P 两层，从工程上保证 real-robot deployability，不仅仅是事后验证。
3. **模型无关**：同一套代码/提示/协议下，DeepSeek-V4.1-Flash 也超过所有训练策略，说明结果不依赖特定 LLM。
4. **RSI 无需梯度**：Guidance 写入 context 的递归自我改进框架，为持续进化提供了干净的路径。

### 局限性

1. **仿真器依赖**：WICL 需要可重置的 [[Simulator]]，真实世界不可直接排练；本文的"真实机器人"实验仍依赖仿真训练的 Memory。
2. **成本高**：GPT-6.1-Sol 总成本 1,911 USD（主要是评估），实际部署中学习成本可摊销，但对低资源场景仍是障碍。
3. **长上下文负担**：Task Memory + 完整 episode 历史在每步都作为输入，token 消耗 2.33M/episode；对于 context window 更小的模型会受限。
4. **感知瓶颈**：166 个失败 episode 中，感知失败（对象识别错误）是主要原因之一，当前深度图方案对光线、遮挡敏感。
5. **泛化边界未充分探索**：当前评估仅在 RoboTwin 2.0（结构化双臂任务）；语义迁移的前提（$\sigma(W)=\sigma(W')$）在更自由形式的场景下是否成立未验证。

### 潜在改进方向

1. 用 [[Learned World Model]] 替换仿真器，支持更广泛对象的 WICL 排练
2. 将 Task Memory 压缩/向量化以降低长上下文成本
3. 与 [[Sim-to-Real Transfer]] 方法结合，解决感知域差距

### 可复现性评估

- [x] 代码开源（github.com/ArcheOne-AI/RoboScript）
- [ ] 预训练模型（Task Memories 未公开）
- [x] 训练细节完整（附录 C 超参数表）
- [x] 数据集可获取（RoboTwin 2.0 公开基准）

---

## 关联笔记

### 基于

- [[In-Context Learning]]：WICL 扩展 ICL，增加世界反馈
- [[Coding Agent]]：coding agent = terminal 是世界；本文 = simulator 是世界
- [[Reflexion]]：Reflexion 在单 episode 内存反思；WICL 跨 episode 蒸馏经验

### 对比

- [[FastWAM]]：干净 77.8% → 随机化 1.9%（训练策略的分布绑定问题典型案例）
- [[4D-WAM]]：最强训练策略基线（50 演示，61.7% 均值）
- [[π0.5]]：VLA 类最强基线（50 演示，58.4% 均值）
- [[HarnessVLA]]：有训练策略的 Agent 基线（随机化 58.4%）

### 方法相关

- [[World In-Context Learning]]：本文提出的核心范式
- [[Task Memory]]：本文提出的可冻结语义知识载体
- [[Recursive Self-Improvement]]：本文实现的无梯度递归改进
- [[In-Context Learning]]：WICL 的前身
- [[Sim-to-Real Transfer]]：Task Memory 跨机体迁移

### 硬件/数据相关

- [[RoboTwin 2.0]]：50 任务双臂操作基准
- [[PiPER]]：真实机器人平台（6-DoF 双臂）
- [[ALOHA]]：仿真中使用的机体

---

## 速查卡片

> [!summary] RoboScript (WICL) — World In-Context Learning for Agentic Robot Manipulation
> - **核心**: 冻结权重 + 仿真器排练 → 语义 Task Memory，跨场景/机体迁移
> - **方法**: WICL 四阶段（技能构建→S1 模仿→S2 排练→部署+RSI），严格 D/P 数据隔离
> - **结果**: 1 演示无梯度 83.4%（RoboTwin 2.0），超最强训练策略 21.7pp；真实 PiPER 机器人 75% 成功率
> - **代码**: [github.com/ArcheOne-AI/RoboScript](https://github.com/ArcheOne-AI/RoboScript)

---

*笔记创建时间: 2026-11-13*

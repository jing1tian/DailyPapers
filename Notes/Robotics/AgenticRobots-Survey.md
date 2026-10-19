---
title: "Towards Agentic Robots: Model, Data, Environment, and Harness"
method_name: "AgenticRobots-Survey"
authors: [Chonghan Qin, Dongming Wu, Kaituo Feng, Jiaxu Wang, Xinlei Yu, Peiwen Sun, Yijia Fan, Xilin He, Bingao Chen, Weitong Lian, Ryan Ji, Wenqi Ren, Xiaobin Hu, Dahua Lin, Yu-Gang Jiang, Xiangyu Yue]
year: 2026
venue: arXiv
tags: [agentic-robotics, survey, vla, world-model, robot-policy, embodied-ai, harness, data-collection]
zotero_collection: 3-Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2610.20442v1
created: 2026-10-19
---

# 论文笔记：Towards Agentic Robots: Model, Data, Environment, and Harness

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CUHK (MMLab), Fudan University, NovaXBot, Tsinghua University, Zhejiang University, Sun Yat-sen University |
| 日期 | October 2026 |
| 项目主页 | [Awesome-Agentic-Robots (GitHub)](https://github.com/) |
| 对比基线 | [[VLA]]、[[World Model]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.20442) / Code: N/A (Survey) |

---

## 一句话总结

> 系统整理 Agentic Robots 领域的 17 类研究方向，覆盖 Model、Data、Environment、Harness 四大域，从 VLA/世界模型演化路径出发，厘清各组件如何协作使机器人能自主规划、执行与从失败中恢复。

---

## 核心贡献

1. **统一框架**: 将 Agentic Robotics 组织为 Model、Data、Environment、Harness 四大域，17 个非互斥类别，提供跨方向的统一视角
2. **系统分析**: 梳理从传统策略 → VLA → World Model → Agentic Robot 的演化路径，并对每类方法提供代表性工作对比表格
3. **研究议程**: 提出 Fair Benchmarking、Real-robot 部署、Efficient Robot Use、Cross-Embodiment Generalization 四大未来方向

---

## 问题背景

### 要解决的问题

现有机器人学习主要聚焦于低层次动作预测，缺乏对长时序任务的目标导向编排能力——面对执行失败、信息缺失或环境变化时，机器人无法自主调整策略。

### 现有方法的局限

- 传统策略（[[Diffusion Policy]]、[[ACT]]）擅长低层控制，但缺乏高层规划和自我纠错
- [[VLA]] 引入了语言推理，但单次调用难以处理长时序多步骤任务中的失败恢复
- [[World Model]] 提供了想象能力，但与执行循环的闭环整合仍是挑战

### 本文的动机

基础模型的崛起正在将机器人学习从动作预测转向**目标导向编排**——即数据收集、训练、规划、执行的全流程自主协调。本文将这一范式命名为"Agentic Robots"，并系统整理其现有研究。

---

## 方法详解

### 从机器人策略到 Agentic Robots 的演化路径

[[Agentic Robot]] 是系统演化的第四阶段：

- **阶段 1**: 基于图像的策略（[[RT-1]]、[[Diffusion Policy]]、[[ACT]]、[[Octo]]）和 3D 感知策略（Transporter、PerAct、RVT、Act3D、[[DP3]]）
- **阶段 2**: [[VLA|Vision-Language-Action Models]] — 通用 VLA（RT-2、OpenVLA）、动作表示（离散 token vs 扩散/Flow decoder）、记忆增强、推理增强、想象力增强
- **阶段 3**: [[World Model|世界模型与世界-动作模型]] — 作为环境（Cosmos、Genie Envisioner）、4D 世界模型（PointWorld、TesserAct）、作为规划器（UniPi、Seer）、联合视频-动作建模（UVA、Motus、DreamZero）
- **阶段 4**: [[Agentic Robot]] — 自主规划与协调，解释反馈，持续改进

### 四大域 · 17 类研究方向

#### Domain 1: Model（模型）

**5 类干预**：

1. **Training（训练）**: 奖励设计（Eureka、RDA、EvoNav、Demo2Reward、ReCoVLA）、策略训练工作流（ENPIRE、HARBOR）、实验管理（RoboLineage、AgenticRobotics）
2. **Perception（感知）**: 主动观测请求、视角选择、场景图、过期语义接地风险
3. **Planning（规划）**: 生成计划（SayPlan、Hi Robot、Gemini Robotics-ER 1.5）、执行前检查、执行中更新
4. **Robot Control（机器人控制）**: 动作指令、目标姿态、运动约束和代价函数（VoxPoser、ReKep）、执行反馈
5. **Adaptation（适应）**: 在线适应策略到新环境或新机器人

#### Domain 2: Data（数据）

**3 类操作**：

1. **Collection（采集）**:
   - *物理采集*: AutoRT、SOAR（优先选择尝试次数少的任务）、Zero2Skill（人类辅助重试）、RoboClaw（VLM 控制器 + 重置策略）、Tether、RoboLineage、FIND、UPS
   - *仿真生成*: RoboGen、Video2Policy（网络视频作为任务来源）、GenSim2、RoboFactory（多机器人）、RoboTalk（机器人间消息）
   - *生成式合成*: RoboCurate（视频生成 + 仿真一致性检查）、V-Dreamer、Wh0（人手视频 → 动作标签）

2. **Annotation（标注）**:
   - *文本*: NILS（任务指令）、EgoTools（时序一致性字幕）、ECoT/TIC-VLA/ERVLA（推理轨迹）
   - *空间*: SPARC、Keypose Exploration（轨迹对齐）
   - *交互*: AFUN、EgoAfford、DextER（可供性与接触标注）
   - *结果*: RoboReward、Zero2Skill（成功标签）、ShieldVLA（安全风险标签）

3. **Refinement（精炼）**:
   - *选择*: RoboLineage、Zero2Skill 审查是否值得训练；HELP（VLAC-CUT 评判器）、ARS 保留有价值片段
   - *监督修正*: Zero2Skill 更新标签；Reflective Test-Time Planning 基于后续结果修正动作评估

#### Domain 3: Environment（环境）

**3 类操作**：

1. **Reconstruction（重建）**:
   - 静态结构: ARSTAG、SimFoundry、Genie Sim 3.0、RoboLab、RoboPlayground、GIF
   - 动态和铰接: Agentic Real2Sim、Video2World、4DCodeBench；Articulate-Anything、Articraft、CoDimRecon
   - 构建接口: SceneSmith、SAGE、EmbodiedGen V2；Code4Scene（基于代码的场景编辑）
   - 下游修正: RAPID、Awomo-SimDataEngine、DexSim2Real（Sim-to-Real 随机化调优）

2. **Task Generation（任务生成）**:
   - 可执行构建: GenSim、RoboGen、Video2Policy、HARBOR、PARTNR、GenPHRI、EmbodiedSmith
   - 自适应分布: CurricuLLM、RATs（从失败记忆选目标）、SimWorld Studio（难度调节）

3. **Benchmarking（基准测试）**:
   - 机器人控制: EmbodiedBench、LIBERO/LIBERO-PRO、RoboHarness、RoboDojo、RobotWorld
   - 主动感知: VABench、ActiveArena-Plan
   - 记忆与安全: RoboGraph、MEMOBench、ReactHuman、ManiGuard
   - 智能体驱动的评估构建: ManipEvalAgent、Embodied-BenchForge、A2Eval

#### Domain 4: Harness（执行框架）

**6 类组件**：

1. **Orchestration（编排）**:
   - *路由与分发*: VoLo（执行中切换）、MistyPilot（持久绑定）、EMOS（Robot Resume 能力评估）
   - *函数协议*: EmbodiedTool（类型化描述）、ROSClaw（速度限制 + 接口白名单）、Thea（独立成功条件检查）
   - *状态连接*: RoboHarness（记忆桥接）、RoboRouter（选择适合当前状态的策略）、RoCo（对话验证可行性）
   - *切换时机*: HarnessWAM（进度估计器）、VLAs-as-Tools（有限执行预算）、Retriever（异步时钟 + 提交规则）

2. **Memory（记忆）**:
   - *写入*: HAM-VLN（记录失败+位置）、MessyMem（每次物理交互后更新图）、MEMORA（地点/对象/活动记录）
   - *更新*: Mimir（任务记忆 vs 世界记忆分离）、SafeMem（不可见对象保留在场景图中）、CoRelNav（跨对象实例共享证据）
   - *检索*: BATON（技能切换时检索任务状态）、2AM（浓缩历史为当前阶段摘要）、MeCo（复用相似成功任务的路径点）

3. **Monitoring & Recovery（监控与恢复）**:
   - *执行监控*: SafeHarness（执行前检查路线）、DoReMi（执行中约束检查）、REFLECT（执行后失败解释）
   - *响应选择*: Goal2Skill（重试/调整/重规划三选一）、WCM（预测-实际差异触发重规划）
   - *恢复动作*: VLA-Corrector（输入恢复提示）、FLARE/VoLo（切换执行器）、AgentChord（切换到预编译恢复分支）、HALTER（先重置场景再恢复）

4. **Skills（技能）**:
   - *从技能到代码策略*: Code as Policies、ProgPrompt（带断言的计划代码）、VoxPoser（代价图代码）、Act-Observe-Rewrite
   - *技能库与复用*: Voyager（自验证后入库）、RoboCoder、Teach and Grow（隔离测试）、CaP-X（9 个蒸馏函数）
   - *技能演化*: ASPIRE（基元级 trace 调试）、RATs（成功促进/失败淘汰）、Zetta（演化评判器-恢复代码）、RHO（变异整个策略仓库）
   - *从演化到训练*: RoboClaw/ENPIRE（微调策略）、HarnessPAI/RACaP（蒸馏到 VLA）

5. **Infrastructure（基础设施）**:
   - *能力接口*: EmbodiedTool（100+ 工具 + 能力卡）、RoboNeuron（ROS 2 消息 → MCP 工具）、Show-Harness（语义笛卡尔步骤）
   - *感知与操作工具链*: Grounding DINO、SAM 2/3、FoundationPose、AnyGrasp、cuRobo（GPU 运动规划）
   - *策略推理与服务*: openpi 远程推理（WebSocket）、LeRobot（策略服务器 + 机器人客户端）、SmolVLA（异步推理减少约 30% 任务时间）、ROSA（GPU 池调度）
   - *运行时门控*: RoboGuard（将越狱成功率从 92.3% 降至 2.3%）、Safety Chip（时序逻辑自动机）、SafeGate（未知危害默认拒绝）

6. **Layered Systems（分层系统）**:
   - *System 2*: VLM/LLM 推理层，低频率（~2-10 Hz）
   - *System 1*: VLA 等视觉运动策略，中频率（~30-200 Hz）
   - *System 0*: 解析控制（IK、MPC、全身控制），高频率（控制率）
   - 代表系统: Helix（7-9 Hz 潜向量 → 200 Hz 策略）、GR00T N1（10 Hz 推理 → 120 Hz 动作块）、Ground Slow Move Fast（2 Hz 规划 → 30 Hz 策略 → MPC 追踪）
   - 决策 API: Jev（System 1 决策模型）、Liquid AI d1（200-300 ms 文本决策）

---

## 关键图表

### Figure 1: 机器人智能演化的四个范式

![Figure 1: Paradigm Mountain](https://arxiv.org/html/2610.20442v1/f0_mountain_svg_compressed.png)

**说明**: 四个范式的对比图：Task-Specific Policies（任务特定策略） → VLAs → World/World-Action Models → Agentic Decision-Making（智能决策）。纵轴表示任务泛化能力，横轴表示自主决策程度。

### Figure 2: Agentic Robots 四大域概览

![Figure 2: Four Domain Overview](https://arxiv.org/html/2610.20442v1/f0_cover_svg_compressed.png)

**说明**: 展示 Model、Data、Environment、Harness 四个域如何相互连接，形成完整的 Agentic Robot 闭环：模型决策驱动机器人执行，执行产生经验数据，数据通过数据域处理后反馈模型训练，环境域提供仿真与测试基础设施。

### Figure 3: 分类法与文献全景

![Figure 3: Taxonomy Landscape](https://arxiv.org/html/2610.20442v1/f1_taxonomy_landscape_compatible_compressed.png)

**说明**: 17 个非互斥类别在四大域中的分布与代表论文映射。展示了各方向的论文密度和相互关联关系。

### Figure 4: 模型侧五类干预（抽屉检索任务示例）

![Figure 4: Model Interventions](https://arxiv.org/html/2610.20442v1/f2_model.png)

**说明**: 以一个抽屉检索任务为例，展示 Training、Perception、Planning、Robot Control、Adaptation 五类模型侧干预的具体体现和各干预点在执行流中的位置。

### Figure 5: 数据侧三类干预

![Figure 5: Data Interventions](https://arxiv.org/html/2610.20442v1/f3_data.png)

**说明**: 在包含失败和恢复的抽屉交互中展示三类数据干预：Collection（采集）、Annotation（标注）、Refinement（精炼）在同一机器人共享经验上的作用。

### Figure 6: 环境域三类操作

![Figure 6: Environment Operations](https://arxiv.org/html/2610.20442v1/f4_environment_compressed.png)

**说明**: 展示针对单个机器人任务构建和测试环境的三种方式：Reconstruction（重建可执行场景）、Task Generation（定义目标和成功条件）、Benchmarking（固定测试条件/指标/评估器）。

### Figure 7: Harness 功能组件图

**说明** (图片未在 HTML 前段出现，根据 PDF 文本描述): 以抽屉任务为核心的 Harness 功能架构图。中间行将任务目标和证据连接到 Decision & Dispatch → 执行器 → 机器人执行 → 反馈。六个组件分别从不同层次管理执行：
- Orchestration: 调用和切换管理
- Memory: 历史信息供给
- Monitoring & Recovery: 监测失败并响应
- Skills: 打包可复用行为
- Infrastructure: 暴露和门控可调用能力
- Layered Systems: 在规划层/学习策略层/解析控制层分配权限和运行速率

### Table 1: 17 类研究方向总览 (概念性)

| 域 | 类别 | 代表工作 |
|---|---|---|
| Model | Training | Eureka, RDA, ENPIRE, HARBOR |
| Model | Perception | SayPlan, Hi Robot, Gemini Robotics-ER 1.5 |
| Model | Planning | SayPlan, Hi Robot |
| Model | Robot Control | VoxPoser, ReKep |
| Model | Adaptation | — |
| Data | Collection | AutoRT, SOAR, Zero2Skill, RoboClaw, RoboFactory |
| Data | Annotation | NILS, ECoT, ShieldVLA |
| Data | Refinement | RoboLineage, HELP, ARS |
| Environment | Reconstruction | SimFoundry, Genie Sim 3.0, Articulate-Anything |
| Environment | Task Generation | GenSim, RoboGen, CurricuLLM, RATs |
| Environment | Benchmarking | LIBERO, RoboDojo, RobotWorld |
| Harness | Orchestration | VoLo, EMOS, HarnessWAM |
| Harness | Memory | MessyMem, MEMORA, BATON, MeCo |
| Harness | Monitoring & Recovery | DoReMi, REFLECT, FLARE, HALTER |
| Harness | Skills | Voyager, ASPIRE, RATs, Zetta |
| Harness | Infrastructure | RoboGuard, ROSA, Safety Chip |
| Harness | Layered Systems | Helix, GR00T N1, Harness VLA |

### Table 13: Orchestration 工作对比（编排）

| 工作 | 执行器类型 | 选择方式 |
|---|---|---|
| VoLo | Policy + Tool | Model-based |
| MistyPilot | Skill | Hybrid |
| EMOS | Policy + Skill + Other Agent | Model-based |
| Physical Agency | Policy + Tool | Model-based |
| Guava | Tool | Model-based |
| EmbodiedTool | Tool + Skill + Other Agent | Model-based |
| ROSClaw | Tool | Hybrid |
| Thea | Tool + Skill | Hybrid |
| RoboHarness | Policy + Tool | Model-based |
| RoboRouter | Policy | Model-based |
| RoCo | Policy + Other Agent | Model-based |
| RoboTalk | Policy + Other Agent | Model-based |
| HarnessWAM | Policy + Tool | Model-based |
| VLAs-as-Tools | Policy + Tool | Model-based |
| Retriever | Policy + Tool + Other Agent | Rule-based |

### Table 14: Memory 工作对比（记忆）

| 工作 | 存储格式 | 写入时机 | 检索方式 | 跨回合持久 |
|---|---|---|---|---|
| HAM-VLN | Graph | Step | Query | ✗ |
| HiMe | Text | Event | Query | ✗ |
| MEMORA | Structured | Step | Query | ✓ |
| MessyMem | Graph | Event | Query | ✗ |
| Mimir | Structured | Step | Query | ✗ |
| SafeMem | Graph | Step | Direct | ✗ |
| BATON | Text | Episode end | Check | ✓ |
| 2AM | Structured | Event | Direct | ✗ |
| MeCo | Structured | Episode end | Similarity | ✓ |

### Table 15: Monitoring & Recovery 工作对比

| 工作 | 检查时机 | 检查器 | 信号 | 响应 |
|---|---|---|---|---|
| SafeHarness | Before, during | Rules | Constraint | Replan |
| DoReMi | During | VLM | Constraint | Stop, replan |
| Code-as-Monitor | During | Code | Constraint | Report, replan |
| REFLECT | After | LLM | Failure | Report, replan |
| AHA | After | VLM | Failure | Report |
| Goal2Skill | After | VLM | Failure | Retry, replan |
| VLA-Corrector | During | Learned | Failure | Adjust |
| FLARE | During | VLM | Failure | Retry, reset |
| AgentChord | During | Code | Failure | Switch |
| HALTER | After | LLM | Success | Reset |
| PEACE | After | Rules | Failure | Replan |

### Table 17: Infrastructure 工作对比（部分）

| 工作 | 目的 | 主要操作 |
|---|---|---|
| Safety Chip | 动作安全 | 编译时序逻辑规则，阻断违规动作 |
| RoboGuard | 计划安全 | Model-check 计划，越狱成功率 92.3% → 2.3% |
| RoboPAIR | 对抗测试 | 对自驾/地面机器人/四足实施 100% 越狱 |
| SafeGate | 任务安全 | 未知危害默认拒绝，运行时监控合约 |
| SmolVLA | 连续执行 | 异步推理，减少约 30% 任务时间 |
| ROSA | 集群推理 | GPU 池调度，满足延迟目标 |
| BitVLA | 模型压缩 | 三元权重量化，匹配 4-bit 基线 |
| vla.cpp | 嵌入式推理 | 在嵌入式板上运行量化 VLA |

### Table 18: Layered Systems 工作对比（部分）

| 工作 | 上层 | 下层 | 下传信息 | 更新时机 |
|---|---|---|---|---|
| SayCan | LLM selector | Learned skills | Skill description | After each skill |
| ReKep | VLM | Constraint solver | Keypoint constraints | Solver ~10 Hz |
| Helix | Vision-language model | 200 Hz policy | Latent vector | Async, 7-9 Hz |
| GR00T N1 | VLM @ 10 Hz | Action @ 120 Hz | Intermediate embeddings | Async |
| π0.5 | Subtask predictor | Action expert | Language subtask | — |
| Gemini Robotics 1.5 | VLM orchestrator | VLA tool | Language subtask | On completion/failure |
| Harness VLA | VLM planner | Analytic tools + frozen VLA | JSON primitive call | After each primitive |

---

## 实验

### 基准测试示例

| 系统 | 基准 | 成功率 |
|---|---|---|
| GPT-6 Astra | RoboDojo (42 tasks) | 22.48% |
| NavMCP (完整协议) | 导航 Q&A | 74.0% |
| NavMCP (仅 episode 接口) | 导航 Q&A | 59.1% |
| CaP-X (特权观测) | — | 65.0% |
| CaP-X (仅 RGB) | — | 22.5% |
| Show-Harness (全步长) | — | 60% (zero-shot) |
| Show-Harness (半步长) | — | 82% (zero-shot) |
| ASPIRE (无执行引擎) | LIBERO-Pro | 14% |
| ASPIRE (有执行引擎) | LIBERO-Pro | 62% |
| RoboGuard (无防护) | 越狱测试 | 92.3% 成功 |
| RoboGuard (有防护) | 越狱测试 | 2.3% 成功 |

---

## 批判性思考

### 优点

1. **覆盖全面**: 17 类 × 4 域的框架兼顾了从数据采集到安全门控的完整系统栈，有助于研究者定位自身工作的位置
2. **表格丰富**: 每类都配有详细对比表（Tables 2-18），维度清晰（执行器类型、选择方式、存储格式等），实用性强
3. **演化视角清晰**: 从传统策略到 Agentic Robot 的演化路径描述系统，尤其是 VLA/WAM 部分覆盖了最新进展（2025-2026 年论文）

### 局限性

1. **缺乏量化综合对比**: 作为综述，缺乏跨方向的统一量化评估；各研究使用不同基准，难以直接比较
2. **Harness 概念边界模糊**: Harness 的定义（"让 Agent 能力对机器人可用的软件"）较新且宽泛，不同社区有不同习惯用法
3. **实机结果稀缺**: 绝大多数工作在仿真中验证，实机 Agentic Robotics 仍处早期阶段

### 潜在改进方向

1. 建立跨方向统一的 Agent 贡献评估标准（固定执行器、固定观测预算的受控对比）
2. 深入分析感知工具链错误的传播路径——工具链是当前物理失败的主要来源
3. 研究 Skill Evolution → 模型权重更新的闭环（目前大多数 Skill 演化不修改编码器权重）

### 可复现性评估

- [x] 代码开源（Awesome-Agentic-Robots GitHub 仓库）
- [ ] 预训练模型（综述不提供）
- [x] 训练细节完整（综述引用的论文多提供细节）
- [x] 数据集可获取（各方向有开源数据集）

---

## 关联笔记

### 基于

- [[VLA]]: 核心前置范式，综述的重要出发点
- [[World Model]]: 第三阶段演化，与 Agentic 紧密相关
- [[Diffusion Policy]]: 代表性传统策略
- [[ACT]]: 代表性传统策略

### 方法相关

- [[Agentic Robot]]: 核心概念
- [[Harness]]: 本综述提出的新概念框架
- [[Code as Policies]]: Skill 域的重要起点
- [[Voyager]]: Skill Library 的范式工作
- [[VoxPoser]]: 代码生成 + 运动规划的代表
- [[ReKep]]: 关键点约束 + 分层控制的代表
- [[Diffusion Policy]]: 传统策略基线
- [[SayCan]]: 经典 LLM + 机器人框架

### 硬件/数据相关

- [[LIBERO]]: 常用操作基准
- [[RoboDojo]]: 新的 Agentic 评估基准（42 任务）
- [[RobotWorld]]: 跨 embodiment 通用基准（84 任务）

---

## 速查卡片

> [!summary] Towards Agentic Robots: Model, Data, Environment, and Harness
> - **核心**: 将 Agentic Robotics 系统组织为 4 域 17 类，从演化路径到未来挑战全面梳理
> - **方法**: Model（训练/感知/规划/控制/适应）+ Data（采集/标注/精炼）+ Environment（重建/任务生成/基准）+ Harness（编排/记忆/监控恢复/技能/基础设施/分层系统）
> - **结果**: 综述性工作，提炼了 RoboGuard（越狱 92.3%→2.3%）、ASPIRE（14%→62%）等代表性数量结果
> - **代码**: [Awesome-Agentic-Robots GitHub](https://github.com/)

---

*笔记创建时间: 2026-10-19*

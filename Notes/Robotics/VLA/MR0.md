---
title: "Meituan-Robotics-0: A Vision-Language-Action Foundation Model for Desktop Manipulation"
method_name: "MR0"
authors: [Caicheng Wang, Chen Zhao, Chengpeng Jin, Duo Xu, Fengjiao Chen, Guanglu Wan, Heqing Yang, Jia Wang, Jiacong Zhou, Jie Hu, Manyuan Zhang, Shijie Zhou, Si Liu, Tianyi Zhang, Xi Zhang, Yanli Tan, Yifu Xu, Zilong Huang, Ziqin Wang]
year: 2026
venue: arXiv
tags: [vla, desktop-manipulation, flow-matching, cross-embodiment, cfg-dagger, imitation-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.08577v1
created: 2026-11-10
---

# 论文笔记：Meituan-Robotics-0

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Meituan Robotics Team |
| 日期 | November 2026 |
| 项目主页 | [meituan-robotics.github.io/Meituan-Robotics-0](https://meituan-robotics.github.io/Meituan-Robotics-0/) |
| 对比基线 | [[pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.08577) / [Code](https://github.com/Meituan-Robotics/Meituan-Robotics-0) |

---

## 一句话总结

> MR0 是一个桌面操作 [[Vision-Language-Action]] 基础模型，以三阶段训练（离散预训练→连续微调→真实世界后训练）和 Rollback 数据收集方法，在 RoboTwin 2.0 上达到 93.6% 成功率，并通过事件驱动智能体实现持久任务状态跟踪。

---

## 核心贡献

1. **三阶段渐进式训练**: 离散动作 token 预训练 + [[Flow-Matching]] 连续控制微调 + 真实世界后训练（SFT/HG-DAgger/RECAP/CFG-DAgger），统一 12 个机器人平台的控制。
2. **Rollback 数据收集**: 在失败前"回退"到锚点让操作员给出人类纠正轨迹，同时产生正负样本供 [[CFG-DAgger]] 使用，无需价值学习或优势标注。
3. **事件驱动智能体系统**: Monitor-Memory-Planner 三层架构，持续维护任务状态，仅在条件变化时触发重规划，配合异步 [[Action Chunking]] 执行实现 3 Hz 推理 + 30 Hz 控制。

---

## 问题背景

### 要解决的问题

桌面操作需要处理多机器人平台的数据异构性、策略自身行为引发的分布偏移、以及超出观测窗口的任务状态丢失三大挑战。

### 现有方法的局限

- 单纯扩大动作模型规模不能解决数据格式不一致问题；
- 现有 DAgger 方法（如 [[HG-DAgger]]）只收集正样本，无法利用失败片段的负监督信号；
- 现有 VLA 只做反应式策略，无法处理跨子任务的任务状态变化。

### 本文的动机

通过统一端效器接口规范化 [[Cross-Embodiment]] 数据，引入 [[Rollback]] 收集正负样本对以启用 [[CFG-DAgger]] 和 [[RECAP]] 后训练，以及 [[Event-Driven Agent]] 维护持久状态，三者互补解决上述问题。

---

## 方法详解

### 模型架构

MR0 采用**双头共享骨干**架构：
- **输入**: 多视角图像 $o_t$ + 语言指令 $l$ + 本体感知状态 $s_t$
- **Backbone**: [[Qwen]] 3.5-4B 预训练语言骨干
- **预训练头**: [[FAST]] 离散动作 token 头（FAST tokenizer）
- **微调头**: [[MMDiT]] 连续流匹配动作头（Multimodal Diffusion Transformer）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H-1} \in \mathbb{R}^{H \times 20}$，$H=50$
- **动作空间**: 每臂 10D 末端执行器动作，双臂共 20D

### 核心模块

#### 模块1：跨机器人动作规范化

**设计动机**: 利用 [[Cross-Embodiment]] 统一 12 个机器人平台的异构动作表示，消除不同本体之间的格式差异。

**具体实现**:
- 以肩部中心坐标系（SC 帧）为基准，每臂动作定义为 $a_t^{(i)} = [p^{SC}_{EEF}, r_{6D}, g]^T \in \mathbb{R}^{10}$
- 位置 $p$ 为 3D 笛卡尔坐标，旋转 $r_{6D}$ 使用 6D 连续旋转表示，夹爪 $g \in [0,1]$ 归一化
- 通过机器人特定变换矩阵 $T^{SC}_B$ 将各平台动作转换到统一 SC 帧

#### 模块2：三阶段渐进训练

**阶段一 — 离散预训练**:
- 用 [[FAST]] tokenizer 将连续动作编码为离散 token，以自回归方式预训练
- 联合 VLA 动作损失和 VQA 语言理解损失（[[Set-of-Mark Prompting]] 生成空间标注）
- 在 256 GPU 上训练 500k 步

**阶段二 — 连续中间训练**:
- 切换到 [[MMDiT]] 流匹配动作头，使用 JiT clean-sample 参数化
- 引入遮蔽损失处理不同机器人平台的动作维度差异（20D 中部分有效）
- "langmix" 采样策略混合子任务级和目标级语言指令
- 在 128 GPU 上训练

**阶段三 — 真实世界后训练**:
- 首先 SFT 在演示数据上微调
- 然后选择三种方法之一：[[HG-DAgger]]（人工门控纠正）、[[RECAP]]（优势条件策略）或 [[CFG-DAgger]]（Rollback 正负样本 + 分类器自由引导）

#### 模块3：Rollback 数据收集

**设计动机**: 传统 [[DAgger]] 只收集正样本；Rollback 在快要失败时让操作员"回退"到之前的锚点并给出纠正续接，自然产生配对的正（人类纠正）和负（被替换的策略片段）轨迹。

**具体流程**:
1. 操作员感知策略即将失败时暂停并后退到锚点 $t_1$
2. 人工接管从 $t_1$ 到 $t_2$ 的片段（正样本 $\tau^+$，$I=1$）
3. 被替换的策略片段成为负样本 $\tau^-$（$I=0$）
4. 若场景无法复原，退化为标准 HIL 仅收集正数据

#### 模块4：异步动作块执行

**设计动机**: VLA 推理约 3 Hz，控制器需要 30 Hz，直接拼接会导致执行间断。[[Asynchronous Inference]] 通过维护命令缓冲区和线性混合实现平滑控制。

**具体实现**:
- 推理线程以 ~3 Hz 预测 $H=50$ 步动作块
- 控制线程以 $f_c=30$ Hz 从缓冲区弹出并执行命令
- 新块到来时：先裁剪延迟失效命令，再对重叠部分线性混合，尾部直接拼接

#### 模块5：事件驱动智能体

**设计动机**: 单步 VLA 策略无记忆，[[Event-Driven Agent]] 通过持久事件状态跟踪任务条件，仅在关键变化时触发重规划，避免频繁调用昂贵的 LLM 规划器。

**三层架构**:
- **Monitor** $M$：每 2 秒观测窗口生成结构化字幕 $C_t = M(W_t, q_t)$
- **Memory Manager** $G$：将字幕转为持久事件 $E_t$，判断是否触发重规划（$z_t \in \{0,1\}$）
- **Planner** $P_{plan}$：被触发时重新评估子任务序列，可维持/推进/等待/重生成

---

## 关键公式

### 公式 1：[[Cross-Embodiment|跨机器人规范化动作]]

$$
a_t^{(i)} = \left[p^{SC}_{EEF,t}{}^{(i)},\ r_t^{(i)},\ g_t^{(i)}\right]^T \in \mathbb{R}^{10},\quad a_t = \left[a_t^{(L)},\ a_t^{(R)}\right]^T \in \mathbb{R}^{20}
$$

**含义**: 以肩部中心坐标系统一表示每臂末端执行器动作，双臂拼接得 20D 规范化动作。

**符号说明**:
- $p^{SC}_{EEF}$: 肩部中心坐标系中末端执行器 3D 位置
- $r_t^{(i)}$: 6D 连续旋转表示（由 $\rho_{6D}$ 映射）
- $g_t^{(i)}$: 归一化夹爪开合度，0=开，1=闭
- 上标 $(L),(R)$: 左臂、右臂

### 公式 2：[[Cross-Embodiment|坐标系转换]]

$$
T^{SC}_{EEF,t}{}^{(i)} = T^{SC}_B \cdot T^B_{EEF,t}{}^{(i)},\quad r_t^{(i)} = \rho_{6D}\!\left(R^{SC}_{EEF,t}{}^{(i)}\right)
$$

**含义**: 通过机器人特定变换矩阵 $T^{SC}_B$ 将各平台末端执行器从本体坐标系转到统一肩部中心坐标系。

**符号说明**:
- $T^{SC}_B$: 机器人特定的本体帧 → 肩部中心帧变换矩阵
- $\rho_{6D}$: 旋转矩阵到 6D 连续表示的映射

### 公式 3：[[Action Chunking|反应式动作块策略]]

$$
\pi_\theta\!\left(a_{t:t+H-1} \mid o_t,\ l,\ s_t\right)
$$

**含义**: 给定当前观测、语言指令和本体感知状态，策略一次性预测 $H$ 步动作块，无时序历史依赖。

**符号说明**:
- $H=50$: 动作块长度（预测未来 50 步）
- $o_t$: 多视角 RGB 图像观测
- $l$: 语言子任务指令
- $s_t$: 关节角等本体感知状态

### 公式 4：[[FAST|离散预训练联合损失]]

$$
\mathcal{L}_{pre} = \lambda_{vla} \cdot \mathcal{L}_{action} + \lambda_{vlm} \cdot \mathcal{L}_{vqa},\quad \lambda_{vla}=1.0,\ \lambda_{vlm}=0.1
$$

**含义**: 预训练阶段同时训练动作预测（VLA）和视觉问答（VLM），两者均为自回归交叉熵损失。

**符号说明**:
- $\mathcal{L}_{action}$: FAST token 动作序列的交叉熵损失
- $\mathcal{L}_{vqa}$: 视觉语言理解（2D Grounding、目标可供性等）的交叉熵损失
- $\lambda_{vla}=1.0$: 动作损失权重
- $\lambda_{vlm}=0.1$: VLM 损失权重

### 公式 5：[[Flow-Matching|遮蔽 JiT 流匹配损失]]

$$
\mathcal{L}_{masked} = \frac{\sum_{h=1}^{H} \sum_{j=1}^{d_{max}} m_j \,\ell_{h,j}}{H \cdot \sum_j m_j}
$$

**含义**: 对不同机器人平台（动作维度不同）使用二值遮蔽 $m_j$ 只计算有效通道的 MSE，实现异构机器人统一训练。

**符号说明**:
- $H$: 动作块长度
- $d_{max}=20$: 最大动作维度（双臂各 10D）
- $m_j \in \{0,1\}$: 第 $j$ 通道的有效性掩码
- $\ell_{h,j}$: 第 $h$ 步第 $j$ 维的流匹配损失

### 公式 6-7：[[Asynchronous Inference|异步块对齐与线性混合]]

$$
\bar{Q}_{new} = Q_{new}[d:],\quad L = \min(|Q_{old}|,\ |\bar{Q}_{new}|),\quad d = \min(k,\ K_{lat})
$$

$$
q_i^{fused} = (1 - \alpha_i) \cdot q_i^{old} + \alpha_i \cdot \bar{q}_i^{new},\quad \alpha_i = \frac{i}{\max(L-1,\,1)}
$$

**含义**: 新块到来时先裁剪延迟失效命令 $d$ 步，再对重叠部分做线性插值混合，消除拼接处的速度跳变。

**符号说明**:
- $k$: 自上次更新以来已消耗的命令数
- $K_{lat}=4$: 最大允许的延迟补偿步数
- $L$: 新旧块的重叠长度
- $\alpha_i \in [0,1]$: 线性混合系数，从 0 渐增到 1

### 公式 8：[[Event-Driven Agent|记忆管理器更新]]

$$
(E_t,\ z_t) = G(B_t,\ H_{<t},\ X_t,\ g),\quad z_t \in \{0,1\}
$$

**含义**: 给定当前字幕 $B_t$、历史事件 $H_{<t}$、任务上下文 $X_t$ 和目标 $g$，生成更新后的事件 $E_t$ 并判断是否触发规划器重评估。

**符号说明**:
- $B_t$: Monitor 生成的当前字幕
- $H_{<t}$: 历史事件序列
- $z_t=1$: 触发规划器重评估
- $X_t$: 当前任务状态上下文

### 公式 9-10：[[RECAP|归一化价值目标]]

$$
r_t = \begin{cases} 0 & t=T,\ \text{success} \\ -T_{max} & t=T,\ \text{failure} \\ -1 & \text{otherwise} \end{cases},\quad R_t = \sum_{t'=t}^{T} r_{t'}
$$

$$
y_t = \text{clip}\!\left(\frac{R_t}{D},\ -1,\ 0\right),\quad D = 2T_{max}
$$

**含义**: 以时间惩罚为基础定义轨迹级回报，归一化到 $[-1, 0]$，成功轨迹值趋近 0，越早失败值越低。

**符号说明**:
- $T_{max}$: 最大允许步数
- $D = 2T_{max}$: 归一化因子
- $y_t \in [-1, 0]$: 归一化价值目标

### 公式 11：[[RECAP|分布式价值模型]]

$$
p_\phi(V \mid o_t,\ell) \in \Delta^B,\quad z_i = V_{min} + i\,\frac{V_{max}-V_{min}}{B-1},\quad i=0,\ldots,B-1
$$

$$
\mathcal{L}_V = -\sum_i q_i \log p_\phi(V = z_i \mid o_t,\ell);\quad V(o_t,\ell) = \sum_i p_\phi(V=z_i \mid o_t,\ell)\,z_i
$$

**含义**: 用 $B=201$ 个原子的分类分布建模价值函数，以交叉熵损失训练，期望值作为标量价值估计。

**符号说明**:
- $B=201$: 价值分布的原子数
- $V_{min}=-1,\ V_{max}=0$: 价值范围
- $q_i$: 由标量目标 $y_t$ 投影到相邻原子的分配权重
- $\Delta z=0.005$: 原子间隔

### 公式 12：[[RECAP|N-步优势估计]]

$$
A_t^{(N)} = \sum_{k=0}^{N-1} \tilde{r}_{t+k} + V(o_{t+N},\ell) - V(o_t,\ell),\quad \tilde{r}_t = y_t - y_{t+1}
$$

$$
I_t = \mathbf{1}\!\left[A_t^{(N)} \geq \varepsilon\right],\quad N=H=50,\quad \varepsilon = \text{70th percentile}
$$

**含义**: 计算 $N$ 步优势，仅对优势超过 70 百分位阈值的时间步附加正向条件词，低优势步骤不加额外条件。

**符号说明**:
- $\tilde{r}_t = y_t - y_{t+1}$: 过渡奖励（相邻步价值差）
- $\varepsilon$: 聚合优势的 70 百分位
- $I_t=1$: 该步被正向条件化

### 公式 13：[[RECAP|动作空间分类器自由引导]]

$$
a_{CFG} = a_\emptyset + \beta\,(a_+ - a_\emptyset),\quad \beta = 1.75
$$

**含义**: RECAP 中推理时以无条件预测为基准，向"优势正"条件方向外推，增强正向执行模式。

**符号说明**:
- $a_\emptyset$: 无条件（condition dropout）分支预测
- $a_+$: 正向条件（"Advantage: positive"）分支预测
- $\beta=1.75$: 引导强度

### 公式 14：[[CFG-DAgger|正负分支分类器自由引导]]

$$
a_{CFG} = a_- + \gamma\,(a_+ - a_-),\quad \gamma = 1.25
$$

$$
D^+ = D_{demo} \cup D_{HIL},\quad D^- = \bigcup_j \tau_j^-
$$

**含义**: CFG-DAgger 直接以负样本预测为排斥基准，向正样本方向外推，使策略远离失败模式，引导强度 $\gamma=1.25$。

**符号说明**:
- $a_-$: 负条件（被替换的失败片段）分支预测
- $a_+$: 正条件（演示 + HIL 纠正）分支预测
- $\gamma=1.25$: CFG-DAgger 引导强度（比 RECAP 更保守）
- $D^-$: 来自 Rollback 的失败片段负样本集

---

## 关键图表

### Figure 1：系统概览

![Figure 1 - MR0 Overview](https://arxiv.org/html/2611.08577v1/main_m42_clean_v2.png)

**说明**: MR0 整体架构图，涵盖多模态数据（147M VL 样本 + 15,060h 机器人轨迹）、三阶段 VLA 训练流程、事件驱动智能体系统和真实机器人部署。右侧 RoboDojo 面板按平均成功率排列比较模型。

### Figure 2：视觉语言数据集构成

![Figure 2 - VL Data](https://arxiv.org/html/2611.08577v1/vqa_data1105.png)

**说明**: 约 1.47 亿 VL 样本的分布，78% 为通用数据，22% 为具身数据，包含 2D Grounding、目标可供性标注、场景描述等多种子类型。

### Figure 3：机器人数据集构成

![Figure 3 - Robot Data](https://arxiv.org/html/2611.08577v1/robot_data_source.png)

**说明**: 15,060 小时机器人轨迹数据在 12 个机器人平台上的分布：RoboDesk 自建数据（3,235h，21.5%）、授权真实机器人数据（7,530h，50.0%）、开放仿真数据（4,295h，28.5%）。

### Figure 4：数据处理流水线

![Figure 4 - Data Pipeline](https://arxiv.org/html/2611.08577v1/data_pipeline_clean.png)

**说明**: 机器人轨迹清洗流水线。S1~S3 过滤阶段处理静止边界裁剪（Eq.1）、运动学一致性检查和视频缺陷去除；A1~A2 对齐阶段规范化动作表示（Eq.2-3）并完成层次任务标注。

### Figure 5：MMDiT 连续动作头

![Figure 5 - MMDiT](https://arxiv.org/html/2611.08577v1/mmdit.png)

**说明**: [[MMDiT]] 连续流匹配动作头的结构，对噪声动作块进行去噪，以 JiT clean-sample 参数化将预测转换为速度场，遮蔽损失处理多机器人异构维度。

### Figure 6：Rollback 数据收集

![Figure 6 - Rollback](https://arxiv.org/html/2611.08577v1/sections/figures/rollback_replaced.png)

**说明**: [[Rollback]] 数据收集示意图。锚点 $t_1$（操作员接管）和 $t_2$（恢复自主执行）定义了被替换的策略片段（$I=0$，负样本）和人类纠正轨迹（$I=1$，正样本）。灰色段为正常自主执行。

### Figure 7：异步动作块合并

> 图片暂无 arXiv HTML 外链，需本地获取。

**说明**: [[Asynchronous Inference|异步动作块合并]]示意图。新块到来时先按延迟 $d$ 裁剪失效命令（Eq.7），对重叠窗口 $L$ 步做线性混合（Eq.8），尾部直接拼接到残差缓冲区。

### Figure 8：事件驱动智能体

![Figure 8 - Agent Pipeline](https://arxiv.org/html/2611.08577v1/agent_pipeline.png)

**说明**: [[Event-Driven Agent]] 架构图。Monitor 生成字幕，Memory Manager 转化为持久事件并判断是否触发规划器，Planner 维护子任务序列，VLA 策略以当前子任务为条件高频执行。

### Figure 9：真实机器人实验任务

![Figure 9 - Real Tasks](https://arxiv.org/html/2611.08577v1/tasks.png)

**说明**: 三个真实机器人实验任务：(a) 字母积木摆放（7个子步骤，PS = k/7），(b) 衣物折叠（进度评分基于时间），(c) 以太网插线（3个子阶段，PS = m/3）。使用双臂 AgileX PiPER 机器人，每个任务 30 次试验。

### Figure 10：真实机器人 HG-DAgger 结果

> 图片暂无 arXiv HTML 外链，需本地获取。

**说明**: MR0 与 π0.5 在三个真实任务上经过 SFT 和 HG-DAgger 后训练的成功率（SR）和进度评分（PS）对比。MR0 在字母积木摆放和衣物折叠上与 π0.5 持平，以太网插线上优势更明显。

### Figure 11：Rollback 轮次对以太网插线的影响

> 图片暂无 arXiv HTML 外链，需本地获取。

**说明**: 展示随 Rollback 数据轮次累积，[[CFG-DAgger]] 和 [[RECAP]] 在以太网插线任务上的性能变化。每轮添加等量片段，第 0 轮为共享 SFT 基准。CFG-DAgger 最终达到 0.70 SR / 0.90 PS。

### Figure 12：HIL 数据累积效果

> 图片暂无 arXiv HTML 外链，需本地获取。

**说明**: 随 HIL 数据轮次增加，三个任务上成功率的变化曲线。各策略均从相同的 mid-training 检查点开始，起点为纯演示数据。

### Figure 13：智能体跨任务事件记忆实验

![Figure 13 - Agent Exp1](https://arxiv.org/html/2611.08577v1/agent_exp1_clean.png)

**说明**: 展示 [[Event-Driven Agent]] 跨任务状态记忆能力。钥匙在螺丝刀子任务期间被移动，系统将其记录为持久事件，后续放置任务中智能体通过检索历史事件定位钥匙。

### Figure 14：子任务等待与条件恢复

![Figure 14 - Agent Exp2](https://arxiv.org/html/2611.08577v1/agent_exp2.png)

**说明**: 展示智能体在临时不可行条件下的等待与恢复。钥匙被移除使放置子任务暂时不可行，智能体识别后进入等待状态；钥匙归位后，智能体自动恢复执行该子任务。

### Table 1：视觉语言监督消融（CALVIN ABC→D）

| 能力类型 | 训练配置 | Avg.Len | T1 | T2 | T3 | T4 | T5 |
|----------|----------|---------|-----|-----|-----|-----|-----|
| — | Baseline | 3.832 | 91.0% | 83.0% | 76.2% | 70.2% | 62.8% |
| 空间定位 | + 2D Grounding | 3.941 | 91.3% | 83.7% | 78.1% | 73.2% | 67.8% |
| 空间定位 | + 2D Pointing | 3.806 | 90.8% | 81.8% | 74.8% | 69.9% | 63.3% |
| 空间定位 | + 3D Grounding | 4.037 | 92.4% | 85.3% | 79.6% | 75.7% | 70.7% |
| 动作理解 | + 2D 目标可供性 | 4.102 | 93.3% | 87.9% | 82.5% | 76.4% | 70.1% |
| 动作理解 | + 3D 目标可供性 | 4.098 | 93.7% | 89.9% | 81.2% | 73.9% | 71.1% |
| 动作理解 | + 2D 轨迹预测 | 3.987 | 92.0% | 85.0% | 79.2% | 74.0% | 68.5% |
| 语义推理 | + 子任务预测 | 3.851 | 90.0% | 82.2% | 76.2% | 71.7% | 65.0% |
| 语义推理 | + 场景描述 | 4.082 | 93.1% | 87.1% | 81.4% | 76.2% | 70.4% |

**关键发现**: 2D 目标可供性标注对 Avg.Len 提升最大（+0.27），3D Grounding 也有显著帮助；子任务预测反而略有下降。

### Table 2：开源机器人数据集统计

| 数据来源 | 时长 | 占比 |
|----------|------|------|
| RoboDesk（真实自建），合计 | 3,235h | 21.5% |
| AgiBotWorld-Beta | 2,174h | 14.4% |
| Galaxea Open-World | 333h | 2.2% |
| RoboCOIN | 311h | 2.1% |
| LET | 172h | 1.1% |
| DROID | 164h | 1.1% |
| RoboMIND | 81h | 0.5% |
| 仿真数据，合计 | 4,295h | 28.5% |
| InternData-A1 | 3,967h | 26.3% |
| Genie Sim 3.0 | 213h | 1.4% |
| RoboTwin 2.0 | 115h | 0.8% |
| 开源合计 | 7,530h | 50% |

### Table 3：预训练数据混合配比

| 训练池 | 数据类别 | 规模 | 池内占比 |
|--------|----------|------|----------|
| VLA 动作池 | RoboDesk（开源真实） | 310.7M 帧 / 3.0kh | 21.5% |
| VLA 动作池 | 授权真实 | 746.3M 帧 / 6.9kh | 51.7% |
| VLA 动作池 | 合成仿真 | 387.5M 帧 / 3.6kh | 26.8% |
| VLA 动作池 | **合计** | 1,444.5M 帧 / ~13.6kh | 100% |
| VLM 协训池 | 真实机器人 Grounding | 66.7M 样本 | 44.2% |
| VLM 协训池 | 通用 VQA/描述 | 50.6M 样本 | 33.6% |
| VLM 协训池 | 空间推理 | 11.9M 样本 | 7.9% |
| VLM 协训池 | 真实机器人可供性 | 8.2M 样本 | 5.5% |
| VLM 协训池 | 仿真 VQA | 8.2M 样本 | 5.4% |
| VLM 协训池 | 任务规划/具身QA | 1.8M 样本 | 1.2% |
| VLM 协训池 | 真实机器人描述 | 1.7M 样本 | 1.1% |
| VLM 协训池 | 星知 VQA | 1.6M 样本 | 1.1% |
| VLM 协训池 | **合计** | 150.7M 有效样本，64 子数据集 | 100% |

### Table 4：RoboTwin 2.0 仿真基准（成功率 %）

| 方法 | Clean | Randomized | Overall |
|------|-------|-----------|---------|
| π0 | 65.9 | 58.4 | 62.2 |
| π0.5 | 82.7 | 76.8 | 79.8 |
| X-VLA | 72.8 | 72.8 | 72.8 |
| ABot-M0 | 86.1 | 85.1 | 85.6 |
| Qwen-VLA-Instruct | 86.1 | 87.2 | 86.7 |
| LingBot-VLA | 86.5 | 85.3 | 85.9 |
| Motus | 88.7 | 87.0 | 87.8 |
| StarVLA | 88.7 | 87.8 | 88.3 |
| InternVLA-A1 | 89.4 | 89.6 | 89.5 |
| Being-H0.7 | 90.2 | 89.6 | 89.9 |
| Fast-WAM | 91.9 | 91.8 | 91.8 |
| LingBot-VA | 92.9 | 91.5 | 92.2 |
| InternVLA-A1.5 | 93.3 | 93.0 | 93.2 |
| GalaxeaVLA (G0.5) | 93.7 | 92.8 | 93.3 |
| **Meituan-Robotics-0** | **93.2** | **93.9** | **93.6** |

**关键发现**: MR0 以 93.6% 总成功率达到最优，在随机化场景下（93.9%）略高于 Clean（93.2%），体现出强泛化性。

### Table 5：RoboDojo 比赛结果（Score/SR %）

| 模型 | 具身数据量 | 平均 | 泛化 | 精度 | 长时序 | 记忆 | 开放 |
|------|----------|------|------|------|--------|------|------|
| **WAM 类** | | | | | | | |
| Liber-0 Lite | — | 29.24/24.23 | 25.36/19.11 | 36.60/31.08 | 45.50/32.92 | 35.68/35.44 | 3.06/2.58 |
| InternW0-Δ | ~23.1kh | 30.77/23.91 | 30.09/22.78 | 31.98/23.25 | 46.29/29.33 | 34.67/34.00 | 10.84/10.17 |
| Awomo-0.5 | — | 35.34/29.64 | 30.31/23.84 | 39.65/33.50 | 40.36/26.75 | 42.27/41.22 | 24.09/22.92 |
| VPP2 | — | 39.26/32.26 | 36.38/29.50 | 43.03/35.17 | 56.28/37.00 | 56.01/55.56 | 4.61/4.08 |
| **VLA 类** | | | | | | | |
| SimpleMemVLA | — | 12.58/9.27 | 6.36/3.95 | 7.42/2.92 | 14.58/5.50 | 33.71/33.22 | 0.85/0.75 |
| KinRT | — | 13.02/8.80 | 14.02/8.61 | 15.65/9.92 | 26.40/18.08 | 4.82/3.56 | 4.23/3.83 |
| Hy-Embodied-0.5-VLA | >10kh | 13.07/8.80 | 11.78/8.39 | 13.81/8.00 | 25.74/14.92 | 13.37/12.11 | 0.65/0.58 |
| Xiaomi-Robotics-1 | >100kh | 20.07/13.93 | 23.54/17.00 | 26.69/18.83 | 38.39/23.67 | 7.81/6.56 | 3.94/3.58 |
| **Meituan-Robotics-0** | ~13.6kh | **20.15/14.63** | 17.43/12.08 | 23.68/15.00 | 27.01/14.38 | 29.03/28.33 | 3.59/3.38 |
| GalaxeaVLA (G0.5) | — | 20.23/14.88 | 18.46/12.83 | 28.25/20.42 | 44.12/32.25 | 8.61/7.33 | 1.73/1.58 |
| DM0.5 | — | 24.90/19.34 | 15.77/10.95 | 24.82/16.75 | 33.70/19.50 | 47.74/47.44 | 2.43/2.08 |
| Rex-M1 Preview | >100kh | 33.79/27.44 | 34.05/26.05 | 32.82/23.83 | 46.89/33.00 | 48.53/48.22 | 6.67/6.08 |
| Simate-beta | — | 33.95/27.96 | 35.09/27.95 | 34.35/26.92 | 57.84/43.42 | 33.33/33.00 | 9.12/8.50 |

**关键发现**: MR0 在 VLA 类中排名第 5（以 Avg. SR 计），以约 13.6kh 训练数据与 Xiaomi-Robotics-1（>100kh）相当，记忆任务得分（29.03）在 VLA 类中领先，体现事件驱动智能体的优势。

---

## 实验

### 数据集

| 数据集/基准 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| RoboDesk | 3,235h | 自建真实机器人多任务数据 | 训练 |
| 授权真实数据 | 7,530h | 多机构合作真实轨迹 | 训练 |
| InternData-A1 仿真 | 3,967h | IsaacSim 高质量仿真 | 训练 |
| RoboTwin 2.0 | 115h | 双臂桌面操作基准（仿真） | 训练+评估 |
| RoboDojo | — | 多维度 VLA 比赛基准 | 评估 |
| CALVIN ABC→D | — | 语言条件连续操作消融基准 | 消融 |

### 实现细节

- **Backbone**: Qwen 3.5-4B 预训练语言视觉模型
- **预训练**: 256 GPU，500k 步，[[FAST]] tokenizer 离散动作头
- **中间训练**: 128 GPU，[[MMDiT]] 连续流匹配头，动作维度 20D
- **后训练**: SFT → HG-DAgger / RECAP / CFG-DAgger 三选一
- **动作块大小**: $H=50$，推理频率约 3 Hz，控制频率 30 Hz
- **硬件（真实）**: 双臂 AgileX PiPER 机器人，30 次/任务

### 可视化结果

- **以太网插线**：CFG-DAgger 最终达 70% SR / 0.90 PS，高于 RECAP（60% SR / 0.67 PS）
- **智能体实验**：事件记忆支持跨子任务状态传递（钥匙位置预测），子任务等待机制在条件恢复后自动续接

---

## 批判性思考

### 优点

1. **完整闭环系统**：从数据（Rollback 收集）到模型（三阶段训练）到部署（异步执行 + 事件智能体）形成完整解决方案，工程成熟度高。
2. **Rollback 创新性**：同时产生正负样本，无需额外标注，自然地为 CFG-DAgger 和 RECAP 提供训练信号，低成本高信息量。
3. **跨机器人规范化**：统一末端执行器接口使 12 个平台数据无缝整合，是少数真正实现大规模跨机器人训练的工作。

### 局限性

1. **基座限于固定底座**：论文明确限定"stationary-base desktop manipulation"，不涵盖移动操作或全身控制，限制了适用范围。
2. **Rollback 依赖人工判断时机**：操作员需主观判断"何时回退"，这个决策的一致性和效率未有定量分析。
3. **RoboDojo 排名中等**：在 VLA 类第 5，整体第 9，远低于 WAM 类方法（如 VPP2），说明 VLA 范式在开放类任务上与 WAM 仍有差距。

### 潜在改进方向

1. 扩展到移动底座（人形或移动操作平台），需处理更复杂的全身动力学；
2. 自动化 Rollback 触发（基于不确定性估计或价值函数预警），减少对操作员主观判断的依赖；
3. 加强 VLA 与智能体的双向反馈，当前智能体调用 VLA 策略是单向的，执行反馈未传回规划器。

### 可复现性评估

- [x] 代码开源（GitHub 链接已提供）
- [ ] 预训练模型（未公开权重）
- [x] 训练细节完整（三阶段配置、超参数等详细）
- [ ] 数据集完全可获取（授权真实数据和部分仿真数据不公开）

---

## 关联笔记

### 基于

- [[FAST]]: 离散动作 token 预训练所用的 tokenizer 方法
- [[Flow-Matching]]: 中间训练和推理的连续动作生成基础
- [[DAgger]]: 在线模仿学习框架，HG-DAgger 和 CFG-DAgger 的基础
- [[Qwen]]: 骨干语言视觉模型

### 对比

- [[pi0.5]]: 真实机器人实验的主要对比基线（相同 HG-DAgger 后训练设置）
- [[ABot-M0]]: RoboTwin 2.0 上的直接竞争对手（85.6%）
- [[Fast-WAM]]: RoboTwin 2.0 上相近的强基线（91.8%）

### 方法相关

- [[Rollback]]: MR0 提出的核心数据收集方法，产生正负样本对
- [[CFG-DAgger]]: 利用 Rollback 数据的正负分支 CFG 后训练方法
- [[RECAP]]: 基于分布式价值学习的优势条件后训练方法
- [[HG-DAgger]]: 人工门控纠正的在线模仿学习基线
- [[MMDiT]]: 连续流匹配动作头使用的多模态扩散变换器架构
- [[Classifier-Free Guidance (CFG)]]: CFG-DAgger 和 RECAP 在动作空间中应用的引导技术
- [[Asynchronous Inference]]: 3Hz 推理 + 30Hz 控制的异步执行方案
- [[Event-Driven Agent]]: Monitor-Memory-Planner 持久状态追踪系统
- [[Cross-Embodiment]]: 统一末端执行器接口规范化 12 个平台数据
- [[Action Chunking]]: MR0 以 H=50 步动作块预测
- [[Set-of-Mark Prompting]]: 用于生成 2D Grounding 和可供性数据的标注方法

### 硬件/数据相关

- [[AgileX PiPER]]: 真实实验使用的双臂机器人平台
- [[RoboTwin 2.0]]: 仿真基准测试环境（也提供训练数据）

---

## 速查卡片

> [!summary] Meituan-Robotics-0 (MR0)
> - **核心**: 桌面操作 VLA 基础模型，Rollback 收集正负样本 + CFG-DAgger 后训练 + 事件驱动智能体
> - **方法**: 离散预训练（FAST）→ 连续中间训练（MMDiT 流匹配）→ 真实世界后训练（Rollback/CFG-DAgger）
> - **结果**: RoboTwin 2.0 93.6%（SOTA），以太网插线 70% SR / 0.90 PS
> - **代码**: [github.com/Meituan-Robotics/Meituan-Robotics-0](https://github.com/Meituan-Robotics/Meituan-Robotics-0)

---

*笔记创建时间: 2026-11-10*

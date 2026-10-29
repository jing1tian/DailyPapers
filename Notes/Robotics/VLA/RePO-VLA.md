---
title: "RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models"
method_name: "RePO-VLA"
authors: [Weijia Liufu, Xiaoyu Guo, Ruiyi Chen, Jingzhi Liu, Kaidong Zhang, Xiwen Liang, Jianqi Lin, Dawei Sun, Yuze Wang, Rongtao Xu, Bingqian Lin, Bowen Yang, Tongtong Cao, Bowen Peng, Dongyu Zhang, Guangrun Wang, Min Wang, Liang Lin, Xiaodan Liang]
year: 2026
venue: NeurIPS 2026
tags: [failure-recovery, vla, imitation-learning, value-function, bimanual-manipulation, policy-optimization, benchmark]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.09410v2
created: 2026-10-29
---

# 论文笔记：RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Sun Yat-sen University 及合作机构 |
| 日期 | May 2026 (NeurIPS 2026) |
| 项目主页 | [RePO-VLA-Web](https://1018weijia.github.io/RePO-VLA-Web/) |
| 对比基线 | [[π₀]]、[[π₀.₅]]、[[SARM]]、[[ConRFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.09410) |

---

## 一句话总结

> 通过对成功、恢复和失败轨迹分配不同监督角色，RePO-VLA 使 [[VLA（视觉-语言-动作模型）]] 在对抗扰动下的平均成功率从 20% 提升至 75%，无需在线故障检测器或重试逻辑。

---

## 核心贡献

1. **[[Recovery-Aware Initialization]] (RAI)**: 将纠正动作序列切分为独立 episode（TSER），消除失败前缀对恢复动作的因果混淆，在 [[Behavior Cloning]] 基础上加入 λ 加权的恢复数据监督。
2. **[[Progress-Aware Semantic Value Function]] (PAS-VF)**: 利用冻结的 [[VJEPA-2]] 视觉编码器和 CLIP 文本编码器，通过余弦相似度对轨迹进度打分，为失败轨迹的有用片段赋予非零值。
3. **[[Value-Conditioned Refinement]] (VCR)**: 为 [[π₀.₅]] 的 [[Flow-Matching]] 策略添加值 token，训练时以逐帧值标签为条件，推理时固定 v = 1.0 引导动作朝成功流形靠拢。
4. **[[FRBench]]**: 标准化的失败-恢复基准，包含 RoboTwin 仿真（46 个任务）和真实双臂操作（4 个任务），定义四种误差注入类型（E1–E4）。

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 在长程、接触丰富的[[Long-horizon Manipulation|长程操作]]任务中极为脆弱：执行偏差（execution drift）累积后几乎不可纠正，现有方法缺乏对失败状态的应对能力。

### 现有方法的局限

- **成功-only 的[[模仿学习]]（SFT）**: 无法提供偏离信号；失败轨迹通常被直接丢弃。
- **VLM 引导的重规划**（[[RACER]]、RoboFAC）: 依赖外部检测器，部署开销大。
- **交互式纠正**（ThriftyDAgger）: 需要人在回路，难以扩展。
- **原始 [[DAgger]] 式聚合**: 将纠正动作拼接在失败前缀后，造成因果混淆——模型误将失败状态与后续恢复动作建立直接关联。

### 本文的动机

作者指出 SFT-based VLA 训练存在"失败利用盲点"（failure-utilization blind spot）：失败轨迹含有丰富的语义恢复信号，但因果结构混乱导致直接使用有害。关键洞察是：**将纠正片段切分为独立 episode，并以密集进度值标签区分有效与无效片段**，可在不修改预训练 backbone 的前提下显著提升鲁棒性。

---

## 方法详解

### 模型架构

RePO-VLA 采用**两阶段微调**框架，以 [[π₀.₅]] 的 [[Flow-Matching]] 策略为 backbone：

- **输入**: 语言指令 $l$ + 视觉观测 $o_t$ + 值 token $e_{val}(v_t)$
- **Backbone**: [[π₀.₅]] (flow-matching policy)
- **核心模块**: [[Recovery-Aware Initialization]] + [[Progress-Aware Semantic Value Function]] + [[Value-Conditioned Refinement]]
- **输出**: 动作序列 $a_t$
- **推理时**: 固定 $v = 1.0$，无在线故障检测器

### 核心模块

#### 模块 1: Recovery-Aware Initialization (RAI)

**设计动机**: 消除原始 [[DAgger]] 式聚合中的因果混淆，利用 [[TSER]] 让纠正动作序列从 $t = 0$ 重新开始。

**具体实现**:
- **TSER（Trajectory Slicing with Episode Reset）**: 丢弃失败前缀，将纠正后缀以新 episode 的形式从 $t = 0$ 开始，使纠正动作与当前不利状态（adverse state）直接绑定。
- 使用专家演示数据 $\mathcal{D}_{expert}$ 和重置后的恢复数据 $\mathcal{D}_{rec}^{reset}$ 联合训练，通过 $\lambda$ 平衡两者。
- 恢复数据来源：基于规划器的控制拦截式误差注入（仿真）和专家对策略失败的现场干预（真实世界远程操作）。

#### 模块 2: Progress-Aware Semantic Value Function (PAS-VF)

**设计动机**: 为混合质量轨迹的每一帧赋予密集进度分，区分"失败但有进度"和"纯失败"的帧。

**具体实现**:
- **视觉编码器**: 冻结的 [[VJEPA-2]] 视频编码器 + 轻量可训练 adapter，将轨迹前缀投影到共享空间得到 $z_t^v$。
- **语言编码器**: 冻结的 CLIP 文本编码器 + 可训练 adapter，将语言指令投影到共享空间得到 $z^l$。
- **对齐训练**: 在成功轨迹上，要求 $\cos(z_t^v, z^l)$ 逼近归一化进度 $t/T$。
- **自参照进度（Self-referential progress）**: 对于失败轨迹，其值为与成功参考聚类 $\mathcal{C}_{ref}$ 中最高的[[Cosine Similarity|余弦相似度]]。

#### 模块 3: Value-Conditioned Refinement (VCR)

**设计动机**: 将进度值信号注入策略网络，让其在推理时通过固定高值 token 向成功流形靠拢。

**具体实现**:
- **Hindsight 标签分配**:
  - 成功帧和有效恢复帧: $v = 1.0$
  - 偏离的恢复前缀: $v = 0.0$
  - 纯失败帧: 可靠性衰减，$v_t = V(\tau_{fail}) \cdot (1 - t/T)^\alpha$，$\alpha = 3.0$
- **值 token**: $\text{MLP}(v_t)$ 嵌入插入 [[π₀.₅]] 的输入中。
- **推理**: 固定 $v = 1.0$，无需在线失败检测或重试逻辑。

---

## 关键公式

### 公式 1: [[Recovery-Aware Initialization|RAI 联合训练损失]] (Eq. 1)

$$
\mathcal{L}_{SFT} = -\mathbb{E}_{\tau \sim \mathcal{D}_{expert}} \sum_t \log \pi(a_t|o_t) - \lambda \, \mathbb{E}_{\tau' \sim \mathcal{D}_{rec}^{reset}} \sum_t \log \pi(a_t|o_t)
$$

**含义**: 以专家演示为主、重置恢复数据为辅的联合[[Behavior Cloning|行为克隆]]损失，消除失败前缀的因果干扰。

**符号说明**:
- $\mathcal{D}_{expert}$: 专家成功演示数据集
- $\mathcal{D}_{rec}^{reset}$: TSER 切分后的重置恢复数据集
- $\lambda$: 平衡两类数据的权重系数
- $\pi(a_t|o_t)$: 策略在观测 $o_t$ 下输出动作 $a_t$ 的概率

---

### 公式 2: [[Progress-Aware Semantic Value Function|进度对齐损失]] (Eq. 2)

$$
\mathcal{L}_{align} = \mathbb{E}_{\tau \in \mathcal{T}_{succ}} \left[ \frac{1}{T_\tau} \sum_t \left( \cos(z_t^v,\, z^l) - \frac{t}{T_\tau} \right)^2 \right]
$$

**含义**: 在成功轨迹上训练 adapter，使视觉-语言[[Cosine Similarity|余弦相似度]]精确追踪归一化任务进度 $t/T$。

**符号说明**:
- $z_t^v$: 时刻 $t$ 的视觉轨迹前缀嵌入（[[VJEPA-2]] + adapter 输出）
- $z^l$: 语言指令嵌入（CLIP + adapter 输出）
- $T_\tau$: 轨迹 $\tau$ 的总时间步长
- $t/T_\tau$: 归一化的任务进度标签

---

### 公式 3: [[Progress-Aware Semantic Value Function|失败轨迹自参照进度]] (Eq. 3)

$$
V(\tau_{fail}) = \max_{z \in \mathcal{C}_{ref}} \cos\!\left(z_{\tau_{fail}}^v,\; z\right)
$$

**含义**: 失败轨迹的整体值定义为其视觉嵌入与成功参考聚类中最相似成员的[[Cosine Similarity|余弦相似度]]，量化"到达了多远"。

**符号说明**:
- $z_{\tau_{fail}}^v$: 失败轨迹的视觉嵌入
- $\mathcal{C}_{ref}$: 成功轨迹参考嵌入聚类
- $V(\tau_{fail}) \in [0, 1]$: 失败轨迹的进度值

---

### 公式 4: [[Value-Conditioned Refinement|失败帧可靠性衰减标签]] (Eq. 4)

$$
v_t = V(\tau_{fail}) \cdot \left(1 - \frac{t}{T}\right)^\alpha, \quad \alpha = 3.0
$$

**含义**: 对纯失败帧按时间指数衰减分配值标签——早期有用动作保留进度值，末期失败状态快速降至 0，避免模型学习失败行为。

**符号说明**:
- $V(\tau_{fail})$: 由 Eq. 3 计算的失败轨迹整体进度值
- $\alpha = 3.0$: 衰减速率超参数（消融中 $\alpha=3$ 最优，$\alpha=1,10$ 均导致性能下降）
- $t/T$: 归一化时间步

---

### 公式 5: [[Value-Conditioned Refinement|VCR 条件对数似然损失]] (Eq. 5)

$$
\mathcal{L}_{VCR} = -\mathbb{E}_{\tau \sim \mathcal{D}_{total}} \left[ \sum_t \log \pi_\theta\!\left(a_t \mid o_t,\, l,\, e_{val}(v_t)\right) \right]
$$

**含义**: 以值 token 为条件的策略微调损失，让策略学会在高值条件下输出高质量动作，低值条件下规避失败行为。

**符号说明**:
- $\mathcal{D}_{total}$: 包含成功、恢复和失败轨迹的完整混合数据集
- $e_{val}(v_t) = \text{MLP}(v_t)$: 将标量值 $v_t$ 映射为值 token 嵌入
- $\pi_\theta$: 带值 token 输入的 [[π₀.₅]] 策略网络

---

### 公式 6: [[FRBench|仿真失败触发条件]] (Eq. 6，附录 A.4.4)

$$
\mathcal{F}_t = \mathbb{1}(t > T_{max})
$$

**含义**: 当执行时间超过所有任务名义成功时间的最大值 $T_{max}$ 时，判定为失败并触发 FRBench 中的误差注入阶段。

**符号说明**:
- $T_{max}$: 所有任务名义成功 rollout 中最大时长（按任务设定，避免误触发）
- $\mathcal{F}_t \in \{0,1\}$: 失败触发标志

---

## 关键图表

### Figure 1: 恢复驱动的轨迹利用策略对比

![Figure 1](https://arxiv.org/html/2605.09410v2/Intro.png)

**说明**: 先前 VLA 仅模仿成功演示（蓝色）。RePO-VLA 对三类轨迹分别分配监督角色：成功轨迹用于 BC、恢复轨迹经 TSER 切分后独立训练、失败轨迹通过 PAS-VF 进度值标注后部分利用。

---

### Figure 2: RePO-VLA 整体框架

![Figure 2](https://arxiv.org/html/2605.09410v2/f_final_main.png)

**说明**: 两阶段管线。阶段一（RAI）：构建恢复数据集（TSER + 数据引擎），混合训练。阶段二（VCR）：PAS-VF 为混合轨迹打分，附加值 token 微调。推理时固定 $v = 1.0$。

---

### Figure 3: 语义进度对齐（PAS-VF 架构）

![Figure 3](https://arxiv.org/html/2605.09410v2/value_function_arch.png)

**说明**: 冻结 [[VJEPA-2]] 和 CLIP 编码器，各添加可训练 adapter 投影到共享空间。通过 Eq. 2 的对齐损失，使余弦相似度成为密集进度信号。

---

### Figure 4a: 失败-恢复 episode 的值景观

![Figure 4a](https://arxiv.org/html/2605.09410v2/value_function_1.png)

**说明**: 失败-恢复 episode 的进度感知值景观可视化。失败阶段（低进度）值低，纠正恢复阶段后值回升，体现 PAS-VF 的进度建模能力。

---

### Figure 4b: 纯失败 episode 的值景观

![Figure 4b](https://arxiv.org/html/2605.09410v2/value_function_2.png)

**说明**: 纯失败 episode 的值景观。可靠性衰减（Eq. 4）使早期有价值的探索动作保留非零值，末期快速衰减至 0。

---

### Figure 5: 失败-恢复数据引擎

![Figure 5](https://arxiv.org/html/2605.09410v2/Data_engine.png)

**说明**: 两路数据生成：(1) 控制拦截式误差注入——在策略执行名义任务时进行物理扰动（E1–E4），由规划器驱动恢复；(2) 策略引导式 rollout——让学习中的策略自然失败，专家远程操作进行干预恢复。

---

### Figure 6: 消融实验面板（3 组）

> **注**: arXiv HTML 中未提供此图的独立图片 URL，以下为文字描述。

**说明**:
- **(a)** 原始数据聚合（Raw DAgger-style）性能低于 Phase I，验证 TSER 的必要性。
- **(b)** Fold Towel 任务：高值 token ($v=1.0$) 将成功率从 46.7% 提升至 73.3%，$v=0$ 则显著下降。
- **(c)** $\alpha$ 消融：$\alpha=1$（40%）、$\alpha=3$（70%）、$\alpha=10$（40%），$\alpha=3$ 最优，过大/过小的衰减均有害。

---

### Figure 7–14: 附录——仿真误差类型可视化

![Figure 7 — Premature Close, Clean](https://arxiv.org/html/2605.09410v2/appendix_1.png)

**说明**: E1（过早关夹爪）在 Clean 环境下的 9 帧关键帧示意：帧 1–3 为误差注入，帧 4–8 为规划器恢复，帧 9 恢复正常执行。

![Figure 8 — Premature Close, Random](https://arxiv.org/html/2605.09410v2/appendix_2.png)

**说明**: E1 在域随机化（Randomized）环境下的可视化，背景纹理和光照随机化。

![Figure 9 — Grasp Slip, Clean](https://arxiv.org/html/2605.09410v2/appendix_3.png)

**说明**: E2（抓取滑落）Clean 环境可视化。

![Figure 10 — Grasp Slip, Random](https://arxiv.org/html/2605.09410v2/appendix_4.png)

**说明**: E2 Random 环境可视化。

![Figure 11 — Position Offset, Clean](https://arxiv.org/html/2605.09410v2/appendix_5.png)

**说明**: E3（抓取位置偏移）Clean 环境可视化。

![Figure 12 — Position Offset, Random](https://arxiv.org/html/2605.09410v2/appendix_6.png)

**说明**: E3 Random 环境可视化。

![Figure 13 — Orientation Mismatch, Clean](https://arxiv.org/html/2605.09410v2/appendix_7.png)

**说明**: E4（抓取朝向不匹配）Clean 环境可视化。

![Figure 14 — Orientation Mismatch, Random](https://arxiv.org/html/2605.09410v2/appendix_8.png)

**说明**: E4 Random 环境可视化。

---

### Figure 15: 真实世界定性评估

![Figure 15](https://arxiv.org/html/2605.09410v2/Real_world_data.png)

**说明**: 四个接触丰富的双臂任务（Pour Water、Cook Vegetable、Tidy Desk、Fold Towel）在强制偏差（外部推力或物体滑落）下的 RePO-VLA 恢复轨迹可视化。基线方法通常对扰动无反应，导致任务失败。

---

### Table 1: FRBench 三阶段评估协议

| Phase | 状态 $\phi_t$ | 控制/干预方式 | 隔离测量对象 |
|-------|--------------|-------------|------------|
| Nominal | $\phi_t = \text{Nominal}$ | 策略在干净或随机化条件下运行 | 基础到达、抓取和任务进度能力 |
| Error | $\phi_t = \text{Error}$ | E1–E4 结构化扰动将场景推入不利状态 | 可重复的物理故障，独立评分 |
| Recovery | $\phi_t = \text{Recovery}$ | 策略从不利状态继续，无外部重试逻辑 | 恢复任务进度的纠正动作能力 |

**说明**: 评估协议将名义任务能力与受控物理故障后的纠正行为解耦，确保对比公平。

---

### Table 2: FRBench-Sim 仿真结果（10 个 RoboTwin 任务均值）

| 方法 | Nominal Clean (%) | Nominal Random (%) | Adversarial Clean (%) | Adversarial Random (%) |
|------|------------------|-------------------|----------------------|----------------------|
| [[π₀]] | 33.9 | 12.9 | — | — |
| [[π₀.₅]] | — | — | 15.0 | 15.4 |
| Phase I (RAI only) | 44.6 | 44.0 | — | — |
| **RePO-VLA Full** | — | — | **37.0** | **43.0** |

**说明**: RePO-VLA 在对抗条件下的恢复成功率约为 [[π₀.₅]] 的 2.5–2.8 倍。RAI 单独使用（Phase I）在名义条件下也大幅超越 baseline。

---

### Table 3: FRBench-Real 真实世界结果（4 个双臂任务，各 10 次盲测）

| 方法 | Standard avg (%) | Adversarial avg (%) |
|------|-----------------|-------------------|
| [[π₀]] | 27.5 | 12.5 |
| [[π₀.₅]] | 25.0 | 20.0 |
| Phase I (RAI only) | 42.5 | 37.5 |
| RePO-VLA Full (1×) | 40.0 | 30.0 |

**说明**: 1× 恢复数据量下 Full 并未稳定超越 Phase I；作者归因于 1× 时失败数据稀疏。随数据量增加，Full 方法显著领先（见 Table 4）。

---

### Table 4: 恢复数据量扩展实验（Pour Water + Fold Towel 均值）

| 方法 | Standard avg (%) | Adversarial avg (%) |
|------|-----------------|-------------------|
| [[π₀.₅]] | 25.0 | 20.0 |
| RePO-VLA Full (2×) | 70 | 65 |
| **RePO-VLA Full (4×)** | **80** | **75** |

**说明**: 4× 数据量下 RePO-VLA 超越 [[π₀.₅]] 基线 55 个百分点（对抗条件），体现出显著的数据扩展趋势。

---

### Table 5: 消融实验——Fold Towel 任务控制组

| 配置 | Standard (%) | Adversarial (%) | 说明 |
|------|-------------|----------------|------|
| Success-only BC | 40 | 20 | 无恢复数据监督 |
| Raw [[DAgger]]-style aggregation | 0 | — | 直接聚合失败轨迹，因果混淆导致崩溃 |
| Phase I + DAgger-style | 40 | 20 | TSER 缺失时 RAI 退化为 DAgger |
| Full-4× without value token | 20 | 20 | 移除值 token 后性能下降 |
| [[SARM]] + RA-BC | 50 | 20 | 强基线，但 adversarial 无提升 |
| **RePO-VLA Full-4×** | **70** | **70** | 完整方法 |

**关键发现**: TSER 和值 token 各自独立贡献显著，缺少任一组件均导致明显性能下降。

---

### Table 6: 离线方法对比（[[ConRFT]]/RECAP vs. RePO-VLA）

| 方法 | Standard avg (%) | Adversarial avg (%) |
|------|-----------------|-------------------|
| RECAP-2× | 75 | 60 |
| **RePO-VLA Full-2×** | **70** | **65** |

**说明**: 两者在标准/对抗条件下各有优势，作者指出无头对头直接比较，不做优劣断言。

---

### Table 7: 缩略语与符号参考（附录 A.1）

| 术语 | 全称 | 含义 |
|------|------|------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略族 |
| RePO-VLA | Recovery-driven Policy Optimization for VLA | 本文训练框架 |
| [[FRBench]] | Failure-Recovery Benchmark | 失败-恢复基准与数据引擎 |
| [[Recovery-Aware Initialization\|RAI]] | Recovery-Aware Initialization | 第一训练阶段，使用专家演示和切分纠正 episode |
| [[TSER]] | Trajectory Slicing with Episode Reset | 丢弃失败前缀，从 t=0 重新开始纠正 |
| [[Progress-Aware Semantic Value Function\|PAS-VF]] | Progress-Aware Semantic Value Function | 基于成功轨迹对齐的密集进度估计器 |
| [[Value-Conditioned Refinement\|VCR]] | Value-Conditioned Refinement | 第二训练阶段，以值标签为条件微调策略 |
| SFT | Supervised Fine-Tuning | 基于动作标注演示的模仿式微调 |
| RLFT | Reinforcement Fine-Tuning | 基于奖励或偏好反馈的策略微调 |
| [[VJEPA-2\|V-JEPA]] | Video Joint-Embedding Predictive Architecture | PAS-VF 使用的冻结时空编码器 |
| E1–E4 | Error types 1–4 | FRBench 四种扰动类型 |

---

### Table 8: FRBench-Sim 数据集统计

| 类别 | 数量 |
|------|------|
| 总 episode 数 | 23,453 |
| 名义成功 episode | 17,061 |
| 验证失败-恢复 episode | 6,392 |
| 任务数 | 46 |
| E1 尝试（过滤前） | 8,022 |
| E2 尝试（过滤前） | 3,516 |
| E3 尝试（过滤前） | 4,686 |
| E4 尝试（过滤前） | 688 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| FRBench-Sim（RoboTwin） | 23,453 episodes, 46 tasks | 仿真双臂，含 4 类误差注入，支持 Clean/Randomized | 仿真评估 |
| FRBench-Real | 4 tasks, 200 demos + 50 recovery eps/task | Dobot X-Trainer 双臂，远程操作恢复数据 | 真实世界评估 |

### 实现细节

- **Backbone**: [[π₀.₅]] (flow-matching VLA)
- **视觉编码器**: 冻结 [[VJEPA-2]] + 轻量 adapter
- **语言编码器**: 冻结 CLIP + 轻量 adapter
- **硬件**: 2× A100 80GB，每次完整训练约 30 小时
- **仿真评估**: 每个任务 50 次 rollout，Clean 和 Randomized 两种设置
- **真实评估**: 每个条件 10 次盲测（blind trial），4 个任务
- **误差注入时机**: 基于语义阶段（semantic phase）触发，而非 wall-clock 时间

### 误差类型定义 (E1–E4)

| 类型 | 名称 | 物理实现 |
|------|------|---------|
| E1 | Premature Close | 过早触发夹爪关闭 |
| E2 | Grasp Slip | 夹爪打开模拟物体滑落 |
| E3 | Position Offset | 控制覆写施加位置偏移 |
| E4 | Orientation Mismatch | 控制覆写施加朝向偏转 |

---

## 批判性思考

### 优点
1. **无需在线检测器**: 推理时固定 $v=1.0$，零额外延迟，部署简洁。
2. **数据扩展性强**: 随恢复数据量 1×→4× 性能持续提升，体现系统性的扩展规律。
3. **模块化微调**: 不修改预训练 backbone，可直接应用于其他 VLA（如 [[π₀.₅]]）。
4. **TSER 的简洁性**: 仅通过切片操作消除因果混淆，无额外网络结构。

### 局限性
1. **恢复泛化有限**: 仅能恢复训练中见过的失败类型；严重遮挡、工具误用、子目标不匹配等未评估。
2. **远程操作瓶颈**: 真实世界恢复数据依赖人类远程操作，扩展成本高。
3. **失败预训练未探索**: 在预训练阶段建模失败（而非仅微调时）是未来方向。
4. **表格数据不完整**: Table 2 中部分 method × condition 组合的原始数值未在论文中明确给出。

### 潜在改进方向
1. 将失败建模引入 VLA 预训练阶段，提升对未见失败类型的零样本恢复能力。
2. 用自动化规划器/仿真生成更大规模的恢复数据，减少对人工操作的依赖。
3. 探索 PAS-VF 在 online RL 微调中的应用，与 [[ConRFT]] 等方法结合。

### 可复现性评估
- [ ] 代码开源（项目页已上线但未见代码仓库）
- [ ] 预训练模型
- [x] 训练细节完整（附录 A.2 提供了超参数、计算量细节）
- [x] FRBench 数据集统计完整（Table 8）

---

## 关联笔记

### 基于
- [[π₀.₅]]: 使用 π0.5 的 flow-matching backbone 作为策略网络
- [[VJEPA-2]]: PAS-VF 使用其冻结视频编码器

### 对比
- [[π₀]]: 直接 baseline，名义任务下 Random 设置性能大幅下降
- [[π₀.₅]]: 主要对比基线，RePO-VLA 基于其微调
- [[SARM]]: 同类恢复方法，消融实验中直接对比
- [[ConRFT]]: RL 微调类恢复方法，Table 6 间接对比
- [[RACER]]: VLM 重规划类方法，相关工作对比

### 方法相关
- [[Recovery-Aware Initialization]]: 核心方法 (RAI)
- [[Progress-Aware Semantic Value Function]]: 核心方法 (PAS-VF)
- [[Value-Conditioned Refinement]]: 核心方法 (VCR)
- [[TSER]]: RAI 的轨迹预处理技术
- [[Behavior Cloning]]: RAI 阶段的基础损失函数
- [[Flow-Matching]]: π0.5 使用的生成模型框架
- [[Value Function]]: VCR 的理论基础
- [[DAgger]]: 对比的聚合方法，证明 TSER 优于直接 DAgger

### 硬件/数据相关
- [[FRBench]]: 本文提出的失败-恢复基准
- [[RoboTwin 2.0]]: FRBench-Sim 使用的仿真环境基础

---

## 速查卡片

> [!summary] RePO-VLA (NeurIPS 2026)
> - **核心**: 为 VLA 分配成功/恢复/失败三类轨迹监督角色，解决失败利用盲点
> - **方法**: RAI (TSER 切片) + PAS-VF (进度感知值函数) + VCR (值条件微调)
> - **结果**: 对抗成功率 20%→75%（仿真），最高 80%（4×真实数据）
> - **代码**: [https://1018weijia.github.io/RePO-VLA-Web/](https://1018weijia.github.io/RePO-VLA-Web/)

---

*笔记创建时间: 2026-10-29*

---
title: "Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization"
method_name: "Zero-WAM"
authors: [Jiaming Zhou, Qihang Zhang, Gangwei Xu, Cunxin Fan, Yujie Zhao, Ruilin Wang, Yiming Luo, Shuai Yang, Xing Zhu, Yujun Shen, Junwei Liang, Yinghao Xu]
year: 2026
venue: arXiv
tags: [world-action-model, video-generation, robot-manipulation, in-context-learning, zero-shot-generalization, human-robot-transfer, flow-matching]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.26103v2
created: 2026-08-28
---

# 论文笔记：Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（来自 arXiv 提交） |
| 日期 | August 2026 |
| 项目主页 | [robbyant-research.github.io/Zero-WAM](https://robbyant-research.github.io/Zero-WAM/) |
| 对比基线 | [[LingBot-VA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.26103) / Code 未公开 |

---

## 一句话总结

> Zero-WAM 以人类视频为任务示范，通过因果视频-动作联合建模和自动化数据生成流水线，实现机器人对未见操作任务的零样本泛化。

---

## 核心贡献

1. **In-Context World-Action Modeling 范式**: 将零样本任务泛化形式化为上下文世界-动作建模，单一因果策略同时接受语言指令和人类示范视频，无需参数更新即可适应新任务。
2. **HumanGen 数据集**: 自动化生成的 74.2K 人机配对 [[In-Context Learning|ICL]] 样本，覆盖 8.6K 个任务，通过 VLM 语义分析 + 图像编辑 + 视频生成流水线将机器人轨迹转换为人类视频。
3. **In-Context Future Chunk Prediction（IFP）训练目标**: 辅助监督多个时间跨步的未来视频块，强制模型从人类视频中提取任务语义而非依赖机器人历史，提升人类视频提示的有效利用。

---

## 问题背景

### 要解决的问题

现有机器人策略在部署到未见任务时需要额外收集标注数据或重新训练，无法像人类那样通过观察一次示范就能执行新任务。如何让机器人在**不更新参数**的情况下，从一段人类视频示范推断出可执行的操作动作？

### 现有方法的局限

- 已有人机配对数据集（MIME、EgoMimic、BC-Z、RH20T）依赖**人工采集**，规模小（任务类别 <1K），难以覆盖开放世界的任务多样性。
- 现有 [[VLA|视觉-语言-动作模型]] 以语言指令为主要任务说明，语言描述对于精细操作的任务步骤刻画能力有限。
- 已有 [[In-Context Imitation Learning|上下文模仿学习]] 方法通常需要同形态的机器人示范，难以跨形态泛化。

### 本文的动机

互联网上有海量的人类操作视频，但由于形态差异（Embodiment Gap），其动作信号难以直接迁移到机器人。Zero-WAM 的关键洞察是：**任务语义层面的对应**（而非动作轨迹层面）足以指导机器人执行同类任务。通过自动化将机器人轨迹转换为语义等价的人类视频，构建大规模跨形态配对数据，再结合因果视频-动作联合建模，使机器人能从人类视频中提取任务意图。

---

## 方法详解

### 模型架构

Zero-WAM 采用 **[[Mixture-of-Transformers|Mixture-of-Transformers (MoT)]]** 架构，以 [[Wan-2.2|Wan-2.2-TI2V-5B]] 视频生成骨干为基础：

- **输入**: 人类示范视频（ICL prompt）或语言指令 $\ell$ + 机器人历史帧 $\mathbf{x}^{\leq i}$ + 历史动作 $\mathbf{a}^{\leq i}$
- **Backbone**: [[Wan2.2-TI2V-5B]]（5B 参数视频 Transformer）
- **核心模块**: [[Mixture-of-Transformers|MoT]] 分离视频流与动作流；[[In-Context Future Chunk Prediction|IFP]] 辅助训练目标
- **位置编码**: [[RoPE]] 坐标偏移区分 ICL 人类视频与机器人观测
- **输出**: 未来机器人视频帧 $\mathbf{x}^{i+1}$ + 可执行动作块 $\mathbf{a}^{i+1}$
- **总参数**: ~5B（基础骨干）

### 核心模块

#### 模块一：因果视频-动作联合建模（Causal Video-Action Modeling）

**设计动机**: 将视频预测与动作预测在同一自回归框架中统一，利用 [[Flow Matching|流匹配]] 目标同时优化视频和动作。

**具体实现**:
- 模型以分块方式自回归预测：每个时间步预测下一帧视频 $\mathbf{x}^{i+1}$ 及对应动作 $\mathbf{a}^{i+1}$
- 采用[[Flow Matching|流匹配]]目标，噪声插值构造训练信号
- 视频流与动作流通过 MoT 独立参数化，共享中间层表示

#### 模块二：In-Context Future Chunk Prediction（IFP）

**设计动机**: 若仅预测下一帧，模型可能忽略 ICL 人类视频而依赖机器人历史短路预测，IFP 通过预测多步未来强制模型利用任务语义。

**具体实现**:
- $K=4$ 个辅助未来块预测器，时间跨步 $s=2$
- 辅助分支仅接收主视频 Transformer 中间层融合表示 $\bm{\phi}^{i+1}$，**不直接访问**人类视频
- 这一架构约束迫使主干网络将人类视频的任务语义编码到 $\bm{\phi}^{i+1}$ 中
- 训练时与主损失联合优化

#### 模块三：HumanGen 数据生成流水线

**设计动机**: 大规模自动化生成人机配对 [[In-Context Learning|ICL]] 数据，绕过人工采集瓶颈。

**具体实现**（四阶段）:
1. **语义分析**: VLM（Gemini 3.1 Pro 或 Qwen3.6-Plus）分析机器人视频，提取任务描述
2. **场景初始化**: 图像编辑模型（Nano Banana 2 或 Qwen-Image-2.0）生成多样化人类场景初始帧
3. **视频合成**: 视频生成模型（Wan 2.7 或 Kling AI 3.0）合成人类操作序列
4. **语义验证**: VLM 验证生成视频的语义一致性和物理合理性

#### 模块四：Task-Diverse VA 预训练数据

**设计动机**: 原始数据集中热门任务轨迹数量远多于长尾任务，导致模型对低频任务的泛化能力差。

**具体实现**:
- 以**任务级别采样**（task-level sampling）替换原始轨迹频率采样
- 每个 epoch 约 400K 条轨迹，覆盖 6000+ 个任务
- 任务多样性是零样本泛化的重要基础

#### 模块五：RoPE 坐标偏移区分 ICL 视频

**设计动机**: 在统一序列中同时编码人类示范视频和机器人观测，需要让模型区分两种来源。

**具体实现**:
- 机器人视频坐标：$pos_{robot}(q,y,x) = (q, y, x)$
- 人类视频坐标：$pos_{human}(q,y,x) = (q, y+\Delta_H, x)$，其中 $\Delta_H > H_{mv}$（$H_{mv}$ 为最大视频高度），保证坐标不重叠
- 时间维度 $q$ 保持一致，仅空间维度偏移

---

## 关键公式

### 公式一：[[Flow Matching|流匹配]]噪声插值

$$
\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\bm{\epsilon}
$$

**含义**: 在干净样本 $\mathbf{x}_0$ 和噪声 $\bm{\epsilon}$ 之间进行线性插值，构造时间步 $t$ 处的带噪样本。

**符号说明**:
- $\mathbf{x}_0$: 干净视频帧（目标）
- $\bm{\epsilon} \sim \mathcal{N}(0, I)$: 高斯噪声
- $t \sim \mathcal{U}(0,1)$: 扩散时间步

---

### 公式二：[[Flow Matching|流匹配]]目标速度

$$
\mathbf{v}^{\star}_t = \bm{\epsilon} - \mathbf{x}_0
$$

**含义**: 流匹配的目标速度场，表示从噪声到干净样本的方向。

**符号说明**:
- $\mathbf{v}^{\star}_t$: 目标速度（恒定，与 $t$ 无关）
- $\bm{\epsilon}$: 噪声端点
- $\mathbf{x}_0$: 干净端点

---

### 公式三：[[Flow Matching|流匹配]]损失

$$
\mathcal{L}_{fm} = \mathbb{E}_{\mathbf{x}_0, \bm{\epsilon}, t}\left[\|\mathbf{v}_{\theta}(\mathbf{x}_t, t, c) - \mathbf{v}^{\star}_t\|_2^2\right]
$$

**含义**: 模型预测速度与目标速度之间的均方误差，适用于视频和动作预测。

**符号说明**:
- $\mathbf{v}_{\theta}(\cdot)$: 神经网络预测的速度场
- $c$: 条件信号（语言指令或人类视频）
- $\mathbb{E}$: 对数据、噪声、时间步的期望

---

### 公式四：因果视频-动作联合分布

$$
p_{\theta}(\mathbf{x}^{i+1}, \mathbf{a}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, c)
$$

**含义**: 给定历史视频帧和动作序列，联合预测下一帧视频和动作。

**符号说明**:
- $\mathbf{x}^{i}$: 第 $i$ 个时间步的视频帧
- $\mathbf{a}^{i}$: 第 $i$ 个时间步的动作
- $c$: 条件信号（语言或人类视频 ICL）

---

### 公式五：因果视频-动作分解预测

$$
\begin{aligned}
p_{\theta}(\mathbf{x}^{i+1}, \mathbf{a}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, c) = &\; p_{\theta}^{vid}(\mathbf{x}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, c) \\
& \cdot p_{\theta}^{act}(\mathbf{a}^{i+1} \mid \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, \mathbf{x}^{i+1}, c)
\end{aligned}
$$

**含义**: 先预测未来视频帧，再以未来帧为条件预测动作，体现"想象-执行"的因果顺序。

**符号说明**:
- $p_{\theta}^{vid}$: 视频预测分支
- $p_{\theta}^{act}$: 动作预测分支（以预测视频帧为额外条件）

---

### 公式六：[[In-Context Future Chunk Prediction|IFP]] 目标帧索引

$$
j_k = (i+1) + 1 + (k-1)s, \quad k = 1, \ldots, K
$$

**含义**: 计算第 $k$ 个辅助预测目标的帧索引，以时间跨步 $s$ 均匀采样未来帧。

**符号说明**:
- $i+1$: 主预测目标的索引
- $s = 2$: 时间跨步（stride）
- $K = 4$: 辅助预测头数量
- $j_k$: 第 $k$ 个辅助目标的帧索引

---

### 公式七：[[In-Context Future Chunk Prediction|IFP]] 损失

$$
\mathcal{L}_{ifp} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_{fm}(\mathbf{x}^{j_k};\; \bm{\phi}^{i+1}, \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, \ell)
$$

**含义**: 对 $K$ 个辅助未来预测的加权流匹配损失之和，以 $\bm{\phi}^{i+1}$（主干中间表示）为条件，不直接访问人类视频。

**符号说明**:
- $w_k$: 第 $k$ 个辅助头的权重
- $\bm{\phi}^{i+1}$: 主视频 Transformer 第 $(i+1)$ 步的中间层融合表示
- $\ell$: 语言指令（辅助条件，不含人类视频）
- $\mathbf{x}^{j_k}$: 第 $j_k$ 帧（未来帧目标）

---

### 公式八：[[RoPE|RoPE]] 坐标偏移区分 ICL 人类视频

$$
pos_{human}(q, y, x) = (q,\; y + \Delta_H,\; x), \quad \Delta_H > H_{mv}
$$

**含义**: 给人类视频帧施加空间维度偏移 $\Delta_H$，使其与机器人观测的位置编码不重叠，从而让模型学会区分两种来源。

**符号说明**:
- $(q, y, x)$: 时间-高度-宽度坐标
- $\Delta_H$: 高度偏移量，需大于最大视频高度 $H_{mv}$
- $pos_{robot}(q,y,x) = (q,y,x)$: 机器人视频保持原始坐标

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1: Zero-WAM Overview](https://arxiv.org/html/2608.26103v2/framework_v1.0.png)

**说明**: Zero-WAM 整体框架。输入为人类视频示范（ICL prompt）或语言指令，通过 [[Mixture-of-Transformers|MoT]] 因果视频-动作模型同时预测未来机器人视频帧和可执行动作。训练数据包括 Task-Diverse VA 数据和 HumanGen ICL 配对数据。

---

### Figure 2: 数据构建与人类视频生成流水线 / Data Construction Pipeline

![Figure 2: Data Construction](https://arxiv.org/html/2608.26103v2/data_combine_v1.1.png)

**说明**: 上方展示训练数据的两大来源：Task-Diverse VA 数据（任务均衡的机器人视频-动作预训练数据）和 HumanGen（包含外部预训练 ICL、内部预训练 ICL、仿真 ICL、现实 ICL 四个子集）。下方为人类视频自动生成流水线：VLM 分析机器人视频语义 → 图像编辑模型生成人类场景 → 视频生成模型合成人类操作序列 → VLM 验证语义一致性。

---

### Figure 3: RoboTwin 2.0 未见任务 / Unseen Tasks in Simulation

![Figure 3: RoboTwin Unseen Tasks](https://arxiv.org/html/2608.26103v2/robotwin_unseen_v1.1.png)

**说明**: 仿真评估中的 7 个未见任务，涵盖未见物体操作、关节物体操作（微波炉）、双臂协同操作和长时序操作（堆叠三块积木）。这些任务在训练阶段均未出现，用于测试零样本泛化能力。

---

### Figure 4: 真实世界定性评估 / Real-World Qualitative Evaluation

![Figure 4: Real-World Demo](https://arxiv.org/html/2608.26103v2/realworld_demo_v1.0.png)

**说明**: Zero-WAM 在双臂 Franka 机器人上的真实世界执行结果（配合人类视频示范）。展示了三类复杂场景：多物体容器放置、三物体顺序操作和桌腿精细插孔装配。

---

### Figure 5: ICL 人类视频提示效果消融 / Effect of Human Video ICL Prompts

![Figure 5: ICL Ablation](https://arxiv.org/html/2608.26103v2/ablation_icl_v1.1.png)

**说明**: 对比有无人类视频 ICL 提示时各任务的成功率。加入人类视频后，7 任务平均成功率从 WAN-Action 基线的 10.98% 提升至 36.36%，验证了人类视频提供了超越语言描述的额外任务信息。

---

### Figure 6: Task-Balanced 预训练与 IFP 消融 / Ablations of Task-Balanced Pre-training and IFP

![Figure 6: IFP Ablation](https://arxiv.org/html/2608.26103v2/ablation_icl_v1.1.png)

**说明**: 左侧雷达图比较 LingBot-VA 与纯文本 Zero-WAM 变体（评估 Task-Diverse 预训练的效果）；右侧对比完整 Zero-WAM 与移除 IFP 的消融变体。完整 Zero-WAM（46.95%）相比无 IFP 版本（28.55%）提升显著，尤其在关节物体任务上。

---

### Table 1: HumanGen 数据集统计 / HumanGen Dataset Statistics

| 数据来源 | 配对数 | 任务数 | 机器人类型 |
|---------|--------|--------|----------|
| Pre-train ICL (External) | 41,188 | — | 45+ 形态（AgiBot、InternData-A1、Open-X、RoboCOIN、RoboMIND） |
| Pre-train ICL (In-house) | 30,247 | — | 双臂 Franka、Galaxea R1 Pro |
| Simulation ICL | — | — | RoboTwin 2.0 |
| Real-world ICL | — | — | 双臂 Franka（真实） |
| **合计** | **74.2K** | **8.6K** | — |

**说明**: HumanGen 规模远超现有人机配对数据集（如 MIME ~3K、RH20T ~44K 但任务类别 <500），任务类别覆盖广度是其核心优势。

---

### Table 2: RoboTwin 2.0 仿真评估结果 / Simulation Results

| 任务 | LingBot-VA | Zero-WAM（文本） | **Zero-WAM（完整）** |
|------|-----------|----------------|-------------------|
| Open microwave | — | — | **59.00%** |
| Place empty cup | — | — | **84.87%** |
| Move stapler to pad | — | — | **69.14%** |
| Stamp seal | — | — | **47.00%** |
| Place object on scale | — | — | **24.67%** |
| Place bread in basket | — | — | **35.00%** |
| Stack blocks three | — | — | **9.00%** |
| **平均** | **17.45%** | **39.44%** | **46.95%** |

**说明**: Zero-WAM 在 7 个未见任务上平均成功率 46.95%，比 LingBot-VA 基线（17.45%）高出 29.5 个百分点。仅使用语言的 Zero-WAM 变体（39.44%）已大幅优于 LingBot-VA，表明 Task-Diverse 预训练本身贡献显著；加入人类视频 ICL 进一步提升 7.51 个百分点。

---

### Table 3: 真实世界评估结果 / Real-World Evaluation

| 任务 | LingBot-VA | **Zero-WAM（完整）** |
|------|-----------|-------------------|
| Object-to-container placement（5 次） | 43.3% | **53.3%** |
| Three-object sequential manipulation（3 次） | 10.0% | **33.3%** |
| Two-table-leg insertion（2 次） | 0.0% | **16.7%** |

**说明**: 真实世界三类复杂任务上 Zero-WAM 全面超越 LingBot-VA，特别在细粒度装配任务（桌腿插孔）上，基线完全失败而 Zero-WAM 成功率 16.7%。

---

### Table 4: 消融实验汇总 / Ablation Summary

| 配置 | 7 任务平均成功率 | 说明 |
|------|---------------|------|
| WAN-Action 基线（无 ICL） | 10.98% | 无人类视频提示 |
| LingBot-VA（语言 VLA 基线） | 17.45% | 原始频率采样预训练 |
| Zero-WAM（仅文本，Task-Diverse） | 39.44% | 任务均衡预训练的效果 |
| Zero-WAM（无 IFP） | 28.55% | 有 ICL 但无 IFP 模块 |
| **Zero-WAM（完整）** | **46.95%** | 所有组件齐备 |

**关键发现**: Task-Diverse 预训练贡献约 +22 pp（从 17.45% 到 39.44%），IFP 在此基础上额外贡献 +18.4 pp（28.55% → 46.95%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| HumanGen | 74.2K 配对 | 8.6K 任务，自动生成的人机配对 | ICL 预训练 |
| Task-Diverse VA | ~400K 轨迹/epoch | 6000+ 任务，任务级别均衡采样 | VA 预训练 |
| RoboTwin 2.0 | 7 未见任务 | 仿真 benchmark，涵盖多类操作 | 测试 |
| 双臂 Franka（真实） | 3 任务，各 10-15 次 | 复杂真实操作 | 测试 |

### 实现细节

- **Backbone**: Wan-2.2-TI2V-5B（5B 参数视频 Transformer）
- **优化器**: AdamW（学习率 $1 \times 10^{-4}$，权重衰减 0.01）
- **预训练数据比**: Task-Diverse VA : HumanGen = 1:5
- **预训练开销**: 15,360 GPU 小时
- **Post-training**: 4,000 步，64 GPU
- **IFP 超参**: $K=4$，步长 $s=2$
- **推理 CFG**: 视频 CFG=5，动作 CFG=1.0
- **硬件**: 64 GPU（型号未公开）

### 可视化结果

- 对于微波炉开门、面包入篮等关节/长时序任务，ICL 人类视频提供了语言难以描述的连续运动轨迹信息。
- 真实世界三物体顺序操作中，Zero-WAM 能正确推断操作顺序（从人类视频）并顺序执行，LingBot-VA 仅达 10% 成功率。
- 桌腿插孔任务对精度要求极高，Zero-WAM 是首个在真实平台上实现非零成功率的零样本方法。

---

## 批判性思考

### 优点

1. **零样本泛化能力强**: 在 7 个完全未见的仿真任务上平均 46.95%，+29.5 pp over 最优基线。
2. **数据飞轮可扩展**: HumanGen 全自动生成流水线无需人工干预，8.6K 任务覆盖远超现有数据集。
3. **人类视频的优越性得到验证**: 消融实验清晰量化了人类视频相比语言提示的额外贡献（+7.51 pp）。
4. **IFP 设计精巧**: 通过切断辅助预测头对人类视频的直接访问，架构上强制主干提取任务语义，是有效的归纳偏置。

### 局限性

1. **成功率仍然较低**: 最难任务（堆叠三块积木）仅 9%，绝对成功率距实际部署仍有差距。
2. **真实世界评估规模小**: 每任务仅 10-15 次实验，统计置信度有限。
3. **视频质量依赖**: 生成的人类视频若出现语义漂移，会直接影响策略质量，但 VLM 验证步骤的过滤率和效果未详细报告。
4. **计算开销大**: 15,360 GPU 小时预训练成本高，限制了学术可复现性。

### 潜在改进方向

1. 探索更轻量级的 ICL 特征提取器，降低推理时的视频编码开销
2. 研究 IFP 在多任务在线微调场景下的效果（持续学习）
3. 引入触觉传感器作为额外 ICL 通道，增强精细操作的成功率

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数详细报告）
- [ ] 数据集可获取（HumanGen 未公开发布）

---

## 关联笔记

### 基于

- [[LingBot-VA]]: 主要对比基线，同样基于 Wan 骨干的视频-动作模型
- [[Wan2.2-TI2V-5B]]: 视频生成骨干网络
- [[Flow Matching]]: 核心训练目标

### 对比

- [[LingBot-VA]]: 直接 baseline，Zero-WAM 超越 29.5 pp
- [[In-Context Imitation Learning]]: 同类范式，Zero-WAM 扩展到跨形态人类视频

### 方法相关

- [[Mixture-of-Transformers]]: MoT 双流架构（视频流 + 动作流）
- [[In-Context Future Chunk Prediction]]: Zero-WAM 提出的核心训练目标（新概念）
- [[RoPE]]: 坐标偏移区分 ICL 视频与机器人观测
- [[Flow Matching]]: 视频和动作的统一生成目标
- [[World Action Model]]: Zero-WAM 所属的 WAM 研究范式

### 硬件/数据相关

- [[HumanGen]]: Zero-WAM 自动生成的人机配对数据集（新概念）
- [[RoboTwin 2.0]]: 仿真评估平台

---

## 速查卡片

> [!summary] Zero-WAM (arXiv 2608.26103)
> - **核心**: 以人类视频为 ICL prompt，实现机器人对未见操作任务的零样本泛化
> - **方法**: Wan-2.2 骨干 + MoT 双流 + IFP 辅助训练 + HumanGen 自动数据生成
> - **结果**: RoboTwin 2.0 均值 46.95%（+29.5 pp vs LingBot-VA），真实世界三任务全面领先
> - **代码**: 未公开；项目主页：https://robbyant-research.github.io/Zero-WAM/

---

*笔记创建时间: 2026-08-28*

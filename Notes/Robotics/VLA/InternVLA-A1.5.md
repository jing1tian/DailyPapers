---
title: "InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization"
method_name: "InternVLA-A1.5"
authors: [Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, Yuqiang Yang, Jiaqi Peng, Xueyuan Wei, Yangkun Zhu, Jiahao Jiang, Xing Gao, Hanqing Wang, Feng Yuan, Kailin Li, Xueyue Zhu, Tai Wang, Yan Ding, Jiangmiao Pang, Jia Zeng, Jingjing Zhang, Bowen Zhou, Yao Mu, Chunhua Shen, Weinan Zhang]
year: 2025
venue: arXiv
tags: [vla, world-model, latent-foresight, flow-matching, compositional-generalization, robot-manipulation]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.04988
created: 2026-07-08
---

# 论文笔记：InternVLA-A1.5

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 上海AI Lab / 复旦大学 / 香港大学 / 浙江大学 |
| 日期 | July 2025 |
| 项目主页 | [internrobotics.github.io](https://internrobotics.github.io/internvla-a15.github.io/) |
| 对比基线 | [[pi0.5]] / [[Motus]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.04988) |

---

## 一句话总结

> 通过可学习的 Foresight Token 以隐式潜变量形式提炼未来预测，统一理解、动态感知与动作生成，在保持实时推理速度的同时实现强组合泛化能力。

---

## 核心贡献

1. **Latent Foresight 机制**: 引入[[Learnable Foresight Token|可学习 Foresight Token]]，以潜变量查询替代像素级未来帧预测，训练期间以冻结的 [[Wan 2.2]] 视频生成模型为监督信号，推理时直接丢弃，零额外成本
2. **Mixture-of-Transformers 统一架构**: 采用 [[Mixture-of-Transformers]] 设计，将预训练 VLM 主干（Qwen3.5 2B）与 460M 轻量 Expert 融合，共享 Full Attention 层但保留独立线性层，兼顾语言语义与动作动力学
3. **双阶段训练 + 大规模数据配方**: 1.2M 机器人回合 + 300万多模态样本[[Co-training|协同训练]]，在仿真与真实世界多项 benchmark 上达到最优，长指令组合泛化尤为突出

---

## 问题背景

### 要解决的问题

现有机器人操作策略难以在**真实世界中实现组合泛化**——即将训练中见过的指令元素灵活组合形成新指令时，成功率大幅下降。此外，[[World Model|世界模型]]辅助策略通常需要在推理时在线预测未来帧，带来显著延迟。

### 现有方法的局限

- **纯 [[Diffusion Policy|扩散策略]]**（如 [[pi0.5]]）：未显式建模未来动态，缺少物理先验
- **像素级视频预测策略**（Cosmos-Policy 等）：推理时需生成完整视频帧，速度慢（数秒/步）
- **直接微调 VLM**：破坏预训练语义，泛化性下降

### 本文的动机

若能把"理解语义（VLM 主干）"和"预见物理动态（视频生成模型）"的能力同时保留在策略中，且以隐变量形式提炼世界知识（无需在线生成视频），就能同时获得**强泛化**与**实时推理**。

---

## 方法详解

### 模型架构

![Figure 1 - Overview](https://arxiv.org/html/2607.04988v1/x1.png)

**InternVLA-A1.5** 采用 **[[Mixture-of-Transformers]]（MoT）** 架构，包含两个主要组件：

- **输入**: 多视角图像 $\{I_k\}$ + 任务指令 $l$ + 控制模式 $m$ + 离散化本体状态 $s_t$（256 bins，范围 $[-1, 1]$）
- **Backbone**: 预训练 [[Qwen]] 3.5 2B，混合注意力设计（[[GatedDeltaNet2|Gated DeltaNet]] 线性层 + 标准 Full Attention）
- **核心模块**: [[Learnable Foresight Token|Learnable Foresight Token]] 查询 [[Wan 2.2]] 潜在未来 + [[FAST]] Tokenizer 离散化动作
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$（[[FAST]] 离散 Token + [[Flow Matching|流匹配]]连续动作）
- **总参数**: VLM 主干 2B + Unified Expert 460M ≈ **2.46B**

### 核心模块

#### 模块1：Unified Expert（统一专家网络）

**设计动机**: 在不干扰 VLM 预训练权重的情况下，引入机器人操作专用能力

**具体实现**:
- 与 VLM 主干**共享 Full Attention 层**（保留语言对齐能力）
- 维护**独立线性层**（学习机器人特定特征变换）
- 对 VLM Token 采用因果注意力；对 Expert Token 采用**分组因果注意力**（组间因果，组内双向）

![Figure 2 - Framework Architecture](https://arxiv.org/html/2607.04988v1/x2.png)

#### 模块2：Latent Foresight 机制

**设计动机**: 以紧凑潜变量提炼物理动态先验，训练时有监督、推理时零开销

**具体实现**:
1. 引入 $N_f$ 个[[Learnable Foresight Token|可学习 Foresight Token]] $Q^f \in \mathbb{R}^{N_f \times d}$
2. 通过 Unified Expert 对当前多模态上下文做交叉注意力，生成 Foresight 条件嵌入 $H^f$
3. 将 $H^f$ 注入冻结的 [[Wan 2.2]] 视频生成模型，预测未来 4 帧
4. 梯度只流经 Foresight Token 更新，**视频生成模型参数完全冻结**
5. 推理时直接丢弃 Foresight 分支，延迟约 **0.1s/步**

![Figure 4 - Foresight Reasoning](https://arxiv.org/html/2607.04988v1/x4.png)

#### 模块3：对话模板与动作 Token 化

**设计动机**: 统一机器人动作样本与 VQA 样本的输入输出格式

**具体实现**:
- 使用 [[FAST]] Tokenizer（词表大小 2048）将连续动作序列离散化为 Token
- 机器人操作样本与 VQA 样本共用同一 prompt-label 模板结构
- 每个样本包含：$K$ 个图像块 + 任务指令 + 控制模式 $m$ + 离散化状态

![Figure 3 - Chat Template](https://arxiv.org/html/2607.04988v1/x3.png)

#### 模块4：注意力掩码策略

**训练期间**:
- VLM Token 遵循 Qwen3.5 因果注意力
- Expert Token 组间因果、组内双向
- Expert 对 FAST 动作 Token 做掩码（防止信息泄露）

**推理期间**: Foresight 分支整体丢弃

![Figure 5 - Attention Masking](https://arxiv.org/html/2607.04988v1/x5.png)

---

## 关键公式

### 公式1：[[Flow Matching|Flow-Matching 动作损失]]

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau \sim \text{Beta}(1.5,\,1.0)} \left\| v\!\left(a^\tau,\, H_t,\, Q^f\right) - \left(a - \varepsilon\right) \right\|^2
$$

**含义**: 用流匹配方法预测连续动作速度场，监督策略网络将噪声样本 $a^\tau$ 映射到真实动作 $a$

**符号说明**:
- $\tau \sim \text{Beta}(1.5, 1.0)$: 插值时间步采样，Beta 分布偏向后期（接近真实动作侧）
- $a^\tau = (1-\tau)\varepsilon + \tau a$: 插值轨迹（线性插值）
- $a$: 真实动作
- $\varepsilon$: 标准高斯噪声
- $v(\cdot)$: 速度场预测网络
- $H_t$: 当前时刻上下文嵌入
- $Q^f$: Foresight Token 嵌入

### 公式2：[[Video Diffusion Model|视频监督损失]]

$$
\mathcal{L}_{\text{video}} = \mathbb{E} \left\| \hat{F}_{t:t+4}\!\left(H^f\right) - F_{t:t+4} \right\|^2
$$

**含义**: Foresight Token 生成的条件嵌入 $H^f$ 输入冻结视频生成模型，预测未来 4 帧；该损失仅反传至 Foresight Token

**符号说明**:
- $H^f$: Foresight Token 产生的条件嵌入
- $\hat{F}_{t:t+4}$: 冻结 [[Wan 2.2]] 预测的未来 4 帧
- $F_{t:t+4}$: 真实未来 4 帧

### 公式3：[[Autoregressive Policy|Stage 1 训练目标（跨熵）]]

$$
\mathcal{L}_{\text{stage1}} = -\sum_i \log p\!\left(y_i \mid y_{<i},\, x\right)
$$

**含义**: 下一 Token 预测交叉熵，目标包含子任务描述、FAST 动作 Token 和 VQA 答案

**符号说明**:
- $y_i$: 第 $i$ 个目标 Token（动作 Token 或文字 Token）
- $x$: 多模态输入上下文

### 公式4：[[Flow Matching|Stage 2 联合训练目标]]

$$
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{stage1}} + \alpha \cdot \mathcal{L}_{\text{video}} + \beta \cdot \mathcal{L}_{\text{action}}
$$

**含义**: Stage 2 同时优化三个目标：语言理解、Foresight 视频预测、连续动作流匹配

**符号说明**:
- $\alpha = 1$: 视频监督损失权重
- $\beta = 10$: 动作流匹配损失权重

### 公式5：[[Flow Matching|推理采样（Euler 积分）]]

$$
a^{\tau_{k+1}} = a^{\tau_k} + \frac{1}{K} \cdot v\!\left(a^{\tau_k},\, H_t,\, Q^f\right)
$$

**含义**: 推理时通过 $K$ 步 Euler 积分从纯噪声 $\varepsilon$ 逐步去噪得到最终动作

**符号说明**:
- $K$: 推理步数
- $a^{\tau_0} = \varepsilon$: 初始噪声
- $a^{\tau_K}$: 最终预测动作

### 公式6：机器人数据采样权重

$$
w_i \propto N_i^{\,\gamma}, \quad \gamma \in (0, 1)
$$

**含义**: 数据源 $i$ 的采样权重按帧数的 $\gamma$ 次幂比例分配，$\gamma < 1$ 防止大数据源主导训练

**符号说明**:
- $N_i$: 数据源 $i$ 的帧数
- $\gamma$: 缩放指数，控制大小数据集采样均衡程度

---

## 关键图表

### Figure 6：训练数据概览

![Figure 6 - Data Overview](https://arxiv.org/html/2607.04988v1/x6.png)

**说明**: 机器人操控数据来自 6 个来源（1 合成 + 5 真实），共 1.2M 回合 / 861M 帧。右侧展示多模态协同训练的 3M 样本分布（General QA / Box QA / Point QA / Trajectory QA）。

### Figure 7：四项真实世界任务

![Figure 7 - Real-World Tasks](https://arxiv.org/html/2607.04988v1/x7.png)

**说明**: 三项指令跟随任务（Sort Tubes / Insert Tubes / Move Tubes）+ 一项长时程推理任务（MOF 化学实验，共 13 个连续子任务）。

### Figure 8：真实世界结果

![Figure 8 - Real-World Results](https://arxiv.org/html/2607.04988v1/x8.png)

**说明**: InternVLA-A1.5 在 4 项任务中有 3 项优于 [[pi0.5]]，在长时程 MOF 任务中以 76.4% 对比 [[pi0.5]] 的 29.3% 大幅领先。

### Figure 9：Seen vs. OOD 指令绑定

![Figure 9 - OOD Generalization](https://arxiv.org/html/2607.04988v1/x9.png)

**说明**: 将成功率分解为训练见过的指令绑定（seen）与未见过绑定（OOD）。InternVLA-A1.5 在 OOD 条件下保持高成功率，验证组合泛化能力。

### Figure 10：训练效率对比

![Figure 10 - Training Efficiency](https://arxiv.org/html/2607.04988v1/x10.png)

**说明**: 在相同 SFT 设置（60K 步）下，InternVLA-A1.5 比 [[pi0.5]] 和 InternVLA-A1 更快收敛到低损失区域，最终损失值最低。

### Figure 11：Foresight 条件未来帧可视化

![Figure 11 - Foresight Rollouts](https://arxiv.org/html/2607.04988v1/x11.png)

**说明**: 展示冻结 [[World Model|世界模型]]在 Foresight 嵌入条件下预测的未来 4 帧，运动预测准确且场景演化物理一致。

### Figure 12：试管分拣任务设置

![Figure 12 - Test Tube Task](https://arxiv.org/html/2607.04988v1/figures/test_tube.png)

**说明**: 左：原始场景（橙色/蓝色试管 + 左/右盒子）；中：训练集内见过的指令绑定；右：Held-out OOD 绑定。

### Figure 13：长时程 MOF 化学实验

![Figure 13 - MOF Task](https://arxiv.org/html/2607.04988v1/x12.png)

**说明**: 完整指令 + 13 个连续子任务（插漏斗 → 倒液体 → 加塞 → 开搅拌机等），展示策略在长时程任务上的执行能力。

### Table 1：仿真 Benchmark 综合结果

| Benchmark | InternVLA-A1.5 | 主要对比基线 | 说明 |
|-----------|----------------|-------------|------|
| [[LIBERO]] (avg) | **98.9%** | — | 五个 LIBERO 套件均达最优或次优 |
| [[LIBERO-Plus]] (zero-shot) | **84.8%** | — | 最强视觉/布局扰动鲁棒性 |
| [[RoboTwin 2.0]] (avg) | **93.2%** | — | Clean 93.3% → Randomized 93.0%，几乎无下降 |
| DOMINO (zero-shot) | **27.7%** | — | 动态操作任务超越所有基线 |
| EBench (test) | **35.2%** | — | Val-Train / Val-Unseen / Test 三项最优 |
| [[SimplerEnv]] (avg) | **80.8%** | [[pi0.5]] 57.1% | Bridge 任务领先 +23.7 points |

### Table 2：消融实验（4 个 Benchmark 平均）

| 配置 | LIBERO | LIBERO-Plus | RoboTwin | DOMINO |
|------|--------|-------------|----------|--------|
| w/o 视频损失 | 下降 1.0~2.5 pt | **-6.8 pt** | -2.1 pt | -2.4 pt |
| w/o Foresight Token | 下降 0.3~0.7 pt | **-6.9 pt** | -3.0 pt | -3.9 pt |
| **Full Model** | **98.9%** | **84.8%** | **93.2%** | **27.7%** |

**关键发现**: 两个组件在零样本泛化任务（LIBERO-Plus / DOMINO）上收益最显著，说明 Latent Foresight 对 OOD 泛化尤其重要。

---

## 实验

### 数据集

| 数据集 | 规模 | 类型 | 用途 |
|--------|------|------|------|
| InternData-A1 | 396M 帧 | 合成 | 训练 |
| [[AgiBotWorld]] | 168M 帧 | 真实 | 训练 |
| [[UMI]] | 114M 帧 | 真实（in-the-wild） | 训练 |
| [[DROID]] | 71M 帧 | 真实多场景 | 训练 |
| Galaxea | 62M 帧 | 真实 | 训练 |
| RoboMind 1.0 | 50M 帧 | 真实 | 训练 |
| InternVLA-M1 | 300万样本 | 多模态（VQA+轨迹） | 协同训练 |

### 实现细节

- **Backbone**: [[Qwen]] 3.5 2B（混合 Gated DeltaNet + Full Attention）
- **Expert**: 460M 参数，共享 Full Attention 层
- **动作表示**: [[FAST]] Tokenizer（词表 2048）离散 + [[Flow Matching|流匹配]]连续动作
- **Stage 1**: 300K 步，batch 1024
- **Stage 2**: 600K 步，batch 1024，后接 60K 步 post-training（batch 128）
- **推理延迟**: 单张 RTX 5090，约 0.1s/步

### 可视化结果

- Foresight Token 生成的条件向量能驱动冻结 [[Wan 2.2]] 准确预测未来 4 帧的动作轨迹
- 长时程 MOF 任务中，策略能跨 13 个子任务维持连贯动作序列

---

## 批判性思考

### 优点

1. **真正零推理开销的 World Model 集成**: 利用[[Learnable Foresight Token|可学习 Foresight Token]]以隐式潜变量提炼物理先验，绕过推理时视频生成的延迟瓶颈
2. **大规模验证的组合泛化**: OOD 指令绑定测试设计严谨，真实说明了语义先验对组合泛化的作用
3. **训练效率高**: 相同 SFT 数据下收敛更快，说明架构设计的惯性强

### 局限性

1. **Foresight 视野受限**: 仅预见当前动作块（单 chunk）的未来，缺乏跨多 chunk 的显式规划能力，对超长时程任务（>30 步）提升有限
2. **冻结视频生成模型的泛化边界**: [[Wan 2.2]] 在通用视频数据上预训练，对高度定制化机器人场景（极端遮挡、新型末端执行器）覆盖有限，Foresight 条件可能偏离真实物理动态
3. **数据依赖合成数据**: 396M 帧 InternData-A1 为合成，Sim-to-Real gap 的影响未充分分析

### 潜在改进方向

1. 多 chunk Foresight 规划：将 Foresight Token 扩展为层级式时序预测，提升超长任务成功率
2. 在线 Foresight 蒸馏：利用推理时轻量扩散步骤动态更新 Foresight 表示

### 可复现性评估

- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（论文给出完整超参）
- [x] 数据集可获取（多数为公开数据集）

---

## 关联笔记

### 基于

- [[InternVL]]: VLM 主干来自 InternVL 系列
- [[Qwen]]: 使用 Qwen3.5 2B 作为 VLM 骨干
- [[FAST]]: 动作 Token 化方法
- [[Wan 2.2]]: 作为冻结的视频生成监督信号

### 对比

- [[pi0.5]]: 主要基线，三项指令跟随任务中有 2 项被超越，MOF 任务被大幅超越
- [[Diffusion Policy]]: Foresight 流匹配扩展了扩散策略框架

### 方法相关

- [[Mixture-of-Transformers]]: 核心架构设计
- [[Flow Matching]]: 连续动作生成方法
- [[Learnable Foresight Token]]: 核心创新组件
- [[World Model]]: Foresight 机制的理论框架
- [[Action Chunking]]: 动作预测范式
- [[Co-training]]: 机器人数据与多模态数据协同训练

### 数据集/Benchmark 相关

- [[LIBERO]]: 仿真 benchmark
- [[LIBERO-Plus]]: 增强泛化评估 benchmark
- [[RoboTwin 2.0]]: 双臂操作 benchmark
- [[SimplerEnv]]: 跨体现迁移评估
- [[DROID]]: 真实世界多场景训练数据
- [[UMI]]: In-the-wild 训练数据

---

## 速查卡片

> [!summary] InternVLA-A1.5
> - **核心**: 以可学习 Foresight Token 提炼冻结世界模型的物理先验，统一语义理解、动态感知与动作生成
> - **方法**: MoT 架构（Qwen3.5 2B + 460M Expert）+ Latent Foresight + FAST + Flow Matching
> - **结果**: LIBERO 98.9% / RoboTwin 93.2% / MOF 任务 76.4%（vs π0.5 29.3%）
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-07-08*

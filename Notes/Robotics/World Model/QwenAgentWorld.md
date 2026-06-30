---
title: "Qwen-AgentWorld: Language World Models for General Agents"
method_name: "QwenAgentWorld"
authors: [Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, Yantao Liu, Qin Zhu, Yinger Zhang, Bowen Yu, Haiquan Zhao, Haiyang Xu, Jianxin Yang, Jiayang Cheng, Junyang Wang, Lianghao Deng, Mingfeng Xue, Tianyi Bai, Yang Fan, Yubo Ma, Yucheng Li, Zeyu Cui, Zhihai Wang, Zhihui Xie, Zhuorui Ye, An Yang, Dayiheng Liu, Jingren Zhou, Ning Ding]
year: 2026
venue: arXiv
tags: [language-world-model, agent, reinforcement-learning, llm, environment-simulation, multi-domain]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.24597
created: 2026-06-30
---

# 论文笔记：Qwen-AgentWorld: Language World Models for General Agents

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team (Alibaba) |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[GPT-4]], [[Claude Opus]], [[Gemini]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.24597) / Code: N/A |

---

## 一句话总结

> Qwen-AgentWorld 是首批跨七个智能体域（MCP/Search/Terminal/SWE/Android/Web/OS）统一模拟的语言世界模型，通过三阶段训练（CPT→SFT→RL）在 AgentWorldBench 上超越 GPT-5.4 等前沿模型，并提出两种增强智能体的应用范式。

---

## 核心贡献

1. **首个统一多域语言世界模型**: 在单一 [[Language World Model|语言世界模型]] 中跨 7 个智能体环境域进行仿真（MCP、Search、Terminal、SWE、Android、Web、OS），规模涵盖 35B 和 397B MoE 两个变体。
2. **三阶段训练流水线**: CPT 注入状态转移动态知识 → SFT 激活显式下一状态预测推理 → [[GSPO]] RL 通过混合五维评分+规则奖励提升仿真保真度。
3. **AgentWorldBench 评测基准**: 基于 5 个前沿模型在 9 个已有 benchmark 上的真实交互构建，包含 2,170 个样本和五维评分体系（格式/事实性/一致性/真实性/质量）。
4. **两种应用范式**: 解耦式（LWM 作为环境仿真器驱动智能体 RL）和统一式（世界模型训练作为下游智能体的有效预热），均在 7 个智能体 benchmark 上实现增益。

---

## 问题背景

### 要解决的问题

[[World Model|世界模型]] 基于当前观测和动作预测环境动态，是推理和规划的核心认知机制。然而现有世界模型主要聚焦于视频或特定任务场景，缺乏能够统一仿真多种真实智能体环境的语言世界模型。

### 现有方法的局限

- 视频世界模型（Genie, Marble, Pan 等）以像素为预测目标，无法直接应用于文本/GUI 智能体任务
- 现有 LLM 智能体缺乏对环境动态的显式建模，导致规划能力受限
- 智能体 RL 训练依赖真实环境，基础设施成本高、扩展性差

### 本文的动机

利用 LLM 的语言理解和推理能力，通过大规模真实轨迹数据训练，使语言模型能够通过 [[Chain-of-Thought Reasoning|长链式推理]] 准确预测不同类型智能体环境的下一状态，从而既可作为可扩展的仿真器，也可作为改善智能体能力的预训练基础。

---

## 方法详解

### 模型架构

Qwen-AgentWorld 采用 **[[Mixture of Experts|MoE]] 语言模型** 架构，包含两个规模变体：

- **Qwen-AgentWorld-35B-A3B**: 35B 总参数，3B 激活参数
- **Qwen-AgentWorld-397B-A17B**: 397B 总参数，17B 激活参数

**输入**: 任务描述 $c$ + 历史观测动作序列 $\{o_{\leq t}, a_{\leq t}\}$  
**核心能力**: [[Chain-of-Thought Reasoning|长链式推理]] 驱动的环境状态转移预测  
**输出**: 预测的下一观测状态 $\hat{o}_{t+1}$

### 统一环境轨迹 Schema

为跨域统一，设计包含五要素的标准化格式：
- **任务描述**: 目标与环境背景
- **动作空间**: 可用动作的形式定义
- **初始状态**: 初始观测
- **演示示例**: 少样本示范轨迹
- **仿真指令**: 域特定的提示规范

### 核心模块

#### 模块1: 三阶段训练流水线（CPT → SFT → RL）

**设计原则**: "CPT 注入，SFT 激活，RL 磨砺"（CPT injects, SFT activates, RL sharpens）

**Stage 1 — 持续预训练（CPT）**:
- 注入状态转移动态和专业语料库知识
- 使用非思考（non-thinking）轨迹，专注于世界知识灌注
- 数据规模：10M+ 真实环境交互轨迹

**Stage 2 — 监督微调（SFT）**:
- 通过 [[Rejection Sampling|拒绝采样]] 激活显式下一状态预测推理模式
- 全局保留率 69.2%（见 Table 4）
- 最终 SFT 数据：7,094 条轨迹

**Stage 3 — 强化学习（RL）**:
- 使用 [[GSPO]]（Zheng et al., 2025）算法
- **混合奖励**: 五维评分奖励 : 规则奖励 = 9:1
- 单轮扩展防止奖励坍塌（stability solution）
- 严格标签提取检测自我表扬（self-praise detection）
- 规则验证锚定奖励信号

#### 模块2: 基于信息论的逐轮损失掩码

**设计动机**: 智能体轨迹中存在大量低信息量的观测（如 API echo、boilerplate），直接训练会引入噪声。利用统计信号识别高价值训练轮次：

四个统计指标：

$$
OL = \frac{|W_{act} \cap W_{obs}|}{|W_{act}|}
$$

$$
Nov = \frac{|W_{obs} \setminus W_{act}|}{|W_{obs}|}
$$

$$
Jac = \frac{|W_{act} \cap W_{obs}|}{|W_{act} \cup W_{obs}|}
$$

$$
R = \frac{|obs|}{|act|}
$$

其中 $W_{act}$ 和 $W_{obs}$ 分别是动作和观测的词集合，$R$ 是长度比。

#### 模块3: AutoResearch 提示优化

通过迭代"提议-评估-精炼"循环自动生成 12 个模板变体（v0-v11），针对不同域优化系统提示，避免人工设计的提示工程瓶颈。

---

## 关键公式

### 公式1: [[Language World Model|语言世界模型]] 预测

$$
\hat{o}_{t+1} = f_\theta(c, o_{\leq t}, a_{\leq t})
$$

**含义**: 给定任务上下文 $c$、历史观测序列 $o_{\leq t}$ 和历史动作序列 $a_{\leq t}$，语言世界模型 $f_\theta$ 预测下一观测状态。

**符号说明**:
- $\hat{o}_{t+1}$: 预测的下一观测状态
- $f_\theta$: 语言世界模型（参数化为 $\theta$）
- $c$: 任务上下文/描述
- $o_{\leq t}$: 时刻 $t$ 前的全部观测历史
- $a_{\leq t}$: 时刻 $t$ 前的全部动作历史

### 公式2: [[Information-Theoretic Loss Masking|信息论损失掩码]] 指标

$$
OL = \frac{|W_{act} \cap W_{obs}|}{|W_{act}|}, \quad Nov = \frac{|W_{obs} \setminus W_{act}|}{|W_{obs}|}, \quad Jac = \frac{|W_{act} \cap W_{obs}|}{|W_{act} \cup W_{obs}|}, \quad R = \frac{|obs|}{|act|}
$$

**含义**: 通过动作-观测词集合重叠度（OL）、新颖性（Nov）、Jaccard 系数（Jac）和长度比（R）四个信号判断观测是否为高价值训练样本。

**符号说明**:
- $W_{act}$: 动作文本的词集合
- $W_{obs}$: 观测文本的词集合
- $OL$（Overlap）: 动作词在观测中的覆盖比例，高 OL → 低信息量（Echo 类）
- $Nov$（Novelty）: 观测中动作未提及的新词比例，高 Nov → 高信息量
- $Jac$（Jaccard）: 两集合交并比
- $R$（Ratio）: 观测相对动作的长度比，极低或极高均可能是低价值

---

## 关键图表

### Figure 1: Qwen-AgentWorld 总体概览

![Figure 1](https://arxiv.org/html/2606.24597v1/x1.png)

**说明**: 上方展示跨 7 个域的统一原生语言世界模型；下方展示两种互补的应用策略——解耦式环境仿真器（驱动智能体 RL）和统一式智能体基础模型（世界模型训练作为预热）。

### Figure 2: 七个统一域

![Figure 2](https://arxiv.org/html/2606.24597v1/x2.png)

**说明**: Qwen-AgentWorld 在单一 LWM 内统一七类交互环境仿真：MCP（JSON 工具调用）、Search（网页查询）、Terminal（Bash 命令）、SWE（代码读写执行）、Android（触摸/滑动/输入）、Web（点击/导航）、OS（鼠标/键盘操作）。

### Figure 3: Terminal 域系统提示结构

![Figure 3](https://arxiv.org/html/2606.24597v1/x3.png)

**说明**: Terminal 域 LWM RL 系统提示的五要素解剖：任务描述、动作空间定义、初始状态、演示示例和仿真指令。

### Figure 4: SWE 和 Android 交互示例

![Figure 4](https://arxiv.org/html/2606.24597v1/x4.png)

**说明**: 展示文本域（SWE 代码操作）和 GUI 域（Android 界面交互）的代表性交互样本，体现观测空间的宽广度——从纯文本工具输出到 UI 视图层级结构。

### Figure 5: 三阶段训练流水线

![Figure 5](https://arxiv.org/html/2606.24597v1/x5.png)

**说明**: Stage 1 CPT 注入世界知识，Stage 2 SFT 通过拒绝采样激活下一状态预测的思维链推理，Stage 3 RL 通过 [[GSPO]] 与混合奖励提升仿真保真度。

### Figure 6: AgentWorldBench 组成与统计

![Figure 6](https://arxiv.org/html/2606.24597v1/x6.png)

**说明**: 展示 AgentWorldBench 的域分布（2,170 样本）、来源 benchmark（Terminal-Bench 1.0/2.0、OSWorld-Verified、Tool Decathlon、MCPMark、WideSearch）以及五维评估体系。

### Figure 7: AgentWorldBench 主要结果

![Figure 7](https://arxiv.org/html/2606.24597v1/x7.png)

**说明**: 各模型在 AgentWorldBench 五维评分均值上的对比结果，Qwen-AgentWorld-397B-A17B 总体均值 58.71 略超 GPT-5.4（58.25）。

### Table 1: 七个域的对比

| 域 | 动作类型 | 观测类型 | 核心能力 |
|-----|---------|---------|---------|
| MCP | JSON 工具调用 | 工具响应 | 事实性知识 |
| Search | 网页查询/提取 | 对话历史 | 事实性知识 |
| Terminal | Bash 命令 | 终端输出 | 长上下文因果推理 |
| SWE | Read/Edit/Bash | 工具输出 | 代码执行推理 |
| Android | Touch/Swipe/Type | UI 视图层级 | 视觉状态推理 |
| Web | Click/Type/Navigate | 可访问性树 | 视觉状态推理 |
| OS | Mouse/Keyboard | 可访问性树 | 视觉状态推理 |

### Table 2: SFT 和 RL 训练数据统计

| 域 | SFT | RL Train | 平均 Tokens | 平均轮次 |
|----|-----|---------|------------|---------|
| MCP | 179 | 4,156 | 62,702 | 28.9 |
| Search | 1,042 | 20,004 | 18,873 | 6.2 |
| Terminal | 1,580 | 34,125 | 5,805 | 12.0 |
| SWE | 249 | 8,181 | 36,734 | 24.7 |
| Android | 1,337 | 11,498 | 30,064 | 19.3 |
| Web | 1,605 | 8,716 | 19,417 | 10.2 |
| OS | 1,102 | 5,628 | 25,439 | 12.4 |
| **Total** | **7,094** | **92,308** | **19,443** | **13.4** |

### Table 3: 逐轮损失掩码类别与保留率

| 类别 | 保留率 | 直觉说明 |
|------|--------|---------|
| Retrieval（检索） | 100% | 读取文件 → 内容展示，高信息量 |
| Expansion（扩展） | 100% | 获取页面 → 页面+元数据，高信息量 |
| Action（动作） | 100% | 发送邮件 → "已发送"，有因果关系 |
| Transform（转换） | 50% | 长输入 → 状态词，中等信息量 |
| Boilerplate（样板） | 10% | API echo 模式，低信息量 |
| Echo（回显） | 5% | Think(x) → {thought:x}，几乎无信息 |
| Other（其他） | 100% | 未分类，默认保留 |

### Table 4: SFT 拒绝采样统计

| 域 | 候选数 | 保留率 | 最终 SFT 数 | 平均轮次 |
|----|--------|--------|------------|---------|
| MCP | 261 | 68.6% | 179 | 24.3 |
| Search | 1,466 | 71.1% | 1,042 | 3.3 |
| Terminal | 1,826 | 86.5% | 1,580 | 5.9 |
| SWE | 402 | 61.9% | 249 | 26.9 |
| Android | 1,975 | 67.7% | 1,337 | 15.9 |
| Web | 2,697 | 59.5% | 1,605 | 3.0 |
| OS | 1,623 | 67.9% | 1,102 | 5.4 |
| **Total** | **10,250** | **69.2%** | **7,094** | **8.5** |

### Table 5: AgentWorldBench 主要结果（五维评分均值）

| 模型 | MCP | Search | Terminal | SWE | Android | Web | OS | Avg |
|------|-----|--------|----------|-----|---------|-----|-----|-----|
| Claude Opus 4.8 | 54.93 | 35.14 | 59.18 | 64.10 | 61.50 | 54.66 | 66.62 | 56.59 |
| Claude Opus 4.6 | 69.90 | 29.30 | 57.51 | 64.55 | 61.74 | 51.42 | 70.20 | 57.80 |
| GPT-5.4 | 70.10 | 37.26 | 53.69 | 66.29 | 60.00 | 51.80 | 68.58 | 58.25 |
| Qwen3.5-35B-A3B | 57.87 | 25.98 | 46.13 | 47.58 | 53.18 | 47.10 | 56.27 | 47.73 |
| **Qwen-AgentWorld-35B-A3B** | 64.79 | 36.69 | 53.96 | 65.63 | 58.17 | 49.55 | 65.92 | 56.39 |
| Qwen3.5-397B-A17B | 68.31 | 30.81 | 55.30 | 64.44 | 54.90 | 48.55 | 60.85 | 54.74 |
| **Qwen-AgentWorld-397B-A17B** | 68.24 | 37.82 | **57.73** | **68.49** | 60.20 | 50.98 | 67.89 | **58.71** |

**关键发现**: 397B 变体在总体均值上以 58.71 微超 GPT-5.4（58.25）；35B 变体以 56.39 逼近 Claude Opus 4.8（56.59）；Terminal（+4.04）和 SWE（+2.20）是相对基座模型（397B）提升最显著的域。

---

## 实验

### 数据集

| 数据集/Benchmark | 规模 | 特点 | 用途 |
|----------------|------|------|------|
| 10M+ 环境轨迹 | 10M+ 条 | 覆盖 7 域的真实交互 | CPT 训练 |
| SFT Pool | 10,250 条 | 拒绝采样后高质量轨迹 | SFT 训练 |
| RL Pool | 92,308 条 | 平均 19,443 tokens, 13.4 轮 | RL 训练 |
| AgentWorldBench | 2,170 样本 | 5 前沿模型 × 9 benchmark 构建 | 评估 |
| Terminal-Bench 1.0/2.0 | - | 终端操作 | AgentWorldBench 来源 |
| OSWorld-Verified | - | OS 操作验证集 | AgentWorldBench 来源 |
| Tool Decathlon | - | 工具调用 | AgentWorldBench 来源 |
| MCPMark | - | MCP 协议评测 | AgentWorldBench 来源 |
| WideSearch | - | 网络搜索 | AgentWorldBench 来源 |

### 实现细节

- **模型基座**: Qwen MoE 系列（35B-A3B 和 397B-A17B）
- **RL 算法**: [[GSPO]]（Zheng et al., 2025）
- **奖励比例**: 五维评分奖励 : 规则验证奖励 = 9:1
- **评估模型**: GPT-5.2 作为参考锚定的 LLM 裁判
- **跨裁判一致性**: Gemini 3 Flash、Claude Sonnet 4.5、GPT-5.2 的 Spearman 相关系数 ρ = 0.92–0.99

### Applications 实验结果

**解耦式（Section 6.1）**:
- 仿真 4,000 个真实 OpenClaw 环境用于智能体 RL 训练
- 仿真器 RL 在 Tool Decathlon、MCPMark、WideSearch 上超越纯真实环境训练
- 可控仿真支持 MCP 环境自适应、Search 虚构世界构建等场景

**统一式（Section 6.2）**:
- 世界模型训练作为预热，在 7 个智能体 benchmark 上全面提升下游性能
- 覆盖 Terminal-Bench 2.0、SWE-Bench Verified & Pro、BFCL v4、Claw-Eval、QwenClawBench、WideSearch

### LWM 推理模式分析（Section 7）

RL 训练涌现出三类高质量推理模式：
1. **审慎自我纠错**: 对生成内容主动回顾和修正
2. **信息泄露防护**: 避免在仿真中暴露参考答案
3. **多步因果推理**: 追踪长轨迹中的状态变化链

RL 训练微观层面的改进：URL 真实性提升（Search 域）、字符级字节运算（Terminal 域）、API Schema 保真度（MCP 域）。

---

## 批判性思考

### 优点

1. **首个统一多域 LWM**: 在单一模型中同时仿真文本域和 GUI 域，是领域内重要的工程与方法创新
2. **评测体系完善**: AgentWorldBench 基于真实前沿模型轨迹构建，参考锚定评估方式跨裁判一致性极高（ρ = 0.92-0.99）
3. **双应用范式互补**: 解耦和统一两种范式分别解决了智能体 RL 的扩展性问题和基础能力问题，实用性强
4. **AutoResearch 自动化提示优化**: 减少了人工提示工程的工作量，迭代提升模板质量

### 局限性

1. **GUI 域依赖可访问性树**: Android/Web/OS 域的观测输入是可访问性树而非像素，对视觉理解的真实性有限制
2. **Unified approach 数字未完整披露**: 7 个智能体 benchmark 的具体改进数字在论文中未全部呈现
3. **评估依赖前沿 LLM 裁判**: AgentWorldBench 需 GPT-5.2 作为裁判，存在评估成本高且不开放的问题

### 潜在改进方向

1. 扩展到像素级 GUI 仿真，支持不依赖可访问性树的视觉域仿真
2. 探索更大规模 RL 训练对仿真保真度的进一步提升
3. 将 LWM 与规划算法（如 MCTS）结合，构建更强的智能体推理框架

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（三阶段流水线描述详细）
- [x] 数据集部分可获取（来源 benchmark 均为公开）

---

## 关联笔记

### 基于

- [[Dreamer-v4]]: 经典视频世界模型，语言 WM 的先驱对比
- [[DIAMOND]]: 基于扩散的世界模型代表
- [[GSPO]]: 本文 RL 阶段使用的算法
- [[Chain-of-Thought Reasoning]]: 核心推理机制

### 对比

- [[WAM-Survey]]: 世界模型综述，提供对比框架
- [[WorldOlympiad]]: 同类多域世界模型评测

### 方法相关

- [[Language World Model]]: 核心技术类别
- [[Mixture of Experts]]: 模型架构
- [[Rejection Sampling]]: SFT 数据筛选
- [[Information-Theoretic Loss Masking]]: 逐轮损失掩码技术

### 硬件/数据相关

- [[AgentWorldBench]]: 本文提出的评测基准

---

## 速查卡片

> [!summary] Qwen-AgentWorld: Language World Models for General Agents
> - **核心**: 首个跨 7 个智能体域（MCP/Search/Terminal/SWE/Android/Web/OS）的统一语言世界模型
> - **方法**: 三阶段训练（CPT→SFT→GSPO RL）+ 信息论损失掩码 + 混合五维评分奖励
> - **结果**: 397B 变体在 AgentWorldBench 总均值 58.71，微超 GPT-5.4（58.25）
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-30*

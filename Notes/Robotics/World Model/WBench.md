---
title: "WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation"
method_name: "WBench"
authors: [Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, Henghui Ding]
year: 2026
venue: arXiv
tags: [benchmark, world-model, video-generation, interactive-world-model, multi-turn-evaluation, video-quality, navigation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.25874
created: 2026-05-28
---

# 论文笔记：WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University, Meituan LongCat Team |
| 日期 | May 2026 |
| 项目主页 | [meituan-longcat.github.io/WBench](https://meituan-longcat.github.io/WBench) |
| 对比基线 | [[VBench]], [[WorldScore]], [[Omni-WorldBench]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.25874) / [Code](https://github.com/meituan-longcat/WBench) |

---

## 一句话总结

> WBench 是首个覆盖双视角、四类交互、五维度评估的多轮交互式视频世界模型综合基准，含 289 个测例、1,058 个交互轮次和 22 个自动化指标，揭示了当前世界模型在导航可控性与渲染质量之间的根本性解耦问题。

---

## 核心贡献

1. **综合多轮评测框架**: 提出覆盖视频质量、场景遵从、交互遵从、一致性、物理合规五个维度的统一评测体系，支持 multi-turn 多步交互评估
2. **统一导航控制协议**: 将文本驱动、相机控制、动作条件三类范式的导航指令统一归一化，实现跨范式公平对比
3. **关键发现揭示**: 通过对 20 个最前沿世界模型的实测，揭示了导航控制能力与渲染质量之间近乎零相关（r=-0.12）的根本性解耦现象

---

## 问题背景

### 要解决的问题

[[交互式世界模型|Interactive World Model]] 领域缺乏统一、全面的评测标准。现有 benchmark 只能从某一侧面评估模型：VBench 只测渲染质量而无交互；WorldScore 有相机轨迹但缺语义交互；MIND 仅测单轮内存；Omni-WorldBench 覆盖多类交互但缺第三人称视角；WorldLens 限于自动驾驶场景。

### 现有方法的局限

| 基准 | Cases | 视角 | 交互类型 | 主要不足 |
|------|-------|------|----------|----------|
| VBench | 946 | — | None | 只评质量，无交互 |
| WorldScore | 3,000 | FPP | Navigation | 仅第一视角，无语义交互 |
| MIND | 250 | FPP | Navigation | 无多轮，无语义 |
| Omni-WorldBench | 1,068 | FPP | Nav, SA, EE | 仅第一视角 |
| WorldLens | 26k | TPP | Navigation | 仅自动驾驶场景 |
| **WBench** | **289** | **Both** | **All four** | — |

### 本文的动机

世界模型需扮演五种互补角色：渲染器（视觉保真）、导演（场景初始化）、控制器（交互执行）、记忆（状态保留）、引擎（物理合规）。现有基准只能部分覆盖，WBench 首次将开放域场景、双视角、四类交互、多轮评估统一在同一框架下。

---

## 方法详解

### 核心形式化定义

[[交互式世界模型]] 的生成过程可形式化为：

$$
o_{t+1} \sim f_\theta(o_{t+1} \mid o_{\leq t},\ a_{\leq t})
$$

**含义**: 世界模型 $f_\theta$ 在给定历史观测序列 $o_{\leq t} = (o_0, \ldots, o_t)$ 和历史动作序列 $a_{\leq t} = (a_0, \ldots, a_t)$ 的条件下，生成下一帧观测 $o_{t+1}$。

**符号说明**:
- $o_t$: 第 $t$ 步的视频帧/观测
- $a_t$: 第 $t$ 步的控制动作（文本指令 / 6-DoF 相机姿态 / 键盘按键）
- $f_\theta$: 参数化的世界模型生成函数
- $o_{\leq t}$: 包含第 0 到第 t 步的历史观测

### 数据集构建

WBench 数据集包含 289 个测试用例、1,058 个交互轮次，覆盖以下分布：

**视角分布**: 62% 第一人称（FPP），38% 第三人称（TPP）

**场景分布**: 自然（31%）、城市（21%）、室内（17%）、工作场所（13%）、奇幻（10%）、运动（8%）

**渲染风格**: 写实（52%）、动画/卡通/CG/油画/水墨/素描/抽象（48%）

**主体类型**: 人类（64%）、动物（9%）、机器人（9%）、载具（7%）、其他（10%）

**交互类型分布**: 导航 Navigation（57%）、主体动作 Subject Action（20%）、事件编辑 Event Editing（17%）、视角切换 Perspective Switching（6%）

### 统一导航控制协议

WBench 将三类不同控制范式统一归一化以实现公平对比：

- **文本驱动模型**: 接收自然语言提示（如 "move forward 5 steps"）
- **相机控制模型**: 接收 6-DoF 姿态矩阵
- **动作条件模型**: 接收离散键盘命令（WASD 等）

视角依赖语义：第一人称旋转产生朝向变化，第三人称旋转产生绕主体的轨道运动。

### 五个评估维度与 22 个指标

#### 维度 1: 视频质量 Video Quality（6 个指标）

- **V.1 Aesthetic Quality**: 感知美观性
- **V.2 Imaging Quality**: 技术图像保真度
- **V.3 Temporal Flickering**: 帧间伪影检测
- **V.4 Dynamic Degree**: 运动强度
- **V.5 Motion Smoothness**: 时序连续性
- **V.6 HPSv3-Norm**: 百分位归一化的人类偏好奖励分数

#### 维度 2: 场景遵从 Setting Adherence（2 个指标）

- **S.1 Scene Adherence**: 评估"初始可见"元素（地形、建筑）及期望出现的"画外"元素
- **S.2 Subject Adherence**: 分解为外观匹配和运动风格一致性，通过 [[VLM]] 打分

#### 维度 3: 交互遵从 Interaction Adherence（4 个指标）

- **I.1 Navigation Score**: 使用 MegaSaM 姿态估计，结合归一化绝对轨迹误差（nATE）和跨轮次一致性
- **I.2 Event Editing Adherence**: 五项二元检查的 VLM 协议（变化检测、事件发生、完成度、细节准确性、无异常）
- **I.3 Subject Action Adherence**: 与 I.2 相同的五项检查，结果归一化为 [0,100] 分
- **I.4 Perspective Switching Adherence**: 更严格的三项分类评估（过渡可见性、目标类型一致性、结构合规性）

#### 维度 4: 一致性 Consistency（8 个指标）

- **C.1 Spatial Consistency**: DreamSim 感知相似度（往返轨迹）
- **C.2 Gated Spatial Consistency**: 带中间帧最小值过滤的空间一致性
- **C.3 Segment Continuity**: TransNetV2 硬切检测，无切割帧比例
- **C.4 Perspective Consistency**: SAM2 主体质心稳定性
- **C.5 Geometric Consistency**: Depth Anything 3 深度重投影位移
- **C.6 Photometric Consistency**: 重投影帧对的像素级 PSNR
- **C.7 Subject Consistency**: DINOv2 + CLIP 双信号追踪
- **C.8 Background Consistency**: 均值成对 CLIP 余弦相似度

#### 维度 5: 物理合规 Physical Compliance（2 个指标）

- **P.1 Causal Fidelity**: 两阶段 VLM 协议（Stage 1 全局合理性；Stage 2 七类物理子维度：流体/烟雾、碰撞、表面轨迹、变形、风力、反射、人体运动）
- **P.2 Visual Plausibility**: 微调的 Qwen3-VL-30B 在 [1,5] 量表上评估几何畸变、物体穿透、非自然变形

---

## 关键公式

### 公式1: [[归一化绝对轨迹误差|导航评分 nATE]]

$$
\text{NavScore} = \frac{1}{2}\left(\text{nATE\_score} + \text{Cross-turn Consistency}\right)
$$

**含义**: 导航得分综合轨迹准确度和跨轮次一致性的均值。

**符号说明**:
- $\text{nATE}$: Normalized Absolute Trajectory Error，使用 MegaSaM 估计的相机姿态与 ground-truth 轨迹对齐后的归一化误差
- $\text{Cross-turn Consistency}$: 多轮交互后的轨迹稳定性

### 公式2: [[VLM 评分协议|交互遵从 VLM 评分]]

$$
\text{Score}_\text{interaction} = \frac{1}{5}\sum_{k=1}^{5} b_k \times 100
$$

**含义**: 五项二元标准的均值归一化为 [0,100] 分。

**符号说明**:
- $b_k \in \{0, 1\}$: 第 $k$ 项二元标准（变化检测、事件发生、完成度、细节准确性、无异常）

---

## 关键图表

### Figure 1: WBench 整体概览

![Figure 1 Overview](https://arxiv.org/html/2605.25874v1/x1.png)

**说明**: 上方展示一个包含导航、主体动作、事件编辑和视角切换的多轮测试用例；下方展示 benchmark 设计，包括世界设置、交互分类、统一导航控制和五维评估体系。

### Figure 2: 数据集组成

![Figure 2 Dataset Composition](https://arxiv.org/html/2605.25874v1/x2.png)

**说明**: WBench 数据集从八个轴向展示组成分布，涵盖视角、场景类型、渲染风格、主体类型和交互类型的多样性。

### Figure 3: 跨维度相关性与场景偏差分析

![Figure 3 Cross-dimension Analysis](https://arxiv.org/html/2605.25874v1/x3.png)

**说明**: (a) 六维度 Pearson 相关矩阵（n=20 模型，导航子集）；(b) 七维度相关矩阵（n=9 文本条件模型，交互拆分为导航和语义）；(c) 五维度 Z-score 偏差，红色=更容易，蓝色=更难。揭示导航与质量/一致性/物理的近乎零相关。

### Figure 4: 多轮性能衰减

![Figure 4 Per-turn Degradation](https://arxiv.org/html/2605.25874v1/x4.png)

**说明**: T4+ 汇聚第 4 轮及之后所有轮次。导航性能从第 1 轮到第 4+ 轮下降约 33 分，事件编辑下降 13 分，主体动作下降 9 分，揭示 multi-turn 交互中的长程记忆退化问题。

### Figure 5: 人类偏好对齐验证

![Figure 5 Human Preference Alignment](https://arxiv.org/html/2605.25874v1/x5.png)

**说明**: 十个评估维度的 Spearman ρ（x 轴为每模型人类胜率，y 轴为 WBench 自动分数）。所有维度达到 ρ≥0.94，四个维度达到 ρ=1.00，证明自动化指标可靠反映人类判断。

### Table 1: 与现有基准的对比

| Benchmark | Cases | Turns | 视角 | 交互类型 | 评估维度 |
|-----------|-------|-------|------|----------|----------|
| VBench | 946 | 946 | — | None | 质量 |
| WorldScore | 3,000 | 3,000 | FPP | Navigation | 质量、遵从 |
| MIND | 250 | — | FPP | Navigation | 质量、一致性 |
| Omni-WorldBench | 1,068 | 1,068 | FPP | Nav, SA, EE | 5 维度 |
| WorldLens | 26k | — | TPP | Navigation | 5 维度（驾驶） |
| **WBench** | **289** | **1,058** | **Both** | **All four** | **All five** |

### Table 2: 主要实验结果（文本驱动模型）

| 模型 | Aesthetic | Imaging | Navigation | Event Edit | Subject Action | Perspective | Setting Adh. |
|------|-----------|---------|-----------|------------|----------------|-------------|--------------|
| Seedance 1.5 | 61.0 | 69.3 | 68.0 | 80.4 | 80.0 | 45.0 | ~77.0 |
| **Wan 2.7** | 61.4 | 68.0 | 66.0 | **84.0** | 83.4 | 55.0 | **91.4** |
| **Kling 3.0** | **63.0** | 68.1 | 70.3 | 81.4 | **85.6** | 55.0 | 91.0 |
| YUME 1.5 | 58.7 | 63.3 | 72.0 | 57.8 | 47.0 | 16.7 | — |
| HY-Video 1.5 | 63.4 | **67.4** | 71.8 | 63.8 | 55.6 | 27.6 | — |
| LTX 2.3 | 57.9 | 61.0 | 67.6 | 53.0 | 51.8 | 25.0 | — |
| LongCat-Video | **66.5** | **69.6** | 63.1 | 50.4 | 48.4 | 18.3 | — |
| Kairos 3.0 | 59.9 | 62.7 | 65.1 | 46.8 | 41.4 | 13.3 | — |
| Cosmos 2.5 | 61.8 | 66.9 | 64.1 | 48.2 | 41.6 | 20.0 | — |

### Table 3: 相机控制 vs 动作条件模型导航评分

| 模型 | 类型 | Aesthetic | Navigation | Geometric Consist. |
|------|------|-----------|-----------|-------------------|
| HY-World 1.5 | Camera | 60.1 | **87.5** | 90.6 |
| LingBot-World | Camera | 66.9 | 79.8 | 92.7 |
| InSpatio-World | Camera | 64.4 | 72.8 | **93.8** |
| Fantasy-World | Camera | 63.0 | 72.1 | 80.6 |
| Astra | Camera | 48.6 | 67.7 | 64.7 |
| Happy Oyster | Action | 56.6 | 85.1 | — |
| Matrix-Game 3.0 | Action | 46.4 | 83.5 | — |
| Matrix-Game 2.0 | Action | 54.0 | 80.6 | — |
| Genie 3 | Action | 51.6 | 73.3 | — |
| Infinite-World | Action | 58.7 | 75.9 | — |
| HY-GameCraft | Action | 52.6 | 67.8 | — |

**关键发现**: 相机控制模型在几何一致性最高（93.8），但视角一致性最低（67.1），揭示了"显式姿态监督无法保证主体控制保真"的重要发现。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| WBench | 289 cases, 1,058 turns | 双视角、四类交互、多渲染风格 | 主要评测基准 |
| 人类标注验证集 | — | 10 个评估维度 | 人机对齐验证 |

### 评估工具

- **几何一致性**: [[Depth Anything 3]] 深度重投影
- **导航姿态估计**: [[MegaSaM]] 相机姿态估计
- **主体追踪**: [[SAM2]] 质心稳定性
- **感知相似度**: [[DreamSim]]
- **场景切割检测**: [[TransNetV2]]
- **语义一致性**: [[CLIP]] / [[DINOv2]] 双信号
- **物理合规评估**: 微调 Qwen3-VL-30B

### 关键发现

1. **导航独立性**: 导航与视频质量（r=-0.12）、一致性（r=-0.05）、物理合规（r=-0.15）近乎零相关，表明强渲染能力不保证可控性
2. **控制-一致性解耦**: 相机控制模型几何一致性最高（93.1），但视角一致性最低（67.1）
3. **场景遵从差距**: 文本驱动模型（Wan 91.4, Kling 91.0）远超世界模型（74.2-72.6）
4. **视角切换最难**: 所有文本驱动模型平均仅 30.7 分
5. **多轮性能衰减**: 导航 -33 分（轮次 1→4+）、事件编辑 -13 分、主体动作 -9 分
6. **物理跟随渲染**: 物理合规与视频质量强相关（r=0.84），但与导航不相关（r=-0.15）
7. **人机对齐**: 所有 10 个维度 Spearman ρ≥0.94，4 个维度达 ρ=1.00

---

## 批判性思考

### 优点
1. **全面性**: 首次在统一框架下覆盖双视角、四类交互、22 个自动化指标，填补领域空白
2. **统一导航协议**: 跨范式归一化使文本/相机/动作三类世界模型可公平对比，解决了领域长期存在的评测不可比问题
3. **高人机对齐**: 所有指标 ρ≥0.94 的强人类对齐验证，证明评测可靠性
4. **重要发现**: 揭示导航控制与渲染质量的根本性解耦，为未来研究指明关键方向

### 局限性
1. **规模相对较小**: 289 个测例相比 VBench（946）和 WorldScore（3000）偏少，可能影响统计稳健性
2. **缺乏训练成本分析**: 未考虑不同模型训练数据量和计算资源差异，难以评估算法本身的效率
3. **视角切换测例不足**: 仅占 6%（约 17 个），对该最难维度的评估统计功效较弱

### 潜在改进方向
1. 扩展测例规模，特别是视角切换类型，提高统计显著性
2. 加入长程多轮（>10 轮）测试，更全面评估世界模型的"记忆"能力
3. 引入任务完成度（goal-conditioned success rate）等面向任务的指标

### 可复现性评估
- [x] 代码开源（GitHub: meituan-longcat/WBench）
- [x] 数据集可获取（HuggingFace: meituan-longcat/wbench）
- [ ] 预训练模型（基于各商业模型 API，难以完全复现）
- [x] 评估细节完整（22 个指标均有说明）

---

## 关联笔记

### 基于
- [[VBench]]: 视频质量评测基准，被 WBench 扩展到交互维度
- [[WorldScore]]: 相机轨迹世界模型基准，WBench 在其基础上增加语义交互和第三人称视角

### 对比
- [[Omni-WorldBench]]: 同类多交互基准，但仅支持第一人称视角
- [[WorldLens]]: 自动驾驶专用评测，WBench 扩展至开放域

### 方法相关
- [[MegaSaM]]: 导航分数中用于相机姿态估计
- [[SAM2]]: 主体追踪一致性评估
- [[DreamSim]]: 感知相似度度量
- [[Depth Anything 3]]: 几何一致性评估
- [[VLM]]: 语义交互评估的核心工具

### 被评测模型
- [[Matrix-Game]]: 动作条件世界模型（Matrix-Game 2.0, 3.0）
- [[HYWorld2]]: 相机控制世界模型（HY-World 1.5）

---

## 速查卡片

> [!summary] WBench
> - **核心**: 首个覆盖双视角、四类交互的多轮交互式视频世界模型综合基准
> - **方法**: 289 测例 / 1,058 轮次 / 22 自动化指标 / 统一导航控制协议
> - **结果**: 无单一模型在所有维度领先；导航控制与渲染质量根本性解耦（r=-0.12）
> - **代码**: [github.com/meituan-longcat/WBench](https://github.com/meituan-longcat/WBench)

---

*笔记创建时间: 2026-05-28*

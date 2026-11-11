---
title: "SWAM: Scaling World Action Models with Cross-Modal Alignment"
method_name: "SWAM"
authors: [Yu Zhou, Xinyuan Xia, Xuankun Yang, Yinghao Yang, Wendi Chen, Bowen Liu, Jianhan Ma, Weicheng Duan, Jun Lv, Cewu Lu, Yanwei Li]
year: 2026
venue: arXiv
tags: [world-action-model, cross-modal-alignment, robot-manipulation, multi-source-data, flow-matching, bimanual-robot, data-scaling]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.12870v1
created: 2026-11-11
---

# 论文笔记：SWAM: Scaling World Action Models with Cross-Modal Alignment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University, Shanghai Innovation Institute, Noematrix Ltd., Westlake University, Tongji University |
| 日期 | November 2026 |
| 项目主页 | [char-lotte-xia.github.io/swam-official](https://char-lotte-xia.github.io/swam-official/) |
| 对比基线 | [[OpenWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.12870) / [Code](https://github.com/shaaaaron/SWAM) / [Dataset](https://huggingface.co/datasets/FreDERicia/SWAM-Pretrain) |

---

## 一句话总结

> SWAM 通过跨模态对齐（语言-视频-动作三级对齐）和 ~47K 小时多源数据的系统性扩展训练，让 [[World Action Model]] 在实体机器人操控任务中实现泛化，Hard 设置真实场景成功率达到 40%，超越最强基线 OpenWAM 的 23.33%。

---

## 核心贡献

1. **三级跨模态对齐框架**: 明确区分语言-视频对齐（保留预训练语义）与语言-视频-动作对齐（桥接操控能力），并发现语言-视频对齐的质量直接决定动作泛化性能。
2. **多源数据集成方案**: 构建 ~47,016 小时数据池，涵盖通用视频、第一视角（Ego）视频、UMI 演示、机器人演示四类来源，系统研究来源比例与动作监督的影响。
3. **信息流向解耦架构**: 基于 [[Mixture-of-Transformers]]（MoT）的双流设计，语言仅通过视频特征间接传入动作分支，防止动作学习破坏视频分支的语言语义。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]]（WAM）在获取机器人动作控制能力的同时，会**削弱**视频预训练阶段建立的语言-视频对齐关系。动作学习会干扰语义表示，导致模型在新场景（Hard 设置，Clean-to-Random）下泛化能力差。

### 现有方法的局限

- 已有工作（[[Fast-WAM]]、[[LingBot-Video]]、[[OpenWAM]] 等）注重架构设计和训练机制，但将视频预训练的语言-视频对齐视为固定不变的先验。
- 多源数据整合时缺乏对语言标注质量的系统性研究（许多机器人数据集存在标注简短、重复率高的问题）。
- UMI 演示等第三方数据中动作标注的完整性对策略学习的影响尚未被量化。

### 本文的动机

通过精细的语言标注（多粒度、环境信息）和分阶段数据引入，主动维护和强化语言-视频对齐；同时用第一视角视频和 UMI 演示弥补机器人演示在场景多样性和动作覆盖度上的不足，再通过持续预训练扩大规模。

---

## 方法详解

### 模型架构

SWAM 采用 **[[Mixture-of-Transformers]] 双流** 架构，基于 [[Wan2.2-TI2V-5B]] 初始化视频分支（动作分支随机初始化）：

- **输入**: 语言指令 $P$ + 视觉历史 $V_{0:n}$（三视角拼帧，256×320）+ 状态 $S_n$
- **视频分支**: 继承 Wan2.2-TI2V-5B，保持独立残差流；语言通过该分支的 Cross-Attention 进入
- **动作分支**: 随机初始化；语言**仅**通过视频特征间接传入（单向注意力），与视频分支共享 Masked Attention
- **信息流向**: 视频 → 动作（单向），语言 → 视频 → 动作
- **输出**: 每 Chunk 2 帧视频 latent + 48 步 [[Action Chunking|动作块]]（16 维双臂末端执行器格式）
- **总参数**: ~5B（Wan2.2-TI2V-5B 规模）

### 核心模块

#### 模块 1：语言-视频对齐（Language-Video Alignment）

**设计动机**: 利用 [[Cross-Modal Alignment|跨模态对齐]] 保留视频预训练的语义先验

**具体实现**:
- **过渡预训练阶段**: 先用通用视频（接近 Wan 原始分布）适配新帧数、分辨率和因果注意力，再引入具身数据
- **多粒度语言标注**（见 Table 3）: 三个维度独立组合——
  - *详细程度*: Brief（动词短语）/ Medium（单句含属性）/ Detailed（时序描述+手部动作）
  - *场景信息*: 是否包含背景、布局、初始状态、光照
  - *时间粒度*: Episode（整轨迹）vs. Subtask（带时间戳的子任务片段）
- **标注工具**: Qwen 3.8 Max（子任务分割）+ Seed 2.1 Pro（字幕生成）
- **机器人数据重标注**: 对原始标注进行改写（Rephrased）和详细化（Detailed + Env.），解决原始数据集中标注高度重复问题（145,430 条轨迹仅 592 条不同指令）

#### 模块 2：UMI 数据整合（Language-Video-Action Alignment）

**设计动机**: 利用 [[UMI]] 演示在保持场景多样性的同时提供 [[Action Chunking|动作标注]]，弥合语言-视频-动作三者的 gap

**具体实现**:
- 使用 RoboPocket（智能手机 + 手持自适应夹爪）采集，含头部视角和双腕视角
- 视觉-惯性追踪轨迹，外部编码器记录夹爪状态
- 共 411,754 条 UMI 轨迹（15,284.9 小时），全部视频均用；动作标签比例可调（消融研究验证：100% 优于 0%）

#### 模块 3：双流 Masked Attention

**设计动机**: 防止动作学习干扰视频分支的语言语义，使用 [[Teacher Forcing]] 训练

**具体实现**:
- 视频 token 和动作 token 保持分离的残差流
- 训练时用过去的干净视频 latent $h_t$ 作为 Context（Teacher Forcing），避免曝光偏差影响动作预测
- 推理时 Chunk 化生成：每次生成 2 帧视频 + 48 步动作

---

## 关键公式

### 公式 1：[[World Action Model|WAM 策略定义]]

$$
A_{n:n+K} \sim \pi_\theta(\cdot \mid V_{0:n}, P, S_n)
$$

**含义**: 给定视觉历史 $V_{0:n}$、语言指令 $P$ 和当前状态 $S_n$，预测未来 $K$ 步动作序列

**符号说明**:
- $A_{n:n+K}$: 从第 $n$ 步开始的 $K$ 步动作序列
- $V_{0:n}$: 时刻 0 至 $n$ 的视觉帧历史
- $P$: 语言指令
- $S_n$: 时刻 $n$ 的本体感知状态

### 公式 2：[[Cross-Modal Alignment|联合分布分解]]

$$
p_\theta(V, A \mid V_{0:n}, P, S_n) = p_\theta(V \mid V_{0:n}, P)\, p_\theta(A \mid V_{0:n+K}, P, S_n)
$$

**含义**: 联合视频-动作分布分解为先预测视频、再以视频为条件预测动作两个独立阶段，视频分支不依赖本体感知状态

**符号说明**:
- $p_\theta(V \mid V_{0:n}, P)$: 视频生成分支（无状态输入）
- $p_\theta(A \mid V_{0:n+K}, P, S_n)$: 动作生成分支（以预测视频为条件）

### 公式 3：[[Flow Matching|流匹配噪声轨迹]]

$$
\begin{aligned}
x_t^\tau &= (1-\tau)x_t + \tau\epsilon_v \\
A_t^\tau &= (1-\tau)A_t + \tau\epsilon_a
\end{aligned}
$$

**含义**: 在噪声时间步 $\tau$ 对视频 latent 和动作分别进行线性插值，构造训练所需的噪声样本

**符号说明**:
- $x_t^\tau$: 时刻 $t$、噪声级别 $\tau$ 下的视频 latent 噪声版本
- $A_t^\tau$: 时刻 $t$、噪声级别 $\tau$ 下的动作噪声版本
- $\epsilon_v, \epsilon_a$: 视频和动作的高斯噪声
- $\tau \sim \mathcal{U}(0, 1)$: 均匀采样的噪声时间步

### 公式 4：[[Flow Matching|视频与动作损失]]

$$
\begin{aligned}
\mathcal{L}_{\text{video}} &= \mathbb{E}\Big[\big\|u^v_\theta(x_t^\tau, h_t, p, \tau) - (\epsilon_v - x_t)\big\|_2^2\Big] \\
\mathcal{L}_{\text{action}} &= \mathbb{E}\Big[\big\|u^a_\theta(A_t^\tau, x_t^\tau, h_t, S_t, p, \tau) - (\epsilon_a - A_t)\big\|_2^2\Big]
\end{aligned}
$$

**含义**: 视频分支和动作分支各自预测流速（噪声与样本之差），用 MSE 监督

**符号说明**:
- $u^v_\theta$: 视频分支的流速预测网络
- $u^a_\theta$: 动作分支的流速预测网络（额外接收视频 latent 和状态）
- $h_t$: [[Teacher Forcing|过去干净视频 latent]]（Teacher Forcing Context）
- $p$: 语言嵌入

### 公式 5：[[Flow Matching|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}}
$$

**含义**: 视频损失和动作损失等权相加，不缺少权重超参数，两个分支联合优化

### 公式 6：[[Partial Success Rate|SR 与 PSR 评估指标]]

$$
\text{PSR} = \frac{1}{|T|}\sum_{i \in T} \frac{\text{completed subtasks}_i}{\text{total subtasks}_i}, \quad \text{SR} = \mathbb{1}[\text{all subtasks completed}]
$$

**含义**: SR 是全任务成功率（所有子任务完成），PSR 是子任务平均完成率（更细粒度）

**符号说明**:
- $T$: 测试 Episode 集合
- $\text{SR}$: 成功率（全任务完成 = 1，否则 = 0 的均值）
- $\text{PSR}$: 部分成功率（子任务完成比例均值）

---

## 关键图表

### Figure 1: Teaser / 方法概览

![Figure 1](https://arxiv.org/html/2611.12870v1/teaser.png)

**说明**: SWAM 的整体动机示意图。多源监督（通用视频、Ego 视频、UMI 演示、机器人演示）协同作用，在保留视频预训练语言-视频关联的前提下，将能力延伸到机器人控制。~47K 小时数据池覆盖四类来源。

### Figure 2: 数据构成与模型架构示意

![Figure 2](https://arxiv.org/html/2611.12870v1/intro_data_model.png)

**说明**: (a) 多源数据的构成分布（通用视频、Ego、UMI、机器人演示）；(b) [[World Action Model]] 整体架构，视频分支与动作分支相互链接，三视角输入拼成一帧（256×320），语言仅通过视频分支的 Cross-Attention 进入。

### Figure 3: 模型详细架构与注意力掩码

![Figure 3](https://arxiv.org/html/2611.12870v1/model_architecture.png)

**说明**: 展示训练时（带 [[Teacher Forcing|干净视频 Context]]）和推理时（Chunk 化生成）的注意力掩码设计。视频 token 和动作 token 保持分离残差流，单向信息流（视频→动作）防止动作梯度干扰视频语义。

### Figure 4: 多源数据与数据精选流水线

![Figure 4](https://arxiv.org/html/2611.12870v1/data_curation_pipeline.png)

**说明**: 多源数据示例（四种来源的视觉差异）以及数据精选流水线，包括视觉质量检查（模糊、亮度、遮挡）、动作过滤（NaN、关节跳变、末端执行器跳变）、子任务分割（Qwen 3.8 Max）和多级标注（Seed 2.1 Pro）。

### Figure 5: 预训练规模扩展曲线（仿真）

> 图片暂无在线 URL，见论文 Figure 5

**说明**: 横轴为预训练视频时长（小时），纵轴为 RoboTwin 2.0 仿真 Easy 和 Hard 成功率。Hard 成功率从 0 小时的 34.56% 随规模增加持续提升，峰值达 59.60%。

### Figure 6: 真实场景成功率对比（Hard 设置）

> 图片暂无在线 URL，见论文 Figure 6

**说明**: Hard 设置下三个真实任务的平均成功率柱状图，对比各基线。SWAM 以 40.00% 领先 OpenWAM（23.33%）和 π0.5（20.00%）。

### Figure 7: 真实场景跨模型执行对比

![Figure 7](https://arxiv.org/html/2611.12870v1/real_world_execution_across_models.png)

**说明**: 三个任务（杯子叠放、毛巾折叠、礼品入盒）在各模型下的执行帧序列对比。SWAM 能稳定完成任务，部分基线在关键子步骤失败。

### Figure 8: 泛化示例（水果摆放、彩虹叠杯）

![Figure 8](https://arxiv.org/html/2611.12870v1/real_world_external_vs_head.png)

**说明**: 展示 SWAM 在未见过对象和场景上的泛化能力。头部视角与外部摄像头视角对比，包含水果摆放（新物体）和彩虹叠杯（新颜色）任务。

### Figure 9: 真实场景评估布局（Easy & Hard）

![Figure 9](https://arxiv.org/html/2611.12870v1/layouts_black.png)

**说明**: Easy 和 Hard 设置的 10 种布局示意。Easy（Clean-to-Clean）布局固定、光照/背景统一；Hard（Clean-to-Random）布局随机、加入背景干扰、物体高度/光照变化，所有模型使用相同布局。

### Figure 10: 三个真实任务的子任务分解

![Figure 10](https://arxiv.org/html/2611.12870v1/subtasks_black.png)

**说明**: 三个任务的子任务图示——杯子叠放（7 个子任务：3 层金字塔）、毛巾折叠（4 个子任务：展开+两次折叠）、礼品入盒（2 个子任务：放置+释放）。

### Figure 11: 真实场景逐任务对比（附录）

> 图片暂无在线 URL，见论文 Figure 11

**说明**: Easy 和 Hard 设置下各任务以及三任务平均的真实世界成功率，逐模型横向对比。

### Figure 12: 真实场景性能 vs 预训练规模（附录）

> 图片暂无在线 URL，见论文 Figure 12

**说明**: 横轴为预训练视频时长（22.5K → 90K 小时），纵轴为真实场景 Hard SR 和 PSR。0 小时模型无法完成任何任务；90K 小时模型达到 48.89% SR 和 61.43% PSR。

### Figure 13: 杯子叠放完整执行序列

![Figure 13](https://arxiv.org/html/2611.12870v1/real_world_episode_stack_cups.png)

**说明**: Hard 设置下 SWAM 完成三层杯子金字塔叠放的完整帧序列。7 个子任务全部完成。

### Figure 14: 毛巾折叠完整执行序列

![Figure 14](https://arxiv.org/html/2611.12870v1/real_world_episode_fold_towel.png)

**说明**: Hard 设置下 SWAM 完成毛巾展开→两次折叠的完整帧序列。4 个子任务全部完成。

### Figure 15: 礼品入盒完整执行序列

![Figure 15](https://arxiv.org/html/2611.12870v1/real_world_episode_giftbox.png)

**说明**: Hard 设置下 SWAM 完成礼品放入盒子并释放的完整帧序列。2 个子任务全部完成。

---

### Table 1: 预训练数据池构成

| 来源 | 时长（小时） | Episodes | 备注 |
|------|------------|---------|------|
| 通用视频（Panda-70M, OpenVid） | 10,909.6 | — | 语言-视频监督 |
| 第一视角视频（Ego4D, EgoVerse, EgoSuite） | 10,986.3 | — | 语言-视频监督 |
| UMI 演示（RoboPocket） | 15,284.9 | 411,754 | 视频+动作标注 |
| 机器人演示（AgiBot, RoboCOIN, RoboMIND, InternData-A1, Galaxea, ABC-130K） | 9,836.0 | 768,356 | 视频+动作标注 |
| **合计** | **47,016.7** | **4,407,996** | — |

**说明**: UMI 演示规模最大（15K+ 小时），机器人演示经过严格质量过滤后保留 768K 条轨迹。

### Table 3: 语言标注维度矩阵

| 时间粒度 \ 详细程度 | Brief（动词短语） | Medium（单句） | Detailed（时序+手部） |
|--------------------|----------------|--------------|-------------------|
| Episode（整轨迹）   | 基线设置 | — | ✓ 实验 |
| Subtask（带时间戳） | ✓ 实验 | ✓ 实验 | ✓ 实验（+Env 最优） |

**说明**: 此外可选择是否包含场景信息（Env.：背景、布局、初始状态、光照）。最佳组合为 Subtask/Detailed + Env.。

### Table 4: 语言-视频对齐消融（固定架构和数据，仅变标注）

| 标注方式 | Easy SR (%) | Hard SR (%) |
|---------|------------|------------|
| Episode / Brief | 81.36 | 44.96 |
| Episode / Detailed | 82.00 | 51.60 |
| Subtask / Brief | 83.60 | 47.20 |
| Subtask / Medium | 82.08 | 49.60 |
| Subtask / Detailed | 83.68 | 53.52 |
| Subtask / Detailed + Env. | **83.92** | **55.04** |
| Subtask / Detailed + Env.（打乱配对） | — | 52.00 |

**关键发现**: Hard 成功率对标注质量高度敏感（+10pp），Easy 变化较小。打乱语言-视频配对后 Hard 下降 3pp，证明准确对齐而非仅"有标注"才有效。

### Table 5: 机器人数据标注质量消融

| 机器人标注方式 | Easy SR (%) | Hard SR (%) |
|--------------|------------|------------|
| 原始标注（大量重复） | 84.64 | 50.80 |
| 改写标注（Rephrased） | 86.08 | 54.76 |
| 详细标注 + 环境信息 | **86.48** | **56.16** |

**关键发现**: 机器人演示的原始指令集中度极高（145,430 条仅 592 种），改写后 Hard +3.96pp，详细化后额外 +1.4pp。

### Table 6: UMI 动作标注覆盖率消融

| UMI 动作标注比例 | Easy SR (%) | Hard SR (%) |
|----------------|------------|------------|
| 0%（仅视频） | — | 46.72 |
| 25% | — | 48.xx |
| 50% | — | 49.xx |
| 75% | — | 50.xx |
| **100%** | — | **51.28** |
| 对比：直接从 Wan 后训练 | — | 34.56 |

**关键发现**: UMI 动作标注的完整性单调提升 Hard 成功率（+4.56pp），且相比直接从 Wan 后训练（34.56%）高出 ~17pp，说明 UMI 阶段的动作预训练至关重要。

### Table 7: 多源数据引入顺序和比例消融

| 配置 | Easy SR (%) | Hard SR (%) |
|------|------------|------------|
| 无过渡预训练 | — | 53.84 |
| + 25K 步通用视频过渡预训练 | — | 56.48 |
| 仅 Ego | — | 低于 1:1 |
| 仅 UMI | — | 低于 1:1 |
| Ego + UMI（1:1） | — | 53.84 |

**关键发现**: 加入 25K 步通用视频过渡预训练（使模型适应新分辨率/帧率后再引入具身数据）可额外提升 +2.64pp。

### Table 8: Ego-UMI 采样比例消融（20K 步）

| Ego : UMI | Easy SR (%) | Hard SR (%) |
|-----------|------------|------------|
| 1 : 0（纯 Ego） | 77.04 | 42.56 |
| 2 : 1 | 82.72 | 52.64 |
| 1 : 1 | 82.64 | **53.84** |
| 1 : 2 | 83.20 | 53.28 |
| 0 : 1（纯 UMI） | 83.28 | 51.28 |

**关键发现**: 均等混合（1:1）在 Hard 成功率上最优；纯 Ego 最差（-11.28pp），说明 UMI 动作监督不可或缺；纯 UMI 也弱于混合，Ego 场景多样性有价值。

### Table 9: 模型架构消融

| 架构变体 | Easy SR (%) | Hard SR (%) |
|---------|------------|------------|
| 单流（Single-Stream） | 86.48 | 52.56 |
| 标准双流（Standard Dual-Stream） | 84.96 | 52.48 |
| **本文设计（语言仅经视频特征）** | **84.80** | **57.36** |
| 本文设计 + 原生视觉输入（无额外编码器） | 83.44 | 49.52 |
| 本文设计 + CLIP 特征 | — | 42.96 |
| 本文设计 + DINOv2 特征 | — | 41.xx |
| 本文设计 + SigLIP2 特征 | — | 40.xx |
| 本文设计 + Qwen3-VL 特征 | — | 39.76 |

**关键发现**: 语言仅通过视频特征间接传入动作分支（本文设计）在 Hard 上高出单流 +4.8pp；添加额外视觉编码器（CLIP/DINO 等）反而大幅下降，说明这些特征干扰了 Wan 的预训练视觉表示。

### Table 10: 两阶段机器人数据预训练

| 配置 | Easy SR (%) | Hard SR (%) | 背景 | 光照 | 杂乱 | 高度 |
|------|------------|------------|------|------|------|------|
| 仅阶段 1（Ego+UMI，75K 步） | 84.80 | 57.36 | — | — | — | — |
| 阶段 1 + 阶段 2（机器人数据，25K 步） | — | **61.52** | ↑ | ↑ | ↑ | ↑ |

**关键发现**: 加入 Stage 2 机器人演示数据后，Hard 成功率再提升 +4.16pp，且在所有扰动维度（背景、光照、杂乱、高度）均有提升。

### Table 11: 与基线方法的主要对比

| 方法 | RoboTwin Easy (%) | RoboTwin Hard (%) | RoboTwin Avg. (%) | Real Easy SR (%) | Real Easy PSR (%) | Real Hard SR (%) | Real Hard PSR (%) |
|------|-----------------|-----------------|-----------------|----------------|-----------------|----------------|-----------------|
| StarVLA | 60.03 | 14.06 | 37.05 | 31.11 | 36.87 | 4.44 | 21.55 |
| π0.5 | 70.66 | 46.00 | 58.33 | 50.00 | 71.87 | 20.00 | 37.78 |
| Xiaomi Robotics-0 | 62.86 | 18.22 | 40.54 | — | — | — | — |
| X-VLA | 67.96 | 20.95 | 44.46 | — | — | — | — |
| Fast-WAM | 77.76 | 1.92 | 39.84 | 35.56 | 49.33 | 6.67 | 14.52 |
| LingBot-VA | 77.04 | 36.16 | 56.60 | 41.11 | 53.61 | 10.00 | 24.01 |
| OpenWAM | **89.30** | 48.70 | 69.00 | 37.78 | 46.35 | 23.33 | 31.67 |
| X-WAM | 69.98 | 25.84 | 47.91 | — | — | — | — |
| 4D-WAM | 81.50 | 41.80 | 61.65 | — | — | — | — |
| **SWAM（本文）** | 88.10 | **61.56** | **74.83** | **56.67** | **68.57** | **40.00** | **50.36** |

**关键发现**: SWAM 在 Hard 设置下大幅领先所有方法（+12.86pp vs OpenWAM）；Easy 略低于 OpenWAM（-1.2pp）但总体平均最高。真实世界 Hard SR 是 OpenWAM 的 1.7 倍。

### Table 12: 优化器超参数（所有阶段共享）

| 超参数 | 值 |
|--------|---|
| 优化器 | [[AdamW]] |
| $\beta$ | (0.95, 0.999) |
| $\epsilon$ | $10^{-8}$ |
| Weight Decay | $10^{-5}$ |
| Gradient Clip | 1.0 |
| Warmup 比例 | 5% |
| 随机种子 | 42 |

### Table 13: 预训练与后训练配置对比

| 阶段 | 学习率 | Batch Size | 步数 | 数据来源 |
|------|--------|-----------|------|---------|
| 预训练（消融） | $4 \times 10^{-5}$ | 256 | 100K | 多源 |
| 预训练（最终/扩展） | $4 \times 10^{-5}$ | 512 | 100K+ | 多源 |
| 仿真后训练 | — | 128 | 20K | RoboTwin Clean（2,500 条） |
| 真实后训练 | — | — | 50K/任务 | 任务专属 |

### Table 14: 真实世界后训练数据规模

| 任务 | Episodes | 时长（小时） | 子任务数 |
|------|---------|------------|--------|
| 杯子叠放（Cup Stacking） | 2,748 | 25.8 | 7 |
| 毛巾折叠（Towel Folding） | 2,852 | 15.1 | 4 |
| 礼品入盒（Gift-in-Box） | 4,399 | 21.7 | 2 |

**说明**: 评估时每个设置使用 10 种布局 × 3 次试验 = 30 次试验/任务。

---

## 实验

### 数据集

| 数据集/来源 | 规模 | 用途 |
|-----------|------|------|
| [[RoboTwin 2.0]] | 50 任务，每任务 50 Clean 演示 | 仿真后训练 + 评估 |
| AgiBot / RoboCOIN / RoboMIND / InternData-A1 / Galaxea / ABC-130K | 9,836 小时 | 机器人演示预训练 |
| Ego4D / EgoVerse / EgoSuite | 10,986 小时 | 第一视角视频预训练 |
| Panda-70M / OpenVid | 10,910 小时 | 通用视频过渡预训练 |
| UMI（RoboPocket） | 15,285 小时 | UMI 演示预训练 |
| Flexiv Rizon 4 任务数据 | ~62.6 小时（3 任务合计） | 真实后训练 |

### 实现细节

- **Backbone**: [[Wan2.2-TI2V-5B]]（视频分支初始化）
- **动作表示**: 16 维双臂末端执行器格式（每手：3D 位置 + wxyz 四元数 + 夹爪值），每源独立 q1/q99 归一化
- **视觉输入**: 三视角（头部 + 双腕）拼成 256×320 单帧
- **推理 Chunk**: 2 帧视频 latent + 48 步动作
- **优化器**: [[AdamW]]，$lr=4 \times 10^{-5}$，详细参数见 Table 12
- **硬件**: 多 GPU（具体型号未在摘要中列出）
- **语言标注**: Qwen 3.8 Max（分割）+ Seed 2.1 Pro（字幕）

### 可视化结果

SWAM 在 Hard 设置下对背景干扰、光照变化、随机杂乱物体和桌面高度变化均保持鲁棒。毛巾折叠和礼品入盒任务中，模型能追踪精细动作；水果摆放、彩虹叠杯等 zero-shot 泛化也表现良好。

---

## 批判性思考

### 优点

1. **系统性消融分析**: 对语言标注粒度、数据来源比例、架构设计的消融实验非常详尽，结论有充分实验支撑
2. **真实世界评估严格**: 10 布局 × 3 试验的 Hard 设置真实反映了部署难度，PSR 指标也能衡量部分完成能力
3. **数据工程实用**: 多级标注、去重改写、动作质量过滤等工程细节对社区有很强的参考价值

### 局限性

1. **推理能力受限**: 语言仅通过视频特征间接传入动作分支，可能限制需要复杂推理（如多步规划、条件分支）的任务
2. **长时程记忆不足**: 当前 Context 仅来自 Teacher Forcing 的近期视频 latent，缺乏跨 Episode 的长时程记忆机制
3. **计算成本未报告**: 47K 小时预训练的具体 GPU 时和推理延迟未量化，实用部署成本不明

### 潜在改进方向

1. 引入记忆机制（如外部记忆、时序压缩）支持长时程任务
2. 在动作分支增加层次规划模块（高层语言规划 → 底层动作生成）
3. 探索更少真实演示的 few-shot 设置

### 可复现性评估

- [x] 代码开源（[github.com/shaaaaron/SWAM](https://github.com/shaaaaron/SWAM)）
- [x] 数据集公开（HuggingFace）
- [x] 训练细节完整（Table 12/13，附录 A/B/C）
- [x] 仿真 Benchmark 公开（RoboTwin 2.0）

---

## 关联笔记

### 基于

- [[Wan2.2-TI2V-5B]]: 视频分支初始化权重，5B 参数 Text-to-Image-to-Video 模型
- [[Flow Matching]]: 视频和动作联合生成的训练目标
- [[UMI]]: 人手持演示采集系统，提供 15K+ 小时动作标注数据

### 对比

- [[OpenWAM]]: 最强基线，Easy 略高但 Hard 差 12.86pp，真实场景差 16.67pp
- [[Fast-WAM]]: Hard 仅 1.92%（极差泛化），说明仅专注速度的 WAM 泛化性不足
- [[LingBot-Video]]: Hard 36.16%，说明语言对齐对 Hard 场景至关重要

### 方法相关

- [[World Action Model]]: 本文核心范式，联合预测视频和动作
- [[Mixture-of-Transformers]]: 双流架构设计基础
- [[Action Chunking]]: 48 步动作 Chunk 的输出方式
- [[Teacher Forcing]]: 训练时使用干净视频 latent 作为 Context
- [[Cross-Modal Alignment]]: 本文核心贡献，三级对齐（语言-视频-动作）

### 硬件/数据相关

- [[RoboTwin 2.0]]: 仿真评估 Benchmark（50 任务）
- [[Bimanual Manipulation]]: 双臂末端执行器控制范式
- [[UMI]]: 数据采集平台（RoboPocket）

---

## 速查卡片

> [!summary] SWAM: Scaling World Action Models with Cross-Modal Alignment
> - **核心**: 三级跨模态对齐（语言-视频-动作）+ ~47K 小时多源数据扩展训练
> - **方法**: MoT 双流架构（视频分支初始化自 Wan2.2），语言仅通过视频特征传入动作分支；Ego+UMI 1:1 混合预训练；多粒度语言标注
> - **结果**: RoboTwin Hard 61.56%（+12.86pp vs OpenWAM）；真实 Hard SR 40.00%（+16.67pp vs OpenWAM）
> - **代码**: [github.com/shaaaaron/SWAM](https://github.com/shaaaaron/SWAM)

---

*笔记创建时间: 2026-11-11*

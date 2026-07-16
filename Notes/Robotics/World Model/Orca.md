---
title: "Orca: The World is in Your Mind"
method_name: "Orca"
authors: [Yihao Wang, Yuheng Ji, Mingyu Cao, Yanqing Shen, Runze Xiao]
year: 2026
venue: arXiv
tags: [world-model, next-state-prediction, embodied-ai, multimodal, vla, foundation-model, video-pretraining]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.30534
created: 2026-07-16
---

# 论文笔记：Orca: The World is in Your Mind

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beijing Academy of Artificial Intelligence (BAAI) |
| 日期 | June 2026 |
| 项目主页 | [orca-wm.github.io](https://orca-wm.github.io/) |
| 对比基线 | [[V-JEPA]], [[pi0.5\|π₀.₅]], [[Qwen3.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.30534) |

---

## 一句话总结

> Orca 是一个通用世界基础模型，通过"无意识学习"（视频密集状态转移）和"有意识学习"（语言条件化事件）统一学习世界隐空间表示，冻结骨干后分别接语言、视觉、动作轻量 Decoder 实现跨模态下游任务。

---

## 核心贡献

1. **[[Next-State Prediction|下一状态预测范式]]**: 将建模目标从 next-token/next-frame/next-action 统一为 Next-State-Prediction，学习通用世界隐空间 $S$
2. **双范式预训练编码器**: [[Unconscious Learning|无意识学习]]（视频自然状态转移）+ [[Conscious Learning|有意识学习]]（语言条件事件）互补覆盖密集与稀疏动态
3. **统一 Encoder-Decoder 框架**: 冻结预训练骨干，通过轻量 Readout 模块支持文本生成、图像预测、机器人动作三类下游任务

---

## 问题背景

### 要解决的问题

现有 AI 模型各自只优化孤立预测任务：LLM 做 Next-Token-Prediction，视频生成模型做 Next-Frame-Prediction，VLA 做 Next-Action-Prediction。这三类系统都不具备统一的世界状态表示能力，无法跨模态共享知识。

### 现有方法的局限

- [[V-JEPA]] 等 JEPA 类方法在隐空间做预测，但仅针对视觉预训练，缺乏语言条件化能力
- [[VLA]] 模型直接从视觉-语言映射到动作，没有显式的世界模型
- 生成式视频模型（如 [[Sora]]）追求像素级重建，监督信号冗余且计算成本高
- 没有一个模型能同时支持文本问答、未来帧预测和机器人动作生成

### 本文的动机

作者认为"智能"应建立在对世界状态的理解之上，而非分别优化各类预测头。类比人脑：无意识的感知流（视频）让人学习自然动态，有意识的语言事件让人学习因果关系，两者共同构成世界模型。更强的预训练世界隐空间应当直接带来更强的下游任务 Readout。

---

## 方法详解

### 模型架构

![Figure A1 - 概念对比](https://arxiv.org/html/2606.30534v2/x9.png)

**说明**: Orca 将建模目标从被动的 next-token/next-frame/next-action 预测转向主动的 Next-State-Prediction。左侧是传统模型的分离预测头，右侧是 Orca 统一的世界隐空间范式。

Orca 采用 **Encoder-Decoder** 架构：
- **输入**: 视频帧序列 $v_t$ + 语言事件标注 $e_{t+\Delta}$ + VQA 问答对
- **Backbone**: 预训练 [[VLM]]（Qwen3.5-4B / Qwen3.5-0.8B），已对齐语言与视觉空间
- **核心模块**: [[World Latent Space|世界隐空间]] $S_t$，通过可学习查询向量 $Q_1, Q_2$ 接口化
- **输出 Decoder**: 语言 LM Head（复用）/ 视觉 MLP+LoRA+SD3.5 / 动作 [[DiT]] Action Expert
- **总参数**: 4B（主模型）或 0.8B（小模型）

![Figure 1 - 整体框架](https://arxiv.org/html/2606.30534v2/x1.png)

**说明**: Orca 整体 Encoder-Decoder 框架。Encoder 通过无意识学习和有意识学习两条路径更新世界隐空间；Decoder 冻结骨干，只训练轻量任务头。

### 核心模块

#### 模块1：无意识学习（Unconscious Learning）

**设计动机**: 人在婴儿期通过连续感知流（无标注视频）获得对物理世界的直觉理解，对应[[Self-Supervised Learning|自监督]]的密集状态转移学习

**具体实现**:
- 输入连续视频帧，用 [[ViT]] 编码当前帧为视觉 token $v^l_t$
- 通过可学习查询 $Q_1$ 聚合上下文，预测下一帧的 ViT 隐空间表示 $\hat{v}^l_{t+1}$
- 无需语言标注，学习自然的密集状态转移概率

$$
S_{t+\Delta} \sim p_\Theta^u(S_{t+\Delta} \mid S_t,\, z_t)
$$

其中 $z_t$ 为隐式动态因子（无法直接观测的物理规律）

#### 模块2：有意识学习（Conscious Learning）

**设计动机**: 人通过语言描述的因果事件（"因为X，所以Y"）获得高层语义理解，对应[[Language-Conditioned|语言条件化]]的稀疏状态转移学习

**具体实现**:
- 输入事件前/后帧对 + 语言事件描述 $e_{t+\Delta}$
- 通过可学习查询 $Q_2$ 在语言条件下预测双向状态转移（事件前←→事件后）
- 同时训练 [[VQA]] 问答头强化语义理解

$$
S_{t+\Delta} \sim p_\Theta^c(S_{t+\Delta} \mid S_t,\, z_t,\, e_{t+\Delta})
$$

#### 模块3：下游 Readout 模块

![Figure 4 - 下游 Readout 架构](https://arxiv.org/html/2606.30534v2/x4.png)

**说明**: 三种 Readout 的结构对比。语言直接复用 LM Head（零参数新增）；视觉通过 MLP+LoRA 驱动 SD3.5 多步去噪；动作通过 DiT Action Expert 生成 30 步动作块。

**语言 Readout**: 直接复用 VLM 的 LM Head，无额外参数，冻结后直接接文本问答

**视觉 Readout**:
- MLP Adapter（556.9M 参数）+ [[LoRA]]（rank=32）映射隐状态到 [[Stable Diffusion 3.5|SD3.5]] 条件空间
- 多步去噪生成目标分辨率 768×768 预测图像
- 目标：物理合理性 + 场景一致性，非照片写实

**动作 Readout**:
- [[DiT]]-based Action Expert（8 块，hidden=1024，heads=12）
- 输入：世界隐状态查询 + 含噪动作嵌入 + 机器人本体感觉
- 使用 [[Flow Matching]] 损失，推理时 4 步去噪
- 输出 30 步 Action Chunk（[[Action Chunking|动作块]]）

---

## 关键公式

### 公式1：[[World State Transition|通用状态转移]]

$$
S_{t+\Delta} \sim p_\Theta(S_{t+\Delta} \mid S_t,\, z_t,\, c_t), \quad \Delta \in \mathbb{Z}_{\neq 0}
$$

**含义**: Orca 的核心建模对象，预测在隐式动态 $z_t$ 和显式条件 $c_t$（语言/事件）下的下一世界状态

**符号说明**:
- $S_t$: 时刻 $t$ 的世界隐状态（World Latent）
- $z_t$: 不可直接观测的隐式动态（物理规律、惯性等）
- $c_t$: 显式条件（语言描述的事件，无意识学习时为空）
- $\Delta \in \mathbb{Z}_{\neq 0}$: 时间偏移量，支持前向和后向预测

### 公式2：[[Latent Matching Loss|隐空间匹配损失]]

$$
\ell_{\text{lat}}(\hat{v}^l, v^l) = 0.1 \|\hat{v}^l - v^l\|^2_2 + 0.9\left(1 - \frac{\langle \hat{v}^l, v^l \rangle}{\|\hat{v}^l\|_2 \|\hat{v}^l\|_2}\right)
$$

**含义**: 预测隐状态与目标 ViT 隐状态之间的组合损失，兼顾距离和方向

**符号说明**:
- $\hat{v}^l$: 模型预测的 ViT 隐层表示
- $v^l$: 真实帧的 ViT 隐层表示（监督信号）
- 第一项（10%）: L2 距离，约束量级
- 第二项（90%）: Cosine 相似度，约束方向

### 公式3：[[Observation-Only Loss|无意识学习损失]]

$$
\mathcal{L}_{\text{obs}} = \mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{t+1},\, v^l_{t+1}\right)\right]
$$

**含义**: 纯视频（无语言条件）的下一帧 ViT 隐状态预测损失

### 公式4：[[Event-Conditioned Loss|有意识学习损失]]

$$
\mathcal{L}_{\text{evt}} = \frac{1}{2}\,\mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{\text{prev}},\, v^l_{\text{prev}}\right) + \ell_{\text{lat}}\left(\hat{v}^l_{\text{next}},\, v^l_{\text{next}}\right)\right]
$$

**含义**: 语言事件条件下的双向状态转移损失（预测事件前后两帧）

**符号说明**:
- $\hat{v}^l_{\text{prev}}$: 预测的事件前帧隐状态
- $\hat{v}^l_{\text{next}}$: 预测的事件后帧隐状态

### 公式5：[[Pre-training Loss|预训练总损失]]

$$
\mathcal{L} = \lambda_{\text{obs}} \cdot \mathcal{L}_{\text{obs}} + \lambda_{\text{evt}} \cdot \mathcal{L}_{\text{evt}} + \lambda_{\text{vqa}} \cdot \mathcal{L}_{\text{vqa}}
$$

**含义**: 三个训练目标的加权组合，无意识学习权重低（下游任务多依赖语言条件），有意识学习权重最高

**符号说明**:
- $\lambda_{\text{obs}} = 0.1$: 纯观测状态转移权重（较低，防止过拟合无标注视频噪声）
- $\lambda_{\text{evt}} = 0.5$: 事件条件状态转移权重（最高，核心学习目标）
- $\lambda_{\text{vqa}} = 0.4$: VQA 语义理解权重（保持语言对齐）

---

## 关键图表

### Figure 2：Encoder 总览

![Figure 2 - Encoder 总览](https://arxiv.org/html/2606.30534v2/x2.png)

**说明**: 无意识学习（左）从连续视频预测下一帧 ViT 隐状态；有意识学习（右）在语言事件条件下双向预测前后帧。两路共享骨干，通过不同查询向量 $Q_1, Q_2$ 区分。

### Figure 3：预训练数据构成

![Figure 3 - 预训练数据](https://arxiv.org/html/2606.30534v2/x3.png)

**说明**: 数据三角形：A 视频数据支持观测状态转移，A+B 事件数据支持事件条件转移，C VQA 数据支持语义理解。125K 小时视频 + 160M 事件标注 + 11.5M VQA。

### Figure 5 & 6：Scaling 曲线

![Figure 5 - Loss Scaling](https://arxiv.org/html/2606.30534v2/x5.png)

**说明**: 预训练 Loss 随模型规模和数据规模单调下降，验证范式的可扩展性。

![Figure 6 - 下游性能 Scaling](https://arxiv.org/html/2606.30534v2/x6.png)

**说明**: 三类下游任务（文本/图像/动作）性能随模型规模和数据量一致提升，印证"更强预训练 → 更强 Readout"核心假设。

### Figure 7：图像预测视觉对比

![Figure 7 - 图像预测对比](https://arxiv.org/html/2606.30534v2/x7.png)

**说明**: Orca 在真实世界图像预测中对场景一致性和物理合理性的保持明显优于 FLUX.2-klein，尽管图像写实度相近。

### Figure 8：机器人故障恢复对比

![Figure 8 - 机器人恢复](https://arxiv.org/html/2606.30534v2/x8.png)

**说明**: 在铲糖任务中，Orca 在多次抓取失败后能够自适应恢复并最终成功，而 π₀.₅ 持续陷入失败循环。体现世界模型对异常状态的内在理解。

### Figure C1：查询向量实现细节

![Figure C1 - 查询实现](https://arxiv.org/html/2606.30534v2/x10.png)

**说明**: $Q_1$（无意识）和 $Q_2$（有意识）作为可学习查询向量，从零开始训练，作为状态转移预测的接口层。

### Table 1：文本生成基准测试

| 模型 | 规模 | MVBench | TemporalBench | 3DSRBench | SWITCH | **平均** |
|------|------|---------|---------------|-----------|--------|---------|
| V-JEPA 2.1 | 10B | 75.4 | 28.5 | — | — | — |
| Emu3 | 8B | 35.2 | 9.5 | 39.1 | 38.0 | 30.4 |
| Emu3.5 | 34B | 39.5 | 9.5 | 31.3 | 38.9 | 29.8 |
| Qwen3.5 (0.8B) | 0.8B | 52.7 | 19.1 | 21.8 | 38.8 | 33.1 |
| Qwen3.5 (4B) | 4B | 67.1 | 25.2 | 48.1 | 42.8 | 46.7 |
| **Orca (4B)** | **4B** | **65.3** | **34.2** | **52.1** | **55.6** | **51.8** |

**关键发现**: Orca-4B 平均分 51.8 超越同规模基础 Qwen3.5-4B（46.7）+5.1 分，在时序推理（TemporalBench +9.0）和空间状态（SWITCH +12.8）提升最显著，但 MVBench 略低（-1.8，因部分 MVBench 测试非状态转移能力）

### Table 2：能力维度分析

| 能力维度 | Qwen3.5-4B | Orca-4B | 提升 |
|---------|-----------|---------|------|
| 状态转移 | 51.86% | 64.13% | **+12.27%** |
| 常识推理 | 57.76% | 62.95% | +5.19% |
| 空间关系 | 54.68% | 55.25% | +0.57% |
| 动态运动 | 57.03% | 65.55% | **+8.52%** |

**关键发现**: Orca 在与世界状态直接相关的"状态转移"和"动态运动"维度提升最大，"空间关系"无明显改变——符合预期，因空间理解主要来自 VLM 的视觉预训练。

### Table 3：图像预测（PRICE-V0.1 基准）

| 模型 | 规模 | Gemini 3.1 | GPT 5.4 | Doubao-Seed | Gemma 4 | **平均** |
|------|------|-----------|---------|------------|---------|---------|
| OmniGen2 | 3+4B | 24.6 | 46.8 | 41.4 | 45.5 | 39.6±10.2 |
| FLUX.1-Kontext | 12B | 21.6 | 46.9 | 42.7 | 52.5 | 40.9±13.5 |
| FLUX.2 [klein] | 4+4B | 29.7 | 64.6 | 60.0 | 70.2 | 56.1±18.1 |
| **Orca (4B)** | **4+2B** | **44.0** | **67.9** | **61.0** | **66.3** | **59.8±10.9** |

**关键发现**: Orca 以更小模型（4+2B vs 12B FLUX）取得最高平均分 59.8，且方差最小（±10.9），说明生成质量更稳定；在 Gemini 3.1 评估维度（物理合理性）上领先最多。

### Table 4：真实机器人动作（OOD 环境）

| 方法 | Rule-based↑ | M25↑ | M50↑ | SR↑ | MaxP-F↑ | FNS↑ | DRR↑ | SQS↑ |
|------|-----------|------|------|-----|--------|------|------|------|
| V-JEPA 2.1 | 15.2 | 40 | 12 | 0 | 23.0 | 13.9 | 25.8 | 0.0 |
| Qwen3.5 | 12.4 | 26 | 10 | 0 | 18.3 | 11.2 | 19.2 | 0.0 |
| π₀.₅ | 27.6 | 54 | 16 | 2 | 27.9 | 17.7 | 31.5 | 1.5 |
| **Orca** | **36.6** | **64** | **16** | **4** | **33.9** | **19.3** | **32.9** | **1.8** |

**关键发现**: OOD 环境下 Orca Rule-based 分数 36.6 比 π₀.₅（27.6）高 32.6%，M25（64 vs 54）明显更高，体现世界模型对未见环境的更强泛化。

### Table 5：消融实验

| $\lambda_{\text{obs}}$ | $\lambda_{\text{evt}}$ | $\lambda_{\text{vqa}}$ | 文本 | 图像 | 动作 | **平均** |
|----|----|----|----|----|----|-----|
| ✓ | — | — | 48.4 | — | 10.2 | 29.3 |
| ✓ | ✓ | — | 58.2 | 30.9 | — | 44.6 |
| ✓ | — | ✓ | 50.5 | — | 32.6 | 41.6 |
| — | ✓ | ✓ | 50.1 | 54.7 | 23.0 | 42.6 |
| **✓** | **✓** | **✓** | **51.8** | **59.8** | **32.4** | **48.0** |

**关键发现**: 三个目标缺一不可；$\mathcal{L}_{\text{evt}}$ 对图像预测最关键（无它时图像性能断崖），$\mathcal{L}_{\text{vqa}}$ 对动作生成最关键（无它时动作从 32.4→10.2 骤降），$\mathcal{L}_{\text{obs}}$ 在三者组合时对文本理解有额外提升。

---

## 实验

### 数据集

| 数据类型 | 规模 | 内容 | 用途 |
|---------|------|------|------|
| 视频数据（全量） | 125,000 小时 | 自我中心互动、外视角操作、机器人执行、自然动态 | 无意识学习 |
| 视频数据（当前使用） | 12,500 小时（1/10） | 同上 | 当前训练 |
| 事件标注 | 160M 条 | 语言描述的场景变化事件 | 有意识学习 |
| VQA 数据 | 11.5M 对 | 视觉问答 | 语义对齐 |
| PRICE-V0.1 | — | 图像预测基准 | 视觉 Readout 评测 |
| 真实机器人任务 | 5 个任务 | Take Book / Stacked Bowls / Pull Out Tissue / Stamp / Scoop Sugar | 动作 Readout 评测 |

### 实现细节

| 组件 | Orca-4B | Orca-0.8B |
|------|---------|-----------|
| 基础 VLM | Qwen3.5-4B | Qwen3.5-0.8B |
| 隐层维度 | 2560 | 1024 |
| 训练节点/GPU | 32 节点 / 256 GPU | 32 节点 / 256 GPU |
| Batch Size（每 GPU） | 8 | 8 |
| 梯度累积 | 2 | 2 |
| 训练步数 | 10,844 | 10,844 |
| 最大序列长度 | 1024 | 1024 |

**优化器**: AdamW，VLM 学习率 $3.5\times10^{-5}$，视觉头 $1.2\times10^{-4}$，$\beta=[0.9, 0.95]$，weight decay $10^{-8}$，Cosine 调度，warmup 200步

**视觉 Readout（SD3.5）**: MLP Adapter 556.9M 参数 + LoRA (rank=32, alpha=32, dropout=0.05)，Global batch=512，训练 200K 步

**动作 Readout（DiT）**: 8 个 DiT 块，hidden=1024，12 头，flow matching 损失，推理 4 步去噪，Global batch=128，训练 20K 步

**训练框架**: [[FlagScale]] + [[FSDP2]]，4.4× 吞吐量加速（vs StarVLA 基线：2.91 vs 0.66 Samples/Sec/GPU）

### 可视化结果

图像预测任务中，Orca 在预测物体被推开、液体倒入、食材切割等场景时保持了高度的物理一致性；在机器人任务中，Orca 成功从铲子抓取失败中自动恢复，展现了对异常状态的规划能力。

---

## 批判性思考

### 优点

1. **范式创新有说服力**: Next-State-Prediction 作为统一框架自然覆盖了文本、图像、动作三类下游任务，消融实验也验证了三路训练目标的必要性
2. **Scaling 特性良好**: 模型和数据规模提升均带来一致的 Loss 下降和下游性能提升，符合 Foundation Model 预期
3. **OOD 泛化明显**: 真实机器人 OOD 环境下相比 π₀.₅ 提升 32.6%，体现了世界模型对物理规律的内在理解

### 局限性

1. **监督信号局限于 ViT 空间**: 隐状态监督信号来自预训练 ViT，而非自由的世界表示，可能限制表征的上限
2. **模态覆盖不完整**: 当前仅有视觉和语言，缺少触觉、音频、力反馈等感知模态
3. **事件标注时间粒度粗**: 事件标注描述分钟级变化，对长时序演化建模能力有限
4. **动作 SR 依然很低**: 环境 OOD 下成功率仅 4%（π₀.₅ 2%），绝对表现仍有较大空间
5. **数据利用率低**: 当前只用了 1/10 的视频数据，结论在全量数据下是否成立有待验证

### 潜在改进方向

1. 引入音频、触觉等多模态信号，突破纯视觉-语言限制
2. 用自由隐状态（非 ViT 对齐）替代现有监督，学习原生世界表示
3. 增加长时序事件标注密度，提升长程规划能力

### 可复现性评估

- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（论文附录 C 有完整超参）
- [ ] 数据集可获取（125K 小时专有视频数据未公开）

---

## 关联笔记

### 基于

- [[V-JEPA]]: JEPA 范式的先驱，Orca 继承了在隐空间做预测的思路
- [[Qwen3.5]]: 作为 Orca 的 VLM 骨干（语言-视觉对齐起点）

### 对比

- [[pi0.5\|π₀.₅]]: 最主要的动作生成基线，OOD 场景下被 Orca 明显超越
- [[V-JEPA]]: 纯视觉预训练世界模型，缺乏语言条件化能力

### 方法相关

- [[Next-State Prediction]]: 核心范式
- [[Unconscious Learning]]: 无意识学习范式
- [[Conscious Learning]]: 有意识学习范式
- [[Flow Matching]]: 动作 Readout 使用的生成方式
- [[DiT]]: 动作专家的网络结构
- [[Action Chunking]]: 动作输出形式
- [[FlagScale]]: 训练基础设施框架

### 硬件/数据相关

- [[FSDP2]]: 分布式训练优化方案

---

## 速查卡片

> [!summary] Orca: The World is in Your Mind
> - **核心**: 以 Next-State-Prediction 为统一范式，学习通用世界隐空间
> - **方法**: 无意识学习（视频）+ 有意识学习（语言事件）双路预训练，冻结骨干接轻量 Readout
> - **结果**: 文本 51.8（+5.1 vs Qwen3.5-4B），图像预测 PRICE 59.8（SOTA），机器人 OOD Rule-based 36.6（+32% vs π₀.₅）
> - **代码**: 暂未开源（项目主页：https://orca-wm.github.io/）

---

*笔记创建时间: 2026-07-16*

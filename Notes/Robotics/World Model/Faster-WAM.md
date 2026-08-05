---
title: "Faster-WAM: Do World Action Models Need Deep Action Modules?"
method_name: "Faster-WAM"
authors: [Liheng Ma, Rui Heng Yang, Zhanguang Zhang, Mateo Clemente, Ziwen Hu, Tongtong Cao, Yingxue Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, inference-efficiency, robot-manipulation, diffusion-transformer, representation-hub, kv-fusion]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.02365
created: 2026-08-05
---

# 论文笔记：Faster-WAM: Do World Action Models Need Deep Action Modules?

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Huawei Noah's Ark Lab；Huawei Celia Team；Department of Foundation Model, 2012 Labs |
| 日期 | August 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Fast-WAM]]、[[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.02365) / Code 暂未开放 |

---

## 一句话总结

> Faster-WAM 提出 [[DoT]]（Dock of Transformer）框架，将预训练视频 Transformer 作为表征枢纽，通过 [[KV-Fusion]] 让单层动作头直接访问 30 层骨干的多层 KV，推理延迟 3.2× 低于 Fast-WAM，同时在 LIBERO 和 RoboTwin 2.0 上保持竞争力，并在 OOD 泛化上大幅领先。

---

## 核心贡献

1. **DoT 框架（Dock of Transformer）**: 将预训练视频 [[Video DiT]] 作为只读表征枢纽，轻量动作头通过对接接口（docking interface）访问骨干的全层表征，彻底解耦动作模块深度与视频骨干深度
2. **KV-Fusion 对接机制**: 通过通道混合（Mode-n product）+ 跨层混合（learnable aggregation）将视频骨干所有层的 KV 融合到单层动作头的注意力计算中
3. **视频-动作 RoPE 对齐**: 将 [[3D RoPE]] 编码的视频 Key 先"反旋"到规范空间再与动作查询 [[Rotary Position Encoding|RoPE]] 对齐融合，解决跨模态位置编码不兼容问题

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model（WAM）]] 已成为机器人操作的主流范式，但现有高性能 WAM（如 [[Fast-WAM]]）仍面临严重的推理延迟问题：Fast-WAM 的每动作块推理延迟为 211.7ms，近 3× 慢于 [[Pi05|π₀.₅]] 的 71.4ms，不满足实时控制需求。

### 现有方法的局限

- **Shared-backbone WAM**：动作 token 与视频 token 在同一 Transformer 内交织处理，动作模块深度与视频骨干深度完全耦合，计算量随骨干规模线性增长
- **[[MoT]]（Mixture-of-Transformers）设计（Fast-WAM 采用）**：要求动作流与视频流一一对应的逐层交互，动作流深度等于视频骨干深度（30 层），无法裁减
- **MotuBrain 的 H-Bridge**：跨流交互限制在固定的中间几层，缺乏灵活的多层表征访问能力
- **轻量 WAM（Light-WAM、Efficient-WAM）**：采用小骨干或简化解码器，性能在挑战性 benchmark 上大幅下滑

### 本文的动机

关键洞察：**动作预测需要的是骨干内丰富的多层表征，而不是骨干的等深度模仿**。如果让轻量动作头直接"对接"骨干的所有层 KV，无需逐层对应，就能在不损失信息的前提下将动作头压缩到 1 层，推理速度提升 3.2×。

---

## 方法详解

### 模型架构

![Figure 2: DoT 架构](https://arxiv.org/html/2608.02365v1/x2.png)

**说明**: DoT 将预训练 [[Wan2.2-TI2V-5B]] 视频 DiT 作为表征枢纽（hub），单层动作头通过对接接口从外部访问骨干的所有层 KV，视频主干参数冻结或联合微调。

Faster-WAM 采用 **视频-动作解耦** 架构：
- **输入**: 语言指令 + 历史帧观测 + 噪声动作序列
- **Backbone**: [[Wan2.2-TI2V-5B]]（30 层 [[Video DiT|Diffusion Transformer]]，参数约 5B）
- **核心模块**: [[KV-Fusion]] 通过 Mode-n 乘积 + 跨层可学习矩阵聚合全部 30 层 KV；[[DoT]] 接口统一管理视频-动作对接
- **动作头**: 单层 [[Transformer]] Decoder（hidden size 1024，24 head × 128 dim）
- **输出**: [[Action Chunking|动作块]] $a_{t:t+32}$（H=32 步）
- **总参数**: ~5.05B 可训练参数（骨干 ~5B + 动作头 ~30M + KV-Fusion ~20M）

### 核心模块

#### 模块 1: KV-Fusion（[[KV-Fusion|关键值融合]]）

**设计动机**: 利用视频骨干各层已计算好的 Key/Value 缓存（[[KV Cache]]），通过可学习矩阵聚合成单套 KV，供单层动作头使用，避免深度耦合

**具体实现** — 两步骤融合：

**步骤 A — 通道混合（Channel Mixing）**：将视频 Key/Value 从视频特征空间投影到动作头特征空间

将所有层 $L_v$ 的 KV 拼接为张量后，通过 Mode-n 乘积（mode-4 product）投影：

$$
\bar{K}^v = \text{reshape}\!\left(\text{reshape}(K^v)\ \times_4\ W^K\right)
$$

$$
\bar{V}^v = \text{reshape}\!\left(\text{reshape}(V^v)\ \times_4\ W^V\right)
$$

**步骤 B — 跨层混合（Layer Mixing）**：按注意力头（head）学习对各视频层的贡献权重

对每个注意力头 $h$，用可学习矩阵 $A_h \in \mathbb{R}^{1 \times L_v}$ 对层维度做加权求和：

$$
\tilde{K}^{h,v} = \bar{K}^{h,v}\ \times_1\ A_h, \quad \tilde{V}^{h,v} = \bar{V}^{h,v}\ \times_1\ A_h
$$

**完整对接机制**（论文统一形式）：

$$
(\tilde{K}^v_j,\ \tilde{V}^v_j) = g_j\!\left(\{(K^v_l,\ V^v_l)\}_{l=1}^{L_v}\right)
$$

其中 $g_j$ 即上述两步可学习函数，$j$ 为动作头层索引（Faster-WAM 中 $j=1$）。

#### 模块 2: 视频-动作 RoPE 对齐

**设计动机**: 视频 backbone 使用 [[3D RoPE]]（三维旋转位置编码，作用于时间 × 高 × 宽），动作头使用标准 1D [[Rotary Position Encoding|RoPE]]（作用于时间步序列），直接融合会导致位置编码空间错位，影响注意力质量

**具体实现** — 三步 RoPE 对齐：

1. **反旋（Un-rotate）**: 将视频 backbone 缓存的 Key $K^v_l$（已乘 3D RoPE 旋转矩阵）先乘逆旋转矩阵，还原到规范（canonical）表征空间：
   $$K^{v,\text{canonical}}_l = R^{-1}_{3D}\ K^v_l$$

2. **融合**: 在规范空间执行 KV-Fusion 的通道混合与跨层混合，得到 $\tilde{K}^{v,\text{canonical}}$

3. **重施 1D RoPE**: 将融合后 KV 乘上与动作查询相同的 1D RoPE 旋转矩阵，确保位置编码一致：
   $$\tilde{K}^v = R_{1D}\ \tilde{K}^{v,\text{canonical}}$$

#### 模块 3: 混合上下文注意力（Mixed-Context Attention）

动作头的注意力层将自身 KV 与对接来的视频 KV 拼接后共同参与计算：

$$
O^a_j = \text{Softmax}\!\left(\frac{Q^a_j\ [K^a_j,\ \tilde{K}^v_j]^\top}{\sqrt{d}}\right) [V^a_j,\ \tilde{V}^v_j]
$$

其中 $Q^a_j$ 为动作头的查询，$[K^a_j, \tilde{K}^v_j]$ 为动作自 KV 与视频融合 KV 的拼接。

---

## 关键公式

### 公式 1: [[KV-Fusion|KV 对接聚合函数]]

$$
(\tilde{K}^v_j,\ \tilde{V}^v_j) = g_j\!\left(\{(K^v_l,\ V^v_l)\}_{l=1}^{L_v}\right)
$$

**含义**: 对接接口 $g_j$ 将视频骨干所有 $L_v$ 层的 Key/Value 集合聚合为单套融合 KV，供第 $j$ 层动作头使用

**符号说明**:
- $\tilde{K}^v_j, \tilde{V}^v_j$: 融合后供动作头使用的视频 Key、Value
- $K^v_l, V^v_l$: 视频骨干第 $l$ 层的 Key、Value 缓存
- $L_v = 30$: 视频骨干总层数（Wan2.2-TI2V-5B）
- $g_j$: 可学习对接函数（由通道混合 + 跨层混合组成）

### 公式 2: [[KV-Fusion|通道混合（Mode-n 乘积）]]

$$
\bar{K}^v = \text{reshape}\!\left(\text{reshape}(K^v)\ \times_4\ W^K\right)
$$

**含义**: 通过 mode-4 张量乘积将视频 KV 从视频特征维度投影到动作头的特征维度，实现跨模态特征空间对齐

**符号说明**:
- $K^v \in \mathbb{R}^{L_v \times T \times H \times W \times d_v}$: 所有层 Key 拼接张量
- $W^K$: 可学习投影矩阵，$d_v \to d_a$（视频特征维 → 动作特征维）
- $\times_4$: Mode-4 张量乘积（作用于第 4 维即特征维）

### 公式 3: [[KV-Fusion|跨层混合（Layer-wise Aggregation）]]

$$
\tilde{K}^{h,v} = \bar{K}^{h,v}\ \times_1\ A_h, \quad \tilde{V}^{h,v} = \bar{V}^{h,v}\ \times_1\ A_h
$$

**含义**: 对每个注意力头 $h$ 独立学习各视频层的贡献权重 $A_h$，使不同注意力头可以专注于骨干不同深度的特征

**符号说明**:
- $\bar{K}^{h,v} \in \mathbb{R}^{L_v \times T'}$: 通道混合后第 $h$ 头的 Key（层 × token 数）
- $A_h \in \mathbb{R}^{1 \times L_v}$: 第 $h$ 头的可学习层权重向量
- $\tilde{K}^{h,v} \in \mathbb{R}^{T'}$: 最终融合的 Key（已沿层维度加权求和）

### 公式 4: [[KV-Fusion|混合上下文注意力]]

$$
O^a_j = \text{Softmax}\!\left(\frac{Q^a_j\ [K^a_j,\ \tilde{K}^v_j]^\top}{\sqrt{d}}\right) [V^a_j,\ \tilde{V}^v_j]
$$

**含义**: 动作头在自注意力计算中同时使用动作自身 KV 和从视频骨干对接来的融合 KV，单层动作头因此获得骨干全 30 层的多尺度特征

**符号说明**:
- $Q^a_j$: 动作头第 $j$ 层的查询（Query）
- $K^a_j, V^a_j$: 动作头自身的 Key、Value
- $\tilde{K}^v_j, \tilde{V}^v_j$: 视频融合 Key、Value（来自 KV-Fusion）
- $[\cdot, \cdot]$: 沿 token 维度的拼接
- $d$: Head dimension（128）

---

## 关键图表

### Figure 1: WAM 架构对比

![Figure 1: 三种 WAM 架构对比](https://arxiv.org/html/2608.02365v1/x1.png)

**说明**: 三种 [[WAM]] 架构设计的对比。MoT（[[Fast-WAM]] 采用）要求视频流与动作流逐层一一对应；H-Bridge（MotuBrain 采用）限制在固定中间层交互；[[DoT]]（Faster-WAM）将视频骨干作为表征枢纽，动作头从外部对接访问多层 KV，动作深度与视频深度完全解耦。

### Figure 2: DoT 架构与 Faster-WAM 实现

![Figure 2: DoT 完整架构](https://arxiv.org/html/2608.02365v1/x2.png)

**说明**: DoT 框架的完整实现。预训练 [[Wan2.2-TI2V-5B]] 视频骨干产生多层 KV 缓存，[[KV-Fusion]] 通过通道混合和跨层混合聚合所有层的 KV，配合视频-动作 [[3D RoPE|RoPE]] 对齐，将融合后的视频 KV 注入单层动作头。

### Figure 3a: 消融实验 — 顺序设计贡献

![Figure 3a: 设计消融](https://arxiv.org/html/2608.02365v1/x3.png)

**说明**: LIBERO-Plus 泛化性能（% 成功率）随各设计组件的累积提升：

| 配置 | LIBERO-Plus 成功率 | 增益 |
|------|--------------------|------|
| 仅最终层（Final-layer only） | 60.3% | — |
| \+ KV-Fusion | 66.8% | +6.5pp |
| \+ RoPE 对齐 | 71.3% | +4.5pp |
| \+ 移除文本交叉注意力 | **75.0%** | +3.7pp |

**关键发现**: KV-Fusion 贡献最大（+6.5pp），说明多层视频表征对泛化至关重要；去除文本交叉注意力在动作头中同样有益（视频骨干已隐式包含语言信息）。

### Figure 3b: KV-Fusion 跨层混合信号可视化

![Figure 3b: 跨层融合信号分布](https://arxiv.org/html/2608.02365v1/x4.png)

**说明**: 可学习的跨层权重矩阵 $A_h$ 在各注意力头上的热图。核心发现：**最强融合信号集中在骨干的中间层**（而非最终层），但所有层均有非可忽略的贡献，验证了全层 KV 访问的设计必要性。

### Table 1: LIBERO Benchmark 对比

| 方法 | Spatial | Object | Goal | Long | **Avg.** |
|------|---------|--------|------|------|----------|
| π₀ | 89.0% | 97.2% | 90.0% | 75.2% | 87.9% |
| π₀.₅ | 97.6% | 99.6% | 97.8% | 92.0% | 96.8% |
| Fast-WAM | 98.2% | 100.0% | 97.0% | 95.2% | 97.6% |
| **Faster-WAM（Ours）** | **98.4%** | **100.0%** | **97.0%** | **97.8%** | **98.5%** |

**关键发现**: Faster-WAM 在 LIBERO 平均成功率上超越 [[Fast-WAM]] 0.9pp（98.5% vs. 97.6%），在 Long 子集上提升 2.6pp，表明单层动作头不仅不劣于深层头，反而略有优势。

### Table 2: RoboTwin 2.0 对比

| 方法 | 使用体态 PT | Clean | Randomized | **Avg.** |
|------|-------------|-------|------------|----------|
| π₀ | ✓ | 65.92 | 58.40 | 62.16 |
| π₀.₅ | ✓ | 82.74 | 76.76 | 79.75 |
| X-VLA | ✓ | 72.9 | 72.8 | 72.85 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| Motus | ✓ | 88.66 | 87.02 | 87.84 |
| Fast-WAM | ✗ | 91.42 | 91.86 | 91.64 |
| **Faster-WAM（Ours）** | **✗** | **89.70** | **88.64** | **89.17** |

**关键发现**: Faster-WAM 在不使用体态预训练的方法中与 [[Fast-WAM]] 竞争性持平（89.17% vs. 91.64%，差距 2.47pp），3.2× 延迟优势显著；使用体态预训练的 LingBot-VA 性能更高（92.20%），但属于不同对比组。

### Table 3: 推理延迟对比

| 方法 | 延迟（ms） | 相对 Fast-WAM |
|------|-----------|----------------|
| Fast-WAM | 211.7 | 1.0× |
| π₀.₅ | 71.4 | 2.96× faster |
| **Faster-WAM（Ours）** | **66.5** | **3.18× faster** |

**关键发现**: Faster-WAM 以 66.5ms 超越 π₀.₅（71.4ms），成为 WAM 中延迟最低的高性能模型，在 24 GB 消费级 GPU 上测试。

### Table 4: LIBERO-Plus OOD 泛化对比

| 方法 | Camera | Texture | Lighting | Noise | Distractors | Size | Orientation | **Avg.** |
|------|--------|---------|----------|-------|-------------|------|-------------|----------|
| Fast-WAM | 23.5% | 59.3% | 66.7% | 37.3% | 60.2% | 65.3% | 56.0% | 51.5% |
| **Faster-WAM** | **75.0%** | **82.1%** | **76.9%** | **81.9%** | **74.7%** | **78.3%** | **61.6%** | **75.0%** |

**关键发现**: Faster-WAM 在 LIBERO-Plus OOD 泛化上大幅超越 [[Fast-WAM]] **+23.5pp**（75.0% vs. 51.5%），摄像机扰动类提升最大（+51.5pp）。这一结果反映出 DoT 的全层 KV 融合学到了更具泛化性的多层次表征，而非仅依赖最终层的任务特定特征。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 2,000 demos / 40 tasks | 桌面操作，标准 in-distribution 评估 | 训练 + 测试（50 trials/task） |
| [[LIBERO-Plus]] | 10,030 instances / 7 扰动类 | 相机、纹理、光照、噪声等 OOD 扰动，无需重训 | OOD 泛化测试 |
| [[RoboTwin 2.0]] | 27,500 demos / 50 tasks | Clean + Randomized 场景，仿真 benchmark | 训练 + 测试（100 trials/task） |

### 实现细节

- **视频骨干**: [[Wan2.2-TI2V-5B]]（30 层 DiT，冻结视频权重，仅训练 KV-Fusion + 动作头）
- **动作头**: 单 Transformer 层，hidden size 1024，24 head × 128 dim
- **KV-Fusion 参数**: ~20M；动作头参数: ~30M；总可训练参数: ~5.05B
- **优化器**: [[AdamW]]，lr = 1×10⁻⁴，weight decay = 0.01，cosine annealing 调度
- **Batch Size**: 128（LIBERO），1024（RoboTwin 2.0）
- **训练轮数**: 10 epochs（LIBERO），5 epochs（RoboTwin 2.0）
- **推理**: 10 步去噪，[[Classifier-Free Guidance (CFG)|CFG]] scale = 1.0，动作块长度 H = 32
- **硬件（推理）**: 24 GB 消费级 GPU

---

## 批判性思考

### 优点

1. **延迟-性能帕累托改进**: 66.5ms 同时低于 Fast-WAM（3.2×）和 π₀.₅（轻微），且 LIBERO 性能不降反升，属真正帕累托前沿提升
2. **OOD 泛化显著提升**: LIBERO-Plus +23.5pp 的幅度远超预期，说明多层 KV 融合学到了更具结构性的表征，不局限于任务特定特征
3. **可扩展性**: DoT 框架与具体骨干无关，可接入任意视频 DiT（如 Wan、CogVideo 等），普适性强
4. **参数效率**: KV-Fusion（~20M）+ 动作头（~30M）共增加约 50M 参数，相对 5B 骨干可忽略不计

### 局限性

1. **仅仿真评估**: 论文未提供真实机器人实验，sim-to-real 迁移能力未知
2. **RoboTwin 略低于 Fast-WAM**: 尽管 OOD 大幅领先，RoboTwin 2.0 仍比 Fast-WAM 低 2.47pp，可能因为任务分布不同对深浅架构的敏感性有差异
3. **骨干依赖性**: 现实中 KV 缓存读取意味着骨干仍需完整前向传播，仅省去了动作流的深度，实际节省取决于原始设计的动作流比例
4. **消融有限**: 仅消融在 LIBERO-Plus 上评估，未系统分析不同骨干规模对 DoT 效果的影响

### 潜在改进方向

1. 在真实机器人上验证（论文也指出未来方向）
2. 探索 DoT 在导航、灵巧手等其他具身任务的适用性
3. 多动作头（>1 层）时的性能-延迟权衡分析

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节较完整（optimizer、lr、batch size、epochs 均有说明）
- [x] 数据集可获取（LIBERO/RoboTwin 2.0 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: Faster-WAM 直接在其基础上提速，共用 Wan2.2-TI2V-5B 骨干和 MoT 对比基线
- [[Flash-WAM]]: 同期竞争方法，采用知识蒸馏路线提速
- [[Wan2.2-TI2V-5B]]: 使用的视频骨干模型

### 对比

- [[Fast-WAM]]: 主对比基线，同骨干但 MoT 架构，211.7ms vs. 66.5ms
- [[Pi05]]: π₀.₅ 作为延迟参考基准（71.4ms）

### 方法相关

- [[DoT]]: 本文提出的核心框架
- [[KV-Fusion]]: 本文提出的对接机制
- [[MoT]]: DoT 对比的传统深度耦合设计
- [[3D RoPE]]: 视频骨干使用的位置编码，需 RoPE 对齐处理
- [[Mixture-of-Transformers]]: Fast-WAM 采用的架构，Faster-WAM 改进的对象

### 硬件/数据相关

- [[LIBERO]]: 主训练和测试 benchmark
- [[LIBERO-Plus]]: OOD 泛化评估集
- [[RoboTwin 2.0]]: 仿真大规模 benchmark

---

## 速查卡片

> [!summary] Faster-WAM: Do World Action Models Need Deep Action Modules?
> - **核心**: DoT 框架将视频骨干作为表征枢纽，单层动作头通过 KV-Fusion 对接全部 30 层 KV，彻底解耦深度耦合
> - **方法**: KV-Fusion（通道混合 + 跨层混合）+ 视频-动作 RoPE 对齐 + 移除动作头文本交叉注意力
> - **结果**: 66.5ms 延迟（3.2× 快于 Fast-WAM），LIBERO 98.5%，RoboTwin 2.0 89.17%，LIBERO-Plus OOD +23.5pp
> - **代码**: 暂未开放

---

*笔记创建时间: 2026-08-05*

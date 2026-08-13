---
title: "Flex-π: A Multi-Stream World-Action Model with Compute Flexibility"
method_name: "Flex-Pi"
authors: [Ge Yan, Jinghao Liu, Yuzhi Fan, Lei Cai, Minwen Liao, Jesse Zhang, Dieter Fox]
year: 2026
venue: arXiv
tags: [world-action-model, multi-modal, robot-manipulation, flow-matching, bimanual, cross-modality]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://export.arxiv.org/html/2608.10860v1
created: 2026-08-13
---

# 论文笔记：Flex-π: A Multi-Stream World-Action Model with Compute Flexibility

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Washington, Allen Institute for AI |
| 日期 | August 2026 |
| 项目主页 | [flex-pi.github.io](https://flex-pi.github.io/) |
| 对比基线 | [[ManiFlow]], [[π0.5]], [[Fast-WAM]], [[GR00T-N1]], [[OpenVLA-OFT]], [[MolmoAct2-Think]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.10860) / [Code](https://github.com/geyan21/flex-pi) |

---

## 一句话总结

> Flex-π 是一个 60 亿参数的多流世界-动作模型，通过对 RGB、3D 点图和 DINO 语义特征联合监督，单一 checkpoint 支持 56 种输入输出组合，在双臂灵巧操作任务上比最强基线高出 2-7 倍。

---

## 核心贡献

1. **免费午餐发现**: 预训练视频 [[VAE]]（Wan-2.2）无需任何点图专项训练，即可近乎无损地编码 3D Pointmap，大幅降低多模态接入门槛。
2. **多流 [[WAM|世界动作模型]]**: 同时对 RGB 外观、3D 几何（Pointmap）和 [[DINOv3]] 语义特征进行联合预测监督，显著增强表征的几何与语义一致性。
3. **[[Cross-Modality Forcing|跨模态强迫训练]]**: 训练时要求模型从残留流预测缺失流，迫使多模态表征互相可预测，性能相对基线提升 47%。
4. **[[Visual Stream Dropout|视觉流 Dropout]] 部署灵活性**: 单一 checkpoint 支持从纯动作推理（60ms）到全联合生成（193ms）等 56 种输入输出组合，无需重新训练。

---

## 问题背景

### 要解决的问题

现有 [[WAM|World Action Model]] 要么依赖单一视觉模态（RGB），无法充分利用 3D 几何和语义信息；要么针对不同部署场景需要分别训练不同模型，缺乏计算灵活性。

### 现有方法的局限

- [[π0.5]]、[[Fast-WAM]] 等模型通常仅使用 RGB 流，忽视了 3D 几何对接触精度任务的关键价值。
- 多模态输入需要为每种组合单独设计和训练模型，工程成本高。
- [[VAE]] 编码点图被认为需要专门训练，限制了几何信息的低成本接入。

### 本文的动机

作者发现，用于视频生成的 Wan-2.2 [[VAE]] 在完全未训练点图的情况下，仍能近乎无损地重建 3D 点图（即"免费午餐"）。基于此，他们设计了一个统一多流架构，引入 [[Cross-Modality Forcing|跨模态强迫]] 和 [[Visual Stream Dropout|流 Dropout]] 机制，使单一模型在不同计算预算下灵活运行。

---

## 方法详解

### 模型架构

[[Flex-Pi|Flex-π]] 采用 **[[Mixture-of-Transformers|Mixture-of-Transformers (MoT)]]** 架构：

- **输入**: 语言指令 $l$（umT5 编码）+ RGB 观测 $o_t$ + 点图 $p_t$ + [[DINOv3]] 特征 $d_t$ + 本体感知状态 $s_t$
- **Backbone**: Wan-2.2-5B [[Mixture-of-Transformers|MoT]] 主干（30 层 Transformer，中间 16 层启用跨流注意力）
- **核心模块**: [[Cross-Modality Forcing|跨模态强迫]] + [[Visual Stream Dropout|流 Dropout]] 实现灵活训练与部署
- **输出**: 动作 $a_t$（[[Action Chunking|动作块]]）+ 下一步视觉预测 $z^o_{t+1}, z^p_{t+1}, d_{t+1}$（可选）
- **总参数**: 6B（MoT 主干 5B + 动作专家 1B）

### 核心模块

#### 模块1: 多流视觉编码

**设计动机**: 利用预训练 [[VAE]] 的通用压缩能力，同时接入外观、几何和语义三种互补信息。

**具体实现**:
- **RGB 流** $z^o_t = \text{Enc}(o_t)$：使用冻结的 Wan-2.2 [[VAE]] 编码器；继承视频预训练先验
- **Pointmap 流** $z^p_t = \text{Enc}(p_t)$：**同一个 [[VAE]] 编码器**（无点图专项训练），发现可近乎无损编码 3D 点图
- **DINO 流** $d_t = \text{DINO}(o_t)$：[[DINOv3]] 编码器，$2 \times 2$ patch folding 将 token 数减少 4×（768→3072 dim），保留空间细节

#### 模块2: Visual Stream Dropout 与跨模态强迫

**设计动机**: 使用 [[Visual Stream Dropout|流 Dropout]] 实现 56 种输入输出组合，并通过 [[Cross-Modality Forcing|跨模态强迫]] 提升多模态表征互预测能力。

**具体实现**:
- 输入掩码 $\mathbf{m}^{in} \in \{0,1\}^3$：每个视觉流独立以概率 0.5 丢弃（至少保留一个流）
- 输出注意力掩码 $\mathbf{m}^{out} \in \{0,1\}^3$：控制联合生成时哪些流参与输出
- **[[Cross-Modality Forcing|跨模态强迫]]**：当某流不在输入中时，仍要求模型从其他流生成该流的预测，迫使模型内化所有流的互相预测关系

#### 模块3: 因果联合生成

**设计动机**: 保持动作与视觉预测的因果顺序，避免视觉预测信息提前泄露。

**具体实现**:
- 动作 token 可以 attend 到所有观测 token 和视觉未来预测 token
- 视觉预测 token **不**可以 attend 到动作 token（单向注意力）
- [[Flow Matching|流匹配]] ODE 以 K=4 Euler 步推理，平衡速度与精度

---

## 关键公式

### 公式1: [[Flow Matching|流匹配]] 损失

$$
\mathcal{L}_{FM}(z_{t+1}) = \mathbb{E}_{z, \epsilon, \tau} \left\| v_\theta\!\left(\cdot \mid \tau, l, z_{1:t}\right) - \left(z_{t+1} - \epsilon\right) \right\|_2^2
$$

**含义**: 训练速度场网络 $v_\theta$ 逼近从噪声 $\epsilon$ 到目标 $z_{t+1}$ 的直线流，用于预测下一步视觉潜变量或动作。

**符号说明**:
- $z^{\tau}_{t+1} = \tau z_{t+1} + (1 - \tau)\epsilon$：插值中间状态
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $\tau \in [0, 1]$：插值系数，均匀采样
- $v_\theta$：模型预测的速度场
- $l$：语言条件

### 公式2: 多流视觉编码输入

$$
\begin{aligned}
z^o_t &= \text{Enc}(o_t) & &\triangleright \text{RGB 外观，继承视频预训练先验} \\
z^p_t &= \text{Enc}(p_t) & &\triangleright \text{3D 几何，共享 RGB VAE 潜空间} \\
d_t &= \text{DINO}(o_t) & &\triangleright \text{对象语义锚定}
\end{aligned}
$$

**含义**: 三种视觉流均编码进共享潜空间，关键发现是同一个 [[VAE]] 编码器无需额外训练即可处理点图。

**符号说明**:
- $\text{Enc}(\cdot)$：冻结的 Wan-2.2 VAE 编码器
- $\text{DINO}(\cdot)$：[[DINOv3]] 特征提取器（2×2 patch folding）
- $o_t$：RGB 观测图像；$p_t$：3D 点图

### 公式3: 总训练目标

$$
\mathcal{L}(\theta) = \lambda_a \mathcal{L}^{FM}_a(a_t) + \sum_{i \in \{o, d, p\}} \lambda_i \mathcal{L}^{FM}_i(i_{t+1})
$$

**含义**: 联合监督动作生成损失与三个视觉流的未来预测损失，所有权重均设为 1。

**符号说明**:
- $\lambda_a, \lambda_i = 1$：各项损失等权重
- $\mathcal{L}^{FM}_a$：动作[[Flow Matching|流匹配]]损失
- $\mathcal{L}^{FM}_o, \mathcal{L}^{FM}_d, \mathcal{L}^{FM}_p$：RGB、DINO、Pointmap 未来预测损失

---

## 关键图表

### Figure 1: 系统概览

![Figure 1 Overview](https://export.arxiv.org/html/2608.10860v1/x1.png)

**说明**: Flex-π 的整体框架。输入 RGB、3D 点图和 [[DINOv3]] 视觉特征，通过 [[Mixture-of-Transformers|MoT]] 主干联合生成未来视觉潜变量和动作，支持灵活的输入输出流组合。

### Figure 2: 模型架构

![Figure 2 Architecture](https://export.arxiv.org/html/2608.10860v1/x2.png)

**说明**: 三路视觉流（RGB、Pointmap 经冻结 [[VAE]] 编码，[[DINOv3]] 语义 token）通过流在场掩码 $\mathbf{m}^{in}$、[[Mixture-of-Transformers|MoT]] Transformer 主干和动作专家，最终生成动作与未来视觉预测，受输出掩码 $\mathbf{m}^{out}$ 控制。

### Figure 3: Wan-VAE 点图重建质量

![Figure 3 Pointmap Reconstruction](https://export.arxiv.org/html/2608.10860v1/x3.png)

**说明**: 核心"免费午餐"发现——仅在 RGB 图像上训练的 Wan-2.2 [[VAE]] 在零点图训练的情况下即可近乎无损重建 3D 点图，验证了跨模态编码假设的可行性。

### Figure 4: 训练机制可视化

![Figure 4 Training Regime](https://export.arxiv.org/html/2608.10860v1/x4.png)

**说明**: 展示输入掩码 $\mathbf{m}^{in}$、输出注意力掩码 $\mathbf{m}^{out}$ 及 [[Cross-Modality Forcing|跨模态强迫]] 的训练样例。当某流从输入中 dropout 时，仍要求模型生成该流的未来预测。

### Figure 5: 真实评估任务

![Figure 5 Hard Tasks](https://export.arxiv.org/html/2608.10860v1/figs/hard_task_diagram.png)

**说明**: 5 个真实双臂 [[YAM]] 机器人任务，包括 Self-Repair Gripper（8 阶段任务，插入间隙仅 ±0.25-0.5mm）和 Soft-Bag Zipping（可变形物体精细操作），代表接触精度与任务复杂度的极限挑战。

### Figure 6: 真实环境任务完成率

![Figure 6 Real World Results](https://export.arxiv.org/html/2608.10860v1/x5.png)

**说明**: 5 个真实操作任务的完成率对比。Flex-π 在全部任务上领先，全联合生成模式平均比最强基线（[[ManiFlow]]）高出 2.3×；动作专一模式仍优于所有基线且推理速度最快。

### Figure 7: 速度-成功率帕累托前沿

![Figure 7 Speed vs Success](https://export.arxiv.org/html/2608.10860v1/x6.png)

**说明**: 5 任务平均完成率 vs 推理延迟散点图。Flex-π 动作专一模式（60ms）和全联合模式（193ms）均位于帕累托前沿右上角，展示单一 checkpoint 在速度-精度权衡上的优势。

### Figure 8: OOD 泛化评估条件

![Figure 8 OOD Conditions](https://export.arxiv.org/html/2608.10860v1/figs/plate_on_rack_conditions.png)

**说明**: "Plate on Rack"任务的分布外泛化评估条件，包含光照变化、干扰物摆放、物体颜色变化等视觉分布偏移场景。

### Figure 9: OOD 泛化与数据效率

![Figure 9 OOD and Data Efficiency](https://export.arxiv.org/html/2608.10860v1/x7.png)

**说明**: (a) OOD 条件下 Flex-π 平均 76.1% vs [[π0.5]] 43.2%；(b) 半数据情况下 Flex-π 与全数据基线持平，体现出[[WAM|世界模型]]联合监督的样本效率优势。

### Figure 10: RoboTwin 数据规模扩展曲线

![Figure 10 Data Scaling](https://export.arxiv.org/html/2608.10860v1/x8.png)

**说明**: [[RoboTwin]] 50 任务平均成功率随 demo 数量增长曲线。Flex-π 在每个数据规模上均领先，低 demo 预算时优势最大（1.9-4.5×），证明 WAM 多流预测目标显著提升了数据效率。

### Figure 11: 消融——流组合与延迟权衡

![Figure 11a Stream Ablation](https://export.arxiv.org/html/2608.10860v1/x9.png)

![Figure 11b Latency vs Success](https://export.arxiv.org/html/2608.10860v1/x10.png)

**说明**: (a) 累计消融结果：仅 RGB 为基础，加入 [[DINOv3]] +6.8%，加入 Pointmap 再 +20%；(b) 单一 checkpoint 在不同生成流数量下的延迟-成功率曲线，体现灵活计算能力。

### Figure 12: 跨模态强迫消融

![Figure 12 Cross-Modality Forcing](https://export.arxiv.org/html/2608.10860v1/x11.png)

**说明**: 在 [[RoboTwin]] 上对比有无 [[Cross-Modality Forcing|跨模态强迫]]（两者均观测全部三流）。去掉强迫训练，性能下降 21%，证明该训练策略对多流表征互预测能力的核心作用。

---

### Table 1: RoboTwin 成功率对比（%）

| 方法 | Clean | Rand. | Avg. |
|------|-------|-------|------|
| X-VLA | 72.9 | 72.8 | 72.9 |
| [[π0.5]] | 82.7 | 76.8 | 79.8 |
| [[Motus]] | 88.7 | 87.0 | 87.8 |
| Fast-WAM | 91.9 | 91.8 | 91.8 |
| LingBot-VA | 92.9 | 91.6 | 92.2 |
| LingBot-VA 2.0 | 93.8 | 93.4 | 93.6 |
| Qwen-RobotManip | 93.7 | 94.0 | 93.9 |
| **Flex-π (action-only)** | **94.5** | **94.6** | **94.6** |
| **Flex-π (full joint)** | **94.3** | **94.8** | **94.6** |

**说明**: Flex-π 在 [[RoboTwin]] 50 任务上取得最高成绩，动作专一与全联合模式均达 94.6%，分别比 [[π0.5]] 高 14.8 个点。

---

### Table 2: LIBERO 成功率对比（%）

| 方法 | LIBERO |
|------|--------|
| [[GR00T-N1]] | 93.9 |
| [[π0.5]] | 96.9 |
| Fast-WAM | 97.6 |
| [[OpenVLA-OFT]] | 97.1 |
| LingBot-VA | 98.5 |
| **Flex-π (action-only)** | **98.4** |
| **Flex-π* (action-only)** | **98.7** |
| [[MolmoAct2-Think]] | 98.1 |
| Qwen-RobotManip | 99.2 |
| **Flex-π (full joint)** | **98.5** |
| **Flex-π* (full joint)** | **99.2** |

**说明**: Flex-π* 在 [[LIBERO]] 达到与 Qwen-RobotManip 并列最高的 99.2%，在不依赖更大预训练语料的条件下达到顶级性能。

---

### Table 3: 消融实验（RoboTwin，输入流消融）

| 配置 | Avg. 成功率 | 说明 |
|------|------------|------|
| RGB Only | 基础 | 单流基线 |
| + [[DINOv3]] | +6.8% | 语义增益 |
| + Pointmap | +20% | 几何增益最大 |
| w/o [[Cross-Modality Forcing\|跨模态强迫]] | -21% | 强迫训练不可或缺 |
| Full Flex-π | 最高 | - |

**关键发现**: Pointmap 贡献的几何信息（+20%）是最大单项增益；[[Cross-Modality Forcing|跨模态强迫]] 缺失导致 21% 下降，证明其对多流表征的核心作用。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[AgiBotWorld|AGIBOT World-Beta]] | ~500h，100 任务 | 双臂操作，30Hz，头部+双腕相机 | 预训练 |
| [[RoboTwin]] | 50 任务，含 domain randomization | 仿真双臂，地面真值 3D 可用 | 仿真评估 |
| [[LIBERO]] | 4 任务套件（Spatial/Object/Goal/Long） | 桌面操作 | 标准 benchmark |
| [[LIBERO-Plus]] | 7 类扰动 | 视觉/空间/语言 OOD | 泛化评估 |
| 真实 [[YAM]] 机器人 | 5 接触丰富任务 | Sub-mm 精度要求 | 真实世界评估 |

### 实现细节

- **Backbone**: Wan-2.2-5B [[Mixture-of-Transformers|MoT]] 初始化
- **动作专家**: 独立 key/query/value/FFN/norm，约 1B 参数，hidden dim 缩减
- **跨流注意力**: 仅在中间 16/30 层启用，早期编码和后期解码保持流独立
- **流 Dropout**: 每流独立 0.5 概率丢弃，至少保留一流
- **推理步数**: K=4 Euler steps（[[Flow Matching|ODE 积分]]）
- **延迟**: 动作专一 60ms / 全联合 193ms
- **最少 Fine-tune**: 每个真实任务域至少 10 epochs 收敛

### 可视化结果

Flex-π 全联合模式生成的未来视觉预测（RGB + Pointmap + DINO）与真实观测高度一致，尤其在接触点附近的几何细节（如夹具位置）预测准确，体现出[[WAM|世界模型]]预测对动作精度的实际增益。

---

## 批判性思考

### 优点

1. **"免费午餐"发现具实用价值**: 无需标注/训练点图数据，直接复用视频 [[VAE]] 实现 3D 接入，工程门槛低。
2. **单 checkpoint 多配置**: [[Visual Stream Dropout|流 Dropout]] 设计使同一模型根据硬件预算灵活部署，真正解决 latency-accuracy tradeoff。
3. **数据效率突出**: 半数据情况下超越全数据基线，[[WAM|WAM 多流监督]]目标本质上是一种数据增强。
4. **任务难度有说服力**: Self-Repair Gripper（8 阶段、sub-mm 精度）和 Soft-Bag Zipping 等任务远超常规桌面操作评估。

### 局限性

1. **收敛慢**: 真实任务需 ≥10 epochs，与需要少量 fine-tune 的 [[π0.5]] 类方法相比工程成本更高。
2. **全联合模式延迟偏高**: 193ms vs 60ms（3×），实时控制场景受限。
3. **LIBERO-Plus 上略逊大预训练对手**: 部分基线受益于更大预训练语料，Flex-π 该场景下有差距。
4. **仍需更强语义推理**: 复杂语言指令场景仍有提升空间。

### 潜在改进方向

1. 引入更强 [[VLA|VLM]] 骨干提升语言理解能力，进一步改善 LIBERO-Plus 泛化。
2. 探索 tactile（触觉）流作为第四个感知流，进一步提升接触精度任务表现。

### 可复现性评估

- [x] 代码开源（github.com/geyan21/flex-pi）
- [ ] 预训练模型（待确认）
- [x] 训练细节完整（论文 Appendix 详述）
- [x] 数据集可获取（[[RoboTwin]]、[[LIBERO]] 均公开）

---

## 关联笔记

### 基于

- [[WAM|World Action Model]]: 本文的框架范式，同时预测未来视觉状态和动作
- [[π0.5]]: 主要对比基线，同为双臂操作 WAM/VLA
- [[Mixture-of-Transformers|MoT]]: 主干架构，来自 Wan-2.2 初始化
- [[Flow Matching]]: 动作与视觉预测的统一生成目标
- [[DINOv3]]: 语义流编码器

### 对比

- [[ManiFlow]]: 真实任务最强基线，Flex-π 超越 2.3×
- [[Fast-WAM]]: WAM 速度优化方向的对比
- [[π0.5]]: OOD 泛化对比中对照组（43.2% vs 76.1%）
- [[GR00T-N1]]: LIBERO 上的机器人基础模型对比

### 方法相关

- [[Cross-Modality Forcing]]: 本文提出的核心训练技术
- [[Visual Stream Dropout]]: 实现计算灵活性的关键机制
- [[VAE]]: Wan-2.2 视频 VAE，同时编码 RGB 和 Pointmap
- [[Action Chunking]]: 动作输出形式

### 硬件/数据相关

- [[YAM]]: 真实评估使用的双臂机器人平台
- [[AgiBotWorld|AGIBOT World-Beta]]: 预训练数据集（~500h）
- [[RoboTwin]]: 仿真评估 benchmark
- [[LIBERO]]: 标准操作 benchmark

---

## 速查卡片

> [!summary] Flex-π (arXiv 2608.10860)
> - **核心**: 多流 WAM，RGB + Pointmap + DINO 联合监督，单 checkpoint 56 种部署组合
> - **方法**: Cross-Modality Forcing + Visual Stream Dropout + 冻结 Wan-VAE 点图编码
> - **结果**: RoboTwin 94.6%，LIBERO 99.2%，真实任务比最强基线高 2.3×，半数据超全数据基线
> - **代码**: [github.com/geyan21/flex-pi](https://github.com/geyan21/flex-pi)

---

*笔记创建时间: 2026-08-13*

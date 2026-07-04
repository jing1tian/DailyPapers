---
title: "VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation"
method_name: "VT-WAM"
authors: [Shuai Tian, Yupeng Zheng, Yuhang Zheng, Songen Gu, Yujie Zang, Yuxing Qin, Weize Li, Haoran Li, Wenchao Ding, Dongbin Zhao]
year: 2026
venue: arXiv
tags: [world-action-model, tactile-sensing, contact-rich-manipulation, flow-matching, mixture-of-transformers, visuomotor, attention-mechanism]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.02503
created: 2026-07-04
---

# 论文笔记：VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | SKL-MAIS, 中科院自动化所; UCAS; TARS Robotics; 新加坡国立大学; 复旦大学 |
| 日期 | July 2026 |
| 项目主页 | [vt-wam.github.io](https://vt-wam.github.io/) |
| 对比基线 | [[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.02503) / Code: Coming Soon |

---

## 一句话总结

> 通过非对称 MoT 注意力将视觉锚帧与时序触觉动态解耦，并用接触门控的辅助损失强制动作查询在接触阶段优先关注触觉，将接触密集型操作平均成功率从 45% 提升至 71.67%。

---

## 核心贡献

1. **非对称 MoT 注意力（Asymmetric MoT Attention）**: 动作 token 访问完整触觉序列以建模接触动态，但只访问首帧视觉锚帧作为场景上下文，从而在推理时跳过未来视觉预测，大幅降低计算开销。
2. **接触门控 AVTAG（Contact-Gated AVTAG）**: 一个仅在训练阶段生效的 hinge ranking 辅助损失，在接触阶段强制动作查询的注意力权重中触觉占比超过视觉，对抗视觉信号天然密集导致的 attention 偏置。
3. **联合 Flow Matching 框架**: 在同一 [[Flow Matching]] 目标下同时学习视觉预测、触觉形变预测和动作预测，使触觉动态直接编码进控制策略。

---

## 问题背景

### 要解决的问题

接触密集型操作（擦拭、插入、刷卡）需要精细的力反馈，而纯视觉策略无法感知微小形变和法向力。[[WAM]] 类方法虽然联合预测未来状态和动作，但尚未将**触觉形变动态**纳入联合建模框架。

### 现有方法的局限

- **直接拼接型触觉策略**（DP+Tactile、OmniVTLA）：将触觉 token 简单拼入输入序列，在联合训练时被密集视觉信号淹没，触觉注意力权重极低。
- **视觉优先的 WAM**（[[Flash-WAM]]）：完整预测未来视觉帧，推理代价大；完全忽视触觉模态，在接触阶段缺乏信息来源。
- **[[Visuo-Tactile World Models]]** 等现有视触觉世界模型：证明了触觉有预测价值，但未解决注意力层面的路由问题——哪些层应允许触觉影响动作生成。

### 本文的动机

触觉响应**时序上极度稀疏**——仅在接触发生的短暂窗口内有效。若将触觉 token 均匀参与所有注意力路径，非接触阶段的噪声反而干扰视觉特征。VT-WAM 的核心洞察是：用**非对称掩码**将触觉序列路由到动作分支（需要它），同时隔离视觉预测分支（不需要它）；再用**接触门控损失**确保模型在训练时真正学到这种优先级。

---

## 方法详解

### 模型架构

VT-WAM 采用 **[[Mixture-of-Transformers|MoT]] Transformer** 架构，三路专家共享骨干但通过非对称掩码控制信息流：

- **输入**: 语言指令 $l$ + 腕部相机序列 $\mathbf{V}_{1:T}$ + 触觉形变场序列 $\mathbf{T}_{1:T}$ + 本体感知状态 $s_t$
- **视觉专家**: 使用 [[Wan2.2]] Video VAE 编码为视觉 token，分为**首帧锚帧** $V_1$ 和**未来帧** $V_{2:T}$
- **触觉专家**: 预训练触觉 VAE 将双传感器 3D 形变场编码为触觉 token 序列 $T_{1:T}$
- **动作专家**: 线性投影 [[Action Chunking|动作块]] $a_{t:t+k}$ 为动作 token
- **跨模态条件**: 语言 $l$ 和本体感知 $s_t$ 通过交叉注意力接入三个专家

### 核心模块

#### 模块 1: 非对称 MoT 注意力（Asymmetric MoT Attention）

**设计动机**: 传统对称 [[MoT]] 允许所有 token 互相访问，但动作预测不需要未来视觉帧（推理时不可用），视觉预测也不需要触觉（模态无关）。非对称设计在**训练阶段**模拟**推理阶段的信息可用性**。

**具体实现**（三类查询，一套 K/V）：

联合查询矩阵 $\mathbf{Q}^{(l)}$ 由三路专家查询拼接：

$$
\mathbf{Q}^{(l)} = [\mathbf{Q}_v^{(l)};\, \mathbf{Q}_t^{(l)};\, \mathbf{Q}_a^{(l)}]
$$

注意力输出通过块状掩码矩阵 $\mathbf{M}$ 控制：

$$
\mathbf{P}^{(l)} = \text{Softmax}\!\left(\frac{\mathbf{Q}^{(l)}(\mathbf{K}^{(l)})^\top}{\sqrt{d}} + \mathbf{M}\right), \quad \mathbf{Y}^{(l)} = \mathbf{P}^{(l)}\mathbf{V}^{(l)}
$$

**掩码规则**（$-\infty$ 表示屏蔽）：

| 查询 \ 键值 | 视觉首帧 $V_1$ | 视觉未来帧 $V_{2:T}$ | 触觉序列 $T_{1:T}$ | 动作 token |
|------------|:---:|:---:|:---:|:---:|
| 视觉专家 $Q_v$ | ✓ | ✓ | ✗ | ✗ |
| 触觉专家 $Q_t$ | ✓ | ✗ | ✓ | ✗ |
| 动作专家 $Q_a$ | ✓ | ✗ | ✓ | ✓ |

- 视觉专家只看视觉 token（独立重建）
- 触觉专家看首帧视觉锚帧 + 触觉序列（场景先验 + 接触动态）
- 动作专家看首帧视觉锚帧 + 完整触觉序列（跳过未来视觉以匹配推理条件）

#### 模块 2: 接触门控 AVTAG（Contact-Gated Action-Visual-Tactile Attention Guidance）

**设计动机**: 即使有非对称掩码，视觉 token 数量远多于触觉 token，自然形成 attention 主导。接触阶段若动作查询仍偏重视觉，则触觉动态无法有效编码进策略。[[AVTAG]] 通过辅助损失在训练时纠正这种偏置。

**具体实现**:

用 stop-gradient 构造辅助注意力分布（不影响主干梯度流向 K/V）：

$$
\mathbf{P}_{\mathrm{vt}} = \text{Softmax}\!\left(\frac{\mathbf{Q}_a \cdot \mathrm{sg}(\mathbf{K}_{\mathrm{vt}})^\top}{\sqrt{d}}\right)
$$

对每个时间步 $r$，计算归一化的视觉占比 $p_v(r)$ 和触觉占比 $p_t(r)$：

$$
p_v(r) = \frac{\alpha_v(r)}{\alpha_v(r) + \alpha_t(r)}, \quad p_t(r) = \frac{\alpha_t(r)}{\alpha_v(r) + \alpha_t(r)}
$$

其中 $\alpha_v(r), \alpha_t(r)$ 分别为动作查询对视觉、触觉键的注意力权重之和。

接触集合 $\mathcal{C}$ 由明显触觉形变的时间步定义，仅对这些步施加 hinge 损失：

$$
\mathcal{L}_{\mathrm{AVTAG}} = \mathbb{E}_{r \in \mathcal{C}}\!\left[\max\!\left(0,\, p_v(r) - p_t(r)\right)\right]
$$

当触觉注意力超过视觉注意力后损失自动归零，不施加反向约束。

### 训练目标与高效推理

#### Flow Matching 训练目标

对三个模态分别定义速度场预测误差（$\mathbf{f}^* = x_1 - x_0$ 为目标流向量）：

$$
\mathcal{L}_v = \mathbb{E}\,\|\hat{\mathbf{f}}^v - \mathbf{f}_v^*\|^2, \quad \mathcal{L}_t = \mathbb{E}\,\|\hat{\mathbf{f}}^t - \mathbf{f}_t^*\|^2, \quad \mathcal{L}_a = \mathbb{E}\,\|\hat{\mathbf{f}}^a - \mathbf{f}_a^*\|^2
$$

联合 [[Flow Matching]] 损失：

$$
\mathcal{L}_{\mathrm{Flow}} = \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t + \lambda_a \mathcal{L}_a
$$

总训练损失（AVTAG 仅在训练阶段生效）：

$$
\mathcal{L}_{\mathrm{Train}} = \mathcal{L}_{\mathrm{Flow}} + \lambda_{\mathrm{AVTAG}} \mathcal{L}_{\mathrm{AVTAG}}
$$

**符号说明**:
- $\lambda_v, \lambda_t, \lambda_a$: 三个模态损失权重
- $\lambda_{\mathrm{AVTAG}}$: AVTAG 辅助损失权重
- $\hat{\mathbf{f}}^v, \hat{\mathbf{f}}^t, \hat{\mathbf{f}}^a$: 模型预测的视觉/触觉/动作速度场
- $\mathbf{f}_v^*, \mathbf{f}_t^*, \mathbf{f}_a^*$: 对应模态的目标流向量

#### 高效视觉缓存推理（Visual-Cache Inference）

训练时联合去噪三个模态；部署时切换至 **Visual-Cache Inference** 模式：
- 将当前视觉观测保存为首帧锚帧缓存
- 仅对触觉 latent 和动作 latent 去噪（10 步）
- 完全跳过未来视觉生成，节省计算

---

## 关键公式汇总

### 公式 1: [[Mixture-of-Transformers|非对称 MoT 注意力计算]]

$$
\mathbf{P}^{(l)} = \text{Softmax}\!\left(\frac{[\mathbf{Q}_v;\mathbf{Q}_t;\mathbf{Q}_a](\mathbf{K}^{(l)})^\top}{\sqrt{d}} + \mathbf{M}\right)
$$

**含义**: 三路查询共用一套 K/V，通过块状掩码矩阵 $\mathbf{M}$ 实现非对称信息路由

**符号说明**:
- $\mathbf{Q}_v, \mathbf{Q}_t, \mathbf{Q}_a$: 视觉、触觉、动作专家的查询矩阵
- $\mathbf{K}^{(l)}, \mathbf{V}^{(l)}$: 第 $l$ 层共享键值矩阵
- $\mathbf{M}$: 块状非对称掩码（$-\infty$ 屏蔽不需访问的 token）
- $d$: 注意力头维度

### 公式 2: [[AVTAG|AVTAG 辅助损失]]

$$
\mathcal{L}_{\mathrm{AVTAG}} = \mathbb{E}_{r \in \mathcal{C}}\!\left[\max\!\left(0,\, p_v(r) - p_t(r)\right)\right]
$$

**含义**: 在接触阶段集合 $\mathcal{C}$ 上，对视觉注意力超出触觉注意力的量施加 hinge 惩罚

**符号说明**:
- $\mathcal{C}$: 接触阶段时间步集合（由触觉形变幅度阈值划定）
- $p_v(r)$: 时间步 $r$ 处动作查询的归一化视觉注意力占比
- $p_t(r)$: 时间步 $r$ 处动作查询的归一化触觉注意力占比
- $\mathrm{sg}(\cdot)$: stop-gradient 操作（AVTAG 梯度不回传到 K/V）

### 公式 3: [[Flow Matching|联合 Flow Matching 总损失]]

$$
\mathcal{L}_{\mathrm{Train}} = \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t + \lambda_a \mathcal{L}_a + \lambda_{\mathrm{AVTAG}} \mathcal{L}_{\mathrm{AVTAG}}
$$

**含义**: 视觉/触觉/动作三模态速度场预测误差加权求和，叠加 AVTAG 注意力正则项

---

## 关键图表

### Figure 1: 触觉稀疏性与方法动机

![Figure 1](https://arxiv.org/html/2607.02503v1/x1.png)

**说明**: (a) 六个任务中触觉响应的时序分布——触觉仅在极短接触窗口内激活，信号时序上高度稀疏；(b) VT-WAM 将平均成功率从 [[Flash-WAM]] 的 45% 提升至 71.67%，逐任务对比。

### Figure 2: VT-WAM 整体架构

![Figure 2](https://arxiv.org/html/2607.02503v1/x2.png)

**说明**: (a) 三路专家的联合视觉-触觉-动作 [[Flow Matching]] 框架；(b) 训练阶段与推理阶段非对称 [[MoT]] 注意力掩码对比——推理时动作分支跳过未来视觉；(c) 接触门控 [[AVTAG]] 的 hinge ranking 损失示意，仅在接触阶段激活。

### Figure 3: 真实机器人实验平台

![Figure 3](https://arxiv.org/html/2607.02503v1/x3.png)

**说明**: 7-DoF xArm7 + Robotiq 2F-85 夹爪 + 腕部相机 + 双路 Xense 触觉传感器（gripper 两侧各一）。传感器测量接触法向力产生的 3D 形变场。

### Figure 4: 六项真实操作基准任务

![Figure 4](https://arxiv.org/html/2607.02503v1/x4.png)

**说明**: 分为两类——**表面交互**（Wipe Board、Wipe Vase、Peel Cucumber）和**受约束插入**（Insert Plug、Swipe Card、Insert Tube）。插入类任务需要精确的力控制，成功率明显低于表面交互类。

### Figure 5: 视觉-触觉预测可视化

![Figure 5](https://arxiv.org/html/2607.02503v1/pics/VTWAM_fig5.png)

**说明**: 跨六任务的腕部相机观测预测与触觉形变场预测。蓝色为 ground truth，橙色为预测。VT-WAM 对触觉形变场的捕捉明显优于 exUMI 和 UVA（余弦相似度 0.749 vs 0.618/0.667）。

### Figure 6: AVTAG 对触觉注意力的提升效果

![Figure 6](https://arxiv.org/html/2607.02503v1/pics/VTWAM_fig6.png)

**说明**: 花瓶擦拭任务中接触恢复阶段，红色曲线（触觉注意力占比）与蓝色曲线（视觉注意力占比）随接触力（虚线）的变化。加入 [[AVTAG]] 后，接触阶段触觉注意力明显超过视觉，使模型能根据力反馈实时调整策略。

### Table 1: 六任务真实操作成功率对比

| Method | Wipe Board | Wipe Vase | Peel Cuc. | **Surface Avg.** | Insert Plug | Swipe Card | Insert Tube | **Insertion Avg.** | **Overall** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DP + Tactile | 30% | 20% | 25% | 25.00% | 5% | 35% | 15% | 18.33% | 21.67% |
| RDP | 45% | 60% | 40% | 48.33% | 15% | 35% | 10% | 20.00% | 34.17% |
| π₀.₅ | 40% | 35% | 35% | 36.67% | 30% | 45% | 10% | 28.33% | 32.50% |
| OmniVTLA | 45% | 30% | 25% | 33.33% | 40% | 35% | 40% | 38.33% | 35.83% |
| Fast-WAM | 70% | 55% | 45% | 56.67% | 20% | 55% | 25% | 33.33% | 45.00% |
| **VT-WAM** | **90%** | **85%** | **70%** | **81.67%** | **60%** | **70%** | **55%** | **61.67%** | **71.67%** |

**关键发现**: VT-WAM 在所有方法中均占优，对 OmniVTLA（最强触觉基线）超出 35.84 个百分点。WAM 基线 Fast-WAM 同样明显优于非 WAM 方法，印证了联合状态-动作建模的有效性。

### Table 2: 触觉形变预测质量

| Method | L₂ Error ↓ | Cosine Similarity ↑ |
|--------|:---:|:---:|
| exUMI | 0.091 | 0.618 |
| UVA | 0.083 | 0.667 |
| **VT-WAM** | **0.077** | **0.749** |

**关键发现**: VT-WAM 的触觉预测不仅服务于动作控制，其预测精度本身也优于专用触觉预测方法，验证了联合建模不牺牲预测质量。

### Table 3: 消融实验——触觉动态建模与注意力引导

| 模型 | 描述 | Wipe Vase | Insert Tube |
|------|------|:---:|:---:|
| M₀ | Fast-WAM（基线） | 55% | 25% |
| M₁ | M₀ + 对称 MoT（全触觉序列） | 65% | 40% |
| M₂ | M₀ + 非对称（仅首帧触觉） | 40% | 30% |
| M₃ | M₀ + 非对称（全触觉序列） | 70% | 50% |
| M₄ | **VT-WAM = M₃ + AVTAG** | **85%** | **55%** |

**关键发现**:
1. M₂ vs M₃：动作分支必须访问**完整触觉序列**（而非单帧），动作预测从 40% 跳至 70%（Wipe Vase）
2. M₁ vs M₃：**非对称**掩码优于对称掩码，隔离视觉未来帧避免训推不一致
3. M₃ vs M₄：AVTAG 进一步提升 15 pp（Wipe Vase），是接触恢复能力的关键

---

## 实验

### 数据集与平台

| 项目 | 规模 / 配置 |
|------|------------|
| 机器人 | 7-DoF xArm7 + Robotiq 2F-85 |
| 触觉传感器 | 双路 Xense（左右夹指各一，3D 形变场输出） |
| 相机 | 腕部单目相机 |
| 数据收集 | 每任务 100 条专家轨迹（运动学示教） |
| 测试次数 | 每任务 20 次（每条件 ≥ 3 次）|

### 实现细节

- **视觉编码**: [[Wan2.2]] Video VAE（冻结权重）
- **触觉编码**: 预训练触觉 VAE（冻结权重）
- **去噪步数**: 推理时 10 步 Flow Matching 去噪
- **推理模式**: Visual-Cache Inference（仅去噪触觉+动作 latent）
- **硬件**: NVIDIA A100 GPU

---

## 批判性思考

### 优点

1. **架构设计优雅**: 非对称掩码同时解决了"训推不一致"和"触觉-视觉注意力竞争"两个问题，且推理时计算量显著低于完整 WAM
2. **AVTAG 轻量有效**: 仅增加一个辅助 loss，无额外推理成本，却带来 15 pp 的显著提升（消融验证）
3. **双指标验证**: 同时评估操作成功率和触觉预测精度，说明联合建模不以牺牲预测质量换取控制性能

### 局限性

1. **数据规模有限**: 每任务仅 100 条轨迹，泛化性存疑；商业 Xense 传感器较难普及复现
2. **任务多样性受限**: 六项任务均在桌面场景，缺少移动操作或动态干扰场景
3. **代码未开源**: 目前无法复现，可复现性差

### 潜在改进方向

1. 将触觉 VAE 替换为通用触觉编码器，支持多种传感器类型（GelSight、DIGIT 等）
2. 探索接触集合 $\mathcal{C}$ 的自动发现（而非依赖形变幅度阈值）
3. 在多指灵巧手上验证方法扩展性

### 可复现性评估

- [ ] 代码开源（Coming Soon）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（论文中有较详细描述）
- [ ] 数据集可获取（未发布）

---

## 关联笔记

### 基于
- [[WAM]]: 世界动作模型框架，VT-WAM 是其视触觉扩展
- [[Flash-WAM]]: 直接基线与 visual-cache 推理思路来源
- [[Flow Matching]]: 三模态联合训练目标
- [[Visuo-Tactile World Models]]: 视触觉世界建模的前序验证工作

### 对比
- [[Flash-WAM]]: 无触觉的 WAM 基线，成功率 45% vs 71.67%
- [[Tactile-WAM]]: 同期提交的另一篇触觉 WAM 工作（arXiv:2606.26663）

### 方法相关
- [[Mixture-of-Transformers]]: 非对称 MoT 注意力的基础架构
- [[AVTAG]]: 接触门控注意力引导，VT-WAM 的 contact-gated 版本
- [[Flow Matching]]: 训练目标
- [[Wan2.2]]: 视觉 VAE 编码器来源
- [[Action Chunking]]: 动作表示方式

### 硬件/数据相关
- xArm7: 7-DoF 机械臂实验平台
- Xense 触觉传感器: 双路 3D 形变场输出传感器

---

## 速查卡片

> [!summary] VT-WAM
> - **核心**: 非对称 MoT 注意力 + Contact-Gated AVTAG，将触觉动态纳入 WAM 框架
> - **方法**: 动作 token 访问完整触觉序列 + 首帧视觉锚帧；AVTAG hinge loss 强制接触阶段触觉注意力 > 视觉注意力
> - **结果**: 六任务平均成功率 71.67%（vs Fast-WAM 45%，vs OmniVTLA 35.83%）
> - **代码**: Coming Soon — https://vt-wam.github.io/

---

*笔记创建时间: 2026-07-04*

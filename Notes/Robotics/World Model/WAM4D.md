---
title: "WAM4D: Fast 4D World Action Model via Spatial Register Tokens"
method_name: "WAM4D"
authors: [Ying Li, Xiaobao Wei, Jiajun Cao, Hao Wang, Xiaowei Chi, Chengyu Bai, Qianpu Sun, Jiajun Li, Xiaojie Zhang, Jian Tang, Sirui Han, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, 3d-spatial-consistency, depth-estimation, robot-manipulation, knowledge-distillation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.14048v1
created: 2026-06-16
---

# 论文笔记：WAM4D: Fast 4D World Action Model via Spatial Register Tokens

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, HKUST, Beijing Innovation Center of Humanoid Robotics |
| 日期 | June 2026 |
| 项目主页 | https://github.com/myendless1/wam4d |
| 对比基线 | [[LingBot-VA]], [[Fast-WAM]], [[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.14048) / [Code](https://github.com/myendless1/wam4d) |

---

## 一句话总结

> WAM4D 通过在训练期间引入轻量级 [[空间寄存器 Token|Spatial Register Token]]，将几何基础模型先验蒸馏进 [[World Action Model|WAM]]，推理时移除几何分支，实现 3D 空间一致的高效机器人操作。

---

## 核心贡献

1. **空间寄存器蒸馏（Spatial Register Distillation）**: 设计可学习的寄存器 token，在训练时作为未来深度读出器，将预训练几何基础模型（[[Depth Anything 3|DA3-GIANT-1.1]]）的几何先验迁移至视频-动作 Transformer，推理时完全移除，不引入额外开销。
2. **因果混合注意力（Causal Mixture Attention）**: 为 [[Mixture-of-Transformers|MoT]] 骨干设计模态专属可见性掩码，定义视频、动作、几何三类 token 间的因果依赖关系，防止非因果信息捷径。
3. **推理高效性**: 部署时仅保留视频-动作流，以 525ms 延迟和 9.71 GiB VRAM 实现与 LingBot-VA（843ms / 12.97 GiB）媲美的操作性能。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model|WAM]] 在 2D 视频空间或潜在空间中进行联合预测，忽略了精密操作任务所必需的 3D 几何约束：视觉上合理的视频预测可能隐藏接触几何误差，直接影响空间精确操作。

### 现有方法的局限

- **TesserAct**：将几何投影为 4D 场景表示，计算开销大
- **Kinema4D**：拼接几何与视觉 token，在骨干网络中处理，推理延迟高
- **X-WAM**：显式合成 RGB-D 未来帧，需要完整几何解码路径
- 上述方法均在推理时保留几何分支，增加延迟和显存消耗

### 本文的动机

几何信息只在训练时约束内部表示即可——推理时可以依赖从视频特征中隐式习得的 3D 感知能力。寄存器 token 作为训练-部署之间的"紧凑桥梁"，蒸馏后被丢弃。

---

## 方法详解

### 模型架构

WAM4D 采用 **[[Mixture-of-Transformers|MoT]] 视频-动作因果架构**：

- **输入**: 语言指令 $l$ + RGB 历史帧 $O^{hist}_t$ + 历史动作 $a_{t-L_a:t-1}$
- **Backbone**: 基于 [[LingBot-VA]] 的因果视频-动作 Transformer，扩展为 MoT
- **核心模块**: [[空间寄存器 Token|Spatial Register Token]] 用于[[深度估计|未来深度预测]]，[[因果混合注意力|Causal Mixture Attention]] 用于 token 间可见性控制
- **输出**: 动作块 $a_{t:t+T_a}$（16D 向量：3D 位置 + 四元数 + 夹爪值，双臂）
- **推理阶段**: 移除寄存器分支和深度块，仅保留视频-动作流

### 核心模块

#### 模块 1: Spatial Register Distillation（空间寄存器蒸馏）

**设计动机**: 利用[[知识蒸馏|Knowledge Distillation]]将预训练几何基础模型先验迁移至视频 Transformer，无需推理时的几何解码开销。

**具体实现**:

1. 定义寄存器网格（12×10），在多视角拼图（3 视角）坐标系下重复至每个未来深度时间步 $t \in \mathcal{T}_d$
2. 通过深度提取块（交替 [[Self-Attention|自注意力]] + [[Cross-Attention|交叉注意力]]）更新寄存器状态，交叉注意力以历史视频特征为键值对
3. 寄存器特征投影至几何头（[[Depth Anything 3|DA3-GIANT-1.1]]）的输入空间
4. 预测未来深度图；[[SmoothL1 Loss|Smooth L1 损失]]反向传播至视频特征，注入几何约束
5. 推理时移除整个寄存器-深度路径

#### 模块 2: Causal Mixture Attention（因果混合注意力）

**设计动机**: MoT 骨干中三类 token（视频、动作、几何）需要差异化的因果关系，防止未来几何 token 泄露因果信息给动作预测。

**可见性规则**:

| Token 类型 | 可见范围 |
|-----------|---------|
| 未来动作 token | 历史视频/动作 + 同时步未来动作（不可见未来视频和寄存器） |
| 寄存器 token | 自身 + 历史视频（不可见未来视频和动作） |
| 未来视频 token | 历史上下文 + 同时步及之前未来视频帧 |

这种设计确保动作预测不从几何分支获取非因果信息捷径。

---

## 关键公式

### 公式 1: [[World Action Model|因果上下文定义]]

$$
\mathcal{C}_t = \{l,\ O^{hist}_t,\ a_{t-L_a:t-1}\}
$$

**含义**: WAM4D 的因果上下文由语言指令、RGB 历史观测、历史动作序列组成；未来帧和动作仅作为预测目标，不参与输入。

**符号说明**:
- $l$: 语言指令（任务描述）
- $O^{hist}_t$: 时刻 $t$ 的 RGB 历史观测序列
- $a_{t-L_a:t-1}$: 过去 $L_a$ 步的历史动作序列

### 公式 2: [[空间寄存器 Token|寄存器深度预测]]

$$
\mathbf{G}_t = \mathcal{P}_g\!\left(\{\mathbf{R}^{\ell+1}_t\}_{\ell \in \mathcal{L}_r}\right),\quad \hat{\mathbf{D}}^{fut}_t = \mathcal{G}_\phi(\mathbf{G}_t)
$$

**含义**: 从中间层寄存器特征中聚合并投影，送入冻结/微调的几何头 $\mathcal{G}_\phi$ 预测未来深度图。

**符号说明**:
- $\mathbf{R}^{\ell+1}_t$: 第 $\ell$ 个深度提取块输出的寄存器 token（时刻 $t$）
- $\mathcal{L}_r$: 寄存器所在的 Transformer 层集合（中间层：12, 14, 16, 18）
- $\mathcal{P}_g$: 投影到几何头输入空间的线性层
- $\mathcal{G}_\phi$: 预训练几何基础模型（[[Depth Anything 3|DA3-GIANT-1.1]]），可微调

### 公式 3: [[SmoothL1 Loss|深度监督损失]]

$$
\mathcal{L}_{depth} = \frac{1}{\sum_\tau |\Omega_\tau|} \sum_\tau \sum_{p \in \Omega_\tau} \text{SmoothL1}\!\left(\hat{D}_{\tau,p},\ D_{\tau,p}\right)
$$

**含义**: 在所有有效深度像素上计算预测深度与真实深度的 Smooth L1 损失，指导寄存器 token 习得几何感知能力。

**符号说明**:
- $\tau$: 未来深度预测时间步索引
- $\Omega_\tau$: 时刻 $\tau$ 的有效深度像素集合
- $\hat{D}_{\tau,p}$: 预测深度（像素 $p$，时刻 $\tau$）
- $D_{\tau,p}$: 真实深度标注

### 公式 4: [[World Action Model|综合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{video} + \lambda_{act} \cdot \mathcal{L}_{action} + \lambda_{depth} \cdot \mathcal{L}_{depth}
$$

**含义**: 同时优化未来视频预测、机器人动作预测和深度几何监督三个目标。

**符号说明**:
- $\mathcal{L}_{video}$: 未来 RGB 帧重建损失（通过 [[VAE]] 潜在空间）
- $\mathcal{L}_{action}$: 动作预测损失
- $\mathcal{L}_{depth}$: 深度监督损失（见公式 3）
- $\lambda_{act} = 1$, $\lambda_{depth} = 1$: 损失权重（默认值）

---

## 关键图表

### Figure 1: 4D World-Action 设计对比

![Figure 1](https://arxiv.org/html/2606.14048v1/x1.png)

**说明**: 对比四种 4D WAM 设计范式。TesserAct 将几何投影为 4D 场景；Kinema4D 拼接几何与视觉 token；X-WAM 使用模态适配进行显式 RGB-D 合成；WAM4D 通过训练时寄存器蒸馏、推理时移除几何分支，实现最轻量部署。

### Figure 2: WAM4D 架构与因果可见性

![Figure 2](https://arxiv.org/html/2606.14048v1/x2.png)

**说明**: 展示 WAM4D 的 [[Mixture-of-Transformers|MoT]] 骨干结构和三类 token（视频、动作、寄存器）的[[因果混合注意力|因果混合注意力]]可见性掩码设计。

### Figure 3: 训练路径 vs 推理路径

![Figure 3](https://arxiv.org/html/2606.14048v1/x3.png)

**说明**: 训练时包含寄存器-深度路径，深度损失反向传播约束视频特征；推理时彻底移除该分支，仅保留视频-动作流，延迟和显存显著降低。

### Figure 4: AstriBot S1 真实机器人任务

![Figure 4](https://arxiv.org/html/2606.14048v1/x4.png)

**说明**: 真实世界评估平台 AstriBot S1 双臂机器人，4 个接触密集任务：抓盘子（Plate）、拿瓶子（Bottle）、搭乐高（Lego）、插笔（Pen），每任务 100 次演示数据，10 次 rollout 评估。

### Figure 5: 空间寄存器注意力可视化

![Figure 5](https://arxiv.org/html/2606.14048v1/x5.png)

**说明**: RoboTwin 随机化样本上的寄存器注意力热力图。物体寄存器跨视角关注相关图像区域，夹爪寄存器专注于初始位姿——验证寄存器 token 确实习得了有意义的几何感知而非随机激活。

### Figure 6: RGB-D 与点云 Rollout 可视化

![Figure 6](https://arxiv.org/html/2606.14048v1/x6.png)

**说明**: 从单帧初始化开始，WAM4D 自回归预测未来 RGB 帧和深度图，并重建点云序列，展示时序一致的 4D 世界状态预测能力（该路径独立于闭环控制部署路径）。

### Figure 7: 失败案例——长程遮挡导致物体身份漂移

![Figure 7](https://arxiv.org/html/2606.14048v1/x7.png)

**说明**: 长自回归 rollout 中，遮挡后物体可能被补全为视觉合理但不同的物体（无显式长期记忆）。该局限不影响闭环控制性能（真实观测连续更新），但限制了离线 4D 预测的忠实度。

---

### Table 1: RoboTwin 2.0 仿真基准（50 任务平均）

| Method | Clean SR (%) | Random SR (%) | Avg SR (%) | 延迟 (ms) | VRAM (GiB) |
|--------|-------------|--------------|-----------|----------|-----------|
| π₀ | 65.9 | 58.4 | 62.2 | 64.16 | 8.45 |
| π₀.₅ | 82.7 | 76.8 | 79.8 | 72.03 | 8.45 |
| Motus | 88.7 | 87.0 | 87.9 | 1516.30 | 11.55 |
| LingBot-VA | 92.9 | 91.6 | 92.3 | 843.57 | 12.97 |
| Fast-WAM | 91.9 | 91.8 | 91.8 | 425.53 | 11.55 |
| **WAM4D** | **93.8** | **89.9** | **91.8** | 525.43 | **9.71** |

**说明**: WAM4D 在 Clean 场景成功率最高（93.8%），整体性能与 Fast-WAM 持平（91.8%），但使用更少显存（9.71 vs 11.55 GiB）。

### Table 2: 真实机器人实验（AstriBot S1，成功率）

| Method | Plate | Bottle | Lego | Pen | Avg |
|--------|-------|--------|------|-----|-----|
| π₀.₅ | 1.0 | 0.8 | 0.7 | 0.74 | 0.74 |
| LingBot-VA | 1.0 | 1.0 | 1.0 | 0.84 | 0.84 |
| Fast-WAM | 0.9 | 1.0 | 0.8 | 0.80 | 0.80 |
| **WAM4D** | 0.9 | 0.9 | 1.0 | **0.90** | **0.90** |

**说明**: WAM4D 在接触密集的插笔任务（Pen，0.90）和搭乐高任务（Lego，1.0）上表现最佳，平均成功率 0.90 优于所有基线，体现 3D 空间感知对精密操作的价值。

### Table 3: 空间寄存器层位置消融（Table 7）

| 寄存器层 | FVD↓ | Clean SR↑ | AbsRel↓ |
|---------|------|-----------|---------|
| 浅层 (2,4,6,8) | **168.8** | 73.1% | 0.061 |
| **中间层 (12,14,16,18)** | 183.2 | 75.2% | **0.053** |
| 深层 (22,24,26,28) | 201.5 | 72.0% | 0.058 |
| 双向注意力 | 175.6 | **76.6%** | 0.055 |

**关键发现**: 中间层寄存器在控制性能（75.2% SR）和几何质量（AbsRel 0.053）间取得最佳权衡；单向（因果）优于双向，防止几何信息捷径。

### Table 4: 几何头设计消融（Table 8）

| 变体 | Clean SR↑ | AbsRel↓ | F-score↑ |
|-----|-----------|---------|---------|
| 无深度监督 | 71.7% | — | — |
| 随机初始化 | 70.0% | 0.059 | 0.589 |
| 预训练固定 | 75.2% | 0.053 | 0.685 |
| **预训练可微调** | **80.1%** | **0.049** | **0.710** |

**关键发现**: 可微调的预训练几何头带来最大增益（+8.4% SR），验证几何基础模型先验的有效迁移；随机初始化反而略低于无深度监督基线，说明几何先验而非结构本身是关键。

### Table 5: 推理计算开销对比（Table 9，单卡 A800 80GB）

| Method | 延迟 (ms) | VRAM (GiB) |
|--------|----------|-----------|
| Motus | 1516.30 | 11.55 |
| LingBot-VA | 843.57 | 12.97 |
| **WAM4D** | 525.43 | **9.71** |
| Fast-WAM | **425.53** | 11.55 |

**说明**: WAM4D 推理延迟高于 Fast-WAM（因 MoT 骨干更重），但 VRAM 最少，显示推理时移除几何分支的显存收益显著。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 50 任务 × (50 clean + 500 random) 轨迹 | 仿真，含干净/随机化场景 | 训练 + 测试 |
| AstriBot S1 Real | 4 任务 × 100 次演示 | 真实双臂机器人，接触密集 | 真实世界测试 |

### 实现细节

- **视频帧采样**: 最多 17 帧；8 个预测目标每 4 步采样一次
- **动作表示**: 16D 向量（每臂：3D 位置 + 四元数 + 夹爪值，双臂）
- **寄存器网格**: 12×10，3 视角拼图布局
- **几何头**: [[Depth Anything 3|DA3-GIANT-1.1]]（预训练，可微调）
- **寄存器层**: 中间层 Transformer 层（12, 14, 16, 18）
- **损失权重**: $\lambda_{act} = 1$，$\lambda_{depth} = 1$
- **硬件**: 单卡 NVIDIA A800 80GB

### 可视化结果

空间寄存器注意力热力图（Figure 5）显示：寄存器 token 自动发现了语义相关的图像区域，物体寄存器跨多视角关注操作目标，夹爪寄存器专注于末端执行器位置——表明几何感知以可解释方式嵌入视频表示。

---

## 批判性思考

### 优点

1. **训练-推理解耦优雅**: 几何蒸馏在训练时注入先验，推理时零开销，架构设计简洁
2. **真实任务验证**: 在接触密集的双臂真实机器人任务上获得 SOTA，不仅限于仿真
3. **显存高效**: 9.71 GiB VRAM 低于所有对比基线，有实际部署价值
4. **几何感知可解释**: 注意力可视化证明寄存器 token 学到了有意义的几何语义

### 局限性

1. **无长期物体记忆**: 长自回归 rollout 中遮挡后物体身份可能漂移（Figure 7），影响离线 4D 预测质量
2. **延迟略高于 Fast-WAM**: 525ms vs 425ms，MoT 骨干引入额外开销
3. **依赖 RoboTwin 2.0 深度标注**: 训练需要精确深度监督，真实场景需要深度传感器或伪标注

### 潜在改进方向

1. **持久化物体记忆**: 引入场景状态跟踪或 3D 记忆库，解决长程遮挡问题
2. **无监督深度蒸馏**: 利用自监督深度估计，减少对标注深度数据的依赖
3. **寄存器 token 压缩**: 减少寄存器数量或引入动态稀疏激活，进一步降低训练开销

### 可复现性评估

- [x] 代码开源（https://github.com/myendless1/wam4d）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（层位置、采样策略、损失权重均有说明）
- [ ] 数据集可获取（RoboTwin 2.0 数据集可获取，真实数据集未公开）

---

## 关联笔记

### 基于

- [[LingBot-VA]]: WAM4D 在其因果视频-动作架构上扩展，引入 MoT 骨干和寄存器蒸馏
- [[Depth Anything 3]]: 预训练几何基础模型，作为被蒸馏的几何头 $\mathcal{G}_\phi$

### 对比

- [[Fast-WAM]]: 同样追求推理效率的 WAM，延迟更低但无 3D 几何约束
- [[LingBot-VA]]: 基础 WAM 架构，WAM4D 的前身，延迟和显存均更高
- [[π₀.₅]]: VLA 基线，无世界模型预测，成功率较低

### 方法相关

- [[空间寄存器 Token]]: 本文核心创新，训练时几何蒸馏桥梁
- [[Mixture-of-Transformers]]: WAM4D 使用的 MoT 骨干
- [[知识蒸馏]]: 将几何基础模型先验蒸馏进视频 Transformer 的核心思想
- [[因果混合注意力]]: 本文设计的模态专属可见性掩码机制
- [[SmoothL1 Loss]]: 深度监督损失函数

### 硬件/数据相关

- [[AstriBot S1]]: 真实世界评估使用的双臂机器人平台
- [[RoboTwin 2.0]]: 主要仿真评估基准

---

## 速查卡片

> [!summary] WAM4D: Fast 4D World Action Model via Spatial Register Tokens
> - **核心**: 训练时用空间寄存器 token 蒸馏几何先验，推理时移除几何分支
> - **方法**: MoT 骨干 + Spatial Register Distillation + Causal Mixture Attention
> - **结果**: RoboTwin 2.0 Clean 93.8% SR；真实机器人 Avg 0.90；9.71 GiB VRAM
> - **代码**: https://github.com/myendless1/wam4d

---

*笔记创建时间: 2026-06-16*

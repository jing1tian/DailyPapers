---
title: "Next Forcing: Causal World Modeling with Multi-Chunk Prediction"
method_name: "NextForcing"
authors: [Gangwei Xu, Qihang Zhang, Jiaming Zhou, Xing Zhu, Yujun Shen, Xin Yang, Yinghao Xu]
year: 2026
venue: arXiv
tags: [world-model, autoregressive-generation, video-prediction, multi-chunk-prediction, robot-manipulation]
zotero_collection: 3-Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.11187
created: 2026-06-11
---

# 论文笔记：Next Forcing: Causal World Modeling with Multi-Chunk Prediction

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Robbyant, HUST, HKUST, HKUST (GZ) |
| 日期 | June 2026 |
| 项目主页 | [gangweix.github.io/next-forcing](https://gangweix.github.io/next-forcing/) |
| 对比基线 | [[Flash-WAM]], [[LingBot-VA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.11187) |

---

## 一句话总结

> Next Forcing 通过多块预测（MCP）框架让自回归视频世界模型同时预测多个未来帧块，消除标准 Teacher Forcing 的近视监督问题，在机器人操作 benchmark 上实现 2.3× 训练加速和 2× 推理加速。

---

## 核心贡献

1. **发现并定义近视监督问题**: 标准 [[Teacher Forcing]] 仅监督下一帧块导致模型走捷径（复制相邻帧），在高帧率（50 fps）时问题最为严重
2. **多块预测（MCP）框架**: 引入轻量辅助模块同时预测 next¹/next²/next³ 多个未来块，形成跨预测深度的因果链，提供密集多尺度时序监督
3. **2× 推理加速**: 保留 MCP 模块在推理时并行预测下一块，无需额外等待，达到与标准推理相当的精度

---

## 问题背景

### 要解决的问题

[[自回归视频生成]] 中的 **近视监督（Myopic Supervision）** 问题：标准 [[Teacher Forcing]] 范式仅训练模型预测紧接的下一个视频块，而相邻块视觉上高度相似，模型可以通过"外观捷径"（appearance shortcuts）最小化损失，而不需要学习真正的时序动力学。

### 现有方法的局限

- **LingBot-VA / Fast-WAM** 等 WAM（World Action Model）方法使用标准 next-chunk 去噪训练，在 12 fps 时勉强可行，但在 **50 fps 高帧率** 下收敛极慢（5k 步仅 45.5%）
- **Diffusion Forcing** 修改噪声调度，**Self Forcing** 修改上下文策略，但都未改变预测目标本身
- 在高帧率下相邻帧差异趋近于零，近视监督问题最为突出

### 本文的动机

若能让模型同时预测多个**更远的**未来块，则模型被迫学习长程时序依赖，无法通过简单复制获益。类比 LLM 中的 [[Multi-Token Prediction]]，但连续视频潜变量需要 [[Flow Matching|迭代去噪]]，需要全新的技术设计。

---

## 方法详解

### 模型架构

Next Forcing 基于 **Wan2.2 Transformer**（30 层）骨干，在其上增加三个轻量 MCP 模块：

- **输入**: 语言指令 $\ell$ + 历史视频块 $\mathbf{x}_0^{1:i-1}$（含带噪历史增强）+ 当前带噪块 $\mathbf{x}_t^{(i)}$
- **Backbone**: Wan2.2 Transformer，30 层，VAE 编码潜空间操作
- **核心模块**: [[Multi-Chunk Prediction|MCP 辅助模块]]，3 个深度，各 3 个 transformer block
- **输出（主模型）**: 当前块的去噪预测；**MCP 输出**: 未来 k=1,2,3 块的预测
- **总参数**: 主模型 Wan2.2 规模，MCP 模块显著轻量（3 × 3 blocks）

```
输入
 └─ Wan2.2 主干（30 层）
       ├─ 主模型输出：预测当前块 x₀^(i)
       └─ 中间层特征 {h_4, h_12, h_20, h_30}
             └─ MLP Fusion → h_fuse
                   ├─ MCP Depth-1 模块 → 预测 next¹ 块
                   ├─ MCP Depth-2 模块（接 Depth-1 输出）→ 预测 next² 块
                   └─ MCP Depth-3 模块（接 Depth-2 输出）→ 预测 next³ 块
```

### 核心模块

#### 模块 1: 时序块偏移（Temporal Chunk Shifting）

**设计动机**: 利用 [[Flow Matching]] 的训练目标，将预测目标向未来偏移 k 块，强制模型学习长程时序依赖

**具体实现**:
- 对第 i 个块，生成偏移目标 $\mathbf{x}_0^{[k][i]}$，指向未来第 min(i+k, F) 块
- k=1,2,3 分别对应三个 MCP 深度
- 使用 [[Rotary Position Embedding|RoPE]] 位置编码调整：将 MCP 目标的位置编码设置为 RoPE(i+k) 而非 RoPE(i)，保持时序一致性

#### 模块 2: 独立噪声注入（Independent Noise Injection）

**设计动机**: 防止 MCP 模块直接"复制"主模型特征，强制其依赖主模型的语义表征

**具体实现**:
- 每个 MCP 深度的偏移目标使用**独立噪声** $\boldsymbol{\varepsilon}_k$，与主模型噪声解耦
- MCP 时间步偏移 $s_{\text{mcp}} = 10$，高于主模型 $s_{\text{main}} = 5$，更高噪声迫使 MCP 更依赖主模型特征
- 消融实验验证：$s_{\text{mcp}} = 5$（与主模型相同）时性能从 85.8% 降至 83.2%

#### 模块 3: 多层特征融合（Multi-Layer Feature Fusion）

**设计动机**: 利用 [[Transformer]] 不同层捕捉不同抽象层次的特征，提供更丰富的时序监督信号

**具体实现**:
- 从主干第 {4, 12, 20, 30} 层收集隐状态 $\mathbf{h}_4, \mathbf{h}_{12}, \mathbf{h}_{20}, \mathbf{h}_{30}$
- 拼接后经两层 MLP 压缩：$\mathbf{h}_{\text{fuse}} \in \mathbb{R}^{B \times N \times d}$
- 梯度可以通过融合反传到骨干所有层，实现"深度时序监督穿透"

#### 模块 4: 因果链（Causal MCP Chaining）

**设计动机**: 利用近未来预测辅助更远未来预测，形成预测深度间的因果链

**具体实现**:
- Depth-1 模块输入：$\mathbf{h}_{\text{fuse}}$ + 带噪未来块 embed
- Depth-k 模块输入：Depth-(k-1) 的输出 $\mathbf{h}_{\text{prev}}^{[k-1]}$ + 对应带噪块 embed
- 通过可训练线性层 $W_k \in \mathbb{R}^{d \times 2d}$ 融合

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配前向插值]]

$$
\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol{\varepsilon}
$$

**含义**: 在干净样本 $\mathbf{x}_0$ 和标准高斯噪声 $\boldsymbol{\varepsilon}$ 之间线性插值，构造时间步 $t$ 处的带噪样本

**符号说明**:
- $\mathbf{x}_0$: 干净视频潜变量
- $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $t \in [0, 1]$: 扩散时间步

### 公式 2: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\varepsilon}}\left[\|v_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\boldsymbol{\varepsilon} - \mathbf{x}_0)\|^2\right]
$$

**含义**: 训练速度预测网络 $v_\theta$ 拟合从 $\mathbf{x}_0$ 到 $\boldsymbol{\varepsilon}$ 的流场方向

**符号说明**:
- $v_\theta$: 速度预测网络（主模型）
- $\mathbf{c}$: 条件信息（语言指令 + 历史帧）
- $\boldsymbol{\varepsilon} - \mathbf{x}_0$: 目标流场方向

### 公式 3: [[Teacher Forcing|自回归视频 Teacher Forcing]]

$$
v_\theta\!\left(\mathbf{x}_t^{(i)}, t, \left[\mathbf{x}_0^{(1:i-1)}, \ell\right]\right) \approx \boldsymbol{\varepsilon}^{(i)} - \mathbf{x}_0^{(i)}
$$

**含义**: 标准 Teacher Forcing 以过去所有干净帧块为条件，训练模型去噪第 i 个块

**符号说明**:
- $\mathbf{x}_0^{(1:i-1)}$: 历史干净帧块（teacher-forced 上下文）
- $\ell$: 语言条件

### 公式 4: [[Multi-Chunk Prediction|时序块偏移]]

$$
\mathbf{x}_0^{[k][i]} = \mathbf{x}_0[\min(i+k, F)]
$$

**含义**: 对深度 k 的 MCP 模块，将训练目标向未来偏移 k 个块，边界处复制最后一帧

**符号说明**:
- $k$: MCP 预测深度（k=1,2,3）
- $F$: 视频总块数
- $i$: 当前块索引

### 公式 5: [[Multi-Chunk Prediction|MCP 独立噪声注入]]

$$
\mathbf{x}_{t_k}^{[k]} = (1 - t_k)\mathbf{x}_0^{[k]} + t_k \boldsymbol{\varepsilon}_k
$$

**含义**: 对偏移后的目标注入独立噪声，$t_k$ 由偏移时间步调度给出，$\boldsymbol{\varepsilon}_k$ 与主模型噪声独立

**符号说明**:
- $\boldsymbol{\varepsilon}_k$: 各深度独立采样的高斯噪声
- $t_k$: 对应 MCP 深度的噪声时间步（由 $s_{\text{mcp}}=10$ 偏移调度决定）

### 公式 6: [[Rotary Position Embedding|MCP 位置编码调整]]

$$
\text{RoPE}\!\left(\mathbf{x}_0^{[k][i]}\right) = \text{RoPE}(i + k)
$$

**含义**: MCP 目标的 RoPE 编码使用未来位置 $i+k$，保持时序位置一致性

### 公式 7: [[Multi-Chunk Prediction|多层特征融合]]

$$
\mathbf{h}_{\text{fuse}} = \text{MLP}\!\left([\mathbf{h}_4;\, \mathbf{h}_{12};\, \mathbf{h}_{20};\, \mathbf{h}_{30}]\right) \in \mathbb{R}^{B \times N \times d}
$$

**含义**: 将骨干 {4, 12, 20, 30} 层的中间隐状态拼接后，经两层 MLP 压缩到模型维度 d

**符号说明**:
- $\mathbf{h}_l$: 第 $l$ 层 Transformer 的隐状态
- $B, N, d$: Batch size、序列长度、模型维度

### 公式 8: [[Multi-Chunk Prediction|因果链跨深度融合]]

$$
\mathbf{z}^{[k]} = W_k\!\left[\mathbf{h}_{\text{prev}}^{[k-1]};\, \text{Embed}\!\left(\mathbf{x}_{t_k}^{[k]}\right)\right], \quad W_k \in \mathbb{R}^{d \times 2d}
$$

**含义**: 深度 k 的 MCP 模块将上一深度的输出与当前带噪块的嵌入拼接并线性变换，实现因果链

**符号说明**:
- $\mathbf{h}_{\text{prev}}^{[k-1]}$: 上一预测深度（k-1）的输出特征
- $\text{Embed}(\cdot)$: 视频潜变量嵌入层
- $\mathbf{h}_{\text{prev}}^{[0]} \triangleq \mathbf{h}_{\text{fuse}}$: 深度 1 以融合特征作为起点

### 公式 9: 联合视频-动作分解

$$
\mathbf{x}_{i+1} \sim p_\theta(\cdot \mid \mathbf{x}_{\leq i}, \mathbf{a}_{<i}, \ell);\quad \mathbf{a}_i \sim g_\psi(\cdot \mid \mathbf{x}_{\leq i+1}, \mathbf{a}_{<i}, \ell)
$$

**含义**: 主模型 $p_\theta$ 生成下一视频块，逆动力学模型 $g_\psi$ 从视频块中恢复动作

### 公式 10: 视频动力学损失

$$
\mathcal{L}_{\text{video}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\varepsilon}}\!\left[\|v_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\boldsymbol{\varepsilon} - \mathbf{x}_0)\|^2\right]
$$

**含义**: 主模型的自回归视频去噪损失

### 公式 11: 动作逆动力学损失

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\varepsilon}}\!\left[\|v_\psi(\mathbf{a}_t, t, \mathbf{c}_a) - (\boldsymbol{\varepsilon} - \mathbf{a}_0)\|^2\right]
$$

**含义**: 逆动力学模型 $v_\psi$ 的动作预测损失，$\mathbf{c}_a$ 为视频块序列作为条件

### 公式 12: 各 MCP 深度损失

$$
\mathcal{L}_k^{\text{MCP}} = \mathbb{E}_{t_k, \mathbf{x}_0^{[k]}, \boldsymbol{\varepsilon}_k}\!\left[\|v_\theta^{[k]}\!\left(\mathbf{x}_{t_k}^{[k]}, t_k, \mathbf{c}\right) - \left(\boldsymbol{\varepsilon}_k - \mathbf{x}_0^{[k]}\right)\|^2\right]
$$

**含义**: 第 k 个 MCP 辅助模块的流匹配损失，与主损失结构相同但作用于偏移目标

### 公式 13: [[Multi-Chunk Prediction|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}} + \sum_{k=1}^{3} w_k \cdot \mathcal{L}_k^{\text{MCP}}
$$

**含义**: 总损失由主视频损失、动作损失和三个 MCP 辅助损失加权求和组成

**符号说明**:
- $w_1 = 0.5,\; w_2 = 0.2,\; w_3 = 0.1$: MCP 各深度损失权重（随深度递减）

### 公式 14: [[SNR Shift|时间步偏移调度]]

$$
\tilde{\sigma}_i = \frac{s \cdot \sigma_i}{1 + (s-1) \cdot \sigma_i}
$$

**含义**: 将原始时间步 $\sigma_i$ 通过偏移系数 $s$ 变换，使采样集中在更高噪声区间

**符号说明**:
- $s$: 偏移系数（主模型 $s=5$，MCP 模块 $s=10$）
- $\sigma_i$: 均匀采样的原始时间步

### 公式 15: 偏移带噪样本

$$
\mathbf{x}_{\tilde{\sigma}_i} = (1 - \tilde{\sigma}_i)\mathbf{x}_0 + \tilde{\sigma}_i \cdot \boldsymbol{\varepsilon}
$$

**含义**: 使用变换后的时间步 $\tilde{\sigma}_i$ 构造带噪样本，对应更高噪声水平

---

## 关键图表

### Figure 1: 训练收敛曲线

![[NextForcing_fig1.png]]

**说明**: RoboTwin 上不同训练步数的任务成功率（%）。Next Forcing 在 12 fps 和 50 fps 两种设置下都收敛更快、最终精度更高。在 50 fps 高帧率下，5k 步时 Next Forcing（70.2/61.6%）vs 基线（45.5/31.9%），体现了近视监督问题在高帧率下的严重性。

### Figure 2: 整体架构图

![[NextForcing_fig2.png]]

**说明**: Next Forcing 整体框架。主模型去噪当前块，同时链式 [[Multi-Chunk Prediction|MCP 模块]] 利用主模型的多层特征预测未来块（next¹, next², …）。独立噪声注入、时序块偏移和因果链设计确保 MCP 模块有效学习长程时序依赖。

### Figure 3: PhyWorld 定性对比

![[NextForcing_fig3.png]]

**说明**: PhyWorld 物理一致性的定性对比，展示 5 帧（起始、3 个中间帧、结束）。上：Ground Truth，中：Next Forcing，下：基线。Next Forcing 生成的视频更符合物理规律，基线常出现物理异常。

### Figure 4: 通用视频预训练 FVD 曲线

![[NextForcing_fig4.png]]

**说明**: 在 3.5M 通用视频数据上预训练时，Next Forcing 与基线的 FVD 随训练步数变化。Test Set 1（人体活动视频）：Next Forcing 在 10k 步超越基线 50k 步；58% FVD 降低（94 vs 225）。

### Figure 5: 注意力掩码设计

![[NextForcing_fig5.png]]

**说明**: 主模型与 MCP 模块的注意力掩码（仅展示视频 token，省略动作 token）。主模型使用因果掩码，MCP 模块各深度的注意力范围与因果链结构对应。

### Table 1: RoboTwin Benchmark 结果

| 方法 | Clean (%) | Random (%) |
|------|-----------|------------|
| X-VLA | 72.9 | 72.8 |
| π₀ | 65.9 | 58.4 |
| π₀.₅ | 82.7 | 76.8 |
| Motus | 88.7 | 87.0 |
| Being-H0.7 | 90.2 | 89.6 |
| Fast-WAM | 91.9 | 91.8 |
| LingBot-VA | 92.9 | 91.5 |
| **Next Forcing** | **94.1** | **93.5** |

**关键发现**: Next Forcing 在 50 个双臂操作任务上全面超越所有 baseline，相比前 SOTA LingBot-VA 提升 1.2/2.0%。

### Table 2: 消融实验

| 配置 | 成功率 (%) | 说明 |
|------|------------|------|
| **Baseline 消融** | | |
| Baseline（默认 s_main=5） | 75.6 | 基础配置 |
| s_main = 1 | 65.3 | 时间步偏移过小，监督信号弱 |
| s_main = 10 | 78.4 | 适度增大偏移有帮助 |
| s_main = 20 | 77.6 | 偏移过大收益递减 |
| s_main = 25 | 77.2 | 同上 |
| 无带噪历史增强 | 69.8 | 历史增强贡献 +5.8% |
| **MCP 模块消融** | | |
| Baseline + MCP（默认） | 85.8 | MCP 带来 +10.2% |
| s_mcp = 5（与主模型相同） | 83.2 | 松耦合降低效果 |
| 无多层融合 | 83.6 | 多层融合贡献 +2.2% |
| 无权重初始化 | 83.8 | 初始化策略有效 |
| Transformer blocks = 1 | 86.5 | 1 块略好，计算更少 |
| Transformer blocks = 5 | 85.0 | 过多块略有下降 |

**关键发现**: MCP 模块带来最大增益（+10.2%），独立噪声耦合（$s_{\text{mcp}} > s_{\text{main}}$）和多层融合各贡献约 2-2.6%。

### Table 3: PhyWorld 物理理解 Benchmark

| 方法 | FVD (Out-of-Template) ↓ | FVD (In-Template) ↓ | 异常率 OOT ↓ | 异常率 IT ↓ |
|------|------------------------|---------------------|-------------|------------|
| LingBot-VA | 5.3 | 3.5 | 12% | 3% |
| **Next Forcing** | **4.7** | **3.2** | **8%** | **2%** |

**关键发现**: Next Forcing 在物理规律遵守方面更优，OOT 异常率降低 33%（12% → 8%）。

### Table 4: 推理加速结果

| 推理模式 | 12fps Clean | 12fps Random | 25fps Clean | 25fps Random | 50fps Clean | 50fps Random |
|---------|------------|-------------|------------|-------------|------------|-------------|
| 标准推理 | 94.1 | 93.5 | 92.6 | 91.4 | 91.8 | 90.5 |
| MCP 加速（2×） | 93.5 | 90.6 | 91.0 | 89.8 | 92.2 | 91.3 |

**关键发现**: 2× 推理加速下，精度损失极小（50fps 下甚至略有提升），验证了 MCP 模块并行预测的可行性。

### Table 5: 训练收敛完整数据（50 fps）

| 设置 | 方法 | 5k | 10k | 20k | 30k | 40k | 50k |
|------|------|-----|-----|-----|-----|-----|-----|
| 50fps Clean | LingBot-VA | 45.5 | 64.8 | 78.5 | 81.2 | 83.8 | 88.6 |
| 50fps Clean | **Next Forcing** | **70.2** | **80.5** | **87.4** | **90.0** | **91.5** | **91.8** |
| 50fps Random | LingBot-VA | 31.9 | 54.7 | 69.4 | 75.6 | 80.4 | 85.2 |
| 50fps Random | **Next Forcing** | **61.6** | **77.6** | **85.0** | **86.8** | **88.4** | **90.5** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin | 2,500 Clean + 25,000 Random 演示 | 50 个双臂操作任务，Clean/Random 扰动 | 主要机器人操作评测 |
| PhyWorld | — | 物理规律遵守测试，含 OOT（Out-of-Template）和 IT 两个集合 | 物理理解评测 |
| 通用视频预训练 | ~3.5M 视频片段（5-10 秒） | 人体活动 + 相机驱动场景动态 | 通用视频生成评测 |

### 实现细节

- **Backbone**: Wan2.2 Transformer，30 层
- **MCP 模块**: 3 个深度，各 3 个 Transformer blocks，轻量设计
- **特征融合层**: 两层 MLP，从 {4, 12, 20, 30} 层收集特征
- **主模型时间步偏移**: $s_{\text{main}} = 5$，带噪历史增强概率 0.5
- **MCP 时间步偏移**: $s_{\text{mcp}} = 10$
- **MCP 损失权重**: $w_1=0.5,\, w_2=0.2,\, w_3=0.1$
- **Chunk 大小**: $M$ 从 $\{1, \ldots, M_{\max}\}$ 随机采样，$M_{\max}=4$
- **训练硬件（主实验）**: 64 GPUs；**消融实验**: 16 GPUs
- **训练步数**: RoboTwin 最多 50k 步；通用视频预训练 50k 步

### 可视化结果

PhyWorld 定性结果（Figure 3）显示：Next Forcing 生成的视频物理约束更强，基线在 Out-of-Template 场景频繁出现物理异常（如物体穿透、违反重力）；Next Forcing 则与 Ground Truth 更接近。

---

## 批判性思考

### 优点

1. **问题定义清晰**: "近视监督"是真实存在的 failure mode，在高帧率下尤为显著，论文用收敛曲线直接论证
2. **工程设计细致**: 独立噪声、时间步偏移解耦、多层融合三个关键设计各有其消融验证，非拍脑袋
3. **推理加速是免费午餐**: zero-overhead 模式不增加推理成本，MCP 加速模式 2× 加速且几乎无精度损失，实用性强
4. **通用性验证充分**: 不仅限于机器人操作，通用视频预训练也显著提升，说明是根本性改进

### 局限性

1. **训练开销增加**: MCP 模块引入额外计算，训练成本高于基线（文中未明确量化开销比）
2. **依赖特定架构**: 多层特征融合强依赖 Transformer 的中间层隐状态，迁移到 CNN 或 SSM 架构需要重新设计
3. **边界处理简单**: $\min(i+k, F)$ 在视频末尾复制最后一帧，可能引入人工偏置

### 潜在改进方向

1. **自适应 MCP 深度**: 动态调整预测深度 k 而非固定为 3，根据任务难度自动决定
2. **与 Diffusion Forcing / Self Forcing 结合**: 文中提到是正交方法，可探索联合训练
3. **MCP 权重调度**: $w_k$ 固定为 {0.5, 0.2, 0.1}，自适应调度可能进一步提升

### 可复现性评估

- [ ] 代码开源（项目主页存在但尚无代码链接）
- [ ] 预训练模型
- [x] 训练细节完整（详细超参数均在 paper 中）
- [ ] 数据集可获取（RoboTwin 公开，通用视频集为内部数据）

---

## 关联笔记

### 基于

- [[LingBot-VA]]: 直接基线方法，Next Forcing 建立在其 WAM 框架之上
- [[Flash-WAM]]: 近期高速 WAM 方法，Table 1 基线之一
- [[Flow Matching]]: 核心生成框架，Next Forcing 在其上设计 MCP 损失

### 对比

- [[Diffusion Forcing]]: 修改噪声调度的"forcing"方法，Next Forcing 修改预测目标，两者正交
- [[Self Forcing]]: 修改上下文策略，与 Next Forcing 也正交可组合

### 方法相关

- [[Multi-Chunk Prediction]]: 核心方法，从 LLM 的 Multi-Token Prediction 迁移到连续视频
- [[Teacher Forcing]]: Next Forcing 要改进的对象
- [[Flow Matching]]: 底层生成框架
- [[SNR Shift|Timestep Shift]]: 关键训练技巧，控制噪声水平
- [[Rotary Position Embedding]]: MCP 位置编码调整所用技术
- [[Transformer]]: 骨干架构（Wan2.2）

### 硬件/数据相关

- [[RoboTwin]]: 主要评测 benchmark，50 个双臂操作任务
- [[PhyWorld]]: 物理理解评测 benchmark

---

## 速查卡片

> [!summary] Next Forcing: Causal World Modeling with Multi-Chunk Prediction
> - **核心**: 识别并解决自回归视频世界模型的"近视监督"问题
> - **方法**: Multi-Chunk Prediction — 三深度辅助模块同时预测 next¹/²/³ 未来块，配合独立噪声注入和多层特征融合
> - **结果**: RoboTwin SOTA 94.1%，2.3× 训练收敛加速（50fps），2× 推理加速
> - **代码**: [项目主页](https://gangweix.github.io/next-forcing/)

---

*笔记创建时间: 2026-06-11*

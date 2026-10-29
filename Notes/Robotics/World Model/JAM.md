---
title: "Vision-Language-Action or World-Action? Just Action Models"
method_name: "JAM"
authors: [Yonghao Yu, Lang Huang, Runyi Li, Toshihiko Yamasaki]
year: 2026
venue: arXiv
tags: [world-action-model, vla, flow-matching, foresight-learning, robot-manipulation, imitation-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.37219
created: 2026-10-29
---

# 论文笔记：Vision-Language-Action or World-Action? Just Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Tokyo |
| 日期 | October 2026 |
| 项目主页 | [y0uroy.github.io/JAM](https://y0uroy.github.io/JAM) |
| 对比基线 | [[FastWAM]] / [[ImageWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.37219) |

---

## 一句话总结

> JAM 通过在训练时引入"先见损失"（预测未来帧编码的辅助头，推理时丢弃），无需视觉生成即可媲美或超越 WAM，同时降低 73% 训练 FLOPs。

---

## 核心贡献

1. **设计空间拆分**: 将 VLA vs. WAM 的争论分解为三个独立选择：骨干初始化方式、对未来观测施加的目标、推理时保留的模块；揭示这三者可以独立变化。
2. **Just Action Model (JAM)**: 在训练时用[[Foresight Loss|先见损失]]（cosine 距离预测冻结的未来帧编码）代替视觉生成目标；训练结束后丢弃预测头与未来锚，推理时与普通动作策略完全相同。
3. **跨骨干泛化**: JAM 在生成式骨干（Wan2.2-5B、Ovis-U1、FLUX.2）、自监督骨干（DINOv2-B/L）和视觉语言骨干（Qwen3.5-2B）上均有效，且 Qwen3.5-2B 在 LIBERO-Plus 上达到 81.1% 的最高成绩。

---

## 问题背景

### 要解决的问题

近年来出现两类机器人策略流派：[[Vision-Language-Action|VLA]]（用语言-视觉大模型初始化）和 [[World Action Model|WAM]]（用视频/图像生成模型初始化，并在训练时加入视觉生成目标）。两类方法争论骨干应当来自"语言世界"还是"视觉生成世界"，但实际上将骨干选择、训练目标和推理模块三个问题混在了一起讨论。

### 现有方法的局限

- **FastWAM / ImageWAM** 等 WAM 方法在训练时需要生成未来帧（视频或图像），计算开销大（15.52 TFLOPs/step for FastWAM）。
- VLA 方法（π0、GR00T N1）依赖大规模网络预训练，在泛化性（LIBERO-Plus 摄像头扰动）上仍存在明显短板。
- 现有方法没有系统区分"生成能力"和"未来表示学习"的独立贡献。

### 本文的动机

如果未来监督信号（表示未来帧的特征）可以通过一个训练后丢弃的轻量预测头学到，则无需在训练或推理时生成完整未来帧。这样既保留了"预测未来有助于控制"的归纳偏置，又大幅降低计算成本。

---

## 方法详解

### 模型架构

JAM 采用 **双路径（训练）→ 单路径（推理）** 架构：

- **输入**: 语言指令 $c$ + 当前观测帧 $f_0$ + 机器人状态（可选）
- **Backbone**: 预训练视觉/VL 骨干（Wan2.2-5B、Yak18/Ovis-U1、FLUX.2 Klein 4B、Qwen3.5-2B、DINOv2-B/L）
- **Action Expert**: 与骨干深度匹配的 [[Flow Matching|流匹配]]动作专家（[[Action Chunking|动作块]]预测）
- **Foresight Head（仅训练）**: 可学习的投影 + 2 层 Transformer + readout 线性层
- **Future Anchor（仅训练）**: 冻结的骨干副本（或冻结 VAE），编码记录的未来帧 $f_k$ 以生成目标 $\mathbf{g}_{k,n}$
- **输出**: 动作块 $a_{t:t+32}$（Wan）或 $a_{t:t+16}$（其他）

**训练时**：骨干+动作专家+预测头联合优化；未来锚参数完全冻结。
**推理时**：预测头、未来锚、未来帧全部丢弃，与普通 VLA 推理完全相同。

### 核心模块

#### 模块1: Foresight Head（先见预测头）

**设计动机**: 用[[Cosine Similarity|余弦相似度]]目标迫使骨干在当前帧特征中编码未来观测的语义信息。

**具体实现**:
- 输入投影：$D \to d$（$d = D/2$）
- 两层共享 Pre-LayerNorm Transformer 块
- 每个目标偏移 $k$ 对应一个独立的线性 readout $d \to D_g$
- 从骨干约 60% 深度处（Wan: 第 18/30 块）的 token 特征读取输入

**丢弃时机**: 最终 checkpoint 保存前移除，不进入推理图。

#### 模块2: Future Anchor（未来锚）

**设计动机**: 提供高质量、稳定的未来帧表示目标，避免在线生成的成本。

**具体实现**:
- 方案 A（Wan）：冻结的 Wan VAE 编码器，直接提取潜变量特征
- 方案 B（Ovis-U1）：冻结的 Yak18 编码器（与骨干同族）
- 方案 C（Qwen3.5-2B）：冻结的 Qwen3.5-2B 编码器（与骨干同族）
- 实验发现：随机初始化锚仍有效（Table 2），骨干预训练比锚预训练更重要（Table 17）

**关键发现**: 未来帧偏移量的选择对性能有显著影响（Wan 用 $k \in \{1,2\}$ 步偏移，Ovis-U1 用 $k=1$）；当前帧偏移 $k=0$ 对 Wan 几乎无益，但对 Ovis-U1 有害（Table 1）。

---

## 关键公式

### 公式1: [[VLA]] 动作目标（基线）

$$
\mathcal{L}_{\mathrm{VLA}} = \mathcal{L}_{\mathrm{act}}
$$

**含义**: 纯动作损失，不对未来观测施加任何目标。

**符号说明**:
- $\mathcal{L}_{\mathrm{act}}$: [[Flow Matching|流匹配]]动作损失

### 公式2: [[World Action Model|WAM]] 总目标

$$
\mathcal{L}_{\mathrm{WAM}} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{G}}\,\mathcal{L}_{\mathrm{G}}
$$

**含义**: 在动作损失基础上加入视觉生成损失 $\mathcal{L}_{\mathrm{G}}$，训练和推理时保留生成头。

**符号说明**:
- $\lambda_{\mathrm{G}}$: 生成损失权重
- $\mathcal{L}_{\mathrm{G}}$: 视觉生成损失（扩散/流匹配于未来帧）

### 公式3: [[Foresight Loss|JAM 总目标]]

$$
\mathcal{L}_{\mathrm{JAM}} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{F}}\,\mathcal{L}_{\mathrm{F}}
$$

**含义**: 用先见损失 $\mathcal{L}_{\mathrm{F}}$ 替代视觉生成损失；先见头仅训练时存在，推理时丢弃。

**符号说明**:
- $\lambda_{\mathrm{F}}$: 先见损失权重（Wan: 0.6，其他: 0.3）
- $\mathcal{L}_{\mathrm{F}}$: 先见损失（见公式4）

### 公式4: [[Foresight Loss|先见损失]]

$$
\mathcal{L}_{\mathrm{F}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \mathbb{E}_{n} \Big[ 1 - \cos\big(\hat{\mathbf{g}}_{k,n},\ \operatorname{sg}[\mathbf{g}_{k,n}]\big) \Big]
$$

**含义**: 预测头输出 $\hat{\mathbf{g}}_{k,n}$ 与冻结未来锚特征 $\mathbf{g}_{k,n}$ 之间的余弦距离，对所有目标偏移 $k$ 和 token 位置 $n$ 取平均。

**符号说明**:
- $\mathcal{K}$: 目标偏移集合（Wan: $\{1,2\}$ 环境步）
- $k$: 具体偏移量（相对当前帧）
- $n$: token 位置索引
- $\hat{\mathbf{g}}_{k,n}$: 预测头对偏移 $k$ 的输出特征
- $\mathbf{g}_{k,n}$: 未来锚编码的第 $k$ 步未来帧特征（冻结目标）
- $\operatorname{sg}[\cdot]$: [[Stop-Gradient|stop-gradient]] 算子（阻止梯度流向未来锚）
- $\cos(\cdot, \cdot)$: [[Cosine Similarity|余弦相似度]]

### 公式5: FLOP 估算

$$
\text{FLOPs} \approx 2NT
$$

**含义**: 用于训练计算量对比分析，$N$ 为活跃参数量，$T$ 为 token 数。

**符号说明**:
- $N$: 模型活跃参数数量
- $T$: 处理的 token 总数

---

## 关键图表

### Figure 1: Design Space / 设计空间

![Figure 1](https://arxiv.org/html/2610.37219v1/fig_designspace_ppt.png)

**说明**: 三个独立选择维度将 VLA、WAM 和 JAM 区分开来。(a) VLA：仅用动作损失；(b) WAM：加入视觉生成损失，部分方法将生成的未来帧反馈给动作专家；(c) JAM：用先见损失代替生成损失，推理时丢弃监督分支，动作分支与 VLA 完全相同。

### Figure 2: JAM 训练与推理

![Figure 2](https://arxiv.org/html/2610.37219v1/fig_method_ppt.png)

**说明**: 左图为训练时：记录的未来帧仅进入未来锚分支，产生冻结目标；预测头从骨干中间层特征预测这些目标。右图为推理时：预测头与未来锚被丢弃，仅保留骨干+动作专家，无需生成任何未来帧。

### Figure 3: LIBERO-Long Matched Rollouts / 对比推理轨迹

**说明**: 在两个 LIBERO-Long 任务（将碗放入抽屉并关闭；将两个摩卡壶放上炉子）上，ImageWAM（Ovis-U1）与 JAM（Ovis-U1）的对比帧序列。ImageWAM 失败而 JAM 成功。

### Figure 4: Decoded Future Change / 未来变化解码对比

![Figure 4a - current](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_current.png)
![Figure 4b - future](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_future.png)
![Figure 4c - true change](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_true_change_signed.png)
![Figure 4d - ImageWAM](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_native_signed.png)
![Figure 4e - JAM offset 0](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_jam_off0_signed.png)
![Figure 4f - JAM offset 1](https://arxiv.org/html/2610.37219v1/Figures/readout_diff/agent_jam_signed.png)

**说明**: 用冻结 MLP 探针从骨干特征重建当前帧和未来帧，差图与记录的真实变化对比。JAM（偏移=1）的重建差图与真实变化最接近，说明先见损失让骨干有效编码了未来变化信息，与 ImageWAM 相当甚至更好。

### Figure 5: Action-to-Visual Attention / 动作-视觉注意力

**说明**: 动作 token 在单帧 LIBERO-Long 观测（视野图和腕部图）上的注意力分布。(a) ImageWAM 与 JAM（同骨干 Ovis-U1）对比，JAM 注意力更集中于任务相关区域；(b) DINOv2-L 和 Qwen3.5-2B 的 JAM 变体，条形图展示注意力在文本/图像/动作 key 上的分配。

### Figure 6: LIBERO-Plus Paired Outcomes / 配对结果分析

**说明**: 固定 checkpoint 后，LIBERO-Plus（1,506 个共享任务）的配对比较。左图：ImageWAM（Ovis-U1）vs JAM，JAM 在摄像头扰动类别上多解 71 个仅 JAM 成功的任务（vs 9 个仅 baseline 成功）。右图：FastWAM（Wan）vs JAM，机器人初始状态扰动类别 45 vs 10；摄像头扰动类别 31 vs 28（接近）。

### Figure 7: RoboTwin Matched Rollouts / 机器人实验对比轨迹

**说明**: 四组对比：(a) LIBERO-Plus，Ovis-U1，"打开炉子并放置摩卡壶"；(b) LIBERO-Plus，Ovis-U1，"打开柜子中间抽屉"；(c) RoboTwin 2.0，Wan，"将罐子放入篮子"；(d) RoboTwin 2.0，Wan，"堆叠红绿蓝积木"。显示 JAM 在 baseline 失败的情况下成功完成任务。

### Figure 8: Action-to-Visual Attention (Agent View) / 动作注意力热力图

**说明**: 热力图行：LIBERO-Goal、LIBERO-Long、LIBERO-Plus（摄像头/布局扰动）。列：原图、ImageWAM（Ovis-U1）、JAM（Ovis-U1）、差图、JAM（DINOv2-L）、JAM（Qwen3.5-2B）。红色表示 JAM 注意力更高，蓝色表示 ImageWAM 更高。JAM 注意力更集中于物体和目标区域。

### Figure 9: Mean Future-Frame PSNR / 未来帧 PSNR

**说明**: 冻结 MLP 探针重建未来帧的 PSNR（dB）。Current 帧：16.13 dB（基准）；ImageWAM：19.43 dB；JAM（偏移=0）：19.29 dB；JAM（偏移=1）：19.57 dB。JAM 在不生成未来帧的情况下，骨干内编码的未来信息量与 ImageWAM 相当。

---

### Table 1: 生成 vs 先见（固定策略架构）

| Policy | $\mathcal{L}_G$ | $\mathcal{L}_F$ | Future Anchor | Readout Offsets | LIBERO | LIBERO-Plus |
|---|:---:|:---:|---|---|---|---|
| FastWAM | ✗ | ✗ | – | – | 95.25 | 45.55 |
| FastWAM | ✓ | ✗ | – | – | 97.00 | 55.31 |
| JAM (Wan) | ✗ | ✓ | Pretrained Wan | [0] | 95.80 | 44.62 |
| JAM (Wan) | ✗ | ✓ | Pretrained Wan | [1,2] | 97.45 | 55.58 |
| JAM (Wan) | ✗ | ✓ | Pretrained Wan | [0,1,2] | 97.40 | 54.18 |
| JAM (Wan) | ✗ | ✓ | Random Wan | [1,2] | 97.65 | 58.63 |
| **JAM (Wan)** | ✗ | ✓ | **Pretrained VAE** | **–** | **98.15** | **64.54** |
| ImageWAM (Ovis-U1) | ✗ | ✗ | – | – | 90.20 | 61.35 |
| ImageWAM (Ovis-U1) | ✓ | ✗ | – | – | 92.90 | 65.27 |
| JAM (Ovis-U1) | ✗ | ✓ | Pretrained Yak | [0] | 94.55 | 69.65 |
| **JAM (Ovis-U1)** | ✗ | ✓ | **Pretrained Yak** | **[1]** | **96.75** | **79.95** |
| JAM (Ovis-U1) | ✗ | ✓ | Pretrained Yak | [0,1] | 95.00 | 72.58 |
| JAM (Ovis-U1) | ✗ | ✓ | Random Yak | [1] | 95.55 | 72.51 |
| JAM (Ovis-U1) | ✗ | ✓ | Pretrained VAE | – | 95.75 | 76.43 |

**关键发现**: 先见损失（无生成）可完全匹配甚至超越 WAM（有生成）。最佳锚的选择与骨干有关：Wan 用 VAE 锚最佳，Ovis-U1 用 Yak 锚最佳。未来偏移量选择对 LIBERO-Plus 影响显著（偏移=0 对 Ovis 有害）。

### Table 2: 骨干与锚的初始化消融（JAM-Wan，LIBERO）

| Backbone | Anchor | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|
| Pretrained | Pretrained | 98.2 | 100.0 | 97.6 | 94.0 | 97.45 |
| Pretrained | Random | 98.0 | 99.8 | 97.0 | 95.8 | 97.65 |
| Random | Random | 85.6 | 98.0 | 88.2 | 65.8 | 84.40 |
| Random | Pretrained | 76.0 | 95.0 | 70.4 | 43.2 | 71.15 |

**关键发现**: 骨干预训练至关重要；锚的预训练影响很小（随机锚甚至略好）；锚预训练无法弥补骨干随机初始化的不足。

### Table 3: JAM 跨预训练骨干（LIBERO & LIBERO-Plus）

| Policy Backbone | Size | LIBERO | LIBERO-Plus |
|---|---|---|---|
| ImageWAM (Ovis-U1) | 1.13B | 92.90 | 65.27 |
| JAM (Ovis-U1) | 1.13B | 96.75 | 79.95 |
| JAM (DINOv2-B) | 86.6M | 95.50 | 68.19 |
| JAM (DINOv2-L) | 304M | 96.30 | 70.39 |
| **JAM (Qwen3.5-2B)** | **2.0B** | **96.55** | **81.14** |

**关键发现**: JAM 的先见损失在所有骨干家族（生成式、自监督、语言-视觉）上均有效。Qwen3.5-2B 在 LIBERO-Plus 上达到最高 81.1%。

### Table 4: RoboTwin 2.0 成功率（%，Clean + Randomized 均值）

| Method | P.T. | AVG |
|---|:---:|---|
| π0 | ✓ | 62.2 |
| π0.5 | ✓ | 79.8 |
| Motus | ✓ | 87.8 |
| ImageWAM (Ovis-U1)† | ✗ | 89.0 |
| JAM (Qwen3.5-2B)† | ✗ | 89.8 |
| JAM (Ovis-U1)† | ✗ | 89.9 |
| FastWAM† | ✗ | 91.2 |
| JAM (Wan)† | ✗ | 91.4 |
| LingBot-VA | ✓ | 92.2 |
| ImageWAM (FLUX.2)† | ✗ | 92.9 |
| **JAM (FLUX.2)†** | **✗** | **93.1** |

### Table 5: LIBERO 四套件成功率（%）

| Method | P.T. | Spatial | Object | Goal | Long | AVG |
|---|:---:|---|---|---|---|---|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| GR00T N1 | ✓ | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| JAM (Qwen3.5-2B)† | ✗ | 97.8 | 99.6 | 97.2 | 91.6 | 96.6 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| FastWAM† | ✗ | 97.4 | 99.6 | 96.6 | 94.4 | 97.0 |
| **JAM (Wan)†** | **✗** | **98.8** | **100.0** | **98.0** | **95.8** | **98.2** |
| ImageWAM (Ovis-U1)† | ✗ | 97.8 | 100.0 | 93.6 | 80.2 | 92.9 |
| JAM (Ovis-U1)† | ✗ | 98.6 | 99.6 | 98.4 | 90.4 | 96.8 |

### Table 6: LIBERO-Plus 各扰动类别成功率（%）

| Method | P.T. | Camera | Robot | Language | Light | Background | Noise | Layout | AVG |
|---|:---:|---|---|---|---|---|---|---|---|
| UniVLA | ✓ | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | ✓ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | ✗ | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| FastWAM† | ✗ | 24.2 | 50.3 | 78.7 | 87.6 | 49.5 | 39.9 | 67.3 | 55.3 |
| JAM (Wan)† | ✗ | 25.4 | 68.0 | 92.6 | 93.2 | 46.2 | 52.7 | 81.4 | 64.5 |
| ImageWAM (Ovis-U1)† | ✗ | 57.9 | 54.8 | 67.6 | 78.9 | 66.7 | 68.6 | 65.0 | 65.3 |
| JAM (Ovis-U1)† | ✗ | 83.8 | 49.2 | 84.8 | 95.7 | 80.1 | 91.9 | 72.3 | 79.9 |
| **JAM (Qwen3.5-2B)†** | **✗** | **80.8** | **61.4** | **71.7** | **93.8** | **81.7** | **94.2** | **84.5** | **81.1** |

**关键发现**: JAM（Qwen3.5-2B）在几乎所有扰动类别上均显著领先，尤其是背景（81.7%）、噪声（94.2%）、布局（84.5%），展现出更强的鲁棒性。

### Table 7: 各骨干的训练目标与动作上下文

| Backbone | 帧偏移（env steps） | 监督层（/总层） | Readout | 动作上下文 |
|---|---|---|---|---|
| Wan | 0, 4, …, 32 | 18/30 | [1,2] | Wan + UMT5（VAE 目标） |
| Yak18 (Ovis-U1) | 0, 16 | 11/18 | [1] | Yak18 + Ovis |
| FLUX.2 | 0, 16 | 15/25 | – | FLUX.2 + Qwen3-4B |
| Qwen3.5-2B | 0, 16 | 13/24 | [1] | 因果图像-文本前缀 |
| DINOv2-B | 16 | 12/12 | [1] | DINOv2 + Qwen3-4B |

### Table 8: 预测头默认尺寸

| Policy | Input D | Target $D_g$ | Trunk $d$ | Layers | Heads | FFN | Readouts |
|---|---|---|---|---|---|---|---|
| JAM (Wan) | 3072 | 3072 | 1536 | 2 | 24 | 6144 | 2 |
| JAM (Ovis-U1) | 1536 | 1536 | 768 | 2 | 12 | 3072 | 1 |
| JAM (FLUX.2) | 3072 | 3072 | 1536 | 2 | 24 | 6144 | 1 |
| JAM (Qwen3.5-2B) | 2048 | 2048 | 1024 | 2 | 16 | 4096 | 1 |
| JAM (DINOv2-B) | 768 | 768 | 384 | 2 | 6 | 1536 | 1 |
| JAM (DINOv2-L) | 1024 | 1024 | 512 | 2 | 8 | 2048 | 1 |

### Table 9: 训练计算量对比

| Policy | Targets | Total TFLOPs | FLOP 降低 | s/step (GH200) | 时间降低 |
|---|---|---|---|---|---|
| FastWAM | – | 15.52 | – | 0.789 | – |
| JAM (Wan) | online | 9.80 | 36.9% | 0.532 | 32.6% |
| **JAM (Wan)** | **cached** | **4.19** | **73.0%** | **0.258** | **67.3%** |
| ImageWAM (Ovis-U1) | – | 5.81 | – | 0.473 | – |
| JAM (Ovis-U1) | online | 4.88 | 15.9% | 0.403 | 14.8% |
| **JAM (Ovis-U1)** | **cached** | **3.67** | **36.8%** | **0.294** | **37.8%** |

**关键发现**: 目标缓存（预先计算未来锚特征）后，JAM（Wan）训练 FLOPs 降低 73%，实测 GPU 时间降低 67.3%。

---

## 附录关键表格

### Table 10: 冻结 vs EMA 未来锚（Wan，LIBERO）

| Policy | Future Anchor | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|
| FastWAM | – | 97.4 | 99.6 | 96.6 | 94.4 | 97.00 |
| FastWAM action-only | – | 95.0 | 99.6 | 94.4 | 92.0 | 95.25 |
| JAM (Wan) | Pretrained Wan | 98.2 | 100.0 | 97.6 | 94.0 | 97.45 |
| JAM (Wan) | Online Wan EMA | 95.0 | 99.6 | 96.4 | 94.6 | 96.40 |

**关键发现**: EMA 锚（移动平均更新）不如完全冻结锚，原因可能是 EMA 目标随训练变化导致不稳定。

### Table 11: 骨干监督深度消融（Wan，LIBERO）

| JAM (Wan) | Depth | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|
| | 14 | 98.0 | 99.8 | 96.4 | 94.2 | 97.10 |
| | 16 | 99.2 | 99.8 | 96.2 | 94.0 | 97.30 |
| | **18** | **98.2** | **100.0** | **97.6** | **94.0** | **97.45** |
| | 20 | 96.8 | 99.2 | 95.4 | 90.4 | 95.45 |

**关键发现**: 最优监督深度为第 18 块（60% 深度）；过深（第 20 块）导致性能下降。

### Table 12: 先见损失权重 $\lambda_F$ 消融（Wan，LIBERO）

| $\lambda_F$ | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|
| 0.4 | 96.8 | 99.4 | 96.6 | 90.0 | 95.70 |
| 0.5 | 95.6 | 99.8 | 95.2 | 97.0 | 96.90 |
| **0.6** | **98.2** | **100.0** | **97.6** | **94.0** | **97.45** |
| 0.7 | 97.4 | 99.8 | 96.0 | 93.8 | 96.75 |

### Table 13: 预测头容量消融（Wan，LIBERO）

| Head Type | Width | FFN | Params (M) | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|---|---|
| Transformer | 3072 | 4096 | 145 | 97.4 | 100.0 | 96.2 | 91.6 | 96.30 |
| **Transformer** | **1536** | **6144** | **71** | **98.2** | **100.0** | **97.6** | **94.0** | **97.45** |
| Transformer | 768 | 6144 | 31 | 96.4 | 100.0 | 97.0 | 89.6 | 95.75 |
| MLP | 3072 | – | 38 | 95.8 | 100.0 | 94.2 | 91.4 | 95.35 |
| MLP | 1536 | – | 17 | 95.8 | 99.2 | 94.0 | 91.2 | 95.05 |

**关键发现**: 中等容量 Transformer 头（71M）优于更大（145M）或更小（31M）的头；MLP 头性能劣于 Transformer 头。

### Table 14: 纯动作 baseline（LIBERO-Plus，%）

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | AVG |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B action-only | 80.00 | 68.02 | 72.54 | 92.55 | 54.84 | 93.02 | 75.45 | 77.03 |
| **JAM (Qwen3.5-2B)** | **80.83** | **61.42** | **71.72** | **93.79** | **81.72** | **94.19** | **84.55** | **81.14** |
| DINOv2-L action-only | 37.92 | 32.99 | 86.48 | 86.96 | 85.48 | 78.29 | 72.73 | 68.26 |
| JAM (DINOv2-L) | 55.42 | – | – | – | – | – | – | 70.39 |

**关键发现**: JAM 相较于同骨干的纯动作 baseline 提升显著，背景鲁棒性（54.84→81.72）和布局鲁棒性（75.45→84.55）是最大收益点。

### Table 15: 独立动作专家 vs 共享骨干（Qwen3.5，LIBERO-Plus，%）

| Policy | Actions | Params | Camera | Robot | Language | Light | Background | Noise | Layout | AVG |
|---|---|---|---|---|---|---|---|---|---|---|
| JAM (Qwen3.5-2B) | **Expert** | 3.60B | 80.8 | 61.4 | 71.7 | 93.8 | 81.7 | 94.2 | 84.5 | **81.14** |
| JAM (Qwen3.5-2B) | Shared | 2.22B | 80.4 | 64.0 | 68.9 | 92.5 | 76.9 | 96.1 | 78.2 | 79.61 |
| JAM (Qwen3.5-4B) | Shared | 4.55B | 80.4 | 67.5 | 74.6 | 91.9 | 76.3 | 95.3 | 77.7 | 80.68 |

### Table 16: 训练预算敏感性

| Policy | Updates | LIBERO | LIBERO-Plus |
|---|---|---|---|
| FastWAM | 10k | 95.90 | 55.25 |
| FastWAM | 20k | 97.00 | 55.31 |
| JAM (Wan) | 10k | 97.45 | 48.47 |
| **JAM (Wan)** | **20k** | **97.45** | **55.58** |

**关键发现**: JAM（Wan）在 10k 步时 LIBERO 已与 20k 步的 FastWAM 相当，但 LIBERO-Plus 需要更多步数（预测头需要更长时间稳定）。

### Table 17: 骨干与目标初始化消融（LIBERO，%）

| Backbone | Anchor | Spatial | Object | Goal | Long | AVG |
|---|---|---|---|---|---|---|
| Pretrained | Pretrained | 98.2 | 100.0 | 97.6 | 94.0 | 97.45 |
| Pretrained | Random | 98.0 | 99.8 | 97.0 | 95.8 | 97.65 |
| Random | Random | 85.6 | 98.0 | 88.2 | 65.8 | 84.40 |
| Random | Pretrained | 76.0 | 95.0 | 70.4 | 43.2 | 71.15 |

### Table 18: LIBERO 完整成功率（%，含套件细节）

| Method | P.T. | Spatial | Object | Goal | Long | AVG |
|---|:---:|---|---|---|---|---|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| GR00T N1 | ✓ | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| JAM (Qwen3.5-2B)† | ✗ | 97.8 | 99.6 | 97.2 | 91.6 | 96.6 |
| Motus | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** |
| FastWAM† | ✗ | 97.4 | 99.6 | 96.6 | 94.4 | 97.0 |
| **JAM (Wan)†** | **✗** | **98.8** | **100.0** | **98.0** | **95.8** | **98.2** |
| ImageWAM (Ovis-U1)† | ✗ | 97.8 | 100.0 | 93.6 | 80.2 | 92.9 |
| JAM (Ovis-U1)† | ✗ | 98.6 | 99.6 | 98.4 | 90.4 | 96.8 |

### Table 19: RoboTwin 2.0 完整成功率（%）

| Method | P.T. | Clean | Rand. | AVG |
|---|:---:|---|---|---|
| π0 | ✓ | 65.92 | 58.40 | 62.16 |
| π0.5 | ✓ | 82.74 | 76.76 | 79.75 |
| JAM (Qwen3.5-2B)† | ✗ | 89.70 | 89.86 | 89.78 |
| Motus | ✓ | 88.66 | 87.02 | 87.84 |
| LingBot-VA | ✓ | 92.90 | 91.50 | 92.20 |
| FastWAM† | ✗ | 91.22 | 91.20 | 91.21 |
| JAM (Wan)† | ✗ | 91.70 | 91.18 | 91.44 |
| ImageWAM (Ovis-U1)† | ✗ | 89.14 | 88.78 | 88.96 |
| ImageWAM (FLUX.2)† | ✗ | 92.84 | 93.04 | 92.94 |
| JAM (Ovis-U1)† | ✗ | 90.34 | 89.46 | 89.90 |
| **JAM (FLUX.2)†** | **✗** | **92.88** | **93.28** | **93.08** |

### Table 20: 统计显著性检验（配对 McNemar 检验）

| Baseline | Benchmark | $\Delta$ | 95% Bootstrap CI | Baseline-only / JAM-only | McNemar p |
|---|---|---|---|---|---|
| FastWAM | LIBERO | +1.15pp | [+0.20, +2.10] | 30/53 | 0.015 |
| ImageWAM | LIBERO | +3.85pp | [+1.00, +7.40] | 42/119 | 1.01×10⁻⁹ |
| FastWAM | LIBERO-Plus | +9.23pp | [+6.84, +11.69] | 111/250 | 1.91×10⁻¹³ |
| ImageWAM | LIBERO-Plus | +14.67pp | [+12.15, +17.13] | 89/310 | 9.69×10⁻³⁰ |

**关键发现**: 所有四项对比的 p 值均 < 0.05，JAM 的提升具有统计显著性。

### Table 21: 三次独立运行（%, 均值 ± 标准差）

| Policy | Run | Spatial | Object | Goal | Long | AVG | LIBERO-Plus |
|---|---|---|---|---|---|---|---|
| JAM (Wan) | 1 | 98.80 | 100.00 | 98.00 | 95.80 | 98.15 | 64.54 |
| JAM (Wan) | 2 | 98.60 | 99.80 | 98.20 | 95.40 | 98.00 | 64.28 |
| JAM (Wan) | 3 | 99.00 | 100.00 | 97.80 | 96.20 | 98.25 | 65.01 |
| **JAM (Wan) Mean ± SD** | | **98.80 ± 0.20** | **99.93 ± 0.12** | **98.00 ± 0.20** | **95.80 ± 0.40** | **98.13 ± 0.13** | **64.61 ± 0.37** |
| JAM (Ovis-U1) | 1 | 98.60 | 99.60 | 98.40 | 90.40 | 96.75 | 79.95 |
| JAM (Ovis-U1) | 2 | 98.80 | 99.60 | 98.40 | 90.80 | 96.90 | 80.35 |
| JAM (Ovis-U1) | 3 | 98.40 | 99.80 | 98.20 | 90.00 | 96.60 | 79.48 |
| **JAM (Ovis-U1) Mean ± SD** | | **98.60 ± 0.20** | **99.67 ± 0.12** | **98.33 ± 0.12** | **90.40 ± 0.40** | **96.75 ± 0.15** | **79.92 ± 0.43** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件，各 50 任务 | 桌面操作，标准评测 | 训练 + 测试 |
| [[LIBERO-Plus]] | 7 类扰动，1,506 任务子集 | 摄像头/光照/背景/噪声等 OOD 泛化评测 | 测试（鲁棒性） |
| [[RoboTwin]] 2.0 | 50 任务，100 trials/task/setting | SAPIEN 3.0，Clean + Randomized | 训练 + 测试 |

### 实现细节

- **骨干**: Wan2.2-5B（30块）/ Yak18/Ovis-U1（18块）/ FLUX.2 Klein 4B / Qwen3.5-2B（24层）/ DINOv2-B/L
- **优化器**: AdamW，betas=(0.9, 0.95)，weight decay=0.01
- **学习率**: $1 \times 10^{-4}$，cosine decay，5% warmup
- **梯度裁剪**: 1.0
- **混合精度**: bf16
- **并行策略**: ZeRO-1（Wan）/ ZeRO-0（其他）
- **动作块长度**: 32 步（Wan）/ 16 步（其他）
- **批大小（LIBERO）**: global batch 128
- **训练步数（LIBERO）**: 20k steps
- **硬件**: 8× NVIDIA GH200

### 可视化结果

- JAM 的动作-视觉注意力热力图显示其更专注于任务相关区域（物体、目标位置），而 ImageWAM 注意力更分散。
- 探针重建实验（PSNR）证明 JAM 骨干内编码的未来语义信息量与 ImageWAM 相当（19.57 vs 19.43 dB）。

---

## 批判性思考

### 优点

1. **概念清晰**：将 VLA vs WAM 的争论拆解为三个可独立操控的维度，提供了系统的分析框架
2. **计算高效**：目标缓存后训练 FLOPs 降低 73%，在不牺牲性能的前提下显著降低训练成本
3. **实验严谨**：配对 McNemar 检验、置信区间、多次独立运行，统计验证充分
4. **广泛验证**：在 5 种不同骨干（生成、自监督、VLM）上均验证了 JAM 的有效性

### 局限性

1. **跨骨干比较不受控**：不同骨干的架构和动作接口不同，无法纯粹隔离预训练的影响；作者也明确承认此点
2. **LIBERO-Plus 为作者自评**：1,506 任务子集由作者定义并重新评测，缺乏第三方复现
3. **训练预算较多**：10k 步时 LIBERO-Plus 仅 48.47%（远低于 20k 步的 55.58%），说明先见目标需要更多步数才能充分训练

### 潜在改进方向

1. 探索更动态的锚（如轻量在线锚），进一步节省离线缓存存储开销
2. 将先见损失应用于真实机器人数据（目前仅在仿真中验证）
3. 研究先见偏移量的自适应选择（当前为手动设定）

### 可复现性评估

- [ ] 代码开源（未提供）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（详细记录于附录）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均公开）

---

## 关联笔记

### 基于

- [[FastWAM]]: JAM 在 Wan 骨干上的直接改造基础，沿用其策略架构并替换生成目标
- [[ImageWAM]]: JAM 在 Ovis-U1 骨干上的对比基准
- [[Flow Matching]]: JAM 动作专家使用流匹配生成动作块

### 对比

- [[pi0.5]]: LIBERO 上的顶级 VLA（96.9%），JAM（Wan）达到 98.2% 超越之
- [[LingBot-VA]]: LIBERO 最高成绩（98.5%），JAM（Wan）接近
- [[WorldVLA]]: LIBERO-Plus 上失败的对比方法（25.0%）

### 方法相关

- [[Foresight Loss]]: 本文提出的核心新概念
- [[Action Chunking]]: 动作块预测机制
- [[DINOv2]]: JAM 使用的自监督骨干之一
- [[Stop-Gradient]]: 先见损失中阻止梯度流向未来锚
- [[World Action Model]]: JAM 分析和对比的主要范式

### 硬件/数据相关

- [[LIBERO]]: 主要评测数据集
- [[RoboTwin]]: 第二评测数据集

---

## 速查卡片

> [!summary] JAM: Just Action Models
> - **核心**: 将未来监督（先见损失）与视觉生成解耦；训练时用轻量预测头学习未来表示，推理时丢弃
> - **方法**: $\mathcal{L}_\mathrm{JAM} = \mathcal{L}_\mathrm{act} + \lambda_F \mathcal{L}_F$，先见损失用 cosine 距离对齐冻结未来锚特征
> - **结果**: LIBERO 98.2%（超π0.5 96.9%）；LIBERO-Plus 81.1%（Qwen3.5 最高）；RoboTwin 93.1%；训练 FLOPs 降低 73%
> - **代码**: 未公开（项目页：y0uroy.github.io/JAM）

---

*笔记创建时间: 2026-10-29*

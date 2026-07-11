---
title: "Multiplayer Interactive World Models with Representation Autoencoders"
method_name: "MIRA"
authors: [Anthony Hu, Václav Volhejn, Adrien Ramanana Rahary, Chris Mulder, Aditya Makkar, Alyx Liao, Amélie Royer, Manu Orsini, Adam Jelley, Eloi Alonso, Florian Laurent, Fredrik Norén, James Swingos, Jan Hünermann, Kent Rollins, Lucas Hosseini, Matthieu Le Cauchois, Maxim Peter, Pim de Witte, Tim Brown, Vincent Micheli, Moritz Böhle, Gabriel de Marmiesse, Viktoriia Sharmanska, Lucia Specia, Michael Black, Patrick Pérez]
year: 2026
venue: arXiv Preprint
tags: [world-model, multiplayer, diffusion-model, flow-matching, video-generation, game-ai, representation-learning, streaming-inference, representation-autoencoder, rocket-league]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2607.05352
created: 2026-07-11
---

# 论文笔记：Multiplayer Interactive World Models with Representation Autoencoders

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | General Intuition; Kyutai; Epic Games |
| 日期 | July 2026 |
| 项目主页 | https://mira-wm.com |
| 对比基线 | [[DIAMOND]], [[Genie2]], [[GameNGen]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.05352) / [Code](https://github.com/mira-wm/mira) |

---

## 一句话总结

> MIRA 是首个多人交互式世界模型，通过冻结 [[DINOv3|DINOv3-L]] 特征构建表示自编码器（RAE）作为潜在空间，结合流匹配 + [[Diffusion Forcing|扩散强制]] + 渐进自蒸馏，在单张 B200 GPU 上以 20fps 实时生成四人火箭联盟对战。

---

## 核心贡献

1. **首个多人世界模型**: 首次建模 4 个同时在线玩家的高动态多智能体游戏环境（Rocket League 2v2），通过 tiled view + 联合空间注意力实现跨玩家交互感知，每位玩家独立动作条件控制
2. **表示自编码器（RAE）**: 冻结预训练 [[DINOv3|DINOv3-L]] 视觉特征作为编解码器潜在空间，天然继承语义结构，将世界模型训练与像素重建目标完全解耦，使潜在空间与像素空间相比更稳定、更可控
3. **实时流式生成系统**: 流匹配 + 扩散强制 + [[PSD|渐进自蒸馏（PSD）]]+ KV-cache 流式推理，实现 ~70ms/步（20fps），可在训练时长 200 倍以上（超 5 分钟）的超长生成中维持分布质量稳定

---

## 问题背景

### 要解决的问题

构建能在高度动态、高速（球速 150 km/h）、多人交互游戏环境中实时生成连贯视频的世界模型，同时支持每位玩家独立动作条件控制，并能在训练时长之外保持长期稳定性。具体而言：如何在多智能体环境中建模玩家间交互？如何在极端动态场景下保持生成质量？如何消除 teacher-forcing 训练带来的 train/inference 分布差距？

### 现有方法的局限

- **像素空间世界模型**（如 GameNGen）：训练不稳定，难以扩展到 5B 参数规模，高速动态场景（球快速运动）的渲染质量差，动作控制信号难以有效利用视觉语义
- **传统 VAE 潜在空间**（如 Stable Diffusion 的 VAE）：潜在空间不保留语义结构，专门优化像素重建损失，使世界模型隐式地需要学习视觉外观细节而非物理交互规律
- **单玩家世界模型**（如 DIAMOND、Genie2）：无法建模多玩家同步动作和玩家间交互，不支持联合多视角生成
- **Teacher-forcing 训练**：强制每步使用真实帧作为上下文，推理时使用自身生成帧则累积误差，导致长程生成快速退化

### 本文的动机

作者认为三个核心见解可以解决上述问题：
1. **冻结预训练特征作为潜在空间**：使用 DINOv3-L 的冻结中间层特征聚合，而非训练专用 VAE。DINOv3-L 特征自然编码语义结构（对象、运动、空间关系），世界模型在此空间学习物理规律更容易，且与解码器的像素重建完全分离
2. **Tiled view + 联合注意力**：将 4 个玩家潜在视角在空间维度拼接，DiT 的双向空间注意力直接处理玩家间交互，无需显式多玩家协调机制
3. **扩散强制 + PSD**：每帧独立噪声时间步消除 train/inference 分布差距（模型在部分污染上下文上训练），PSD 蒸馏使推理仅需 1-2 次函数评估，实现实时生成

---

## 方法详解

### 模型架构

<!-- 使用 [[概念]] 内联链接所有技术术语 -->

MIRA 采用 **两阶段解耦** 架构：视频表示编解码器（600M 参数）+ 潜在空间世界模型（5B 参数）。

- **输入**: 720p 视频帧 $x_t \in \mathbb{R}^{H \times W \times 3}$ + 4 个玩家动作 $a_t^{1:P}$（9 个按键：W/A/S/D/Q/E/SPACE/LSHIFT/LCTRL，15Hz）
- **编码器**: 冻结 [[DINOv3|DINOv3-L/16]]（特征聚合自块 $\{11,13,15,17,19,21,23\}$ 均值 + 最深层）→ 学习的线性层 2×2×2 时空下采样（1024→32 通道，空间/32，时间 2×，10Hz）
- **世界模型**: 5B 参数流匹配 [[DiT|扩散 Transformer]]，联合建模所有玩家潜在 $z^{1:P}_{t+1}$
- **解码器**: 空间 2× 上采样（/32→/16）+ 因果时空 [[ViT|ViT-XL]]（28层，宽度 1152，16头）+ 时间 2× 上采样
- **总参数**: 编解码器 ~600M；世界模型 5B

### 核心模块

#### 模块1: 表示自编码器（Representation Autoencoder, RAE）

**设计动机**: 利用 [[DINOv3|DINOv3-L]] 自监督视觉特征构建潜在空间，使世界模型天然继承语义结构（对象中心性、空间连贯性），避免从零优化像素重建目标，同时将编解码器训练与世界模型训练完全解耦。

**具体实现**:
- **编码器（冻结）**: 提取 DINOv3-L/16 第 $\{11,13,15,17,19,21,23\}$ 层 + 最深层 patch 特征，取均值聚合 → 学习线性层将 1024 维 2×2×2 时空下采样至 32 通道（输出分辨率：空间/32，时间 10Hz）
- **解码器（可训练）**: 空间 2× 上采样 → 因果时空 ViT-XL（28 层，隐维 1152，16 注意力头，因果时间注意力 + 双向空间注意力）→ 时间 2× 上采样
- **损失函数**: L1 重建 + [[LPIPS]] 感知损失 + DINOv3 特征对齐正则化，各项损失权重使用自适应梯度归一化平衡

#### 模块2: 多人潜在世界模型（5B DiT）

**设计动机**: 以 [[Flow Matching|流匹配]] 为训练目标，通过因式化时空注意力高效建模时序动力学，通过 tiled view 将多玩家交互统一为空间问题，通过 [[Diffusion Forcing|扩散强制]] 消除 train/inference 分布差距，通过 [[AdaLN|自适应层归一化]] 注入动作条件。

**具体实现**:
- **多玩家 Tiled View**: 4 个玩家潜在视角 $z_t^{1:P}$ 在高度方向拼接为 $(2P) \times W$ 的 grid → 单次联合空间注意力处理所有玩家间交互
- **因式化时空注意力**: 空间注意力（帧内双向，处理玩家间交互）+ 时间注意力（同一空间位置跨帧因果，处理时序动力学）；两种注意力交替应用于各 DiT 层
- **动作条件化**: 每玩家 9 键动作 + 玩家 ID 嵌入 → 线性映射为 [[AdaLN]] 参数 $(\gamma, \beta)$，调节每个 DiT 层归一化；以 $p=0.1$ 概率将动作替换为学习的 "absent token"（训练中随机丢弃支持无动作自动驾驶模式）
- **扩散强制**: 每帧独立采样流时间 $\tau^i \sim \mathcal{U}(0,1)$，上下文帧设 $\tau=0$（干净），以部分污染上下文训练使模型能处理自生成历史
- **架构细节**: 隐维 4096，16 Transformer 层，32 注意力头 / 8 KV 头（[[GQA|分组查询注意力]]），因式化 [[RoPE|旋转位置编码]]（空间+时间独立），[[SwiGLU]] FFN，寄存器 token

#### 模块3: 渐进自蒸馏（Progressive Self-Distillation, PSD）

**设计动机**: 将推理时所需的函数评估次数从数十步降至 1-2 步，实现实时生成，同时不显著损失生成质量。

**具体实现**:
- **教师网络**（冻结 $\theta$）：在区间 $[s,t]$ 内执行两个子步骤 $[s,u]$ 和 $[u,t]$，计算等效速度场目标
- **学生网络** $\phi$：一步拟合教师的两步等效结果，迭代蒸馏后仅需 1-2 次 NFE（函数评估次数）
- **额外条件**：学生同时以 $t-s$（步长）作为额外输入，以适应不同步长

#### 模块4: 流式推理系统

**具体实现**:
- [[KV Cache|KV-cache]]：缓存历史帧的 Key-Value 注意力对，仅计算新帧注意力
- **滚动上下文窗口**: 维持最近 $T=20$ 个潜在帧（对应 2 秒历史）
- **少步采样**: PSD 蒸馏后 1-2 次 NFE
- **速度**: 单 Nvidia B200 GPU，~70ms/步 → 20fps 实时生成，超训练序列长度 200 倍（>5 分钟）分布质量稳定

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

### 公式1: [[DINOv3|DINOv3-L 特征层聚合]]

$$
\bar{f} = \frac{1}{|S|}\sum_{\ell \in S} f_\ell + f_{\ell_{\max}}
$$

**含义**: 对 DINOv3-L 的指定中间层集合取均值并叠加最深层特征，得到编解码器编码器所使用的聚合视觉特征。

**符号说明**:
- $f_\ell$: DINOv3-L/16 第 $\ell$ 层的 patch 特征
- $S = \{11, 13, 15, 17, 19, 21, 23\}$: 采样的中间层索引集合
- $f_{\ell_{\max}}$: 模型最深层特征（提供最抽象语义）

---

### 公式2: [[Codec|视频编解码器损失]]

$$
\mathcal{L} = \|x - \hat{x}\|_1 + \lambda_p \,\text{LPIPS}(x, \hat{x}) + \lambda_d \,\|\phi(x) - \phi(\hat{x})\|^2_2
$$

$$
\lambda = \frac{\|\nabla_\theta \mathcal{L}_{rec}\|}{\|\nabla_\theta \mathcal{L}_{perc}\| + \epsilon}
$$

**含义**: 结合 L1 像素重建损失、[[LPIPS]] 感知损失和 DINOv3 特征对齐损失训练解码器（编码器冻结），其中各项权重通过自适应梯度归一化动态调整。

**符号说明**:
- $\lambda_p$: LPIPS 感知损失权重（自适应梯度归一化动态调整）
- $\lambda_d$: DINOv3 特征对齐损失权重
- $\phi(\cdot)$: 冻结 DINOv3-L 特征提取器
- $\lambda$: 自适应权重，使各损失项梯度范数保持平衡

---

### 公式3: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau, z_0, z_1}\left[\|v_\theta(z_\tau, \tau) - (z_1 - z_0)\|^2_2\right]
$$

$$
z_\tau = \tau z_1 + (1 - \tau) z_0
$$

**含义**: 学习速度场 $v_\theta$，使其在插值路径上与最优传输方向（直线）一致，从噪声 $z_0$ 推向数据 $z_1$。

**符号说明**:
- $z_0 \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $z_1$: 真实潜在编码（来自冻结 DINOv3-L 编码器）
- $\tau \sim \mathcal{U}(0, 1)$: 流时间（均匀采样）
- $z_\tau$: 时刻 $\tau$ 的插值状态

---

### 公式4: [[Diffusion Forcing|扩散强制]] 世界模型条件

$$
p_\theta\bigl(z^{1:P}_{t+1} \mid z^{1:P}_{\leq t},\, a^{1:P}_{\leq t}\bigr)
$$

每帧独立采样 $\tau^i \sim \mathcal{U}(0, 1)$，上下文帧设 $\tau^{ctx} = 0$（干净），目标帧 $\tau^{tgt} \sim \mathcal{U}(0, 1)$。

**含义**: 世界模型条件于所有历史玩家帧（可能含部分噪声）和动作，联合预测下一时刻 $P$ 个玩家视角。独立帧噪声消除 teacher-forcing 与推理间的分布差距。

---

### 公式5: [[PSD|渐进自蒸馏（PSD）]]目标

$$
v^*_\phi(z_s, s, t-s) = \frac{1}{2}\left[v_\theta(z_s, s, u-s) + v_\theta(\hat{z}_u, u, t-u)\right]
$$

$$
\hat{z}_u = z_s + (u - s)\,v_\theta(z_s, s, u-s)
$$

**含义**: 教师网络（冻结 $\theta$）将 $[s,t]$ 区间的流匹配积分分为 $[s,u]$ 和 $[u,t]$ 两子步计算等效速度场目标，学生网络 $\phi$ 以一步拟合该目标，使推理步数减半。

**符号说明**:
- $v_\theta$: 教师速度场（冻结）
- $v_\phi$: 学生速度场（训练中）
- $u = (s + t)/2$: 中间时间点
- $t - s$: 步长（学生的额外输入条件）

---

### 公式6: [[AdaLN|自适应层归一化]]（动作条件化）

$$
\text{AdaLN}(x, c) = (1 + \gamma(c))\,\text{LN}(x) + \beta(c)
$$

**含义**: 将玩家动作嵌入 $c$（动作向量 + 玩家 ID）通过学习的线性映射生成尺度 $\gamma$ 和偏移 $\beta$，对 DiT 每层的层归一化输出进行动作条件化调制。

**符号说明**:
- $c$: 动作嵌入（9键状态 + 玩家 ID 拼接后线性映射）
- $\gamma(c), \beta(c)$: 条件化线性层输出
- $\text{LN}(\cdot)$: 标准层归一化

---

### 公式7: [[ARR|动作可恢复比]]（可控性评估指标）

$$
\text{ARR}(a) = \frac{\text{AP}_{gen}(a)}{\text{AP}_{recon}(a)}
$$

**含义**: 量化世界模型对特定动作 $a$ 的条件控制能力；ARR=1 表示生成视频与重建视频（理想上界）等效地反映动作信号，ARR 接近 0 表示动作控制失效。

**符号说明**:
- $\text{AP}_{gen}(a)$: 在生成视频上识别动作 $a$ 的平均精度（Average Precision）
- $\text{AP}_{recon}(a)$: 在重建视频（编解码器过一次）上识别动作 $a$ 的 AP（理想上界）

---

## 关键图表

<!-- 图片：下载到本地 assets/ 文件夹，用 ![[]] wikilink 嵌入 -->
<!-- 命名规范: MultiWM_fig{N}.png -->
<!-- 下载后必须验证：文件 >10KB、Read 确认内容正确 -->

### Figure 1: Overview / 系统概览

![[MultiWM_fig1.png|600]]

**说明**: MIRA 系统概览（Paper Figure 1 "World model imagination"）。4 位玩家视角（P1-P4 行）×3 个时间步（$t_0$, $t_0$+6s, $t_0$+8s 列）的生成对战帧网格。给定初始帧和玩家动作流，MIRA 想象出包含球高速运动、四辆车协调争球的动态比赛场景，各玩家视角保持时间和空间一致性（如计分板实时更新）。

---

### Figure 2: System Architecture / 整体架构

![[MultiWM_fig2.png|600]]

**说明**: MIRA 整体流程（Paper Figure 2 "Method overview"）。左侧：4 个玩家的输入帧经共享的表示编解码器编码为 per-player 潜在，形成固定长度上下文窗口（T=20）。中央：潜在世界模型在上下文条件和动作流条件下以 10Hz 因果预测下一潜在帧，结合高效推理（KV-cache + 流式滚动窗口 + 少步扩散蒸馏）。右侧：编解码器解码器将预测潜在重建为每玩家 20fps 视频帧（1 潜在步产生 2 视频帧）。

---

### Figure 3: Representation Autoencoder / 表示自编码器

![[MultiWM_fig3.png|600]]

**说明**: 表示自编码器（RAE）详细结构（Paper Figure 3 "Codec"）。编码器（FROZEN）：冻结的 DINOv3-L/16 预训练图像特征提取器通过特征聚合获得 patch 特征，再经学习的线性 2×2×2 时空瓶颈下采样至潜在 $z$（32通道，/32空间，10Hz）。解码器（全部 TRAINED）：空间上采样层（/32→/16）→ 因果时空 ViT-XL 解码器（全空间注意力 + 因果时间注意力）→ 时间上采样。训练目标：L1 + LPIPS + P-DINO 三项损失，自适应梯度归一化平衡。

---

### Figure 4: World Model DiT / 世界模型 DiT 架构

![[MultiWM_fig4.png|600]]

**说明**: 世界模型训练流程与内部架构（Paper Figure 4 "World model training and architecture"）。**顶部（训练流程）**：P 个玩家视角帧经共享冻结编码器映射为 per-player 潜在 $z_1^p$，拼接为联合干净潜在 $z_1$；高斯噪声 $z_0 \sim \mathcal{N}(0,I)$ + 扩散强制（每帧独立 $\tau \in [0,1]$）生成带噪潜在 $z_\tau$；与所有玩家动作 $a^{1:P}$ 一起送入潜在世界模型预测速度 $v_\theta$，流匹配损失 $\|v_\theta - (z_1-z_0)\|^2$ 训练。**底部（Transformer 块）**：N 个堆叠的 AdaLN 时空 Transformer 块，每块包含空间注意力（双向，GQA+寄存器token）→ 时间注意力（因果，GQA）→ AdaLN 调制 SwiGLU MLP。动作嵌入 $\text{emb}(a^{1:P})$ + 时间嵌入 $\text{emb}(\tau)$ 求和得条件向量 $c$，MLP 输出 AdaLN 参数 $\gamma, \beta$。

---

### Figure 5: Demo - Demolition Event / 击碎事件演示

![[MultiWM_fig5.png|600]]

**说明**: 多玩家世界一致性演示（Paper Figure 5 "World model imagination of a demolition event"）。展示 2v2 沙漠竞技场对战中的击碎（demolition）事件，4 个玩家视角（行）×3 个时间步（$t_0, t_1, t_2$）。事件在各视角中保持完美一致性：$t_2$ 时 Player 1 视角显示"DEMOLITION"标注（攻击者），Player 3 视角显示"BOOM!"爆炸特效（被击碎者），其他玩家视角看到远处的爆炸烟雾。计时钟（4:12）在所有视角中保持一致。

---

### Figure 6: Demo - Goal Event / 进球事件演示

![[MultiWM_fig6.png|600]]

**说明**: 多玩家进球一致性演示（Paper Figure 6 "World model imagination of a goal"）。展示神殿竞技场进球瞬间，4 个玩家视角×3 个时间步。$t_1$ 时所有视角同步显示"SCORED!"标注和蓝色球门爆炸特效，各玩家的摄像机位置和渲染角度不同，但事件在同一时刻跨所有视角一致出现。计分牌从 2-1 更新为 3-1 在所有视角的 HUD 中一致更新。

---

### Figure 7: PSD Few-Step Sampling / 渐进自蒸馏少步采样质量

![[MultiWM_fig7.png|600]]

**说明**: 渐进自蒸馏（PSD）少步采样效果（Paper Figure 11 "Few-step sampling"）。横轴为流匹配步数（NFE），纵轴分别为 gFID、gFVD、gFDD。蓝线为基线（无蒸馏），橙线为 PSD 蒸馏模型。PSD 在所有步数下均优于基线，尤其在 1-2 步低步数区间优势显著：2步时 gFID 约 10.7（基线约 15），实现实时推理同时保持生成质量。

---

### Figure 8: Game State Probing / 游戏状态探测

![[MultiWM_fig8.png|600]]

**说明**: 游戏状态探测演示（Paper Figure 21 "Multiplayer rollout with game-state probe"）。3 行对应 3 个不同 rollout 帧（Frame 81, 121, 161）。每行左侧展示 4 个玩家视角的生成帧（2×2 grid），右侧展示对应的顶视图竞技场地图：灰色圆/方块为球/车的真实物理状态（ground truth），橙色轮廓圆/方块为游戏状态探针从世界模型激活预测的状态。探针精确预测球位置（$R^2$ 高）和所有 4 辆车的位置，验证世界模型内部表征编码了丰富的游戏物理信息。

---

### Figure 9: Emergent Cross-View Attention / 涌现跨视角注意力

![[MultiWM_fig9.png|600]]

**说明**: 多玩家跨视角注意力涌现特性（Paper Figure 17 "The multiplayer model attends across views to shared objects and state"）。顶行为 4 个玩家的生成视角帧（列=玩家）。后续 3 行展示交叉视角空间注意力热图：查询 Player 1 的车辆 token（通过 Player 2 视角），查询球的 token（通过 Player 1 视角），查询计时钟 token（通过 Player 1 视角）。暖色=强注意力，冷色=弱注意力。Player 1 的车在其他视角中能被精确定位；球 token 在所有视角中激活；计时器 token 激活所有视角的 HUD 顶部区域，展示多玩家 tiled view 使模型在没有显式跨玩家设计的情况下学会了跨视角一致性推理。

---

### Table 1: 编解码器消融实验

| 配置 | rFID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | P-DINO ↓ |
|------|--------|--------|--------|---------|----------|
| Full RAE | **2.1** | **26.4** | **0.78** | **0.122** | **0.039** |
| w/o DINOv3 对齐损失 | 3.8 | 25.1 | 0.74 | 0.158 | 0.071 |
| w/o 冻结编码器（端到端训练） | 4.2 | 24.7 | 0.72 | 0.171 | 0.083 |
| 像素空间（无编解码器） | 18.6 | 21.3 | 0.61 | 0.312 | 0.198 |

**说明**: RAE 编解码器的消融实验。冻结 DINOv3-L 编码器和 DINOv3 特征对齐损失对重建质量均有显著贡献，移除任一组件均导致 rFID 显著上升。

---

### Table 2: 旗舰模型对比

| 方法 | gFID ↓ | gFVD ↓ | gFDD ↓ | ARR ↑ |
|------|--------|--------|--------|-------|
| 像素空间（plain） | 104.9 | 1456.3 | 16.19 | 0.61 |
| 潜在 + Teacher-forcing | 32.5 | 944.1 | 2.21 | 0.89 |
| 潜在 + Diffusion Forcing | 14.2 | 312.7 | 0.82 | 0.90 |
| **MIRA（潜在+DF+PSD）** | **10.7** | **163.1** | **0.55** | **0.91** |

**说明**: 旗舰对比结果。MIRA 在所有生成质量指标（gFID/gFVD/gFDD）上显著优于像素空间基线和 teacher-forcing 基线，ARR=0.91 表明动作控制能力接近上界（1.0）。潜在空间相对像素空间将 gFID 从 104.9 降至 10.7（-90%），扩散强制进一步降低长程生成退化，PSD 实现实时推理且仅微小损失质量。

---

### Table 3: 每动作可控性分析（ARR）

| 动作 | ARR |
|------|-----|
| 加速（boost） | 0.94 |
| 漂移（drift） | 0.92 |
| 跳跃（jump） | 0.88 |
| 空中翻滚（air-roll） | 0.91 |
| **平均** | **0.91** |

**说明**: 各动作类型的 ARR 分析，验证 MIRA 对不同类型动作的控制能力。高速动态动作（boost：0.94）的控制最准确，跳跃动作（0.88）相对较难恢复（可能因跳跃帧视觉变化较微妙）。所有动作 ARR 均高于 0.88，表明动作控制能力一致稳健。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **Rocket Science** | 10,000 match-hours | 82,983 场对战，331,932 条每玩家视角录像，3 个竞技场，Nexto bot（顶级 AI） | 预训练 + 精调 |
| Champions Field | 约 60% | 官方锦标赛标准竞技场 | 主要训练 |
| Forbidden Temple | 约 20% | 不同地形的竞技场 | 多样性训练 |
| Deadeye Canyon | 约 20% | 西部风格竞技场 | 多样性训练 |

**数据采集细节**:
- 分辨率：720p / 20fps（原始 30fps 下采样）
- 动作频率：15Hz（9 个二值键位：W/A/S/D/Q/E/SPACE/LSHIFT/LCTRL）
- 对战配置：4 玩家 2v2，均使用 Nexto bot（业余中前 0.1% 段位）
- 数据集已在 HuggingFace 公开：https://huggingface.co/datasets/mira-wm/rocket-science

### 实现细节

- **编解码器 Backbone**: 冻结 DINOv3-L/16（ViT-Large，patch size 16，不微调）
- **世界模型参数**: 5B（隐维 4096，16 层，32/8 GQA 头，因式化 RoPE，SwiGLU FFN）
- **编解码器规模**: ~600M（主要来自 ViT-XL 解码器，28L/1152D/16H）
- **训练策略**: 先在单玩家数据（1P）上预训练，再在 4 人多人数据上精调
- **优化器**: AdamW（精确参数见论文附录）
- **推理硬件**: 单 Nvidia B200 GPU
- **上下文窗口**: T=20 个潜在帧（约 2 秒历史）
- **蒸馏步骤**: PSD 迭代 3 轮，最终学生 1-2 NFE

### 可视化结果

MIRA 在 5 分钟超长生成（训练时长 200 倍以上）中维持分布稳定性：
- gFVD 保持在 163.1 左右（vs. Teacher-forcing 在 30 秒后急速退化至 2000+）
- 高速运动区域（球、汽车）渲染清晰，无明显模糊或伪影
- 游戏状态探针 $R^2$ 在长程生成中保持稳定（球位置 $R^2 \approx 0.87$）
- 4 个玩家视角保持多玩家交互一致性（例如球在不同视角中的相对位置关系正确）
- 动作控制响应即时：boost 动作在 1-2 帧内对速度场有可见影响

---

## 批判性思考

### 优点

1. **冻结特征潜在空间的洞见深刻**: 用冻结 DINOv3-L 特征作为潜在空间是本文最核心的创新。这一设计既避免了 VAE 潜在空间与世界模型训练目标冲突的问题，又天然提供了语义丰富的表示，使动作控制信号更易利用视觉语义
2. **多玩家 tiled view 优雅简洁**: 将多玩家问题化约为空间问题，无需设计专门的跨玩家通信机制，充分利用 DiT 的大容量空间注意力
3. **长程稳定性突出**: 5 分钟超过训练时长 200 倍的稳定生成，在当前世界模型中属领先水平，扩散强制是关键设计

### 局限性

1. **数据集高度特定**: 仅在 Rocket League 2v2 场景上验证，泛化到其他游戏或真实世界视频尚未探索；特定竞技场和 Nexto bot 行为分布可能限制泛化性
2. **计算资源要求高**: 5B 世界模型 + 600M 编解码器，训练需要大量 GPU 资源，仅实验中使用 B200 等高端硬件，普通研究者难以复现全量训练
3. **动作空间简单**: Rocket League 仅 9 个二值键位，与机器人控制的连续动作空间或复杂游戏的高维动作空间相比较简单；扩展到连续动作的效果未知
4. **图像分辨率限制**: 720p 输出分辨率（/32 潜在空间），对于需要更高分辨率细节的应用可能不足

### 潜在改进方向

1. **扩展到真实世界视频**: RAE 的冻结 DINOv3-L 特征天然适合真实世界视觉，探索将 MIRA 框架迁移至机器人操作或自动驾驶等真实场景
2. **连续动作空间**: 将离散键位动作扩展为连续控制信号（如操纵杆、肌肉控制），探索在机器人世界模型中的应用
3. **分层多玩家架构**: 当玩家数量更多时（如 5v5）tiled view 会使分辨率线性下降，探索层次化或稀疏交互机制
4. **语言条件控制**: 结合自然语言指令和动作条件，探索语言条件多玩家世界模型

### 可复现性评估

- [x] 代码开源（https://github.com/mira-wm/mira）
- [ ] 预训练模型（未公开世界模型权重）
- [x] 训练细节完整（论文附录提供超参数）
- [x] 数据集可获取（Rocket Science 数据集在 HuggingFace 公开）

---

## 关联笔记

### 基于

- [[DINOv3]]: 冻结编码器提供语义潜在空间，RAE 的核心组件
- [[Flow Matching]]: MIRA 世界模型的训练目标框架（Lipman et al. 2022）
- [[Diffusion Forcing]]: 独立帧流时间采样策略（Chen et al. 2024），消除 train/inference 差距
- [[Video Diffusion Transformer]]: 因式化时空注意力 DiT 架构基础

### 对比

- [[DIAMOND]]: 像素空间 Atari 游戏世界模型，MIRA 通过潜在空间显著超越
- [[Genie2]]: Google DeepMind 单玩家交互世界模型，MIRA 首次扩展到多玩家
- [[GameNGen]]: Doom 像素空间世界模型基线，MIRA 在动态场景质量上大幅领先

### 方法相关

- [[Representation Autoencoder]]: 核心编解码器设计，冻结特征潜在空间
- [[Factorized Attention]]: 因式化时空注意力，MIRA 世界模型效率关键
- [[PSD|Progressive Self-Distillation]]: 渐进自蒸馏实现实时推理
- [[AdaLN|Adaptive Layer Normalization]]: 动作条件化注入机制
- [[GQA|Grouped Query Attention]]: 世界模型 KV 头分组（32/8），降低推理显存

### 硬件/数据相关

- [[Rocket League]]: 数据来源游戏，3D 多玩家高速球类竞技
- [[Nvidia B200]]: 实时推理使用的 GPU 硬件
- [[Rocket Science Dataset]]: 本文贡献的 10,000 小时多玩家游戏数据集

---

## 速查卡片

> [!summary] Multiplayer Interactive World Models with Representation Autoencoders (MIRA)
> - **核心**: 首个多人世界模型，冻结 DINOv3-L 特征作为 RAE 潜在空间，流匹配 + 扩散强制 + PSD 蒸馏
> - **方法**: 600M RAE（冻结 DINOv3-L 编码器 + 因果 ViT-XL 解码器）+ 5B 流匹配 DiT（因式化时空注意力 + AdaLN 动作条件 + 多玩家 tiled view + 扩散强制 + PSD）
> - **结果**: gFID=10.7 / gFVD=163.1 / ARR=0.91，单 B200 20fps 实时生成，超 5 分钟长期稳定
> - **代码**: https://github.com/mira-wm/mira

---

*笔记创建时间: 2026-07-11*

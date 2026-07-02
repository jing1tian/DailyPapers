---
title: "DVG-WM: Disentangled Video Generation Enables Efficient Embodied World Model for Robotic Manipulation"
method_name: "DVG-WM"
authors: [Ziyu Shan, Zhenyu Wu, Xiaofeng Wang, Zheng Zhu, Ziwei Wang]
year: 2026
venue: arXiv
tags: [world-model, video-generation, robotic-manipulation, flow-matching, disentangled-learning, diffusion-policy, video-prediction]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.32028v1
created: 2026-07-02
---

# 论文笔记：DVG-WM: Disentangled Video Generation Enables Efficient Embodied World Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Nanyang Technological University; Beijing University of Posts and Telecommunications; GigaAI |
| 日期 | June 2026 |
| 项目主页 | [zyshan0929.github.io/DVGWM](http://zyshan0929.github.io/DVGWM) |
| 对比基线 | [[LongScape]] / CogVideoX |
| 链接 | [arXiv](https://arxiv.org/abs/2606.32028) |

---

## 一句话总结

> DVG-WM 将具身世界模型解耦为低分辨率**动力学预览**和高分辨率**视觉精炼**两阶段，利用[[Flow Matching]]实现 3.97× 推理加速，同时在真实机器人操作任务上取得 62.2% 成功率。

---

## 核心贡献

1. **解耦视频生成框架**: 将[[世界模型]]的动力学学习与视觉合成显式分离为两阶段流水线，低分辨率预览阶段专注接触动力学，高分辨率精炼阶段专注视觉质量。
2. **基于 [[Flow Matching]] 的级联机制**: 精炼阶段使用流匹配在低分辨率潜码与高分辨率潜码之间建立直接映射路径，仅需 4 步去噪即可完成上采样，大幅降低冗余计算。
3. **潜空间退化机制 (Latent Degradation)**: 在潜在空间而非像素空间对预览特征注入高斯噪声扰动，防止模型走捷径直接上采样，迫使精炼阶段重建接触细节，在真实操作（插接头、齿轮装配等）任务上效果尤为突出。

---

## 问题背景

### 要解决的问题

视频预测式具身世界模型同时承担**动力学建模**（物体运动、接触关系）和**视觉合成**（纹理、光照、高分辨率细节）两项任务，二者耦合导致模型必须以高分辨率处理整个去噪轨迹，推理速度慢且对接触细节的学习被视觉纹理干扰。

### 现有方法的局限

- 端到端高分辨率视频生成（如 [[CogVideoX]]）需要大量去噪步骤，推理延迟高（354.2 s），难以满足在线规划需求。
- 简单的级联超分辨率（DDPM 条件方式）会将低分辨率视频作为条件拼接输入，模型仍从噪声中从头生成高分辨率视频，未能利用预览阶段已有的动力学信息。
- 像素空间退化（先下采样再上采样）在接触密集区域（如夹爪与物体接触面）容易丢失语义细节。

### 本文的动机

若将视频世界模型解耦：预览阶段用小模型学习低分辨率动力学（快），精炼阶段用 [[Flow Matching]] 将低分辨率潜码**直接映射**到高分辨率潜码（不重新采样噪声，仅做流匹配推进），则可以在保留动力学准确性的同时大幅减少计算量。潜空间退化机制则保证精炼阶段不退化为简单插值。

---

## 方法详解

### 模型架构

DVG-WM 采用**两阶段级联视频生成**架构：

- **输入**: 语言指令 $c$ + 初始观测帧 $x_0 \in \mathbb{R}^{H \times W \times 3}$
- **预览 Backbone**: [[CogVideoX]]-5B + [[LoRA]] 微调（rank 128），分辨率 256×384
- **精炼 Backbone**: [[CogVideoX]]-2B，分辨率 480×720
- **时序压缩**: [[Causal VAE|3D Causal VAE]]（时间×空间压缩比 4×8×8）
- **动作专家**: 视觉-仅 [[Diffusion Policy]]（逆动力学模型，[[Inverse Dynamics Model|IDM]]）
- **输出**: 预测视频帧 $\hat{x}_{1:N}$ → 操作动作序列 $a_{0:T}$

### 核心模块

#### 模块 1: 预览阶段 (Preview Stage)

**设计动机**: 利用 [[CogVideoX]] 的视频生成先验，在低分辨率下快速学习机器人操作的**接触动力学**，使大量去噪步骤只在廉价的低分辨率空间发生。

**具体实现**:
- 使用 [[LoRA]]（rank 128）对 [[CogVideoX]]-5B 进行领域微调，仅训练 10,000 迭代
- 输入语言指令 $c$ 和初始帧 $x_0^d$（下采样到 256×384），执行 50 步 DDPM 去噪
- 输出低分辨率预测视频潜码 $z_{lr}$，编码物理交互轨迹

**关键发现（Fig. 6）**: 随 LoRA 微调步数增加，预览阶段**逐渐学习接触动力学，同时丢失纹理细节**——正是期望的"解耦"行为，动力学学习和视觉渲染被分离到不同阶段。

#### 模块 2: 精炼阶段 (Refinement Stage with Flow Matching)

**设计动机**: 利用 [[Flow Matching]] 在预览低分辨率潜码 $z_{lr}$ 与高分辨率目标 $z_{hr}$ 之间建立**直线路径**，将精炼过程转化为有限步流推进（仅 4 步），而非从头生成。

**具体实现**:
- 将 $z_{lr}$ 双线性上采样得到 $\bar{z}_{lr}$
- 对上采样结果施加**潜空间退化** $z̃_{lr} = \alpha_s \bar{z}_{lr} + \beta_s \varepsilon$，$\varepsilon \sim \mathcal{N}(0, I)$
- 使用 [[CogVideoX]]-2B 学习从 $z̃_{lr}$ 到 $z_{hr}$ 的向量场 $v_\theta$
- 推理时仅需 $K=4$ 步 Euler 推进即可得到高分辨率潜码 $z^K \approx z_{hr}$

**与 DDPM 条件化对比（Fig. 3）**: DDPM 条件方式将低分辨率潜码拼接为条件输入，仍从纯噪声开始完整生成，未能利用 $z_{lr}$ 的动力学信息；DVG-WM 的流匹配从 $z̃_{lr}$ 出发，沿直线路径推进，冗余计算大幅减少。

#### 模块 3: 潜空间退化 (Latent Degradation)

**设计动机**: 若精炼阶段输入是从高分辨率编码得到的低分辨率潜码（无噪声），模型容易走捷径——直接学习对潜码做上采样插值，而不重建接触细节。潜空间退化通过注入噪声打破这种捷径。

**具体实现**:
1. 先对高分辨率视频做像素下采样：$\hat{x}_{lr} = \text{Deg}_{pix}(x_{hr})$
2. 用 3D Causal VAE 编码：$z_{deg} = \mathcal{E}(\hat{x}_{lr})$
3. 在潜空间叠加高斯噪声：$z̃_{lr} = \alpha_s z_{deg} + \beta_s \varepsilon$，满足 $\alpha_s^2 + \beta_s^2 = 1$
4. 训练时用 $z̃_{lr}$ 替代干净预览潜码，强迫精炼网络重建被噪声遮盖的接触细节

**与像素退化对比（Fig. 4）**: 像素退化在放大视图中丢失接触区域细节，潜空间退化则能重建精细接触结构（夹爪关节纹路等）。

#### 模块 4: 逆动力学模型 (Inverse Dynamics Model / Action Expert)

**设计动机**: 利用 [[Inverse Dynamics Model|IDM]] 从预测视频中提取动作序列，使世界模型预测与底层控制解耦。

**具体实现**:
- 采用视觉-仅 [[Diffusion Policy]] 作为动作专家，不需要语言条件
- 以世界模型预测视频帧为条件，输出操作动作序列
- 训练分两阶段：先训练世界模型，再联合微调动作专家

---

## 关键公式

### 公式 1: [[世界模型|两阶段世界模型分解]]

$$
g_p(z_{lr} \mid x_0^d, c); \quad g_r(z_{hr} \mid z_{lr}, x_0, c)
$$

**含义**: 将世界模型分解为预览阶段 $g_p$（从下采样初始帧 $x_0^d$ 和指令 $c$ 生成低分辨率潜码）和精炼阶段 $g_r$（从低分辨率潜码和完整初始帧生成高分辨率潜码）。

**符号说明**:
- $x_0^d$: 下采样到 256×384 的初始帧
- $c$: 语言指令
- $z_{lr}$: 低分辨率预测视频潜码
- $z_{hr}$: 高分辨率预测视频潜码

---

### 公式 2: [[Flow Matching|流匹配插值路径]]

$$
z_\tau = (1-\tau)\tilde{z}_{lr} + \tau z_{hr}, \quad \tau \sim \mathcal{U}(0,1)
$$

**含义**: 在退化低分辨率潜码 $\tilde{z}_{lr}$ 与高分辨率目标 $z_{hr}$ 之间定义线性插值路径，构成条件流匹配的训练轨迹。

**符号说明**:
- $\tau$: 流匹配时间步，均匀采样自 $[0,1]$
- $\tilde{z}_{lr}$: 经过潜空间退化的上采样低分辨率潜码
- $z_{hr}$: 高分辨率真实视频潜码

---

### 公式 3: [[Flow Matching|目标速度场]]

$$
u^\star(z_\tau, \tau) \triangleq \frac{dz_\tau}{d\tau} = z_{hr} - \tilde{z}_{lr}
$$

**含义**: 沿插值路径的目标速度为常数向量，等于终点与起点之差，流匹配学习逼近该恒定速度场。

**符号说明**:
- $u^\star$: 目标向量场（ground truth velocity）
- $v_\theta$: 网络参数化的预测速度场

---

### 公式 4: [[Flow Matching|流匹配训练损失]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{z_{hr}, \tilde{z}_{lr}, \tau}\left[ \left\| v_\theta(z_\tau, \tau \mid \tilde{z}_{lr}, x_0, c) - (z_{hr} - \tilde{z}_{lr}) \right\|_2^2 \right]
$$

**含义**: 最小化精炼网络 $v_\theta$ 预测的速度与目标速度之间的 L2 距离，使网络学习从退化低分辨率潜码到高分辨率潜码的直线映射。

**符号说明**:
- $v_\theta(\cdot)$: 精炼阶段神经网络（CogVideoX-2B）的输出速度
- $x_0$: 完整分辨率初始帧（提供外观先验）

---

### 公式 5: [[Flow Matching|推理 Euler 更新]]

$$
z^{(k+1)} = z^{(k)} + \Delta\tau \cdot v_\theta\!\left(z^{(k)}, \tau_k \mid z_{lr}, x_0, c\right), \quad k = 0, \ldots, K-1
$$

**含义**: 推理时从初始化潜码 $z^{(0)}$ 出发，用 Euler 积分沿学到的速度场推进 $K$ 步（默认 $K=4$），得到高分辨率预测潜码 $z^{(K)}$。

**符号说明**:
- $\Delta\tau = 1/K$: 每步步长
- $K$: 精炼步数（消融最优值为 4）
- $z^{(0)} = \alpha_{s_{inf}} \bar{z}_{lr} + \beta_{s_{inf}} \varepsilon$: 推理时初始化，对上采样潜码加噪

---

### 公式 6: [[潜空间退化|Latent Degradation]]

$$
\tilde{z}_{lr} = \text{Deg}_{lat}(z_{deg}; s) = \alpha_s z_{deg} + \beta_s \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

满足归一化约束：

$$
\alpha_s^2 + \beta_s^2 = 1
$$

**含义**: 对编码后的低分辨率潜码 $z_{deg}$ 按信噪比参数 $s$ 混合高斯噪声，生成退化潜码 $\tilde{z}_{lr}$，防止精炼阶段走捷径。

**符号说明**:
- $\alpha_s, \beta_s$: 信号和噪声权重，由噪声强度参数 $s$ 控制
- $z_{deg} = \mathcal{E}(\hat{x}_{lr})$: VAE 编码的像素下采样视频潜码

---

### 公式 7: [[Diffusion Policy|动作扩散策略损失]]

$$
\mathcal{L}_{DP} = \mathbb{E}\left[ \left\| \varepsilon - \varepsilon_\phi(a_\sigma, \sigma \mid h_t) \right\|_2^2 \right]
$$

**含义**: 动作专家（Diffusion Policy）通过去噪分数匹配，以世界模型预测帧的视觉特征 $h_t$ 为条件，学习预测加噪动作 $a_\sigma$ 的去噪方向。

**符号说明**:
- $\varepsilon$: 真实噪声
- $\varepsilon_\phi$: 去噪网络预测的噪声
- $a_\sigma$: 加噪声级 $\sigma$ 后的动作
- $h_t$: 世界模型预测视频提取的视觉特征

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2606.32028v1/x1.png)

**说明**: DVG-WM 整体工作流程。给定初始观测和语言指令，预览阶段生成低分辨率动力学轨迹（左），精炼阶段以更快速度重建高分辨率视频（右）。右侧对比图同时展示真实任务中的推理加速效果。

---

### Figure 2: DVG-WM 完整流水线

![Figure 2](https://arxiv.org/html/2606.32028v1/x2.png)

**说明**: 完整系统结构图。预览阶段（上）以语言指令和初始帧为输入，通过 [[CogVideoX]]-5B + [[LoRA]] 生成低分辨率轨迹潜码 $z_{lr}$；精炼阶段（中）以 [[Flow Matching]] 将 $z_{lr}$ 映射到高分辨率 $z_{hr}$，经 [[Causal VAE]] 解码；逆动力学模型（下）从预测视频提取动作序列。

---

### Figure 3: 精炼方法对比

![Figure 3](https://arxiv.org/html/2606.32028v1/x3.png)

**说明**: (a) 传统 DDPM 条件化：将低分辨率潜码作为条件拼接，从纯噪声生成高分辨率视频，冗余计算高。(b) DVG-WM 流匹配：直接在 $z_{lr}$ 和 $z_{hr}$ 之间建立直线路径，Euler 积分仅需 4 步，极大降低计算成本。

---

### Figure 4: 像素退化 vs 潜空间退化

![Figure 4](https://arxiv.org/html/2606.32028v1/x4.png)

**说明**: 左侧像素退化在放大图中丢失接触细节；右侧潜空间退化通过噪声扰动促使精炼网络重建细粒度接触结构（夹爪接触面、关节纹理等），对操作任务至关重要。

---

### Figure 5: LIBERO 仿真定性对比

![Figure 5](https://arxiv.org/html/2606.32028v1/x5.png)

**说明**: DVG-WM 与 [[CogVideoX]] 在 [[LIBERO]] 任务上的视频预测对比。红圈标注 CogVideoX 的错误交互细节（物体穿透、夹爪位置偏差等），DVG-WM 正确捕获接触时序和物体状态变化。

---

### Figure 6: 预览阶段 LoRA 微调步数效果

![Figure 6](https://arxiv.org/html/2606.32028v1/x6.png)

**说明**: 蓝圈显示不同微调步数下碗的纹理变化，红圈显示夹爪与抽屉的接触细节。随微调步数增加，接触动力学逐渐清晰而纹理逐渐模糊，验证预览阶段成功实现"动力学-视觉"解耦。

---

### Figure 7: 真实世界实验平台

![Figure 7](https://arxiv.org/html/2606.32028v1/x7.png)

**说明**: Galaxea A1 机械臂平台实验设置，包含场景相机和腕部相机双视角，用于 9 项真实操作任务评测（覆盖拾取放置、堆叠、倒水、扫地、分拣等）。

---

### Figure 8: 真实操作任务定性结果

![Figure 8](https://arxiv.org/html/2606.32028v1/x8.png)

**说明**: "将面包放入碗中"任务的视频预测结果。DVG-WM 生成高保真预测视频，准确捕获抓取接触细节和语言指令对应的交互序列，验证世界模型在真实场景的泛化能力。

---

### Figure B.1: Pine 7K 数据集可视化（附录）

![Figure B.1](https://arxiv.org/html/2606.32028v1/x9.png)

**说明**: Pine 7K 真实数据集中随机三个任务的示例帧（开始/中间/结束 + 腕部视角），展示数据集覆盖的多样操作场景和双视角标注。

---

### Table 1: LIBERO 仿真定量对比

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | 物体准确率 |
|------|--------|--------|---------|-------|------------|
| CogVideoX | — | — | — | — | — |
| LVP-14B | — | — | — | — | — |
| [[LongScape]] | 19.977 | 0.788 | 0.123 | 153.72 | — |
| **DVG-WM (Ours)** | **20.019** | 0.783 | **0.120** | **152.36** | **89%** |

**说明**: DVG-WM 在 PSNR、LPIPS、FVD 上超越 [[LongScape]] 基线，同时实现 3.97× 推理加速（88.7 s vs. LVP-14B 的 354.2 s）。

---

### Table 2: 推理效率对比

| 方法 | 推理时间 (s) | 相对加速 |
|------|-------------|---------|
| LVP-14B | 354.2 | 1× (基线) |
| CogVideoX | — | — |
| **DVG-WM (Ours)** | **88.7** | **3.97×** |

**说明**: DVG-WM 通过两阶段分解（50 步低分辨率 + 4 步高分辨率精炼）将总推理时间压缩至 88.7 秒，相比 LVP-14B 实现近 4× 加速。

---

### Table 3: 真实世界任务成功率

| 方法 | 平均成功率 | 提升 |
|------|-----------|------|
| CogVideoX（基线） | 34.0% | — |
| **DVG-WM (Ours)** | **62.2%** | **+28.2pp** |

**说明**: 在 Galaxea A1 和 UR7e 两个平台的 9 项操作任务（每项 80 次 rollout）上，DVG-WM 以 62.2% 的平均成功率大幅超越 CogVideoX 基线（34.0%），在接触密集任务（插接头、齿轮装配）上提升最显著。

---

### Table 4: 消融实验

| 配置 | SSIM ↑ | FVD ↓ | 说明 |
|------|--------|-------|------|
| 无精炼阶段 | 大幅下降 | 187.69 | 动力学与视觉全耦合 |
| 无预览阶段 | 大幅下降 | — | 缺少动力学先验 |
| 像素退化（替换潜空间退化）| 0.768 | — | 接触细节丢失 |
| DDPM 条件（替换流匹配）| — | 187.69 | 低效、冗余生成 |
| 流匹配 + 像素退化 | 0.768 | — | 退化方式次优 |
| **完整 DVG-WM** | **0.783** | **152.36** | — |

**关键发现**: 两阶段解耦缺一不可；潜空间退化优于像素退化（SSIM 0.783 vs. 0.768）；流匹配显著优于 DDPM 条件化（FVD 152.36 vs. 187.69）；最优精炼步数 $K=4$ 平衡效率与质量。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | ~5,000 轨迹（8:2 划分）| 仿真桌面操作，多样任务 | 训练 + 评测 |
| Pine 7K（内部）| 6,041 集（505,779 帧，~14 h）| 双视角，80 任务，424×240 | 真实机器人训练 |

### 实现细节

- **预览 Backbone**: [[CogVideoX]]-5B，[[LoRA]] rank 128，50 步 DDPM，分辨率 256×384
- **精炼 Backbone**: [[CogVideoX]]-2B，$K=4$ 步流匹配，分辨率 480×720
- **视频时长**: 49 帧（预览）→ 49 帧（精炼）
- **预览训练**: 10,000 迭代，batch size 4，1× A100-80G
- **精炼训练**: 10 epoch，batch size 6，8× A100-80G (~24 小时)
- **训练策略**: 先训练世界模型，再联合微调动作专家
- **动作专家**: 视觉-仅 [[Diffusion Policy]]，无语言条件

### 可视化结果

- 预览阶段随 LoRA 微调步数增加，接触动力学逐渐清晰（红圈）而物体纹理逐渐模糊（蓝圈），符合设计预期（Fig. 6）
- 真实场景中 DVG-WM 生成视频与真实执行序列的接触时序高度一致（Fig. 8）
- 仿真对比中 CogVideoX 在接触关键帧出现穿透或位置错误，DVG-WM 无此问题（Fig. 5）

---

## 批判性思考

### 优点

1. **效率与质量兼顾**: 解耦设计在不显著牺牲视频质量（PSNR/SSIM/FVD 持平或小幅领先）的前提下实现近 4× 推理加速，工程实用价值高。
2. **接触细节敏感**: 潜空间退化机制针对操作任务的关键挑战（接触-密集区域）设计，真实任务成功率提升显著（+28.2pp）。
3. **方法可扩展**: 两阶段分解框架对 Backbone 无强耦合，可替换更强的视频生成基础模型。

### 局限性

1. **数据依赖**: Pine 7K 为内部数据集，未公开，影响可复现性和社区复用。
2. **两阶段误差累积**: 预览阶段错误（动力学偏差）无法在精炼阶段被修正，可能在长序列或高精度任务上积累。
3. **精炼仅做空间超分**: 当前精炼阶段专注空间分辨率提升，对时序一致性（帧间抖动）的建模未明确处理。

### 潜在改进方向

1. 引入预览-精炼循环反馈（精炼结果反哺预览的动力学估计）减少误差累积。
2. 将动作专家与世界模型做更紧密的联合优化（如 end-to-end 梯度传递），而非两阶段分离训练。
3. 开放 Pine 7K 数据集或提供 LIBERO 全量复现代码，提升社区可复现性。

### 可复现性评估

- [ ] 代码开源（项目页面提供，但截至写作时未确认代码库状态）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（论文附录提供详细超参）
- [ ] Pine 7K 数据集可获取（内部数据集，未公开）

---

## 关联笔记

### 基于

- [[CogVideoX]]: 预览和精炼阶段的 Backbone 基础模型（5B / 2B 版本）
- [[Flow Matching]]: 精炼阶段的核心训练和推理框架
- [[Diffusion Policy]]: 动作专家（逆动力学模型）

### 对比

- [[LongScape]]: 主要定量对比基线（视频质量指标）
- [[CogVideoX]]: 直接应用于操作任务的对比基线（真实场景成功率）

### 方法相关

- [[Causal VAE]]: 视频压缩的 3D 时序 VAE
- [[LoRA]]: 预览阶段的参数高效微调方法
- [[Inverse Dynamics Model]]: 从视频预测提取动作的逆动力学模型
- [[潜空间退化]]: 本文提出的核心正则化机制（需创建概念）

### 硬件/数据相关

- [[LIBERO]]: 仿真评测 benchmark
- Galaxea A1 / UR7e: 真实实验机械臂平台

---

## 速查卡片

> [!summary] DVG-WM
> - **核心**: 解耦世界模型 = 低分辨率动力学预览 + 流匹配高分辨率精炼
> - **方法**: CogVideoX-5B (preview) + Flow Matching + Latent Degradation + Diffusion Policy
> - **结果**: 3.97× 推理加速，真实操作 62.2% 成功率（vs. 34.0% baseline）
> - **代码**: [zyshan0929.github.io/DVGWM](http://zyshan0929.github.io/DVGWM)

---

*笔记创建时间: 2026-07-02*

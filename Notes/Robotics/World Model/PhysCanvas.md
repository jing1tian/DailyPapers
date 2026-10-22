---
title: "Physical Canvas: A Vision Tactile World Model For Robot Policy Evaluation And Learning"
method_name: "PhysCanvas"
authors: [Zebin Xing, Yuhang Zheng, Yupeng Zheng, Yichen Zhang, Zichen Li, Shangqing Zhou, Yujie Zang, Yuchen Liu, Zhongpu Xia, Haoran Li, Qichao Zhang]
year: 2026
venue: arXiv
tags: [visuo-tactile-world-model, contact-rich-manipulation, policy-evaluation, world-model, grpo, flow-matching, tactile-sensing]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.28613
created: 2026-10-22
---

# 论文笔记：Physical Canvas

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Institute of Automation CAS; Anyverse Robotics; Open-Phoenix Team; NUS; Xi'an Jiaotong Univ.; Beijing Inst. of Technology |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[Open-Sora]], [[WoVR]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.28613) / Code — |

---

## 一句话总结

> PhysCanvas 将多视角 RGB、触觉图像、标记位移和任务状态统一编码到单一视觉画布，结合三个冻结奖励模型的一致性后训练，实现精确的视触觉世界模型，支持真实机器人策略评估和 [[GRPO]] 在线强化学习。

---

## 核心贡献

1. **统一视觉画布（Unified Canvas）**: 将多视角 RGB、[[触觉传感器|触觉图像]]、XYZ→RGB 标记位移和任务状态统一平铺到单一图像，直接复用预训练 [[Wan2.2-TI2V-5B]] 视频 VAE，无需专用触觉编码器。
2. **一致性感知后训练（Consistency-Aware Post-Training）**: 三个冻结的奖励模型——Scene（场景保留）、State（任务成功预测）、Modality（RGB-触觉模态一致性）——在后训练阶段提供梯度监督，将 Pearson 相关从 0.602 提升至 0.885。
3. **世界模型内策略 RL（Policy RL in Imagination）**: 用世界模型作为虚拟环境，State Reward Model 提供二值奖励，通过 [[GRPO]] 在想象轨迹中更新策略，将 [[π₀.₅]] 真实成功率从 59.1%（SFT）提升至 70.8%。

---

## 问题背景

### 要解决的问题

接触密集型机器人操作（如拧瓶盖、摆放试管）中，纯视觉世界模型无法捕捉接触物理，导致世界模型预测与真实任务结果相关性差，不能可靠地用于策略评估和在线 RL。

### 现有方法的局限

- 现有视触觉世界模型（[[OmniVTA]]、[[FeelWorld]]、Dream-Tac、TACO 等）通常需要**独立的触觉编码器**，结构复杂且泛化性差。
- 纯视觉世界模型视觉上合理但忽略接触信息，导致世界模型评分与真实成功率相关性低。
- 缺乏直接利用世界模型在想象中进行在线 RL 的方法。

### 本文的动机

作者提出：是否可以用一个简单机制将异构模态编码到同一空间？是否可以不增加额外预测网络而注入高层语义？答案是将所有模态平铺到一个视觉"画布"上，并用冻结的奖励模型提供语义监督。

---

## 方法详解

### 模型架构

![Figure 1: PhysCanvas Teaser](https://arxiv.org/html/2610.28613v1/teaser.png)

**说明**: PhysCanvas 作为多模态仿真器，用于策略评估和强化学习。

PhysCanvas 以 **[[流匹配|流匹配扩散 Transformer]]** 为骨干（[[Wan2.2-TI2V-5B]]），统一处理多模态输入：

- **输入**: 初始 RGB 锚帧 + 当前多模态参考帧 + 目标关节角轨迹序列
- **Backbone**: [[Wan2.2-TI2V-5B]]（视频生成，5B 参数）
- **核心模块**: [[统一视觉画布]] → 空间动作图像注入 → 一致性后训练
- **输出**: 未来 H 帧预测（RGB + 触觉图像 + 标记位移）
- **训练规模**: 64 加速器，约 3 天；3,805 条训练轨迹

![Figure 3: PhysCanvas 方法概览](https://arxiv.org/html/2610.28613v1/main_fig.png)

**说明**: PhysCanvas 整体架构与一致性引导后训练流程。三个奖励模型冻结参数，梯度通过 VAE 解码器反传到世界模型。

---

### 核心模块

#### 模块 1: 统一视觉画布（Unified Visual Canvas）

![Figure 2: 任务概览与多模态输入](https://arxiv.org/html/2610.28613v1/View-Tactile.png)

**说明**: 四个任务的真实执行示例，及其多视角 RGB 与触觉观测。

**设计动机**: 利用 [[VAE|预训练视频 VAE]] 对触觉图像天然的重建能力（PSNR 35.51 vs. UniT VAE 的 31.72），无需触觉专用编码器。

**具体实现**:
- 多视角 RGB + [[触觉传感器|触觉图像]]（直接当图像）+ 标记位移（XYZ 映射为 RGB）+ 任务状态，拼接为单一大图
- 共享的 [[Wan2.2-TI2V-5B]] 视频 VAE 编码全部模态
- 任务状态用特定颜色编码，在 Canvas 中区分不同状态

![Figure 4: 多模态 VAE 重建](https://arxiv.org/html/2610.28613v1/PhysCanvas_VAE_20260926.png)

**说明**: 对比预训练 Wan VAE 与触觉专用 UniT VAE 的重建效果，Wan VAE 在触觉图像上 PSNR 更高（35.51 vs. 31.72）。

![Figure 5: t-SNE 多模态 token 分布](https://arxiv.org/html/2610.28613v1/tsne_readability_row.png)

**说明**: [[t-SNE]] 可视化多模态 token 在各 Transformer 层的分布演变。早期层跨模态注意力强，深层以自注意力为主。

![Figure 12: 任务状态 Canvas 编码](https://arxiv.org/html/2610.28613v1/state_canvas_visualization.png)

**说明**: State 0 和 State 1 在视觉画布上的编码示意，不同状态用不同颜色区域标注。

---

#### 模块 2: 动作图像条件化（Action Image Conditioning）

**设计动机**: 将关节动作序列通过[[前向动力学模型|前向运动学]]和相机投影渲染为与画布空间对齐的"动作图像"，实现空间位置感知的动作注入。

**具体实现**:
- 用 [[前向动力学模型|正向运动学（FK）]] + 相机投影，将 URDF 线框投影到相机视角生成动作图像
- 空间位置逐点与 Canvas token 相加（排除任务状态区域）
- 消融实验（表 3）表明，相比 AdaLN 注入，空间注入（M1）将末端执行器跟踪误差降低 32.6%

![Figure 13: 动作投影帧](https://arxiv.org/html/2610.28613v1/assets/tasks/action_projection_frame.png)

**说明**: 将 URDF 线框通过正向运动学和相机投影渲染到相机视角，生成空间对齐的动作图像。

![Figure 6: 动作干预效果](https://arxiv.org/html/2610.28613v1/action_condition_ablation.png)

**说明**: 强制开合夹爪的干预实验，展示 PhysCanvas 能正确响应动作，产生预期的物理效果（方块掉落）。

---

#### 模块 3: 一致性感知后训练（Consistency-Aware Post-Training）

三个冻结的奖励模型（[[ResNet-18]] 骨干，ResNet + MLP/Transformer 解码器，ImageNet 预训练）：

| 奖励模型 | 输入 | 判断 | 架构 |
|----------|------|------|------|
| Scene Reward Model | 初始 GT 帧 + 当前 WM 帧 | 场景是否保留 | 共享编码器 + 2层 [[Transformer解码器]] |
| State Reward Model | 三视角 RGB | 任务是否完成 | 共享编码器 + MLP（1536→256→1）|
| Modality Reward Model | RGB + 触觉图像 + 标记位移 | 视觉与触觉是否一致 | 各模态独立编码器 + 4层跨模态 [[Transformer解码器]] |

全部输出二值逻辑（BCEWithLogitsLoss），冻结参数，梯度流过 VAE 解码器回传到世界模型。

![Figure 7: 帧一致性消融](https://arxiv.org/html/2610.28613v1/frame_consistency_ablation.png)

**说明**: 场景一致性监督与模态一致性监督的消融对比，展示后训练减少时序漂移和触觉噪声的效果。

---

#### 模块 4: 世界模型内策略 RL（Policy RL in Imagination）

**设计动机**: 利用 [[GRPO]] 在世界模型提供的虚拟环境中进行在线策略优化，避免真实机器人数据采集成本，同时通过 State Reward Model 提供可靠的任务成功信号。

**具体实现**:
- 世界模型作为冻结环境，State Reward Model 作为冻结评估器
- 一个真实 reset 产生 8 条想象轨迹（一组）
- [[动作分块]] 策略以 10Hz 运行，FM/ACT 预测 32 步，[[π₀.₅]] 预测 16 步

---

## 关键公式

### 公式 1: [[流匹配|流匹配训练目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2
$$

**含义**: 学习速度场 $v_\theta$ 预测从噪声 $x_0$ 到数据 $x_1$ 的流方向；PhysCanvas 以此损失在 Wan 2.2 DiT 骨干上训练多模态未来帧生成。

**符号说明**:
- $t \sim \mathcal{U}(0,1)$: 流时间步
- $x_t = (1-t)x_0 + tx_1$: 线性插值潜变量
- $v_\theta$: 待学习的速度预测网络（Wan2.2 DiT）
- $x_0$: 噪声潜变量；$x_1$: 真实 Canvas 潜变量

---

### 公式 2: [[GRPO]] 策略损失

$$
\mathcal{L}_{GRPO} = \mathbb{E} \left[ \min\!\left( r_\theta \hat{A},\ \mathrm{clip}(r_\theta,\ 0.8,\ 1.28)\ \hat{A} \right) \right]
$$

其中双截断（dual clip）对负优势额外限制系数为 3.0：

$$
r_\theta(a|s) = \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}, \quad \hat{A} = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}
$$

**含义**: 基于 PPO-clip 的在想象世界中的策略梯度目标；无评论家（Critic）、无 KL 惩罚；组内归一化奖励为优势。

**符号说明**:
- $r_\theta$: 重要性权重（当前策略 / 旧策略概率比）
- $\hat{A}$: 组归一化优势（基于 State Reward Model 的二值奖励 $R_i$）
- $\mathrm{clip}(\cdot, 0.8, 1.28)$: PPO 截断范围
- 3.0: 负优势双截断系数（来自 Appendix G）

---

### 公式 3: [[奖励函数|奖励模型二值奖励]]

$$
R_i = \mathbf{1}\!\left[\sigma\!\left(f_\phi^{\text{state}}(o_T)\right) \geq 0.7\right]
$$

**含义**: 对想象轨迹末帧，State Reward Model 输出成功概率 $\geq 0.7$ 时给予奖励 1，否则为 0；全成功或全失败组被丢弃，只保留 1-7 次成功的组参与训练。

**符号说明**:
- $f_\phi^{\text{state}}$: 冻结的 State Reward Model（三视角 RGB → 1）
- $o_T$: 想象轨迹最终帧
- $\sigma(\cdot)$: Sigmoid 函数；0.7 为成功阈值

---

## 关键图表

### Figure 8: 状态监督对策略评估的影响

（原 HTML 中无图片 src）

**说明**: State Reward Model 监督将任务级 Pearson 相关从 0.602 提升至 0.885，$R^2$ 从 0.413 到 0.791，在 Test Tube 和 Blocks 任务上提升最显著，Rubik's Cube 提升最小。

---

### Table 2: RGB 预测质量对比

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | EEF 误差 ↓ |
|------|--------|--------|---------|-------|-------|-----------|
| EVAC | — | — | — | — | — | — |
| Open-Sora | — | — | — | — | — | — |
| WoVR | — | — | — | — | — | — |
| **PhysCanvas (Ours)** | **21.172** | **0.735** | **0.121** | **21.459** | **116.274** | **0.685 px** |

**说明**: PhysCanvas 在所有 RGB 预测指标上超越基线，末端执行器跟踪误差比次优基线低约 51.9%。

---

### Table 3: 消融实验（动作注入 + 一致性监督）

| 配置 | 说明 | 效果 |
|------|------|------|
| M0: AdaLN 注入 | 基础动作条件 | EEF 误差较高 |
| M1: 空间动作图像注入 | 正向运动学 + 相机投影 | EEF 误差降低 32.6% |
| M1 + Scene 监督 | 加入场景保留奖励 | LPIPS、FVD 下降 |
| M1 + Scene + State 监督 | 加入任务状态奖励 | FID、FVD 进一步下降 |
| M4: 全模型（+ Modality） | 三奖励模型全部 | 接触 MAE、标记位移误差最低 |

**关键发现**: 空间动作注入是精确动作条件化的关键；三奖励模型共同作用达到最佳视触觉一致性。

---

### Table 4: 策略改进（真实机器人）

| 策略 | 方法 | 平均成功率 |
|------|------|-----------|
| π0.5 | SFT | 59.1% |
| π0.5 | RL in WM (M1) | 67.5% |
| π0.5 | **RL in WM+RM (M4)** | **70.8%** |
| FM | SFT | 11.2% |
| FM | RL in WM+RM (M4) | 19.1% |

**说明**: 世界模型内 RL 将 [[π₀.₅]] 真实成功率提升 11.7 个百分点；FM 提升 7.9 个百分点。Test Tube 和 Rubik's Cube 对 FM 仍为 0%。

---

### Table 7: GRPO 超参数配置

| 参数 | 值 | 参数 | 值 |
|------|----|------|----|
| 组大小 | 8 条轨迹 | Actor global batch | 128 |
| Rollout epochs/update | 4 | 轨迹/update | 512 |
| 学习率 | 5×10⁻⁶ | Adam (β1, β2) | (0.9, 0.95) |
| 梯度截断 | 1.0 | Ratio clip | [0.8, 1.28] |
| Dual clip | 3.0 | KL/Entropy 系数 | 0.0/0.0 |
| 奖励阈值 | 0.7 | 计算节点 | 2×8 GPU |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人操作轨迹 | 3,805 条训练轨迹；每任务 80 条测试 | ARX AC One 双臂 + 4 个 Xense 触觉传感器 | 世界模型训练 |
| 奖励模型标注 | 真实轨迹人工标注 | 帧级任务完成 + 场景漂移 + 模态不一致 | 三奖励模型训练 |

**四个任务**:
1. **试管放置（Test Tube Placement）**: 精密接触，要求位姿精准
2. **瓶盖拧开（Bottle-Cap Unscrewing）**: 旋转操作，触觉信号关键
3. **方块搬运（Block Relocation）**: 相对简单，作为基准
4. **魔方操作（Rubik's Cube Manipulation）**: 最复杂，接触多变

### 实现细节

- **骨干**: [[Wan2.2-TI2V-5B]]（Wan 2.2 TI2V-5B，5B 参数扩散 Transformer）
- **奖励模型**: [[ResNet-18]]（ImageNet 预训练）+ MLP / [[Transformer解码器|Transformer 解码器]]
- **硬件**: 64 块加速器（世界模型）；2 节点 × 8 GPU（GRPO）
- **训练**: ~3 天（世界模型）；100 epochs（奖励模型，AdamW）
- **采样频率**: 10Hz；FM 32 步积分，[[π₀.₅]] 10 步积分

### 硬件平台

![Figure 9: 硬件布局](https://arxiv.org/html/2610.28613v1/hardware_layout.png)

**说明**: ARX AC One 双臂机器人平台，三路相机（左手腕 + 右手腕 + 头顶），四个 Xense 光学触觉传感器。

![Figure 10: Xense 触觉传感器](https://arxiv.org/html/2610.28613v1/assets/hardware/sensor_front_cutout.png)

**说明**: Xense 光学触觉传感器的前视（a）、后视（b）和安装在夹爪上的状态（c）。

![Figure 11: 四任务轨迹](https://arxiv.org/html/2610.28613v1/task_cases_rollout.png)

**说明**: 四个双臂任务的轨迹展示，包含配对的触觉观测示例。

![Figure 14: 触觉数据统计](https://arxiv.org/html/2610.28613v1/tactile_statistics_rollout.png)

**说明**: 各任务触觉数据的百分位-箱线图分布，展示不同任务的接触模式差异。

### 可视化结果

![Figure 18: 基线定性对比](https://arxiv.org/html/2610.28613v1/baseline_stat_comparison.png)

**说明**: 四任务上 PhysCanvas 与 Open-Sora、WoVR 等基线的定性对比，PhysCanvas 更好地保持了小物体和场景布局的一致性。

![Figure 19: 逐层多模态注意力分析（Bottle case 01）](https://arxiv.org/html/2610.28613v1/assets/analysis/bottle_case01_attention.png)

**说明**: [[t-SNE|逐层注意力]] 分析（layer 0/14/29）。Layer 0 跨模态注意力占比高（RGB 0.39, 触觉 0.45, 标记 0.29），Layer 14/29 以自模态注意力为主（对角值 0.86/0.66/0.84），表明**跨模态融合集中在浅层，深层专注自模态精化**。

---

## 批判性思考

### 优点

1. **极简模态融合**: "画布"设计无需额外网络，直接复用大规模预训练视频模型，优雅解决多模态异构问题。
2. **语义监督可扩展**: 冻结奖励模型的梯度监督框架模块化，奖励模型可独立训练和替换。
3. **完整闭环验证**: 从世界模型预测精度到策略评估相关性再到真实机器人改进，验证链路完整。

### 局限性

1. **低层重建指标混合**: 论文本身承认低层重建指标（PSNR 等）提升有限，说明画质增益来自语义而非感知细节，可能限制视觉保真度要求高的应用。
2. **部分任务提升不显著**: FM 策略在 Test Tube 和 Rubik's Cube 上仍为 0%，说明世界模型对策略基础能力不足时无法弥补；[[π₀.₅]] 在 Rubik's Cube 上改进也最小。
3. **奖励模型依赖人工标注**: 三个奖励模型需要帧级人工标注（任务完成、场景漂移、模态不一致），标注成本高，难以快速迁移到新任务。
4. **单平台验证**: 目前仅在 ARX AC One 一款双臂平台上验证，跨平台泛化能力未知。

### 潜在改进方向

1. 自动奖励标注（利用 VLM 替代人工标注）以降低新任务迁移成本
2. 扩展到更多异构传感器（力矩、视觉事件相机）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（Appendix 详细描述）
- [x] 数据集规模已公布

---

## 关联笔记

### 基于
- [[Wan2.2-TI2V-5B]]: 世界模型骨干（5B 参数视频扩散 Transformer）
- [[流匹配]]: 训练目标（Flow Matching）
- [[GRPO]]: 策略在线强化学习算法

### 对比
- [[Open-Sora]]: RGB 视频预测基线
- [[OmniVTA]]: 先前视触觉世界模型
- [[FeelWorld]]: 先前视触觉世界模型

### 方法相关
- [[π₀.₅]]: 受益策略之一，成功率从 59.1% 升至 70.8%
- [[VAE]]: 复用预训练视频 VAE 编码多模态
- [[ResNet-18]]: 奖励模型骨干
- [[触觉传感器]]: 核心传感器（Xense 光学触觉）
- [[前向动力学模型]]: 动作图像渲染所用
- [[动作分块]]: 策略动作表示
- [[PPO]]: GRPO 依赖的 PPO-clip 机制

### 硬件/数据相关
- [[Dual-Arm Robot]]: ARX AC One 双臂平台

---

## 速查卡片

> [!summary] PhysCanvas (arXiv 2610.28613)
> - **核心**: 统一视觉画布融合 RGB + 触觉 + 标记 + 状态，加三奖励模型一致性后训练
> - **方法**: Wan2.2-TI2V-5B 骨干 + 空间动作图像注入 + GRPO 世界模型内 RL
> - **结果**: Pearson 相关 0.602→0.885；π0.5 成功率 59.1%→70.8%
> - **代码**: 未公开

---

*笔记创建时间: 2026-10-22*

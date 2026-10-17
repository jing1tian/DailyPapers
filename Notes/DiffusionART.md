---
title: "DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models"
method_name: "DiffusionART"
authors: [Zhenglin Wan, Jiayu Chen, Zhengfeiyang Zhang, Pengfei Zhou, Wangbo Zhao, Fangyuan Kong, Pengfei Wan, Kun Gai, Yang You]
year: 2026
venue: arXiv
tags: [diffusion-rl, reward-fine-tuning, flow-matching, video-generation, reinforcement-learning, policy-optimization]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2610.12572
created: 2026-10-17
---

# 论文笔记：DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NUS HPC-AI Lab, Kling AI, HKUST |
| 日期 | October 2026 |
| 项目主页 | https://github.com/vanzll/DiffusionART |
| 对比基线 | [[DiffusionNFT]], [[FlowGRPO]], [[Flash-GRPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.12572) / [Code](https://github.com/vanzll/DiffusionART) |

---

## 一句话总结

> DiffusionART 通过聚合奖励引导方向消除 NFT 训练中的隐式一致性力，并用信任域掩码的双更新代替 EMA 实现自适应复原，在 HunyuanVideo-13B 上以 12× 更少 GPU 时间达到相同奖励水平。

---

## 核心贡献

1. **诊断 NFT 不稳定根源**：证明正负联合训练只能部分抵消前向-反向一致性力（forward-reverse consistency force），而 EMA 的真正作用是提供"复原力"而非离策略数据
2. **聚合优先奖励引导**：按 prompt 聚合奖励协方差得到统一引导方向 $G(c)$，彻底消除一致性力，且无需奖励可微或差分奖励模型
3. **信任域掩码双更新**：通过两个不相交 mini-batch 的分步更新实现漂移自适应复原，无需调节 EMA 超参数

---

## 问题背景

### 要解决的问题

[[扩散模型]]和[[流匹配]]模型通常通过噪声/速度目标回归训练，无法直接优化下游黑盒奖励（如视觉质量、文本对齐）。在线 RL 可利用黑盒奖励但存在信用分配问题（奖励仅在完整生成后才可得）。

### 现有方法的局限

- **反向过程 RL**（[[FlowGRPO]]、DanceGRPO）：在去噪轨迹上训练，需要 ODE→SDE 转换，单步信用分配弱
- **前向过程 RL**（[[DiffusionNFT]]）：对已生成样本加噪后回归速度，效率高但不稳定，需精心调节每任务 EMA 参数
- 根本问题：NFT 的正负联合训练存在一个隐式"一致性力"无法在每个状态层面消除，而 EMA 虽能稳定训练但需任务特定调节

### 本文的动机

作者识别出 NFT 不稳定的两个根本原因：(1) 一致性力未被完全消除；(2) 复原力（restoration）才是 EMA 的真正作用机制。DiffusionART 在设计上完全消除一致性力，同时通过双更新实现无超参的自适应复原。

---

## 方法详解

### 背景：DiffusionNFT 回顾

[[DiffusionNFT]] 采用前向过程 RL：对每个 prompt $c$，从冻结 rollout 策略 $p_{old}$ 采样 $K$ 个端点 $x_0$，计算奖励权重，用正负样本加权 MSE 训练。

[[整流流]]定义插值 $x_t = \alpha_t x_0 + \sigma_t \varepsilon$，速度目标 $u_t = x_0 - \varepsilon$。NFT 在每次更新后用 [[指数移动平均]] 更新 rollout 策略：$\theta_{old} \leftarrow \eta\theta_{old} + (1-\eta)\theta$。

### 核心分析（Sec 3）

#### 3.1 一致性力诊断

**Lemma 1**：理想 NFT 引导等于条件奖励协方差 $g = \text{Cov}[r, u_t | s_t]$。正负分支在同一方向上仅步长不同。

设前向-反向速度差 $e = v_{fwd}^{old} - v_{rvs}^{old}$，则：

**Proposition 1**：NFT 引导力（在 $v_\theta = v_{rvs}^{old}$ 处）为：

$$
\mathcal{F}_{NFT}(s_t) = 2g + (2q-1)e
$$

其中 $q = \mathbb{E}[r | s_t]$ 是条件均值奖励，$g = \text{Cov}(r, u_t | s_t)$。

**符号说明**：
- $g$：奖励与目标速度的条件协方差（期望引导方向）
- $q$：状态 $s_t$ 处的条件均值奖励权重
- $e = v_{fwd}^{old} - v_{rvs}^{old}$：前向与反向速度的差（一致性力来源）
- $(2q-1)e$：残余一致性力——仅当 $q = 1/2$ 时为零，但这只在均值上成立，不在每个状态成立

**Insight 1**：联合训练只能在统计均值上消除一致性力，无法在每个状态上消除。

#### 3.2 EMA 与复原力

**Proposition 2**：NFT 损失可分解为：

$$
\mathcal{L}_{NFT} = \mathbb{E}_{\mathcal{D}_{old}}\left[-2\beta(2r-1)\langle d, h \rangle + \beta^2 \|d\|^2\right] + \text{const}
$$

**符号说明**：
- $d = v_\theta - v_{old}$：当前模型与 rollout 策略的速度差（漂移量）
- $h = u_t - v_{old}$：目标速度与 rollout 策略的差
- $\beta$：NFT 正负分支步长系数
- $-2\beta(2r-1)\langle d, h \rangle$：线性奖励引导项
- $\beta^2\|d\|^2$：二次复原项（将 $v_\theta$ 拉回 $v_{old}$）

消融实验发现：EMA 策略 rollout（无旧策略锚点）仍然崩溃；两次更新无 EMA 可避免崩溃但慢 15.2×；第二次更新时复原力的输出力 RMS 比引导力小 9.4×，但参数梯度范数大 4.9×，因此主导聚合更新。

**Insight 2**：复原力是稳定训练的关键机制，但 EMA 参数需要任务特定调节（如 OCR 任务的默认 EMA 设置不稳定，迁移到 FLUX 也不稳定）。

### 4. DiffusionART 方法

#### 4.1 聚合优先引导（Aggregation-First Guidance）

**Lemma 2**（奖励倾斜的协方差解释）：在 $\tau = 0$ 处，倾斜分布端点均值对 $\tau$ 的导数等于奖励-端点协方差：

$$
\frac{\partial}{\partial \tau}\mathbb{E}_{p_\tau}[x_0 | c]\bigg|_{\tau=0} = \text{Cov}_{p_{old}}(R, x_0 | c)
$$

**含义**：奖励倾斜（reward tilting）在端点空间中的一阶效应就是奖励-端点协方差，直接给出"向更高奖励方向移动"的方向。

**协方差估计与归一化**：

$$
\overline{\text{Cov}}(R, x_0 | c) = \frac{1}{K}\sum_{i=1}^{K}(R_i - \bar{R}_c)(x_0^{(i)} - \bar{x}_{0,c})
$$

$$
G(c) = \frac{\overline{\text{Cov}}(R, x_0 | c)}{\text{RMS}(\overline{\text{Cov}}(R, x_0 | c))}
$$

**符号说明**：
- $K$：每个 prompt 的采样数
- $R_i$：第 $i$ 个样本的奖励值
- $\bar{R}_c$：prompt $c$ 的均值奖励
- $x_0^{(i)}$：第 $i$ 个生成端点
- $\bar{x}_{0,c}$：端点均值
- $\text{RMS}(\cdot)$：均方根归一化，消除奖励尺度差异

**冻结训练目标**：在 prompt $c$ 的所有反向轨迹状态上使用同一冻结目标：

$$
v_{tgt}^{ART}(s_t) = v_\theta^{(0)}(s_t) + G(c)
$$

**含义**：目标速度在 rollout 策略基础上加上 per-prompt 奖励引导方向。由于 $G(c)$ 不含前向速度项，一致性力完全不存在。

#### 4.2 双更新自适应复原（Two-Update Restoration）

将 rollout batch 分为两个不相交 mini-batch，各做一次 MSE 更新：

- 第一次更新：梯度正比于 $G(c)$（纯引导）
- 第二次更新：梯度正比于 $G(c) - d(s_t)$，其中 $d(s_t) = v_\theta - v_\theta^{(0)}$ 是第一次更新引入的漂移

$$
\nabla \mathcal{L}_{2nd} \propto G(c) - d(s_t)
$$

**含义**：复原力随漂移大小自适应增长——漂移大时复原力强，漂移小时复原力弱，无需 EMA 超参调节。

#### 4.3 信任域掩码（Trust-Region Masking）

ODE rollout 是确定性的，因此用 Gaussian 参考转移（沿用 Flow-GRPO 的 ODE→SDE 转换）定义 KL 代理：

$$
m(s_t) = \frac{\|\mu(s_t, v_\theta) - \mu(s_t, v_\theta^{(0)})\|^2}{2D\sigma_{ref}^2(t)}
$$

per-state 权重为：

$$
w(s_t) = t \cdot \left[1 - \mathbf{1}\{m \geq \delta\} \cdot \mathbf{1}\{\langle d, G - d\rangle > 0\}\right]
$$

**符号说明**：
- $\mu(s_t, v)$：从状态 $s_t$ 出发、速度预测为 $v$ 时的参考均值转移
- $D$：状态空间维度
- $\sigma_{ref}^2(t)$：参考转移方差（$\nu = 0.8$，$\delta = 10^{-5}$）
- $t$：时间步权重（强调高噪声阶段）
- 掩码逻辑：当漂移超出信任域 AND 方向为外扩时屏蔽该状态，允许方向内收的修正

灵感来自 [[PPO]] 的裁剪机制和 Flow-DPPO。参数 $\nu = 0.8$，$\delta = 10^{-5}$ 无需任务特定调节。

#### 4.4 DiffusionART-Pairwise（成对奖励变体）

将 $K$ 个端点分为 $K/2$ 对，每对分配胜/负标签 $(y_i \in \{0, 1/2, 1\})$，用标签替代标量奖励计算协方差：

$$
\overline{\text{Cov}} = \frac{1}{4M}\sum_{j \text{ non-tied}} (x_{0,j}^{win} - x_{0,j}^{lose})
$$

**含义**：平局对贡献零引导，胜者-负者差直接作为引导方向，支持 pairwise 奖励模型（无需绝对奖励分数）。

---

## 关键图表

### Figure 1: Training Curves + Visual Results / 训练曲线与视觉结果

![Figure 1](https://arxiv.org/html/2610.12572v1/diffusionart_cover_base_model_20261008.png)

**说明**：左图展示 HunyuanVideo-13B 上均等权重设置（VQ:MQ=1:1）的训练曲线，DiffusionART 比 DiffusionNFT 和 Flow-GRPO-Fast 分别快 12.02× 和 11.72× 达到相同奖励水平。右侧为同 prompt、同种子下微调前后的视频帧对比。

### Figure 2: 正负分支消融与一致性力分析

**说明**：(a) 对正负目标分支的消融；(b) 展示联合训练如何只能部分消除一致性力——在状态级别 $(2q-1)e$ 项无法被完全抵消。

### Figure 3: EMA 与复原力消融

![Figure 3](https://arxiv.org/html/2610.12572v1/fig3_review_20261008_v2.png)

**说明**：(a) On-policy NFT 使用两个 micro-batch 比单 micro-batch 慢 15.2×；(b) EMA 策略 rollout（无旧策略锚点）仍然崩溃，说明离策略数据不是稳定因素；(c) 第二次更新中复原力 vs. 引导力的 RMS 与参数梯度范数对比；(d) 第二次更新的分解：复原力主导聚合方向。

### Figure 4: 聚合优先引导示意图

**说明**：对比 NFT 的正负分支在不同状态产生相互冲突的力（一致性力残余），与 DiffusionART 将 $G(c)$ 直接施加于反向速度状态的方式。DiffusionART 的目标无前向速度项，彻底消除一致性力。

### Figure 5: SD3.5-M 单/多奖励训练曲线

**说明**：4 个奖励（GenEval、PickScore、CLIPScore、HPSv2.1）联合优化训练曲线，DiffusionART 在所有设置下均稳定上升，相比 Flow-GRPO 在 GenEval/OCR/PickScore 任务上分别以 10.0×/5.2×/9.6× 更少步数达到相同奖励。

### Figure 6: FLUX.1-dev GenEval + OCR 复原消融

![Figure 6](https://arxiv.org/html/2610.12572v1/arxiv_t2i_flux_restoration_fonts_20261009.png)

**说明**：(a) FLUX.1-dev 上 GenEval 训练曲线，DiffusionART eval reward 达 0.9684 vs. Flow-GRPO-Fast 0.8374；(b) OCR 任务上有/无第二次复原更新的消融——1 次更新快速学习后退化，2 次更新稳定；NFT 2 次更新先短暂崩溃后恢复。

### Figure 7: 视频生成（T2V + I2V）训练曲线

**说明**：HunyuanVideo-13B T2V（TA:VQ:MQ=7:2:1）和 Wan2.1-I2V-14B I2V（VQ:MQ=1:1）的训练曲线，包含 VQ、MQ、TA 三个指标。DiffusionART 在所有指标上均优于基线。

### Figure 8: T2V 信任域掩码消融

**说明**：VQ:MQ=1:1 设置下，无掩码版本在约 step 200 出现下降再恢复，有掩码版本持续稳定提升。诊断显示有掩码时前 300 步最大第一次更新梯度范数为 1.96，无掩码为 3.33。

### Figure 9: SD3.5-M 联合多奖励优化（500 步）

**说明**：在 SD3.5-M 上联合优化 GenEval、CLIPScore、PickScore、HPSv2.1 四个奖励，500 步训练曲线展示 DiffusionART 的稳定多目标优化能力。

### Figure 10: T2V VQ:MQ=1:1 训练曲线（步数 & GPU 时间）

**说明**：左图按步数，右图按 GPU 小时展示 VQ 和 MQ 训练曲线。DiffusionART 在 VQ 6 时仅需 20 步（44.2 GPU 时），DiffusionNFT 需 50 步，Flow-GRPO-Fast 需 427 步。

### Figure 11: DiffusionART vs. ReFL 对比

**说明**：ReFL 使用可微分奖励模型梯度（K=1），DiffusionART 使用标量奖励值（K=8）。在等权重下 DiffusionART 在更少优化器更新内同时提升两个奖励。

### Figure 12: 图像生成信任域消融

**说明**：在 SD3.5-M 和 FLUX.1-dev 上的信任域掩码消融实验，验证掩码对训练稳定性的贡献。

### Figure 13: HunyuanVideo 信任域诊断

**说明**：训练过程中的梯度范数和速度漂移诊断图，展示信任域掩码如何控制梯度峰值（有掩码 1.96 vs. 无掩码 3.33）。

### Figure 14: GenEval 引导分配方式对比

**说明**：在 GenEval 上对比 prompt 全局引导（24 张图形成 12 对）vs. 成对局部引导的效果。Step 400 时全局引导达到 0.91 held-out GenEval，局部引导仅 0.78。

### Figure 15: T2V 生成质量对比（DiffusionART 前后）

![Figure 15](https://arxiv.org/html/2610.12572v1/diffusionart_t2v_selected24_p1_base_model_20261009.png)

**说明**：T2V 生成质量对比，24 个精选样本，同 prompt 同种子下微调前后各 4 帧。展示 DiffusionART 在视觉质量和运动质量上的提升。

### Figure 16: T2I 多奖励优化结果（页面 1）

![Figure 16](https://arxiv.org/html/2610.12572v1/diffusionart_t2i_selected16_p1_base_model_20261009.png)

**说明**：SD3.5-M 在 GenEval、CLIPScore、PickScore、HPSv2.1 联合优化后的文生图结果，16 个精选样本。

### Figure 17: T2I 多奖励优化结果（页面 2）

![Figure 17](https://arxiv.org/html/2610.12572v1/diffusionart_t2i_selected16_p2_base_model_20261009.png)

**说明**：T2I 附加对比，上两行为多奖励优化，下两行为 PickScore 专项优化。

### Table 1: 成对反馈与组大小消融（T2V VQ, step 400）

| 方法 | GPU 小时 ↓ | VQ ↑ |
|------|-----------|------|
| 标量奖励, K=8 | 891.1 | 13.637 |
| Pairwise, K=2 | 260.8 | 12.235 |
| Pairwise, K=4 | 465.6 | 10.883 |
| Pairwise, K=8 | 883.3 | 12.092 |

**说明**：K=2 pairwise 仅需标量 K=8 的 3.4× 更少 GPU 时间，最终 VQ 无明显劣化，是实际应用中高效选择。

### Table 2: FLUX.1-dev GenEval 评估奖励对比

| 方法 | Eval Reward |
|------|------------|
| Flow-GRPO-Fast | 0.8374 |
| Flow-DPPO | 0.9183 |
| DiffusionNFT (EMA) | 0.8490 |
| **DiffusionART** | **0.9684** |

**说明**：DiffusionART 在 FLUX.1-dev 上显著超越所有基线，比最强竞争者 Flow-DPPO 高 5.5%。

### Table 3: T2V 主实验结果（TA:VQ:MQ = 7:2:1，Step 400）

| 方法 | VQ ↑ | MQ ↑ | TA ↑ |
|------|------|------|------|
| 预训练基线 | 5.195 | 1.943 | 1.118 |
| Flow-GRPO-Fast | 3.149 | 1.874 | 1.667 |
| DanceGRPO | 3.922 | 1.206 | 1.685 |
| DiffusionNFT (EMA) | 6.827 | 2.683 | 2.096 |
| DiffusionNFT (On-policy) | 5.543 | 1.839 | −4.480 |
| **DiffusionART** | **8.970** | **4.224** | **2.111** |

**说明**：DiffusionART 相比最强基线 DiffusionNFT (EMA) 提升 VQ 31.4%、MQ 57.4%。On-policy DiffusionNFT 的 TA 指标崩溃（−4.480），证实不稳定问题的严重性。

### Table 4: I2V 实验结果（VQ:MQ = 1:1，Wan2.1-I2V-14B）

| 方法 | VQ ↑ | MQ ↑ |
|------|------|------|
| Flow-GRPO-Fast | 5.771 | 4.710 |
| DanceGRPO | 5.123 | 3.266 |
| DiffusionNFT (EMA) | 5.516 | 4.161 |
| **DiffusionART** | **11.196** | **10.476** |

**说明**：I2V 任务上 DiffusionART 大幅领先，DiffusionNFT 早期平台期被归因于残余一致性力的影响。

---

## 实验

### 数据集与任务

| 任务 | 模型 | 奖励 | 分辨率 |
|------|------|------|--------|
| 文生图（T2I）| SD3.5-M, FLUX.1-dev | GenEval, OCR, PickScore, CLIPScore, HPSv2.1 | 512×512 |
| 文生视频（T2V）| HunyuanVideo-13B | VideoReward (VQ, MQ), TA Score | 480×480, 53 帧, 8FPS |
| 图生视频（I2V）| Wan2.1-I2V-14B | VideoReward (VQ, MQ) | 400×640, 53 帧, 8FPS |

### 实现细节

- **图像 LoRA**：rank 32，$\alpha = 64$，学习率 $3 \times 10^{-4}$，10 步 rollout，无 CFG，无 KL 正则
- **视频全量微调**：学习率 $10^{-5}$，32 prompts × 8 videos，16 步采样
- **I2V**：4 步采样，FLUX.1-dev 参考图
- **超参**：$\nu = 0.8$，$\delta = 10^{-5}$，无任务特定调节

### 效率总结（VQ:MQ=1:1）

| 里程碑 | DiffusionART 步数 | GPU 小时 | 基线对比 |
|--------|-----------------|---------|---------|
| VQ 达到 6 | 20 步 | 44.2h | DiffusionNFT 50步，Flow-GRPO-Fast 427步 |
| MQ 达到 3 | 19 步 | 42.9h | — |
| 晚期均值奖励达到 DiffusionNFT 水平 | — | 63.4h | DiffusionNFT 需 1401.4h（22.1×） |

---

## 批判性思考

### 优点
1. **理论分析深入**：对 NFT 不稳定性的诊断（一致性力 + 复原机制）提供了清晰的数学解释，有助于理解现有方法的失败模式
2. **实用性强**：无需可微分奖励模型、无需 EMA 超参调节，支持 pairwise 反馈，大幅降低实际部署门槛
3. **规模化验证**：在 HunyuanVideo-13B（视频）和 Wan2.1-I2V-14B（图生视频）这两个大型模型上验证，具有实用价值

### 局限性
1. **组大小敏感**：K=2 pairwise 虽省资源，但在某些任务上比 K=8 低；最优 K 仍需权衡
2. **ODE-to-SDE 转换依赖**：信任域掩码仍使用 Flow-GRPO 的 ODE→SDE 转换建立参考分布，与 SDE 转换的近似误差有关
3. **超参迁移性**：虽然不需 EMA 调节，但 $\nu$ 和 $\delta$ 在所有任务上设为固定值，边界情况的鲁棒性未完全验证

### 潜在改进方向
1. 将 DiffusionART 框架扩展到 DiT 架构的图像编辑任务
2. 探索更大 K 在多样化 prompt 下的分布估计质量
3. 与 RLHF 成对标注数据集结合，用于视频生成的人类偏好对齐

### 可复现性评估
- [x] 代码开源（GitHub: vanzll/DiffusionART）
- [ ] 预训练权重（未提及）
- [x] 训练细节完整（LoRA rank、学习率、采样步数均有说明）
- [x] 数据集可获取（使用 HunyuanVideo、SD3.5-M、FLUX、Wan2.1 等公开模型）

---

## 关联笔记

### 基于
- [[DiffusionNFT]]: NFT-style 前向过程 RL，DiffusionART 直接在其基础上分析和改进
- [[FlowGRPO]]: 反向过程 RL 基线，代表信用分配弱的一类方法
- [[整流流]]: 基础生成框架（Rectified Flow）
- [[流匹配]]: 更一般的连续流生成框架

### 对比
- [[Flash-GRPO]]: 同为 Flow-GRPO 加速变体（CPS），但用单步随机策略而非 NFT 前向过程
- [[DiffusionNFT]]: 最近的前向过程 RL 方法，DiffusionART 诊断其不稳定根源
- [[PPO]]: 信任域掩码的灵感来源

### 方法相关
- [[指数移动平均]]: DiffusionART 通过双更新机制取代 EMA
- [[Trust Region]]: 信任域掩码的理论基础
- [[HunyuanVideo]]: 主要实验平台之一

### 硬件/数据相关
- [[HunyuanVideo]]: 13B 参数视频生成模型，主要测试平台

---

## 速查卡片

> [!summary] DiffusionART
> - **核心**: 通过聚合奖励协方差消除 NFT 一致性力，双更新实现自适应复原
> - **方法**: 按 prompt 聚合引导方向 $G(c) = \text{Cov}(R, x_0|c)/\text{RMS}$，信任域掩码双 mini-batch 更新
> - **结果**: HunyuanVideo-13B 上以 12× 更少 GPU 时间达到 DiffusionNFT 相同奖励，T2V VQ +31.4%、MQ +57.4%
> - **代码**: https://github.com/vanzll/DiffusionART

---

*笔记创建时间: 2026-10-17*

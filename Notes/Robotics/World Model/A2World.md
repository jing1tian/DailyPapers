---
title: "Learning Transferable Dynamics Priors from Action to World Modeling"
method_name: "A2World"
authors: [Ze Huang, Jiahui Zhang, Hairuo Liu, Chenxi Zhang, Ran Cheng, Li Zhang]
year: 2026
venue: ECCV 2026
tags: [world-model, robot-manipulation, diffusion-model, action-conditioned, multi-view, policy-learning]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.29501
created: 2026-07-01
---

# 论文笔记：Learning Transferable Dynamics Priors from Action to World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University; Shanghai Innovation Institute; Shanghai Jiao Tong University; McGill University |
| 日期 | June 2026 |
| 项目主页 | https://github.com/LogosRoboticsGroup/A2World |
| 对比基线 | [[Cosmos-Predict2]], [[Prophet]], [[π0]], [[GE-Act]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.29501) / [Code](https://github.com/LogosRoboticsGroup/A2World) |

---

## 一句话总结

> A2World 是一个基于 2.1M+ 机器人轨迹预训练的动作条件扩散世界模型，通过动作驱动的视觉预测学习可迁移动力学先验，并统一适配为长程仿真器（A2World-sim）和策略模型（A2World-policy）。

---

## 核心贡献

1. **Action-to-Video 预训练范式**: 用真实机器人动作标注监督大规模世界模型预训练，学习跨具身形态可迁移的交互动力学先验，而非仅捕获外观级视频生成能力
2. **双用途统一架构**: 同一预训练 A2World 分叉为 A2World-sim（长程自回归仿真器）和 A2World-policy（指令条件机器人策略），实现仿真与策略的统一基础
3. **关键技术创新**: 姿态引导历史帧采样（Pose-guided History Sampling）、跨视角一致性注意力、MoE 式视频-动作联合扩散块（MoE-like Video-Action Blocks）

---

## 问题背景

### 要解决的问题

现有机器人学习方法中，世界模型通常只用于单一目的（仿真或策略生成），缺乏能同时服务两者的统一动力学表示。而基于文本条件的视频生成预训练忽视了动作信息，无法准确建模机器人与环境的交互因果关系。

### 现有方法的局限

- 文本条件世界模型（如 Cosmos、VideoPrediction 系列）：忽略动作作为因果信号，生成与指令相关但与实际执行动作不一致的视觉序列
- 纯策略模型（如 π0、π0.5）：不显式建模视觉变化，对长程任务和分布外场景泛化能力弱
- 独立仿真器：通常针对特定具身形态训练，难以跨平台迁移

### 本文的动机

动作是视觉场景变化的因果驱动力。以动作为条件的视觉预测迫使模型学习"动作如何改变世界"的物理知识，这种交互动力学先验可迁移至仿真（用历史帧预测未来状态）和策略学习（将视频-动作联合建模）两个下游任务。

---

## 方法详解

### 模型架构

A2World 系列采用 **[[Diffusion Transformer (DiT)|DiT]] 扩散架构**（2.5B 参数），从 [[Cosmos]]-Predict2-2B-Video2World 权重初始化，并以 2.1M+ 机器人操作轨迹进行动作条件预训练：

- **输入**: 当前观测帧 $o_t$ + 动作序列 $a_{t+1:t+k}$（多视角打包至时间维度）
- **Backbone**: [[Diffusion Transformer (DiT)|DiT]] 基础架构，支持跨视角注意力（Cross-view Attention）
- **核心模块**: [[Action Conditioning|动作条件化]] 注入扩散时间步嵌入；[[Cross-view Attention|跨视角注意力]] 保证多相机空间一致性
- **输出**: 预测未来 $k$ 步的视觉帧序列 $o_{t+1:t+k}$
- **总参数**: 2.5B

### 核心模块

#### 模块 1: A2World 基础世界模型

**设计动机**: 利用 [[Action Conditioning|动作条件化]] 建立视觉预测的因果结构，使模型学习动作驱动的场景变化规律。

**具体实现**:
- 动作序列经 MLP 编码为嵌入向量，注入至 [[Diffusion Model|扩散模型]] 的 timestep 嵌入中
- 多视角图像沿时间维度拼接，加入可学习视角嵌入（view embeddings）区分不同相机
- Cross-view Attention 模块在空间上强制跨视角一致性，保证多相机输出在三维空间中相互兼容

**建模目标**:

$$
\text{A2World}: p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k})
$$

#### 模块 2: A2World-sim — 长程仿真器

**设计动机**: 利用 [[Autoregressive Generation|自回归生成]] 实现长时域策略评估，通过历史帧感知避免误差累积导致的生成漂移。

**具体实现**:

- **姿态引导历史帧采样（Pose-guided History Sampling）**: 根据相对动作计算加权弧长，选取覆盖运动轨迹的关键历史帧，融合平移与旋转分量：

  - 平移距离：$d_t^{trans} = \|\Delta p_t\|_2$
  - 旋转距离：$d_t^{rot} = \|\Delta r_t\|_{\angle}$（旋转角度）
  - 加权步距：$d_t = w_t \cdot d_t^{trans} + w_r \cdot d_t^{rot}$（$w_t=1.0, w_r=0.3$）
  - 沿累积弧长 $\{d_1, d_1+d_2, ...\}$ 均匀采样 $T_h$ 帧

- **历史注入双通道（Dual-pathway History Injection）**:
  - 与历史 token 序列做 Cross-Attention
  - 将历史帧全局记忆拼接到 Self-Attention 中

- **Self-forcing 训练**: 随机将历史帧替换为模型自身生成的帧，让模型在训练期即暴露于自身预测误差，提升鲁棒性

**建模目标**:

$$
\text{A2World-sim}: p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}, \mathcal{H}_{t-1})
$$

#### 模块 3: A2World-policy — 视频-动作联合策略

**设计动机**: 利用 [[Joint Diffusion|联合扩散]] 同时生成视觉预测和动作序列，使世界模型知识直接辅助动作生成。

**具体实现**:

- **MoE 式模态块（MoE-like Blocks）**: 视频和动作共享 Self-Attention 层，但各自拥有独立的 [[AdaLN]] 归一化层和 MLP 分支，实现模态感知的特征处理
- **独立噪声调度（Independent Noise Schedules）**: 每个模态独立设置噪声尺度
  - $\sigma_v = \alpha_v \cdot \sigma_{\text{base}}$（视频，$\alpha_v = \sqrt{6}$）
  - $\sigma_a = \alpha_a \cdot \sigma_{\text{base}}$（动作，$\alpha_a = 0.5$）
- **Per-modality [[Classifier-Free Guidance (CFG)|分类器自由引导]]**: 对每个模态分别应用 CFG

**建模目标**:

$$
\text{A2World-policy}: p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)
$$

---

## 关键公式

### 公式 1: [[Action-Conditioned World Model|A2World 基础预训练目标]]

$$
\mathcal{L}_{\text{A2World}}(\sigma) = \mathbb{E}_{\mathbf{z},\mathbf{n}}\left[\left\|\mathrm{A2World}(\mathbf{z}+\mathbf{n};\,\sigma,\,\mathbf{c}=\mathbf{0},\,a) - \mathbf{z}\right\|_2^2\right]
$$

**含义**: 在 [[EDM (Elucidated Diffusion Model)|EDM]] 框架下，以动作 $a$ 为条件，无分类器引导（$\mathbf{c}=\mathbf{0}$），让模型预测真实潜码 $\mathbf{z}$，去除噪声 $\mathbf{n}$

**符号说明**:
- $\mathbf{z}$: 真实视觉帧的潜码（latent code）
- $\mathbf{n}$: 添加的噪声，$\sigma$ 决定其强度
- $\sigma$: 扩散噪声级别
- $\mathbf{c}$: 文本/语言条件（预训练阶段置 $\mathbf{0}$）
- $a$: 动作序列条件

---

### 公式 2: [[Joint Diffusion|A2World-policy 联合训练目标]]

$$
\mathcal{L}_{\text{A2World-policy}}(\sigma_v, \sigma_a) = \mathbb{E}\left[\mathbf{w}(\sigma_v)\|\hat{\mathbf{z}}^v - \mathbf{z}^v\|_2^2 + \lambda_a \mathbf{w}(\sigma_a)\|\hat{\mathbf{z}}^a - \mathbf{z}^a\|_2^2\right]
$$

**含义**: 同时优化视频预测损失（$v$）和动作预测损失（$a$），通过权重系数 $\lambda_a$ 平衡两个模态

**符号说明**:
- $\hat{\mathbf{z}}^v, \hat{\mathbf{z}}^a$: 模型预测的视频/动作潜码
- $\mathbf{z}^v, \mathbf{z}^a$: 真实视频/动作潜码
- $\mathbf{w}(\sigma)$: 噪声级别相关的权重函数（来自 EDM 框架）
- $\sigma_v = \alpha_v \sigma_{\text{base}}$, $\sigma_a = \alpha_a \sigma_{\text{base}}$: 各模态独立噪声尺度
- $\lambda_a$: 动作损失权重系数

---

### 公式 3: [[Classifier-Free Guidance (CFG)|Per-modality 分类器自由引导]]

$$
\hat{\mathbf{z}}_{\text{cfg}}^m = \hat{\mathbf{z}}_u^m + s_m\left(\hat{\mathbf{z}}_c^m - \hat{\mathbf{z}}_u^m\right)
$$

**含义**: 对模态 $m$（视频或动作）分别应用 CFG，通过引导尺度 $s_m$ 增强条件信号强度

**符号说明**:
- $m$: 模态标识（$v$ 视频 / $a$ 动作）
- $\hat{\mathbf{z}}_u^m$: 无条件预测（unconditional）
- $\hat{\mathbf{z}}_c^m$: 条件预测（conditional）
- $s_m$: 模态 $m$ 的引导尺度

---

### 公式 4: [[Pose-guided History Sampling|姿态引导历史帧采样]]

$$
d_t = w_t \cdot d_t^{\text{trans}} + w_r \cdot d_t^{\text{rot}}
$$

**含义**: 综合平移距离和旋转距离计算加权步距，再沿累积弧长均匀采样历史帧，使采样覆盖实际运动轨迹

**符号说明**:
- $d_t^{\text{trans}} = \|\Delta p_t\|_2$: 第 $t$ 步末端执行器平移距离
- $d_t^{\text{rot}} = \|\Delta r_t\|_{\angle}$: 第 $t$ 步旋转角度变化
- $w_t = 1.0$: 平移权重
- $w_r = 0.3$: 旋转权重
- $T_h = 20$: 历史帧数

---

## 关键图表

### Figure 1: 整体框架概览

![[A2World_fig1.png]]

**说明**: A2World 预训练流水线。以 2.1M+ 多具身形态机器人轨迹（含动作标注）训练 [[Diffusion Transformer (DiT)|DiT]] 基础世界模型，再分叉为 A2World-sim（长程仿真器）和 A2World-policy（动作策略）两条适配路径。

---

### Figure 2: 模块架构详图

![[A2World_fig2.png]]

**说明**: A2World 三个变体的详细模块设计。展示动作注入方式（MLP → timestep）、跨视角注意力结构、历史注入双通道以及视频-动作 MoE 块的具体连接关系。

---

### Figure 3: 真实机器人平台

![[A2World_fig3.png]]

**说明**: 双臂 Flexiv Rizon 4S 机器人平台，配备 Robotiq 夹爪、正面 Intel RealSense D435i 和腕部 D405 相机，用于 5 个双臂操作任务的真机实验。

---

### Figure 4: DROID 数据集多样化生成

![[A2World_fig4.png]]

**说明**: 在 DROID 数据集上，A2World 基础模型展示可操控的多物体抓取与失败模式仿真，验证动作条件精确驱动视觉变化的能力。

---

### Figure 5: RoboCoin 全自由度控制

![[A2World_fig5.png]]

**说明**: 在 RoboCoin 数据集上以未见脚本指令做全自由度控制，验证基础模型的泛化生成能力。

---

### Figure 6: OOD 泛化滚动

![[A2World_fig6.png]]

**说明**: 在 RoboMind 和 VIOLA 数据集上的分布外（OOD）生成结果，展示模型对新场景和新任务的泛化能力。

---

### Figure 7: 长程生成质量对比

![[A2World_fig7.png]]

**说明**: "Put chain in box" 任务上 A2World-sim 与基线的长程生成对比，A2World-sim 保持稳定任务进展，而对比方法出现明显漂移。

---

### Figure 8: 仿真器-真实机器人一致性

![[A2World_fig8.png]]

**说明**: A2World-sim 评估成功率与真实机器人成功率的散点图，Spearman ρ=0.916，Pearson r=0.965，R²=0.930，说明仿真器可有效替代真机评估。

---

### Figure 9: 真机执行结果

![[A2World_fig9.png]]

**说明**: A2World-policy 与 π0.5 和 LingBot-VA 在 5 个双臂操作任务（举盒、翻盒、插 RAM、拨开关、链条入盒）上的真机执行对比。

---

### Figure 10: 真机评估详细结果

![[A2World_fig10.png]]

**说明**: 5 个操作任务各阶段成功率详细数据，A2World-policy 在接触丰富的长程任务上优势最显著。

---

### Figure 11: 视频-动作联合训练耦合曲线

![[A2World_fig11.png]]

**说明**: 联合训练过程中视频一致性与动作预测质量的共同提升曲线，验证视频-动作联合扩散的正向耦合效应。

---

### Figure 12: AgiBot 精细双臂控制

![[A2World_fig12.png]]

**说明**: 在 AgiBot 平台上以不同幅度指令精确控制双臂，展示基础模型对动作幅度变化的敏感响应。

---

### Figure 13: π0.5 闭环在 A2World-sim 内的滚动

![[A2World_fig13.png]]

**说明**: 使用 [[π0.5]] 策略在 A2World-sim 中进行闭环推理，验证仿真器与外部策略模型的接口兼容性。

---

### Figure 14: OOD 动作条件生成对比

![[A2World_fig14.png]]

**说明**: 分布外动作条件下 A2World-sim 与基线的生成对比，A2World 更准确地响应未见动作模式。

---

### Figure 15: 历史采样策略消融可视化

![[A2World_fig15.png]]

**说明**: 姿态引导采样、滑动窗口采样、无历史三种策略的定性对比，姿态引导采样生成帧更一致连贯。

---

### Figure 16: 复杂接触任务补充执行结果

![[A2World_fig16.png]]

**说明**: "链条入盒"等复杂接触丰富任务的补充真机执行结果，A2World-policy 展示稳定的长程操作能力。

---

### Table 1: 预训练数据集统计

| 数据集 | 规模 | 具身形态 |
|--------|------|---------|
| AgiBot | 1.003M 轨迹 | Genie-1 |
| DROID | 76k 轨迹 | Franka |
| OPEN-X | 19k 轨迹 | 多具身形态 |
| InternData-A1 | 630k 轨迹 | Genie-1, Franka, 其他 |
| InternData-M1-Agilex | 105k 轨迹 | 双臂 Franka |
| RoboCoin | 246k 轨迹 | 15 种具身形态 |
| Galaxea | 77k 轨迹 | Galaxea R1 Lite |
| **总计** | **2.156M 轨迹** | **20+ 具身形态** |

---

### Table 2: 世界模型滚动质量（精调后）

| 方法 | LIBERO PSNR↑ | LIBERO SSIM↑ | LIBERO tSSIM↑ | Real Robot PSNR↑ |
|------|-------------|-------------|--------------|-----------------|
| Cosmos-Predict2 | 25.36 | .8792 | .7631 | 24.99 |
| Ctrl-World | 23.60 | .8632 | .7445 | 21.91 |
| Prophet | 26.12 | .8887 | .7789 | 25.55 |
| A2World-sim-T-pre | 26.18 | .8892 | .7794 | 24.64 |
| **A2World-sim** | **26.64** | **.8957** | **.7862** | **25.95** |

**关键发现**: A2World-sim 在所有指标上均超越基线，LIBERO PSNR 提升约 0.52 dB（vs Prophet），验证动作条件预训练的优势。

---

### Table 3: 仿真器-真实机器人一致性指标

| 指标 | 值 |
|------|-----|
| Spearman ρ | 0.916 |
| Pearson r | 0.965 |
| R² | 0.930 |

**关键发现**: 高达 0.965 的皮尔逊相关系数说明 A2World-sim 可作为真机评估的有效替代品。

---

### Table 4: LIBERO 域内策略性能（Table 5）

| Suite | A2World-policy 成功率 |
|-------|----------------------|
| LIBERO-Spatial | 98.2% |
| LIBERO-Object | 99.1% |
| LIBERO-Goal | 98.5% |
| LIBERO-Long | 98.6% |
| **平均** | **98.6%** |

---

### Table 5: OOD 迁移至 LIBERO-Plus Spatial（Table 6）

| 方法 | 平均成功率↑ |
|------|------------|
| Cosmos Policy | 85.6% |
| π₀ | 78.6% |
| GE-Act | 87.5% |
| **A2World-policy (A-pre)** | **88.5%** |

**关键发现**: A2World-policy 在 OOD 场景下超越所有基线，验证动作-视频联合预训练的泛化能力。

---

### Table 6: 预训练策略消融（Table 7）

| 预训练策略 | LIBERO 平均成功率 |
|-----------|-----------------|
| C-init（Cosmos 初始化）| 97.0% |
| T-pre（文本条件预训练）| 97.4% |
| **A-pre（动作条件预训练）** | **98.6%** |
| P-pre（策略目标预训练）| 98.8% |

**关键发现**: 动作条件预训练（A-pre）相比文本条件预训练（T-pre）提升 1.2%，说明动作先验显著优于纯视觉先验。

---

### Table 7: 历史采样策略消融（Table 8）

| 历史采样策略 | PSNR↑ | SSIM↑ | EPE↓ |
|------------|-------|-------|------|
| 无历史 | 25.41 | .8806 | .3969 |
| 滑动窗口 | 25.63 | .8840 | .3900 |
| **姿态引导（Ours）** | **26.64** | **.8957** | **.3498** |

**关键发现**: 姿态引导历史采样在 PSNR 和光流误差（EPE）上均显著优于其他策略，是长程稳定生成的关键。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 多具身预训练集（7个子集）| 2.156M 轨迹 | 20+ 具身形态，统一 14 维动作格式 | 预训练 |
| LIBERO | 6k 轨迹 | Franka，4个任务套件 | 精调+评估 |
| LIBERO-Plus-Spatial | 3.7k 轨迹 | OOD 评估集 | 迁移测试 |
| RoboNet | 100k 轨迹 | 多机器人 | 仿真器精调 |
| Flexiv 自采集数据 | 2k 轨迹 | 双臂 Flexiv Rizon 4S，5个任务 | 真机评估 |

### 实现细节

- **Backbone**: Cosmos-Predict2-2B-Video2World（DiT，2.5B 参数）
- **预训练硬件**: 64 × H200 GPU，batch size 12/GPU，梯度累积 4
- **预训练学习率**: 1e-4，weight decay 0.1，训练 2 epochs
- **采样调度**: 35 步，$t_{\min}=0.01$，$t_{\max}=200.0$
- **A2World-sim 精调**: 8 × H200，batch size 24，历史长度 $T_h=20$，$w_r=0.3$，$w_t=1.0$
- **A2World-policy 训练**: 32 × H200，全局 batch size 256，LIBERO 训练 20k 步，OOD 评估训练 24k 步，$\alpha_v=\sqrt{6}$，$\alpha_a=0.5$
- **动作统一**: 所有具身形态动作标准化为 14 维双臂格式（每臂 7D：末端执行器位姿 + 夹爪状态）

### 可视化结果

- 基础模型在 DROID 上可精确操控不同物体的抓取序列，支持失败模拟
- A2World-sim 在长程"链条入盒"任务上生成帧保持物理一致性，基线方法出现明显漂移
- A2World-policy 在接触丰富任务（插 RAM、拨开关）上优势最大，任务完成率提升约 10-20%

---

## 批判性思考

### 优点

1. **因果监督信号**: 以动作为条件建立视觉预测的因果结构，比文本条件更精准地捕捉机器人与环境交互
2. **数据规模**: 2.1M+ 多具身形态轨迹覆盖广泛，动作统一到 14 维格式有效缓解具身差异问题
3. **双用途验证充分**: 同一基础模型适配仿真和策略两个下游任务，且均超越当前 SOTA，范式创新性强
4. **仿真器替代真机**: ρ=0.916 的相关性意味着 A2World-sim 可大幅降低真机评估成本

### 局限性

1. **计算成本高**: 预训练需 64 × H200，推理用 35 步扩散采样，部署门槛较高
2. **动作格式统一限制**: 强制统一到双臂 14 维格式可能损失单臂或移动底座的特定信息，对非双臂具身形态的迁移效果待验证
3. **视频生成质量与策略质量的解耦**: 联合训练假设视频质量与动作质量正相关，但实验中两者权重需手动调整（$\lambda_a$，$\alpha_v$，$\alpha_a$）

### 潜在改进方向

1. 探索更轻量化的蒸馏版本，降低推理延迟以满足实时控制需求
2. 扩展到移动操作（Mobile Manipulation）场景，验证 14 维格式的扩展性
3. 研究 A2World-sim 的主动学习闭环，自动生成难例轨迹提升策略鲁棒性

### 可复现性评估

- [x] 代码开源（GitHub: LogosRoboticsGroup/A2World）
- [ ] 预训练模型（未明确提供权重）
- [x] 训练细节完整（论文中详细列出超参数）
- [ ] 数据集可获取（部分子集如 InternData 非公开）

---

## 关联笔记

### 基于

- [[Cosmos]]: 基础 DiT 架构权重来源（Cosmos-Predict2-2B-Video2World）
- [[EDM (Elucidated Diffusion Model)|EDM]]: 采用 EDM 框架作为扩散训练目标
- [[Self-forcing]]: A2World-sim 的自回归稳定训练策略

### 对比

- [[Cosmos-Predict2]]: 文本条件视频生成基线，无动作先验
- [[π0]]: OOD 迁移对比策略基线（78.6% vs 88.5%）
- [[GE-Act]]: 最强策略对比基线（87.5% vs 88.5%）
- [[Prophet]]: 世界模型生成质量对比基线

### 方法相关

- [[Action-Conditioned World Model|动作条件世界模型]]: 核心建模范式
- [[Diffusion Transformer (DiT)|DiT]]: 骨干网络架构
- [[Classifier-Free Guidance (CFG)|CFG]]: Per-modality 推理引导策略
- [[Pose-guided History Sampling|姿态引导历史采样]]: A2World-sim 关键创新

### 硬件/数据相关

- [[LIBERO]]: 主要评估 benchmark
- [[Flexiv Rizon]]: 真机实验平台（双臂 Rizon 4S）
- [[AgiBot Genie]]: 最大预训练数据来源（1.003M 轨迹）

---

## 速查卡片

> [!summary] A2World (ECCV 2026)
> - **核心**: 动作条件预训练学习可迁移动力学先验，统一适配仿真器与策略
> - **方法**: DiT 2.5B + 2.1M 多具身轨迹 + 姿态引导历史采样 + MoE 式视频-动作联合扩散
> - **结果**: LIBERO 98.6% 成功率；OOD 迁移 88.5%（超越 π0 10%）；仿真-真机相关 ρ=0.916
> - **代码**: https://github.com/LogosRoboticsGroup/A2World

---

*笔记创建时间: 2026-07-01*

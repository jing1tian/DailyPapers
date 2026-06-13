---
title: "WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation"
method_name: "WEAVER"
authors: [Arnav Kumar Jain, Yilin Wu, Jesse Farebrother, Gokul Swamy, Andrea Bajcsy]
year: 2026
venue: arXiv
tags: [world-model, robotic-manipulation, flow-matching, diffusion-forcing, policy-evaluation, policy-improvement, test-time-planning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.13672
created: 2026-06-13
---

# 论文笔记：WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila / McGill University; Carnegie Mellon University |
| 日期 | June 2026 |
| 项目主页 | [https://arnavkj1995.github.io/WEAVER/](https://arnavkj1995.github.io/WEAVER/) |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13672) / Code: 暂未开源 |

---

## 一句话总结

> WEAVER 是首个同时满足高保真度、长时序一致性和高效推理三重目标的机器人操作[[世界动作模型|世界模型]]，通过[[流匹配]] + [[Diffusion Forcing]] + 潜在奖励头，在策略评估（ρ=0.870）、策略改进（+38%）和测试时规划（+14%，速度 20×）三大应用上均取得 SOTA。

---

## 核心贡献

1. **三重目标统一**: 首个同时满足 fidelity（FID/FVD SOTA）、consistency（长时序稳定）、efficiency（生成速度）的机器人[[世界动作模型|世界模型]]架构，打破"三选二"的传统权衡
2. **高效推理加速**: [[KV 缓存]]跨去噪步复用 + 余弦噪声调度 + [[Rectified Flow|ReFlow 后训练蒸馏]]，推理速度比 Ctrl-World 快最高 20×
3. **轻量级潜在验证器**: 基于 [[AdaPool]] 的奖励头 + [[TD(λ)|Critic 网络]]直接在潜在空间做策略评估，无需解码到像素，验证三大下游应用均在真实 Franka 机器人上完成

---

## 问题背景

### 要解决的问题

机器人操作[[世界动作模型|世界模型]]需要同时满足三个核心目标：
- **保真度（Fidelity）**: 仿真轨迹与真实世界高度相关（ρ 接近 1.0）
- **一致性（Consistency）**: 长时序预测（10s+）保持时序连贯性，不发散
- **效率（Efficiency）**: 快到足以支撑测试时规划的实时推理速度

### 现有方法的局限

- **视频生成模型**（如 Cosmos）：保真度高但推理极慢（>42s/步），无法用于实时规划
- **JEPA 式世界模型**：效率高但潜在表示不可解码，难以做可视化策略评估
- **DreamerV3/V4**：从头训练编码器，OOD 鲁棒性差
- **[[Ctrl-World]]**（当前 SOTA）：多视角生成但速度低于实时，且不预测本体状态导致可变形物体操作效果差

### 本文的动机

整合多个社区的最优设计：视频生成社区的[[Diffusion Forcing]]和[[流匹配]]、潜在世界模型的奖励预测头、JEPA 的未来潜在预测目标、Ctrl-World 的多视角架构——同时引入 SD3 VAE 编码器保留视觉语义先验，避免从头训练的 OOD 脆弱性。

---

## 方法详解

### 模型架构

WEAVER（World Estimation Across Views for Embodied Reasoning）采用 **编码器-动力学-奖励-解码器** 四级架构：

- **输入**: 语言指令 $\ell$ + 多视角 RGB 观测 $(I^{ext}_t, I^{wrist}_t)$ + 本体状态 $q_t \in \mathbb{R}^8$ + 动作序列 $\mathbf{a}_t$
- **编码器**: 预训练 [[Stable Diffusion 3]] VAE（冻结权重），H×W 图像 → 潜在 token
- **核心模块**: 32 层 2D [[Transformer]] + [[稀疏记忆|稀疏长期记忆]] + [[Diffusion Forcing]] 动力学预测
- **加速**: [[KV 缓存]] + [[SPRINT]] token 剪枝 + 余弦噪声调度 + [[Rectified Flow|ReFlow 后训练]]
- **输出**: 预测未来潜在状态 $\hat{\mathbf{z}}_t$、奖励 $\hat{r}_t$、可选解码图像 $\hat{o}_t$
- **总参数**: 928M

### 核心模块

#### 模块1: 多视角潜在编码

**设计动机**: 利用[[Stable Diffusion 3]] VAE 的强大视觉先验，避免从头训练编码器导致的 OOD 鲁棒性损失；多视角同时预测解决单摄像头遮挡问题

**具体实现**:
- SD3 VAE 对外置相机（Zed 2i）和腕部相机（Zed Mini）分别编码为 $H \times W$ 个 patch token
- 本体状态 $q_t$（关节角度）通过 MLP 投影到相同的 token 维度后拼接
- 两路视角 token 在 Transformer 中通过空间注意力交叉融合

#### 模块2: 稀疏记忆 + 短期历史

**设计动机**: 处理遮挡和视角变化，同时保持长时序感知能力，无需存储每帧完整状态

**具体实现**:
- **稀疏记忆**: $\mathbf{z}_t^{mem} = (\ldots, z_{t-2k}, z_{t-k})$，每隔 $k=5$ 步采一帧，默认保留 6 帧历史
- **短期历史**: $\mathbf{z}_t^{hist} = (z_{t-1}, z_t)$，保留最近两帧维持局部连续性
- 两者拼接作为条件输入潜在动力学模型

#### 模块3: 潜在动力学模型（核心预测器）

**设计动机**: 用[[流匹配]]替代 DDPM 训练目标，实现更快推理；[[Diffusion Forcing]]对每时间步独立采样噪声水平提升长时序一致性

**具体实现**:
- 32 层 2D [[Transformer]]：空间注意力（patch 维度）+ 因果时序注意力（帧间）
- 每个 dynamics block：[[RMSNorm]] → 空间 Multi-Head Attention（[[RoPE]] + [[QK Normalization]]）→ 时序 Causal Attention → [[SwiGLU]] FFN
- 参数：32 层，1536 隐藏维，16 注意力头，96 头维度
- **[[SPRINT]] 加速**：训练时随机 drop 50% patch token，推理时全量保留；[[KV 缓存]]跨去噪步复用记忆/历史 token

#### 模块4: 潜在验证器（奖励头 + Critic 网络）

**设计动机**: 直接在潜在空间估计奖励，无需像素解码，实现轻量高效的策略评估和优势估计

**具体实现**:
- **奖励头**: [[AdaPool]] 对 $H \times W$ 个 token 自适应聚合 → 两层 MLP → 预测 $[0,1]$ 任务成功概率
- **Critic 网络**: 同架构，预测 [[TD(λ)|bootstrapped λ-returns]]，估算想象视野之外的累积价值
- 语言指令 $\ell$ 通过 Cross-Attention 注入两个头

#### 模块5: ReFlow 后训练（推理加速）

**设计动机**: [[Rectified Flow|Rectified Flow 蒸馏]] 将多步生成的 $(x^0, x^1)$ 对用于重新训练，使流轨迹趋于直线，支持极少步数（1-4 NFE）高质量生成

**具体实现**:
- 用预训练 WEAVER（教师，50 NFE）生成高质量样本对
- 以教师输出替换真实数据，用相同[[流匹配]]目标重新训练（学生继承教师权重）
- 配合余弦噪声调度：$k = 1 - \cos(i\pi / 2K)$，集中推理步数在关键区间

---

## 关键公式

### 公式1: [[流匹配|Flow Matching 噪声插值]]

$$
x^\tau_t = \tau x^1_t + (1 - \tau) x^0_t, \quad \tau \sim \mathcal{U}(0,1)
$$

**含义**: 在干净潜在 $x^1_t$ 和标准高斯噪声 $x^0_t$ 之间线性插值，构造[[流匹配]]训练所需的中间态；[[Diffusion Forcing]]对每时间步独立采样 $\tau$，使模型学会在任意噪声水平下预测

**符号说明**:
- $x^1_t$：目标干净潜在表示（SD3 VAE 编码的真实观测）
- $x^0_t$：标准高斯噪声 $\mathcal{N}(0, I)$
- $\tau \sim \mathcal{U}(0,1)$：插值系数，对每时间步独立采样（Diffusion Forcing 关键）

### 公式2: [[Diffusion Forcing|世界模型训练损失]]

$$
\mathcal{L}^{WM}(\phi) = \mathbb{E}_{x^0, x^1, \tau} \left[ \left\| (x^1_t - x^0_t) - f_\phi\!\left(\mathbf{z}_t^{hist}, \mathbf{z}_t^{mem}, \mathbf{a}_t, x^\tau_t, \tau\right) \right\|_2^2 \right]
$$

**含义**: 训练动力学模型 $f_\phi$ 预测从 $x^\tau_t$ 到 $x^1_t$ 的速度场方向（即"去噪方向"），结合[[Diffusion Forcing]]使模型在任意噪声水平下都能稳定预测

**符号说明**:
- $f_\phi$：32 层潜在动力学 Transformer
- $\mathbf{z}_t^{hist}$：短期历史潜在（最近 2 帧）
- $\mathbf{z}_t^{mem}$：稀疏记忆潜在（每 5 步采样 6 帧）
- $\mathbf{a}_t$：动作序列条件

### 公式3: [[Rectified Flow|ReFlow 后训练蒸馏损失]]

$$
\mathcal{L}^{ReFlow}(\phi) = \mathbb{E} \left[ \left\| (\hat{x}^1_t - x^0_t) - f_\phi\!\left(\mathbf{z}_t^{hist}, \mathbf{z}_t^{mem}, \mathbf{a}_t, x^\tau_t, \tau\right) \right\|_2^2 \right]
$$

**含义**: 用教师模型（50 NFE）生成的配对样本 $\hat{x}^1_t$ 替换真实数据，蒸馏出直线流轨迹，使学生模型在 1-4 NFE 时即可高质量生成

**符号说明**:
- $\hat{x}^1_t$：教师 WEAVER（50 NFE）生成的高质量样本，替代真实观测 $x^1_t$

### 公式4: [[余弦噪声调度|推理余弦噪声调度]]

$$
k_i = 1 - \cos\!\left(\frac{i\pi}{2K}\right), \quad i \in \{0, 1, \ldots, K\}
$$

**含义**: 推理时按余弦曲线分配 $K$ 个去噪步骤的噪声水平，使早期（高噪声阶段）步距小、后期（精细去噪）步距大，比线性调度在相同 NFE 下质量更优

**符号说明**:
- $i$：当前去噪步编号
- $K$：总去噪步数（NFE）
- $k_i$：第 $i$ 步的噪声水平（$0$ = 纯噪声，$1$ = 干净）

### 公式5: [[TD(λ)|自举 λ-return 价值估计]]

$$
\mathbf{v}_t^\lambda = R(z_t, \ell) + \gamma\!\left((1-\lambda)V(z_{t+1}, \ell) + \lambda\, \mathbf{v}_{t+1}^\lambda\right)
$$

$$
\mathbf{v}_{t+K}^\lambda = V(z_{t+K}, \ell)
$$

$$
\mathcal{L}^{critic}(V) = \left\| V(z_t, \ell) - \mathbf{v}_t^\lambda \right\|_2^2
$$

**含义**: 用折扣 λ-return 自举训练 Critic 网络，在 $K$ 步想象视野内递推估算累积价值，末端用 Critic bootstrap 替代真实奖励

**符号说明**:
- $R(z_t, \ell)$：奖励头预测的即时任务成功概率
- $V(z_t, \ell)$：Critic 网络预测的状态价值
- $\gamma = 0.995$：折扣因子
- $\lambda = 0.95$：TD(λ) 混合系数（0 = TD(0)，1 = Monte Carlo）

### 公式6: [[Advantage Estimation|蒙特卡洛优势估计]]

$$
\hat{A}_t^b = \sum_{\ell=1}^{H} \gamma^{\ell-1} R\!\left(\hat{z}_{t+\ell}^b, \ell\right) + \gamma^H V\!\left(\hat{z}_{t+H}^b, \ell\right) - V(z_t, \ell)
$$

**含义**: 计算第 $b$ 条 rollout 轨迹相对基线价值的优势，用于：策略改进时筛选高质量合成轨迹；测试时规划时选择最优动作块

**符号说明**:
- $\hat{z}_{t+\ell}^b$：第 $b$ 条想象轨迹第 $\ell$ 步的潜在状态
- $H = Kh$：总想象步数（$K$ 个动作块，每块 $h$ 步）
- $V(z_t, \ell)$：当前状态基线价值
- $\epsilon_{adv}$：优势过滤阈值，防止在无法完成的状态继续优化

### 公式7: [[动作适配器|关节位置增量预测损失]]

$$
\mathcal{L} = \mathcal{L}_{joint} + 5.0 \cdot \mathcal{L}_{gripper}
$$

$$
q_{t+k} = q_t + \hat{\Delta}_{joint,k}, \quad g_{t+k} = g_t + \hat{\Delta}_{gripper,k}
$$

**含义**: 3 层 MLP 动作适配器将 π₀.₅ 的关节速度命令转换为关节位置增量，夹爪损失权重放大 5× 以保证精确夹持，输出 128→T×8 的 15 步动作块

**符号说明**:
- $q_t \in \mathbb{R}^7$：当前关节角度
- $g_t$：当前夹爪开合状态
- $\hat{\Delta}_{joint,k}$：第 $k$ 步关节位置增量预测

---

## 关键图表

### Figure 1: 系统概览（三目标 + 三应用）

![Figure 1](https://arxiv.org/html/2606.13672v1/x1.png)

**说明**: WEAVER 同时满足高保真度、长时序一致性和高效生成三目标，解锁三类下游应用：策略评估（中）、策略改进（右上）和测试时规划（右下）。输入多视角观测 + 动作序列，核心在潜在空间推理，奖励头选择最优动作。

### Figure 2: WEAVER 完整架构

![Figure 2](https://arxiv.org/html/2606.13672v1/x2.png)

**说明**: 左侧：世界模型编码稀疏记忆、短期历史和动作序列，在潜在空间用[[流匹配]] + [[Diffusion Forcing]]预测未来 rollout；中间：潜在验证器（[[AdaPool]] 奖励头 + Critic 头）通过[[Advantage Estimation|优势估计]]引导策略分布；右侧：不同动作序列对应的解码生成结果，对比好/差动作的视觉差异。

### Figure 3: 长时序 FID 对比

![Figure 3](https://arxiv.org/html/2606.13672v1/x3.png)

**说明**: 在 50/150 步（3s/10s）时序长度下 WEAVER 相比 Ctrl-World 的 FID 对比。WEAVER 在所有视野长度均保持更低 FID，体现[[Diffusion Forcing]]对长时序一致性的核心贡献；即使用更少 NFE（16 vs 50）WEAVER 依然胜出。

### Figure 4: 奖励预测 + 测试时规划优势筛选

![Figure 4](https://arxiv.org/html/2606.13672v1/x4.png)

**说明**: 左图：WEAVER 预测奖励曲线与 Robometer 真实进度奖励高度吻合（定量验证潜在验证器有效性）；右图：基于[[Advantage Estimation|优势值]]的动作样本筛选，高亮轨迹对应最高优势值，在想象空间中对应更好的任务结局。

### Figure 5: FVD vs 推理时间 Pareto 前沿

![Figure 5](https://arxiv.org/html/2606.13672v1/x5.png)

**说明**: WEAVER 在 NFE=8、16、32、50 各点上均 Pareto 主导 Ctrl-World——在更低推理时间下实现更低 FVD（外置和腕部两个视角均成立，DROID val 和 OOD 任务两个数据集均成立）。推理时间最高差异约 30-50×。

### Figure 6: 策略评估结果

![Figure 6](https://arxiv.org/html/2606.13672v1/x6.png)

**说明**: 左图定性对比：仅 WEAVER 和 WEAVER-FT 能正确想象"毛巾入篓"（PnP Towel）和"咖啡豆散落"（Pour Beans）等物理细节，Ctrl-World 预测发散；右图定量：WEAVER-FT 达到 ρ=0.870 皮尔逊相关，显著超越 Ctrl-World 的 ρ≈0.60。

### Figure 7: 策略改进（微调对比 + 数据扩展）

![Figure 7](https://arxiv.org/html/2606.13672v1/x7.png)

![Figure 7b](https://arxiv.org/html/2606.13672v1/x8.png)

**说明**: 左图：各数据来源（基础策略 / 真实 / 合成 / 混合）在 5 任务上的平均成功率对比，合成+真实混合超越纯真实数据 11%；右图（Pour Beans 任务）：随合成数据量从 1K 增至 5K 段，性能持续提升且超越纯真实数据，验证合成数据的数据扩展定律有效性。

### Figure 9: 测试时规划（Best-of-4）结果

![Figure 9](https://arxiv.org/html/2606.13672v1/x10.png)

**说明**: WEAVER 测试时规划在五个任务上平均比基础策略 π₀.₅ 提升 14%，最高提升 20%（弱策略基线）。相比 Ctrl-World：质量相当时速度快约 20×（RTX A6000 单卡）。

### Figure 10: 硬件配置与五任务

![Figure 10](https://arxiv.org/html/2606.13672v1/x11.png)

**说明**: 左图：Franka Emika Panda + 右侧外置 Zed 2i + 腕部 Zed Mini 两视角配置；右图：五个操作任务示例（Stack Bowls / PnP Bag / PnP Marker / PnP Towel / Pour Beans），每个任务含初始状态（上行）和目标状态（下行）。

### Table 1: 视频生成质量对比（FID↓ / FVD↓ / 推理时间↓）

| 数据集 | 方法 | NFE | Ext. FID↓ | Ext. FVD↓ | Wrist FID↓ | Wrist FVD↓ | Time (s)↓ |
|--------|------|-----|-----------|-----------|-----------|-----------|----------|
| DROID val | Ctrl-World | 16 | 26.09 | 78.73 | 33.83 | 195.37 | 14.65 |
| DROID val | Ctrl-World | 50 | 22.44 | 55.05 | 25.32 | 91.77 | 42.33 |
| DROID val | **WEAVER** | 16 | **10.20** | **27.83** | **21.50** | 90.72 | **4.78** |
| DROID val | **WEAVER** | 50 | **9.51** | **26.54** | **16.75** | **66.89** | 14.25 |
| Task OOD | Ctrl-World | 16 | 36.16 | 139.54 | 38.76 | 277.13 | 14.65 |
| Task OOD | Ctrl-World | 50 | 31.44 | 91.48 | 33.47 | 145.86 | 42.33 |
| Task OOD | **WEAVER** | 16 | **23.95** | **88.27** | **30.77** | 184.62 | **4.78** |
| Task OOD | **WEAVER** | 50 | **23.48** | **87.03** | **27.37** | **145.04** | 14.25 |

**关键发现**: WEAVER 在分布内（DROID val）外部相机 FID 从 22.44→9.51（-58%），同时推理时间 WEAVER 16 NFE 比 Ctrl-World 16 NFE 快 3×，实现质量与速度的双提升。

### Table 2: 策略评估皮尔逊相关系数

| 世界模型 | Pearson ρ↑ |
|---------|-----------|
| Ctrl-World（DROID 预训练） | ~0.60 |
| WEAVER（DROID 预训练） | ~0.65–0.70 |
| **WEAVER-FT（任务微调）** | **0.870** |

**关键发现**: 任务特定微调（仅 50 rollout × 5 任务，16K steps）对策略评估质量至关重要，相关系数从 ~0.65 跳升至 0.870。

### Table 3: 策略改进各配置成功率

| 数据来源 | 片段数 | 平均成功率提升 |
|---------|-------|------------|
| 基础策略 π₀.₅ | — | 0%（基准） |
| FT w/ 真实数据 | 1,000 | Δ_real |
| **FT w/ 合成数据（WEAVER）** | 1,000 | Δ_real − 4% |
| **FT w/ 混合数据** | 2,000 | Δ_real + 11% |
| **总体提升（混合）** | — | **+38%** |

**关键发现**: 合成数据仅比真实数据落后 4%；混合后超越纯真实 11%，说明合成数据有效补充真实数据的覆盖盲区；数据扩展定律在合成数据上有效（1K→5K 持续提升）。

### Table 4: 模型超参数

| 组件 | 超参 | 值 |
|------|------|-----|
| Transformer | 层数 | 32 |
| | 嵌入维度 | 1536 |
| | 注意力头数 | 16 |
| | 头维度 | 96 |
| | SPRINT drop 概率 | 0.5 |
| 预训练 | batch size | 32 |
| | batch length | 8 |
| | 记忆帧数 / 步长 | 6 / 5 |
| | 学习率 | 1e-4 |
| | warmup steps | 10,000 |
| | EMA decay | 0.9999 |
| | 总训练步数 | 1,000,000 |
| 微调 | 学习率 | 2e-5 |
| | 总训练步数 | 16,000 |
| 策略 | 折扣 γ | 0.995 |
| | λ | 0.95 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | 76K 轨迹 | 多机器人、多场景、多语言指令 | 世界模型预训练 |
| 任务微调数据 $\mathcal{D}^{FT}_{real}$ | 50 rollouts × 5 任务 | 目标场景特定 | 世界模型微调 |
| 策略评估验证集 $\mathcal{D}^{val}_{real}$ | 20 rollouts × 5 任务 | 含多质量策略 | 超参调优与评估 |

### 五个操作任务

| 任务 | 描述 | 挑战 |
|------|------|------|
| Stack Bowls | 将一只碗叠放到另一只碗上 | 精确对齐，刚性物体 |
| PnP Bag | 将可变形薯片袋放到盘子上 | 可变形物体 |
| PnP Marker | 拾取并将马克笔插入杯子 | 精确插入 |
| PnP Towel | 将软毛巾放入篮子 | 可变形物体，布料 |
| Pour Beans | 将咖啡豆倒入碗中 | 散粒物质，非刚性流 |

### 实现细节

- **编码器**: Stable Diffusion 3 VAE（冻结权重）
- **动力学模型**: 32 层 2D Transformer，928M 参数
- **优化器**: AdamW，lr=1e-4（预训练），lr=2e-5（微调）
- **EMA**: decay=0.9999
- **Batch Size**: 32，Batch Length 8
- **预训练**: 1M steps，4 × H100 GPU，约 10 天
- **微调**: 16K steps，4 × H100 GPU，约 6 小时
- **基础策略**: [[π₀.₅]] VLA（在 DROID 数据集上预训练）
- **动作频率**: 5 Hz（3× 降采样），动作块 15 步
- **测试时规划**: Best-of-4，想象步长 $h=12$

---

## 批判性思考

### 优点

1. **三目标首次统一**: 在此前被认为相互冲突的三目标（质量/一致性/速度）上同时取得领先，方法论贡献清晰
2. **工程设计精细**: KV 缓存、ReFlow 后训练、SPRINT 块、余弦调度多重加速组合，实际加速比有说服力（20×）
3. **合成数据价值验证**: 仅 50 rollout 真实数据微调世界模型，即可生成媲美真实数据的合成训练集，大幅降低数据采集成本
4. **真实硬件全面验证**: 三大应用场景均在真实 Franka 机器人上验证，不依赖仿真

### 局限性

1. **部分可观测性**: 纯视觉世界模型无法感知接触力/触觉信息，接触密集任务的轨迹预测存在固有盲区
2. **可变形物体建模薄弱**: 缺乏物理先验，布料/流体/颗粒的动态预测精度有限（论文附录 A5 坦承）
3. **规划视野短**: 测试时规划仅选最优单 action chunk（$h=12$ 步），无法做多轮迭代长时序 MCTS 式规划
4. **奖励监督噪声依赖**: 依赖 Robometer 作为监督信号，Robometer 本身存在噪声，限制了奖励头的精度上限
5. **多机器人泛化未验证**: 当前仅在单一 Franka 平台验证，跨体态泛化能力未知

### 潜在改进方向

1. 引入触觉传感器作为额外观测通道，改善接触状态的感知盲区
2. 结合基于物理的可变形模拟（FEM/PBD）作为软约束，改善非刚性物体预测
3. 扩展为多轮树搜索规划（MCTS-style），发挥长时序一致性优势

### 可复现性评估

- [ ] 代码开源（项目主页提及，截至论文提交时未确认）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（附录 A2 包含完整超参数表格）
- [x] 数据集可获取（DROID 为公开数据集，π₀.₅ 权重公开）

---

## 关联笔记

### 基于

- [[Ctrl-World]]: 主要对比基线，WEAVER 继承其多视角设计并显著提速
- [[Diffusion Forcing]]: 用于长时序一致性生成的核心训练范式
- [[流匹配]]: 替代 DDPM 的训练目标，加速推理收敛
- [[Rectified Flow]]: ReFlow 后训练的理论基础
- [[π₀.₅]]: 下游策略基础模型，在五个任务上被 WEAVER 改进

### 对比

- [[Ctrl-World]]: 当前 SOTA，WEAVER 在 FID/FVD/速度上全面 Pareto 支配
- [[DreamerV3]]: 从头训练编码器方案，OOD 鲁棒性较弱

### 方法相关

- [[流匹配]]: 核心训练目标
- [[Diffusion Forcing]]: 长时序一致性训练范式
- [[KV 缓存]]: 跨去噪步复用推理加速
- [[Rectified Flow]]: ReFlow 后训练减少 NFE
- [[AdaPool]]: 奖励头 token 自适应聚合
- [[TD(λ)]]: Critic 网络训练目标
- [[Advantage Estimation]]: 策略改进和测试时规划的筛选机制
- [[SPRINT]]: 训练时 token 剪枝加速方法
- [[RoPE]]: Transformer 位置编码
- [[SwiGLU]]: 前馈网络激活函数
- [[RMSNorm]]: 归一化方案
- [[QK Normalization]]: 注意力头归一化

### 硬件/数据相关

- [[DROID]]: 预训练数据集（76K 轨迹）
- [[Franka Emika Panda]]: 实验机械臂平台
- [[Stable Diffusion 3]]: VAE 编码器来源

---

## 速查卡片

> [!summary] WEAVER: World Model for Robotic Manipulation
> - **核心**: 首个同时满足高保真度/长时序一致性/高效推理三重目标的机器人操作世界模型
> - **方法**: SD3 VAE 编码 + 稀疏记忆 + Diffusion Forcing + Flow Matching + KV 缓存加速 + ReFlow 蒸馏 + 潜在奖励/Critic 头
> - **结果**: 策略评估 ρ=0.870，策略改进 +38%（合成+真实混合），测试时规划 +14%（速度 20× vs Ctrl-World）
> - **代码**: [https://arnavkj1995.github.io/WEAVER/](https://arnavkj1995.github.io/WEAVER/)（暂未开源）

---

*笔记创建时间: 2026-06-13*

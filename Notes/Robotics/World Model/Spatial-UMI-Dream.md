---
title: "Spatial-UMI-Dream: Scaling World Action Models with Metric Geometry for UMI Manipulation"
method_name: "Spatial-UMI-Dream"
authors: [Weiduo Yuan, Ze Yuan, Sihan Liu, Jionghao Wang, Liang Heng, Zhenyu Zhao, Yue Wang]
year: 2026
venue: arXiv
tags: [world-action-model, umi, metric-depth, bimanual-manipulation, flow-matching, stereo-vision, robot-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.07229v1/
created: 2026-11-07
---

# 论文笔记：Spatial-UMI-Dream

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | — |
| 日期 | November 2026 |
| 项目主页 | [spatial-umi-dream](https://weiduoyuan.com/spatial-umi-dream/) |
| 对比基线 | [[π₀.₅]]、RDT2-FM、[[Fast-WAM]]、LingBot-VA |
| 链接 | [arXiv](https://arxiv.org/abs/2611.07229) / Code N/A |

---

## 一句话总结

> Spatial-UMI-Dream 是首个在 [[UMI]] 场景下融合立体度量几何（RGB-D）的 5B 参数[[世界动作模型（WAM）]]，配套 [[RoboTwin-UMI]] 基准，在真实与仿真中均显著超越 [[π₀.₅]]。

---

## 核心贡献

1. **Spatial-UMI-Dream 世界动作模型**: 在 605 小时 [[UMI]] RGB-D 数据上预训练的 5B 参数 [[世界动作模型（WAM）]]，联合预测未来 RGB 帧、度量深度和机器人动作，以 [[Wan 2.2]] 视频骨干为基础。
2. **立体度量几何集成**: 对 UMI 夹爪三目摄像头做立体标定，用 Fast [[FoundationStereo]] 估计度量深度，编码后与 RGB 拼成双夹爪 RGB-D 画布，无需独立 3D 骨干网络即可融入视觉预训练知识。
3. **[[RoboTwin-UMI]] 仿真基准**: 基于 [[RoboTwin]] 构建 20 任务 UMI 专属仿真基准，系统评测光照变化和 embodiment 迁移（Aloha-AgileX → ARX X5）下的策略泛化性。

---

## 问题背景

### 要解决的问题
[[UMI]] 的手持夹爪摄像头视角近、持续移动，相机运动、物体运动和场景几何完全纠缠，仅凭 RGB 难以推断度量尺度和空间布局，导致纯 RGB 的 [[世界动作模型（WAM）]] 在 UMI 场景下表现受限；且双臂协作需要在同一 3D 工作空间中协调两个独立运动视图。

### 现有方法的局限
- 基于 [[Wan 2.2]] 的通用 WAM（Fast-WAM、LingBot-VA）仅预测 RGB 未来帧，缺失深度几何信息；
- 几何感知策略（3D Diffusion Policy、SpatialVLA）需要显式 3D 输入或额外对齐教师，系统复杂度高；
- 现有 UMI 策略缺少系统化的 embodiment 迁移和环境泛化测试基准。

### 本文的动机
将度量深度作为世界状态的一部分联合预测，让 [[Wan 2.2]] 的视频预训练知识天然融合空间几何，同时用 [[RoboTwin-UMI]] 量化 UMI 策略的真实泛化能力。

---

## 方法详解

### 模型架构

Spatial-UMI-Dream 以 [[Wan 2.2]] 视频生成模型为骨干，采用双流（世界 + 动作）的 [[世界动作模型（WAM）]] 架构：

- **输入**: 语言指令 $\ell$ + 双夹爪 RGB-D 画布 $C_t$ + 机器人状态 $s_t$
- **Backbone**: [[Wan 2.2]] 视频 Transformer（5B 参数）
- **核心模块**: [[Block Causal Attention]] 时序建模 + Fast [[FoundationStereo]] 度量深度 + 双夹爪 RGB-D 画布
- **输出**: 未来 RGB-D 画布帧 + [[动作分块]] $a_{t:t+k}$
- **总参数**: 5B

### 核心模块

#### 模块1: 立体度量深度提取

**设计动机**: [[UMI]] 夹爪摄像头视角变化剧烈，仅凭 RGB 无法推断度量深度，立体视觉提供无歧义的度量尺度信息。

**具体实现**:
- GenRobot DAS UMI 系统每个夹爪配备三个标定摄像头（中/左/右鱼眼），中间 + 右侧摄像头构成立体对
- 立体对经极线矫正后，由 Fast [[FoundationStereo]] 估计视差图
- 视差通过立体标定转换为度量 Z 深度（单位：米），再重映射回中间摄像头的原始鱼眼坐标系，同时矫正镜头畸变
- 离线预处理与部署时处理流程完全一致，确保无分布偏移

#### 模块2: 双夹爪 RGB-D 画布

**设计动机**: 将空间几何信息无缝融入视频生成的视觉空间，无需修改预训练 [[Wan 2.2]] Transformer 架构，利用预训练 VAE 统一编码深度和 RGB。

**具体实现**:
- 深度图经深度编码（Eq.1）归一化后用 Turbo colormap 映射为三通道图像
- 2×2 画布（Eq.2）：行对应两个夹爪，列分别是 RGB 和编码深度，每格 160×160，整体 320×320 像素
- 整个画布由 [[Wan 2.2]] 预训练图像 VAE 统一编码为视频特征，无单独深度分支

#### 模块3: 两阶段训练策略

**设计动机**: 先用纯世界目标将视频骨干适配到机器人 RGB-D 域，再联合训练动作流，避免动作信号过早干扰视觉预训练知识，同时允许动作编码器/解码器从随机初始化充分收敛。

**具体实现**:
- **阶段一（世界适配）**: 仅优化 $\mathcal{L}_{\text{world}}$，禁用动作分支和状态条件；对预训练 [[Wan 2.2]] 施加梯度，605 小时数据中所有轨迹均可用
- **阶段二（联合训练）**: 从阶段一 checkpoint 初始化，激活动作编码器/解码器（随机初始化），联合优化 $\mathcal{L} = \mathcal{L}_{\text{world}} + \lambda_a \mathcal{L}_{\text{action}}$；仅使用通过运动学可行性过滤的轨迹
- 通过[[行为克隆]]（教师强制，Teacher Forcing）训练

#### 模块4: 轨迹过滤与夹爪补偿

**设计动机**: 手持演示不受机器人运动学约束，需过滤不可行轨迹；UMI 夹爪存在响应延迟，部署时需补偿。

**具体实现**:
- **轨迹过滤**: 对 Dexmate Vega1 求逆运动学并在 ManiSkill 中重放，丢弃不可行片段；同时丢弃速度/加速度/冲击超过数据集 99th 百分位的 episode
- **夹爪指令补偿**: 标定一阶延迟模型参数 $(\delta, \tau, w_{\max})$ 并求逆（Eq.6），在部署时对期望轨迹进行前馈补偿

---

## 关键公式

### 公式1: [[深度估计|深度编码]]

$$
\Phi(D)=\phi\!\left(\frac{d_{\max}-\operatorname{clip}(D,d_{\min},d_{\max})}{d_{\max}-d_{\min}}\right)
$$

**含义**: 将度量深度图归一化后，用 Turbo colormap $\phi$ 映射为三通道图像，使深度信息可被预训练图像 VAE 有效编码。

**符号说明**:
- $D$: 原始度量深度图（单位：米）
- $d_{\min} = 0.01\,\text{m}$，$d_{\max} = 1.5\,\text{m}$: 数据集范围边界（固定值）
- $\phi(\cdot)$: Turbo colormap，将 $[0,1]$ 映射为三通道 RGB 值
- 无效深度像素（如遮挡区域）映射为黑色（全零）

### 公式2: [[统一视觉画布|双夹爪 RGB-D 画布]]

$$
C_t=\begin{bmatrix}I_t^0 & \Phi(D_t^0)\\ I_t^1 & \Phi(D_t^1)\end{bmatrix}
$$

**含义**: 将两个夹爪的 RGB 图像与其对应深度编码拼成 2×2 画布，作为统一视频帧输入预训练 VAE 和 Transformer。

**符号说明**:
- $I_t^0, I_t^1$: 时刻 $t$ 两个夹爪的 RGB 图像（各 160×160）
- $\Phi(D_t^0), \Phi(D_t^1)$: 对应夹爪的编码深度图（各 160×160，三通道）
- $C_t \in \mathbb{R}^{320 \times 320 \times 3}$: 合成后的视频帧

### 公式3: [[Block Causal Attention|块因果自回归生成]]

$$
(Z_i, A_i) \sim p_{\theta}\!\left(Z_i, A_i \mid Z_{<i}, s_i, \ell\right)
$$

**含义**: 给定历史视觉块、当前状态和语言指令，[[Block Causal Attention]] Transformer 联合采样当前 RGB-D 潜变量和[[动作分块]]。

**符号说明**:
- $Z_i$: 第 $i$ 步的 RGB-D 画布潜变量（由 [[Wan 2.2]] VAE 编码）
- $A_i$: 第 $i$ 步的 [[动作分块]]（连续末端执行器轨迹）
- $Z_{<i}$: 所有历史视觉块序列
- $s_i$: 机器人本体状态（关节角等）
- $\ell$: 语言指令（通过 cross-attention 注入 Transformer）

### 公式4: [[流匹配|加权流匹配损失]]

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}\!\left[w(\tau)\left\|v_{\theta}(x_{\tau},\tau;c)-(\epsilon-x)\right\|_2^2\right]
$$

**含义**: 视频（世界）和动作流均用加权[[流匹配]]训练，$w(\tau)$ 对不同噪声水平加权以平衡各时间步的贡献。

**符号说明**:
- $x_{\tau}=(1-\tau)x+\tau\epsilon$: $\tau \in [0,1]$ 时刻的线性插值样本
- $x$: 干净数据（真实视频帧或动作），$\epsilon \sim \mathcal{N}(0,I)$: 高斯噪声
- $v_{\theta}$: 网络预测的速度场
- $c$: 条件（视觉历史 $Z_{<i}$、机器人状态 $s_i$、语言 $\ell$）
- $w(\tau)$: 噪声水平权重函数

### 公式5: [[世界动作模型（WAM）|联合训练目标]]

$$
\mathcal{L}=\mathcal{L}_{\mathrm{world}}+\lambda_a\mathcal{L}_{\mathrm{action}}
$$

**含义**: 世界预测损失与动作预测损失等权重加和，共同驱动空间感知世界动作模型的训练。

**符号说明**:
- $\mathcal{L}_{\mathrm{world}}$: 未来 RGB-D 画布的加权[[流匹配]]损失
- $\mathcal{L}_{\mathrm{action}}$: [[动作分块]]的加权[[流匹配]]损失
- $\lambda_a = 1$: 动作损失权重（实验中取等权重）

### 公式6: [[执行偏差|夹爪指令补偿]]

$$
u_k=\operatorname{clip}\!\left(\frac{r_{k+1+\delta}-\alpha r_{k+\delta}}{1-\alpha},\,0,\,w_{\max}\right)
$$

**含义**: 通过反转一阶延迟模型，将期望夹爪轨迹转换为补偿后的实际指令，消除部署时的响应延迟，使实际运动匹配预测动作。

**符号说明**:
- $r_{k+\delta}$: 延迟 $\delta$ 步后的期望开合宽度
- $\alpha = \exp(-h/\tau)$: 一阶模型衰减系数（$h$ 为控制周期，$\tau$ 为时间常数）
- $w_{\max}$: 最大夹爪开合宽度（物理限制）

### 公式7: KB4 鱼眼图像渲染

$$
\mathbf{q}_j=\pi_{K_j}\!\left(R_j^{\top}\pi_{\mathrm{KB4}}^{-1}(\mathbf{p})\right)
$$

**含义**: 在 [[RoboTwin-UMI]] 仿真中，从四个共中心的针孔视图合成 Kannala–Brandt (KB4) 鱼眼图像，用于从渲染帧中采样以生成与真实 UMI 摄像头一致的鱼眼观测。

**符号说明**:
- $\mathbf{p}$: 目标鱼眼图像像素坐标
- $\pi_{\mathrm{KB4}}^{-1}(\mathbf{p})$: KB4 鱼眼模型反投影，将像素映射到 3D 射线方向
- $R_j, K_j$: 第 $j$ 个针孔视图的旋转矩阵和内参矩阵
- $\mathbf{q}_j$: 在第 $j$ 个针孔视图中的投影像素坐标（用于双线性采样）

---

## 关键图表

### Figure 2: 模型架构与两阶段训练

![Figure 2](https://arxiv.org/html/2611.07229v1/fig2_method.png)

**说明**: Spatial-UMI-Dream 整体架构与两阶段训练流程。左侧阶段一仅预测未来 RGB-D 画布（世界目标），动作分支被旁路；右侧阶段二从阶段一 checkpoint 初始化，激活动作编码器/解码器实现联合 RGB-D 和[[动作分块]]预测。两个阶段均使用双夹爪 RGB-D 画布作为视觉输入，通过 [[Block Causal Attention]] 实现块因果自回归生成。

### Figure 3: 度量深度构建流程

![Figure 3](https://arxiv.org/html/2611.07229v1/figures/fig3_times_roman_v7.jpg)

**说明**: 每个 [[UMI]] 夹爪三目摄像头的度量深度提取流程。中间与右侧鱼眼摄像头经极线矫正后，由 Fast [[FoundationStereo]] 估计视差图，转换为度量深度后重映射回中间摄像头的鱼眼坐标系，最终用 Turbo colormap 编码为三通道深度图像，与 RGB 并排构成 RGB-D 画布。

### Figure 4: 真实世界任务套件

![Figure 4](https://arxiv.org/html/2611.07229v1/figures/real_task_suite_zed_v5.png)

**说明**: 六个真实世界任务的以自我为中心的关键帧（仅用于可视化，不作为策略输入）。从左至右：Pick-and-Place（将毛绒玩具放入碗中）、Bimanual Handover（双手转移木块入箱）、Scan-and-Place（扫码后放置商品）、Pack-and-Close（装球并关箱）、Handover-and-Pour（瓶子换手后浇花）、Table Wiping（从右向左擦桌子）。

### Figure 5: RoboTwin-UMI 评测结果

（arXiv HTML 中暂未提取到图片 URL）

**说明**: [[RoboTwin-UMI]] 基准下 20 个任务的评测结果。Panel (a) 展示 Aloha-AgileX 和 ARX X5 两种 embodiment 的平均成功率对比；Panel (b) 展示 ARX X5 随机光照设置下的逐任务成功率。Spatial-UMI-Dream 在所有设置和任务上均领先所有基线。

### Figure 6: 深度和预训练消融

（arXiv HTML 中暂未提取到图片 URL）

**说明**: RGB-D vs RGB（无深度）和两阶段预训练的消融对比，覆盖四种仿真设置（Aloha clean、Aloha 随机光照、ARX X5 随机光照、ARX X5 随机光照+环境）和真实任务平均成功率。

### Table I: 真实世界 6 任务评测

| 方法 | 平均 SR (%) | 平均 Score |
|------|-------------|-----------|
| RDT2-FM | 40.0 | — |
| LingBot-VA | 43.3 | — |
| Fast-WAM | 48.3 | — |
| [[π₀.₅]] | 60.8 | 73.3 |
| **Spatial-UMI-Dream** | **85.0** | **93.2** |

**关键发现**: 比最强基线 [[π₀.₅]] 高 24.2pp SR（85.0% vs 60.8%），双手转移任务（Bimanual Handover）提升最显著（80.0% vs 45.0%），抓取和放置精度的提升归因于度量深度信息。

### Table II: RoboTwin-UMI 泛化评测（平均成功率 %）

| 方法 | Aloha Clean | Aloha Rand. Light | ARX X5 Rand. Light |
|------|------------|-------------------|--------------------|
| RDT2 | 38.55 | 16.55 | 12.40 |
| Fast-WAM | 61.20 | 18.85 | 16.90 |
| LingBot-VA | 35.55 | 34.85 | 28.60 |
| [[π₀.₅]] | 74.45 | 66.15 | 61.85 |
| **Spatial-UMI-Dream** | **79.10** | **73.35** | **72.15** |

**关键发现**: in-domain 领先 [[π₀.₅]] 4.65pp；随机光照下领先 7.20pp；联合 embodiment+环境迁移（ARX X5 Rand. Light）下领先 10.30pp，体现最强泛化性，且 embodiment 迁移掉点最小。

### Table III: 训练配置

| 阶段 | 更新步数 | Batch Size | GPU 数量 |
|------|---------|-----------|---------|
| 阶段一（世界适配） | 30k | 192 | 24× NVIDIA RTX PRO 6000 |
| 阶段二（联合训练） | 30k | 192 | 24× NVIDIA RTX PRO 6000 |
| 真实世界微调 | 22.5k | 64 | 8× |
| [[RoboTwin-UMI]] 微调 | 10k | 128 | 8× |

### Table IV: RoboTwin-UMI 20 任务列表

共 20 个任务，步数限制从 400 到 1400 步不等，涵盖抓取、放置、堆叠、插入等操作类型，在 Aloha-AgileX 和 ARX X5 两种 embodiment 下评测。

### Table V: 真实世界演示数据

| 任务 | 演示数 | 帧数 |
|------|--------|------|
| Pick-and-Place | 20 | — |
| Bimanual Handover | 20 | — |
| Scan-and-Place | 40 | — |
| Pack-and-Close | 40 | — |
| Handover-and-Pour | 30 | — |
| Table Wiping | 30 | — |
| **合计** | **180** | **69,032** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[UMI]] RGB-D | 605 小时 | 两个物体抓取集合 + 高质量操作集合 | 两阶段预训练 |
| [[RoboTwin-UMI]] | 50 次/任务 × 20 任务 | Aloha-AgileX，仿真干净环境 | 仿真微调 |
| 真实演示 | 180 次，69,032 帧 | Dexmate Vega1 双臂，6 任务 | 真实世界微调 |

### 实现细节

- **Backbone**: [[Wan 2.2]] 视频生成模型（5B 参数）
- **优化器**: AdamW，学习率 1e-5，余弦调度，5% warmup，bfloat16，DeepSpeed ZeRO-2
- **Batch Size**: 预训练 192，微调 64/128
- **训练步数**: 预训练各阶段 30k 步，微调 22.5k / 10k 步
- **硬件**: 24×/8× NVIDIA RTX PRO 6000 GPU

### 消融实验

| 配置 | Aloha Clean (%) | ARX X5 Embodiment Shift (相对下降) |
|------|----------------|-----------------------------------|
| Wan2.2 直接微调（无预训练） | 39.50 | — |
| 仅阶段一（世界适配） | 61.25 | — |
| 两阶段完整预训练 | **79.10** | 基线 |
| 仅视频预训练（无动作预训练） | — | −15.35pp |
| RGB-only（无度量深度） | −5.75pp | −8.25pp |

**关键发现**:
- 两阶段预训练贡献最大：直接微调 39.50% → 完整预训练 79.10%（+39.60pp）
- 仅世界适配（阶段一）已带来 +21.75pp，联合训练（阶段二）再增 +17.85pp
- 度量深度（RGB-D vs RGB）in-domain +5.75pp，跨 embodiment+环境迁移 +8.25pp，真实世界 +4.2pp（80.8% → 85.0%）

---

## 批判性思考

### 优点
1. **度量深度集成优雅**: 将立体深度编码为伪 RGB 图像（Turbo colormap），无需修改预训练 [[Wan 2.2]] 架构，完全利用现有 VAE 和 Transformer 权重，即插即用。
2. **两阶段预训练有效**: 先世界后动作的分阶段策略显著优于直接微调（39.50% → 79.10%，+39.60pp），设计直觉清晰。
3. **泛化性最强**: 仅用 Aloha 数据即可泛化到 ARX X5，embodiment 迁移掉点远小于所有基线，展示了真实的跨 embodiment 迁移能力。
4. **基准贡献**: [[RoboTwin-UMI]] 填补了 UMI 策略系统化泛化测试的空白，20 任务 + 两种 embodiment + 光照随机化，设计全面。

### 局限性
1. **立体深度质量依赖**: Fast [[FoundationStereo]] 在近距离（<10cm）或纹理缺失区域可能估计不准；高速运动引起的运动模糊可能降低视差质量。
2. **硬件依赖**: 需要 GenRobot DAS 三目摄像头，标准双目 UMI 夹爪不能直接使用，限制了方法的通用性。
3. **计算成本高**: 605 小时数据 + 60k 预训练步 + 24× RTX PRO 6000 GPU，工业级资源需求，学术复现门槛高。
4. **代码未开源**: 目前仅有项目主页和论文，无代码/模型权重发布。

### 潜在改进方向
1. 探索单目深度估计（如 Depth Anything）替代立体深度，降低硬件要求；
2. 将 [[RoboTwin-UMI]] 扩展到更多 embodiment（如灵巧手）和任务类型（如液体操作）；
3. 研究减少预训练计算量的高效策略，如参数高效微调（LoRA）。

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文详细描述超参数、硬件、过滤策略）
- [x] 数据集可获取（UMI 社区数据 + RoboTwin-UMI 仿真）

---

## 关联笔记

### 基于
- [[Wan 2.2]]: 5B 参数视频生成骨干模型
- [[UMI]]: 数据采集框架和部署硬件平台
- [[RoboTwin]]: [[RoboTwin-UMI]] 仿真基准的基础框架

### 对比
- [[π₀.₅]]: 最强基线，通用 VLA，真实 60.8% SR vs 本文 85.0%
- [[Fast-WAM]]: 通用 WAM 基线（无度量深度），真实 48.3% SR

### 方法相关
- [[FoundationStereo]]: Fast FoundationStereo 的基础模型，用于度量深度估计
- [[Block Causal Attention]]: 时序块因果自回归生成的核心机制
- [[流匹配]]: 世界和动作两个流的训练目标（Flow Matching）
- [[动作分块]]: 末端执行器轨迹的动作表示方式
- [[统一视觉画布]]: 双夹爪 RGB-D 画布的设计思路

### 数据集/硬件相关
- [[RoboTwin-UMI]]: 本文提出的 UMI 专属仿真泛化基准（20 任务）

---

## 速查卡片

> [!summary] Spatial-UMI-Dream
> - **核心**: 5B 参数 WAM，联合预测双夹爪 RGB-D 未来帧 + 动作，立体度量深度无缝融入视频生成骨干
> - **方法**: 双夹爪 RGB-D 画布 + 两阶段预训练（世界适配→联合训练）+ Fast FoundationStereo 度量深度
> - **结果**: 真实 85.0% SR（vs π0.5 60.8%，+24.2pp）；仿真 79.10%（+4.65pp vs π0.5）；跨 embodiment 72.15%（+10.30pp）
> - **代码**: 暂未开源（项目主页：https://weiduoyuan.com/spatial-umi-dream/）

---

*笔记创建时间: 2026-11-07*

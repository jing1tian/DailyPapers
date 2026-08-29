---
title: "GameWAM: A World Action Model for Video Games"
method_name: "GameWAM"
authors: [Yuncheng Guo, Zhanqiu Zhang, Yiwen Guo, Weijia Li]
year: 2026
venue: arXiv
tags: [world-action-model, video-game, flow-matching, diffusion-transformer, block-causal, gui-control, minecraft]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.26200
created: 2026-08-29
---

# 论文笔记：GameWAM: A World Action Model for Video Games

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开 |
| 日期 | August 2026 |
| 项目主页 | [yunncheng.github.io/GameWAM](https://yunncheng.github.io/GameWAM/) |
| 对比基线 | [[Game-TARS]], [[OpenHA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.26200) / 代码未公开 |

---

## 一句话总结

> GameWAM 是首个面向视频游戏的[[World Action Model|世界动作模型]]，通过并行 Video/Action DiT + 块因果条件 + 流匹配，实现原生键鼠控制的闭环游戏，在 Minecraft MCU 基准上 ASR 达 50.7%，并首次揭示 LASI 生成控制失效现象。

---

## 核心贡献

1. **首个游戏闭环 WAM**: 提出 GameWAM，将[[World Action Model|世界动作模型]]范式扩展到视频游戏领域，支持原生键鼠/GUI 混合控制的闭环自主游戏。
2. **块循环控制（Block-Cycle Control）**: 设计预测-执行解耦机制（预测 $P$ 步，只执行 $E$ 步），结合层次化视觉历史（近期缓冲 + 长程记忆槽）实现长期自主游戏。
3. **LASI 现象发现**: 首次发现并系统量化"低频动作源印记"失效模式（Low-Frequency Action Source Imprinting），揭示生成控制中高度稳定但语义无关的频率耦合问题。

---

## 问题背景

### 要解决的问题

现有[[World Action Model|WAM]]方法（如机器人操控领域的 [[Fast-WAM]]、[[AdaWAM]] 等）主要面向机器人连续动作空间，缺乏对视频游戏场景的支持。视频游戏有其独特挑战：
- **异构原生控制**：键盘离散按键 + 鼠标连续位移 + GUI 点击三种交互模式混合
- **长时程交互**：游戏任务往往需要数百步甚至更长的持续交互
- **闭环自主性**：模型需在无人类干预下持续感知-规划-执行

### 现有方法的局限

- 多模态游戏 Agent（Game-TARS、OpenHA）依赖 LLM 语言推理，无法直接生成原生键鼠动作
- 传统[[World Model|世界模型]]（Genie、GameGen-O 等）仅做视频预测，不生成可执行动作
- 已有 WAM 不支持 GUI 与游戏控制的动态路由

### 本文的动机

将[[Flow Matching|流匹配]]生成框架与块因果自回归结构结合，同时建模未来视觉观测与原生动作，实现真正意义上的游戏世界-动作联合建模。

---

## 方法详解

### 模型架构

![Figure 1: GameWAM 系统概览](https://arxiv.org/html/2608.26200v1/gamewam_overview.png)

GameWAM 采用**并行双流 [[Diffusion Transformer|DiT]]** 架构：
- **输入**: 任务指令 $\ell$ + 历史视觉观测 + 当前观测 $o_t$ + 控制状态
- **Video DiT**: 去噪未来视觉帧序列 $\mathcal{V}_j$（[[Flow Matching|流匹配]]目标）
- **Action DiT**: 去噪原生动作序列 $\mathcal{A}_j$（键盘/鼠标/GUI 混合）
- **块因果交互**: 两路 DiT 通过 Block-Causal Attention 共享历史前缀 $\mathcal{P}_{c,k}$
- **输出**: 可执行的原生键鼠动作序列（每个周期执行前 $E$ 步）

### 核心模块

#### 模块1: Block-Causal World–Action Modeling

**设计动机**: 利用[[Causal Attention|因果注意力]]保证训练与推理语义一致——只有真正执行过的动作产生的观测才能作为干净因果上下文。

**具体实现**:
- 每个 Block $B_k$ 包含一个预测视觉块 $\mathcal{V}_k$ 和对应动作块 $\mathcal{A}_k$
- 训练时条件上下文 $\Gamma_{c,k} = (C_{c,k}, \ell, H_c, s_{c,k})$，其中 $H_c$ 为跨周期持久历史，$C_{c,k}$ 为周期内已实现的观测
- 模态解耦可见性掩码（默认）确保 Video DiT 和 Action DiT 各自独立访问前缀，避免未来视频信息泄露给动作流

#### 模块2: 原生动作路由（Native Action Routing）

**设计动机**: 游戏中 GUI 操作（如打开背包、点击菜单）与游戏操控（WASD 移动、视角转动）语义差异巨大，需动态路由。

**具体实现**:
- 每个动作时刻 $\tau$ 预测路由 logit $\rho_\tau$，由 BCE 损失监督
- 推理时：$\hat{r}_\tau = [\text{sigmoid}(\rho_\tau) > 1/2]$，根据预测路由选择游戏或 GUI 动作分支
- 各分支有独立的归一化参数和有效性掩码

#### 模块3: 层次化历史（Hierarchical History）

**设计动机**: 视频游戏需要数百步以上的长期记忆（如"记得之前挖了多少钻石"），单靠 KV Cache 无法胜任。

**具体实现**:
- **近期缓冲** $R_c$：FIFO 队列，保存最近 $K_R$ 个 segment 的 Conv3D 压缩特征
- **长程记忆槽** $M_c$：$L$ 个时间尺度 × $K_M$ 个槽位，通过门控注意力池化更新
- **时序编码** $\eta_c(t)$：编码 segment 在周期内的相对位置、时长信息
- 历史表示：$H_c = [M_c; R_c]$，跨周期持久存在

---

## 关键公式

### 公式1: [[Block-Causal Conditioning|块因果联合分布]]

$$
p_\theta(B_{k:k+R-1} \mid \Gamma_{c,k}) = \prod_{r=0}^{R-1} p_\theta(B_{k+r} \mid \Gamma_{c,k}, B_{k:k+r-1})
$$

**含义**: WAM 的块级自回归分解——在给定条件上下文 $\Gamma_{c,k}$ 和已生成块的条件下，依次生成每个预测块。

**符号说明**:
- $B_k$: 第 $k$ 个 Block（含视觉帧 $\mathcal{V}_k$ 和动作 $\mathcal{A}_k$）
- $\Gamma_{c,k}$: 条件上下文（历史 $H_c$、周期内实现观测 $C_{c,k}$、语言指令 $\ell$）
- $R$: 每次预测的 Block 数（即预测窗口）

### 公式2: [[Flow Matching|流匹配]]前向过程

$$
X_{\sigma_m} = (1 - \sigma_m) X_0^m + \sigma_m \epsilon_m, \quad U_m = \epsilon_m - X_0^m
$$

**含义**: 对视觉/动作变量的噪声插值（线性流），目标速度场 $U_m$ 指向从噪声到数据的方向。

**符号说明**:
- $X_0^m$: 干净目标（视频帧或动作序列）
- $\sigma_m \sim \mathcal{U}(0,1)$: 噪声水平（均匀采样）
- $\epsilon_m \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $U_m$: 速度场目标，用于[[Flow Matching|流匹配]]回归

### 公式3: [[Native Action Routing|原生动作路由]]

$$
\hat{r}_\tau = [\text{sigmoid}(\rho_\tau) > 1/2], \quad \hat{U}_\tau^a = (1-\hat{r}_\tau)\hat{U}_\tau^{game} + \hat{r}_\tau \hat{U}_\tau^{gui}
$$

**含义**: 根据预测路由 $\hat{r}_\tau$ 动态选择游戏控制或 GUI 控制的动作流速度场，实现异构控制的软路由。

**符号说明**:
- $\rho_\tau$: 路由 logit（每动作时刻独立预测）
- $\hat{U}_\tau^{game}$: 游戏控制分支的预测速度场
- $\hat{U}_\tau^{gui}$: GUI 控制分支的预测速度场

### 公式4: [[Block-Cycle Control|预测-执行解耦]]

$$
\hat{A}_{c,k}^{plan} \in \mathbb{R}^{P \times d_a}, \quad E < P, \quad \hat{A}_{c,k}^{exec} = \hat{A}_{c,k}^{plan}[1:E]
$$

**含义**: 模型预测 $P$ 步动作但只提交执行前 $E$ 步，随后从新观测出发重新规划，实现预测滚动（Receding Horizon）控制。

**符号说明**:
- $P$: 预测步数（超越执行的前瞻窗口）
- $E$: 执行步数（$E < P$，提交给环境的动作数）
- $d_a$: 动作维度

### 公式5: [[Hierarchical Memory|长程记忆门控更新]]

$$
\tilde{M}_{c} = \text{Attn}(M_c, \bar{S}_c, \bar{S}_c), \quad M_{c+1} = \text{LN}[M_c + g_c \odot (\tilde{M}_c - M_c)]
$$

**含义**: 通过对近期 segment 特征做注意力池化后，以门控残差方式更新长程记忆槽，门控系数 $g_c$ 控制更新幅度。

**符号说明**:
- $M_c$: 当前周期的长程记忆槽
- $\bar{S}_c$: 近期 segment 特征（来自近期缓冲）
- $g_c$: 门控系数（由多时间尺度先验 $\alpha_\ell(\Delta)$ 初始化）
- $\text{LN}$: Layer Normalization

### 公式6: [[Training Objective|总训练目标]]

$$
\mathcal{L} = \lambda_v \mathcal{L}_v + \lambda_a \mathcal{L}_a + \lambda_m \mathcal{L}_{mode} + \lambda_h \mathcal{L}_{hist}
$$

**含义**: 四项损失的加权和——视频流损失、动作流损失、路由监督损失、历史预测损失。

**符号说明**:
- $\mathcal{L}_v = \mathbb{E}[w_v(\sigma_v)\|\hat{U}_v - U_v\|_2^2]$: 视频[[Flow Matching|流匹配]]损失，含噪声水平加权
- $\mathcal{L}_a$: 动作坐标流损失（连续 + 离散分量）
- $\mathcal{L}_{mode} = \sum_\tau \nu_\tau \text{BCE}(\rho_\tau, r_\tau^*) / \sum \nu_\tau$: 路由 BCE 损失（按有效性掩码）
- $\mathcal{L}_{hist}$: 历史预测正则损失（余弦 + 均方误差）

### 公式7: [[DCT|DCT 频率分解]]（LASI 分析）

$$
\tilde{X}_0 = C X_0, \quad \tilde{Z} = C Z, \quad C^T C = I
$$

**含义**: 对动作序列和噪声分别做 DCT 变换，在频率域分析低频分量对生成动作的影响，揭示 LASI 机制。

**符号说明**:
- $C$: 离散余弦变换矩阵（正交阵）
- $\tilde{X}_0$: 动作序列的 DCT 系数（低频分量主导动作模式）
- $\tilde{Z}$: 噪声的 DCT 系数（低频分量具有"源印记"效应）

---

## 关键图表

### Figure 1: GameWAM 系统概览

![Figure 1: Overview](https://arxiv.org/html/2608.26200v1/gamewam_overview.png)

**说明**: GameWAM 整体流程——从游戏观测帧出发，通过[[Block-Causal Conditioning|块因果条件化]]的并行 Video/Action DiT，联合预测未来帧与原生键鼠动作，实现闭环游戏控制。

### Figure 2: 模型架构（训练与推理）

![Figure 2: Architecture](https://arxiv.org/html/2608.26200v1/gamewam_overall_framework.png)

**说明**: 并行 Video DiT 与 Action DiT 共享历史前缀 $\mathcal{P}_{c,k}$，通过模态解耦可见性掩码进行块内交互。动作流分游戏/GUI 两个分支，路由器实时决定执行哪套控制流。

### Figure 3: 块循环控制与层次历史

![Figure 3: Block-Cycle Control](https://arxiv.org/html/2608.26200v1/gamewam_schedule.png)

**说明**: 展示"预测 $P$、执行 $E$"的滚动规划机制。跨周期边界时近期缓冲和长程记忆槽持续保留，周期内 KV Cache 重置，实现长程自主游戏。

### Figure 4: ViZDoom 评测结果

![Figure 4: ViZDoom Results](https://yunncheng.github.io/GameWAM/assets/images/vizdoom-results.webp)

**说明**: GameWAM 在四张 FPS 地图（Battle、Battle2、Defend Center、Defend Line）上与多模态 Agent 的平均奖励对比。GameWAM 在所有场景下均达到竞争性或领先水平（50 episodes/地图）。

### Figure 5: LASI 受控实验证据

![Figure 5: LASI Causal Evidence](https://yunncheng.github.io/GameWAM/assets/images/lasi-causal-evidence.webp)

**说明**: 三组受控实验：固定条件关联（Yaw DCT0 相关系数 $r = 0.890$）、低频源替换（94.8% 试验跟随捐赠者低频）、低频置零（消除 99.25% 相关输出方差），系统量化 LASI 效应。

### Figure 6: 模态解耦可见性掩码

![Figure 6: Decoupled Mask](https://arxiv.org/html/2608.26200v1/gamewam_mask_decoupled.png)

**说明**: 默认的模态解耦配置——Video DiT 和 Action DiT 各自可见历史前缀，但周期内当前块视觉/动作变量相互不可见，避免信息泄露。

### Figure 7: 联合耦合可见性掩码（对比配置）

![Figure 7: Joint Coupled Mask](https://arxiv.org/html/2608.26200v1/gamewam_mask_coupled.png)

**说明**: 联合耦合的替代设计——当前块内视觉和动作变量共享注意力上下文，允许两个模态在去噪过程中交换信息（消融实验中评估此配置）。

### Table 1: MCU 基准主要结果（> 800 任务）

| Method | Steps (Emb) | ASR Mini (Emb) | ASR All (Emb) | Steps (GUI) | ASR Mini (GUI) | ASR All (GUI) | Avg ASR Mini | Avg ASR All |
|--------|-------------|----------------|---------------|-------------|----------------|---------------|--------------|-------------|
| OpenHA | 287 | 37.0±15.9 | 30.1±13.9 | 314 | 33.3±13.3 | 32.5±9.2 | — | — |
| Game-TARS | 373 | — | 50.4±20.7 | 406 | — | 39.1±27.5 | — | — |
| **GameWAM** | **138** | **70.0±32.2** | **47.5±36.0** | **155** | **43.0±32.9** | **60.0±38.6** | **50.7** | **46.6** |

**关键发现**: GameWAM 在 Embodied 任务上 ASR Mini 达 70.0%，远超 OpenHA（37.0%）；交互步数仅为竞品的 1/3—1/2，效率显著更高。

### Table 2: 消融实验（MCU Mini）

| 配置 | Embodied | GUI | Combat | 平均 |
|------|----------|-----|--------|------|
| Action-only 监督 | 60.0±27.6 | 34.0±35.8 | 13.0±13.5 | 35.7 |
| 更粗时间采样 | 60.0±29.7 | 34.0±40.0 | 16.0±36.7 | 36.7 |
| 无事件锚点采样 | 64.0±28.7 | 33.0±30.0 | 17.0±14.9 | 38.0 |
| 统一动作分布 | 59.0±25.5 | 35.0±27.7 | 21.0±29.1 | 38.3 |
| P=E（无前瞻） | 63.0±33.8 | 32.0±36.8 | 29.0±28.1 | 41.3 |
| 无跨周期历史 | 75.0±30.1 | 31.0±29.8 | 34.0±31.0 | 46.7 |
| **Full GameWAM** | **70.0±32.2** | **43.0±32.9** | **39.0±25.9** | **50.7** |

**关键发现**: 移除视频监督损失（Action-only）导致平均下降 15 个百分点（最大影响 Combat 任务）；预测-执行解耦（P>E）对 GUI 和 Combat 任务尤为重要。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| VPT 轨迹（Minecraft） | 大规模自然交互 | 键鼠录制、涵盖广泛游戏行为 | 预训练 |
| 事件锚点 VPT（Event-Anchored） | 定制采样 | 任务事件密集采样 + 稀疏背景 | 提升任务密度 |
| 脚本化 GUI 轨迹 | 补充 | 覆盖界面交互场景 | GUI 能力强化 |
| MCU Benchmark | >800 任务，3 类 | Embodied / GUI / Combat | 主要评测 |
| ViZDoom | 4 张地图，50 eps/图 | FPS 射击游戏 | 跨游戏泛化评测 |

### 实现细节

- **架构**: 并行 Video DiT + Action DiT，共享 Block-Causal 注意力前缀
- **流匹配**: 线性插值，噪声水平均匀采样 $\sigma \sim \mathcal{U}(0,1)$
- **预测/执行比**: $P > E$（具体值见附录 C）
- **记忆配置**: $L$ 层时间尺度，每层 $K_M$ 个槽位；近期缓冲 $K_R$ 个 segment
- **动作归一化**: 游戏/GUI 控制各有独立归一化参数和有效性掩码
- **推理策略**: 动作推理（不做视频去噪）+ 每个规划步重采样噪声源（缓解 LASI）

### LASI 实验发现

Low-Frequency Action Source Imprinting 是本文发现的新型失效模式：

| 实验 | 指标 | 结果 |
|------|------|------|
| 固定条件下 Yaw DCT0 相关性 | Pearson $r$ | 0.890 |
| 低频源替换后跟随率 | % trials following donor | 94.8% |
| 低频置零后方差消除 | % variance removed | 99.25% |

缓解策略：推理时每步重采样独立噪声源（而非复用前一步噪声），可显著降低 LASI 效应。

---

## 批判性思考

### 优点

1. **范式创新**: 首次将 WAM 范式完整应用于视频游戏原生控制，验证了联合视觉-动作生成在游戏智能体中的可行性
2. **实用性强**: 直接生成键鼠事件，无需额外视觉-语言转译层，降低系统延迟
3. **LASI 发现有价值**: 揭示生成式控制中普遍存在但此前未被识别的低频噪声源依赖问题，对整个生成控制领域有借鉴意义

### 局限性

1. **泛化范围有限**: 实验仅涵盖 Minecraft 和 ViZDoom 两类游戏，对其他类型游戏（如 RTS、RPG）的泛化能力未验证
2. **计算开销**: 并行双流 DiT + 层次记忆的结构相比单流模型计算量更大，实时性存在挑战
3. **LASI 根本解决**: 重采样噪声源只是缓解手段，LASI 的根本原因（频率域的隐式耦合）仍未从架构层面消除

### 潜在改进方向

1. 探索更高效的单流视频-动作联合建模（减少双 DiT 开销）
2. 引入课程学习（Curriculum Learning）以在更多类型游戏上泛化
3. 从[[DCT|DCT 频率域]]设计正则化方法，从根本上抑制 LASI

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（附录详细描述超参数和数据构建流程）
- [x] 数据集可获取（MCU、ViZDoom 均为公开基准）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: WAM 范式的代表性工作，GameWAM 沿用其[[Flow Matching|流匹配]]框架
- [[World Action Model]]: 世界动作模型的通用范式定义
- [[Diffusion Transformer]]: Video/Action DiT 的基础架构

### 对比

- [[Game-TARS]]: 基于 LLM 推理的游戏 Agent，GameWAM 在步数效率和 Embodied 任务上均更优
- [[OpenHA]]: 另一多模态游戏 Agent 基线

### 方法相关

- [[Block-Causal Conditioning]]: GameWAM 的核心时序建模机制
- [[Flow Matching]]: 视觉和动作的联合生成目标
- [[Hierarchical Memory]]: 长程跨周期记忆设计
- [[DCT]]: LASI 分析中用于频率分解的工具

### 硬件/数据相关

- [[Minecraft Universe]]: 主要评测基准（>800 任务）
- [[ViZDoom]]: FPS 游戏跨域泛化评测

---

## 速查卡片

> [!summary] GameWAM: A World Action Model for Video Games
> - **核心**: 首个视频游戏闭环 WAM，支持原生键鼠/GUI 混合控制
> - **方法**: 并行 Video/Action DiT + 块因果条件化 + 块循环控制 + 层次历史
> - **结果**: MCU 平均 ASR 50.7%，ViZDoom 四图竞争性领先；发现 LASI 生成失效现象
> - **代码**: 未公开（项目页 https://yunncheng.github.io/GameWAM/）

---

*笔记创建时间: 2026-08-29*

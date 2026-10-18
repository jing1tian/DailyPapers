---
title: "Multics: Generative Multi-Player World Model with Hierarchical Communication"
method_name: "Multics"
authors: [Kaiyi Huang, Wangbo Yu, Yukun Huang, Jiwen Yu, Kunhao Liu, Donglin Yang, Wenbo Hu, Ying Shan, Xihui Liu]
year: 2026
venue: arXiv
tags: [world-model, video-generation, multi-player, hierarchical-communication, game-world-model, attention-mechanism]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.17201v1/
created: 2026-10-18
---

# 论文笔记：Multics: Generative Multi-Player World Model with Hierarchical Communication

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong; ARC Lab, Tencent IEG |
| 日期 | October 2026 |
| 项目主页 | [https://karine-huang.github.io/Multics/](https://karine-huang.github.io/Multics/) |
| 对比基线 | [[Solaris]], [[γ-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.17201) / Code: N/A |

---

## 一句话总结

> Multics 提出分层玩家-组-全局通信框架，通过参数共享的组令牌和基于相机轨迹的动态分组，在保持线性通信开销的同时实现多玩家视频世界模型的跨玩家一致性生成与玩家数泛化。

---

## 核心贡献

1. **分层通信框架**: 设计玩家层→组层→全局层的三级 [[自注意力机制|Self-Attention]] 通信拓扑，避免全连接的二次方代价与中心化方法的跨事件干扰
2. **参数共享组令牌**: [[Group Token|组令牌]] 在多个组槽间共享初始参数 $\mathbf{E}^L$，每个槽通过与所属玩家的注意力操作形成不同表示，使模型可无缝扩展到不同玩家数量
3. **预算约束动态分组**: 基于玩家间相机轨迹计算相关性矩阵 $\mathbf{R}^{(t)}$，在保持与中心化方法等价计算量的约束下，通过预算感知的贪心算法决定哪些玩家进行组内通信

---

## 问题背景

### 要解决的问题

多玩家视频[[世界模型 (World Model)|世界模型]]面临两大挑战：(1) **跨玩家一致性生成** — 多个玩家在共享场景中的行为与视角必须互相一致；(2) **玩家数量的可扩展性** — 模型需在推理阶段泛化到训练时未见过的玩家数量（如训练4人、推理10人）。

### 现有方法的局限

- **全连接方法（Solaris）**: 所有玩家令牌互相注意，通信开销随玩家数 $N$ 呈 $O(N^2)$ 增长，不可扩展
- **中心化方法（γ-World, MultiWorld）**: 所有玩家仅通过共享全局 hub 令牌 $\mathbf{h}^G$ 通信，不同场景事件（如两侧对战）的信息混入同一全局表示，导致跨事件干扰

### 本文的动机

在多玩家场景中，并非所有玩家都彼此相关——物理上相近或互相可见的玩家才需要细粒度信息交换。因此，将通信分为"组内直接通信 + 跨组通过全局令牌间接通信"的分层结构，既能减少无关玩家间的干扰，又能在线性代价内捕捉局部强相关性。

---

## 方法详解

### 模型架构

Multics 基于 [[Wan2.2-TI2V-5B]]（Wan2.2 TI2V 5B）[[DiT（扩散变换器）|DiT]] 骨干，在每个自注意力层中插入额外的**组令牌** $\mathbf{h}^L$ 和**全局令牌** $\mathbf{h}^G$：

- **输入**: 初始图像 $I_i \in \mathbb{R}^{c \times h \times w}$（每个玩家） + 文本提示 $P_i$ + 动作序列 $A_i$ + 相机轨迹 $C_i$
- **Backbone**: Wan2.2 TI2V 5B [[DiT（扩散变换器）|DiT]]（大部分权重冻结）
- **核心模块**: [[Group Token|组令牌]] $\mathbf{h}^L$ 聚合局部组内信息，[[自注意力机制|全局令牌]] $\mathbf{h}^G$ 聚合跨组信息
- **输出**: 每个玩家的视频（165帧，352×640）
- **总参数**: 约5B（Wan2.2-TI2V-5B 基础上微调）

### 核心模块

#### 模块1: 分层通信 (Hierarchical Communication)

**设计动机**: 利用 [[自注意力机制|Self-Attention]] 掩码实现可控的通信拓扑，避免全连接的二次方代价

**四种通信拓扑对比**:

- **全连接（Solaris）**: 所有玩家令牌互相关注 $\mathbf{x}_i \leftrightarrow \mathbf{x}_j$，$\forall i,j$
- **中心化（γ-World）**: 仅玩家与全局 hub 通信 $\mathbf{x}_i \leftrightarrow \mathbf{h}^G$
- **基于组**: 同组内玩家互相通信，加上与全局令牌通信 $\mathbf{x}_i \leftrightarrow \mathbf{h}^G$
- **分层（Multics）**: 玩家 → 组令牌 → 全局令牌 $\mathbf{x}_i \leftrightarrow \mathbf{h}^L_{g(i)} \leftrightarrow \mathbf{h}^G$，跨组信息需经由全局令牌传递

**序列组织**: 将所有玩家令牌、组令牌、全局令牌拼接为单一序列，通过注意力掩码控制通信路径：

$$\mathbf{x}_{\mathrm{hier}} = [\mathbf{x}_1, \ldots, \mathbf{x}_N, \mathbf{h}^L, \mathbf{h}^G]_{\mathrm{seq}}$$

#### 模块2: 参数共享组令牌 (Shared Group Tokens)

**设计动机**: 使模型对玩家数量的变化具有不变性，无需增加新参数即可处理更多玩家

**具体实现**:
- 定义 $M = \lceil \log_2 N \rceil$ 个组槽（4玩家时 $M=2$，6玩家时 $M=3$）
- 所有组槽共享初始参数 $\mathbf{E}^L \in \mathbb{R}^{k_L \times d}$（$k_L = 5$）
- 每个槽独立执行与其分配玩家的注意力计算，形成不同表示
- 推理时玩家数超过训练时，只需增加槽数（无需新参数）

#### 模块3: 预算约束动态分组 (Budget-Constrained Dynamic Grouping)

**设计动机**: 基于相机轨迹的相关性自动决定哪些玩家应归入同一组，在控制计算代价的前提下最大化有效通信

**具体实现**:
- 每 48 帧更新一次分组策略
- 计算玩家对相关性矩阵 $\mathbf{R}^{(t)}$（取场景相关性与互视相关性的最大值）
- 使用贪心算法在组容量 $K$ 约束内选择最大化覆盖率的分组
- 约束：每个玩家至多属于 2 个组，任意两组至多共享 1 名玩家（锚节点）

#### 模块4: 动作与相机条件注入

**动作条件化（参考 GameFactory）**:
- 动作 $A_i = (D_i, U_i)$：$D_i \in \mathbb{R}^{(f-1) \times 23}$ 离散控制 + $U_i \in \mathbb{R}^{(f-1) \times 2}$ 相对偏航/俯仰
- 离散控制通过[[Cross-Attention|跨注意力]]注入；鼠标控制通过时序模块融合
- 每帧使用因果窗口内 $r \cdot w_c$ 步动作条件化

**相机条件化（参考 ReCamMaster）**:
- 所有玩家相机位姿统一于第一玩家初始帧的坐标系：$\mathbf{C} \in \mathbb{R}^{N \times f \times 3 \times 4}$
- 可学习适配器将相机特征注入玩家的 Query 和 Key
- 组令牌和全局令牌使用零相机特征

---

## 关键公式

### 公式1: [[自注意力机制|分层通信序列]]

$$
\mathbf{x}_{\mathrm{hier}} = [\mathbf{x}_1, \ldots, \mathbf{x}_N, \mathbf{h}^L, \mathbf{h}^G]_{\mathrm{seq}}, \quad \mathbf{x}_i \leftrightarrow \mathbf{h}^L_{g(i)} \leftrightarrow \mathbf{h}^G
$$

**含义**: Multics 的注意力序列结构，玩家令牌只与自身所在组令牌通信，组令牌再与全局令牌通信

**符号说明**:
- $\mathbf{x}_i \in \mathbb{R}^{f_z \times L \times d}$: 玩家 $i$ 的视觉令牌
- $\mathbf{h}^L \in \mathbb{R}^{f_z \times M \times k_L \times d}$: 所有组令牌（$M$ 个组，每组 $k_L=5$ 个令牌）
- $\mathbf{h}^G \in \mathbb{R}^{f_z \times k_G \times d}$: 全局令牌（$k_G=6$）
- $g(i)$: 玩家 $i$ 所属组的索引

### 公式2: [[Group Token|组槽数量]]

$$
M = \left\lceil \log_2 N \right\rceil
$$

**含义**: 组槽数随玩家数对数增长，确保通信效率的同时允许不同玩家数的泛化

**符号说明**:
- $N$: 玩家总数（4玩家时 $M=2$，6玩家时 $M=3$，10玩家时 $M=4$）

### 公式3: [[Dynamic Grouping|玩家对相关性矩阵]]

$$
R^{(t)}_{ij} = \max\left(R^{(t),\mathrm{scene}}_{ij},\ R^{(t),\mathrm{mutual}}_{ij}\right)
$$

**含义**: 综合场景邻近性和互视可见性，计算每对玩家的通信相关性

**符号说明**:
- $R^{(t)} \in [0,1]^{N \times N}$: 时刻 $t$ 的对称相关性矩阵
- $R^{(t),\mathrm{scene}}_{ij}$: 基于相机位置距离和视角端点距离的场景相关性
- $R^{(t),\mathrm{mutual}}_{ij}$: 互视相关性（见下方公式）

### 公式4: [[Dynamic Grouping|场景相关性子项]]

$$
r^{\mathrm{prox}}_{ij}(\tau) = \exp\!\left(-\frac{\|\mathbf{p}_i(\tau) - \mathbf{p}_j(\tau)\|_2}{\sigma_p}\right), \quad r^{\mathrm{view}}_{ij}(\tau) = \exp\!\left(-\frac{\|\mathbf{e}_i(\tau) - \mathbf{e}_j(\tau)\|_2}{\sigma_v}\right)
$$

**含义**: 分别衡量玩家相机位置的接近程度，以及视线方向端点的接近程度

**符号说明**:
- $\mathbf{p}_i(\tau)$: 时刻 $\tau$ 时玩家 $i$ 的相机位置
- $\mathbf{e}_i(\tau) = \mathbf{p}_i(\tau) + \ell \mathbf{v}_i(\tau)$: 视线方向端点（$\mathbf{v}_i(\tau)$ 为朝向，$\ell$ 为投影距离）
- $\sigma_p, \sigma_v$: 带宽参数

### 公式5: [[Dynamic Grouping|互视相关性]]

$$
r^{\mathrm{mutual}}_{ij}(\tau) = \sqrt{Q_{i \to j}(\tau) \cdot Q_{j \to i}(\tau)}
$$

**含义**: 玩家 $i$ 是否能在视野中看到玩家 $j$，以及 $j$ 是否能看到 $i$，取几何均值作为互视相关性

**符号说明**:
- $Q_{i \to j}(\tau)$: 方向可见性分数（玩家 $j$ 在玩家 $i$ 视锥内的程度）
- 对称形式保证 $R^{(t)}$ 为对称矩阵

### 公式6: [[Budget-Constrained Grouping|中心化通信代价（参考基准）]]

$$
\mathcal{C}_{\mathrm{central}} = A_w \left(2NLk_C + k_C^2\right)
$$

**含义**: 中心化方法（γ-World）的自注意力 FLOP 数，作为 Multics 预算上限

**符号说明**:
- $A_w$: 注意力窗口步数
- $L$: 每玩家空间令牌数
- $k_C$: 中心化 hub 令牌数

### 公式7: [[Budget-Constrained Grouping|分层基础代价与剩余预算]]

$$
\mathcal{C}_{\mathrm{hier\text{-}base}} = A_w\!\left(2NLk_L + Mk_L^2 + 2Mk_Lk_G + k_G^2\right)
$$

$$
\Delta\mathcal{C} = \mathcal{C}_{\mathrm{central}} - \mathcal{C}_{\mathrm{hier\text{-}base}}, \quad B_{\mathrm{pair}} = \left\lfloor \frac{\Delta\mathcal{C}}{B_b L^2} \right\rfloor
$$

**含义**: 分层框架自身的固定代价及可用于组内玩家直接通信的剩余预算 $B_\mathrm{pair}$

**符号说明**:
- $B_b$: 每块的基础代价系数
- $B_\mathrm{pair}$: 可容纳的玩家对数（用于推导组容量 $K$）

### 公式8: [[Budget-Constrained Grouping|组容量]]

$$
K = \min\!\left(N,\ \left\lfloor \frac{1}{2} + \sqrt{\frac{1}{4} + B_{\mathrm{pair}}} \right\rfloor\right)
$$

**含义**: 从预算反推每个组最多可以容纳多少名玩家

**符号说明**:
- $K$: 组内最大玩家数（由二次公式 $\frac{K(K-1)}{2} \leq B_\mathrm{pair}$ 反解而来）

### 公式9: [[Diffusion Forcing|隐式帧数]]

$$
f_z = \frac{f - 1}{r} + 1
$$

**含义**: VAE 时序下采样后的潜在帧数（$r$ 为时序压缩比）

**符号说明**:
- $f$: 原始视频帧数（165帧）
- $r$: VAE 时序压缩比
- $f_z$: 潜在帧数，用于组/全局令牌的时序维度

---

## 关键图表

### Figure 1: Teaser — Multics 系统概览

![Figure 1: Multics Overview](https://arxiv.org/html/2610.17201v1/teaser.png)

**说明**: Multics 将多玩家通信分为玩家层（player-level）、组层（group-level）和全局层（global-level）。相比全连接拓扑（Solaris）和纯中心化拓扑（γ-World），Multics 在保持线性代价的同时减少了跨不相关事件的干扰。

### Figure 2: 训练流程与通信拓扑对比

![Figure 2: Pipeline and Communication Topologies](https://arxiv.org/html/2610.17201v1/pipeline.png)

**说明**: (a) Multics 整体训练管线：以 [[Wan2.2-TI2V-5B]] 为骨干，在自注意力层中插入 [[Group Token|组令牌]] 和全局令牌；(b-d) 四种通信拓扑的注意力掩码对比（全连接/中心化/基于组/分层）。注意冻结层（VAE、文本编码器、FFN）与微调层（自注意力、动作模块、相机适配器、令牌嵌入）的划分。

### Figure 3: 4玩家与6玩家定性对比

![Figure 3: Qualitative Comparison 4 and 6 Players](https://arxiv.org/html/2610.17201v1/qualitative-4_6players.png)

**说明**: 与 Solaris（全连接）和 γ-World（中心化）的定性比较。Multics 在多玩家场景中产生跨视角更一致的场景内容，尤其在两侧独立战场同时存在时，中心化方法产生明显的跨事件混淆。

### Figure 4: 玩家数泛化（4训练→6/8/10推理）

![Figure 4: Generalization to More Players](https://arxiv.org/html/2610.17201v1/4infer_more.png)

**说明**: 仅训练4玩家模型，在6、8、10玩家场景下直接推理。[[Group Token|参数共享组令牌]] 允许模型通过增加槽数来处理更多玩家，无需增加参数或重新训练。Multics 在所有设置下均优于 Solaris 和 γ-World。

### Figure 5: 基于组方法的消融对比

![Figure 5: Group-based Ablation](https://arxiv.org/html/2610.17201v1/group-based.png)

**说明**: 纯基于组的通信方法（组内直接连接，无独立组令牌）在场景重游时产生不一致性：因为直接连接使玩家令牌同时承担生成与通信的双重作用，引入干扰。分层组令牌将两者解耦。

### Figure 6: 计算效率对比

**（图片 URL 未能从 arXiv HTML 中提取）**

**说明**: 不同通信方法随玩家数 $N$ 增长的自注意力 FLOP 对比。Multics 的代价与中心化和基于组方法持平（均随 $N$ 线性增长），而全连接方法呈二次增长。在10玩家时，Multics 需要约 **9.6 TFLOPs** 自注意力代价，全连接方法需 **69.1 TFLOPs**。

### Figure 7: 通信预算对 GeoKPM 的影响

**（图片 URL 未能从 arXiv HTML 中提取）**

**说明**: 将通信预算从中心化代价基准（1.0×）向上或向下调整的消融实验。低于 1.0× 时 [[GeoKPM]] 性能下降，高于 1.0× 时基本稳定，说明当前预算设置已接近效果上限。

### Figure 8: 玩家数泛化相对性能下降

![Figure 8: Player Generalization Metrics](https://arxiv.org/html/2610.17201v1/figures/player-generalization-serif.png)

**说明**: 将4玩家训练模型外推至6玩家时，8个指标上的相对性能下降幅度对比。Multics 在6/8个指标上下降最小，印证其 [[Group Token|参数共享组令牌]] 带来的泛化优势。

### Figure 9: EgoTeam 跨域测试定性结果

![Figure 9: EgoTeam Cross-Domain](https://arxiv.org/html/2610.17201v1/egoteam.png)

**说明**: 在未见过的 EgoTeam 数据集（第一人称团队视频）上的跨域测试。Multics 生成的多玩家视频在跨视角语义一致性上显著优于 γ-World，接近 Solaris 的视觉质量，但在 GeoKPM 和 FVD 上均超过 Solaris。

---

### Table 1: 主要实验结果（4玩家 & 6玩家，训练集评测）

| 设置 | 方法 | GeoKPM↑ | IDM↑ | RotErr↓ | TransErr | CamMC↓ | FID↓ | FVD↓ |
|------|------|---------|------|---------|----------|--------|------|------|
| 4玩家 | Solaris | 48.51% | 93.87 | 2.572 | — | 15.00 | 28.41 | 171.9 |
| 4玩家 | γ-World | 49.42% | 93.98 | 2.728 | — | 15.26 | 29.36 | 200.5 |
| 4玩家 | **Multics** | **54.17%** | **94.17** | **2.434** | — | **13.21** | **26.39** | **162.3** |
| 6玩家 | Solaris | 54.65% | 94.32 | 3.991 | — | 14.19 | 18.83 | 115.1 |
| 6玩家 | γ-World | 52.80% | 94.34 | 3.761 | — | 14.29 | 20.16 | 128.4 |
| 6玩家 | **Multics** | **56.84%** | 94.33 | **3.669** | — | **13.21** | **16.34** | **101.2** |

**关键发现**: Multics 在 [[GeoKPM]]（跨玩家几何一致性）和 FVD（视频质量）上取得最佳，[[IDM|IDM分数]]（动作准确性）持平；6玩家 SSIM 略低于 Solaris（0.4687 vs. 0.4728），为唯一例外。

### Table 2: 分组策略消融（4玩家）

| 分组策略 | GeoKPM↑ | RotErr↓ |
|---------|---------|---------|
| 随机分组 | 46.76% | — |
| 基于组（固定） | 50.71% | — |
| 基于组（无共享参数） | 48.26% | — |
| **动态分组（Multics）** | **54.17%** | **2.434** |

**关键发现**: [[Dynamic Grouping|动态分组]] 相比固定分组提升 +3.46% GeoKPM，证明基于相机轨迹的相关性矩阵 $\mathbf{R}^{(t)}$ 能有效识别需要细粒度通信的玩家对。

### Table 3: 共享 vs 非共享组令牌（4→6玩家泛化）

| 组令牌初始化 | GeoKPM↑（6玩家推理）|
|-----------|---------------------|
| 复制组0（非共享） | ~43.6% |
| 复制组1（非共享） | ~44.2% |
| 平均（非共享） | ~45.0% |
| **共享参数（Multics）** | **54.33%** |

**关键发现**: [[Group Token|参数共享]] 带来约 +9–11% GeoKPM 的提升，说明共享参数使组槽具有置换不变性，是玩家数泛化的关键。

### EgoTeam 跨域测试（Table）

| 方法 | GeoKPM↑ | FID↓ | FVD↓ |
|------|---------|------|------|
| Solaris | 57.65% | — | — |
| γ-World | 52.31% | — | — |
| **Multics** | **58.62%** | — | — |

**说明**: 跨域泛化到 EgoTeam 数据集，Multics 在一致性指标上取得最高 GeoKPM。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OpenCS2 | 10玩家 CS2 比赛录像 | 每样本含10名玩家165帧视频（352×640），随机抽取 $N$ 名玩家 | 训练 + 测试（200样本） |
| EgoTeam | 第一人称团队运动视频 | 跨域，未见场景 | 跨域测试（100样本） |

### 实现细节

- **Backbone**: [[Wan2.2-TI2V-5B]]（Wan2.2 TI2V 5B，约5B参数）
- **冻结层**: VAE、文本编码器、FFN 层及大部分预训练 DiT 权重
- **微调层**: 自注意力、文本跨注意力、动作模块、相机 Q/K 适配器、组/全局令牌嵌入
- **训练策略**: [[Diffusion Forcing|扩散强制]]，因果块（每块3个潜在帧）
  - 阶段一：10K步，7潜在帧注意力窗口
  - 阶段二：25K步，21帧注意力窗口
- **优化器**: [[AdamW]]，学习率 $3 \times 10^{-5}$，bf16 精度
- **Batch Size**: 有效批大小16（16块 GPU）
- **硬件**: 16块 GPU（型号未说明）
- **推理**: 50步去噪，CFG scale = 5.0
- **分组更新频率**: 每48帧

### 可视化结果

- 在4和6玩家设置下，Multics 产生的视频在两侧独立战场同时存在时，双侧场景内容比 γ-World 更一致
- 泛化到8/10玩家时，整体视觉质量保持，[[Group Token|参数共享组令牌]] 无需重训即可适应

---

## 批判性思考

### 优点

1. **框架优雅**: 分层通信从 MARL 领域借鉴思想，优雅地解决了多玩家视频生成的通信代价与干扰问题
2. **参数共享设计巧妙**: 组令牌共享参数实现"训练时固定 $N$，推理时泛化"的无缝扩展，工程上几乎零额外代价
3. **相机驱动分组理论上合理**: 仅用相机轨迹（无语义特征）即可判断玩家相关性，轻量且可解释

### 局限性

1. **游戏场景特化**: 基于射击游戏数据集 OpenCS2 训练，相关性度量依赖 FPS 游戏的视角特性，未必泛化到其他类型场景（如体育、驾驶）
2. **长时序持久状态缺失**: 未处理跨片段的世界状态持续性（玩家位置历史、物品状态），每次生成独立
3. **基础模型伪影**: 继承 Wan2.2 在大幅动态人体运动时的伪影
4. **分组频率硬编码**: 每48帧更新一次分组，高速场景可能不够及时

### 潜在改进方向

1. 将相关性矩阵扩展为可学习的 soft 分组，而非离散的硬分组
2. 引入跨片段的持久状态表示，支持更长时序的多玩家世界建模
3. 在更多游戏类型和真实视频数据上验证跨域泛化能力

### 可复现性评估

- [ ] 代码开源（截至论文提交未开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（训练步数、LR、硬件均有描述）
- [ ] 数据集可获取（OpenCS2 未见公开说明）

---

## 关联笔记

### 基于

- [[Wan2.2-TI2V-5B]]: 基础视频生成骨干模型
- [[Diffusion Forcing]]: 训练策略（因果扩散）
- [[DiT（扩散变换器）]]: 整体 DiT 架构

### 对比

- [[Solaris]]: 全连接多玩家世界模型基线（γ-World 的 FPS 扩展）
- [[γ-World]]: 中心化 hub 通信的多玩家世界模型

### 方法相关

- [[Group Token|组令牌]]: 本文核心设计——参数共享的组级通信令牌
- [[Dynamic Grouping|动态分组]]: 基于相机轨迹的预算约束分组算法
- [[自注意力机制]]: 通过注意力掩码实现可控通信拓扑
- [[GeoKPM]]: 主要跨玩家一致性评估指标
- [[IDM|IDM 分数]]: 动作准确性评估指标

### 硬件/数据相关

- [[Wan2.2-TI2V-5B]]: 5B 视频扩散基础模型

---

## 速查卡片

> [!summary] Multics: Generative Multi-Player World Model with Hierarchical Communication
> - **核心**: 分层玩家-组-全局通信框架，参数共享组令牌 + 相机驱动动态分组
> - **方法**: 在 DiT 自注意力层插入组令牌 $\mathbf{h}^L$（$k_L=5$）+ 全局令牌 $\mathbf{h}^G$（$k_G=6$），通过注意力掩码控制拓扑；$M=\lceil\log_2 N\rceil$ 组槽共享参数 $\mathbf{E}^L$
> - **结果**: 4玩家 GeoKPM 54.17%（vs. Solaris 48.51%）；10玩家推理时注意力代价 9.6 vs. 69.1 TFLOPs
> - **代码**: N/A（[项目主页](https://karine-huang.github.io/Multics/)）

---

*笔记创建时间: 2026-10-18*

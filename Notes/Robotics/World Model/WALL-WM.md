---
title: "WALL-WM: Carving World Action Modeling at the Event Joints"
method_name: "WALL-WM"
authors: [Shalfun Li, Victor Yao, Charles Yang, Truth Qu, Regis Cheng, Ryan Yu, Howard Lu, Newton Von, Vincent Chen, Yohann Tang, Maeve Zhang, Ellie Ma, Gody Li, Sage Yang, Lorien Shu, J.W. Gao, Ethan Chen, Colin Ye, Yu Sun, Elise Mon, PS Zhang, Neo Li, Lily Li, James Wang, Ping Yang, Chris Pan, Lucy Liang, Hang Su, Roy Gan, Hao Wang, Qian Wang]
year: 2026
venue: arXiv
tags: [world-action-model, vla, embodied-ai, diffusion-policy, video-generation, event-grounded-learning, robot-manipulation]
zotero_collection: 3-Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.01955
created: 2026-06-03
---

# 论文笔记：WALL-WM: Carving World Action Modeling at the Event Joints

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | X-Square Robot |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[视觉语言动作模型]]、[[扩散世界模型]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.01955) / [Code](https://github.com/X-Square-Robot/wall-x) |

---

## 一句话总结

> WALL-WM 以"语义一致的动作事件"为原子学习单元，通过事件对齐的视频-动作联合预训练解决语言描述、视觉动态与控制时间尺度之间的三重错位，实现了大规模真实机器人泛化的最优性能。

---

## 核心贡献

1. **事件中心世界动作建模**: 将固定时钟分块替换为"语义动作事件"作为训练原子单元，消除语言描述与视觉动态之间的结构性错位
2. **多视图几何一致性建模**: 通过视锥注意力遮罩（Sight-Cone Masking）和管状补丁遮罩（Tube Patch Masking）增强跨视图几何一致性，支持无标定多相机输入
3. **楼梯式潜空间推理（Staircase Latent CoT）**: 将 Transformer 深度分区，实现并行潜链式思考生成，同时保留视觉-语言分层先验

---

## 问题背景

### 要解决的问题

现有 [[视觉语言动作模型|VLA]] 方法将视频和动作切分为**固定长度块（fixed-length chunks）**，这造成三个层面的错位：
1. **语言描述** 以任务/子任务粒度描述，与定长块不对应
2. **视觉动态** 在语义事件边界处发生自然变化，而非均匀切分处
3. **控制时间尺度** 需要在接触、抬起、放置等关键时刻做出不同精度的决策

### 现有方法的局限

- [[扩散策略|Diffusion Policy]] 等方法使用固定时钟切分，导致块内部出现语义断裂
- 基于 [[流匹配]] 的视频预训练模型（如 Wan）针对单视图设计，无法直接用于多相机机器人部署
- [[视觉语言动作模型|VLA]] 的链式推理路径通常是串行的，引入了不必要的推理延迟

### 本文的动机

**"固定块由时钟切分；语义事件由具身动态切分。"** 作者认为，若将语义一致的动作事件作为预训练的原子单元，模型可以同时在语言理解、视觉预测和控制执行三个维度上建立自然对齐。

---

## 方法详解

### 模型架构

![[WALL-WM_fig3_p7.png]]

WALL-WM 采用**事件对齐的视频-动作联合预训练**架构，分为三大模块：

- **输入**: 多视图 RGB 视频帧 + 自然语言指令 + 机器人末端执行器状态
- **视频骨干**: 基于 [[流匹配]] 的 [[视频扩散模型|DiT（Diffusion Transformer）]]，继承 Wan text-to-video 权重
- **核心模块**: [[多视图几何建模|多视图视觉世界事件建模]] + [[事件中心动作动态建模]] + [[Staircase Latent Decoding|楼梯式语言引导推理]]
- **输出**: 变长事件视频预测 + 对应时间段的机器人动作序列
- **推理模式**: 事件模式（variable-length）/ 统一模式（fixed-length chunk with staircase reasoning）

### 核心模块

#### 模块1：多视图视觉世界事件建模（Multi-View Visual World Events Modeling）

**设计动机**: 机器人部署通常需要多相机，但主流[[视频扩散模型]]（如 Wan DiT）仅针对单视图设计，直接扩展会破坏几何一致性。

**具体实现**:

- **[[多视图适配|Multi-View Adaptation]]**: 在单视图 Wan DiT 的每个 Transformer 块中，插入带**零初始化投影**的跨视图自注意力：

$$
\mathbf{h}_i^v \leftarrow \mathbf{h}_i^v + g_i W_{\text{view}} \text{CrossViewAttn}_i(\mathbf{h}_i^v)
$$

  $W_{\text{view}}$ 初始化为 0，保证训练初期不破坏单视图先验；$g_i$ 是可学习门控。

- **[[Camera RoPE]]**: 可学习旋转位置编码，提供无需外参标定的相机身份表示

- **[[Cross-View Geometric Masking|跨视图几何遮罩]]**: 两种互补机制：

  1. **视锥注意力遮罩（Sight-Cone Masking）**: 仅允许视野几何相交的像素块之间互相注意：

     像素 $u$ 的视锥射线为：

     $$
     \mathbf{p}(u, t) = \mathbf{p}_0(u) + t\hat{\mathbf{v}}(u)
     $$

     两射线最近点时刻：

     $$
     (t_1, t_2) = \arg\min_{t_1, t_2} \|\mathbf{p}(u, t_1) - \mathbf{p}(u', t_2)\|_2
     $$

     深度范围内截断：

     $$
     \hat{t}_1 = \text{clamp}(t_1, d_{\min}, d_{\max})
     $$

     $$
     \hat{t}_2 = \text{clamp}\left(\arg\min_{t_2} \|\mathbf{p}(u, \hat{t}_1) - \mathbf{p}(u', t_2)\|_2, d_{\min}, d_{\max}\right)
     $$

     视锥相交判断：

     $$
     \mathcal{C}(u) \cap \mathcal{C}(u') \Longleftrightarrow \|\mathbf{p}(u, \hat{t}_1) - \mathbf{p}(u', \hat{t}_2)\|_2 \leq \hat{t}_1 \gamma(u) + \hat{t}_2 \gamma(u')
     $$

     遮罩矩阵：

     $$
     \mathcal{M}_{sc}[u, u'] = 1 \Longleftrightarrow \mathcal{C}(u) \cap \mathcal{C}(u') \neq \emptyset
     $$

  2. **管状补丁遮罩（Tube Patch Masking）**: 对时序管状区域施加更高损失权重，强化自我一致性

- **视频[[流匹配]]损失（默认）**:

  $$
  \mathcal{L}_v = w_v(t^v) \|\hat{\mathbf{C}}^v - \mathbf{C}^{v\star}\|^2
  $$

  带管状遮罩加权版本：

  $$
  \mathcal{L}_v = w_v(t^v) \left( \sum_{u \notin \mathcal{T}} \|\hat{C}^v_u - C^{v\star}_u\|^2 + \lambda_{\text{mask}} \sum_{u \in \mathcal{T}} \|\hat{C}^v_u - C^{v\star}_u\|^2 \right)
  $$

  $\mathcal{T}$ 为管状区域集合，$\lambda_{\text{mask}}$ 为加权系数。

#### 模块2：事件中心动作动态建模（Event-Centric Action Dynamics Modeling）

**设计动机**: 动作序列与视频必须在时间语义上对齐，但两者天然处于不同的时间尺度（帧率 vs. 控制频率）。

**具体实现**:

- **逐层耦合（Layer-Wise Coupling）**: 动作 Transformer 在匹配深度处对视频特征进行[[交叉注意力]]，不修改视频流权重，保留预训练先验

- **时间对齐双嵌入系统**:

  跨视图连接后加时间嵌入：

  $$
  \tilde{h}^v_i = \text{ViewConcat}(\mathbf{h}^v_{\pi(i)}) + E_\tau(\boldsymbol{\tau}^v) + E_{\text{abs}}(\mathbf{t}_{\text{abs}})
  $$

  $E_\tau$：窗口内相对时序索引；$E_{\text{abs}}$：滑动块绝对位置

  事件中心窗口时序索引定义：

  $$
  \tau^v_{(f,h,w)} = f, \quad \tau^0_a = 0, \quad \tau^{1+k}_a = \lfloor k / K_p \rfloor + 1, \quad k \in [0, T_a)
  $$

  观测中心窗口：

  $$
  \mathbf{h} \mathrel{+}= E_\tau(\tau) + E_{\text{abs}}(t_{\text{abs}})
  $$

- **非对称时步映射（Asymmetric 1-to-$N_a$ Timestep Mapping）**: 所有动作噪声步骤均交叉注意到单个锚定视频步骤 $s^\star$：

  $$
  m(j) = s^\star \quad \text{for all } j \in \{0, \ldots, N_a - 1\}
  $$

  每步优化器步骤并行抽取 $K=6$ 个独立动作噪声：

  $$
  t^k_a \sim \text{u.a.r.} \, \Phi_a, \quad k = 1, \ldots, K
  $$

- **动作[[流匹配]]损失**:

  $$
  \mathcal{L}_a = \frac{1}{K} \sum_{k=1}^K \mathcal{L}^k_a, \quad \mathcal{L}^k_a = w(t^k_a) \|\hat{y}^k_a - y^{k\star}_a\|^2
  $$

- **DCT 辅助损失**（接触密集场景）:

  $$
  \mathcal{L}_a^+ = \frac{w_{\text{act}}}{K} \sum_{k=1}^K \mathcal{L}^k_{\text{act}}
  $$

  $$
  \mathcal{L}^k_{\text{act}} = \left(1 - \frac{t^k_a}{T}\right)^2 \sum_{j=0}^{T_a-1} e^{-j/(T_a/4)} \|\text{DCT}(\hat{a}^{(k)}_0)_j - \text{DCT}(a_0)_j\|^2
  $$

  指数衰减权重优先惩罚低频（全局形状）成分误差。

#### 模块3：楼梯式语言引导推理（Staircase Latent Decoding）

**设计动机**: 串行链式思考引入推理延迟；现有方法将语言推理与视觉动态解耦，丢失层次先验。

**具体实现**:

- **VLM 文本条件**（Qwen3.5-9B 骨干，投影头可训练）:

  $$
  \mathbf{c}_\ell = \mathcal{M}_t(\mathcal{P}_Q(\mathbf{H}_q[\mathbf{m}_{\text{txt}}]))
  $$

- **楼梯潜空间 CoT 生成**: 将 Transformer 深度按中继深度 $N_r$ 分区——下层计算共享视觉-语言基础，上层并行生成 $K_c$ 个推理潜状态：

  $$
  \hat{y}_{1:K_c} = \mathcal{F}_{\text{stair}}(x; N_r)
  $$

- **冻结重建监督**（避免推理状态坍塌）: 通过冻结语言模型从潜推理状态重建文本 CoT：

  $$
  P_\phi(r_{1:M_r} \mid \mathbf{z}_{1:K_c})
  $$

  重建损失：

  $$
  \mathcal{L}_{\text{CoT}} = -\sum_{m=1}^{M_r} \log P_\phi(r_m \mid \mathbf{z}_{1:K_c}, r_{<m})
  $$

---

## 关键图表

### Figure 1：概念图与整体性能

![[WALL-WM_fig1_p2.png]]

**说明**: 左侧展示模态层次结构（语言 → 视觉 → 动作），说明不同模态在语义粒度上的天然差异；右侧展示 WALL-WM 在真实机器人泛化评估上的整体性能对比，WALL-WM 达到 SOTA。

### Figure 2：训练方案对比

![[WALL-WM_fig2_p4.png]]

**说明**: 对比"下一事件训练（Next-Event Training）"与"等长块方案（Equilong-Chunk Scheme）"。事件对齐方案的块长度随语义内容动态变化，等长方案在语义边界处产生错位。

### Figure 3：WALL-WM 整体框架

![[WALL-WM_fig3_p7.png]]

**说明**: 完整系统架构图，展示视频 DiT、动作 DiT 与楼梯式 VLM 推理模块的组合关系，以及事件中心窗口的输入输出流程。

### Figure 4：跨视图遮罩机制

![[WALL-WM_fig4_p13.png]]

**说明**: 展示视锥注意力遮罩（左）和管状补丁遮罩（右）的工作原理。视锥遮罩基于几何相交判断哪些补丁对之间可以交互；管状遮罩在时序连续帧的对应空间位置施加更强的一致性约束。

### Figure 5：观测中心窗口（M=1）

![[WALL-WM_fig5_p15.png]]

**说明**: 展示在 next-chunk 适配阶段（Stage 5）中，以单帧历史（M=1）、锚点帧和 N 帧未来为输入的观测中心窗口结构。VAE 编解码规则：$1 + 4M + 4N$ 原始帧压缩为 $1 + M + N$ 潜变量。

### Figure 6：三种 CoT 推理方案对比

![[WALL-WM_fig6_p17.png]]

**说明**: 对比串行 CoT（Sequential）、并行无监督（Parallel Unsupervised）与楼梯式蒸馏（Staircase Distillation）三种推理调度方案。楼梯式方案实现了并行生成同时保留重建监督，兼顾速度与质量。

### Figure 7：部署平台

![[WALL-WM_fig7_p25.png]]

**说明**: 内部自研的机器人部署平台，展示 WALL-WM 在真实机器人上的硬件部署环境。

### Figure 8：数据集来源分布图

![[WALL-WM_fig8_p28.png]]

**说明**: 训练数据来源的四象限分布图：一般互联网视频（OpenVID 1.2M clips）、以自我为中心的人类动作视频、XRZero-G0 可穿戴设备采集的无机身数据、以及异构机器人遥操作与公开数据集。

### Figure 9：视频+动作数据统计分布

![[WALL-WM_fig9_p29.png]]

**说明**: 训练集中视频和动作数据的分布统计，包括事件时长分布、数据规模统计等，反映数据集的覆盖广度。

### Figure 10：XRZero-G0 无机身数据采集装置

![[WALL-WM_fig10_p35.png]]

**说明**: XRZero-G0 可穿戴采集装置，允许在无机器人本体的情况下收集 UMI 风格的手部操作数据，大幅降低数据采集成本并拓展场景多样性。

### Figure 11：视频+动作数据的时间同步

![[WALL-WM_fig11_p37.png]]

**说明**: 展示视频-动作相位对齐方法：通过光流估算视觉运动信号，通过末端执行器位置有限差分估算动作运动信号，再做跨模态时间相关对齐，消除采集时的相位偏移。

### Figure 12：单集的四级标注轨迹

![[WALL-WM_fig12_p38.png]]

**说明**: 展示对单个机器人操作集进行四级时序标注的示例：任务级（全集目标）→ 子任务级（语义阶段）→ 动作级（操作原语：到达、抓取、提起、放置）→ 片段级（最细粒度事件）。

### Figure 13：平衡采样的标注粒度分布

![[WALL-WM_fig13_p30.png]]

**说明**: 展示聚类均衡采样后各标注粒度的分布统计，反映两轮离线聚类（视觉-语言聚类 + 动作轨迹聚类）如何改善数据多样性覆盖。

### Table 1：各训练阶段可训练/冻结组件矩阵

| 组件 | 视频预训练 | 动作预训练 | VLM 文本条件 | 楼梯蒸馏 | next-chunk 适配 |
|------|-----------|-----------|-------------|---------|----------------|
| 3D 因果 VAE $\mathcal{E}_v$ | ✗ | ✗ | ✗ | ✗ | ✗ |
| T5 文本编码器 | ✗ | ✗ | ✗ | — | ✗ |
| VLM 骨干（Qwen3.5-9B） | — | — | ✗ | ✗ | ✗ |
| VLM 投影输出头 / 辅助头 | — | — | ✓ | ✗ | ✗ |
| 视频 DiT（含视图注意力） | ✓ | ✗ | ✗ | ✗ | ✓ |
| 动作 DiT（含层耦合） | — | ✓ | ✗ | ✗ | ✓ |
| 楼梯 MoT 分支 | — | — | — | ✓ | ✗ |

**关键发现**: VAE 全程冻结以保留压缩先验；VLM 骨干全程冻结仅训练适配头，体现参数高效微调策略；视频与动作 DiT 在 next-chunk 阶段联合微调。

---

## 训练流程

WALL-WM 采用五阶段渐进式训练：

### Stage 1：事件中心视频预训练

- 继承 Wan text-to-video 权重，T5 编码器冻结
- **长度感知标注丢弃调度**（避免短事件缺乏文本指导）:

  $$
  \rho(L_e) = \begin{cases}
  \rho_{\min}, & L_e \leq L_{\min} \\
  \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot \frac{1 - \cos\left(\pi \frac{L_e - L_{\min}}{L_{\max} - L_{\min}}\right)}{2}, & L_{\min} < L_e < L_{\max} \\
  \rho_{\max}, & L_e \geq L_{\max}
  \end{cases}
  $$

  $\rho$ 越大代表越高概率丢弃标注，短事件更需保留标注指导

- 边界遮罩排除超出画面区域的帧外区域
- EMA 权重更新（$\beta_{\text{ema}} = 0.9999$）

### Stage 2：动作预训练

- 视频塔在调度锚点 $s^\star = 45$（50步调度）处冻结
- 非对称 1-to-50 映射：全部动作步骤交叉注意到单一视频步骤
- 每步并行 $K=6$ 个独立动作噪声采样

### Stage 3：VLM 文本条件预训练

- 适配 Qwen3.5-9B 骨干，可学习投影头
- 视觉编码器冻结

### Stage 4：楼梯式蒸馏

- 训练轻量级 Mixture-of-Transformers（MoT）耦合到冻结 VLM 骨干
- 中继深度 $N_r$ 划分共享计算与并行推理分区

### Stage 5（可选）：Next-Chunk 适配

- 以观测为中心的窗口：M 历史帧 + 锚点帧 + N 未来帧
- VAE 编解码规则：$1 + 4M + 4N$ 原始帧 → $1 + M + N$ 潜变量

---

## 训练数据生态

### 数据集概况

| 数据来源 | 规模 | 特点 | 用途 |
|---------|------|------|------|
| OpenVID 通用视频 | 1.2M clips | 通用互联网视频 | 视频预训练 |
| 以自我为中心的人类动作视频 | — | 操作先验 | 视频预训练 |
| XRZero-G0 无机身采集 | — | UMI 风格，无机器人本体 | 动作数据 |
| 异构机器人遥操作+公开数据集 | — | 多场景多任务 | VLA 微调 |

### 数据处理关键技术

- **光流时间同步**: 通过帧间图像变化估算视觉运动信号，与末端执行器位置差分对齐，消除相位偏移
- **四级层次标注**: 任务 → 子任务 → 动作原语 → 细粒度片段
- **双重聚类均衡采样**: 视觉-语言聚类均衡指令-场景覆盖 + 动作轨迹聚类均衡轨迹空间
- **接触丰富恢复数据增强**: 在接触事件附近进行局部姿态扰动，创建受控的恢复行为数据

恢复数据分布混合：

$$
\tilde{p}_{\text{train}}(\mathbf{q}, e) = (1 - \alpha) p_{\text{nominal}}(\mathbf{q}, e) + \alpha \mathbb{E}_e[p(\mathbf{q}|e)]
$$

---

## 推理模式

### 事件模式（Event Mode）

- [[大语言模型|VLM]] 或人工提出下一事件描述
- 系统执行变长视频-动作片段
- 无固定控制时域；时长跟随任务语义动态确定

### 统一模式（Unified Mode）

- 楼梯解码器并行生成 $K_c$ 个连续潜推理状态
- 固定长度块预测，保持标准部署模式
- 同时维持梯度连续的 VLA 路径

---

## 实验

### 评估任务类别

1. **多样化操作（Diverse Manipulation）**: 宽泛的任务覆盖度
2. **推理操作（Reasoning Manipulation）**: 需要任务分解的复杂任务
3. **灵巧操作（Dexterous Manipulation）**: 需要精细手部控制的任务
4. **泛化（Generalization）**: 新颖物体、场景和指令下的泛化
5. **消融（Ablations）**: 事件建模与视图建模组件的影响

### 实现细节

- **优化器**: Muon（大规模预训练）
- **计算优化**: 细粒度计算重叠 + 多事件序列打包提升吞吐
- **模型压缩**: 大模型蒸馏 + FP8 量化支持
- **视频骨干**: 继承 Wan DiT 权重

---

## 批判性思考

### 优点

1. **三重错位根治**: 从根本上解决语言-视觉-控制时间尺度错位，而非事后弥补
2. **先验保留型扩展**: 在保留视频生成预训练先验（Caption-to-Video）的同时引入多视图和动作能力，避免灾难性遗忘
3. **两种推理模式覆盖**: 事件模式支持高级任务分解，统一模式保持与现有部署管线的兼容性
4. **数据飞轮设计完整**: XRZero-G0 无机身采集 + 四级标注 + 双重聚类采样，形成系统化的数据工程体系

### 局限性

1. **实验数值细节不公开**: arXiv 版本中具体的量化指标（成功率等）在截至本笔记创建时尚未完全公开
2. **五阶段训练复杂度高**: 训练流程涉及五个阶段、多组件渐进冻结/解冻，工程实现门槛较高
3. **VLM 骨干固定**: Qwen3.5-9B 骨干全程冻结，限制了多模态联合优化的上限

### 潜在改进方向

1. 探索事件边界的自动检测，减少对标注的依赖
2. 研究 VLM 骨干低秩微调（LoRA）以提升语言-视觉协同
3. 将楼梯推理扩展到更长时域的任务规划

### 可复现性评估

- [ ] 代码开源（GitHub 链接存在但尚未确认完整性）
- [ ] 预训练模型（未公开）
- [ ] 训练细节完整（论文中有较详细描述）
- [ ] 数据集可获取（部分数据集为内部自有）

---

## 关联笔记

### 基于

- [[视频扩散模型]]: 继承 Wan DiT text-to-video 权重作为视频骨干
- [[流匹配]]: 视频和动作的去噪目标均采用 flow-matching 框架
- [[视觉语言动作模型]]: 将 VLA 扩展为 World Action Model

### 对比

- [[扩散策略|Diffusion Policy]]: 固定长度块方案的典型代表，WALL-WM 解决其时语义错位问题
- [[扩散世界模型]]: 现有世界模型未将动作事件与视频语义对齐

### 方法相关

- [[多视图几何建模]]: 视锥遮罩的几何约束基础
- [[流匹配]]: 视频和动作的核心去噪目标
- [[动作分块]]: Next-chunk 适配阶段的输出表示
- [[Camera RoPE]]: 无标定相机身份编码

### 硬件/数据相关

- [[XRZero-G0]]: 无机身数据采集可穿戴装置

---

## 速查卡片

> [!summary] WALL-WM: Carving World Action Modeling at the Event Joints
> - **核心**: 以语义动作事件为原子单元的视频-动作联合预训练，解决语言/视觉/控制三重时间尺度错位
> - **方法**: 事件中心 DiT + 视锥几何多视图遮罩 + 楼梯式并行潜推理
> - **结果**: 真实机器人大规模泛化评估 SOTA，覆盖多样化/推理/灵巧操作与泛化四类任务
> - **代码**: https://github.com/X-Square-Robot/wall-x

---

*笔记创建时间: 2026-06-03*

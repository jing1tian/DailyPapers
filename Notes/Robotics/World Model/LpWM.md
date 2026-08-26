---
title: "LpWM: A Case for Sparse Representations in World Models"
method_name: "LpWM"
authors: [Yilun Kuang, Yash Dagade, Quentin Le Lidec, Lucas Maes, Randall Balestriero, Yann LeCun]
year: 2026
venue: arXiv
tags: [world-model, jepa, sparse-representation, planning, self-supervised-learning, representation-learning, robotics]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.22764
created: 2026-08-26
---

# 论文笔记：LpWM — A Case for Sparse Representations in World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | (Brown University · Meta FAIR 等) |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[LeWM]] · [[VICReg]] · [[DINO-WM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.22764) / [HTML](https://arxiv.org/html/2608.22764) |

---

## 一句话总结

> 通过理论证明非线性 Lipschitz 动力学可被高维 one-hot 空间中的线性动力学任意近似，提出用稀疏非负 [[JEPA|联合嵌入预测]] 世界模型 LpWM，在 PushT 任务上规划成功率比 dense [[LeWM]] 提高最多 **57%**。

---

## 核心贡献

1. **线性化理论**：证明紧致状态空间上的均匀 Lipschitz 非线性动力学，可被充分高维 one-hot 潜空间中的动作条件线性动力学（排列矩阵）任意近似；rollout 误差随维度增长趋于零。
2. **稀疏非负编码**：提出 [[RDMReg|Rectified Distribution Matching Regularization]]，将 latent 边缘分布匹配到 ℓp（p≤1）范数约束下的 Rectified Generalized Gaussian，诱导稀疏非负编码；用 [[RepReLU]] 解决 dying-ReLU 问题。
3. **解释性结构发现**：稀疏表示自发将动力学组织成离散 regime：二值支撑编码不同接触模式（94–99% 可解码），特征幅值捕捉 regime 内的连续状态信息。
4. **多谓词实验验证**：在 Wall、PushT、Piecewise 合成导航、OGBench-Cube 等多个环境中系统比较 dense vs. sparse，揭示"中等复杂度预测器"下稀疏优势最显著。

---

## 问题背景

### 要解决的问题

现有 [[JEPA]] 风格世界模型（如 [[LeWM]]、[[VICReg]]）使用**密集连续** latent 表示，预测器需要较高容量才能拟合复杂接触动力学。本文探讨：**稀疏非负 latent 能否降低预测器复杂度需求、同时提升规划性能？**

### 现有方法的局限

- **密集表示**（Dense JEPA）：latent 分布为各向同性高斯，不区分离散动力学模式，预测器需高容量才能应对接触等非线性。
- **[[VICReg]]**：variance / invariance / covariance 三项组合，latent 连续，缺乏稀疏结构，解释性差。
- **One-hot 精确离散化**：维度诅咒使误差以 $O(N^{-1/d})$ 衰减，实际不可行。

### 本文的动机

作者从理论上证明：一维 one-hot 编码对 Lipschitz 动力学是最优的，分布式稀疏编码（Distributed Sparse Codes）是其实用化松弛。通过将 latent 分布约束为 ℓp 球上的 Rectified Generalized Gaussian，可以在连续潜空间中近似 one-hot 的线性化效果，同时保持端到端可微训练。

---

## 方法详解

### 模型架构

[[LpWM]] 是标准 [[JEPA]] 框架：编码器 $\text{enc}_\theta$ 将观测 $o_t$ 映射到**稀疏非负** latent $z_t$；动作条件预测器 $\text{pred}_\phi$ 在 latent 空间前推 $\hat{z}_{t+1}$；用 [[RDMReg]] 替代 [[SIGReg]] 以诱导稀疏性。

- **输入**: RGB 像素观测 $o_t$ + 动作 $a_t$
- **编码器**: [[Vision Transformer|ViT]] + [[RepReLU]] 激活，输出非负稀疏编码 $z_t \geq 0$
- **预测器**: 多种复杂度配置（LTI、MLP∘LTI、MLP∘LTV、Deep-AdaLN）
- **正则**: [[RDMReg]]（Rectified Distribution Matching Regularization）
- **规划**: [[Cross-Entropy Method|CEM]] + 隐空间目标距离

### 核心模块

#### 模块 1: RDMReg — Rectified Distribution Matching Regularization

**设计动机**: 将 latent 边缘分布从各向同性高斯（密集）替换为 Rectified Generalized Gaussian（稀疏非负），诱导 [[Sparse Representation|稀疏]] 结构，使 latent 在高维下近似 one-hot 行为。

**具体实现**:
- 目标分布: $y \sim \prod_{i=1}^d \text{ReLU}(\mathcal{GN}_p(\mu, \sigma))$，其中 $p \leq 1$
- 用 [[Cramér-Wold 定理]] 思路：匹配沿随机方向 $c$ 的一维投影分布
- 损失 $\mathcal{L}$ 为 Wasserstein 或 KL 型 1D 分布匹配

#### 模块 2: RepReLU — Reparameterized ReLU

**设计动机**: 编码器输出需要精确零（稀疏性），ReLU 会导致 [[Dying ReLU]] 问题，阻断梯度。RepReLU 前向等价 ReLU，反向等价 GELU，兼顾稀疏性与梯度流通。

**具体实现**:

$$
\text{RepReLU}(x) = \text{sg}(\text{ReLU}(x)) + \text{GELU}(x) - \text{sg}(\text{GELU}(x))
$$

其中 $\text{sg}(\cdot)$ 为 stop-gradient；前向传递输出 $\text{ReLU}(x)$（精确零），反向传递梯度通过 $\text{GELU}(x)$ 流动。

#### 模块 3: 多级复杂度预测器

提供五种预测器以系统研究预测器容量影响：

| 预测器 | 函数形式 | 复杂度 |
|--------|---------|--------|
| LTI(1) | $\hat{z}_{t+1} = \sigma(A z_t + B a_t)$ | 最低 |
| LTI(k) | $\hat{z}_{t+1} = \sigma(\sum_{i=0}^{k-1} A_i z_{t-i} + B a_t)$ | 低 |
| MLP∘LTI(k) | $\hat{z}_{t+1} = \sigma(W \text{ReLU}(\sum_i A_i z_{t-i} + B a_t))$ | 中 |
| MLP∘LTV(k) | $\hat{z}_{t+1} = \sigma(W \text{ReLU}(\sum_i A_i(z_t) z_{t-i} + B(z_t) a_t))$ | 中-高 |
| Deep-AdaLN(k) | $\hat{z}_{t+1} = \sigma(\text{AdaLN}^{(6)}(z_{t-k+1:t}, a_t))$ | 最高 |

---

## 关键公式

### 公式 1: [[JEPA]] 训练目标

$$
\min_{\theta, \phi} \|\hat{z}_{t+1} - z_{t+1}\|_2 + \lambda_{\text{RDMReg}} \cdot \mathcal{R}(z_{t+1})
$$

**含义**: 最小化预测误差，同时用 RDMReg 正则约束 latent 分布为稀疏非负，防止 [[Representation Collapse|表示塌缩]]。

**符号说明**:
- $\hat{z}_{t+1}$: 预测器输出的未来 latent
- $z_{t+1}$: 编码器输出的真实未来 latent
- $\lambda_{\text{RDMReg}}$: 正则权重超参

### 公式 2: [[RDMReg]] 正则项

$$
\mathcal{R}(z) = \mathbb{E}_{c \sim \text{Unif}(\mathcal{S}^{d-1}_{\ell_2})} \left[ \mathcal{L}\!\left(\mathbb{P}_{c^\top z} \,\|\, \mathbb{P}_{c^\top y}\right) \right]
$$

其中目标分布:

$$
y \sim \prod_{i=1}^{d} \text{ReLU}\!\left(\mathcal{GN}_p(\mu, \sigma)\right)
$$

**含义**: 沿单位球面上均匀采样的随机方向 $c$，匹配 latent 投影与目标 Rectified Generalized Gaussian 投影的 1D 分布，诱导稀疏非负结构。

**符号说明**:
- $c$: 随机单位方向向量
- $\mathbb{P}_{c^\top z}$: latent 沿方向 $c$ 的投影分布
- $\mathcal{GN}_p$: 广义高斯分布，$p \leq 1$ 时对应超高斯/稀疏分布

### 公式 3: Proposition 1 — 非线性动力学的线性化近似

对紧致状态空间 $X \subset \mathbb{R}^d$、动作空间 $A \subset \mathbb{R}^p$ 上的均匀 Lipschitz 动力学 $f(x, a)$，存在有限维 $N$、one-hot 编码器 $E_\varepsilon$、解码器 $D_\varepsilon$ 和动作条件排列矩阵 $P_\varepsilon(a)$，使得：

$$
\sup_{x_t \in X,\, a_t \in A} \|f(x_t, a_t) - D_\varepsilon(P_\varepsilon(a_t) E_\varepsilon(x_t))\| \leq (L+1)\varepsilon
$$

**含义**: 充分高维的 one-hot 潜空间可以将非线性动力学线性化为排列矩阵操作，误差任意小。

### 公式 4: Corollary 1 — 维度与误差的权衡

对单位超立方体 $X = [0,1]^d$ 划分为 $n^d$ 个网格单元，覆盖半径：

$$
\varepsilon_N = \frac{\sqrt{d}}{2} N^{-1/d}
$$

单步预测误差上界：

$$
\sup_{x, a} \|f(x,a) - D_{\varepsilon_N}(P_{\varepsilon_N}(a) E_{\varepsilon_N}(x))\| \leq \frac{\sqrt{d}}{2}(L+1) N^{-1/d} = O(N^{-1/d})
$$

有限 horizon $H$ 的 rollout 误差：

$$
\|x_H - \hat{x}_H\| \leq \frac{\sqrt{d}}{2} N^{-1/d} \sum_{k=0}^{H} L^k
$$

**含义**: 维度诅咒以 $O(N^{-1/d})$ 衰减，维度越高近似越精，但分布式稀疏编码是更实际的近似手段。

### 公式 5: [[RepReLU]]

$$
\text{RepReLU}(x) = \text{sg}(\text{ReLU}(x)) + \text{GELU}(x) - \text{sg}(\text{GELU}(x))
$$

**含义**: 前向通过 ReLU（精确零，实现稀疏性），反向通过 GELU（光滑梯度，避免 dying-ReLU）。

### 公式 6: LTV 结构化参数化

$$
A_i(z_t) = A_i + U_i \,\text{diag}(g_i(z_t))\, V_i^\top, \quad g(z_t) = \text{sigmoid}(G z_t) \in [0,1]^r
$$

**含义**: LoRA 风格的低秩门控矫正，使状态依赖的动力学算子在基础线性算子上叠加输入自适应调制。

### 公式 7: [[MPC]] 规划代价

$$
C = \|\hat{z}_T - z_g\|_2
$$

**含义**: 规划器（[[Cross-Entropy Method|CEM]]）优化使 rollout 终态 latent 接近目标 latent 的动作序列。

### 公式 8: Jaccard 指数（硬版本）

$$
J(x, y) = \frac{\sum_{i=1}^{D} \mathbf{1}_{x_i=1 \wedge y_i=1}}{\sum_{i=1}^{D} \mathbf{1}_{x_i=1 \vee y_i=1}}
$$

**含义**: 衡量两个稀疏二值支撑之间的重叠度，用于分析 latent 支撑的 regime 结构。

### 公式 9: Soft Jaccard（可微松弛）

$$
J_S(x, y) = \frac{\sum_{i=1}^{D} \min(x_i, y_i)}{\sum_{i=1}^{D} \max(x_i, y_i) + \varepsilon}
$$

**含义**: 对连续非负向量的 Jaccard 指数可微近似，用于损失函数计算。

### 公式 10: Temporal Jaccard Loss

$$
\mathcal{L}_{\text{TJ}} = \frac{1}{B(T-1)} \sum_{b=1}^{B} \sum_{t=1}^{T-1} \left(1 - J_S(Z[b,t,:],\, Z[b,t+1,:])\right)
$$

**含义**: 对连续帧鼓励支撑稳定（时间慢性先验），促使支撑变化与物理接触事件对齐。

### 公式 11: μP 学习率缩放

$$
\eta_W \propto \frac{1}{d_{\text{in}}}, \qquad \eta_W(D) = \eta_{\text{base}} \cdot \frac{D_0}{D}
$$

**含义**: [[Maximal Update Parameterization|μP]] 保证跨宽度变化时特征学习动态一致，支持宽度扫描实验。

---

## 关键图表

### Figure 1a: LpWM 整体架构示意

![[LpWM_fig1a_schematic.png]]

**说明**: LpWM 的整体结构。ViT 编码器通过 [[RepReLU]] 输出稀疏非负 latent $z_t$；预测器（多种容量配置）在 latent 空间动作条件前推；[[RDMReg]] 将分布约束到 Rectified Generalized Gaussian 以诱导稀疏性。

### Figure 1b: PushT 规划性能 vs. 预测器容量

![[LpWM_fig1b_pusht_planning.png]]

**说明**: 在 PushT 操作任务上，sparse LpWM 在**中等预测器复杂度**时超越 dense [[LeWM]] 最多 **57%** 规划成功率；高容量预测器下两者性能趋于接近。

### Figure 2a: Piecewise 环境支撑 Jaccard 热力图

![[LpWM_fig2a_piecewise_heatmap.png]]

**说明**: 在 Piecewise 合成导航环境中，对应不同 regime（分段区域）的 latent 支撑形成高度对角化的 Jaccard 相似度矩阵，验证支撑确实编码了离散动力学 regime。

### Figure 2b: 状态解码实验

![[LpWM_fig2b_state_decoding.png]]

**说明**: 从 LpWM 的稀疏 latent 解码 ground-truth 状态标签，94–99% 准确率说明支撑模式对 regime 信息高度可解码。

### Figure 2c: Piecewise 规划评估

![[LpWM_fig2c_piecewise_planning.png]]

**说明**: Piecewise 环境下，sparse LpWM 的规划成功率在低/中预测器复杂度时均优于 dense 基线。

### Figure 3a: Wall 环境 LeWM vs. LpWM

![[LpWM_fig3a_wall_comparison.png]]

**说明**: 在 Wall 简单导航环境（线性动力学主导），两种表示性能相当，线性预测器已足够。

### Figure 3b: PushT 环境 VICReg vs. LpWM

![[LpWM_fig3b_pusht_vicreg.png]]

**说明**: 在 PushT 接触操作任务，稀疏 LpWM 在中等预测器复杂度时显著优于 dense [[VICReg]]，体现接触 regime 离散性与稀疏表示的匹配。

### Figure 4a: 时间 Jaccard 正则化效果

![[LpWM_fig4a_temporal_jaccard.png]]

**说明**: 添加 $\mathcal{L}_{\text{TJ}}$ 后，latent 支撑的变化更集中于物理接触事件，提升 regime 边界的时间对齐性。

### Figure 4c: OGBench-Cube 支撑 Raster 图

![[LpWM_fig4c_cube_raster.png]]

**说明**: 在 OGBench-Cube 机器人操作任务中，latent 支撑的 raster 可视化显示不同抓取接触状态对应不同的稀疏编码模式。

### Figure 5: 实验环境可视化

![[LpWM_fig5_environments.png]]

**说明**: 四个评测环境（从左到右）：Wall（简单 2D 导航）、PushT（接触操作）、Piecewise（合成分段导航）、OGBench-Cube（机器人 cube 操作）。

### Table 1: 预测器功能形式汇总

| 预测器 | 函数形式 | 参数规模 |
|--------|---------|---------|
| Deep-AdaLN(k) | $\sigma(\text{AdaLN}^{(6)}(z_{t-k+1:t}, a_t))$ | 最大 |
| MLP∘LTV(k) | $\sigma(W \text{ReLU}(\sum_i A_i(z_t)z_{t-i} + B(z_t)a_t))$ | 中-大 |
| MLP∘LTI(k) | $\sigma(W \text{ReLU}(\sum_i A_i z_{t-i} + B a_t))$ | 中 |
| LTI(k) | $\sigma(\sum_i A_i z_{t-i} + B a_t)$ | 小 |
| LTI(1) | $\sigma(A z_t + B a_t)$ | 最小 |

**关键发现**: 稀疏 LpWM 在 LTI(k) 和 MLP∘LTI(k) 层级即可达到 dense 表示需要 Deep-AdaLN 才能达到的规划性能。

---

## 实验

### 数据集 / 环境

| 环境 | 特点 | 主要测试内容 |
|------|------|------------|
| Wall | 2D 导航，线性动力学 | 验证线性预测器充分性 |
| PushT | 接触操作，非线性动力学 | 稀疏 vs. 密集主要对比场景 |
| Piecewise | 合成分段导航，ground-truth regime 已知 | 验证支撑结构与 regime 对应关系 |
| OGBench-Cube | 机器人三维 cube 操作 | 接触追踪与稀疏编码演示 |

### 实现细节

- **编码器**: ViT + [[RepReLU]]（末层激活），产生非负稀疏 latent
- **正则项**: [[RDMReg]]（$p \leq 1$）for LpWM；[[SIGReg]] for dense [[LeWM]] 基线
- **规划器**: [[Cross-Entropy Method|CEM]]，目标为最小化 latent 距离 $\|\hat{z}_T - z_g\|_2$
- **宽度扫描**: 采用 [[Maximal Update Parameterization|μP]] 保证跨宽度实验可比
- **可选辅助损失**: Temporal Jaccard Loss $\mathcal{L}_{\text{TJ}}$（促进接触对齐，默认关闭）

### 主要实验结论

1. **Wall（线性环境）**: 稀疏与密集表示性能相近，LTI 预测器已足够；无稀疏优势。
2. **PushT（接触环境）**: 稀疏 LpWM 在中等预测器容量时最多领先 dense LeWM **57%**；高容量时差距收窄。
3. **Piecewise（合成）**: Jaccard 热力图确认稀疏支撑几乎完美恢复 ground-truth 离散 regime，94–99% state 解码准确率。
4. **OGBench-Cube**: 稀疏 latent 追踪接触状态，raster 图可视化清晰展示抓取模式切换。

---

## 批判性思考

### 优点

1. **理论扎实**: Proposition 1 给出了"为什么稀疏有效"的严格数学依据（Lipschitz 动力学线性化），不只是启发式直觉。
2. **可解释性强**: 离散 regime 编码在支撑模式中，连续状态在幅值中，两类信息自然分离，比密集 latent 更易分析。
3. **代价可控**: RepReLU 优雅解决 dying-ReLU，RDMReg 是对 SIGReg 的自然扩展，无需引入全新训练范式。

### 局限性

1. **维度诅咒**: 理论近似误差以 $O(N^{-1/d})$ 衰减，高维状态空间下需要极高维 latent 才能保证近似质量，实际可行性存疑。
2. **环境依赖性**: 稀疏优势主要在接触操作（PushT）等离散模式切换显著的任务中体现，线性动力学环境（Wall）下无明显收益。
3. **支撑语义无保证**: 未加 $\mathcal{L}_{\text{TJ}}$ 时，vanilla LpWM 不保证支撑变化与物理事件对齐，regime 边界可能与物理无关。
4. **规划器局限**: 仍依赖 CEM 等基于模型的规划，未探索与学习型策略的结合。

### 潜在改进方向

1. 探索更高效的分布式稀疏编码，缓解维度诅咒，减少所需潜在维度。
2. 将支撑结构与符号规划或程序合成结合，利用离散 regime 实现层次化任务分解。
3. 研究稀疏 world model 与模型蒸馏的结合，压缩接触动力学先验。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（μP、超参、预测器配置均有描述）
- [x] 数据集可获取（PushT、OGBench 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[LeWM]]: 密集 JEPA 世界模型基线，使用 [[SIGReg]] 正则，LpWM 以其为主要对比对象
- [[JEPA]]: Joint-Embedding Predictive Architecture，LpWM 的基础框架
- [[VICReg]]: 早期密集表示正则基线，同在 PushT 上对比

### 对比

- [[DINO-WM]]: 基于预训练 DINO 编码器的 JEPA 世界模型，密集表示，规划慢
- [[PLDM]]: 另一种 JEPA 风格世界模型，多 loss 训练，密集表示

### 方法相关

- [[Sparse Representation]]: 核心表示范式
- [[Rectified Generalized Gaussian]]: RDMReg 的目标分布族
- [[RepReLU]]: 解决 dying-ReLU 的核心激活函数
- [[Maximal Update Parameterization]]: 宽度扫描实验所用 μP 参数化
- [[Cross-Entropy Method]]: 规划所用 CEM 优化器
- [[Vision Transformer]]: 编码器骨干

### 硬件/数据相关

- [[OGBench]]: 机器人操作评测基准，含 Cube 任务
- [[PushT]]: 接触操作基准，主要评测场景

---

## 速查卡片

> [!summary] LpWM: A Case for Sparse Representations in World Models
> - **核心**: 理论证明高维 one-hot 可线性化非线性动力学，提出稀疏非负 JEPA 世界模型 LpWM
> - **方法**: ViT + RepReLU + RDMReg（ℓp 分布匹配），多容量预测器系统对比
> - **结果**: PushT 任务中等预测器复杂度下规划成功率比 dense LeWM 提升最多 57%
> - **代码**: 未公开（截至 2026-08）

---

*笔记创建时间: 2026-08-26*

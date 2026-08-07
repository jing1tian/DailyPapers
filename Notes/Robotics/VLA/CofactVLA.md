---
title: "CofactVLA: Deconfounding Vision-Language-Action Models via Counterfactual Intervention"
method_name: "CofactVLA"
authors: [Yan Zhang, Yinan Wu, Haoran Duan, Jungong Han]
year: 2026
venue: arXiv
tags: [vla, causal-inference, counterfactual-learning, robotic-manipulation, ood-generalization, flow-matching]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.04396
created: 2026-08-07
---

# 论文笔记：CofactVLA: Deconfounding Vision-Language-Action Models via Counterfactual Intervention

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Department of Automation, Tsinghua University |
| 日期 | August 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Pi05\|π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.04396) / Code 暂未开源 |

---

## 一句话总结

> 通过因果干预框架（DDG + OPG + CCR）消除 [[VLA]] 模型中视觉混淆变量对语言指令的遮蔽，在真实机器人 OOD 场景下取得 +52.3% 的绝对性能提升。

---

## 核心贡献

1. **视觉覆盖现象的因果建模**: 用 [[Dual-path Deconfounding Graph|双路去混淆图（DDG）]] 形式化描述 VLA 中视觉混淆变量绕过语言语义的虚假因果路径，提供理论分析基础。
2. **动作级去混淆 — OPG**: 提出 [[Orthogonal Projection Guidance|正交投影引导（OPG）]]，在 [[Flow Matching|流匹配]] 速度场中将事实分支速度投影到反事实速度的正交补空间，提取纯语义意图。
3. **特征级去混淆 — CCR**: 提出 [[Counterfactual Covariance Reduction|反事实协方差消减（CCR）]]，通过广义特征分解分离潜在表示中的观测驱动子空间与混淆变量子空间，直接在特征层面剔除视觉捷径。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型在训练时，密集的视觉流信号往往主导稀疏的语言指令信号，导致策略在执行时忽视文本指令，转而依赖视觉捷径（如显著物体、熟悉布局）完成任务——即 [[Vision-Override Phenomenon|视觉覆盖现象（vision-override phenomenon）]]。这一问题在 [[Out-of-Distribution Generalization|分布外（OOD）]] 场景下尤为突出：当视觉背景发生变化时，模型性能急剧下降。

### 现有方法的局限

- **模态融合方法**（如多模态注意力）：仅在特征层面融合，未从因果层面切断虚假视觉路径。
- **[[Classifier-Free Guidance|无分类器引导（CFG）]]**：简单相减 `v_cond - v_uncond`，会产生方向偏移，可能破坏原始动作流形。
- **数据增强策略**：覆盖有限的扰动类型，泛化能力不足。

### 本文的动机

将视觉混淆变量明确建模为 [[因果推断|因果干预]] 对象，利用反事实分支（仅有视觉无语言）显式估计视觉偏差，再通过几何正交投影（动作级）与特征子空间分解（特征级）双管齐下消除偏差，从根本上实现语义引导的 [[Robotic Manipulation|机器人操作]]。

---

## 方法详解

### 模型架构

CofactVLA 采用**双路反事实干预**架构，以 [[Pi05|π₀.₅]] 为骨干网络：

- **输入（事实分支）**: 语言指令 $l$ + 观测 $o_t$，送入完整 VLA 模型，输出速度场 $v_\text{cond}$
- **输入（反事实分支）**: 仅观测 $o_t$（屏蔽语言），送入相同 VLA 模型，输出速度场 $v_\text{uncond}$
- **核心模块1**: [[Orthogonal Projection Guidance|OPG]] — 在 [[Flow Matching|流匹配]] 解码阶段对速度场施加几何正交投影
- **核心模块2**: [[Counterfactual Covariance Reduction|CCR]] — 在 Transformer 中间层特征上执行子空间投影消偏
- **输出**: 经去混淆的速度场驱动 [[Action Chunking|动作块]] $a_{t:t+50}$（chunk size = 50）
- **总参数**: 基于 π₀.₅（具体参数量论文未列出）

### 核心模块

#### 模块1: Action-Level OPG（动作级正交投影引导）

**设计动机**: [[Flow Matching|流匹配]] 框架中的速度场 $v_\theta$ 与 [[Score Function|得分函数]] $\nabla \log p^\tau$ 之间存在线性等价关系（公式2），因此可以在速度场域直接进行因果干预，无需引入额外网络。

**具体实现**:
- 计算事实速度在反事实速度上的投影分量 $v_\text{proj}$（公式3）
- 提取正交分量 $v_\perp = v_\text{cond} - v_\text{proj}$，代表纯语义驱动的速度方向（公式4）
- 以可调强度 $\gamma$ 增强正交分量，输出最终干预速度 $v_\text{causal}$（公式5）

与 [[Classifier-Free Guidance|CFG]] 的核心区别：CFG 做向量相减可能跨越动作流形边界，而 OPG 始终保持在事实速度所在方向附近，仅去除与视觉偏差共线的分量。

#### 模块2: Feature-Level CCR（特征级反事实协方差消减）

**设计动机**: 仅做动作级干预不够彻底，视觉混淆变量已渗入中间层潜表示。通过[[广义特征值分解|广义特征值分解]] 可以精确分离出混淆变量主成分子空间 $\mathcal{S}_C$，再将特征在该子空间上的投影归零，实现根源消偏。

**具体实现**:
- 计算反事实与事实特征的协方差差 $\Delta\Sigma = \Sigma_\text{cf} - \Sigma_f$（公式6）
- 利用[[增益-偏差分解|增益-偏差分解]]将 $\Delta\tilde{\Sigma}$ 分解为观测驱动项 $\Delta\tilde{O}$ 和混淆项 $\Delta\tilde{C}$（公式7）
- 通过对比特征间隙假设（公式8），保证 $\mathcal{S}_C$ 和 $\mathcal{S}_O$ 严格分离
- 提取混淆子空间基向量 $U_\text{bias}$，从特征中减去该方向的投影（公式9）
- 仅在 Transformer 第 15、16 层执行 CCR（$\beta = 0.15$）

---

## 关键公式

### 公式1: [[条件熵|条件熵间隙]]（视觉覆盖的信息论刻画）

$$
H(A \mid O) - H(A \mid O, C) = I(A; C \mid O) > 0
$$

**含义**: 在给定观测 $O$ 的条件下，混淆变量 $C$（视觉偏差）对动作 $A$ 仍有额外信息增益 $>0$，即模型受到视觉混淆变量影响。

**符号说明**:
- $H(\cdot)$: 条件熵
- $I(\cdot;\cdot\mid\cdot)$: 条件互信息
- $A$: 动作，$O$: 观测，$C$: 视觉混淆变量

---

### 公式2: [[Flow Matching|流匹配速度场]]与[[Score Function|得分函数]]等价关系

$$
\nabla_{A^\tau} \log p^\tau(A^\tau \mid \cdot) = -\frac{A^\tau + \tau v_\theta(A^\tau, \tau \mid \cdot)}{1 - \tau}
$$

**含义**: 流匹配速度场 $v_\theta$ 与扩散模型的得分函数之间存在线性等价，因此可以在速度场空间直接进行因果干预，无需转换框架。

**符号说明**:
- $A^\tau$: 时间步 $\tau$ 的噪声动作
- $\tau \in [0,1]$: 流匹配时间参数
- $v_\theta$: 网络预测的速度场

---

### 公式3 & 4: [[Orthogonal Projection Guidance|OPG — 正交速度分解]]

$$
v_\text{proj} = \frac{\langle v_\text{cond},\, v_\text{uncond} \rangle}{\|v_\text{uncond}\|_2^2 + \varepsilon} \cdot v_\text{uncond}
$$

$$
v_\perp = v_\text{cond} - v_\text{proj}
$$

**含义**: $v_\text{proj}$ 是事实速度在反事实视觉偏差方向的投影；$v_\perp$ 是与视觉偏差完全正交的纯语义分量。

**符号说明**:
- $v_\text{cond}$: 条件（事实）分支速度场
- $v_\text{uncond}$: 无条件（反事实）分支速度场
- $\varepsilon$: 数值稳定项

---

### 公式5: [[Orthogonal Projection Guidance|OPG — 因果干预速度]]

$$
v_\text{causal} = v_\text{cond} + \gamma \cdot v_\perp
$$

**含义**: 以引导尺度 $\gamma$ 增强语义正交分量，最终推理时使用 $v_\text{causal}$ 驱动动作解码。$\gamma = 2.0$ 为最优值。

**符号说明**:
- $\gamma$: 因果引导尺度（超参数，默认 2.0）
- $v_\perp$: 纯语义正交速度分量

---

### 公式6: [[Counterfactual Covariance Reduction|CCR — 协方差差]]

$$
\Delta\Sigma = \Sigma_\text{cf} - \Sigma_f
$$

**含义**: 反事实分支与事实分支特征协方差之差，捕获因语言屏蔽而产生的偏差信号。

**符号说明**:
- $\Sigma_\text{cf}$: 反事实（仅视觉）分支特征协方差
- $\Sigma_f$: 事实（视觉+语言）分支特征协方差

---

### 公式7: [[增益-偏差分解]]

$$
\tilde{\Delta} = \tilde{\Delta}_O + \tilde{\Delta}_C
$$

**含义**: 将白化协方差差分解为观测驱动项 $\tilde{\Delta}_O$（正常信号）和混淆变量项 $\tilde{\Delta}_C$（需消除的偏差）。

---

### 公式8: [[对比特征间隙假设]]

$$
\lambda_\min(M \mid_{\Sigma_0^{1/2}\mathcal{S}_C}) > \lambda_\max(M \mid_{\Sigma_0^{1/2}\mathcal{S}_O})
$$

**含义**: 混淆子空间 $\mathcal{S}_C$ 对应的特征值均大于观测子空间 $\mathcal{S}_O$，保证两个子空间可以通过特征值切分严格分离。

**符号说明**:
- $M := \Sigma_0^{-1/2} \Sigma \Delta \Sigma_0^{-1/2}$: $\Sigma_0$-白化矩阵
- $\lambda_\min, \lambda_\max$: 最小/最大特征值
- $\mathcal{S}_C$: 混淆变量子空间，$\mathcal{S}_O$: 观测驱动子空间

---

### 公式9: [[Counterfactual Covariance Reduction|CCR — 特征去混淆]]

$$
F_\text{causal} = F + \Delta O = F - \beta (F U_\text{bias}) U_\text{bias}^\top
$$

**含义**: 从特征 $F$ 中减去在混淆子空间基向量 $U_\text{bias}$ 方向上的投影，实现特征层面的因果净化。$\beta = 0.15$ 为最优干预强度。

**符号说明**:
- $F$: 原始特征矩阵（来自 Transformer 中间层）
- $U_\text{bias}$: 混淆子空间的正特征向量矩阵（$\Delta\Sigma$ 正谱的主成分）
- $\beta$: 干预强度超参数（默认 0.15）

---

### 公式10-12: [[广义特征值分解]]（CCR 理论推导）

$$
M := \Sigma_0^{-1/2} \Sigma \Delta \Sigma_0^{-1/2}
$$

$$
\Sigma \Delta b = \lambda \Sigma_0 b \quad \Longleftrightarrow \quad Mu = \lambda u, \quad u := \Sigma_0^{1/2} b
$$

**含义**: 广义特征值问题可转化为标准特征值问题，通过 $\Sigma_0$-白化矩阵 $M$ 的特征分解求解混淆子空间基。

---

## 关键图表

### Figure 1: 动机与双路去混淆图（DDG）

![Figure 1](https://arxiv.org/html/2608.04396v1/x1.png)

**说明**: 左侧示意 [[Vision-Override Phenomenon|视觉覆盖现象]]：机器人忽视"pick red cube"指令，错误抓取显著蓝色物体。右侧为 [[Dual-path Deconfounding Graph|DDG]] 的因果图：实线为事实路径（语言+视觉→动作），虚线为虚假路径（视觉混淆变量绕过语言直接影响动作），两种干预分别在动作层（OPG）和特征层（CCR）截断虚假路径。

---

### Figure 2: CofactVLA 整体架构

![Figure 2](https://arxiv.org/html/2608.04396v1/x2.png)

**说明**: 展示双路前向传播（事实分支与反事实分支），以及 [[Orthogonal Projection Guidance|OPG]] 在解码阶段和 [[Counterfactual Covariance Reduction|CCR]] 在 Transformer 第 15-16 层的具体位置。两个分支共享权重，反事实分支通过屏蔽语言 token 实现无条件前向传播。

---

### Figure 3: 真实机器人实验结果对比

![Figure 3](https://arxiv.org/html/2608.04396v1/x3.png)

**说明**: 标准条件下 CofactVLA（90.8%）vs. π₀.₅ 基线；OOD 场景下 CofactVLA（75.8%）vs. π₀.₅（23.5%），绝对提升 **+52.3%**。OOD 扰动包括不同背景、新颖物体布局、光照变化等。

---

### Figure 4: 超参数敏感性分析

![Figure 4](https://arxiv.org/html/2608.04396v1/x4.png)

**说明**: 引导尺度 $\gamma$（OPG）在 2.0 时性能最优，过大则损害任务多样性；干预强度 $\beta$（CCR）在 0.15 时最优，过大则过度消除语义信息。两个超参数均有明确最优区间，调参容易。

---

### Figure 5: 仿真与真实场景视觉覆盖消除定性对比

![Figure 5](https://arxiv.org/html/2608.04396v1/x5.png)

**说明**: 在 LIBERO 仿真中，baseline 抓取错误物体（对显著物体产生视觉覆盖），CofactVLA 正确遵循语言指令；真实机器人场景下同样展示了正确的语义引导行为。

---

### Figure 6: 仿真 OOD 泛化对比（棋盘格纹理背景）

![Figure 6](https://arxiv.org/html/2608.04396v1/x6.png)

**说明**: 在从未见过的棋盘格纹理背景下，CofactVLA 成功完成任务，而基线方法因视觉分布漂移失效，体现 [[Out-of-Distribution Generalization|OOD 泛化]] 能力。

---

### Figure 7: 真实机器人 OOD 泛化可视化

![Figure 7](https://arxiv.org/html/2608.04396v1/x7.png)

**说明**: 真实机器人在新背景与新物体布局条件下的执行轨迹可视化，CofactVLA 保持语义引导的抓取准确性。

---

### Figure 8: 机器人平台

![Figure 8](https://arxiv.org/html/2608.04396v1/x8.png)

**说明**: 实验使用 AgileX PiPer 机械臂 + 双目相机（腕部摄像头 + 第三视角摄像头），约 400 条专家轨迹用于微调。

---

### Figure 9 & 10: 双视角数据采集样例

![Figure 9](https://arxiv.org/html/2608.04396v1/x9.png)

![Figure 10](https://arxiv.org/html/2608.04396v1/x10.png)

**说明**: 四个真实任务（Task I-IV）的双视角采集示例。Task I/II 见 Fig 9，Task III/IV 见 Fig 10，展示不同物体抓取与放置的操作场景。

---

### Table 1: LIBERO 基准测试（%）

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| **CofactVLA** | **99.0** | **100.0** | **98.0** | **97.0** | **98.5** |

**关键发现**: CofactVLA 超越所有基线，在 LIBERO-Object 上达到 100%，Long-horizon 任务提升显著（96.9%→97.0% vs π₀.₅），整体平均 98.5%。

---

### Table 2: LIBERO-Plus 零样本 OOD 测试（%）

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|------|--------|-------|----------|-------|-----------|-------|--------|-------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT (w) | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 55.8 |
| OpenVLA-OFT (m) | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| **CofactVLA** | **44.7** | **49.7** | **71.8** | **85.6** | **83.6** | **78.0** | **70.2** | **69.1** |

**关键发现**: CofactVLA 在 7 类 OOD 扰动中的总分最高（69.1%），相比 π₀ 基线（53.6%）提升 +15.5%，在 Robot（新机器人形态）和 Language（新指令描述）两维度表现尤为突出。

---

### Table 3: 消融实验（%）

**模块消融（LIBERO 基准）:**

| 配置 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| Baseline（π₀.₅） | 97.0 | 99.0 | 96.0 | 96.0 | 97.0 |
| w/ CCR only | 99.0 | 99.0 | 98.0 | 94.0 | 97.5 |
| w/ OPG only | 100.0 | 99.0 | 98.0 | 95.0 | 98.0 |
| Full CofactVLA | 99.0 | 100.0 | 98.0 | 97.0 | **98.5** |

**关键发现**: OPG 和 CCR 均独立有效，组合后在 Long-horizon 任务上进一步提升（97.0%），验证双层干预的互补性。

**反事实干预设计对比:**

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| Add（简单相加）| 98.0 | 98.0 | 92.0 | 88.0 | 94.0 |
| Sub（CFG风格相减） | 97.0 | 99.0 | 98.0 | 90.0 | 96.0 |
| CAG（条件注意力引导） | 98.0 | 96.0 | 99.0 | 97.0 | 97.5 |
| **OPG（Ours）** | **99.0** | **100.0** | **98.0** | **97.0** | **98.5** |

**关键发现**: OPG 通过几何正交投影保留动作流形结构，优于 CFG 风格的简单相减（+2.5%）和条件注意力引导（+1.0%）。

---

### Table 4: 训练超参数配置

| 超参数 | 值 | 超参数 | 值 |
|--------|-----|--------|-----|
| 计算资源 | 4×H100-96GB | 因果引导尺度 γ | 2.0 |
| 骨干网络 | π₀.₅ | 干预强度 β | 0.15 |
| Batch size | 32/GPU | 干预层 | [15, 16] |
| 学习率 | 2.5e-5 | 冻结视觉编码器 | 否 |
| 训练步数 | 6K | 冻结动作专家 | 否 |
| 权重衰减 | 0.01 | Action chunk size | 50 |
| 优化器 | AdamW | 预热步数 | 1K |
| Betas | [0.9, 0.95] | 梯度检查点 | 是 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 子集 × 10 任务 | 桌面操作，多任务 | 仿真测试（标准） |
| LIBERO-Plus | 7 类扰动 | 摄像机、机器人、语言、光照、背景、噪声、布局变化 | 仿真 OOD 测试 |
| 真实机器人数据集 | ~400 条专家轨迹 | AgileX PiPer，双目摄像头，4 个任务 | 真实机器人微调+测试 |

### 实现细节

- **Backbone**: [[Pi05|π₀.₅]]（基于 [[LeRobot]] 框架）
- **优化器**: AdamW，lr=2.5e-5，betas=[0.9, 0.95]，weight_decay=0.01
- **Batch Size**: 32/GPU
- **训练步数**: 6K steps，预热 1K steps
- **硬件**: 4× NVIDIA H100 (96GB)
- **CCR 干预层**: Transformer 第 15、16 层

### 可视化结果

- 标准仿真场景（Fig 5）：CofactVLA 正确遵循语言指令抓取目标物体，基线出现视觉覆盖错误
- OOD 背景（Fig 6）：新棋盘格纹理背景下 CofactVLA 维持正常操作，基线失效
- 真实机器人（Fig 7）：新布局和背景条件下轨迹稳定，语义引导鲁棒性强

---

## 批判性思考

### 优点

1. **理论基础扎实**: DDG 因果图 + 广义特征值分解提供严谨数学证明，非启发式方法。
2. **双层互补干预**: OPG 在动作解码期间实时干预，CCR 在特征提取阶段预处理，两种机制相互补充。
3. **训练无需额外标注**: 反事实分支通过屏蔽语言 token 在推理时获得，无需额外标注数据或辅助模型。
4. **OOD 提升显著**: 真实机器人 OOD 场景 +52.3% 绝对提升，实用价值明确。
5. **即插即用**: 仅需微调（6K steps），可方便地适配其他 [[Flow Matching|流匹配]] 型 VLA 基座。

### 局限性

1. **依赖 VLM 零样本能力**: CCR 需要 VLM 基座具备良好的零样本语言理解能力，弱基座效果下降。
2. **严重遮挡场景鲁棒性差**: 物体被大面积遮挡时，双路分支均无法获得有效视觉特征，未来需引入多视角融合。
3. **单臂操作限制**: 当前实验仅验证单臂操作，双臂协作场景尚未探索。
4. **额外推理开销**: 双路前向传播使推理计算量约翻倍，实时性有一定压力。

### 潜在改进方向

1. **多视角融合**: 引入多目摄像头以缓解遮挡导致的性能下降
2. **自适应 γ/β**: 根据语言-视觉对齐度动态调整干预强度，而非固定超参数
3. **扩展到非流匹配 VLA**: 将 OPG 思想推广至 AR（自回归）型 VLA 的 logit 空间

### 可复现性评估

- [ ] 代码开源（论文暂未提供 GitHub）
- [ ] 预训练模型（暂未提供）
- [x] 训练细节完整（Table 4 超参数完整）
- [x] 数据集可获取（LIBERO 使用 MIT 协议，LeRobot 使用 Apache 2.0）

---

## 关联笔记

### 基于

- [[Pi05|π₀.₅]]: 骨干 VLA 网络，CofactVLA 在其上进行微调
- [[Flow Matching]]: 核心生成框架，OPG 在其速度场空间操作

### 对比

- [[Classifier-Free Guidance|CFG]]: 相同的无条件引导思路，但简单相减破坏流形，OPG 通过正交投影解决此问题
- [[CAG]]: 条件注意力引导方法，同为动作级反事实干预，OPG 性能更优（97.5% vs 98.5%）
- [[X-VLA]]: LIBERO 最强基线（98.1%），CofactVLA 以 98.5% 超越
- [[Pi05|π₀.₅]]: 真实 OOD 基线（23.5%），CofactVLA 以 75.8% 大幅超越

### 方法相关

- [[Dual-path Deconfounding Graph|DDG]]: 本文提出的因果图形式化框架
- [[Orthogonal Projection Guidance|OPG]]: 动作级去混淆核心模块
- [[Counterfactual Covariance Reduction|CCR]]: 特征级去混淆核心模块
- [[因果推断|Causal Intervention]]: 整体方法的理论基础
- [[Score Function]]: 与流匹配速度场等价，支撑 OPG 理论

### 硬件/数据相关

- [[AgileX PiPer]]: 实验用机械臂平台
- [[LIBERO]]: 主要仿真 benchmark

---

## 速查卡片

> [!summary] CofactVLA (Tsinghua, 2026)
> - **核心**: 用双路反事实干预消除 VLA 中视觉覆盖现象
> - **方法**: DDG 因果建模 + OPG（动作级正交投影）+ CCR（特征级协方差消减）
> - **结果**: LIBERO 98.5% SOTA；真实机器人 OOD 场景 75.8%，比 π₀.₅ 高 +52.3%
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-08-07*

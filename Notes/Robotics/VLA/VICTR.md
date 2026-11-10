---
title: "VICTR: Value-Informed In-Context Learning for Vision-Language-Action Models"
method_name: "VICTR"
authors: [Luca Macesanu, Tianluo Zhang, Aditya Patil, Archit Sharma, Kevin Angers, Christopher M. Clark, Chen Feng, Juexiao Zhang]
year: 2026
venue: arXiv
tags: [vla, in-context-learning, retrieval, value-function, robot-manipulation, bimanual]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.08698
created: 2026-11-10
---

# 论文笔记：VICTR

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | New York University (ai4ce Lab) |
| 日期 | November 2026 |
| 项目主页 | [ai4ce.github.io/VICTR](https://ai4ce.github.io/VICTR) |
| 对比基线 | [[Pi0.5\|π0.5 fine-tuned]], [[RICL]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.08698) / Code（即将开源） |

---

## 一句话总结

> VICTR 通过 [[IC-VFE]]（In-Context Value Function Estimator）估计当前任务进度，从参考演示中检索同阶段示例块，驱动 [[Pi0.5|π0.5]] 进行上下文感知的动作生成，使未见域外任务成功率从 6.2% 提升至 23.8%。

---

## 核心贡献

1. **[[IC-VFE]]（In-Context Value Function Estimator）**: 以参考演示视频、语言指令和当前观测为输入，输出任务相对进度（0–1），无需测试时参数更新。
2. **进度对齐检索**: 利用 IC-VFE 的进度估计替代纯视觉相似度，从演示池中检索与当前执行阶段匹配的示范块，解决 [[RICL]] 在错误阶段检索的问题。
3. **VICTR 实机评估框架**: 在 16 个任务（8 域内 + 8 域外）的双臂操作平台 YOR 上验证，OOD 任务进度率 55.5%，远超视觉检索基线 42.1%。

---

## 问题背景

### 要解决的问题

预训练 [[VLA（视觉-语言-动作模型）|VLA]] 模型适配新任务通常需要收集大量演示并重新微调。[[In-Context Imitation Learning|基于检索的上下文学习（ICL）]] 允许通过少量参考演示直接条件化策略，但现有方法（如 [[RICL]]）使用视觉相似度选择示范块，可能检索到**错误执行阶段**的片段（例如在任务末尾检索到开始阶段的动作），导致策略混乱。

### 现有方法的局限

- **视觉相似度检索**（[[RICL]]）：外观相似的帧未必处于相同任务阶段，尤其在域外任务上失效明显。
- **基于时间的进度估计**（RECAP）：用时间戳归一化进度，对不同速度执行的任务缺乏语义对齐。
- **在线奖励模型**（Robometer、TOPReward）：不以参考演示为条件，无法感知当前检索上下文。

### 本文的动机

若能准确估计当前观测在任务执行进度中的位置，就可以有针对性地检索**同阶段**的演示块，使上下文信息对策略真正有用。VICTR 将"以参考演示为条件的进度估计"和"进度对齐检索"统一到一个轻量推理框架中。

---

## 方法详解

### 整体框架

VICTR 由两个核心组件构成：

- **[[IC-VFE]]**：给定参考演示 $\tau_{ctx}$、语言指令 $\ell$、当前观测 $o_t$，输出进度估计 $\hat{v}_t \in [0, 1]$
- **IC-VLA**：[[Pi0.5|π0.5]] 经过后训练，接收检索到的演示块 $c_t$、当前观测 $o_t$、指令 $\ell$，生成动作块 $a_{t:t+H}$

每个演示块事先用 IC-VFE 和 [[DINOv2]] 编码（进度值 + 视觉嵌入），构成可检索的索引。

### 模型架构：IC-VFE

**输入处理**：
- 视觉骨干：Gemma 3 4B 冻结视觉编码器，输入 $896 \times 896$，每视角输出 256 个 token
- 参考演示压缩：两层 patch-merging，将参考帧压缩为 16 个 summary token
- 总输入序列 $H^{(0)} = [C_1; \ldots; C_K; O_t; E_\ell; r]$，其中 $r$ 是价值查询 token

**解码器**：Gemma 3 270M，18 层，在第 6、12、18 层使用 [[Block Causal Attention|分块因果注意力]]；其余层块内使用双向注意力。

**输出**：价值查询 token 经分类头预测 $B = 201$ 个分档的概率分布，期望值作为进度估计。

### 检索策略

IC-VFE 对演示池中的每个块 $c_i$ 和当前观测 $o_t$ 均计算进度估计，取进度差最小的块：

- **纯进度检索**（VICTR）：选 $i^* = \arg\min_i |{\hat{v}_t - \hat{v}_i}|$
- **进度 + 视觉联合检索**（VICTR+vision）：加权融合 [[DINOv2]] 视觉距离和进度距离（z-score 归一化后相加）

### IC-VLA 训练

使用 oracle 进度采样上下文块，以 [[Flow Matching|流匹配]] 动作损失训练 [[Pi0.5|π0.5]]（[[FAST|FAST 动作 token]]，每块保留 20/30 个 DCT bins，动作块长 $H = 30$，20D 双臂动作，30 Hz）。

---

## 关键公式

### 公式 1–2: [[VLA（视觉-语言-动作模型）|策略与进度索引]]

$$
a_{t:t+H} = \pi_\theta(o_t, \ell)
$$

$$
\hat{v}_i = f_\phi(o_i, \ell, \tau_{ctx})
$$

**含义**：策略从观测和指令预测动作块；IC-VFE 对演示池中每帧计算进度估计作为索引。

**符号说明**:
- $a_{t:t+H}$: 时间步 $t$ 起的动作块，长度 $H$
- $\pi_\theta$: VLA 策略网络
- $f_\phi$: IC-VFE
- $\tau_{ctx}$: 参考演示上下文
- $\ell$: 语言指令

### 公式 3–5: [[IC-VFE|进度对齐检索]]

$$
\hat{v}_t = f_\phi(o_t, \ell, \tau_{ctx})
$$

$$
d_{val,i} = |\hat{v}_t - \hat{v}_i|
$$

$$
i_t^* = \arg\min_{i \in \{1,\ldots,M\}} d_{val,i}, \quad c_t = c_{i_t^*}
$$

**含义**：对当前观测估计进度，取演示池中进度最接近的块作为上下文。

**符号说明**:
- $d_{val,i}$: 当前进度与第 $i$ 块进度的差
- $M$: 演示块总数
- $c_t$: 检索到的上下文块

### 公式 6–8: [[DINOv2|视觉+进度联合检索]]

$$
d_{vis,i} = \|h_t - h_i\|_2
$$

$$
g_i = z(d_{vis,i}) + z(d_{val,i})
$$

$$
z(d_i) = \frac{d_i - \mu_d}{\sigma_d + 10^{-8}}
$$

**含义**：将 [[DINOv2]] 视觉距离和 IC-VFE 进度距离 z-score 归一化后求和，用于联合检索（VICTR+vision）。

**符号说明**:
- $h_t, h_i$: [[DINOv2]] 特征向量
- $z(\cdot)$: z-score 归一化

### 公式 9: [[In-Context Imitation Learning|IC-VLA 策略]]

$$
a_{t:t+H} = \pi_\theta^{ICL}(c_t, o_t, \ell)
$$

**含义**：IC-VLA 以检索到的演示块 $c_t$ 为额外上下文，生成动作块。

### 公式 10–11: [[IC-VFE|IC-VFE 输入与输出]]

$$
H^{(0)} = [C_1; \ldots; C_K; O_t; E_\ell; r]
$$

$$
\hat{v}_t = 1 + \sum_{b=0}^{B-1} \nu_b \cdot p_{\phi,b}(o_t, \ell, \tau_{ctx})
$$

**含义**：解码器输入为参考演示压缩 token + 当前观测 token + 语言 token + 价值查询 token；输出为 201 个分档的期望进度（偏移至 $[0,1]$）。

**符号说明**:
- $C_1, \ldots, C_K$: 参考演示帧的 summary token
- $r$: 价值查询 token
- $B = 201$: 进度分档数
- $\nu_b$: 第 $b$ 档的分档值
- $p_{\phi,b}$: IC-VFE 对第 $b$ 档的预测概率

### 公式 12–13: [[Value Function|IC-VFE 训练目标]]

$$
v_t = \frac{\text{clip}(r_t, 0, 100)}{100}, \quad b_t = \text{round}((B-1) \cdot v_t)
$$

$$
\mathcal{L}_{VFE}(\phi) = -\frac{1}{N} \sum_{j=1}^{N} \log p_{\phi, b_j}(o_j, \ell_j, \tau_{ctx}^{(j)})
$$

**含义**：用 [[Robo-Dopamine]]（特权离线教师，可见最终帧）生成软标签 $v_t$，IC-VFE 以交叉熵损失拟合离散化进度分布。

**符号说明**:
- $r_t$: Robo-Dopamine 给出的进度奖励（0–100）
- $b_t$: 目标分档索引
- $N$: 训练样本数

---

## 关键图表

### Figure 1: 方法动机与主要结果

![Figure 1: VICTR pull figure](https://arxiv.org/html/2611.08698v1/victr-pull-fig.png)

**说明**: IC-VFE 估计任务进度，驱动进度对齐检索。OOD 任务进度率从 π0.5 的 26.8%、视觉 RICL 的 43.0% 提升至 VICTR 的 58.2%。

### Figure 2: VICTR 整体架构

![Figure 2: VICTR overview](https://arxiv.org/html/2611.08698v1/blicl_method.png)

**说明**: IC-VFE 对当前观测评分，可选融合视觉相似度，检索匹配进度的演示块；IC-VLA 接收该块 token 和当前观测，通过 [[Block Causal Attention|分块因果注意力]] 生成动作。

### Figure 3: IC-VFE 注意力模式

**说明**: IC-VFE 解码器在大多数层块内独立处理参考帧和当前观测；在第 6、12、18 层通过 [[Block Causal Attention|分块因果注意力]] 跨块通信，实现上下文感知的进度估计。

### Figure 4: 域内 / 域外成功率与任务进度率

| 方法 | OOD 成功率 | OOD 进度率 | 域内成功率 | 域内进度率 |
|------|-----------|-----------|-----------|-----------|
| π0.5 fine-tuned | 6.2% | 28.9% | — | 82.2% |
| RICL | 15.0% | 42.1% | — | 87.5% |
| VICTR（进度） | **23.8%** | **55.5%** | — | ~90% |
| VICTR+vision（进度+视觉） | 21.2% | 47.8% | — | ~88% |

**说明**: 进度对齐检索在 OOD 任务上显著优于视觉检索，域内任务各方法差异较小。

### Figure 5: 检索上下文示例对比

![Figure 5: Retrieval examples](https://arxiv.org/html/2611.08698v1/retrieval_examples.png)

**说明**: DINOv2 在 circle-on-peg 任务上检索了错误阶段的示范块；IC-VFE 在 wipe-the-plate 任务上找到了阶段对齐的示范块，说明进度估计的实际效果。

### Figure 6: 不同进度估计方法的任务进度率

**说明**: 在相同 IC-VLA 策略下，IC-VFE 的 OOD 进度率（62.7%）显著高于 RECAP（~50%）、Robometer（~35%），验证了上下文条件化对进度估计质量的重要性。

### Figure 7: 演示池大小 vs 成功率

**说明**: 演示池从 1 增至 10，成功率总体递增，约在 5 个演示时趋于饱和，表明 VICTR 对演示数量的需求适中。

### Figure 8: 训练与部署检索信号的匹配性

**说明**: 训练与部署使用相同检索信号时性能最优；信号不匹配（如训练用进度、部署用视觉）仍有一定效果，但弱于匹配配置。

### Table I: VFE 质量（与人工进度曲线的一致性）

| 方法 | 域内 Kendall τ_b | 域内 Pearson r | 域外 Kendall τ_b | 域外 Pearson r |
|------|----------------|---------------|----------------|---------------|
| **IC-VFE（本文）** | **0.51** | **0.62** | **0.52** | **0.65** |
| 无上下文 | 0.26 | 0.36 | 0.38 | 0.49 |
| 无分块嵌入 | 0.46 | 0.55 | 0.37 | 0.45 |
| RECAP | 0.48 | 0.60 | 0.43 | 0.57 |
| Robometer | 0.26 | 0.34 | 0.27 | 0.38 |
| Robometer FT | 0.32 | 0.44 | 0.33 | 0.46 |

**关键发现**: IC-VFE 在域外任务上显著优于所有基线；去除上下文（参考演示）使 OOD 相关性从 0.52 降至 0.38，证明上下文条件化的必要性；去除分块嵌入使 OOD τ_b 从 0.52 降至 0.37。

### Table II: 离线奖励模型质量（训练监督信号）

| 方法 | 域内 Kendall τ_b | 域内 Pearson r | 域外 Kendall τ_b | 域外 Pearson r |
|------|----------------|---------------|----------------|---------------|
| [[Robo-Dopamine]] | 0.79 | 0.88 | 0.79 | **0.90** |
| Robo-Dopamine FT | 0.64 | 0.72 | 0.81 | 0.88 |
| Robometer | 0.56 | 0.70 | 0.58 | 0.73 |
| Robometer FT | 0.65 | 0.78 | 0.68 | 0.81 |
| TOPReward | **0.80** | 0.74 | **0.81** | 0.79 |
| GVL | 0.34 | 0.41 | 0.29 | 0.37 |

**关键发现**: [[Robo-Dopamine]] 是最强的离线教师，论文选其作为训练监督信号。TOPReward 的 Pearson r 偏低（0.79），可能因进度随时间退化。

### Table III: LIBERO-100（30 个未见任务）成功率

| 骨干 | 方法 | Top-k | 动作插值 | 成功率 |
|------|------|-------|---------|--------|
| π0-FAST | 非 ICL 基线 | — | — | 21.5% |
| π0-FAST | RICL | — | 是 | 32.6% |
| **π0.5** | **非 ICL 基线** | — | — | **37.8%** |
| π0.5 | RICL | 1 | 否 | 41.2% |
| **π0.5** | **VICTR** | **1** | **否** | **45.9%** |

**关键发现**: VICTR 在 LIBERO-100 上也优于 RICL（45.9% vs 41.2%）；动作插值对 π0.5 有害；k=1 比 k=4 更优。

### Table IV: 评测任务列表

| 类别 | 任务 |
|------|------|
| 域内（8） | 清洁盘子 / 右臂用锤子击打黄方块 / 左→右传递红辣椒 / 左臂将茄子放上盘子 / 将红辣椒放入杯中 / 将圆形放到圆柱上 / 左臂将茄子放入盒子 / 拔开蓝色马克笔盖 |
| 域外-新物体 | 将方块放到圆柱上 / 拔开黑色马克笔盖 / 将芥末酱放入盒子 / 用锤子击打茄子 |
| 域外-技能组合 | 将芥末酱传到盘子上 / 将方块放入盒子后击打 |
| 域外-相似未见任务 | 叠三个方块塔 |
| 域外-长时程 | 将物品分类放入容器 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| YOR 实机数据 | 28 个任务，1,180 episodes，7.5 小时 | 双臂真实操作 | 训练（7h）+ 评测演示池（30min） |
| LIBERO-100 | 100 个任务 | 仿真 benchmark，30 个未见任务用于评测 | 附录验证 |

### 实现细节

- **机器人平台**: YOR 双臂（两个 7-DoF 臂，20D 动作空间）
- **动作块长度**: $H = 30$，30 Hz
- **FAST 动作编码**: 每块保留 20/30 个 DCT bins
- **IC-VFE 解码器**: Gemma 3 270M，18 层，分块因果注意力
- **视觉编码器**: Gemma 3 4B（冻结），输入 $896 \times 896$
- **参考帧压缩**: 2 层 patch-merging → 16 个 summary token
- **训练监督**: [[Robo-Dopamine]] 离线教师，提供 0–100 进度奖励
- **演示池大小**: 10 个演示（评测时）

---

## 批判性思考

### 优点
1. **无需测试时更新**: IC-VFE 仅在推理时前向传播，不需要梯度更新，部署友好。
2. **进度估计可解释**: 进度估计在各任务上与人工标注高度相关（OOD Pearson r = 0.65），具备实际可用性。
3. **方法解耦**: 进度对齐检索模块理论上可与任意 ICL-ready VLA 结合，不绑定 π0.5。

### 局限性
1. **绝对成功率仍偏低**: OOD 任务成功率仅 23.8%（论文引言处为 25.0%，数字存在不一致）。
2. **单一物理环境**: 所有实验在同一实验室环境中进行，泛化性未充分验证。
3. **参考帧空间池化损失细节**: 将参考视频压缩到 16 个 token 可能丢失精细视觉信息，限制进度估计精度。
4. **数据不一致**: 论文不同位置引用的数字（成功率、进度率）不完全一致，实验严谨性存疑。

### 潜在改进方向
1. 引入高效注意力或测试时训练（TTT）适应参考演示。
2. 使用人类演示视频作为参考，降低机器人演示采集成本。
3. 在更多环境和机器人平台上验证泛化性。

### 可复现性评估
- [ ] 代码开源（项目页面链接尚为占位符）
- [ ] 预训练模型（未开放）
- [x] 训练细节基本完整
- [ ] 实机数据集（未公开）

---

## 关联笔记

### 基于
- [[Pi0.5|π0.5]]: IC-VLA 以 π0.5 为策略骨干，通过后训练支持上下文生成
- [[RICL]]: VICTR 建立在 RICL 框架之上，以进度检索替代视觉检索
- [[Robo-Dopamine]]: 用作 IC-VFE 的训练监督（特权离线教师）

### 对比
- [[RICL]]: 视觉相似度检索，OOD 进度率 42.1% vs VICTR 55.5%
- [[In-Context Imitation Learning|RECAP]]: 时间归一化进度，OOD τ_b 0.43 vs IC-VFE 0.52

### 方法相关
- [[IC-VFE]]: 核心模块，上下文条件化价值函数估计
- [[Block Causal Attention]]: IC-VFE 解码器的关键架构组件
- [[DINOv2]]: 视觉相似度检索的骨干特征提取
- [[FAST]]: IC-VLA 使用的动作 token 格式
- [[Flow Matching]]: IC-VLA 策略的动作生成方式
- [[Value Function]]: IC-VFE 的理论基础

### 硬件/数据相关
- YOR 双臂平台（NYU ai4ce Lab）
- LIBERO-100（附录仿真验证）

---

## 速查卡片

> [!summary] VICTR
> - **核心**: 进度对齐检索解决 VLA 上下文学习的阶段错配问题
> - **方法**: IC-VFE 估计任务相对进度 → 检索同阶段演示块 → π0.5 ICL 策略生成动作
> - **结果**: OOD 成功率 6.2% → 23.8%，进度率 28.9% → 55.5%（16 个双臂操作任务）
> - **代码**: [ai4ce.github.io/VICTR](https://ai4ce.github.io/VICTR)（即将开源）

---

*笔记创建时间: 2026-11-10*

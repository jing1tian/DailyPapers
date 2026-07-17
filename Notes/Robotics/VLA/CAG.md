---
title: "When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs"
method_name: "CAG"
authors: [Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, Mingyu Ding]
year: 2026
venue: arXiv
tags: [vla, counterfactual-robustness, language-grounding, diffusion-policy, benchmark, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2602.17659v2
created: 2026-07-17
---

# 论文笔记：When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of North Carolina at Chapel Hill |
| 日期 | February 2026 |
| 项目主页 | [vla-cf.github.io](https://vla-cf.github.io/) |
| 对比基线 | [[OpenVLA-OFT]], [[π₀]], [[π₀.₅]], [[X-VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.17659) / Code N/A |

---

## 一句话总结

> 提出 LIBERO-CF 反事实评测基准，揭示 [[VLA（视觉-语言-动作模型）]] 普遍依赖视觉捷径而忽略语言指令，并以推理时双分支 [[Counterfactual Action Guidance]] (CAG) 有效缓解这一问题。

---

## 核心贡献

1. **LIBERO-CF 基准**: 首个专为 [[VLA（视觉-语言-动作模型）]] 语言跟随能力设计的反事实评测基准，含 CF-Spatial / CF-Object / CF-Long / CF-OOD 四套任务
2. **视觉捷径诊断**: 系统性实验证明所有主流 VLA（[[OpenVLA-OFT]]、[[π₀]]、[[π₀.₅]]）均存在严重的视觉模态主导现象
3. **CAG 推理方案**: 无需修改模型架构，推理时融合有条件/无条件两路输出，以 [[Classifier-Free Guidance (CFG)]] 思路增强语言条件化

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 在遭遇"反事实"指令（即与训练场景不同的语言指令）时，倾向于执行学习过的场景习惯动作，而非真正理解并遵循语言。

### 现有方法的局限

- 现有评测（如 LIBERO、LIBERO-PLUS）测试的是相同任务的语义等价改写，无法暴露视觉捷径
- 数据增强方案（如 CAST）依赖大规模重新训练，成本高昂
- 尚无通用、架构无关的推理时解决方案

### 本文的动机

从 [[贝叶斯推断]] 角度分析：理想条件分布 $P(a|o,l)$ 应同时依赖观测 $o$ 和语言 $l$，但实际 VLA 因训练数据的语言-视觉相关性偏置而退化为近似仅依赖视觉的 $p(a|o)$。[[Classifier-Free Guidance (CFG)]] 正是一种利用无条件估计放大条件信号的成熟机制，可直接借鉴。

---

## 方法详解

### 核心思路：视觉捷径的 Bayesian 视角

当前 [[VLA（视觉-语言-动作模型）]] 在推理时实际表现为：

$$
p(a \mid o, l) \approx p(a \mid o)
$$

这是因为训练数据中视觉特征与任务之间的强相关，导致模型学会了"看场景猜任务"。理想的条件分布应满足 [[贝叶斯推断|贝叶斯分解]]：

$$
P(a \mid o, l) \propto P(a \mid o) \cdot P(l \mid a, o)
$$

其中 $P(a \mid o)$ 是视觉先验，$P(l \mid a, o)$ 是语言-动作似然项（模型对"给定视觉和动作，语言有多合理"的衡量）。

### CAG 推理公式

[[Counterfactual Action Guidance]] (CAG) 直接类比 [[Classifier-Free Guidance (CFG)]]，在推理阶段将有条件输出和无条件输出做加权外推：

$$
\pi_{\mathrm{CAG}}(a \mid o, l) = \pi_{\mathrm{uncond}}(a \mid o, \emptyset) + \omega \cdot \bigl(\pi_{\mathrm{cond}}(a \mid o, l) - \pi_{\mathrm{uncond}}(a \mid o, \emptyset)\bigr)
$$

其中 $\omega > 1$ 是[[引导尺度]]（guidance scale）。该公式等价于对语言似然项的指数放大：

$$
P_{\mathrm{CAG}}(a \mid o, l) \propto P(a \mid o) \cdot P(l \mid a, o)^{\omega}
$$

对数空间形式（用于扩散模型得分估计）：

$$
\log P_{\mathrm{CAG}}(a \mid o, l) = \log P(a \mid o) + \omega \cdot \bigl(\log P(a \mid o, l) - \log P(a \mid o)\bigr)
$$

### 两种实现策略

#### Training-Free (TF) 策略

直接复用已训练的 VLA，推理时将语言输入置空（$l = \emptyset$）得到 $\pi_{\mathrm{uncond}}$。优点是完全无需额外训练；缺点是语言被强制置空可能导致 [[分布外泛化|分布外]]（OOD）推断。

#### Vision-Action (VA) 策略

额外训练一个**仅视觉输入**的 [[扩散策略|扩散模型]] 作为专用无条件分支，与原始 VLA 组合使用。VA 分支接受完全相同的视觉观测，但从不接收语言。训练时语言输入被完全去除，其余与原始 VLA 相同。

### 架构兼容性

CAG 不依赖任何特定架构，已在以下 VLA 上验证：

| 模型 | 参数量 | 架构 | 最优 ω |
|------|--------|------|--------|
| [[OpenVLA-OFT]] | 7B | Prismatic VLM + LoRA | 3.0 |
| [[π₀]] | 3B | 扩散模型 | 1.5 |
| [[π₀.₅]] | 3B | 扩散模型 | 1.5 |
| [[X-VLA]] | 0.9B | 跨体态模型 | 1.5 |

---

## 关键公式

### 公式1：[[贝叶斯推断|Bayesian 条件分解]]

$$
P(a \mid o, l) \propto P(a \mid o) \cdot P(l \mid a, o)
$$

**含义**: 理想的动作分布应由视觉先验和语言似然共同决定；当前 VLA 因训练偏置导致语言似然项退化

**符号说明**:
- $a$: 机器人动作
- $o$: 视觉观测
- $l$: 自然语言指令
- $P(a \mid o)$: 视觉动作先验（无语言）
- $P(l \mid a, o)$: 语言-动作一致性似然

### 公式2：[[Counterfactual Action Guidance|CAG 推理公式]]

$$
\pi_{\mathrm{CAG}}(a \mid o, l) = \pi_{\mathrm{uncond}}(a \mid o, \emptyset) + \omega \cdot \bigl(\pi_{\mathrm{cond}}(a \mid o, l) - \pi_{\mathrm{uncond}}(a \mid o, \emptyset)\bigr)
$$

**含义**: 从无条件基线出发，沿"有条件 − 无条件"的方向以步长 ω 外推，放大语言信号

**符号说明**:
- $\pi_{\mathrm{cond}}(a \mid o, l)$: 原始 VLA 的有语言条件输出
- $\pi_{\mathrm{uncond}}(a \mid o, \emptyset)$: 无语言条件的视觉输出（TF 置空或 VA 模型）
- $\omega$: [[引导尺度]]，控制语言强化程度；$\omega=1$ 退化为原始策略

### 公式3：[[Classifier-Free Guidance (CFG)|CFG 通用形式]]

$$
f_{\mathrm{CFG}}(x \mid c) = f(x \mid \emptyset) + s \cdot \bigl(f(x \mid c) - f(x \mid \emptyset)\bigr)
$$

**含义**: CAG 是 CFG 在动作空间的直接应用，$s$ 对应 ω，条件 $c$ 对应语言指令 $l$

**符号说明**:
- $f(x \mid \emptyset)$: 无条件得分估计
- $f(x \mid c)$: 有条件得分估计
- $s > 1$: 引导尺度

### 公式4：[[Counterfactual Action Guidance|CAG 对数空间等价形式]]

$$
\log P_{\mathrm{CAG}}(a \mid o, l) = \log P(a \mid o) + \omega \cdot \bigl(\log P(a \mid o, l) - \log P(a \mid o)\bigr)
$$

**含义**: 对数空间形式便于扩散模型在去噪步骤中直接操作得分函数

**符号说明**:
- 等价于 $P_{\mathrm{CAG}}(a \mid o, l) \propto P(a \mid o) \cdot P(l \mid a, o)^{\omega}$

---

## 关键图表

### Figure 1: 方法概览

![Figure 1](https://arxiv.org/html/2602.17659v2/x1.png)

**说明**: (a) VLA 的反事实失败示意：即使指令改变，VLA 仍执行习惯动作；(b) LIBERO-CF 基准四套任务；(c) CAG 双分支推理框架；(d) 跨多种 VLA 架构的实验验证概述。

### Figure 2: 视觉捷径的实证证据

![Figure 2](https://arxiv.org/html/2602.17659v2/x2.png)

**说明**: (a) 50 次试验中抓取位置分布的热力图——VLA 在反事实指令或空指令下仍执行训练任务；(b) 移除训练对象后，反事实指令的成功率大幅提升（CF-Spatial grounding 提升 25.4%，success 提升 20.0%）。

### Figure 3: CAG 方法图解

![Figure 3](https://arxiv.org/html/2602.17659v2/x3.png)

**说明**: [[Counterfactual Action Guidance]] 的双分支推理流程。上分支为原始 VLA（接收语言 + 视觉），下分支为无条件分支（仅视觉，TF 策略置空语言或 VA 策略用独立模型）。推理时以引导尺度 ω 对两路输出做加权外推。

### Figure 4: 引导尺度 ω 的影响分析

![Figure 4](https://arxiv.org/html/2602.17659v2/x4.png)

**说明**: 随 ω 增大，语言跟随能力（grounding rate）单调提升；但过大的 ω 导致任务成功率下降（过度引导）。[[OpenVLA-OFT]] 需要较大的 ω=3.0（视觉偏置更强），[[π₀]]/[[π₀.₅]] 在 ω=1.5 时最优。

### Figure 5: 真实机器人实验

![Figure 5](https://arxiv.org/html/2602.17659v2/x5.png)

**说明**: Franka Research 3 + Robotiq-2F85 夹爪平台，ZED 2i 立体相机 + ZED Mini 腕部相机。测试四个维度：物体识别（+13.3% success）、空间推理（+16.6% grounding, +13.3% success）、目标定向（+36.7% success）、OOD 泛化（零样本有效缓解）。

### Table I: LIBERO 基准上的视觉捷径证据

| 模型 | 输入模态 | Spatial | Object | Goal | Long | Avg. |
|------|----------|---------|--------|------|------|------|
| OpenVLA-OFT | V+L | 97.6% | 98.4% | 97.9% | 94.5% | 97.1% |
| | V only | 83.2% | 98.2% | 10.0% | 85.0% | 69.1% |
| | L only | 0.0% | 0.0% | 3.4% | 0.0% | 0.9% |
| π₀ | V+L | 96.8% | 98.8% | 95.8% | 85.2% | 94.2% |
| | V only | 28.0% | 26.8% | 3.4% | 15.6% | 18.5% |
| | L only | 0.0% | 0.0% | 2.2% | 0.0% | 0.6% |
| π₀.₅ | V+L | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| | V only | 68.2% | 56.6% | 10.2% | 71.6% | 51.7% |
| | L only | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**关键发现**: 所有 VLA 在仅视觉输入下仍能完成大量任务，但仅语言输入时成功率接近 0，证明模型几乎不依赖语言进行策略决策。

### Table II: 移除训练对象的影响

| 模型 | 指标 | CF-Spatial | CF-Focused (移除后) |
|------|------|-----------|-------------------|
| OpenVLA-OFT | Grounding | 6.8% | 31.9% |
| | Success | 1.1% | 14.4% |
| π₀ | Grounding | 70.9% | 92.7% |
| | Success | 34.4% | 51.1% |
| π₀.₅ | Grounding | 39.3% | 68.5% |
| | Success | 24.4% | 54.4% |
| 平均 | Grounding | 39.0% | 64.4% (+25.4%) |
| | Success | 20.0% | 40.0% (+20.0%) |

**关键发现**: 场景中存在训练对象会显著干扰 VLA 的语言跟随，是视觉捷径的直接来源。

### Table III: LIBERO-CF 完整评测结果（π₀.₅）

| 指标 | CAG | CF-Spatial F | CF-Spatial B | CF-Object F | CF-Object B | CF-Long F | CF-Long B | CF-OOD F | CF-OOD B | Avg. F | Avg. B |
|------|-----|-------------|-------------|------------|------------|----------|----------|---------|---------|--------|--------|
| Grounding | / (baseline) | 39.3% | 61.3% | 11.4% | 68.8% | 51.6% | 58.4% | 20.7% | 74.0% | 30.8% | 65.6% |
| | TF | 52.0% | 45.3% | 24.6% | 65.2% | 60.7% | 56.7% | 24.8% | 71.9% | 40.5% | 59.8% |
| | VA | 50.7% | 45.2% | 34.0% | 49.8% | 64.2% | 61.1% | 36.4% | 52.8% | **46.3%** | 52.2% |
| Success | / (baseline) | 24.4% | 56.9% | 5.8% | 66.8% | 15.8% | 50.4% | 6.9% | 69.6% | 13.2% | 60.9% |
| | TF | 27.3% | 33.3% | 9.2% | 41.4% | 20.9% | 36.7% | 9.9% | 62.7% | 16.8% | 43.5% |
| | VA | 31.6% | 36.0% | 18.0% | 29.8% | 26.7% | 41.8% | 10.3% | 37.1% | **21.7%** | 36.2% |

F = Faithful（指令正确），B = Biased（指令指向训练对象）。**关键发现**: VA 策略在忠实任务上大幅提升，但对偏向训练任务的指令执行率（Biased）下降，符合预期。

### Table III（附：OpenVLA-OFT 和 π₀）

**OpenVLA-OFT**:

| 指标 | CAG | Avg. Faithful | Avg. Biased |
|------|-----|---------------|-------------|
| Grounding | / | 4.7% | 83.6% |
| | TF | 5.8% | 81.9% |
| | VA | **11.3%** | 65.4% |
| Success | / | 0.4% | 78.6% |
| | TF | 0.9% | 76.3% |
| | VA | **2.1%** | 41.0% |

**π₀**:

| 指标 | CAG | Avg. Faithful | Avg. Biased |
|------|-----|---------------|-------------|
| Grounding | / | 28.8% | 63.0% |
| | TF | 31.9% | 59.3% |
| | VA | **37.3%** | 52.0% |
| Success | / | 9.6% | 45.0% |
| | TF | 9.6% | 27.6% |
| | VA | **10.4%** | 25.8% |

### Table IV: 消融实验——不同训练策略（π₀.₅）

| 策略 | 说明 | Grounding (CF-Spatial F) | Success (CF-Spatial F) |
|------|------|-------------------------|----------------------|
| R1 | 原始 π₀.₅（baseline） | 39.3% | 24.4% |
| R2 | 用完整 VLA 训练无条件分支（+语言） | 9.5% | 3.7% |
| R3 | TF 策略（置空语言） | 52.0% | 27.3% |
| R4 | VA 部分微调 | 52.3% | 27.7% |
| R5 | VA 完整微调（最终方案） | 50.7% | 31.6% |

**关键发现**: R2 策略（无条件分支仍接受语言）反而使结果下降，证明无条件分支必须与语言完全解耦。TF 和 VA 都优于 baseline，VA 完整微调在 success 上略优。

### Table V: X-VLA 上的 CAG 结果

| 指标 | CAG | Avg. Faithful | Avg. Biased |
|------|-----|---------------|-------------|
| Grounding | / | 37.3% | 56.1% |
| | TF | 39.7% | 55.6% |
| | VA | **41.0%** | 51.6% |
| Success | / | 13.8% | 39.3% |
| | TF | **19.5%** | 42.6% |
| | VA | 17.1% | 31.1% |

**关键发现**: 即使对 0.9B 的轻量跨体态模型，CAG 也带来一致提升。

### Table VI: 真实机器人场景设置

| 维度 | 场景 | 训练任务 | 反事实任务 |
|------|------|----------|-----------|
| 物体识别 | 可乐/雪碧/芬达 | 抓可乐 | 抓雪碧/芬达 |
| | 胶带/芥末/薯片 | 抓胶带 | 抓芥末/薯片 |
| 空间推理 | 中/左/右 | 抓中间杯 | 抓左/右杯 |
| | 桌/盘/碗 | 抓桌上罐 | 抓盘中/碗里罐 |
| 目标定向 | 叠杯/放盘/放篮 | 叠杯 | 放盘/放篮 |
| OOD 泛化 | 杯/魔方/篮球 | 抓杯子 | 抓魔方/篮球 |
| 长时序 | 移动并倒水 | 移右倒可乐 | 倒雪碧/芬达 |
| | 苹果与香蕉 | 苹果然后香蕉 | 香蕉然后苹果/仅苹果 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 130 任务 × 50 演示 | 桌面操作，标准 LIBERO 套件 | 原始训练/标准评测 |
| LIBERO-CF | 50 任务（4 套件） | 反事实指令，专为语言跟随设计 | 反事实评测 |
| 真实机器人数据 | 场景特定 | Franka FR3，实际演示 | 真实机器人训练 |

### 实现细节

- **OpenVLA-OFT**: 7B 参数，Prismatic VLM，[[LoRA]] 微调 300k 步，ω=3.0
- **π₀ / π₀.₅**: 3B 参数，扩散模型，完整微调 30k 步，ω=1.5，[[指数移动平均|EMA]] 衰减 0.999
- **X-VLA**: 0.9B 参数，跨体态，微调 50k 步，ω=1.5
- **VA 分支训练**: Batch size 16-32，学习率 1e-4 ~ 5e-5，完全去除语言输入
- **真实机器人**: Franka Research 3 + Robotiq-2F85 夹爪，ZED 2i 立体相机 + ZED Mini 腕部相机

### 可视化结果

真实机器人实验中，CAG 显著改善了物体识别（物品混淆率下降）、空间推理（左/右/中定位准确率提升）和 OOD 泛化（对未见物体如魔方、篮球的语言响应能力）。

---

## 批判性思考

### 优点

1. **问题挖掘深刻**: 以 Bayesian 视角清晰阐明视觉捷径的成因，而非仅描述现象
2. **无架构修改**: 推理时方案，可即插即用于任意 VLA，无需重训练原始模型
3. **多模型验证**: 跨 4 种主流架构（autoregressive、diffusion、跨体态）一致有效
4. **真实机器人实验**: 不止于仿真，在 Franka 平台上验证了实际价值

### 局限性

1. **VA 分支需要额外训练**: 虽然 TF 版无需训练，但效果最好的 VA 版仍需为每个 VLA 训练一个专用视觉分支
2. **整体成功率仍然偏低**: 在反事实任务上最好成绩也仅 21.7%（π₀.₅+CAG VA），说明 VLA 的语言跟随能力本身还有很大提升空间
3. **Biased success 下降明显**: CAG 增强忠实任务的代价是对"恰好与训练任务一致"的指令执行率下降，存在 tradeoff

### 潜在改进方向

1. 在训练阶段加入对比语言-视觉数据，从根本上解决视觉捷径
2. 将 CAG 与反事实数据增强（如 CAST）结合，形成训练+推理双重强化
3. 自适应引导尺度（根据指令与视觉的相关性动态调整 ω）

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码状态未确认）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（论文中提供了完整的超参数）
- [x] 数据集可获取（LIBERO 公开，LIBERO-CF 随论文发布）

---

## 关联笔记

### 基于

- [[Classifier-Free Guidance (CFG)]]: CAG 的核心推理机制直接类比于生成模型中的 CFG
- [[LIBERO]]: 仿真评测使用 LIBERO 环境，LIBERO-CF 是其反事实扩展
- [[π₀.₅]]: 主要评测和消融实验基于此模型

### 对比

- [[OpenVLA-OFT]]: 7B autoregressive VLA，视觉偏置最严重（V-only 69.1%）
- [[π₀]]: 3B 扩散 VLA，视觉偏置较轻（V-only 18.5%），CAG 增益也较小
- [[X-VLA]]: 轻量跨体态 VLA，验证方法的通用性

### 方法相关

- [[Counterfactual Action Guidance]]: 本文提出的核心方法
- [[贝叶斯推断]]: 视觉捷径的理论分析框架
- [[引导尺度]]: 控制 CAG 语言放大强度的关键超参数

### 硬件/数据相关

- [[Franka Research 3]]: 真实机器人平台
- [[LIBERO]]: 仿真评测基准

---

## 速查卡片

> [!summary] When Vision Overrides Language (CAG)
> - **核心**: VLA 存在视觉捷径问题，忽略语言指令
> - **方法**: CAG = 类比 CFG 的双分支推理，无需修改架构
> - **结果**: π₀.₅ 平均 grounding +15.5pp（30.8%→46.3%），success +8.5pp（13.2%→21.7%）
> - **代码**: [vla-cf.github.io](https://vla-cf.github.io/)

---

*笔记创建时间: 2026-07-17*

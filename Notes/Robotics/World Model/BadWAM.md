---
title: "BadWAM: When World-Action Models Dream Right but Act Wrong"
method_name: "BadWAM"
authors: [Qi Li, Xingyi Yang, Xinchao Wang]
year: 2026
venue: arXiv
tags: [adversarial-attack, world-action-model, embodied-ai, robustness, security]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.15207
created: 2026-07-19
---

# 论文笔记：BadWAM: When World-Action Models Dream Right but Act Wrong

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore |
| 日期 | July 2026 |
| 项目主页 | — |
| 对比基线 | [[BadVLA]]、随机扰动 |
| 链接 | [arXiv](https://arxiv.org/abs/2607.15207) |

---

## 一句话总结

> BadWAM 首次揭示 [[WAM|World-Action Model]] 特有的"世界动作漂移"漏洞：小幅视觉扰动可在保持预测未来视觉合理性的同时，劫持动作输出导致任务彻底失败。

---

## 核心贡献

1. **识别 WAM 特有攻击面**：提出 [[World-Action Drift]] 概念，指出 WAM 的动作输出可在想象未来保持正常的情况下被单独劫持，这是传统 VLA 攻击研究未曾揭示的新漏洞类。
2. **双变体统一攻击框架**：设计纯动作攻击（Action-Only Attack）和保形攻击（[[Imagination-Preserving Attack]]），覆盖强度—隐蔽性 tradeoff 谱。
3. **全面实证分析**：在 [[LIBERO]] 和 [[RoboTwin]] 上对三种 WAM 架构变体（Action-only、Joint、[[IDM|IDM WAM]]）系统评估，揭示空间与长时程任务最易受攻击，并验证现有防御手段不足。

---

## 问题背景

### 要解决的问题

[[WAM|World-Action Model]] 被设计为同时预测未来世界状态和动作序列，研究者希望借助世界预测作为天然安全信号——若预测未来合理，动作应该也合理。BadWAM 质疑这一假设是否成立。

### 现有方法的局限

针对反应式策略（reactive policy）的对抗攻击研究（如对 [[对抗补丁攻击]] 和 [[PGD]] 的研究）均直接针对动作输出，而未考虑 WAM 特有的世界预测-动作耦合结构。攻击 WAM 需要同时处理：(1) 查询型黑盒设定（无梯度访问）；(2) 闭环累积效应；(3) 如何在扰动动作时维持预测未来不变。

### 本文的动机

实证观察（Figure 1）揭示：失败轨迹对应显著更大的动作位移，但成功/失败轨迹的预测未来位移几乎重叠。这表明 **动作偏移是失败的强信号，而想象漂移不是**，为世界动作漂移攻击提供了实验依据。

---

## 方法详解

### 威胁模型

BadWAM 在 **查询型黑盒** 设定下运行：

- **输入访问权限**：可读取并修改观测 $o_t$
- **输出访问权限**：可查询动作输出 $a_{t:t+H-1}$ 和（可选）解码后的预测未来帧 $v_{t+1:t+K}$
- **无梯度访问**：不假设可访问模型权重或内部激活
- **扰动约束**：$\ell_\infty$ 范数约束 $\|\delta_t\|_\infty \leq \varepsilon$，$\varepsilon = 0.06$（像素范围 0–1）

### WAM 的三种架构变体

在 [[WAM]] 的统一框架下，论文针对三种架构变体进行攻击：

1. **Action-only WAM**：标准反应式策略，仅输出 [[Action Chunking|动作块]]，无世界预测头
2. **Joint WAM**：联合建模未来状态与动作的联合分布 $p_\theta(z_{t+1:t+K}, a_{t:t+H-1} \mid o_t, g)$
3. **IDM WAM**（[[IDM|Inverse Dynamics Model]] 范式）：先预测未来 $z_{t+1:t+K}$，再通过逆动力学模型推断动作 $a_{t:t+H-1}$

### 核心攻击流程

BadWAM 在每个重规划步骤在线优化扰动 $\delta_t$：

1. 从当前观测 $o_t$ 初始化扰动 $\delta_t = 0$
2. 通过 [[Zeroth-Order Gradient Estimation|零阶梯度估计]] 迭代更新 $\delta_t$（8 次迭代，每步查询 WAM 2 次，共 17 次查询）
3. 将 $\tilde{o}_t = \text{clip}(o_t + \delta_t)$ 注入 WAM，执行偏移后的动作
4. 在下一个重规划点重新执行上述过程

---

## 关键公式

### 公式 1: [[WAM|反应式策略]] 基线

$$
a_{t:t+H-1} \sim \pi_\theta(o_t, g)
$$

**含义**: 标准反应式策略将当前观测 $o_t$ 和语言目标 $g$ 映射为动作块。

**符号说明**:
- $a_{t:t+H-1}$：动作块，时间步 $t$ 到 $t+H-1$
- $\pi_\theta$：参数为 $\theta$ 的策略网络
- $g$：语言指令

---

### 公式 2: [[WAM|联合 WAM 分布]]

$$
(z_{t+1:t+K}, a_{t:t+H-1}) \sim p_\theta(z_{t+1:t+K}, a_{t:t+H-1} \mid o_t, g)
$$

**含义**: Joint WAM 对未来潜状态与动作序列的联合分布建模，两个输出相互条件化。

**符号说明**:
- $z_{t+1:t+K}$：未来 $K$ 步的潜在世界状态
- $p_\theta$：联合概率分布（参数 $\theta$）

---

### 公式 3: [[WAM|IDM WAM 顺序推理]]

$$
z_{t+1:t+K} \sim W_\theta(o_t, g), \quad a_{t:t+H-1} \sim A_\theta(o_t, g, z_{t+1:t+K})
$$

**含义**: IDM WAM 先生成未来世界状态，再将其作为条件输入逆动力学模型推断动作。

**符号说明**:
- $W_\theta$：世界预测模块
- $A_\theta$：动作推断模块（[[IDM]]）

---

### 公式 4: [[对抗补丁攻击|反应式策略攻击]]（文献基线）

$$
\max_{\|\delta\|_\infty \leq \varepsilon} D\bigl(\pi_\theta(o_t + \delta, g),\ \pi_\theta(o_t, g)\bigr)
$$

**含义**: 传统对动作输出的对抗攻击，无需考虑世界预测。

**符号说明**:
- $\delta$：视觉扰动
- $\varepsilon$：扰动预算（$\ell_\infty$ 范数）
- $D$：距离度量（如 $\ell_2$）

---

### 公式 5: 带扰动的观测

$$
\tilde{o}_t = \text{clip}(o_t + \delta_t), \quad \|\delta_t\|_\infty \leq \varepsilon
$$

**含义**: 将有界扰动叠加到原始观测并裁剪到合法像素范围。

**符号说明**:
- $\text{clip}(\cdot)$：将像素值投影到 $[0, 1]$

---

### 公式 6: BadWAM 约束优化（原始形式）

$$
\max_{\|\delta\|_\infty \leq \varepsilon} D_{\text{act}}(a^\delta,\ a) \quad \text{s.t.} \quad D_{\text{img}}(z^\delta,\ z) \leq \tau_{\text{img}}
$$

**含义**: [[World-Action Drift|世界动作漂移攻击]] 的统一框架：最大化动作偏移量，同时约束想象漂移不超过阈值。

**符号说明**:
- $D_{\text{act}}$：动作空间距离度量
- $D_{\text{img}}$：想象空间距离度量
- $\tau_{\text{img}}$：想象漂移上界
- 上标 $\delta$ 表示扰动后的模型输出

---

### 公式 7: 纯动作攻击（Action-Only Attack）

$$
\delta_t^* = \arg\max_{\|\delta_t\|_\infty \leq \varepsilon} D_{\text{act}}(a_{t:t+H-1}^\delta,\ a_{t:t+H-1})
$$

**含义**: 无想象约束的版本，直接最大化动作偏移，攻击强度最强但隐蔽性较低。

---

### 公式 8: [[Imagination-Preserving Attack|保形攻击]]（Lagrangian 松弛形式）

$$
\delta_t^* = \arg\max_{\|\delta_t\|_\infty \leq \varepsilon} \Bigl[ D_{\text{act}}(a_{t:t+H-1}^\delta,\ a_{t:t+H-1}) - \lambda\, D_{\text{img}}(z_{t+1:t+K}^\delta,\ z_{t+1:t+K}) \Bigr]
$$

**含义**: 将约束化为惩罚项，$\lambda$ 控制动作破坏力与想象隐蔽性之间的权衡。

**符号说明**:
- $\lambda$：Lagrange 乘子，控制隐蔽性-强度权衡（论文中测试 $\lambda \in \{0.1, 0.5, 1.0, 2.0\}$）

---

### 公式 9: 想象距离（帧级）

$$
D_{\text{img}}(v_{t+1:t+K}^\delta,\ v_{t+1:t+K}) = \frac{1}{K} \sum_{k=1}^{K} d_{\text{frame}}(v_{t+k}^\delta,\ v_{t+k})
$$

**含义**: 将扰动前后的解码视频帧逐帧比较，取 $K$ 步平均作为想象漂移指标。

**符号说明**:
- $v_{t+k}$：第 $t+k$ 帧的解码视频帧
- $d_{\text{frame}}$：帧级距离函数（如像素 MSE）

---

### 公式 10: [[Zeroth-Order Gradient Estimation|零阶梯度估计]]

$$
\hat{\nabla}J(\delta_t) = \frac{1}{m} \sum_{i=1}^{m} \frac{J(\delta_t + c u_i) - J(\delta_t - c u_i)}{2c} \cdot u_i
$$

**含义**: 在无梯度访问的黑盒设定下，通过有限差分法估计目标函数梯度。

**符号说明**:
- $\hat{\nabla}$：估计梯度
- $J$：攻击目标函数（公式 7 或 8）
- $u_i$：随机 Rademacher 方向
- $c$：有限差分半径
- $m$：每步采样方向数

---

### 公式 11: 投影梯度更新

$$
\delta_t \leftarrow \Pi_{[-\varepsilon, \varepsilon]}\bigl(\delta_t + \eta\,\hat{\nabla}J(\delta_t)\bigr)
$$

**含义**: 梯度上升后将扰动投影回 $\ell_\infty$ 可行集，保证约束满足。

**符号说明**:
- $\Pi_{[-\varepsilon, \varepsilon]}$：投影算子，逐元素裁剪到 $[-\varepsilon, \varepsilon]$
- $\eta$：步长

---

### 公式 12: 任务成功率（Task Success Rate）

$$
\text{Success} = \frac{1}{T} \sum_{i=1}^{T} \frac{1}{N_i} \sum_{j=1}^{N_i} \mathbb{1}[s_{i,j} = 1]
$$

**含义**: 在所有任务和多次 rollout 上的平均成功率，是主要评价指标。

**符号说明**:
- $T$：任务总数
- $N_i$：任务 $i$ 的 episode 数
- $s_{i,j}$：成功指示量（1 = 成功，0 = 失败）
- $\mathbb{1}[\cdot]$：指示函数

---

### 公式 13: Pass@k 指标

$$
\text{pass@}k = \frac{1}{T} \sum_{i=1}^{T} \frac{1}{N_i} \sum_{j=1}^{k} \mathbb{1}[s_{i,j} = 1]
$$

**含义**: [[Pass@k]] 在前 $k$ 次试验中的平均成功率，衡量攻击在有限试验预算下的累积成功率。

---

## 关键图表

### Figure 1: 实验动机 / 动作位移与失败的相关性

![Figure 1](https://arxiv.org/html/2607.15207v1/x1.png)

**说明**: 在干净和攻击条件下，失败 episode 的动作位移显著大于成功 episode，而预测未来位移在成功/失败之间几乎无差别。这为 [[World-Action Drift]] 提供了实证基础：**动作偏移是任务失败的强预测信号，想象漂移不是**。

---

### Figure 2: 纯动作攻击的结构化动作位移

![Figure 2](https://arxiv.org/html/2607.15207v1/x2.png)

**说明**: 跨三种 WAM 变体，攻击主要集中于**连续动作通道**和**动作时域的特定部分**，而非均匀噪声。这表明扰动被优化器集中分配到最脆弱的动作维度和时间步。

---

### Figure 3: BadWAM 框架总览

![Figure 3](https://arxiv.org/html/2607.15207v1/x3.png)

**说明**: BadWAM 在不访问模型权重的情况下，向观测 $o_t$ 注入小幅视觉扰动，通过查询冻结 WAM 进行在线 [[Zeroth-Order Gradient Estimation|零阶优化]]，每个重规划步骤独立执行 8 次迭代（17 次 WAM 查询）。

---

### Figure 4a: 无保形约束的攻击想象（有漂移）

![Figure 4a](https://arxiv.org/html/2607.15207v1/x4.png)

**说明**: 纯动作攻击（Action-Only Attack）在破坏动作的同时，导致预测未来出现明显视觉异常——世界想象"泄露"了攻击痕迹。

---

### Figure 4b: 有保形约束的攻击想象（无漂移）

![Figure 4b](https://arxiv.org/html/2607.15207v1/x5.png)

**说明**: [[Imagination-Preserving Attack|保形攻击]] 在同等破坏动作的同时，预测未来与干净推理高度相似，**安全监控器无法从预测未来中检测到攻击**。

---

### Figure 5: 各任务失败分布

![Figure 5](https://arxiv.org/html/2607.15207v1/x6.png)

**说明**: 攻击后各任务成功率的分布直方图。失败是系统性的（横跨多个任务），而非偶发性异常，表明 BadWAM 对策略的破坏是普遍而非局部的。

---

### Figure 6: 与随机扰动的对比

![Figure 6](https://arxiv.org/html/2607.15207v1/x7.png)

**说明**: 在相同 $\ell_\infty$ 预算下，BadWAM 优化扰动远强于同预算随机噪声。白盒 [[PGD]] 攻击（需要梯度访问）提供了攻击强度上界。

---

### Figure 7: Pass@k 曲线

![Figure 7](https://arxiv.org/html/2607.15207v1/x8.png)

**说明**: 在不同试验次数 $k$ 下，BadWAM 在所有 [[LIBERO]] 子集上持续压低成功率。攻击不只影响第一次尝试，在 20 次重复试验中始终维持高攻击效果，排除偶发失败的解释。

---

### Figure 8: 各 LIBERO 子集的任务成功率

![Figure 8](https://arxiv.org/html/2607.15207v1/x9.png)

**说明**: 按 [[LIBERO]] 四个子集分解的成功率。**LIBERO-Spatial（空间推理）和 LIBERO-Long（长时程）任务遭受最严重的攻击破坏**，而 LIBERO-Object（对象中心）任务相对鲁棒，提示空间关系和长序列依赖是 WAM 的脆弱维度。

---

### Figure 9: LIBERO-10 干净 vs 攻击 rollout 对比

![Figure 9](https://arxiv.org/html/2607.15207v1/x10.png)

**说明**: 定性比较干净和攻击条件下的 rollout。攻击后机器人仍在执行动作，但轨迹路径被系统性偏移，导致无法完成抓取或放置目标。

---

### Figure 10: 累积动作位移随时间的演化

![Figure 10](https://arxiv.org/html/2607.15207v1/x11.png)

**说明**: 在每个重规划步骤，攻击诱导的动作位移持续累积。**失败 episode 在整个执行过程中保持显著更大的累积位移**，证明攻击效果通过闭环动力学不断放大。

---

### Figure 11: 保形攻击的搜索动态

![Figure 11](https://arxiv.org/html/2607.15207v1/x12.png)

**说明**: [[Zeroth-Order Gradient Estimation|零阶优化器]] 在每次迭代中持续提升保形攻击目标函数值，收敛稳定，约 16 次查询后边际收益递减。

---

### Figure 12: 等强度下的隐蔽性比较

![Figure 12](https://arxiv.org/html/2607.15207v1/x13.png)

**说明**: 在匹配动作位移强度的条件下，[[Imagination-Preserving Attack|保形攻击]] 产生的预测未来漂移始终低于纯动作攻击，**验证 $\lambda$ 惩罚项有效保持想象一致性**，实现强度-隐蔽性 Pareto 改进。

---

### Figure 13: 未来保形权重 λ 的消融

![Figure 13](https://arxiv.org/html/2607.15207v1/x14.png)

**说明**: $\lambda$ 控制攻击强度与隐蔽性的权衡。**适中的 $\lambda$ 值（约 0.5–1.0）能在显著降低想象漂移的同时保持接近最强攻击效果**，过大的 $\lambda$ 会过度牺牲攻击强度。

---

### Figure 14: 预算与效率消融

![Figure 14](https://arxiv.org/html/2607.15207v1/x15.png)

**说明**: (左) 扰动预算 $\varepsilon$ 对攻击强度影响显著，$\varepsilon = 0.20$ 可将成功率降至 0%；(右) 查询次数超过 16 次后收益递减，BadWAM 默认使用 8 次迭代（17 次查询）以平衡效果与效率。

---

### Figure 15a: 增强一致性检测的 ROC 曲线（无保形约束）

![Figure 15a](https://arxiv.org/html/2607.15207v1/x16.png)

**说明**: 基于增强一致性的检测器在无保形约束的攻击下，5% 误报率时召回率仅 13–21%，**检测效果很差**。

---

### Figure 15b: 增强一致性检测的 ROC 曲线（有保形约束）

![Figure 15b](https://arxiv.org/html/2607.15207v1/x17.png)

**说明**: 在 [[Imagination-Preserving Attack|保形攻击]] 下，基于增强一致性的检测器更难识别攻击，进一步验证预测未来不能作为可靠的安全信号。

---

### Table 1: 主要闭环攻击结果

| 模型变体 | 基准 | 干净 (%) | 纯动作攻击 (%) | 保形攻击 (%) | 下降 (pp) |
|---------|------|---------|--------------|------------|---------|
| Action-only WAM | LIBERO | 96.5 | 43.1 | — | ↓53.4 |
| Joint WAM | LIBERO | 98.1 | 61.5 | 63.0 | ↓36.6 / ↓35.1 |
| IDM WAM | LIBERO | 98.4 | 66.1 | 68.1 | ↓32.3 / ↓30.3 |
| Action-only WAM | RoboTwin | 92.1 | 84.4 | — | ↓7.7 |
| Joint WAM | RoboTwin | 90.9 | 84.4 | 85.2 | ↓6.5 / ↓5.7 |
| IDM WAM | RoboTwin | 91.4 | 83.7 | 85.1 | ↓7.7 / ↓6.3 |

**关键发现**: [[LIBERO]] 上攻击效果远强于 [[RoboTwin]]（53 pp vs 8 pp），可能源于任务复杂度和动作维度差异。纯动作攻击在 Action-only WAM 上最有效（无世界预测作为"缓冲"）。

---

### Table 2: 跨 WAM 变体的迁移攻击结果

| 攻击类型 | 迁移方向 | 目标干净 (%) | 目标攻击下 (%) | 下降 (pp) |
|---------|---------|------------|--------------|---------|
| 纯动作 | Action-only→Joint | 98.3 | 64.2 | ↓34.2 |
| 纯动作 | Joint→IDM | 100.0 | 59.2 | ↓40.8 |
| 纯动作 | IDM→Joint | 98.3 | 61.7 | ↓36.7 |
| 保形 | Joint→IDM | 100.0 | 60.8 | ↓39.2 |
| 保形 | IDM→Joint | 98.3 | 63.3 | ↓35.0 |

**关键发现**: 迁移攻击导致目标 WAM 成功率下降 30–40 pp，表明 [[World-Action Drift]] 漏洞不是特定模型架构的产物，而是 WAM 类模型的普遍脆弱性。

---

### Table 3: 非自适应防御效果

| 模型 | 防御方法 | 干净成功率 (%) | 攻击下成功率 (%) |
|------|---------|-------------|---------------|
| Joint WAM | 无 | 98.3 | 57.1 |
| Joint WAM | 高斯模糊 | 98.3 | 85.0 |
| Joint WAM | Resize-Crop 集成 | 52.5 | 0.0 |
| Joint WAM | JPEG-Noise 集成 | 94.2 | 89.2 |
| IDM WAM | 无 | 100.0 | 54.6 |
| IDM WAM | 高斯模糊 | 98.3 | 89.2 |
| IDM WAM | Resize-Crop 集成 | 54.2 | 8.3 |
| IDM WAM | JPEG-Noise 集成 | 93.3 | 90.0 |

**关键发现**: **Resize-Crop 集成虽然有效破坏攻击，但同时将干净成功率从 ~98% 降低到 ~53%**，无法实用部署。高斯模糊和 JPEG 噪声能提升防御效果但不能完全抵御攻击。

---

### Table 4: 实验范围说明

| 结果类型 | 当前协议 | 计划更新 |
|---------|---------|---------|
| 主要闭环结果（Table 1）| 完整 LIBERO；RoboTwin 完整协议 | RoboTwin 默认 5 epoch |
| 随机/白盒基线（Figure 6）| 完整 LIBERO | 不变 |
| Pass@k、子集分析、rollout | 完整 LIBERO | 不变 |
| 闭环误差、搜索动态 | 完整 LIBERO traces | 不变 |
| 等强度隐蔽性（Figure 12）| 完整 LIBERO（选定 WAM）| 扩展到其他变体 |
| 未来保形消融（Figure 13）| LIBERO 平衡子集 | 重跑完整 LIBERO |
| 预算消融（Figure 14）| LIBERO 平衡子集 | 重跑完整 LIBERO |
| 迁移攻击（Table 2）| LIBERO 平衡子集 | 重跑完整 LIBERO |
| 防御与检测（Table 3）| LIBERO 平衡子集 | 重跑完整 LIBERO |

---

### Table 5: 训练配置

| 配置项 | LIBERO Joint/IDM WAMs | RoboTwin Joint/IDM WAMs |
|-------|----------------------|------------------------|
| 数据集 | libero_2cam | robotwin |
| 相机数 | 2 | 3 |
| 图像分辨率 | 224×224 | 240×320 |
| 动作维度 | 7 | 14 |
| 状态维度 | 8 | 14 |
| 训练窗口 | 33 帧 | 33 帧 |
| 动作/视频频率比 | 4 | 4 |
| 每卡 Batch Size | 16 | 16 |
| GPU 数量 | 8 × H100 | 8 × H100 |
| 优化器调度 | Cosine | Cosine |
| 学习率 | 1×10⁻⁴ | 1×10⁻⁴ |
| 训练时长 | 10 epoch | 50k steps |

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 130 任务，每任务 20 episode | 四个子集（Spatial/Object/Goal/Long），桌面操纵 | 主要评测基准 |
| [[RoboTwin]] | 双臂 + 灵巧手操纵 | 更高动作维度（14 维），更复杂接触 | 跨基准泛化验证 |

### 实现细节

- **WAM 主干**: 论文使用 Cosmos Tokenizer（视频编码）+ Diffusion Transformer 动作头
- **优化器（攻击）**: [[Zeroth-Order Gradient Estimation|零阶有限差分]]，步长 $\eta = 0.01$，$c = 0.01$，$m = 1$（单向量估计）
- **攻击迭代**: 每重规划步 8 次迭代（17 次 WAM 查询）
- **扰动预算**: $\varepsilon = 0.06$（默认），消融测试 $\varepsilon \in \{0.02, 0.06, 0.10, 0.20\}$
- **训练硬件**: 8 × NVIDIA H100

---

## 批判性思考

### 优点

1. **揭示新攻击面**：首次系统研究 WAM 特有的世界动作漂移漏洞，填补了 WAM 安全研究的空白
2. **实用威胁模型**：黑盒查询设定比白盒梯度攻击更接近真实部署场景
3. **双变体设计**：Action-Only 和 [[Imagination-Preserving Attack|保形攻击]] 形成连续谱，全面覆盖攻击者的不同意图（最强效果 vs 最强隐蔽性）

### 局限性

1. **防御方案缺失**：论文重点展示攻击，对有效防御路径只给出初步探索，"动作-想象同步"的有效防御仍是开放问题
2. **RoboTwin 效果有限**：在更复杂的 RoboTwin 基准上攻击效果（~7–8 pp drop）远弱于 LIBERO（~30–53 pp），高动作维度和精细接触任务可能天然更鲁棒
3. **消融实验范围**：λ、ε 消融和迁移实验在 LIBERO 平衡子集上运行，论文承认需要在完整基准上复现验证

### 潜在改进方向

1. **自适应防御**: 设计专门针对动作-想象解耦的检测机制（如监控两者的不一致性）
2. **更强的攻击设定**: 探索多步规划攻击或在更长时间窗口上优化扰动
3. **物理世界扰动**: 研究将数字扰动转化为物理对抗贴纸对 WAM 的影响

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（Table 5 提供详细超参数）
- [x] 数据集可获取（LIBERO 和 RoboTwin 均公开）

---

## 关联笔记

### 基于

- [[WAM]]: BadWAM 专门针对 WAM 类模型的攻击框架
- [[WAM-Survey]]: 提供 WAM 分类体系（Joint / IDM / Action-only 三变体）
- [[PGD]]: 投影梯度下降，白盒攻击上界参考
- [[对抗补丁攻击]]: 传统视觉对抗攻击，BadWAM 的攻击机制研究基础

### 对比

- [[BadVLA]]: 针对 VLA 的后门攻击（训练时植入触发器），BadWAM 是推理时的在线对抗攻击，威胁模型不同

### 方法相关

- [[World-Action Drift]]: 本文发现的核心漏洞概念
- [[Zeroth-Order Gradient Estimation]]: 黑盒查询优化方法
- [[Imagination-Preserving Attack]]: 保形攻击变体
- [[Pass@k]]: 多次试验成功率评测指标

### 硬件/数据相关

- [[LIBERO]]: 主要评测基准
- [[RoboTwin]]: 跨基准泛化测试
- [[IDM]]: 作为 WAM 的一种架构变体被攻击评估

---

## 速查卡片

> [!summary] BadWAM (2026)
> - **核心**: [[WAM]] 的动作输出可在想象未来保持正常时被单独劫持（[[World-Action Drift]]）
> - **方法**: 黑盒 [[Zeroth-Order Gradient Estimation|零阶查询攻击]]，两变体（强度/隐蔽性可调）
> - **结果**: LIBERO 成功率从 96.5% 降至 43.1%（纯动作）；保形攻击维持等强度同时想象漂移更小
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-19*

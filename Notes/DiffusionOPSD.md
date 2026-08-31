---
title: "On-Policy Self-Distillation in Diffusion Models"
method_name: "DiffusionOPSD"
authors: [Wei Zhou, Xiongwei Zhu, Lingdong Kong, Bo Chen, Lei Zhang, Yongyuan Liang, Xiaoxia Hou, Ye Tian, Xian Sun, Yingshuo Wang, Linfeng Li, Shengqiong Wu, Leigang Qu, Feng Li, Wei Liu, Julian McAuley, Tat-Seng Chua]
year: 2026
venue: arXiv
tags: [diffusion-alignment, reward-optimization, on-policy-distillation, rlhf, diffusion-model, flow-matching, post-training]
zotero_collection: N/A
image_source: online
created: 2026-08-31
---

# 论文笔记：On-Policy Self-Distillation in Diffusion Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ByteDance Seed、National University of Singapore、UC San Diego、UC Berkeley、University of Maryland、HKUST、Duke、Oxford |
| 日期 | August 2026 |
| 项目主页 | [diffusionopsd.github.io](https://diffusionopsd.github.io/) |
| 对比基线 | [[FlowGRPO]]、[[ReFL]]、[[DiffusionNFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.24646) / [Code](https://github.com/worldbench/DiffusionOPSD) / [Checkpoints](https://huggingface.co/WeiChow/DiffusionOPSD) |

---

## 一句话总结

> DiffusionOPSD 将图像级 reward 梯度转化为去噪中间步骤的显式正负监督目标，通过在线策略自蒸馏循环实现高效、可分析的扩散模型后训练对齐。

---

## 核心贡献

1. **方法**：提出[[On-Policy Distillation|在线策略自蒸馏]]框架，将图像级 reward 梯度转化为中间去噪预测的有界正负目标（detached supervision），目标随行为策略变化持续重建。
2. **结果**：在 SD3.5-M 和步骤蒸馏模型 Z-Image-Turbo 上，19/20 reward 匹配设置中取得最佳 held-out 分数，比 DiffusionNFT 训练效率提升 40%（SD3.5-M）和 63%（Z-Image-Turbo）。
3. **分析**：通过相同 query 的对照实验证明：更大的目标构建增益不一定带来更大的有限步拟合增益，从而将目标构建与有限实现分离为两个独立可测量的阶段。

---

## 问题背景

### 要解决的问题

[[Diffusion Model|扩散模型]]的 [[RLHF|强化学习对齐]] 面临结构性不匹配：每个样本通过一系列相互依赖的去噪预测生成，而 reward 仅在序列终点（解码后的图像）处观测。如何将终点级 reward 转化为中间去噪 query 处的可操作监督，是核心挑战。

### 现有方法的局限

- **[[FlowGRPO]]**：通过有限群组估计轨迹信用分，依赖 likelihood ratio，对采样器选择、discretization 敏感。
- **[[ReFL]]**：对单个截断前缀 rollout 末段的 clean-output 预测反向传播可微 reward，将 reward 评估与模型优化耦合在一起；不执行后缀也不解码终点。
- **[[DiffusionNFT]]**：用 reward 加权的终点监督扩散目标重加权 rollout 终点，目标仍然是终点本身，而非当前中间预测应如何改进的显式描述。

以上方法的共同问题：期望的中间改变隐式编码在 advantage 权重、参数梯度或终点监督中，缺乏**显式的本地改进目标**。

### 本文的动机

在模型更新**之前**先构建 reward 改进目标，将其作为 detached 中间监督拟合，并在行为策略变化时重建。这设计使目标构建和有限拟合（finite fitting）可以**分别测量和分析**，并与传统 RL 回传解耦以降低计算成本。

---

## 方法详解

### 整体框架

DiffusionOPSD 将扩散 reward 优化视为[[On-Policy Distillation|在线策略自蒸馏]]循环，分为三个阶段：

- **阶段 1 — Query 采集与 Anchor**：冻结[[Behavior Policy|行为策略]] $v_{old}$ 采样 $K$ 条轨迹，在低噪 query 处提取 clean-output anchor $y_0$
- **阶段 2 — 目标构建**：用[[Reward Gradient|reward 梯度]]在 $y_0$ 附近构建有界正目标 $\bar{y}_+$ 和负目标 $\bar{y}_-$
- **阶段 3 — 有限拟合与行为策略刷新**：可训练策略以 $M_{fit}$ 步拟合 detached 目标，[[EMA|指数移动平均]]更新行为策略，进入下一轮迭代

![Figure 4 - DiffusionOPSD 框架概览](https://diffusionopsd.github.io/assets/method-overview.png)

**Figure 4**：DiffusionOPSD 框架。行为策略采集低噪 query，reward 梯度构建有界正负目标，可训练策略拟合 detached 监督，更新后的行为策略为下一迭代提供新 anchor。

---

### 模块 1：Clean-Output 预测坐标

**设计动机**：在[[Rectified Flow|整流流]]路径 $z_\sigma = (1-\sigma)y + \sigma\epsilon$ 下，速度目标 $v = \epsilon - y$，因此 velocity 预测和 clean-output 预测之间存在仿射双射关系。

$y_\theta(s)$ 就是在固定 query $s = (c, z_\sigma, \sigma)$ 处，模型预测的 clean-output（即去噪后的干净图像 latent），可以被 decoder 解码并由 reward 函数评分。

**低噪 query 的优势**（Proposition B.1）：

$$
\|y_v - y_{v^*}\|_2 \leq \sigma \|v - v^*\|_2
$$

低噪处 velocity 扰动在 clean-output 空间被 $\sigma$ 衰减，使 clean-output 预测更稳定、语义更丰富，同时策略仍保留本地改进的自由度。

---

### 模块 2：Query 采集与 Anchor

在第 $i$ 次外层迭代，冻结行为策略对每个 prompt 采样 $K$ 条轨迹，选取最接近请求噪声水平 $\sigma^*$ 的 query 状态：

$$
q = \arg\min_j |\sigma_j - \sigma^*|, \quad s = (c, z_q, \sigma_q)
$$

行为 anchor 是行为策略在该 query 处的 clean-output 预测：

$$
y_0 = z_q - \sigma_q v_{old}(z_q, c, \sigma_q)
$$

终点 reward 计算群组归一化拟合权重（per-prompt centering + 全局 rollout batch 标准差）：

$$
\bar{r}_c = \frac{1}{K}\sum_{j=1}^{K} r^j_c, \quad \hat{\sigma}_{B_i} = \left(\frac{1}{|B_i|}\sum_{(c,j)\in B_i}(r^j_c - \bar{r}_{B_i})^2\right)^{1/2}
$$

$$
Z_{B_i} = c_{adv}(\hat{\sigma}_{B_i} + \epsilon_Z), \quad \omega^k_c = \frac{1}{2} + \frac{1}{2}\text{clip}\!\left(\frac{r^k_c - \bar{r}_c}{Z_{B_i}}, -1, 1\right)
$$

**符号说明**：
- $\omega^k_c \in [0,1]$：第 $k$ 个 rollout 的拟合权重（高 reward → 偏向正分支，低 reward → 偏向负分支）
- $c_{adv}$：共享 clipping 和 loss 缩放系数
- $\bar{r}_{B_i}$：全 batch reward 均值；per-prompt centering 消除 prompt 级 reward 偏移

---

### 模块 3：目标构建（Bounded Target Construction）

从 $y_0$ 出发，用稳定化归一化 reward 梯度步骤构建正负目标。初始化 $y^{(0)}_+ = y^{(0)}_- = y_0$，迭代 $m = 0, \ldots, M_{tgt}-1$：

$$
y^{(m+1)}_+ = y^{(m)}_+ + h_{step}\frac{\nabla_y \tilde{R}(y^{(m)}_+, c)}{\|\nabla_y \tilde{R}(y^{(m)}_+, c)\|_2 + \epsilon_g}
$$

$$
y^{(m+1)}_- = y^{(m)}_- - h_{step}\frac{\nabla_y \tilde{R}(y^{(m)}_-, c)}{\|\nabla_y \tilde{R}(y^{(m)}_-, c)\|_2 + \epsilon_g}
$$

步长 $h_{step} = \eta_{tgt}\rho\|y_0\|_2 / M_{tgt}$，其中 $\eta_{tgt} > 0$ 为步长乘数（默认 1），$\rho$ 为目标半径系数。

每步后投影回 trust-region 球内：

$$
y^{(m+1)}_\pm \leftarrow y_0 + \Pi_{\rho\|y_0\|_2}\!\left(y^{(m+1)}_\pm - y_0\right), \quad \Pi_r(d) = \begin{cases} d, & \|d\|_2 \leq r \\ r\,d/\|d\|_2, & \|d\|_2 > r \end{cases}
$$

保证目标始终在 trust-region 内：

$$
\|y^{(m)}_\pm - y_0\|_2 \leq \min\{m\,h_{step},\, \rho\|y_0\|_2\} \leq \rho\|y_0\|_2
$$

最终 detached 目标（stop-gradient）：

$$
\bar{y}_+ = \text{sg}(y^{(M_{tgt})}_+), \quad \bar{y}_- = \text{sg}(y^{(M_{tgt})}_-)
$$

单步的本地 reward 变化（一阶近似）：

$$
\tilde{R}(y^{(1)}_\pm, c) - \tilde{R}(y_0, c) = \pm h_{step}\frac{\|g_0\|_2^2}{\|g_0\|_2 + \epsilon_g} + O(h^2_{step})
$$

**符号说明**：
- $\tilde{R}(y, c) = R(D(y), c)$：local reward，在 clean-output 解码后评分
- $\epsilon_g$：梯度稳定器，防止除零，同时作为隐式置信门控
- $\text{sg}(\cdot)$：stop-gradient 算子

---

### 模块 4：有限拟合（Finite Fitting）

可训练速度场在同一 query 处评估，形成正负拟合分支（branch coefficient $\beta$）：

$$
y^\theta_+ = \beta y_\theta + (1-\beta)y_0, \quad y^\theta_- = (1+\beta)y_0 - \beta y_\theta
$$

**含义**：单个输出 $y_\theta$ 同时编码对 $\bar{y}_+$ 的吸引和对 $\bar{y}_-$ 的排斥，通过 $\beta$ 调节分支灵敏度。

Detached 自适应归一化器：

$$
\gamma_+ = \max\{\text{sg}(\text{mean}|y^\theta_+ - \bar{y}_+|),\, \epsilon_\gamma\}, \quad \gamma_- = \max\{\text{sg}(\text{mean}|y^\theta_- - \bar{y}_-|),\, \epsilon_\gamma\}
$$

**OPSD 分支损失**：

$$
\mathcal{L}_{OPSD} = \omega\,\frac{\text{mean}[(y^\theta_+ - \bar{y}_+)^2]}{\gamma_+} + (1-\omega)\,\frac{\text{mean}[(y^\theta_- - \bar{y}_-)^2]}{\gamma_-}
$$

实际实现目标为 $c_{adv}\mathcal{L}_{OPSD}$，$c_{adv}$ 同时作用于 reward 归一化的 clipping range 和 loss 缩放。

Stop-gradient 约束确保 query、anchor、权重和目标在有限拟合期间不被优化：

$$
\nabla_\theta z_q = \nabla_\theta \omega = \nabla_\theta y_0 = \nabla_\theta \bar{y}_\pm = 0
$$

**理想输出空间最小化器**（固定归一化器时，配方法推导）：

$$
\delta^*_\theta = \frac{a_+ d_+ - a_- d_-}{\beta(a_+ + a_-)}
$$

其中 $a_+ = \omega/\gamma_+$，$a_- = (1-\omega)/\gamma_-$，$d_+ = \bar{y}_+ - y_0$，$d_- = \bar{y}_- - y_0$。局部对称情况下两分支收敛到同一 reward 改进方向，权重 $\omega$ 取消。

**符号说明**：
- $\beta$：分支系数，控制 $y_\theta$ 偏离 $y_0$ 的灵敏度（默认 $\beta=1$）
- $\gamma_\pm$：自适应归一化器，稳定不同 query 的 loss 量级
- $\omega \in [0,1]$：拟合权重，高 reward rollout → $\omega$ 大 → 主要拟合正目标

---

### 模块 5：在线训练循环

第 $i$ 次外层迭代获得 detached 数据集：

$$
\mathcal{D}_i = \{(c, z_q, \sigma_q, \omega, \bar{y}_+, \bar{y}_-)\}_{c\sim C,\, z_q\sim Q^{old,i}_{roll,\sigma_q}}
$$

拟合阶段执行 $M_{fit}$ 步优化器更新；拟合期间 $y_0$ 从冻结的行为策略重新计算（不存储）。行为策略 [[EMA|EMA 更新]]：

$$
\theta_{old} \leftarrow \eta^{(u)}_{beh}\theta_{old} + (1 - \eta^{(u)}_{beh})\theta
$$

同时维护 checkpoint 均值用于最终输出：

$$
\theta_{ckpt} \leftarrow \eta^{(u)}_{ckpt}\theta_{ckpt} + (1 - \eta^{(u)}_{ckpt})\theta
$$

---

## 关键图表

### Figure 1：训练与 Held-Out 质量曲线

![Figure 1 - Training and held-out quality](https://diffusionopsd.github.io/assets/training-curves.png)

**说明**：归一化增益（10 种 reward + 2 种 backbone 平均）随 GPU-hours 的变化。DiffusionOPSD 改进最快，最终增益最高；横轴为归一化累计 GPU-hours。

### Figure 2：生成样本

**说明**：DiffusionOPSD 在 held-out 文本提示下生成的样本，展示 prompt fidelity 和 fine detail 的改进。见项目主页 hero mosaic（https://diffusionopsd.github.io/ 的 `assets/hero-mosaic/`）。

### Figure 3：Reward-to-Policy 范式对比

**说明**（论文 Figure 3）：对比四种将 reward 转化为策略监督的范式：
- **轨迹信用**（FlowGRPO）：对 rollout 终点 backward credit
- **晚期状态反向传播**（ReFL）：对单个 late-state clean-output 反向传播 reward
- **终点模仿**（DiffusionNFT）：重加权 rollout 终点 $x_0$
- **显式中间目标**（DiffusionOPSD）：在 query 处构建 $y_+, y_-$ 后 finite fitting

### Figure 4：DiffusionOPSD 框架概览

![Figure 4 - DiffusionOPSD Overview](https://diffusionopsd.github.io/assets/method-overview-animated.gif)

**说明**：完整框架示意。左：latent 流形与向量场、去噪轨迹；中：query state → behavior anchor $y_0$ → 本地 reward landscape → 正负目标构建；右：LOPSD 损失与 EMA 更新。

### Figure 5：训练动态

**说明**（论文 Figure 5）：71 个单 reward 训练轮次中，DiffusionOPSD 的 median terminal position 为 98%（即达到各轮次观测到的 native reward 范围的 98% 位置），比 ReFL（97%）、DiffusionNFT（90%）和 FlowGRPO（83%）更高，表明优化稳定性最强。

### Figure 6：联合三 Reward 动态

**说明**（论文 Figure 6）：SD3.5-M 上的联合 PickScore + CLIPScore + HPSv2.1 训练曲线。DiffusionOPSD 三项 reward 持续改善至 update 300，而 DiffusionNFT 的 HPSv2.1 在 update 200 后下降。

### Figure 7：Native 训练 Reward

**说明**（论文 Figure 7）：两种 backbone 上 10 个目标的 native reward 与累计 GPU-hours 曲线。DiffusionNFT 在 Z-Image-Turbo 上 10 个指标中 8 个低于未对齐基模型；DiffusionOPSD 在几乎所有设置中均有改善。

### Figure 8：Held-Out 质量与计算效率 Pareto 前沿

**说明**（论文 Figure 8）：DiffusionOPSD 在 Z-Image-Turbo 所有 10 个前沿和 SD3.5-M 9/10 个前沿上达到最高 held-out 分数（例外：SD3.5-M Aesthetic，ReFL 领先 0.01）。

### Figure 9：消融动态

**说明**（论文 Figure 9）：(a) reward 梯度方向训练持续改善；随机残差目标（residual）后期崩溃。(b) rollout query state 与 forward-noised 控制几乎无差异，CLIPScore 仅差 0.0033。

### Figure 10：目标构建与有限拟合

**说明**（论文 Figure 10）：DiffusionOPSD 正目标使 fixed-suffix reward 增加 +0.03511；DiffusionNFT 终点使 reward 降低 −0.03551。单步 fresh-AdamW 更新后，HPSv2.1 的目标排序在 62.3% 的 prompt 上发生逆转，证明构建增益不决定拟合后增益。

### Figure 11–14：Held-Out 定性比较（共 4 页）

各页展示 5 个 prompt（共 20 个），每行对比 DiffusionOPSD / DiffusionNFT / Base / FlowGRPO / ReFL 的生成结果。DiffusionOPSD 在渲染文字、运动模糊、多物体组合、材质细节上一致性最高；DiffusionNFT 经常丢失请求的细节或语义结构。

### Figure 15：目标与实现检查

**说明**（论文 Figure 15）：
- (a) 方向：gradient target 0.3117 >> random 0.2311 > no-op 0.2280 >> residual 0.1456（CLIPScore）
- (b) 实现控制：Steps/η/NFE/Sampler 等单因素控制变量均在默认值 0.0079 范围内
- (c) CFG 依赖：无 CFG 配置超过 0.3117

### Figure 16：鲁棒性与人类偏好

![Figure 16 - Robustness](https://diffusionopsd.github.io/assets/robustness.png)

**说明**：
- (a) 鲁棒性：σ∈[0.28,0.60]、ρ∈[0.08,0.40] 范围内 CLIPScore 稳定在默认值附近；σ=0.90 和 β=10 显著下降
- (b) 人类偏好：标注者对 DiffusionOPSD 偏好率：vs Base 64%、vs FlowGRPO 71%、vs DiffusionNFT 90%、vs ReFL 61%
- (c) 胜出原因分布：prompt alignment、visual quality、realism、text/details 均有贡献

---

### Table 1：主对比结果

> 7 个开源评估器 + 3 个内部偏好模型，更高更好。CFG = classifier-free guidance。**粗体** = 最优，下划线 = 次优（单阶段块内）。

| Model | Pick↑ | CLIP↑ | HPSv2.1↑ | Aes↑ | ImgR↑ | HPSv3↑ | DeQA↑ | AltCLIP↑ | Point↑ | Pair↑ |
|-------|-------|-------|----------|------|-------|--------|-------|----------|--------|-------|
| **SD3.5-M（单 reward 专家，100 updates）** | | | | | | | | | | |
| ReFL | 23.92 | 0.308 | 0.358 | 12.09 | 1.28 | 9.33 | 4.85 | 0.408 | 0.193 | 0.290 |
| DiffusionNFT | 23.43 | 0.298 | 0.336 | 9.11 | 1.46 | 9.14 | 4.76 | 0.412 | 0.199 | 0.323 |
| **DiffusionOPSD** | **24.94** | **0.340** | **0.390** | 12.08 | **1.76** | **13.34** | **4.94** | **0.450** | **0.214** | **0.465** |
| **SD3.5-M（单 checkpoint，300 updates）** | | | | | | | | | | |
| DiffusionNFT | 23.62 | 0.294 | 0.340 | 6.02 | 1.45 | 8.36 | 4.32 | 0.397 | 0.150 | 0.294 |
| **DiffusionOPSD** | **25.51** | **0.333** | **0.389** | **6.03** | **1.51** | **9.41** | **4.40** | **0.399** | **0.170** | **0.345** |
| FlowGRPO | 22.51 | 0.293 | 0.274 | 5.32 | 1.06 | 2.65 | 4.01 | 0.393 | 0.181 | 0.388 |
| **Z-Image-Turbo（单 reward 专家，100 updates）** | | | | | | | | | | |
| Base | 22.86 | 0.276 | 0.296 | 5.41 | 0.96 | 6.19 | 4.44 | 0.392 | 0.213 | 0.422 |
| ReFL | 24.54 | 0.313 | 0.380 | 9.79 | 1.37 | 13.77 | 4.60 | 0.441 | 0.227 | 0.481 |
| DiffusionNFT | 22.28 | 0.280 | 0.277 | 6.07 | 0.58 | 1.58 | 3.37 | 0.363 | 0.166 | 0.357 |
| **DiffusionOPSD** | **25.15** | **0.320** | **0.390** | **10.74** | **1.79** | **14.44** | **4.78** | **0.451** | **0.243** | **0.551** |

**关键发现**：DiffusionNFT 在 Z-Image-Turbo 上 8/10 指标低于未对齐基模型（step distillation 使终点目标失配），而 DiffusionOPSD 在 native 行为策略状态处锚定目标，全部超越基模型。

---

### Table 2：训练效率对比（每 100 updates，8 GPU）

| Method | SD3.5-M GPU-hours | Z-Image-Turbo GPU-hours |
|--------|------------------|------------------------|
| FlowGRPO | >5k updates（>50 GPU-h 等价） | — |
| ReFL | 47.7 | 102.1 |
| DiffusionNFT | ~47 | ~405 |
| **DiffusionOPSD** | **28.2（↓40% vs NFT）** | **149.8（↓63% vs NFT）** |

**关键发现**：DiffusionOPSD 的 Z-Image-Turbo 峰值显存不低于 DiffusionNFT（因为分配了更多 VRAM），但整体 GPU-hours 大幅减少。

---

### 算法 1：DiffusionOPSD 训练流程

```
Require: 初始可训练策略 vθ, reward R, decoder D, prompts C,
         群组大小 K, 请求 query 噪声级 σ*, 目标半径 ρ,
         目标步数 Mtgt, 步长乘数 ηtgt, 拟合预算 Mfit,
         分支系数 β, 共享 clipping/loss 系数 cadv,
         稳定器 ϵZ, ϵg, ϵγ, 行为和 checkpoint EMA 保留率 ηbeh, ηckpt

初始化: θold ← θ, θckpt ← θ, 更新计数 u ← 0

for 每次外层迭代 i:
    选择当前 prompt batch Ci
    for c ∈ Ci, k = 1,...,K:
        rollout 冻结 vθold → 终点 reward rk_c + detached query (c, zk_q, σk_q)
    计算 ωk_c（Per-prompt centering + global std）
    初始化 Di ← ∅
    for 每个 (c, k):
        y0 ← zk_q - σk_q vθold(zk_q, c, σk_q)
        构建 ȳ+ 和 ȳ-（reward 梯度步 + trust-region 投影）
        加入 Di: (c, zk_q, σk_q, ωk_c, ȳ+, ȳ-)
    for j = 1,...,Mfit:
        从 Di 采样，用冻结 vθold 重计算 y0
        用 cadv·LOPSD 更新 θ; u ← u+1; 更新 θckpt
    θold ← EMA(θold, θ)     ← 仅在 fitting 完成后更新

输出: Checkpoint 均值对齐策略 vθckpt
```

---

## 实验

### 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| Pick-a-Pic prompts | 训练 prompt 集合 | 开源人类偏好文本-图像对 |
| DrawBench | Held-out 评估 | 多样化文本-图像评估基准 |

### 实现细节

- **Backbone**：SD3.5-M（512²，10-step CFG-free rollout）和 Z-Image-Turbo（1024²，native 9-step rollout）
- **Rollout 采样器**：训练用 DPM-Solver++ 2M，评估用确定性 40-step flow sampling
- **Query 噪声级**：$\sigma^* = 0.278$（低噪，默认）
- **目标半径**：$\rho = 0.10$（trust-region 10%）
- **分支系数**：$\beta = 1$
- **目标步数**：$M_{tgt} = 2$（reward-ascent steps）
- **群组大小**：$K$（多轨迹 per-prompt rollout）
- **硬件**：8 GPU 实测 per-step 墙钟时间计算

### 关键实验发现

**目标构建（同 query 对照）**：
- DiffusionOPSD 正目标使 fixed-suffix reward 增加 +0.03511，reward-gradient alignment = 0.7094
- DiffusionNFT 终点目标 reward 变化 −0.03551，alignment = −0.000203（即便 radius 匹配也是 −0.01887）
- 结论：构建增益主要来自 reward 梯度方向，而非目标半径

**有限拟合逆转（HPSv2.1）**：
- reward-gradient 目标构建增益 +0.00245，random 目标 −0.03251
- 单步 fresh-AdamW 更新后：reward-gradient 实现 −0.000740，random 实现 −0.000021
- 逆转率 62.3%（95% CI: 58.2%–66.6%）
- 结论：构建增益和有限实现应分别测量

**消融关键结论**：
- reward 方向是最大消融差距来源（缺失后 CLIPScore 从 0.3117 降至 0.1456–0.2311）
- query 来源（rollout vs forward-noised）差距仅 0.0014（相对 1.1%），远小于方向因素
- 负分支效果：移除后 CLIPScore 0.3137（提升 0.002），属于实现细节范围内

---

## 批判性思考

### 优点

1. **可分析性**：目标构建与有限拟合解耦，使 reward pipeline 中的每个环节可单独诊断，这在工程上非常有价值
2. **计算效率**：避免对 reward/decoder 图的 backward 传播（detached 目标），大幅减少训练 GPU-hours
3. **Step-distilled 模型兼容性**：通过 native 行为策略的轨迹锚定目标，解决了 DiffusionNFT 在 Z-Image-Turbo 上终点目标失配的问题

### 局限性

1. **单 query 局限**：每次 rollout 只在单个低噪 query 处构建目标，未利用整条去噪轨迹的多步信息
2. **Reward 函数依赖**：需要 reward 函数在 clean-output 空间可微，限制了对不可微 reward 的适用性（如纯排序/人类标注）
3. **逆转现象未解决**：论文发现构建增益不等于实现增益（逆转率高达 62.3%），但主方法并未针对此设计对策

### 潜在改进方向

1. **多步目标构建**：在多个 query 噪声级处联合构建目标，利用全轨迹的 reward 梯度信息
2. **OPSD + 不可微 reward**：结合离线 reward 模型拟合，将 OPSD 框架扩展到纯排序或 pairwise 标注数据
3. **有限拟合的自适应步数**：根据 target-update 逆转率动态调整 $M_{fit}$，减少无效拟合

### 可复现性评估

- [x] 代码开源（Apache 2.0）
- [x] 预训练模型（HuggingFace: WeiChow/DiffusionOPSD）
- [x] 训练细节完整（Algorithm 1 + Appendix C 详细 protocol）
- [x] 数据集可获取（Pick-a-Pic 开源，DrawBench 开源）

---

## 关联笔记

### 基于

- [[Rectified Flow]]：DiffusionOPSD 的 clean-output 坐标基于整流流路径 $z_\sigma = (1-\sigma)y + \sigma\epsilon$ 的仿射关系
- [[On-Policy Distillation]]：On-policy 蒸馏的框架思想，DiffusionOPSD 的结构性前身
- [[EMA|指数移动平均]]：用于行为策略刷新和 checkpoint 平均，保证训练稳定性

### 对比

- [[FlowGRPO]]：group-relative trajectory credit，依赖 likelihood ratio，本文主要竞争基线之一
- [[ReFL]]：differentiable reward backpropagation，将 reward 与模型优化耦合
- [[DiffusionNFT]]：endpoint-conditioned forward-process regression，在 Z-Image-Turbo 上显著失效

### 方法相关

- [[DiffusionOPSD]]：本方法概念页
- [[Behavior Policy]]：冻结行为策略是 on-policy 采集的核心机制
- [[Trust Region]]：目标构建采用 trust-region 球内投影，保证目标有界
- [[RLHF]]：DiffusionOPSD 是扩散模型 RLHF 对齐的新框架

### 硬件/数据相关

- SD3.5-M（Stable Diffusion 3.5 Medium）：主要实验 backbone 之一
- Z-Image-Turbo：步骤蒸馏后的 few-step diffusion backbone，1024² 分辨率

---

## 速查卡片

> [!summary] DiffusionOPSD
> - **核心**：将图像级 reward 梯度转化为去噪中间步骤的有界正负 detached 监督目标，在线策略循环重建
> - **方法**：冻结行为策略采集低噪 query → reward 梯度 trust-region 构建 $\bar{y}_\pm$ → 可训练策略拟合 → EMA 刷新
> - **结果**：19/20 reward 设置 SOTA，训练效率比 DiffusionNFT 提升 40-63%，人类偏好 vs ReFL 61%
> - **代码**：https://github.com/worldbench/DiffusionOPSD

---

*笔记创建时间: 2026-08-31*

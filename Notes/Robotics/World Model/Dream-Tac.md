---
title: "Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation"
method_name: "Dream-Tac"
authors: [Yunfan Lou, Yifan Ye, Yankai Fu, Jun Cen, Xiaowei Chi, Yaoxu Lyu, Peidong Jia, Sirui Han, Zhihe Lu, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [tactile-sensing, world-action-model, robot-manipulation, diffusion-model, visuotactile]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.08737
created: 2026-06-10
---

# 论文笔记：Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 北京大学、香港科技大学、南京大学 |
| 日期 | June 2026 |
| 项目主页 | https://github.com/LYFCLOUDFAN/Dream-Tac |
| 对比基线 | [[Cosmos-Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.08737) / [Code](https://github.com/LYFCLOUDFAN/Dream-Tac) |

---

## 一句话总结

> Dream-Tac 是首个统一触觉感知的 World Action Model，通过接触门控视触觉融合和联合多模态预测，在 6 个接触密集操作任务上平均成功率达 83.3%，比基线提升 31.7%。

---

## 核心贡献

1. **统一触觉 World Action Model**: 首次将触觉动态（tactile dynamics）纳入 [[世界动作模型|World Action Model]] 的联合建模框架，同时预测动作、未来视觉观测和触觉序列
2. **接触感知自注意力（CASA）**: 提出基于接触门（contact gate）的门控 logit bias 机制，让模型在接触发生时选择性地集成触觉信号，无需额外学习参数
3. **双层加速设计**: 通过 FlashBias 算子融合和扩散步缓存（diffusion-step caching），实现 2.9× 训练加速和 1.8× 推理加速，支持机器人实时部署

---

## 问题背景

### 要解决的问题

接触密集型操作（如削黄瓜、切水果、插 USB）需要精细的力感知，但现有的 [[世界动作模型|World Action Model]] 仅依赖视觉信号。视觉信号对接触状态（如 USB 是否成功插入、抓握力是否足够）的感知存在天然局限：接触前后的 RGB 图像几乎没有变化，无法捕捉关键的力-形变信息。

### 现有方法的局限

- **纯视觉 WAM**（如 [[Cosmos-Policy]]）：忽视触觉模态，在接触密集任务上泛化差
- **[[ForceVLA]]**：使用力矩传感器，但传感器噪声大、信号不稳定，且直接将力信号输入策略而非建模其动态
- **π₀、π₀.₅**（[[OpenPI]]）：视觉语言动作模型，无触觉模态，在精细接触任务上表现弱

### 本文的动机

触觉传感器（如 Xense Photon）输出 RGB 图像序列，与视觉输入共享相同的模态空间，天然适合纳入视频 [[扩散变换器|Diffusion Transformer]] 框架。通过联合建模视觉、触觉和动作，模型可以学习到接触与动作之间的因果关系，从而在预测动作时利用想象中的触觉反馈。

---

## 方法详解

### 模型架构

Dream-Tac 采用基于预训练视频 [[扩散变换器|Diffusion Transformer]] 的 **多模态联合生成** 架构：

- **输入**: 语言指令 $l$ + 视觉观测 $o$ + 触觉历史 $x$ + 本体感知状态
- **Backbone**: 预训练视频 DiT（[[Video Diffusion Transformer]]）
- **核心模块**: [[接触感知自注意力|Contact-Aware Self-Attention (CASA)]] 用于选择性触觉融合
- **输出**: 动作块 $a_{1:H}$ + 未来视觉帧 $v_{1:T}$ + 未来触觉帧 $x_{1:T}$
- **编码器**: 共享 [[VAE|变分自编码器]] 将视觉、触觉、动作统一编码到隐空间

所有 token 序列拼接后通过 [[DiT]] 的 [[Joint Self-Attention|联合自注意力]] 统一处理，接触感知 bias 通过 CASA 层注入。

### 核心模块

#### 模块 1: Contact-Aware Self-Attention (CASA)

**设计动机**: 触觉信号并非始终有意义——只有在接触发生时（如夹持物体时）触觉才携带关键信息。直接将触觉 token 与视觉/动作 token 全等融合会引入噪声，需要一个动态门控机制。

**具体实现**:
- 计算相邻触觉帧的逐像素差分 $\delta_t^L, \delta_t^R$（左右两个触觉传感器）
- 取最大值得到每帧触觉事件强度 $\rho_t$
- 通过标准化 + sigmoid 将 $\rho_t$ 映射为连续门控值 $g_t \in [g_{min}, g_{max}]$
- 将 $g_t$ 作为 logit bias 注入到 [[多头自注意力|多头注意力]] 的打分矩阵：仅作用于"非触觉 token 关注触觉 token"的注意力路径（asymmetric）

**关键设计**: 门控值 $g_t$ **无需学习**，完全由触觉图像变化量自动计算，保持训练简洁性的同时实现了动态加权。

#### 模块 2: FlashBias 加速

**设计动机**: 朴素实现中，logit bias 需要先计算完整注意力矩阵再加 bias，导致 $O(n^2)$ 内存开销。

**具体实现**:
- 将 CASA 的 logit bias 融合进 [[FlashAttention]] 的 tiling 计算内核
- bias 矩阵稀疏（仅非触觉→触觉路径非零），利用稀疏性跳过无效 tile
- 结合 [[扩散步缓存|Diffusion-Step Caching]] 跳过重复的去噪步，训练吞吐提升 2.9×，推理延迟从约 1.5s 降至约 0.8s

---

## 关键公式

### 公式 1: [[视觉运动策略|标准视觉运动策略]]

$$
p(a_{1:H} \mid o, l)
$$

**含义**: 给定视觉观测 $o$ 和语言指令 $l$，预测动作序列。这是不含世界模型的基线形式。

**符号说明**:
- $a_{1:H}$: 长度为 $H$ 的动作块（[[Action Chunking]]）
- $o$: 当前视觉观测
- $l$: 语言指令

### 公式 2: [[世界动作模型|联合 World Action Model]]

$$
p(a_{1:H}, v_{1:T} \mid o, l) = p(v_{1:T} \mid o, l)\, p(a_{1:H} \mid o, l, v_{1:T})
$$

**含义**: 先预测未来视觉帧，再以"想象"的未来视觉为条件预测动作——这是纯视觉 WAM 的分解形式。

**符号说明**:
- $v_{1:T}$: 未来 $T$ 帧视觉帧
- $H$: 动作预测步长（论文中 $H=20$）

### 公式 3: [[Dream-Tac|Dream-Tac 联合建模]]

$$
p(a_{1:H},\, v_{1:T},\, x_{1:T} \mid o, x, l)
$$

**含义**: Dream-Tac 在 WAM 基础上加入触觉模态的联合预测，同时生成动作、未来视觉和未来触觉序列。

**符号说明**:
- $x_{1:T}$: 未来 $T$ 帧触觉图像序列
- $x$: 当前触觉历史观测

### 公式 4: [[接触感知自注意力|Contact-Aware Self-Attention (CASA)]] Logit Bias

$$
\text{logit}_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + \alpha\, g_t\, (1 - M_i)\, M_j
$$

**含义**: 在标准注意力 logit 上叠加一个门控 bias，只有当 query token 是非触觉（$M_i=0$）且 key token 是触觉（$M_j=1$）时，bias 才激活。

**符号说明**:
- $\mathbf{q}_i, \mathbf{k}_j$: 第 $i$ 个 query token 和第 $j$ 个 key token 的向量
- $d$: 注意力头维度
- $\alpha$: 可学习缩放系数
- $g_t$: 第 $t$ 帧的接触门控值
- $M_i$: token 模态掩码，触觉 token 为 1，其余为 0

### 公式 5: [[触觉事件强度|触觉帧差计算]]

$$
\delta_t^L = \frac{1}{255}\mathbb{E}_{p,c}\bigl[|I_t^L(p,c) - I_{t-1}^L(p,c)|\bigr], \quad
\delta_t^R = \frac{1}{255}\mathbb{E}_{p,c}\bigl[|I_t^R(p,c) - I_{t-1}^R(p,c)|\bigr]
$$

$$
\rho_t = \max(\delta_t^L,\, \delta_t^R)
$$

**含义**: 通过左右触觉传感器相邻帧的归一化像素差分均值，计算每个时刻的接触强度指标 $\rho_t$。

**符号说明**:
- $I_t^L(p,c)$: 第 $t$ 帧左传感器在像素位置 $p$、颜色通道 $c$ 的值
- $\delta_t^L, \delta_t^R$: 左/右传感器帧差均值（归一化到 [0,1]）
- $\rho_t$: 两传感器中较大的帧差，作为当前时刻触觉事件强度

### 公式 6: [[门控机制|接触门控值计算]]

$$
z_t = k\,\frac{\rho_t - m}{s + \epsilon}, \qquad
\tilde{g}_t = \sigma(z_t) = \frac{1}{1+e^{-z_t}}, \qquad
g_t = g_{min} + (g_{max} - g_{min})\,\tilde{g}_t
$$

**含义**: 将触觉事件强度 $\rho_t$ 标准化后通过 sigmoid 映射，最终缩放到 $[g_{min}, g_{max}]$ 范围内的门控值。

**符号说明**:
- $m, s$: 训练集上 $\rho_t$ 的均值和标准差（统计量，非学习参数）
- $k$: 斜率超参数（控制 sigmoid 的陡峭程度）
- $\epsilon$: 数值稳定项
- $g_{min}, g_{max}$: 门控值的最小/最大边界

### 公式 7: [[扩散模型|去噪训练损失]]

$$
\mathcal{L}_{denoise} = \mathbb{E}_{y,\epsilon,\sigma}\bigl[\|f_\theta(\tilde{y}, \sigma, o, x, l) - \epsilon\|_2^2\bigr], \qquad \tilde{y} = y + \sigma\epsilon
$$

**含义**: 标准 EDM 风格去噪损失，预测加入噪声 $\sigma\epsilon$ 后的隐变量 $\tilde{y}$ 中的噪声成分。

**符号说明**:
- $y$: 干净的隐变量序列（视觉+触觉+动作的 VAE 编码）
- $\sigma$: 噪声强度
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $f_\theta$: Dream-Tac 去噪网络

### 公式 8: [[多任务学习|联合训练损失]]

$$
\mathcal{L} = \mathcal{L}_{act} + \lambda_v\,\mathcal{L}_{img} + \lambda_t\,\mathcal{L}_{tac}
$$

**含义**: 动作预测损失、未来视觉生成损失和未来触觉生成损失的加权求和，通过联合监督实现多模态对齐。

**符号说明**:
- $\mathcal{L}_{act}$: 动作 token 的去噪损失
- $\mathcal{L}_{img}$: 未来视觉帧的去噪损失
- $\mathcal{L}_{tac}$: 未来触觉帧的去噪损失
- $\lambda_v, \lambda_t$: 视觉和触觉损失权重系数

---

## 关键图表

### Figure 1: 系统概览与任务对比

![Figure 1](https://arxiv.org/html/2606.08737v1/x1.png)

**说明**: 左侧对比 RGB 视觉与触觉传感器在接触检测上的差异——视觉图像无法区分接触与非接触状态，而触觉 RGB 图像有明显的形变差异。右侧展示 Dream-Tac 在 6 个任务上的成功率对比（Dream-Tac 83.3% vs Cosmos Policy 51.7%）以及任务场景图。

### Figure 2: Dream-Tac 模型架构

![Figure 2](https://arxiv.org/html/2606.08737v1/x2.png)

**说明**: 多模态输入（视觉帧、触觉帧、语言指令、本体感知）经共享 VAE 编码为 token 序列，输入堆叠的 [[Video Diffusion Transformer]] DiT blocks。每个 DiT block 内嵌 Contact-Aware Self-Attention (CASA)，接触门 $g_t$ 由触觉帧差自动计算，通过 FlashBias 高效注入注意力矩阵。

### Figure 3: 六任务实验结果（成功率）

![Figure 3](https://arxiv.org/html/2606.08737v1/x3.png)

**说明**: Dream-Tac 在全部 6 个任务上均优于 4 个基线方法（π₀、π₀.₅、[[ForceVLA]]、[[Cosmos-Policy]]）。其中"Pick Baguette"和"Play Mahjong"达到 100%，"Insert USB"（精细接触）达 85%，平均成功率 83.3%。

### Figure 4: 环境泛化能力测试

![Figure 4](https://arxiv.org/html/2606.08737v1/x4.png)

**说明**: 在台面高度变化（±5cm）、物体空间排布扰动、外观变化（Mahjong 换牌面）、背景变化四种条件下，Dream-Tac 表现稳健（最低 75%），而 [[Cosmos-Policy]] 在台面高度+5cm 时完全失败（0%）。

### Figure 5: 训练与推理效率分析

![Figure 5](https://arxiv.org/html/2606.08737v1/x5.png)

**说明**: 左图展示 FlashBias + [[扩散步缓存|Diffusion-Step Caching]] 的组合将训练迭代时间从 80.82s 降至 27.48s（2.9×）；右图展示不同去噪步数下的推理延迟与成功率，Dream-Tac 在约 0.8s 延迟时保持高成功率，支持实时部署。

### Figure 6: 触觉表征的 t-SNE 可视化

![Figure 6](https://arxiv.org/html/2606.08737v1/x6.png)

**说明**: 对学到的触觉隐表征做 [[t-SNE]] 降维，不同操作动作（削黄瓜、切水果、插 USB 等）形成清晰分离的聚类，说明 Dream-Tac 学到了接触状态的判别性表征。

### Figure 7: 接触门时序分析

![Figure 7](https://arxiv.org/html/2606.08737v1/x7.png)

**说明**: 展示 5 个训练 episode 中触觉帧差 $\rho_t$ 和对应门控值 $g_t$ 随时间的变化。接触发生时（如夹持阶段）$\rho_t$ 明显升高，$g_t$ 随之增大，证明门控机制能准确捕捉接触时刻。

### Figure 8: 实验平台

![Figure 8](https://arxiv.org/html/2606.08737v1/x8.png)

**说明**: 实验硬件平台——[[Franka Emika Panda]] 机械臂 + 2 个 Intel RealSense D435i（固定 + 腕部安装）+ 2 个 Xense Photon 触觉传感器（夹爪两侧）。

### Figure 9: 任务执行过程可视化

![Figure 9](https://arxiv.org/html/2606.08737v1/x9.png)

**说明**: 六个任务的执行关键帧序列，展示 Dream-Tac 能够完成削黄瓜、切水果、插 USB、打麻将、拾取和擦白板等接触密集型任务。

### Figure 10: 接触门统计分布

![Figure 10a](https://arxiv.org/html/2606.08737v1/x10.png)

![Figure 10b](https://arxiv.org/html/2606.08737v1/x11.png)

**说明**: 上图为训练数据中触觉事件强度 $\rho_t$ 的分布直方图，呈现双峰——低值对应非接触阶段，高值对应接触阶段。下图展示 $\rho_t \to g_t$ 的 sigmoid 映射曲线，验证门控机制的合理性。

---

### Table 1: 触觉融合与接触感知注意力消融实验

| 方法 | 触觉输入 | 注意力 Bias | 平均成功率 (%) |
|------|---------|-----------|--------------|
| Visual WAM | ✗ | ✗ | 51.7 |
| Visuo-tactile WAM | ✓ | ✗ | 74.2 |
| Visuo-tactile WAM + Bias (Dream-Tac) | ✓ | ✓ | **83.3** |

**关键发现**: 触觉融合本身带来 +22.5% 的提升（51.7%→74.2%），接触感知 Bias 进一步贡献 +9.1%（74.2%→83.3%），两个组件缺一不可。

### Table 2: 专家演示数据统计

| 任务 | 演示数 | 平均回合长度 | 遥操时间 (s) | 最大步数 |
|------|-------|------------|------------|---------|
| Peel Cucumber | 100 | 220 | 7.3 | 450 |
| Cut Banana | 100 | 277 | 9.2 | 500 |
| Insert USB | 100 | 618 | 20.6 | 1100 |
| Play Mahjong | 100 | 200 | 6.7 | 400 |
| Pick and Place | 100 | 733 | 24.4 | 1200 |
| Clean Whiteboard | 100 | 833 | 27.8 | 1400 |

**说明**: 所有任务各采集 100 条遥操作演示，Pick and Place 和 Clean Whiteboard 回合最长（约 30s），是最具挑战性的任务。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 6 任务演示数据集 | 600 条（每任务 100 条） | 真实世界遥操作，包含视觉+触觉+本体感知 | 训练/验证 |
| 泛化测试 | 4 种环境扰动各 20 trials | 台高变化、排布变化、外观变化、背景变化 | 测试泛化性 |

### 实现细节

- **Backbone**: 预训练视频 Diffusion Transformer（基于 Cosmos-Policy）
- **优化器**: Fused Adam，lr=10⁻⁴，β₁=0.9，β₂=0.99
- **批量大小**: 每 GPU 16-25（视任务而定）
- **训练步数**: 20,000 步，含 2,000 步 warmup，衰减至 0.3
- **精度**: [[混合精度训练|混合 bfloat16 精度]]
- **动作块长度**: H=20
- **采集频率**: 30 Hz
- **硬件**: 训练 8× NVIDIA H100 GPU；推理 NVIDIA A800 GPU

### 可视化结果

- t-SNE 分析表明不同操作任务的触觉表征形成清晰分离的聚类，触觉模态确实学到了判别性信息
- 接触门时序曲线与人类对接触时刻的直觉一致，无需标注接触标签
- 泛化实验中，Dream-Tac 在台高+5cm 时仍有 75% 成功率，而 Cosmos Policy 直接失败（0%）

---

## 批判性思考

### 优点

1. **优雅的无监督门控设计**: 接触门 $g_t$ 完全由触觉图像帧差计算，无需接触标签，大幅降低数据采集门槛
2. **模态统一性强**: 将触觉传感器输出（也是 RGB 图像）自然融入视频 DiT 框架，无需特殊设计的触觉编码器
3. **工程实用性好**: FlashBias + 步缓存实现 2.9×/1.8× 加速，解决了 WAM 部署的实时性瓶颈

### 局限性

1. **触觉传感器强绑定**: 依赖 Xense Photon 这种基于视觉的触觉传感器，无法直接适配力矩传感器或电阻式传感器
2. **演示数据规模有限**: 每任务仅 100 条演示，任务种类少（6 个），泛化能力的上限尚未验证
3. **门控超参数敏感**: $g_{min}, g_{max}, k$ 等门控参数需在不同任务间调节，自适应性待提升

### 潜在改进方向

1. **多模态触觉支持**: 将门控机制推广到力矩传感器、电阻阵列等其他触觉模态
2. **大规模预训练**: 在更大规模遥操数据上预训练触觉 WAM，提升跨任务泛化
3. **自适应门控学习**: 将 $g_{min}, g_{max}$ 等参数也纳入端到端学习，减少人工调参

### 可复现性评估

- [x] 代码开源（GitHub: LYFCLOUDFAN/Dream-Tac）
- [ ] 预训练模型（代码已开源，模型权重未确认）
- [x] 训练细节完整（超参数、硬件配置详尽）
- [ ] 数据集可获取（演示数据为自采集，未公开）

---

## 关联笔记

### 基于

- [[Cosmos-Policy]]: Dream-Tac 的 Backbone 基于 Cosmos-Policy 的视频 DiT，并在其基础上加入触觉模态
- [[Video Diffusion Transformer]]: 核心架构基础，联合视觉-动作-触觉的 token 生成

### 对比

- [[Cosmos-Policy]]: 纯视觉 WAM，作为主要消融基线（51.7% vs 83.3%）
- [[ForceVLA]]: 力矩传感器驱动的触觉策略，同类接触感知方法
- [[OpenPI]]: π₀、π₀.₅，视觉语言动作模型，无触觉模态

### 方法相关

- [[接触感知自注意力|Contact-Aware Self-Attention (CASA)]]: 本文核心创新机制
- [[FlashAttention]]: FlashBias 的底层加速内核
- [[扩散步缓存|Diffusion-Step Caching]]: 推理加速技术
- [[Action Chunking]]: 动作预测的基本单元
- [[门控机制]]: 接触门的设计思想

### 硬件/数据相关

- [[Franka Emika Panda]]: 实验机械臂平台
- [[Intel RealSense D435]]: 视觉传感器（固定 + 腕部）

---

## 速查卡片

> [!summary] Dream-Tac
> - **核心**: 首个将触觉动态纳入联合建模的 World Action Model，通过接触门控自注意力选择性融合触觉信号
> - **方法**: 预训练视频 DiT + CASA（门控 logit bias）+ FlashBias 加速，联合预测动作/视觉/触觉
> - **结果**: 6 任务平均成功率 83.3%（vs Cosmos-Policy 51.7%），2.9× 训练加速，1.8× 推理加速
> - **代码**: https://github.com/LYFCLOUDFAN/Dream-Tac

---

*笔记创建时间: 2026-06-10*

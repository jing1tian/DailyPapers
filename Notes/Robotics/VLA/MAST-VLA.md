---
title: "MAST-VLA: Skill-Conditioned Adaptation with Trajectory-Affordance Priors for Multi-Skill Robot Manipulation"
method_name: "MAST-VLA"
authors: [Jiachi Yang, Zhiyi Xu, Hongquan Zhang, Xin Tan, Zhizhong Zhang, Jingyu Gong, Yuan Xie]
year: 2026
venue: CoRL 2026
tags: [vla, skill-conditioning, affordance, lora-moe, manipulation, long-horizon, multi-skill]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.32312v1
created: 2026-10-27
---

# 论文笔记：MAST-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Innovation Institute; Tianjin University; East China Normal University; Tongji University |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[VLA-Adapter]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.32312) |

---

## 一句话总结

> MAST-VLA 通过技能感知的 LoRA-MoE 适配器和轨迹可供性空间先验，对 VLA 主干进行双路条件化，使长时序多技能机器人操作的成功率达到 CALVIN ABC→D SOTA（Avg. Len 4.65）。

---

## 核心贡献

1. **双路条件化框架**: 同时建模"做哪个技能"（primitive-skill semantics）和"在哪里接触/如何运动"（object-interaction semantics），弥补现有 [[VLA（视觉-语言-动作模型）]] 的两大建模盲点。
2. **技能感知 [[LoRA-MoE]] 适配器**: 在 LLM 主干中加入技能查询 token 和每层路由偏置，使专家分配随原语技能类型软性偏移，避免跨任务共享单一更新。
3. **轨迹可供性空间先验**: 将预测的 [[可供性（Affordance）]] 热图下采样为 patch 级注意力偏置，引导动作头关注目标交互区域，无需修改视觉编码器或动作表征格式。

---

## 问题背景

### 要解决的问题

多技能长时序操作中，技能切换处的误差积累会导致序列中断。现有 [[VLA（视觉-语言-动作模型）]] 策略在"当前应执行哪类原语技能"和"应在何处以何种方式接触物体"两个语义层面建模不足。

### 现有方法的局限

- 适配器方案（LoRA、Adapter 等）对所有技能共用同一更新，无法针对特定原语技能激活差异化专家。
- 空间引导方案（Transporter、CLIPort 等）通常修改动作表征格式或视觉编码器，迁移成本高；基于演示轨迹的可供性方法也未直接用于动作解码注意力偏置。

### 本文的动机

通过将**技能类型**编码为可学习的查询 token 和路由偏置，使 MoE 的专家激活随技能软性切换；通过将**未来轨迹**背投影为 2D 高斯密度热图，在推理时由当前观测预测该热图并注入动作头注意力，从而用训练时可得的 oracle 轨迹监督推理时的空间先验。

---

## 方法详解

### 模型架构

MAST-VLA 采用**双路条件化** [[VLA（视觉-语言-动作模型）]] 架构：

- **输入**: 语言指令 $\mathbf{l}$ + 当前观测 $\mathbf{o}_t$ + 粗粒度技能条件 $s_t$
- **Backbone**: Qwen2.5-0.5B（LLM 主干）+ 技能感知 [[LoRA-MoE]] 适配器层
- **空间先验模块**: 可供性热图头（FiLM 调制 CNN）→ patch 级注意力偏置 $\mathbf{M}$
- **动作头**: 64 个动作查询，[[Action Chunking]] 方式输出 $H=8$ 步 7-D 动作
- **总参数**: 基于 Qwen2.5-0.5B，LoRA rank=64，6 个专家

### 核心模块

#### 模块 1：技能感知 [[LoRA-MoE]] 适配器

**设计动机**: 不同原语技能（旋转 vs 抬起 vs 放置）对特征空间的需求差异显著，用一组共享 LoRA 更新无法充分区分。通过 [[Mixture-of-Experts]] 与 [[LoRA]] 的结合，同时保持 VLA 预训练知识。

**具体实现**:

- **技能查询 bank**: $\mathbf{E}_{\mathrm{query}} \in \mathbb{R}^{N_s \times N_q \times d}$，每种技能类型 $s_t$ 取出 $N_q=32$ 个查询 token，拼入输入序列。
- **路由偏置 bank**: 每层维护 $\mathbf{E}_{\mathrm{route}}^{(\ell)} \in \mathbb{R}^{N_s \times N_e}$，对当前技能取出偏置向量 $\mathbf{b}_s^{(\ell)} \in \mathbb{R}^{N_e}$，加到门控 logits 上。
- **Top-$k$ 专家选择**: 路由概率计算后选取得分最高的 $k=2$ 个专家，加权聚合 LoRA 残差更新。

#### 模块 2：轨迹可供性空间先验

**设计动机**: 接触点和运动方向是操作成功的关键。将演示轨迹中的未来末端执行器位置背投影到像素空间，构建可学习的密度先验。

**具体实现**:
- 训练时使用 oracle 未来轨迹 $\{\mathbf{g}_\tau\}$ 构建目标热图 $H^*$（见关键公式）。
- 推理时由当前观测通过热图头预测 $\hat{H}$：以注意力池化的 VLA 隐状态为上下文，点图 CNN（膨胀卷积） + [[FiLM]] 调制产生 logits。
- 将 $\hat{H}$ 归一化后下采样到 patch 分辨率得到 $\mathbf{M}$，作为动作头视觉交叉注意力的加性偏置。

---

## 关键公式

### 公式 1：[[VLA（视觉-语言-动作模型）|策略输入输出映射]]

$$
\pi_\theta : (\mathbf{o}_t, \mathbf{l}, s_t) \mapsto \hat{\mathbf{a}}_{t:t+H-1}
$$

**含义**: VLA 策略将观测、语言指令和技能条件映射到未来 $H$ 步动作块。

**符号说明**:
- $\mathbf{o}_t$: 当前观测（RGB-D 图像）
- $\mathbf{l}$: 语言指令
- $s_t$: 粗粒度技能条件（来自预定义技能分类，共 7 类）
- $H$: 动作块长度（$H=8$）

---

### 公式 2：[[LoRA-MoE|技能 bank 定义]]

$$
\mathbf{E}_{\mathrm{query}} \in \mathbb{R}^{N_s \times N_q \times d}, \quad \mathbf{E}_{\mathrm{route}}^{(\ell)} \in \mathbb{R}^{N_s \times N_e}
$$

**含义**: 两个可学习的参数 bank——技能查询 bank 提供附加 token，路由偏置 bank 在各层软性偏移专家激活概率。

**符号说明**:
- $N_s$: 技能类别数（7 类）
- $N_q$: 每个技能的查询 token 数（$N_q=32$）
- $d$: 隐状态维度
- $N_e$: 每层专家数（$N_e=6$）
- $\ell$: 层索引

---

### 公式 3：[[LoRA-MoE|输入序列拼接]]

$$
\mathbf{X}_t = [\mathbf{V}_t;\, \mathbf{L}_t;\, \mathbf{A};\, \mathbf{Q}_{s_t}]
$$

**含义**: 将视觉 token、语言 token、动作查询 token 与当前技能对应的技能查询 token 拼接为完整输入序列。

**符号说明**:
- $\mathbf{V}_t$: 视觉 token（256 个 patch）
- $\mathbf{L}_t$: 语言 token
- $\mathbf{A}$: 动作查询 token（64 个）
- $\mathbf{Q}_{s_t}$: 技能查询 token（从 $\mathbf{E}_{\mathrm{query}}$ 按 $s_t$ 取出）

---

### 公式 4：[[LoRA-MoE|技能偏置路由与残差更新]]

$$
\begin{aligned}
\mathbf{p} &= \operatorname{softmax}(\mathbf{W}_{\mathrm{gate}}\mathbf{h} + \mathbf{b}_s^{(\ell)}) \\
\mathcal{T} &= \operatorname{TopK}(\mathbf{p}, k) \\
\bar{p}_i &= \frac{p_i}{\sum_{j \in \mathcal{T}} p_j} \\
\Delta\mathbf{h} &= \sum_{i \in \mathcal{T}} \bar{p}_i \mathbf{B}_i \mathbf{A}_i \mathbf{h}
\end{aligned}
$$

**含义**: 路由门控加上技能偏置后，选 top-$k=2$ 个专家，对其路由概率重归一化，以加权和计算 LoRA 残差更新。

**符号说明**:
- $\mathbf{W}_{\mathrm{gate}}$: 门控权重矩阵
- $\mathbf{b}_s^{(\ell)}$: 当前技能对应的第 $\ell$ 层路由偏置
- $\mathcal{T}$: 被选中的 top-$k$ 专家集合
- $\mathbf{B}_i, \mathbf{A}_i$: 第 $i$ 个专家的 LoRA 上/下投影矩阵

---

### 公式 5：[[可供性（Affordance）|轨迹可供性热图目标]]

$$
\begin{aligned}
d_u^2 &= \min_{\tau} \| \mathbf{x}_u - \mathbf{g}_\tau \|_2^2 \\
H^*(u) &= \exp\!\left(-\frac{d_u^2}{2\sigma^2}\right) \mathbb{I}(d_u \le r_{\max})
\end{aligned}
$$

**含义**: 对每个像素 $u$，计算其 3D 背投影点到未来末端执行器轨迹的最近距离，构造高斯衰减的可供性目标热图。

**符号说明**:
- $u$: 像素坐标
- $\mathbf{x}_u$: 像素 $u$ 的 3D 背投影点
- $\mathbf{g}_\tau$: 时间步 $\tau$ 的未来末端执行器位置
- $\sigma$: 高斯带宽（$\sigma = 0.05$ m）
- $r_{\max}$: 截断半径（$r_{\max} = 0.20$ m）

---

### 公式 6：[[可供性（Affordance）|注意力空间先验注入]]

$$
\mathbf{A} = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}_{\mathrm{vis}}^\top}{\sqrt{d_h}} + \tau \mathbf{M}\right)
$$

**含义**: 在动作查询对视觉 patch 的交叉注意力中，加入归一化下采样热图 $\mathbf{M}$ 作为加性偏置，使注意力向目标接触区域偏移，而不屏蔽其他区域。

**符号说明**:
- $\mathbf{Q}$: 动作查询的 Query 矩阵
- $\mathbf{K}_{\mathrm{vis}}$: 视觉 patch 的 Key 矩阵
- $d_h$: 注意力头维度
- $\tau$: 先验强度缩放系数
- $\mathbf{M}$: patch 级空间先验（由 $\hat{H}$ 下采样得到）

---

### 公式 7：[[LoRA-MoE|联合训练目标]]

$$
\mathcal{L} = \underbrace{\|\hat{\mathbf{a}}_{t:t+H-1} - \mathbf{a}_{t:t+H-1}\|_1}_{\mathcal{L}_{\mathrm{action}}} + \lambda_{\mathrm{hm}} \mathcal{L}_{\mathrm{hm}} + \lambda_{\mathrm{moe}} \mathcal{L}_{\mathrm{moe}}
$$

**含义**: 动作 L1 回归损失 + 热图加权 BCE 损失 + MoE 路由负载均衡辅助损失的加权总和。

**符号说明**:
- $\hat{\mathbf{a}}_{t:t+H-1}$: 预测动作块
- $\mathbf{a}_{t:t+H-1}$: 真实动作块
- $\mathcal{L}_{\mathrm{hm}}$: 热图预测的加权 BCE 损失（正样本权重截断到 [1, 20]）
- $\mathcal{L}_{\mathrm{moe}}$: 标准路由辅助损失（负载均衡 + z-loss）
- $\lambda_{\mathrm{hm}} = 0.005$，负载均衡系数 $10^{-2}$，z-loss 系数 $10^{-4}$

---

## 关键图表

### Figure 1: Method Overview / 系统概览

![Figure 1](https://arxiv.org/html/2610.32312v1/method_overview.png)

**说明**: MAST-VLA 的整体架构。左路为技能感知 [[LoRA-MoE]] 适配器——技能查询 token 拼入输入序列，路由偏置向量调整每层的专家激活偏好；右路为轨迹可供性先验——热图头从 VLA 隐状态预测交互区域密度，下采样后作为动作头视觉交叉注意力的加性偏置。

---

### Figure 2: Real Robot Tasks / 真实机器人任务

![Figure 2](https://arxiv.org/html/2610.32312v1/real_robot_tasks.png)

**说明**: Franka Research 3 机器人在两个桌面场景中的原语技能演示，展示旋转、推/滑动、关节体操作、抬起/抓取、放置/存放、堆叠/嵌套、切换 7 类技能的真实执行情况。

---

### Figure 3: Real Robot Results / 真实机器人结果

![Figure 3](https://arxiv.org/html/2610.32312v1/real_robot_results.png)

**说明**: 两个场景 8 类长度为 3 的序列（共 160 次试验）中各方法 Success@1-3 的对比结果，MAST-VLA 在 Avg. Len 上以 2.62 超越 π0.5（2.41）和 OpenVLA-OFT（1.85）。

---

### Figure 4: LoRA-MoE Routing Signatures / 路由签名可视化

![Figure 4](https://arxiv.org/html/2610.32312v1/moe_routing_top2_signature_grid.png)

**说明**: 各技能类型下 Top-2 [[LoRA-MoE]] 专家选择的 routing signature 热图网格，验证不同技能确实激活了不同的专家组合，证明技能偏置路由的有效性。

---

### Figure 5: Affordance Heatmap Visualization / 可供性热图可视化

![Figure 5](https://arxiv.org/html/2610.32312v1/figures/heatmap_visualization.png)

**说明**: 不同技能（抓取、放置、推动等）下预测的 [[可供性（Affordance）]] 热图，热区集中于目标物体接触面，与 oracle 目标热图高度吻合。

---

### Table 1：CALVIN ABC→D 基准对比

| Method | @1 | @2 | @3 | @4 | @5 | Avg. Len |
|--------|------|------|------|------|------|----------|
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| PointMapPolicy | 96.1 | 88.6 | 80.5 | 72.3 | 63.6 | 4.01 |
| EL3DD | 96.6 | 90.0 | 82.4 | 74.2 | 66.2 | 4.09 |
| OpenVLA-OFT | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.10 |
| Seer-Large | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.33 |
| FALCON | 98.4 | 94.5 | 88.6 | 82.5 | 75.5 | 4.40 |
| VLA-Adapter-Pro | 98.5 | 95.0 | 90.5 | 85.3 | 80.0 | 4.50 |
| FLOWER | 99.3 | 95.9 | 90.5 | 84.8 | 77.5 | 4.54 |
| **MAST-VLA** | **99.4** | **97.4** | **94.2** | **89.2** | **84.7** | **4.65** |

**关键发现**: MAST-VLA 在所有 Success@k 上均达到 SOTA，尤其 @4（89.2% vs 85.3%）和 @5（84.7% vs 80.0%）相对 VLA-Adapter-Pro 提升最大，说明双路条件化对抑制长时序误差积累效果显著。

---

### Table 2：消融实验（CALVIN ABC→D）

| Query | LoRA-MoE | Aff. Heatmap | Skill Cond. | @1 | @2 | @3 | @4 | @5 | Avg. Len |
|-------|----------|--------------|-------------|------|------|------|------|------|----------|
| — | — | — | — | 98.5 | 95.0 | 90.5 | 85.3 | 80.0 | 4.50 |
| ✓ | — | — | matched | 98.7 | 95.6 | 91.3 | 86.4 | 81.1 | 4.53 |
| ✓ | ✓ | — | matched | 98.8 | 96.3 | 92.8 | 88.0 | 83.1 | 4.59 |
| ✓ | — | ✓ | — | 99.0 | 96.7 | 93.4 | 88.6 | 83.7 | 4.61 |
| ✓ | ✓ | ✓ | matched | 99.4 | 97.4 | 94.2 | 89.2 | 84.7 | **4.65** |
| ✓ | ✓ | ✓ | mismatched | 98.8 | 96.2 | 92.7 | 86.6 | 80.9 | 4.55 |

**关键发现**: 
- LoRA-MoE 单独贡献约 +0.06 Avg. Len；Affordance Heatmap 单独贡献约 +0.08；两者协同达到 +0.15。
- 将 matched 技能条件替换为 mismatched 随机条件，Avg. Len 从 4.65 降至 4.55，证明准确的技能条件信息是性能的必要保障。

---

### Table 3：技能分类（附录 A.1）

| 技能 ID | 技能名称 | 描述 |
|---------|----------|------|
| 0 | rotation | 旋转操作 |
| 1 | sliding/pushing | 滑动或推动物体 |
| 2 | articulated-object operation | 关节体操作（开关抽屉等）|
| 3 | lifting/picking | 抬起或抓取 |
| 4 | placing/storing | 放置或存放 |
| 5 | stacking/nesting | 堆叠或嵌套 |
| 6 | switching | 切换开关 |

**说明**: 技能映射使用归一化词汇匹配和模糊匹配，覆盖 CALVIN 任务集的所有子任务类型。

---

### Table 4：模型配置（附录 A.2）

| 参数 | 值 |
|------|-----|
| Backbone | Qwen2.5-0.5B |
| 输入分辨率 | 224 × 224 |
| 视觉 patch token 数 | 256（每帧）|
| 动作块长度 $H$ | 8 |
| 动作维度 | 7-D |
| 动作查询数 | 64 |
| 技能查询 token 数 $N_q$ | 32（每技能）|
| 每层 LoRA-MoE 专家数 $N_e$ | 6 |
| Top-$k$ 路由 | 2 |
| LoRA rank | 64，$\alpha = 128$ |
| 负载均衡系数 | $10^{-2}$ |
| z-loss 系数 | $10^{-4}$ |

---

### Table 5：训练超参数（附录 B.2）

| 超参数 | 值 |
|--------|-----|
| 训练步数 | 50,000 |
| GPU 数量 | 8 |
| Batch size（每 GPU）| 16 |
| 优化器 | AdamW |
| LoRA LR | $2 \times 10^{-4}$ |
| 路由 LR | $2 \times 10^{-5}$ |
| 技能查询 LR | $1 \times 10^{-4}$ |
| 动作头 LR | $5 \times 10^{-5}$ |
| $\lambda_{\mathrm{hm}}$ | 0.005 |

---

### Table 6：真实机器人评测序列（附录 C.2）

**场景 1** (4 条 length-3 序列，20 次/条)

| 序列 | 子任务 1 | 子任务 2 | 子任务 3 |
|------|----------|----------|----------|
| S1-1 | Pick cup | Nest cup | Push box |
| S1-2–S1-4 | (其余三类长度3序列) | — | — |

**场景 2** (4 条 length-3 序列，20 次/条)

| 序列 | 子任务 1 | 子任务 2 | 子任务 3 |
|------|----------|----------|----------|
| S2-1 | Pick flower | Insert flower | Stack blocks |
| S2-2–S2-4 | (其余三类长度3序列) | — | — |

---

### Table 7：真实机器人总体成功率

| Method | Success@1 | Success@2 | Success@3 | Avg. Len |
|--------|-----------|-----------|-----------|----------|
| OpenVLA-OFT | — | — | — | 1.85 |
| π0 | — | — | — | 2.18 |
| π0.5 | — | — | — | 2.41 |
| VLA-Adapter-Pro | — | — | — | 2.02 |
| **MAST-VLA** | **98.8%** | **92.5%** | **70.6%** | **2.62** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[CALVIN]] ABC→D | 1,000 测试序列 | 训练域 A-C，测试未见域 D，5 子任务连续 | 主要基准 |
| 真实机器人数据 | 自采集 | Franka R3 + 双 RealSense D435i，2 桌面场景 | 真实部署验证 |

### 实现细节

- **Backbone**: Qwen2.5-0.5B
- **优化器**: AdamW，差异化学习率（LoRA $2\times10^{-4}$，路由 $2\times10^{-5}$）
- **Batch Size**: 16 per GPU
- **训练步数**: 50,000 steps
- **硬件**: 8 × GPU

### 可视化结果

- 可供性热图在不同技能下呈现显著差异：抓取任务热区集中于物体顶部接触点，推动任务热区延伸至运动方向；验证了轨迹先验的直觉正确性。
- MoE 路由签名可视化显示：旋转类技能与放置类技能激活了不同的专家子集，体现了技能感知路由的有效分化。

---

## 批判性思考

### 优点

1. 双路条件化框架设计简洁，与现有 VLA 架构兼容性好，无需修改视觉编码器。
2. 技能分类仅需 7 类粗粒度标签，实际部署中可从语言指令自动匹配。
3. CALVIN 上显著提升（4.65 vs 4.54），尤其在 @4/@5 高难度指标上优势明显。
4. 真实机器人实验覆盖 160 次试验，较全面地验证了鲁棒性。

### 局限性

1. 没有显式的低层次偏差校正机制；末端执行器实际轨迹与预测轨迹偏差无法实时修正，长时序中可能积累。
2. 可供性先验依赖高质量 RGB-D 深度信息；遮挡、反光/透明表面或标定误差会降级热图质量。

### 潜在改进方向

1. 引入在线重规划（closed-loop replanning）以修正低层次轨迹偏差。
2. 对深度图做更鲁棒的预处理（inpainting 已部分缓解，可进一步用深度补全网络）。
3. 将技能条件扩展为细粒度分层分类或软标签，减少边界模糊问题。

### 可复现性评估

- [ ] 代码开源（论文未提及代码仓库）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（附录提供详细超参数）
- [x] 数据集可获取（CALVIN 公开；真实数据自采集）

---

## 关联笔记

### 基于

- [[VLA-Adapter]]: 基线方法，MAST-VLA 在其基础上引入双路条件化
- [[VLA（视觉-语言-动作模型）]]: 核心架构类型
- [[LoRA]]: LoRA-MoE 的基础适配方法
- [[Mixture-of-Experts]]: MoE 路由机制

### 对比

- [[UniVLA]]: CALVIN 基线（Avg. Len 3.80 vs 4.65）
- [[可供性（Affordance）]]: 可供性引导方法的相关先验工作

### 方法相关

- [[LoRA-MoE]]: 技能感知适配器核心方法
- [[可供性（Affordance）]]: 轨迹可供性先验
- [[FiLM]]: 热图头中的调制机制
- [[Action Chunking]]: 动作输出表征

### 硬件/数据相关

- [[CALVIN]]: 主要评测 benchmark

---

## 速查卡片

> [!summary] MAST-VLA (CoRL 2026)
> - **核心**: 双路条件化 VLA——技能感知 LoRA-MoE 适配器 + 轨迹可供性空间先验
> - **方法**: 技能查询 token + 路由偏置 → 差异化 MoE 激活；轨迹热图 → patch 级注意力偏置
> - **结果**: CALVIN ABC→D Avg. Len **4.65** (SOTA)，真实机器人 Avg. Len **2.62**
> - **代码**: 未公开

---

*笔记创建时间: 2026-10-27*

---
title: "SG-WAM: Self-Guided World Modeling in Geometry-Aware Policy Space"
method_name: "SG-WAM"
authors: [Ruiteng Zhao, Zhengshen Zhang, Yue Su, Wenshuo Wang, Jiahui Li, Zhiyuan Yang, "Francis E.H. Tay", "Marcelo H. Ang Jr.", Haiyue Zhu]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, flow-matching, geometric-supervision, vla, zero-shot-transfer, dynamics-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.01397
created: 2026-08-05
---

# 论文笔记：SG-WAM: Self-Guided World Modeling in Geometry-Aware Policy Space

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[VLA-JEPA]], [[Spatial Forcing]], [[Flash-WAM\|Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.01397) |

---

## 一句话总结

> SG-WAM 通过在策略自身的几何感知表示空间内做自引导潜在动力学预测，以 0.9B 参数量无需大规模具身预训练即在 LIBERO 上达到 98.5% 成功率并在零样本迁移上超越所有 7B 基线。

---

## 核心贡献

1. **可学习动力学 Token（Learnable Dynamics Tokens）**: 将 $N_q$ 个可学习 token 插入 [[Vision-Language-Action Model|VLM]] 序列，让其在动作条件下演化以隐式编码未来状态，无需预测像素或外部辅助目标。
2. **自引导世界预测器（SGWP）**: 利用 [[EMA|指数移动平均]]（EMA）维护策略骨干的动量拷贝生成预测目标，保证预测目标与动作生成路径严格对齐，避免目标-策略不匹配问题。
3. **几何监督（Geometric Supervision）**: 引入冻结的 [[VGGT]] 3D 基础模型对视觉 token 施加余弦相似度约束，将空间结构信息注入表示，同时在推理阶段零成本（几何教师被丢弃）。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model|WAM]] 方法在构建世界模型目标时存在两类问题：一是预测像素级或重建级目标（显式 WAM）带来感知负担；二是使用与策略无直接关联的辅助潜在目标（Auxiliary Latent WAM）导致目标-策略不匹配，限制了表示对动作生成的正向迁移。

### 现有方法的局限

- **显式 WAM**（如 [[WorldVLA]]）：预测像素级未来帧，计算开销大，且感知细节对操作未必有用。
- **辅助潜在 WAM**（如 [[VLA-JEPA]]）：预测目标来自独立编码器，与策略学习路径脱节，存在表示对齐困难。
- **大模型路线**（如 [[π₀]]、[[GR00T N1.6]]）：依赖大规模具身数据预训练，参数量 3B+，资源需求高。

### 本文的动机

若将世界建模的预测目标和预测空间都置于策略自身的表示中（"自引导"），则动力学学习天然与动作生成对齐，表示质量互相促进。同时，几何监督可在无额外推理成本的条件下为视觉 token 注入空间归纳偏置，改善对位置敏感任务的泛化。

---

## 方法详解

### 模型架构

SG-WAM 采用 **[[Vision-Language-Action Model|VLM]] + 流匹配动作头** 架构，训练时附加自引导世界预测器和几何教师，推理时仅保留主干和动作头：

- **输入**: 语言指令 $\bm{f}^L$ + 多视角观测 $\bm{f}^V_t$（主视角 + 辅助视角）+ 可学习动力学 token $\bm{Q}$
- **Backbone**: [[Qwen|Qwen3.5]]（0.8B）VLM，联合上下文化所有输入
- **核心模块**: [[Learnable Dynamics Tokens|动力学 Token]] + [[Self-Guided World Predictor|SGWP]] + [[VGGT]] 几何教师
- **输出**: [[Action Chunking|动作块]] $\bm{A}_t^{(\Delta)} \in \mathbb{R}^{\Delta \times d_a}$（$\Delta = 8$ 步）
- **总参数**: ~0.9B（VLM 0.8B + 动作头 + 附加模块）

### 核心模块

#### 模块 1：可学习动力学 Token（Learnable Dynamics Tokens）

**设计动机**: 在 [[Transformer]] 序列中引入一组可学习 token，让其从当前观测中提取动作相关状态，并作为 SGWP 的预测起点与目标。

**具体实现**:
- $N_q = 8$ 个可学习 token $\bm{Q}$ 拼接到视觉和语言 token 之后，送入 [[Self-Attention|自注意力]] VLM 联合上下文化
- 编码后得到动力学 token 表示 $\bm{H}_t^Q$ 和 EMA 目标 $\bar{\bm{H}}_{t+\Delta}^Q$
- 通过投影头 $\mathcal{P}_\phi$ 得到当前潜态 $\bm{z}_t^Q$ 和目标潜态 $\bar{\bm{z}}_{t+\Delta}^Q$

#### 模块 2：自引导世界预测器（SGWP）

**设计动机**: 用 [[EMA|指数移动平均]] 构建动量教师，保证预测目标始终在策略自身的表示流形内，利用 [[Stop-Gradient]] 操作阻断目标梯度传播。

**具体实现**:
- SGWP 网络 $\mathcal{F}_\psi$ 先通过 [[Self-Attention|自注意力]] 上下文化当前动力学 token 状态 $\bm{z}_t^Q$
- 再通过 [[Cross-Attention|交叉注意力]] 以动作嵌入 $\bm{e}_t^A$ 为键值，预测未来潜态 $\hat{\bm{z}}_{t+\Delta}^Q$
- 目标来自 EMA 骨干在未来帧 $t+\Delta$ 上的推断（经 [[Stop-Gradient]] 截断）

#### 模块 3：几何教师（Geometry Teacher）

**设计动机**: 利用冻结的 [[VGGT]] 3D 基础模型提供几何监督，使视觉 token 的分布保留空间结构，同时推理时完全无额外开销。

**具体实现**:
- 冻结 VGGT 对主视角图像提取 3D 特征 $\bm{Z}_t^G \in \mathbb{R}^{N_v^m \times d}$（$37 \times 37 = 1369$ 个 patch 经 $8 \times 8$ 自适应平均池化压缩为 64 个）
- 可学习几何投影头 $\mathcal{G}_\gamma$ 对主视角 VLM 特征 $\bm{H}_t^{V,m}$ 做投影 $\hat{\bm{Z}}_t^G$
- 用余弦相似度损失对齐 $\hat{\bm{Z}}_t^G$ 与 $\bm{Z}_t^G$，迫使视觉 token 保留空间信息

---

## 关键公式

### 公式 1：[[Action Chunking|动作块]] 定义

$$
\bm{A}_{t}^{(\Delta)} = \left[\bm{a}_{t}, \ldots, \bm{a}_{t+\Delta-1}\right]
$$

**含义**: 从当前时刻 $t$ 起连续 $\Delta$ 步动作组成的动作块，作为流匹配的预测目标。

**符号说明**:
- $\bm{a}_t \in \mathbb{R}^{d_a}$: 单步动作向量
- $\Delta$: 动作块长度（实验中 $\Delta=8$）

### 公式 2：[[VLM|VLM 上下文化]]

$$
\bm{H}_{t} = \mathcal{E}_{\theta}\left([\bm{f}_{t}^{V}, \bm{f}^{L}, \bm{Q}]\right), \quad \bm{H}_{t} = [\bm{H}_{t}^{V}, \bm{H}^{L}, \bm{H}_{t}^{Q}]
$$

**含义**: VLM 编码器 $\mathcal{E}_\theta$ 联合上下文化多视角视觉特征、语言特征和动力学 token，输出拆分为视觉、语言、动力学三部分。

**符号说明**:
- $\bm{f}_t^V$: 多视角视觉特征
- $\bm{f}^L$: 语言指令特征
- $\bm{Q}$: 可学习动力学 token 参数
- $\bm{H}_t^Q$: 上下文化后的动力学 token 表示

### 公式 3：[[EMA|EMA 动量更新]]

$$
\bar{\Theta} \leftarrow \mu \bar{\Theta} + (1-\mu)\Theta, \quad \mu = 0.999
$$

**含义**: 动量教师参数 $\bar{\Theta}$ 以 EMA 方式跟踪在线策略参数 $\Theta$，提供稳定的预测目标而不直接反向传播。

**符号说明**:
- $\Theta$: 在线策略骨干参数
- $\bar{\Theta}$: EMA 动量教师参数
- $\mu = 0.999$: 动量系数

### 公式 4：[[EMA|EMA 目标生成]]

$$
\bar{\bm{H}}_{t+\Delta}^{Q} = \left[\mathcal{E}_{\bar{\theta}}\left([\bm{f}_{t+\Delta}^{V}, \bm{f}^{L}, \bar{\bm{Q}}]\right)\right]^{Q}
$$

$$
\bar{\bm{z}}_{t+\Delta}^{Q} = \mathcal{P}_{\bar{\phi}}\left(\bar{\bm{H}}_{t+\Delta}^{Q}\right)
$$

**含义**: 用 EMA 骨干处理未来帧 $t+\Delta$ 的观测，得到目标动力学 token 的潜态表示 $\bar{\bm{z}}_{t+\Delta}^Q$。

**符号说明**:
- $\mathcal{E}_{\bar{\theta}}$: EMA 骨干编码器
- $\bar{\bm{Q}}$: EMA 目标的动力学 token
- $\mathcal{P}_{\bar{\phi}}$: EMA 投影头
- $\bar{\bm{z}}_{t+\Delta}^Q$: 目标潜态（经 stop-gradient 后参与损失计算）

### 公式 5：[[Self-Guided World Predictor|SGWP 预测]]

$$
\hat{\bm{z}}_{t+\Delta}^{Q} = \mathcal{F}_{\psi}\left(\bm{z}_{t}^{Q}, \bm{e}_{t}^{A}\right), \quad \hat{\bm{z}}_{t+\Delta}^{Q} \approx \operatorname{sg}\left(\bar{\bm{z}}_{t+\Delta}^{Q}\right)
$$

**含义**: SGWP 以当前动力学潜态 $\bm{z}_t^Q$ 和动作嵌入 $\bm{e}_t^A$ 为输入预测未来潜态，目标经 [[Stop-Gradient]] 截断梯度。

**符号说明**:
- $\mathcal{F}_\psi$: SGWP 网络（自注意力 + 交叉注意力）
- $\bm{z}_t^Q$: 当前动力学潜态
- $\bm{e}_t^A$: 干预动作块的嵌入
- $\operatorname{sg}(\cdot)$: stop-gradient 算子

### 公式 6：[[Self-Guided World Predictor|预测损失]] $\mathcal{L}_\text{pred}$

$$
\mathcal{L}_{\mathrm{pred}} = \frac{1}{N_q d} \left\| \hat{\bm{z}}_{t+\Delta}^{Q} - \operatorname{sg}\left(\bar{\bm{z}}_{t+\Delta}^{Q}\right) \right\|_{F}^{2}
$$

**含义**: 以 Frobenius 范数衡量 SGWP 预测的动力学潜态与 EMA 目标之间的距离，监督 stop-gradient 保证梯度只流向 SGWP 和在线骨干。

**符号说明**:
- $N_q$: 动力学 token 数量（$=8$）
- $d$: 潜态维度
- $\|\cdot\|_F$: Frobenius 范数

### 公式 7：[[Geometric Supervision|几何投影]]

$$
\hat{Z}_{t}^{G} = \mathcal{G}_{\gamma}\left(\bm{H}_{t}^{V,m}\right)
$$

**含义**: 几何投影头将主视角 VLM 特征映射到 VGGT 特征空间，用于与冻结 VGGT 特征对齐。

**符号说明**:
- $\mathcal{G}_\gamma$: 可学习几何投影头
- $\bm{H}_t^{V,m}$: 主视角 VLM 特征（经 $8 \times 8$ 自适应平均池化）

### 公式 8：[[Geometric Supervision|几何损失]] $\mathcal{L}_\text{geo}$

$$
\mathcal{L}_{\mathrm{geo}} = \frac{1}{N_v^m} \sum_{j=1}^{N_v^m} \left[1 - \cos\left(\hat{\bm{Z}}_{t,j}^{G}, \bm{Z}_{t,j}^{G}\right)\right]
$$

**含义**: 逐 patch 计算余弦距离，最小化几何投影与 VGGT 特征的余弦不相似度，迫使视觉 token 保留空间结构。

**符号说明**:
- $N_v^m$: 主视角 patch 数（$= 64$，经 $8\times 8$ 池化后）
- $\hat{\bm{Z}}_{t,j}^G$: 第 $j$ 个 patch 的投影特征
- $\bm{Z}_{t,j}^G$: VGGT 提供的第 $j$ 个 patch 的目标特征
- $\cos(\cdot, \cdot)$: 余弦相似度

### 公式 9：[[Conditional Flow Matching|动作流匹配损失]] $\mathcal{L}_\text{act}$

$$
\mathcal{L}_{\mathrm{act}} = \mathbb{E}_{\bm{A}_t, \bm{\epsilon}, \tau} \left[ \left\| v_{\omega}\left(\bm{A}_{t}^{\tau}, \tau, \bm{H}_{t}\right) - \left(\bm{A}_{t} - \bm{\epsilon}\right) \right\|_2^2 \right]
$$

**含义**: 训练流速度网络 $v_\omega$ 从高斯噪声流向真实动作，以上下文化表示 $\bm{H}_t$ 为条件。

**符号说明**:
- $v_\omega$: 流速度网络（动作专家）
- $\bm{A}_t^\tau$: 时间 $\tau$ 的中间动作（插值）
- $\tau \sim \mathcal{U}(0,1)$: 流时间步
- $\bm{\epsilon} \sim \mathcal{N}(0, I)$: 高斯噪声

### 公式 10：[[SG-WAM|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{geo}} \mathcal{L}_{\mathrm{geo}} + \lambda_{\mathrm{pred}} \mathcal{L}_{\mathrm{pred}}
$$

**含义**: 三项加权损失联合优化，$\lambda_\text{geo} = \lambda_\text{pred} = 0.1$ 保证辅助监督不压制主动作学习。

**符号说明**:
- $\lambda_\text{geo} = 0.1$: 几何损失权重
- $\lambda_\text{pred} = 0.1$: 预测损失权重

### 公式 11：空动作消融（Null-Action Ablation）

$$
\widetilde{\bm{A}}_{t}^{(\Delta)} = \bm{0}, \quad \widetilde{\bm{e}}_{t}^{A} = \mathcal{A}_{\eta}\left(\widetilde{\bm{A}}_{t}^{(\Delta)}\right)
$$

**含义**: 消融实验中将干预动作置为全零，验证 SGWP 中动作条件的有效性。

**符号说明**:
- $\widetilde{\bm{A}}_t^{(\Delta)}$: 全零动作块
- $\mathcal{A}_\eta$: 动作嵌入网络

---

## 关键图表

### Figure 1: Conceptual Comparison / 概念对比

![Figure 1](https://arxiv.org/html/2608.01397v1/x1.png)

**说明**: 现有 WAM 与 SG-WAM 的概念对比。显式 WAM 预测像素，辅助潜变量 WAM 目标与策略空间脱节；SG-WAM 直接在几何结构化的策略表示空间内做动作条件动力学预测，消除目标-策略不匹配。

### Figure 2: SG-WAM Framework Overview / 框架总览

![Figure 2](https://arxiv.org/html/2608.01397v1/x2.png)

**说明**: SG-WAM 整体框架。[[Qwen|Qwen3.5 VLM]] 联合处理多视角观测、语言和[[Learnable Dynamics Tokens|动力学 Token]]；[[Conditional Flow Matching|流匹配]]动作头生成动作块；[[Self-Guided World Predictor|SGWP]] 和[[VGGT|几何教师]]仅在训练时激活，推理时完全去除。

### Figure 3: Real-World Platform / 真实机器人平台

![Figure 3](https://arxiv.org/html/2608.01397v1/x3.png)

**说明**: UR5e 机器人平台与三个真实操作任务（Pick and Place、Towel Folding、Toolbox Organization）的可视化，展示了 SG-WAM 的实物验证环境。

### Figure 4: Attention Map with Geometric Supervision / 有几何监督的注意力图

![Figure 4](https://arxiv.org/html/2608.01397v1/x4.png)

**说明**: 动力学 token 对主视角图像 token 的中间层[[Self-Attention|注意力图]]。几何监督使得注意力聚焦于操作相关的空间关系（如物体位置），而无几何监督时注意力分散。

### Figure 5: Self-Guided World Predictor (SGWP) / 自引导世界预测器

![Figure 5](https://arxiv.org/html/2608.01397v1/x5.png)

**说明**: SGWP 详细结构。动力学 token 潜态先经[[Self-Attention|自注意力]]上下文化，再通过[[Cross-Attention|交叉注意力]]与干预动作嵌入交互，预测未来潜态；目标由[[EMA|EMA 动量教师]]生成并经[[Stop-Gradient|stop-gradient]]截断。

### Figure 6: Pick and Place Task / 抓取放置任务

![Figure 6](https://arxiv.org/html/2608.01397v1/x6.png)

**说明**: Pick and Place 任务的完整操作序列可视化（ID 条件），展示了 SG-WAM 75% 成功率的执行过程。

### Figure 7: Towel Folding Task / 毛巾折叠任务

![Figure 7](https://arxiv.org/html/2608.01397v1/x7.png)

**说明**: Towel Folding 任务的完整操作序列可视化（ID 条件），需要两步折叠操作，45% 成功率。

### Figure 8: Toolbox Organization Task / 工具箱整理任务

![Figure 8](https://arxiv.org/html/2608.01397v1/x8.png)

**说明**: Toolbox Organization 任务的完整序列可视化，包含螺丝刀 + 两个齿轮 + 关闭盖子，50% 成功率。

### Figure 9: OOD Settings Visualization / 分布外设置可视化

![Figure 9](https://arxiv.org/html/2608.01397v1/x9.png)

**说明**: 三种 OOD 设置示例：背景变换（Background Shift）、光照变化（Light Change）、新颖物体（Novel Object），用于测试泛化鲁棒性。

### Figure 10: Attention Weight Matrix / 注意力权重矩阵

![Figure 10](https://arxiv.org/html/2608.01397v1/x10.png)

**说明**: 所有动力学 token 对主视角图像 token 注意力权重矩阵的可视化对比，直观展示几何监督促使注意力在不同 token 间产生更有结构的空间分工。

### Figure 11: Pick and Place — Background Shift

![Figure 11](https://arxiv.org/html/2608.01397v1/x11.png)

**说明**: Pick and Place 任务在背景变换 OOD 条件下的执行可视化（成功率 55%）。

### Figure 12: Pick and Place — Light Change

![Figure 12](https://arxiv.org/html/2608.01397v1/x12.png)

**说明**: Pick and Place 任务在光照变化 OOD 条件下的执行可视化（成功率 60%）。

### Figure 13: Pick and Place — Novel Object

![Figure 13](https://arxiv.org/html/2608.01397v1/x13.png)

**说明**: Pick and Place 任务在新颖物体 OOD 条件下的执行可视化（成功率 40%）。

### Figure 14: Towel Folding — Background Shift

![Figure 14](https://arxiv.org/html/2608.01397v1/x14.png)

**说明**: Towel Folding 任务在背景变换 OOD 条件下的执行可视化（成功率 25%）。

### Figure 15: Towel Folding — Light Change

![Figure 15](https://arxiv.org/html/2608.01397v1/x15.png)

**说明**: Towel Folding 任务在光照变化 OOD 条件下的执行可视化（成功率 35%）。

### Figure 16: Towel Folding — Novel Object

![Figure 16](https://arxiv.org/html/2608.01397v1/x16.png)

**说明**: Towel Folding 任务在新颖物体 OOD 条件下的执行可视化（成功率 25%）。

### Figure 17: LIBERO-Spatial 可视化

![Figure 17](https://arxiv.org/html/2608.01397v1/x17.png)

**说明**: LIBERO-Spatial 任务示例："Pick up the black bowl on the wooden cabinet and place it on the plate."，SG-WAM 成功率 99.4%。

### Figure 18: LIBERO-Object 可视化

![Figure 18](https://arxiv.org/html/2608.01397v1/x18.png)

**说明**: LIBERO-Object 任务示例："Pick up the alphabet soup and place it in the basket."，SG-WAM 成功率 99.8%。

### Figure 19: LIBERO-Goal 可视化

![Figure 19](https://arxiv.org/html/2608.01397v1/x19.png)

**说明**: LIBERO-Goal 任务示例："Turn on the stove."，SG-WAM 成功率 98.6%。

### Figure 20: LIBERO-Long 可视化

![Figure 20](https://arxiv.org/html/2608.01397v1/x20.png)

**说明**: LIBERO-Long 任务示例："Put both moka pots on the stove."，长时序任务，SG-WAM 成功率 96.2%。

### Figure 21: LIBERO-Plus 视角扰动可视化

![Figure 21](https://arxiv.org/html/2608.01397v1/x21.png)

**说明**: LIBERO-Plus 相机视角扰动条件下的任务执行示例："Put the wine bottle on top of the cabinet."。

### Figure 22: LIBERO-Plus 其他扰动可视化

![Figure 22](https://arxiv.org/html/2608.01397v1/x22.png)

**说明**: LIBERO-Plus 其他分布偏移条件（机器人本体/语言/光照/背景/噪声/布局）下的任务执行可视化。

---

### Table 1: LIBERO 仿真结果

| Method | Params | Embodied PT. | Spatial | Object | Goal | Long | Avg. |
|--------|--------|:------------:|---------|--------|------|------|------|
| OpenVLA-OFT | 7B | ✓ | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ | 3.3B | ✓ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π₀-FAST | 3.3B | ✓ | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀.₅ | 3.3B | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T N1.6 | 3B | ✓ | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| Spatial Forcing | 7B | ✓ | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| WorldVLA | 7B | ✗ | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| LAPA | 7B | ✓ | 55.4 | 58.8 | 74.6 | 73.8 | 65.7 |
| RynnVLA-002 | 7B | ✗ | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Mantis | 5.8B | ✓ | 98.8 | 99.2 | 94.4 | 94.2 | 96.7 |
| UniVLA | 7B | ✓ | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Fast-WAM | 6B | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| VLA-JEPA | 2B | ✗ | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |
| **SG-WAM** | **0.9B** | **✗** | **99.4** | **99.8** | **98.6** | **96.2** | **98.5** |

**说明**: SG-WAM 以 0.9B 参数、无具身预训练，在四个 LIBERO 子任务上均达到最优或并列最优，总均值 98.5% 与 Spatial Forcing（7B，有具身预训练）持平，参数量仅为其 1/7.8。

### Table 2: LIBERO-Plus 零样本迁移结果（七类分布偏移）

| Method | Params | Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|--------|--------|--------|-------|----------|-------|------------|-------|--------|---------|
| WorldVLA | 7B | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| Spatial Forcing | 7B | 20.1 | 13.4 | 40.9 | 29.1 | 33.4 | 25.7 | 39.3 | 29.1 |
| Mantis | 5.8B | 15.7 | 41.8 | 45.9 | 45.1 | 28.9 | 39.2 | 62.5 | 39.8 |
| UniVLA | 7B | 4.3 | 50.3 | 71.8 | 59.1 | 80.0 | 25.3 | 34.3 | 41.5 |
| Fast-WAM | 6B | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 50.0 |
| π₀ | 3.3B | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| VLA-JEPA | 2B | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |
| OpenVLA-OFT | 7B | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| **SG-WAM** | **0.9B** | **58.6** | **48.9** | **81.4** | **89.8** | **86.1** | **80.7** | **74.2** | **73.0** |

**说明**: SG-WAM 在 7 类分布偏移的整体成功率（73.0%）超越所有基线，在相机视角（58.6%）、语言变化（81.4%）两类上取得最优；Robot 迁移（48.9%）略低于 VLA-JEPA（55.7%），显示本体泛化仍有改进空间。

### Table 3: 真实机器人实验结果（含 OOD 条件）

| Model | P&P (ID) | P&P (BG) | P&P (Light) | P&P (Novel) | TF (ID) | TF (BG) | TF (Light) | TF (Novel) | Toolbox (ID) |
|-------|----------|----------|-------------|-------------|---------|---------|------------|------------|--------------|
| VLA-JEPA | 35% | 20% | 25% | 20% | 20% | 10% | 10% | 15% | 20% |
| VPP | 30% | 15% | 10% | 10% | 35% | 15% | 15% | 10% | 30% |
| **SG-WAM** | **75%** | **55%** | **60%** | **40%** | **45%** | **25%** | **35%** | **25%** | **50%** |

*P&P = Pick and Place；TF = Towel Folding；BG = Background*

**说明**: SG-WAM 在所有真实世界条件下大幅超越 VLA-JEPA 和 VPP，ID 成功率提升 2 倍以上，OOD 条件下优势更显著，体现了几何监督带来的空间感知鲁棒性。

### Table 4: 消融实验——几何监督 × 世界建模

| Geo. | WM. | Spatial | Object | Goal | Long | Avg. |
|------|------|---------|--------|------|------|------|
| ✗ | ✗ | 97.0 | 97.6 | 95.6 | 91.0 | 95.3 |
| ✓ | ✗ | 98.2 | 97.8 | 98.0 | 92.2 | 96.6 |
| ✗ | ✓ | 97.8 | 99.8 | 98.2 | 94.4 | 97.6 |
| ✓ | ✓ | **99.4** | **99.8** | **98.6** | **96.2** | **98.5** |

**关键发现**: 世界建模（+2.3pp）和几何监督（+1.3pp）各自独立提升显著，且二者互补、联合后收益叠加；去除世界建模对长时序（Long）影响最大（-5.2pp），印证了 SGWP 在时序推理中的作用。

### Table 5: 消融实验——动力学 Token 数量

| Token 数 | Spatial | Object | Goal | Long | Avg. |
|----------|---------|--------|------|------|------|
| 1 | 97.2 | 98.8 | 98.2 | 90.2 | 96.1 |
| 4 | 99.0 | 98.4 | 97.4 | 95.2 | 97.5 |
| **8** | **99.4** | **99.8** | **98.6** | **96.2** | **98.5** |
| 16 | 99.4 | 99.4 | 97.6 | 92.4 | 97.2 |

**关键发现**: 8 个动力学 token 为最优，增至 16 时性能反而下降（尤其 Long -3.8pp），表明过多 token 引入冗余并干扰表示。

### Table 6: 真实机器人子任务成功率

| Model | TF-First Fold | TF-Second Fold | Toolbox-Screwdriver | Toolbox-First Gear | Toolbox-Second Gear | Toolbox-Close |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| VLA-JEPA | 50% | 20% | 50% | 40% | 20% | 20% |
| VPP | 60% | 35% | 70% | 50% | 40% | 30% |
| **SG-WAM** | **75%** | **45%** | **80%** | **60%** | **50%** | **50%** |

**关键发现**: SG-WAM 在第二折和关闭盖子等长链末尾子任务上优势尤为突出，显示了更强的时序规划能力。

### Table 7: 动作条件消融（Null-Action Ablation）

| 变体 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| Null-Action Sequence ($\widetilde{\bm{A}}_t^{(\Delta)}=\bm{0}$) | 98.4 | 99.2 | 98.0 | 94.6 | 97.6 |
| **SG-WAM（完整）** | **99.4** | **99.8** | **98.6** | **96.2** | **98.5** |

**关键发现**: 将 SGWP 输入动作置零导致平均下降 0.9pp，确认 SGWP 确实利用了干预动作信息而非仅靠观测捷径。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 suites × 10 tasks | 桌面操作仿真，标准基准 | 仿真训练/测试 |
| LIBERO-Plus | 7 类分布偏移 | 相机/本体/语言/光照/背景/噪声/布局扰动 | 零样本迁移测试 |
| Real-World (UR5e) | 300 demos（各 100） | Pick & Place / Towel Folding / Toolbox Org. | 真实机器人训练/测试 |

### 实现细节

- **Backbone**: [[Qwen|Qwen3.5]]（0.8B），联合上下文化多模态输入
- **动作头**: [[Conditional Flow Matching|条件流匹配]]，动作序列长度 8 步
- **优化器**: AdamW；学习率 1e-5（VLM）/ 2.5e-5（仅训练模块）/ 1e-4（动作头）
- **Batch Size**: 96
- **训练步数**: 仿真 40,000 步 / 真实机器人 40 epochs
- **动力学 Token 数**: $N_q = 8$
- **几何池化**: 主视角 $37 \times 37$ 特征经 $8 \times 8$ 自适应平均池化 → 64 个 patch
- **EMA 动量**: $\mu = 0.999$
- **损失权重**: $\lambda_\text{geo} = \lambda_\text{pred} = 0.1$

### 可视化结果

- 几何监督显著改变了动力学 token 的注意力分布，使其关注操作相关的空间区域（如抓取目标位置），验证了空间归纳偏置的有效性（Figure 4 & 10）
- SG-WAM 在长时序任务（LIBERO-Long、Toolbox Organization）中相对优势最明显，说明潜在动力学预测有助于规划多步操作序列

---

## 批判性思考

### 优点

1. **参数效率高**: 0.9B 参数无具身预训练达到与 7B 有预训练模型相当的仿真性能，大幅降低部署门槛。
2. **推理零开销**: 几何教师、SGWP、EMA 教师均仅在训练时使用，推理阶段只需 VLM + 动作头，无额外延迟。
3. **设计自洽**: "自引导"目标从策略自身生成，避免了辅助潜变量方法的目标-策略不匹配问题，理论逻辑清晰。
4. **OOD 鲁棒性强**: LIBERO-Plus 零样本迁移超越所有基线，证明几何感知表示确实提升了空间泛化能力。

### 局限性

1. **机器人本体泛化弱**: LIBERO-Plus Robot 迁移（48.9%）低于 VLA-JEPA（55.7%），跨本体泛化待改进。
2. **真实任务成功率有限**: Towel Folding（ID 45%）和 Novel Object 条件（25-40%）表明在柔性物体和新颖物体上仍有提升空间。
3. **多步预测未探索**: 当前仅预测一步未来（$t+\Delta$），多步滚动预测是否进一步提升长时序能力未验证。
4. **依赖 VGGT 质量**: 几何监督的效果上界受冻结 VGGT 特征质量约束，对 VGGT 未能覆盖场景的泛化性未知。

### 潜在改进方向

1. 引入多步潜在预测（multi-step rollout）进一步增强时序规划
2. 用在线几何模型替代冻结 VGGT 以实现端到端几何学习
3. 在跨本体预训练数据上微调以提升 Robot 维度的迁移性

### 可复现性评估

- [x] 训练细节完整（超参数、学习率、epoch 均详细报告）
- [ ] 代码开源（未在论文中提供代码链接）
- [ ] 预训练模型开放
- [x] 数据集可获取（LIBERO 为公开 benchmark）

---

## 关联笔记

### 基于

- [[Qwen]]: VLM 骨干 Qwen3.5 (0.8B)
- [[Conditional Flow Matching]]: 动作生成范式
- [[VGGT]]: 提供几何监督的冻结 3D 基础模型
- [[EMA|指数移动平均]]: 构建动量教师的核心机制

### 对比

- [[VLA-JEPA]]: 同为辅助潜变量 WAM，目标由独立编码器生成；LIBERO-Plus Robot 迁移略优于 SG-WAM
- [[Spatial Forcing]]: 7B 有具身预训练基线，LIBERO 性能与 SG-WAM 持平但参数量是其 7.8 倍
- [[WorldVLA]]: 显式像素预测 WAM，LIBERO 仿真（81.8%）和 LIBERO-Plus（25.0%）均显著弱于 SG-WAM
- [[Flash-WAM]]: 6B WAM 基线，LIBERO 均值 97.6%，LIBERO-Plus 50.0%

### 方法相关

- [[Learnable Dynamics Tokens]]: 核心设计，动作条件动力学表示
- [[Self-Guided World Predictor]]: 自引导预测模块，Stop-gradient + EMA
- [[Stop-Gradient]]: 防止目标崩溃的关键操作
- [[Geometric Supervision]]: 几何感知表示学习

### 硬件/数据相关

- [[LIBERO]]: 主要仿真基准
- [[UR5e]]: 真实机器人平台

---

## 速查卡片

> [!summary] SG-WAM (2026)
> - **核心**: 在策略自身的几何感知表示空间内做自引导潜在动力学预测，消除目标-策略不匹配
> - **方法**: 可学习动力学 Token + EMA 自引导世界预测器 + VGGT 几何监督
> - **结果**: LIBERO 98.5%（0.9B，无具身预训练）；LIBERO-Plus 73.0%（超所有 7B 基线）
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-05*

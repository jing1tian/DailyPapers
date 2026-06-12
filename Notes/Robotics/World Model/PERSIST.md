---
title: "Beyond Pixel Histories: World Models with Persistent 3D State"
method_name: "PERSIST"
authors: [Samuel Garcin, Thomas Walker, Steven McDonagh, Tim Pearce, Hakan Bilen, Tianyu He, Kaixin Wang, Jiang Bian]
year: 2026
venue: ICML 2026
tags: [world-model, 3d-representation, video-generation, neural-rendering, interactive-simulation]
zotero_collection: N/A
image_source: online
arxiv_html: https://arxiv.org/html/2603.03482
created: 2026-06-05
---

# 论文笔记：Beyond Pixel Histories: World Models with Persistent 3D State

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft Research Cambridge |
| 日期 | March 2026（v2: June 2026） |
| 项目主页 | [francelico.github.io/persist.github.io](https://francelico.github.io/persist.github.io/) |
| 对比基线 | [[Oasis]]、[[WorldMem]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.03482) / Code: N/A |

---

## 一句话总结

> PERSIST 用持久化的潜在3D场景状态（而非像素历史窗口）作为世界模型的记忆，实现长时域、空间一致的交互式视频生成。

---

## 核心贡献

1. **持久化3D状态表示**: 提出将世界状态分解为以智能体为中心的动态[[3D体素网格|世界帧]]，替代基于像素上下文窗口的有限历史记忆，打破自回归模型"只能看几秒历史"的瓶颈。
2. **三模块解耦架构**: PERSIST 将模拟分解为独立的世界帧模型（$\mathcal{W}_\theta$）、相机模型（$\mathcal{C}_\theta$）和像素生成模型（$\mathcal{P}_\theta$），各模块可独立训练与替换。
3. **涌现的3D感知能力**: 显式3D表示带来像素模型天然不具备的能力：3D场景编辑、屏幕外动力学、单张图初始化的多样3D生成。

---

## 问题背景

### 要解决的问题

交互式世界模型（如游戏模拟）需要长时域空间一致性：同一场景重访时应保持几何一致，屏幕外区域的变化应影响重回视野时的观测结果。现有像素历史方法无法保证这一点。

### 现有方法的局限

- [[自回归视频生成|AR视频模型]]（如 Oasis）仅条件化于有限像素历史窗口（通常几秒），无法感知更长时间前的空间信息。
- [[像素历史方法]]（WorldMem 等）即使使用更长上下文，高维像素表示也带来极高计算代价，且缺乏显式几何结构。
- 已有3D方法（[[NeRF]]、[[3D Gaussian Splatting]]）对动态场景重建代价过高，无法满足实时交互需求。

### 本文的动机

世界状态天然是三维的。若模型直接在潜在3D空间中进行状态转移，相机参数就成为从3D记忆"读取"信息的查找键，从而以低维紧凑表示覆盖无限时间跨度的空间历史，摆脱固定上下文窗口的限制。

---

## 方法详解

### 模型架构

PERSIST 采用**三阶段自回归**架构，在每个时间步依次运行：

- **输入**: 动作序列 $A^t_{t-K}$ + 历史相机参数 $C^{t-1}_{t-K-1}$ + 历史像素帧 $\bar{O}^{t-1}_{t-K}$ + 历史世界帧 $\bar{W}^{t-1}_{t-K}$
- **世界帧模型** $\mathcal{W}_\theta$: [[扩散Transformer]]预测下一时刻3D世界帧 $\bar{w}_t$（48³体素潜变量）
- **相机模型** $\mathcal{C}_\theta$: 1D Transformer预测相机位姿变化（位置、俯仰角、偏航角、FOV）
- **世界投影算子** $\mathcal{R}(\boldsymbol{c}, \boldsymbol{w})$: [[深度剥离光栅化]]将3D特征投影到屏幕空间
- **像素生成模型** $\mathcal{P}_\theta$: [[扩散Transformer]]基于投影3D特征渲染像素观测 $\bar{o}_t$
- **参数量**: 3D-VAE（3D-ResNet骨干）+ 2D-VAE（ViT骨干）各独立训练

### 核心模块

#### 模块1：世界帧预测（$\mathcal{W}_\theta$）

**设计动机**: 利用[[扩散强制训练|Diffusion Forcing]]使去噪器对自曝光偏差（exposure bias）鲁棒——推理时输入的是模型自身生成的世界帧而非ground truth帧。

**具体实现**:
- 输入：过去K个世界帧（3D体素特征）+ 动作 + 相机参数 + 历史像素
- [[扩散Transformer]]主体带空间注意力（体素内）、时间注意力（帧间）、交叉注意力（与动作/像素条件）
- 附加10%随机噪声增强（flat noise augmentation）进一步提升对生成帧噪声的鲁棒性
- 推理时20步[[矩形流匹配|Rectified Flow]]去噪，可用2-4步实现3×加速

#### 模块2：相机模型（$\mathcal{C}_\theta$）

**设计动机**: 相机状态作为访问3D世界帧的"空间查找键"，需要独立预测以解耦几何变换与内容生成。

**具体实现**:
- 1D Transformer，预测相机位置 $\Delta x, \Delta y, \Delta z$，旋转角 $\Delta\text{pitch}, \Delta\text{yaw}$，视场角 $\Delta\text{FOV}$
- 输入：历史动作序列 + 历史相机参数（窗口K=16）

#### 模块3：世界到像素生成（$\mathcal{P}_\theta$）

**设计动机**: 利用显式几何投影将3D信息"注入"2D渲染，让像素模型专注于外观细节而非几何推理。

**具体实现**:
- [[深度剥离光栅化]]：GPU原生三角形光栅化，为每个像素生成按深度排序的体素特征列表（深度剥离栈）
- 投影结果与线性深度 $\boldsymbol{d}$ 一起输入像素扩散模型
- 偏置通道权重（biased channel weighting）：加大3D投影特征相对于像素历史的权重，防止模型忽略3D信息

---

## 关键公式

### 公式1：[[交互式世界模拟|世界模拟形式化定义]]

$$
\langle \mathbb{S}, \mathbb{O}, \mathbb{A}, \Omega, p \rangle
$$

**含义**: 交互式世界模拟定义为五元组，包含状态空间 $\mathbb{S}$、观测空间 $\mathbb{O}$、动作空间 $\mathbb{A}$、观测函数 $\Omega$ 和状态转移概率 $p$。

**符号说明**:
- $\mathbb{S}$: 世界状态空间
- $\mathbb{O}$: 像素观测空间
- $\mathbb{A}$: 玩家动作空间
- $\Omega: \mathbb{S} \to \mathbb{O}$: 将状态映射为观测的函数
- $p: \mathbb{S} \times \mathbb{A} \to \mathbb{S}$: 状态转移概率

### 公式2：[[扩散强制训练|世界帧自回归预测]]

$$
\bar{\boldsymbol{w}}_t \sim \mathcal{W}_\theta\!\left(\bar{\mathbf{w}}_t \mid \bar{W}^{t-1}_{t-K},\, A^t_{t-K},\, C^{t-1}_{t-K-1},\, \bar{O}^{t-1}_{t-K-1}\right)
$$

**含义**: 世界帧模型以过去K帧的世界帧、动作、相机参数和像素帧为条件，生成当前时刻世界帧 $\bar{\boldsymbol{w}}_t$。

**符号说明**:
- $\bar{\boldsymbol{w}}_t$: 时刻 $t$ 的生成世界帧（3D潜变量）
- $\bar{W}^{t-1}_{t-K}$: 过去K帧的世界帧序列
- $A^t_{t-K}$: 过去K步的动作序列
- $C^{t-1}_{t-K-1}$: 过去K步的相机参数序列
- $\bar{O}^{t-1}_{t-K-1}$: 过去K帧的像素观测序列
- $\mathcal{W}_\theta$: 世界帧扩散模型

### 公式3：[[世界投影算子|3D-to-2D几何投影]]

$$
\mathcal{R}(\boldsymbol{c}, \boldsymbol{w}) = (\tilde{\boldsymbol{w}}_{2D},\, \boldsymbol{d})
$$

**含义**: 世界投影算子 $\mathcal{R}$ 以相机参数 $\boldsymbol{c}$ 和3D世界帧 $\boldsymbol{w}$ 为输入，输出投影到屏幕空间的2D特征图 $\tilde{\boldsymbol{w}}_{2D}$ 和线性深度图 $\boldsymbol{d}$。

**符号说明**:
- $\boldsymbol{c}$: 相机状态（位置、旋转、FOV）
- $\boldsymbol{w}$: 3D世界帧（体素特征网格）
- $\tilde{\boldsymbol{w}}_{2D}$: 按深度排序的屏幕空间体素特征栈（深度剥离结果）
- $\boldsymbol{d}$: 每像素线性深度值

### 公式4：[[条件视频生成|像素帧自回归预测]]

$$
\bar{\boldsymbol{o}}_t \sim \mathcal{P}_\theta\!\left(\bar{\mathbf{o}}_t \mid {W_{2D}}^{t}_{t-K},\, A^t_{t-K},\, \bar{O}^{t-1}_{t-K}\right)
$$

**含义**: 像素生成模型以屏幕空间投影特征、动作历史和像素历史为条件，渲染当前时刻的像素观测。

**符号说明**:
- $\bar{\boldsymbol{o}}_t$: 时刻 $t$ 的生成像素帧
- ${W_{2D}}^{t}_{t-K}$: 过去K帧通过 $\mathcal{R}$ 投影得到的2D特征序列
- $\bar{O}^{t-1}_{t-K}$: 过去K帧的历史像素观测
- $\mathcal{P}_\theta$: 像素扩散模型

### 公式5：[[矩形流匹配|Rectified Flow 去噪目标]]

$$
\mathcal{L}_\theta = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| v_\theta(x_\tau, \tau, \text{cond}) - (x_1 - x_0) \right\|^2 \right]
$$

**含义**: Rectified Flow 训练目标，令模型预测速度场（从噪声到数据的直线方向），$x_\tau = (1-\tau)x_0 + \tau x_1$ 为噪声插值点。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 均匀采样的噪声强度（时间步）
- $x_0 \sim \mathcal{N}(0,I)$: 高斯噪声
- $x_1$: 真实数据（世界帧或像素帧）
- $v_\theta$: 模型预测的速度向量场
- $\text{cond}$: 条件输入（动作、相机、历史帧等）

### 公式6：[[扩散强制训练|曝光偏差缓解—随机噪声增强]]

$$
\tilde{w}^{(i)} = \sqrt{1 - \sigma^2} \cdot w^{(i)} + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**含义**: 在训练时，以10%的概率对历史世界帧 $w^{(i)}$ 加入随机高斯噪声，模拟推理时使用模型自身生成帧（而非ground truth帧）带来的分布偏移。

**符号说明**:
- $w^{(i)}$: 历史世界帧（ground truth）
- $\sigma$: 噪声强度（从推理时典型噪声水平采样）
- $\tilde{w}^{(i)}$: 加噪后的历史帧（训练时作为条件输入）

---

## 关键图表

### Figure 1: PERSIST 整体流水线

![Figure 1: PERSIST Pipeline](https://arxiv.org/html/2603.03482v2/figs/Pipeline.png)

**说明**: PERSIST 的自回归循环。以单帧像素初始化，每步依次：（1）世界帧模型 $\mathcal{W}_\theta$ 去噪3D世界帧；（2）相机模型 $\mathcal{C}_\theta$ 预测相机参数；（3）世界投影算子 $\mathcal{R}$ 将3D特征投影到屏幕空间；（4）像素模型 $\mathcal{P}_\theta$ 渲染最终像素帧。

### Figure 2: 持久化3D空间记忆

![Figure 2: Persistent 3D Spatial Memory](https://arxiv.org/html/2603.03482v2/figs/rotate_rgb.png)

**说明**: PERSIST 以智能体为中心的3D世界帧作为动态空间记忆，相机参数充当查找键，通过几何投影从世界帧中提取当前视角信息。即使智能体转身或离开区域，3D状态依然保留环境结构信息。

### Figure 3: 初始化方式对比（RGB vs RGB+3D）

![Figure 3: Initialization Variants](https://arxiv.org/html/2603.03482v2/figs/qual.jpg)

**说明**: PERSIST 支持两种初始化方式——仅单张RGB帧，或RGB帧+3D世界帧。可视化600步自回归滚动生成的世界帧和像素视频。即使仅用单张RGB初始化，PERSIST 也能生成连贯的演化3D世界。

### Figure 4: 世界投影——深度剥离光栅化

![Figure 4: World Projection Renderer](https://arxiv.org/html/2603.03482v2/figs/renderer.png)

**说明**: [[深度剥离光栅化]]工作流程。体素特征被投影到屏幕空间，生成每像素按深度排序的特征栈和线性深度图 $\boldsymbol{d}$，供像素扩散模型使用。

### Figure 5: 与基线的定性对比（600步）

![Figure 5: Qualitative Comparison over 600 Timesteps](https://arxiv.org/html/2603.03482v2/figs/compare_87.jpg)

**说明**: PERSIST、Oasis、WorldMem 在600步生成视频帧的对比。PERSIST 在长时域内保持一致的环境结构，而基线方法随时间推移出现明显的几何不一致和场景退化。

### Figure 6: FID随时间变化曲线

![Figure 6: FID over Time](https://arxiv.org/html/2603.03482v2/figs/fid_vs_t_wmshift.png)

**说明**: 600帧内 FID 分数随时间步变化。PERSIST 各配置保持稳定，基线（仅像素历史）快速退化，证明持久化3D状态对长时域生成质量的关键作用。

### Figure 7: 3D感知的场景编辑

![Figure 7: 3D Scene Edits](https://arxiv.org/html/2603.03482v2/figs/edits_single.jpg)

**说明**: PERSIST 的显式3D表示允许在片段中途直接编辑世界状态，支持全局地形/生物群系编辑和小型资产（如树木）放置，实现精细的几何感知控制。

### Figure 8: 屏幕外动力学

![Figure 8: Off-Screen Dynamics](https://arxiv.org/html/2603.03482v2/figs/dynamic.jpg)

**说明**: 展示三种屏幕外动力学能力：（上）玩家后退碰到视野外的树木；（中）洞穴在视野外自主被水填满；（下）视野外的水流动并涌入屏幕产生涌现效果。这些在纯像素方法中不可能实现。

### Figure 9: 用户研究界面

![Figure 9: User Study Interface](https://arxiv.org/html/2603.03482v2/figs/UI.png)

**说明**: 基于网页的用户研究平台截图，参与者对视频对的时间一致性和空间一致性进行评分。共28名参与者，800+次评估，采用5点Likert量表。

### Figure 11: 相机不一致性示例

![Figure 11: Camera Inconsistency](https://arxiv.org/html/2603.03482v2/figs/camera_clip.jpg)

**说明**: 展示PERSIST当前局限性——长时间生成中出现的相机轨迹不一致情况，系曝光偏差累积所致。

### Figure 12: 单图多样3D世界生成

![Figure 12: Diverse 3D Generation from Single Image](https://arxiv.org/html/2603.03482v2/figs/w0_generations.png)

**说明**: 以单张RGB帧为条件，PERSIST 可生成多个多样且连贯的3D世界（即 $w_0$ 初始化）。展示了模型对场景外观的多样想象能力。

### Table 1: FVD 分数对比（不同时间跨度）

| 方法 | 200 帧 | 400 帧 | 600 帧 |
|------|--------|--------|--------|
| Oasis | 409 | 687 | 875 |
| WorldMem | 358 | — | — |
| PERSIST-S（消融） | 159 | 170 | 179 |
| PERSIST | 129 | 141 | 148 |
| PERSIST+w₀ | 80 | 93 | 104 |

**关键发现**: PERSIST 在200/400/600帧各跨度均大幅领先基线，且 FVD 随时间步增加几乎不退化（仅从129到148），而Oasis从409退化到875。提供3D初始化（PERSIST+w₀）进一步将FVD降低到80。

### Table 2: 用户主观评分（1-5分制，均值±标准差）

| 方法 | 视觉保真度 | 3D一致性 | 时间一致性 | 综合 |
|------|-----------|---------|----------|------|
| Oasis | 2.1±0.1 | 1.9±0.1 | 1.8±0.1 | 1.9±0.1 |
| WorldMem | 1.7±0.09 | 1.7±0.09 | 1.5±0.08 | 1.5±0.07 |
| PERSIST-S | 2.8±0.1 | 2.7±0.1 | 2.5±0.1 | 2.6±0.09 |
| PERSIST | 2.8±0.09 | 2.5±0.09 | 2.5±0.09 | 2.6±0.08 |
| PERSIST+w₀ | **3.2±0.1** | **2.8±0.1** | **2.8±0.1** | **3.0±0.1** |

**关键发现**: PERSIST 在所有维度上均显著优于基线。PERSIST+w₀ 综合评分达到3.0，是唯一综合评分超过3.0的方法。WorldMem 表现甚至不如 Oasis，验证了纯像素历史方法的局限。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Luanti（自采集） | ~4000万次环境交互，~10万条轨迹，460小时，24Hz | 开源体素游戏引擎，支持程序化世界生成和48³体素GT | 训练（所有模型）|
| 168条保留世界轨迹 | 168条评估轨迹 | 来自未见过的世界，覆盖多种地形 | 定量评估（FVD/FID/人工评分）|

### 实现细节

- **世界帧模型骨干**: [[扩散Transformer]]，含空间/时间/交叉注意力
- **像素模型骨干**: [[扩散Transformer]]，基于屏幕空间投影特征条件化
- **3D-VAE**: 3D-ResNet，MSE+KL损失，12天在8块A100 GPU上训练
- **2D-VAE**: ViT骨干，MSE+KL损失，4天在8块A100 GPU上训练
- **世界帧去噪器（3D-S）**: 3天在8块H100 GPU上训练
- **像素去噪器**: 10天在8块H100 GPU上训练
- **优化器**: AdamW，学习率 1e-4
- **上下文窗口**: K∈{8-16}（各组件不同）
- **推理**: 每帧20步去噪；2-4步可实现3×加速
- **3D分辨率**: 48³体素潜变量网格

### 评估策略

- **FVD（Fréchet Video Distance）**: 主要量化指标，与GT比较
- **FID（per-frame）**: 逐帧质量评估
- **评估轨迹策略**: Free Play、向前移动、向后移动、环绕旋转

---

## 批判性思考

### 优点

1. **真正的无限时域记忆**: 3D状态不受固定上下文窗口限制，理论上可存储无限历史的空间信息，这是像素历史方法的根本性突破。
2. **涌现的几何感知能力**: 显式3D表示自然带来屏幕外动力学、3D编辑等能力，无需为这些功能专门设计模块。
3. **模块化可替换架构**: 三个独立模块（世界帧/相机/像素）允许分别改进，降低研究迭代成本。

### 局限性

1. **需要3D监督**: 当前方法依赖游戏引擎提供GT体素观测，直接用于真实世界视频需要2D-to-3D基础模型将RGB lift到3D，存在误差累积风险。
2. **曝光偏差仍存在**: Diffusion Forcing + 噪声增强只是缓解，未完全解决；作者建议通过生成轨迹上的端到端后训练来根本解决。
3. **有限空间记忆范围**: 48³体素网格以智能体为中心，不能覆盖无界开放世界；建议用3D记忆库扩展。
4. **推理速度未达实时**: 每帧需要多次去噪前向传播，目前不满足实时交互需求。

### 潜在改进方向

1. 引入2D-to-3D基础模型，摆脱对GT体素数据的依赖，扩展到真实世界视频
2. 端到端在生成轨迹上后训练，从根本上解决曝光偏差问题
3. 基于[[稀疏体素|稀疏体素]]或[[3D记忆库]]设计无界空间记忆
4. 引入流式推理（如一致性模型）实现近实时生成

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（附录A给出完整超参数）
- [ ] 数据集可获取（Luanti游戏数据需自行采集）

---

## 关联笔记

### 基于

- [[Oasis]]: 基于像素历史的AR世界模型基线
- [[WorldMem]]: 带外部像素记忆的AR世界模型基线
- [[矩形流匹配|Rectified Flow]]: 扩散去噪的训练框架
- [[扩散强制训练|Diffusion Forcing]]: 训练自回归扩散模型的框架，PERSIST用于缓解曝光偏差

### 对比

- [[Oasis]]: 像素历史AR世界模型，FVD在长时域快速退化
- [[WorldMem]]: 带外部像素记忆的方法，人工评分甚至不如Oasis
- [[NeRF]]: 静态3D重建，计算代价过高不适用于交互仿真
- [[3D Gaussian Splatting]]: 动态3D表示，但仍面临重建成本问题

### 方法相关

- [[扩散Transformer]]: 世界帧模型和像素模型的骨干架构
- [[深度剥离光栅化]]: 3D-to-2D的核心投影方法
- [[3D体素网格]]: 世界帧的底层3D表示
- [[扩散强制训练|Diffusion Forcing]]: 曝光偏差缓解策略
- [[矩形流匹配|Rectified Flow Matching]]: 扩散去噪框架

### 硬件/数据相关

- [[Luanti]]: 开源体素游戏引擎，训练数据来源
- [[A100]]: 训练硬件（3D-VAE、2D-VAE）
- [[H100]]: 训练硬件（3D-S去噪器、像素去噪器）

---

## 速查卡片

> [!summary] Beyond Pixel Histories: World Models with Persistent 3D State (PERSIST)
> - **核心**: 用持久化潜在3D场景（而非像素历史窗口）作为世界模型记忆
> - **方法**: 三模块解耦：世界帧扩散模型 + 相机预测Transformer + 深度剥离光栅化 + 像素扩散模型
> - **结果**: FVD(600帧) 148 vs Oasis 875；人工评分综合3.0 vs 1.9（Oasis）
> - **代码**: 未开源（项目主页: francelico.github.io/persist.github.io）

---

*笔记创建时间: 2026-06-05*

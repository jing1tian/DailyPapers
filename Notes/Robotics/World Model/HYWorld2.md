---
title: "HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds"
method_name: "HYWorld2"
authors: [Team HY-World, Tengfei Wang]
year: 2026
venue: arXiv
tags: [3d-world-generation, world-model, 3d-gaussian-splatting, diffusion-model, multi-view-synthesis, scene-reconstruction, panorama-generation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2604.14268v1
created: 2026-05-19
---

# 论文笔记：HY-World 2.0

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent Hunyuan |
| 日期 | April 2026 |
| 项目主页 | [HY-World](https://3d-models.hunyuan.tencent.com/world/) |
| 对比基线 | Marble（闭源） |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14268) / [Code](https://github.com/Tencent-Hunyuan/HY-World-2.0) |

---

## 一句话总结

> HY-World 2.0 是首个开源的多模态世界模型，通过四阶段流水线（全景生成 → 轨迹规划 → 世界扩展 → 场景合成）将文本/图像/视频统一转化为可交互的 [[3D Gaussian Splatting|3DGS]] 场景。

---

## 核心贡献

1. **统一生成与重建**: 首个开源框架同时支持世界生成（从文字/单图）和世界重建（从多视图/视频），实现 Text/Image → 可导航 3D 场景的端到端流水线
2. **HY-World 2.0 四大升级模块**: HY-Pano 2.0、WorldNav、WorldStereo 2.0、WorldMirror 2.0 全面升级，性能可比闭源模型 Marble
3. **WorldLens 渲染平台**: 高性能 3DGS 渲染引擎，支持碰撞检测、IBL 光照和角色动画，实现实时交互探索

---

## 问题背景

### 要解决的问题

如何从多模态输入（文本、单张图像、多视图、视频）生成可导航、可交互的沉浸式 3D 场景，同时兼容标准图形渲染管线？

### 现有方法的局限

- **闭源方案**（如 Marble）：性能强但不可复现，社区无法在其基础上继续研究
- **生成与重建割裂**：现有开源方法要么只做生成，要么只做重建，无法统一
- **视频 VAE 不适用于多视图**：时序压缩导致高频细节丢失，大视角变化下一致性差
- **固定分辨率训练**：限制了在不同分辨率下的推理能力

### 本文的动机

构建首个开源、系统性的多模态世界模型，通过模块化四阶段流水线解耦各子问题（全景初始化、轨迹规划、多视图扩展、3DGS 合成），使每个模块可独立优化。

---

## 方法详解

### 整体架构

HY-World 2.0 采用**四阶段流水线**架构：

![Figure 2: 整体架构](https://arxiv.org/html/2604.14268v1/x2.png)

**说明**: HY-World 2.0 整体架构。多模态输入经过 (1) 全景生成初始化世界，(2) 轨迹规划生成探索路径，(3) 记忆驱动的世界扩展生成多视图关键帧，(4) 世界合成构建最终 [[3D Gaussian Splatting|3DGS]] 资产。

- **输入**: 文本提示 / 单视图图像 / 多视图图像 / 视频
- **Stage I**: [[HY-Pano 2.0]] — 全景生成
- **Stage II**: [[WorldNav]] — 轨迹规划
- **Stage III**: [[WorldStereo 2.0]] — 多视图世界扩展
- **Stage IV**: [[WorldMirror 2.0]] + [[3D Gaussian Splatting|3DGS]] 优化 — 场景合成
- **输出**: 可导航的 3DGS 场景

---

### Stage I: 全景生成（HY-Pano 2.0）

![Figure 3: HY-Pano 2.0 架构](https://arxiv.org/html/2604.14268v1/x3.png)

**说明**: HY-Pano 2.0 全景生成架构。左侧为整体流水线，右侧展示环形填充（Latent Space）和像素混合（Pixel Space）实现无缝全景输出。

**核心设计**:
- 基于 [[Multi-Modal Diffusion Transformer|MMDiT]] 实现自适应透视图到等矩形（Perspective-to-Equirectangular）映射
- **隐式映射策略**：绕过显式相机元数据，直接从任意视点输入图像映射
- **环形填充（Circular Padding）**：在 Latent Space 中实现 360° 无缝拼接
- **像素混合（Pixel Blending）**：Pixel Space 中消除拼接边界伪影
- **训练数据**：真实全景数据 + Unreal Engine 合成资产混合

---

### Stage II: 轨迹规划（WorldNav）

![Figure 4: 初始场景解析](https://arxiv.org/html/2604.14268v1/x4.png)

**说明**: 轨迹规划的初始场景解析流程，从全景图获取点云、Mesh、语义 Mask 和 [[NavMesh]]。

![Figure 5: 五种轨迹模式](https://arxiv.org/html/2604.14268v1/x5.png)

**说明**: WorldNav 五种轨迹模式示意图（部分轨迹已简化可视化）。

**五种轨迹模式**:

| 模式 | 最大数量 | 附着物体 | 迭代 | 描述 |
|------|----------|----------|------|------|
| Regular（常规） | 9 | ✗ | ✗ | ±120° 方位角、+45°/−60° 仰角轨道运动 |
| Surrounding（环绕） | 5 | ✓ | ✗ | 自适应环绕显著物体 |
| Recon-Aware（重建感知） | 10 | ✓ | ✓ | 迭代覆盖欠观测区域 |
| Wandering（漫游） | 3 | ✗ | ✗ | 8 个角度扇区最远可导航点 |
| Aerial（航拍） | 8 | — | ✗ | +45° 仰角增强 |

**总计最多 35 条轨迹**

**几何处理**:
- 从等矩形投影（ERP）提取 42 个透视视图（标准做法为 12 个）
- GPU 加速 LSMR 求解器完成深度对齐
- NavMesh 通过射线投射、边界腐蚀、桥接合成精化

---

### Stage III: 世界扩展（WorldStereo 2.0）

![Figure 6: WorldStereo 2.0 三阶段训练](https://arxiv.org/html/2604.14268v1/x6.png)

**说明**: WorldStereo 2.0 三阶段训练流程，逐步启用相机控制、记忆一致性和快速推理。

![Figure 7: WorldStereo 2.0 整体流水线](https://arxiv.org/html/2604.14268v1/x7.png)

**说明**: WorldStereo 2.0 整体流水线。(a) 主 [[Video Diffusion Transformer|Video DiT]] 分支通过 SSM++ 提供细粒度一致性；(b) 相机控制分支由全景点云引导，作为 GGM 确保精确轨迹跟随。

#### 关键设计：Keyframe-VAE

![Figure 8: VAE 对比](https://arxiv.org/html/2604.14268v1/x8.png)

**说明**: 不同 VAE 变体的重建与新视图生成对比，Keyframe-VAE 在大视角变化下保持外观一致性。

![Figure 9: Keyframe-VAE 架构](https://arxiv.org/html/2604.14268v1/x9.png)

**说明**: Keyframe-VAE 与标准 Video-VAE 对比。Video-VAE 进行时空压缩，Keyframe-VAE 仅做空间压缩，更好保留高频细节、减少伪影。

**与 Video-VAE 的对比**:

| 方案 | 受感野 | 轨迹长度/数量 | 潜在空间 | 帧质量 | 精确相机控制 | 一致性 | 效率 |
|------|--------|--------------|----------|--------|------------|--------|------|
| Native VDM | 双向 | 长/单条 | 视频片段 | ✓ | — | — | ✓ |
| Autoregressive | 自回归 | 长/单条 | 视频片段 | — | — | — | — |
| WorldStereo 2.0 | 双向 | 中等/多条 | 关键帧图像 | ✓ | ✓ | ✓ | ✓ |

#### 记忆机制

![Figure 10: 记忆训练数据构建](https://arxiv.org/html/2604.14268v1/x10.png)

**说明**: WorldStereo 2.0 记忆训练数据构建。(a) GGM 训练的全局点云（1 参考视图 + Tg=2 目标视图）；(b) SSM++ 训练的轨迹检索策略：多视图数据的时序错位检索（上）与合成数据的多轨迹检索（下）。

**Global-Geometric Memory（GGM）**:
- 从 Tg=2 目标视图构建扩展点云作为全局 3D 先验
- 渲染扩展点云维持 360° 结构一致性
- 数据增强：50% 双线性下采样深度、10% 高斯模糊（模拟浮点伪影）、50% 保留原始单目深度

**Spatial-Stereo Memory++（SSM++）**:
- 选择性检索相关关键帧（替代显式点云引导）
- 修改 [[Rotary Position Encoding|RoPE]] 支持水平拼接的时序索引

![Figure 11: SSM++ 中的 RoPE 修改](https://arxiv.org/html/2604.14268v1/x11.png)

**说明**: SSM++ 中的 RoPE 修改示意图。目标帧与对应检索参考视图沿水平轴拼接（宽度变为 2W），每个检索视图继承配对目标帧的时间索引后再输入主 DiT。

#### 三阶段训练

1. **Domain-Adaptation**: 使用参考点云的相机控制
2. **Middle-Training**: 记忆机制精化
3. **Post-Training**: [[Distribution Matching Distillation|DMD]] 蒸馏为 4 步学生模型

---

### Stage IV: 世界合成（WorldMirror 2.0 + 3DGS）

#### WorldMirror 2.0 架构

![Figure 12: WorldMirror 2.0 架构](https://arxiv.org/html/2604.14268v1/x12.png)

**说明**: WorldMirror 2.0 统一前向重建模型架构，共享 Transformer 骨干网络，通过任务专用 DPT 解码头同时预测密集点云、深度图、法线图、相机参数和 3DGS 属性。

**WorldMirror 1.0 vs 2.0 对比**:

| 组件 | WorldMirror 1.0 | WorldMirror 2.0 |
|------|-----------------|-----------------|
| **模型改进** | | |
| 位置编码 | 绝对 RoPE | 归一化 RoPE |
| 深度监督 | 仅 GT 深度 | GT 深度 + GT/伪法线 |
| 无效像素建模 | 仅置信度 | 置信度 + 深度 Mask 头 |
| 加速 | 无 | Token/Frame SP + BF16 + FSDP |
| **数据改进** | | |
| 训练数据 | 开源数据集 | + 内部 UE 渲染 |
| 伪标签 | ✗ | 法线伪标签 |
| **训练策略** | | |
| 分辨率采样 | 独立采样 | Token 预算动态采样 |
| 课程阶段 | 2 阶段 | 3 阶段 |
| 分辨率范围 | 100K–250K 像素 | 50K–500K 像素 |
| **新增能力** | | |
| 灵活分辨率推理 | — | ✓ |
| 深度几何一致性 | — | ✓ |
| 鲁棒无效像素处理 | — | ✓ |
| 推理效率 | — | ✓ |

---

## 关键公式

### 公式1: [[Rotary Position Encoding|归一化位置编码]]

$$
\hat{x}_i = \frac{2i + 1}{H_p} - 1, \quad \hat{y}_j = \frac{2j + 1}{W_p} - 1
$$

**含义**: 将 Patch 索引归一化到 $[-1, 1]$ 范围，将"分辨率外推"转化为"插值"，确保不同分辨率间 RoPE 编码高度一致（跨分辨率余弦相似度 >0.95）。

**符号说明**:
- $\hat{x}_i, \hat{y}_j$: 归一化后的空间坐标，取值范围 $[-1, 1]$
- $H_p, W_p$: Patch 网格的高和宽（分辨率无关的归一化分母）
- $i, j$: Patch 的行列索引

![Figure 13: 归一化位置编码分析](https://arxiv.org/html/2604.14268v1/x13.png)

**说明**: 归一化 RoPE 与标准 RoPE 的跨分辨率一致性对比。(a) 归一化 RoPE 保持 >0.95 余弦相似度，标准 RoPE 显著退化；(b)(c) 展示不同分辨率下编码值的均值和标准差。

---

### 公式2: [[Point Cloud Warping|点云投影]]

$$
\mathbf{P}_i^{tar}(x) \simeq \mathbf{R}_i^{c \to w} D(x) \mathbf{K}_i^{-1} \hat{x}
$$

**含义**: 将参考视图中的像素反投影到世界坐标，构建用于 GGM 的全局点云。

**符号说明**:
- $\mathbf{R}_i^{c \to w}$: 第 $i$ 帧的相机到世界旋转矩阵
- $D(x)$: 单目深度估计值
- $\mathbf{K}_i$: 相机内参矩阵
- $\hat{x}$: 齐次像素坐标

---

### 公式3: [[Point Cloud Aggregation|全局点云扩展]]

$$
\mathbf{P}^{glo} = [\mathbf{P}^{ref}, \hat{\mathbf{P}}] \in \mathbb{R}^{(N + \hat{N}) \times 3}
$$

**含义**: 将参考视图点云 $\mathbf{P}^{ref}$ 与额外新视图点云 $\hat{\mathbf{P}}$ 拼接，构建用于 GGM 的扩展全局点云。

**符号说明**:
- $\mathbf{P}^{ref}$: 参考视图的点云（$N$ 个点）
- $\hat{\mathbf{P}}$: 额外新视图生成的点云（$\hat{N}$ 个点）
- $\mathbf{P}^{glo}$: 最终全局点云

---

### 公式4: [[Distribution Matching Distillation|DMD 蒸馏损失]]

$$
\nabla \mathcal{L}_{DMD} = -\mathbb{E}_t \left( \int (s_{real}(x_t, t) - s_{fake}(x_t, t)) \frac{dx_t}{d\theta} \, dz \right)
$$

**含义**: 通过最小化真实分布和生成分布之间的 Score 差异，将多步扩散模型蒸馏为 4 步快速推理的学生模型。

**符号说明**:
- $s_{real}(x_t, t)$: 冻结的教师模型 Score 函数
- $s_{fake}(x_t, t)$: 可训练的学生模型 Score 函数
- $\theta$: 学生模型参数

---

### 公式5: [[Depth-to-Normal Conversion|深度转法线]]

$$
\tilde{N}_i(x) = \text{normalize}\left(\frac{\partial \mathbf{P}_i}{\partial u} \times \frac{\partial \mathbf{P}_i}{\partial v}\right), \quad \mathbf{P}_i = \mathbf{K}^{-1} \hat{D}_i \cdot [u, v, 1]^\top
$$

**含义**: 通过对深度图反投影到 3D 点后计算法向量（叉积），从预测深度派生出表面法线，用于几何一致性监督。

**符号说明**:
- $\tilde{N}_i(x)$: 由深度派生的法线
- $\mathbf{P}_i$: 通过内参矩阵 $\mathbf{K}^{-1}$ 反投影得到的 3D 点
- $\hat{D}_i$: 预测的深度图
- $u, v$: 像素坐标

---

### 公式6: [[Depth-to-Normal Loss|深度-法线一致性损失]]

$$
\mathcal{L}_{d2n} = \frac{1}{|\mathcal{V}|} \sum_{x \in \mathcal{V}} \arccos\left(\frac{\tilde{N}_i(x) \cdot \hat{N}_i(x)}{\|\tilde{N}_i(x)\| \|\hat{N}_i(x)\|}\right)
$$

**含义**: 以角度误差监督派生法线与目标法线之间的一致性，显式耦合深度和法线的几何约束。

**符号说明**:
- $\mathcal{V}$: 有效像素集合
- $\tilde{N}_i(x)$: 从深度图派生的法线
- $\hat{N}_i(x)$: GT 法线（合成数据）或伪法线（真实数据）

---

## 关键图表

### Figure 1: Teaser — 多样化应用

![Figure 1: Teaser](https://arxiv.org/html/2604.14268v1/pics/teaser0.jpeg)

**说明**: HY-World 2.0 的多样化应用展示，统一世界生成（从文本/单图合成可探索 3D 世界）和世界重建（从多视图恢复 3D 表示）。

---

### Table 1: WorldNav 轨迹详情

| 模式 | Regular | Surrounding | Recon-Aware | Wandering | Aerial |
|------|---------|-------------|-------------|-----------|--------|
| 最大数量 | 9 | 5 | 10 | 3 | 8 |
| 附着物体 | ✗ | ✓ | ✓ | ✗ | — |
| 迭代执行 | ✗ | ✗ | ✓ | ✗ | ✗ |

**关键发现**: 五种轨迹覆盖从常规俯仰到对象环绕、从漫游到航拍的全方位场景采样，最多生成 35 条轨迹确保覆盖完整。

---

### Table 2: 视频生成方案对比

| 方案 | 受感野 | 轨迹长度/数量 | 潜在空间 | 帧质量 | 无冗余 | 精确相机控制 | 一致性 | 效率 |
|------|--------|--------------|----------|--------|--------|------------|--------|------|
| Native VDM | 双向 | 长/单 | 视频片段 | ✓ | — | — | — | ✓ |
| Autoregressive | 自回归 | 长/单 | 视频片段 | — | — | — | — | — |
| **WorldStereo 2.0** | **双向** | **中/多** | **关键帧** | **✓** | **✓** | **✓** | **✓** | **✓** |

**关键发现**: WorldStereo 2.0 是唯一在帧质量、无冗余、精确相机控制、一致性和效率五项全部满足的方案。

---

## 实验

### 实现细节

- **WorldMirror 2.0 训练数据**: 开源多视图数据集 + 内部 Unreal Engine 渲染数据
- **训练阶段**: 3 阶段课程学习（vs v1.0 的 2 阶段）
- **分辨率采样**: Token 预算动态采样，50K–500K 像素（vs v1.0 的固定 100K–250K）
- **精度与并行**: BF16 + Token/Frame 分片并行（SP）+ 完全分片数据并行（FSDP）
- **WorldStereo 2.0 蒸馏**: DMD 蒸馏为 4 步学生模型

### 定性结果

- 整体性能可比闭源模型 Marble，是首个达到该水平的开源系统
- 支持多种输入模态：Text-to-3D、Image-to-3D、Video-to-3D、Multi-View-to-3D

---

## 批判性思考

### 优点

1. **开源完整性**: 代码、权重、技术细节全部开源，真正推动社区研究
2. **模块化设计**: 四阶段流水线各模块可独立升级，扩展性强
3. **Keyframe-VAE 创新**: 空间压缩替代时空压缩，解决大视角变化下的一致性问题
4. **多模态统一**: 单一框架同时处理 Text/单图/多视图/视频，Input 灵活性高

### 局限性

1. **量化结果不足**: 论文提供的定量比较表格不完整，主要以闭源 Marble 为参照但未提供具体数值
2. **流水线复杂度**: 四阶段流水线涉及多个模型，端到端推理速度和资源消耗较高
3. **室内/简单场景偏重**: 全景点云 + NavMesh 的方案在大规模室外场景或无限场景中的泛化性待验证

### 潜在改进方向

1. 将生成流水线压缩为端到端模型（减少级联误差累积）
2. 引入 Language-Conditioned 场景编辑能力（在已生成 3D 世界中进行语义修改）
3. 结合物理仿真器，使 3DGS 场景具备物理交互能力

### 可复现性评估

- [x] 代码开源（GitHub: Tencent-Hunyuan/HY-World-2.0）
- [x] 预训练模型（已发布权重）
- [x] 训练细节完整（三阶段课程、Token 预算采样等均有描述）
- [x] 数据集可获取（开源数据集 + 内部 UE 渲染数据）

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: 最终场景表示格式，支持实时渲染
- [[Video Diffusion Transformer]]: WorldStereo 2.0 的骨干网络
- [[Rotary Position Encoding]]: 归一化 RoPE 的基础
- [[Distribution Matching Distillation]]: WorldStereo 2.0 蒸馏训练策略

### 对比

- [[Marble]]: 闭源竞争对手，HY-World 2.0 的性能对标目标

### 方法相关

- [[NavMesh]]: 轨迹规划中用于碰撞检测的导航网格
- [[Multi-Modal Diffusion Transformer]]: HY-Pano 2.0 使用的架构
- [[Depth-to-Normal Loss]]: WorldMirror 2.0 的几何一致性损失

### 相关论文

- [[World Model]]: 宏观概念背景

---

## 速查卡片

> [!summary] HY-World 2.0
> - **核心**: 首个开源多模态 3D 世界生成+重建统一框架
> - **方法**: 四阶段流水线（HY-Pano 2.0 → WorldNav → WorldStereo 2.0 → WorldMirror 2.0 + 3DGS）
> - **结果**: 性能可比闭源 Marble，首个达到该水平的开源系统
> - **代码**: [GitHub](https://github.com/Tencent-Hunyuan/HY-World-2.0)

---

*笔记创建时间: 2026-05-19*

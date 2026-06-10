---
title: "Latent Spatial Memory for Video World Models"
method_name: "Mirage"
authors: [Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, Bohan Zhuang]
year: 2026
venue: arXiv
tags: [video-world-model, latent-spatial-memory, novel-view-synthesis, diffusion-model, 3d-consistency, camera-control]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.09828
created: 2026-06-10
---

# 论文笔记：Latent Spatial Memory for Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Microsoft |
| 日期 | June 2026 |
| 项目主页 | [aka.ms/latent-spatial-memory](https://aka.ms/latent-spatial-memory) |
| 对比基线 | [[Spatia]], [[Voyager]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09828) / [Code](https://github.com/microsoft/LatentSpatialMemory) |

---

## 一句话总结

> Mirage 将三维场景信息直接缓存在 [[扩散模型|Diffusion Model]] 的 [[VAE|潜空间]] 中，消除 RGB 点云方法反复"光栅化→编码"的瓶颈，实现 10.57× 加速与 55× 内存压缩。

---

## 核心贡献

1. **潜空间空间记忆（Latent Spatial Memory）**: 提出将三维场景表示为 `{(p_i, f_i)}` 的结构，其中 `f_i` 是 [[VAE]] 潜特征而非 RGB 颜色，彻底规避"渲染→重编码"循环
2. **深度引导反投影 + 直接潜空间变换（Direct Latent-Space Warping）**: 在潜分辨率下完成投影与可见性判断（[[z-Buffering]]），无需上采样至像素空间
3. **动态目标过滤机制**: 用 [[Qwen3-VL]] + [[SAM3]] 检测并剔除天空与运动物体，防止瞬态内容污染持久记忆，3D 一致性从 80.88 提升至 92.21

---

## 问题背景

### 要解决的问题

给定单张输入图像和用户指定的相机轨迹，生成**几何一致的长程视频**——摄像机移动时场景结构保持稳定、无穿帮或漂移。

### 现有方法的局限

以 Voyager、Spatia 为代表的 RGB 点云方法维护一个显式彩色点云 $\mathcal{M}_\text{rgb} = \{(\mathbf{p}_i, \mathbf{c}_i)\}$，每个条件步都需要：
1. 将点云**光栅化**到目标视角（GPU 密集型操作）
2. 将 RGB 渲染图**重新编码**为 [[VAE]] 潜变量

这两步引入巨大的计算与内存开销，随视频长度线性增长。

### 本文的动机

[[VAE]] 编码器将 RGB 图像映射至紧凑潜空间后，该潜特征已包含丰富的语义与结构信息。若直接在**潜空间**中存储和投影场景特征，可跳过解码→渲染→重编码的往返，且投影在低分辨率（$w \times h = W/s \times H/s$，$s=16$）上进行，计算量大幅减少。

---

## 方法详解

### 模型架构

Mirage 采用**条件视频扩散**架构：

- **输入**: 初始帧 $I_0$ + 相机轨迹 $\{\mathbf{E}^t, K^t\}$
- **Backbone**: [[Wan2.2-TI2V-5B]]（5B 参数，VAE stride $s=16$，$C=48$ 通道）
- **核心模块**: [[Latent Spatial Memory]] 缓存 + [[ControlNet]] 风格条件注入分支
- **输出**: 几何一致的视频帧序列
- **可训练参数**: 条件侧分支 + Rank-64 [[LoRA]] 适配器（仅占主干极小比例）

三阶段循环（**初始化 → 读出 → 更新**）：

- **初始化**: 对 $I_0$ 编码为 [[VAE]] 潜变量，通过**深度引导反投影**构建初始 3D 潜缓存 $\mathcal{M}$
- **读出（Readout）**: 将缓存特征投影到目标相机网格，生成条件潜变量 $\hat{\mathbf{z}}^t$
- **更新（Update）**: 对生成帧重编码，将新区域的潜特征反投影追加到 $\mathcal{M}$

### 核心模块

#### 模块 1: 潜空间空间记忆（Latent Spatial Memory）

**设计动机**: 利用 [[VAE]] 潜空间的紧凑性，将场景三维信息直接存入特征空间，避免 RGB 点云的"光栅化→编码"往返。

**具体实现**:
- 记忆集合 $\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}$，世界坐标 $\mathbf{p}_i \in \mathbb{R}^3$，潜特征 $\mathbf{f}_i \in \mathbb{R}^{C}$（$C=48$）
- 通过[[深度估计|Depth Anything 3]]获取单目深度图，[[针孔相机模型|反投影]]将潜格子点提升到三维空间
- 在**潜分辨率**（$w \times h$）下完成投影与 [[z-Buffering]]，无需全分辨率操作

#### 模块 2: 深度引导反投影（Depth-Guided Back-Projection）

**设计动机**: 利用 [[单目深度估计]] 为每个潜格子点赋予三维坐标，构建与真实几何对应的稠密点云。

**具体实现**:
- 使用 Depth Anything 3 估计深度图 $D(u,v)$
- 在**潜分辨率内参** $K^\ell$（由 $K^\ell = \text{diag}(w/W, h/H, 1) K$ 从像素内参缩放）下进行反投影
- 对潜格子点 $(u,v)$ 构造三维坐标 $\mathbf{p}_{uv}$ 并关联潜特征 $\mathbf{F}_{uv}$

#### 模块 3: 动态目标过滤（Dynamic Object Filtering）

**设计动机**: 移动物体和天空违反刚性场景假设，若纳入持久记忆会污染场景几何，导致 3D 一致性下降。

**具体实现**:
- 使用 [[Qwen3-VL]]（2B）进行语义实体提取，识别动态/天空区域
- 使用 [[SAM3]]（Segment Anything with Concepts）生成精确分割掩码
- 被掩码覆盖的格子点在更新步骤（Eq. 6）中从 $\Lambda^t$ 中排除，不进入持久缓存

#### 模块 4: ControlNet 风格条件注入

**设计动机**: 利用已有 [[ControlNet]] 范式，以最小参数量将空间记忆读出结果注入冻结 Backbone。

**具体实现**:
- 侧分支镜像 Wan2.2 的 VACE 块结构
- 接入层: 块索引 $\{0, 4, 8, 12, 16, 20, 24, 28\}$，共 8 个注入点
- 输入通道 48，与 [[VAE]] 潜变量通道一致
- 可见性掩码（$\hat{\mathbf{z}}^t$ 与全零背景）帮助 Backbone 区分已知区域与待合成区域

#### 模块 5: 两阶段训练策略

**阶段 1 — 侧分支预热**:
- 冻结 Backbone 与 VAE
- 仅训练条件侧分支
- 学习率 $10^{-5}$，目标是学习空间条件注入的基础能力

**阶段 2 — LoRA 精调**:
- 解锁 Rank-64 [[LoRA]] 适配器（作用于自注意力 $\{q,k,v,o\}$ 投影，全部 30 个 Transformer 块）
- $\alpha = 64$，dropout $= 0.05$
- 学习率 $10^{-4}$，启用梯度检查点，文本 dropout $= 0.2$
- 优化器: AdamW（$\beta=(0.0, 0.999)$，权重衰减 $10^{-3}$），余弦调度，bfloat16 混合精度

---

## 关键公式

### 公式 1: [[RGB 点云记忆|RGB 点云内存表示]]

$$
\mathcal{M}_{\text{rgb}}=\bigl\{(\mathbf{p}_{i},\mathbf{c}_{i})\bigr\},\quad\mathbf{p}_{i}\in\mathbb{R}^{3},\;\mathbf{c}_{i}\in[0,1]^{3}
$$

**含义**: 传统方法将场景表示为带颜色的三维点集，每点存储世界坐标与 RGB 颜色。

**符号说明**:
- $\mathbf{p}_i$: 世界坐标系中的三维点坐标
- $\mathbf{c}_i \in [0,1]^3$: 对应的 RGB 颜色值

---

### 公式 2: [[视频扩散条件化|RGB 条件化流水线]]

$$
\hat{\mathbf{z}}^{t}=\mathcal{E}\bigl(\mathrm{Rasterise}(\mathcal{M}_{\text{rgb}};\,\mathbf{E}^{t},K^{t})\bigr)
$$

**含义**: 每步条件化需先将 RGB 点云光栅化为图像再重新编码为潜变量，引入巨大开销。

**符号说明**:
- $\hat{\mathbf{z}}^t$: 目标视角的条件潜变量
- $\mathcal{E}$: [[VAE]] 编码器
- $\mathrm{Rasterise}(\cdot)$: 点云光栅化渲染器
- $\mathbf{E}^t, K^t$: 目标相机外参与内参矩阵

---

### 公式 3: [[Latent Spatial Memory|潜空间记忆表示]]

$$
\mathcal{M}=\bigl\{(\mathbf{p}_{i},\mathbf{f}_{i})\bigr\},\quad\mathbf{p}_{i}\in\mathbb{R}^{3},\;\mathbf{f}_{i}\in\mathbb{R}^{C}
$$

**含义**: Mirage 将颜色替换为潜特征，直接在特征空间中存储三维场景信息。

**符号说明**:
- $\mathbf{p}_i \in \mathbb{R}^3$: 世界坐标
- $\mathbf{f}_i \in \mathbb{R}^C$: 对应的 VAE 潜特征（$C=48$）

---

### 公式 4: [[深度引导反投影|深度反投影构建记忆]]

$$
\mathbf{p}_{uv}=\pi^{-1}(u,v,D(u,v);\,K,\mathbf{E}),\quad\mathbf{F}_{uv}=\mathbf{z}[:,v,u]
$$

**含义**: 将每个潜格子点 $(u,v)$ 利用深度值反投影到世界坐标，并关联对应潜特征。

**符号说明**:
- $\pi^{-1}$: 针孔相机反投影函数
- $D(u,v)$: 深度估计值（来自 Depth Anything 3）
- $K, \mathbf{E}$: 像素分辨率内参与外参
- $\mathbf{F}_{uv}$: 坐标 $(u,v)$ 处的潜特征切片

---

### 公式 5: [[z-Buffering|潜空间记忆读出（z-buffering）]]

$$
i^{t}(u,v)=\operatorname*{arg\,min}_{i\in\Omega^{t}(u,v)}\bigl[\mathbf{E}^{t}\mathbf{p}_{i}\bigr]_{z},\quad\hat{\mathbf{z}}^{t}(u,v)=\mathbf{F}_{i^{t}(u,v)}
$$

**含义**: 对投影到同一潜格子的多个记忆点，选取深度最小（最近）的点作为该格子的条件特征；无投影时输出零向量。

**符号说明**:
- $\Omega^t(u,v)$: 投影到目标视角格子 $(u,v)$ 的候选点集（见 Eq. 10）
- $[\mathbf{E}^t \mathbf{p}_i]_z$: 点 $\mathbf{p}_i$ 在目标相机坐标系中的深度
- $\hat{\mathbf{z}}^t(u,v)$: 读出的条件潜特征

---

### 公式 6: [[自回归记忆更新|自回归缓存更新]]

$$
\mathcal{M}\leftarrow\mathcal{M}\cup\bigl\{(\mathbf{p}_{uv},\mathbf{F}_{uv})\bigr\}_{(u,v)\in\Lambda^{t}}
$$

**含义**: 每生成一帧后，将新区域（非动态遮罩）的潜特征反投影追加到持久缓存。

**符号说明**:
- $\Lambda^t$: 当前帧中允许写入缓存的格子集合（剔除动态目标掩码区域）
- $(\mathbf{p}_{uv}, \mathbf{F}_{uv})$: 新生成帧的坐标-特征对

---

### 公式 7: [[潜分辨率内参缩放|潜空间内参]]

$$
K^{\ell}=\mathrm{diag}(w/W,\,h/H,\,1)\,K
$$

**含义**: 将像素分辨率内参缩放到潜分辨率，使投影直接在低分辨率特征图上进行。

**符号说明**:
- $K$: 原始像素分辨率内参矩阵
- $K^\ell$: 潜分辨率内参矩阵
- $w/W, h/H$: 潜图相对像素图的空间缩放比（等于 $1/s$，$s=16$）

---

### 公式 8: [[针孔相机模型|针孔反投影操作]]

$$
\pi^{-1}(u,v,d;\,K^{\ell},\mathbf{E})=\mathbf{E}^{-1}\!\begin{bmatrix}d\,(K^{\ell})^{-1}\bigl[u+\tfrac{1}{2},\,v+\tfrac{1}{2},\,1\bigr]^{\top}\\ 1\end{bmatrix}\Bigg|_{1:3}
$$

**含义**: 将潜格子中心点坐标乘以深度值，经相机内参逆变换和外参逆变换，得到世界坐标。

**符号说明**:
- $u+\tfrac{1}{2}, v+\tfrac{1}{2}$: 格子中心（半像素偏移）
- $d$: 深度值
- $\mathbf{E}^{-1}$: 相机外参逆矩阵（相机到世界变换）
- $|_{1:3}$: 取齐次坐标的前三维

---

### 公式 9: [[透视投影|投影到潜格子]]

$$
\pi^{\ell}(\mathbf{q}_{i})=(\lfloor x\rfloor,\lfloor y\rfloor),\quad[x,y,1]^{\top}=K^{\ell}\,\mathbf{q}_{i}/[\mathbf{q}_{i}]_{z}
$$

**含义**: 将相机坐标系下的三维点 $\mathbf{q}_i$ 投影到潜格子坐标，取整得到所在格子。

**符号说明**:
- $\mathbf{q}_i = \mathbf{E}^t \mathbf{p}_i$: 点在目标相机坐标系下的坐标
- $[\mathbf{q}_i]_z$: 深度（用于透视除法）
- $\lfloor \cdot \rfloor$: 向下取整

---

### 公式 10: [[可见性判断|候选集构造]]

$$
\Omega_{t}(u,v)=\bigl\{i:\pi^{\ell}(\mathbf{q}_{i})=(u,v),\;[\mathbf{q}_{i}]_{z}>0\bigr\}
$$

**含义**: 收集所有投影到格子 $(u,v)$ 且位于相机前方（$z>0$）的记忆点，供 z-buffering 选择最近点。

**符号说明**:
- $[\mathbf{q}_i]_z > 0$: 可见性条件（点在相机前方）
- $\pi^\ell(\mathbf{q}_i) = (u,v)$: 投影位置匹配条件

---

## 关键图表

### Figure 1: 几何一致视频生成展示

![Figure 1](https://arxiv.org/html/2606.09828v1/x1.png)

**说明**: Mirage 给定单张输入图像与用户指定相机轨迹（左），通过在[[潜空间]]中缓存三维信息，生成几何一致的视频。对比 RGB 点云方法，背景结构保持稳定无漂移。

---

### Figure 2: 潜空间记忆 vs RGB 点云对比

![Figure 2](https://arxiv.org/html/2606.09828v1/x2.png)

**说明**: 直观对比[[Latent Spatial Memory]]与传统 RGB 点云记忆的流水线差异。RGB 方法（上）需要反复光栅化→重编码；Mirage（下）直接在[[VAE]]潜空间读出条件特征，省去编码往返。

---

### Figure 3: Mirage 整体架构概览

![Figure 3](https://arxiv.org/html/2606.09828v1/x3.png)

**说明**: 展示三步循环——初始化（深度引导反投影构建 $\mathcal{M}$）、读出（[[z-Buffering]]投影到目标视角）、更新（新帧特征追加到缓存）。[[ControlNet]]风格侧分支将读出潜特征注入 Backbone。

---

### Figure 4: 开放域视频对比

![Figure 4](https://arxiv.org/html/2606.09828v1/x5.png)

**说明**: 在 RealEstate10K 训练分布之外的室外/自然场景上进行泛化测试，Mirage 在分布外场景仍保持几何一致性，体现[[潜空间]]表示的泛化优势。

---

### Figure 5: 效率随视频时长的扩展性

![Figure 5](https://arxiv.org/html/2606.09828v1/x6.png)

**说明**: 在单块 NVIDIA H100 上测量每帧缓存读出时间（左）与峰值缓存占用（右）随视频长度的变化。Mirage 的缓存大小随镜头移动缓慢增长，不像 RGB 点云方法随分辨率剧增。

---

### Figure 6: RealEstate10K 视频对比

![Figure 6](https://arxiv.org/html/2606.09828v1/x7.png)

**说明**: 在 RealEstate10K 轨迹上与 ViewCrafter、FlexWorld、Voyager、Spatia 的定性对比。Mirage 在房间结构和纹理一致性上均表现最优。

---

### Figure 7: 闭环回访对比

![Figure 7](https://arxiv.org/html/2606.09828v1/x8.png)

**说明**: 相机轨迹逐渐回到起始点的闭环测试。Mirage 通过持久记忆保证回访时场景外观一致，对比方法出现明显漂移。

---

### Figure 8: 室内复杂轨迹对比

![Figure 8](https://arxiv.org/html/2606.09828v1/x9.png)

**说明**: 在有遮挡、转角、远近景变化的室内轨迹上，Mirage 维持整条轨迹的空间布局一致，展示[[Latent Spatial Memory]]对长程几何的记忆能力。

---

### Table 1: WorldScore 综合评测

| Method | Avg ↑ | Static ↑ | Dynamic ↑ | Camera Ctrl ↑ | Object Ctrl ↑ | 3D Const ↑ | Photo Const ↑ | Style Const ↑ |
|--------|-------|----------|-----------|--------------|--------------|-----------|--------------|--------------|
| WonderJourney | 54.19 | 63.75 | 44.63 | 84.60 | 37.10 | 80.60 | 79.03 | 62.82 |
| InvisibleStitch | 51.95 | 61.12 | 42.78 | 93.20 | 36.51 | 88.51 | 89.19 | 32.37 |
| WonderWorld | 61.79 | 72.69 | 50.88 | 92.98 | 51.76 | 86.87 | 85.56 | 70.57 |
| Voyager | 66.08 | 77.62 | 54.53 | 85.95 | 66.92 | 81.56 | 85.99 | 84.89 |
| FlashWorld | 60.23 | 70.85 | 49.60 | 84.43 | 50.28 | 85.87 | 86.72 | 79.36 |
| LucidDreamer | 59.84 | 70.40 | 49.28 | 88.93 | 41.18 | 90.37 | 90.20 | 48.10 |
| Spatia | 69.73 | 72.63 | 66.82 | 75.66 | 52.32 | 86.40 | 89.10 | 80.09 |
| CogVideoX-I2V | 60.64 | 62.15 | 59.12 | 38.27 | 40.07 | 86.21 | 88.12 | 83.22 |
| Wan2.1 | 55.21 | 57.56 | 52.85 | 23.53 | 40.32 | 78.74 | 78.36 | 77.18 |
| **Mirage (Ours)** | **70.36** | **73.60** | **67.11** | 55.36 | **74.17** | **92.21** | **93.95** | **96.91** |

**关键发现**: Mirage 在综合得分（70.36）上超越所有对比方法（含 RGB 点云最强基线 Spatia 69.73）；3D 一致性（92.21）、光度一致性（93.95）、风格一致性（96.91）三项均为最优，动态场景得分（67.11）同样领先。

---

### Table 2: RealEstate10K 新视角合成评测

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PSNR_C ↑ | SSIM_C ↑ | LPIPS_C ↓ |
|--------|--------|--------|---------|---------|---------|----------|
| SEVA | 13.07 | 0.515 | 0.445 | — | — | — |
| VMem | 14.62 | 0.522 | 0.426 | — | — | — |
| ViewCrafter | 15.78 | 0.580 | 0.396 | 14.79 | 0.481 | 0.365 |
| FlexWorld | 16.25 | 0.593 | 0.370 | 12.20 | 0.428 | 0.598 |
| Voyager | 17.79 | 0.636 | 0.297 | 17.66 | 0.540 | 0.380 |
| Spatia | 18.58 | 0.646 | 0.254 | 19.38 | 0.579 | 0.213 |
| **Mirage (Ours)** | 18.38 | **0.779** | **0.250** | **20.05** | **0.825** | **0.228** |

**关键发现**: Mirage 在 SSIM（0.779 vs 0.646）、LPIPS（0.250 vs 0.254）、闭环 PSNR（20.05 vs 19.38）和闭环 SSIM（0.825 vs 0.579）上全面领先，闭环一致性指标（$C$ 下标）提升尤为显著，验证持久记忆在长程轨迹中的优势。

---

### Table 3: 消融实验（WorldScore）

| 变体 | Avg ↑ | Static ↑ | Dynamic ↑ | 3D Cons ↑ | Photo Cons ↑ |
|------|-------|----------|-----------|-----------|--------------|
| Mirage（完整） | **70.36** | **73.60** | **67.11** | **92.21** | **93.95** |
| 显式 RGB 点云 | 67.71 | 70.49 | 64.93 | 90.75 | 91.10 |
| 特征上采样（像素分辨率投影） | 60.85 | 62.41 | 59.28 | 84.90 | 79.81 |
| 无动态目标过滤 | 61.20 | 62.69 | 59.70 | 80.88 | 76.10 |
| 单阶段训练 | 63.18 | 65.15 | 61.20 | 87.11 | 84.47 |

**关键发现**:
- 用 RGB 点云替换潜空间记忆：Avg 下降 2.65，证明潜特征表示本身的价值（非仅效率）
- 在像素分辨率投影后上采样替代潜分辨率直接投影：Avg 下降 9.51，说明低分辨率直接投影精度更高
- 移除动态目标过滤：3D 一致性从 92.21 跌至 80.88（-11.33），是最关键组件
- 单阶段训练（跳过侧分支预热）：Avg 下降 7.18，两阶段策略不可省略

---

### Table 4: 深度来源鲁棒性

| 深度来源 | Avg ↑ | 3D Cons ↑ | Photo Cons ↑ |
|---------|-------|-----------|--------------|
| Depth Anything 3（默认） | **70.36** | **92.21** | **93.95** |
| MapAnything | 69.66 | 91.89 | 93.32 |
| UniDepth | 69.13 | 91.63 | 92.79 |

**关键发现**: 方法对深度来源不敏感，三种深度估计器的性能差距在 1.2 分以内，说明框架具有良好的深度鲁棒性。

---

### Table 5: 深度下采样方式对缓存空洞率的影响

| 下采样方法 | 空洞率 ↓ |
|-----------|---------|
| 双线性插值（默认） | **42.53%** |
| 最近邻 | 47.78% |
| 区域池化 | 53.72% |
| 中值池化 | 52.22% |

**关键发现**: 双线性插值在从像素分辨率深度图下采样到潜分辨率时产生最少空洞，是缓存覆盖率最优的方案。

---

## 实验

### 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| [[RealEstate10K]] | 训练 + 测试 | 室内场景视频，带相机轨迹、内外参；经 Depth Anything 3 处理深度，Qwen3-VL-2B + SAM3 生成动态掩码 |
| [[WorldScore]] | 测试（综合评测） | 多维度视频生成 benchmark，涵盖静态/动态/相机控制/一致性等 13 个指标 |

### 实现细节

- **Backbone**: [[Wan2.2-TI2V-5B]]（5B 参数，Transformer hidden 3072，FFN 14336，24 头，30 块，VAE stride 16，$C=48$）
- **优化器**: AdamW，$\beta=(0.0, 0.999)$，weight decay $10^{-3}$，余弦调度，bfloat16 混合精度
- **学习率**: Stage 1 $10^{-5}$（侧分支），Stage 2 $10^{-4}$（含 LoRA）
- **Batch Size**: 64（全局）
- **硬件**: 32× NVIDIA A100
- **LoRA 配置**: Rank-64，$\alpha=64$，dropout 0.05，作用于全部 30 块的 $\{q,k,v,o\}$ 投影
- **条件注入**: ControlNet 风格侧分支，注入层 $\{0,4,8,12,16,20,24,28\}$
- **深度估计**: Depth Anything 3（monocular depth）
- **动态掩码**: Qwen3-VL-2B 语义提取 + SAM3 分割

### 可视化结果

- **室内场景**: 走廊、客厅等 RealEstate10K 典型场景上，相机前进/旋转时墙面纹理、家具位置保持一致，无"橡皮泥"形变
- **开放域泛化**: 室外自然场景（训练分布外）同样保持几何一致，说明潜空间记忆的泛化能力
- **闭环测试**: 相机回到原点时场景外观与初始帧高度一致，RGB 点云基线出现明显视觉漂移

---

## 批判性思考

### 优点

1. **效率突破**: 10.57× 加速和 55× 内存压缩是量级提升，为长程视频生成打开实用窗口
2. **质量同步提升**: 不仅更快，WorldScore 和 RealEstate10K 指标均超越 RGB 基线，说明潜特征记忆不是质量-效率的折中
3. **架构简洁**: 无需修改主干，仅通过 ControlNet 侧分支和轻量 LoRA 即可接入，迁移成本低
4. **动态过滤设计务实**: 将 VLM（Qwen3-VL）和分割模型（SAM3）用于掩码生成，是工程上简单有效的解法

### 局限性

1. **动态物体不持久**: 当前方法过滤掉动态物体，无法在长程视频中追踪/记住运动实体的位置和状态，限制了以动态内容为主的场景
2. **相机控制精度偏低**: WorldScore 相机控制得分（55.36）明显低于 WonderWorld（92.98）等专注相机控制的方法，说明精确轨迹跟随是弱项
3. **内容对齐有限**: WorldScore 内容对齐得分（42.09）在对比方法中垫底，文本条件控制生成内容的能力有待改进
4. **刚性场景假设**: 深度反投影和记忆缓存均基于刚性世界假设，对可变形体（布料、树叶）处理不当
5. **训练数据单一**: 仅在 RealEstate10K 上微调，可能限制对极端视角变化或风格多样场景的泛化

### 潜在改进方向

1. **动态物体记忆模块**: 为运动实体建立独立的可变形记忆，支持长程动态场景
2. **精确相机控制**: 结合相机姿态损失或 epipolar attention 提升轨迹跟随精度
3. **多模态条件融合**: 更强的文本-场景对齐，解决 content alignment 得分低问题
4. **在线深度校准**: 利用视频帧间一致性动态校正深度估计噪声

### 可复现性评估

- [x] 代码开源（github.com/microsoft/LatentSpatialMemory）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（超参数、数据处理流程均有描述）
- [x] 数据集可获取（RealEstate10K 公开）

---

## 关联笔记

### 基于

- [[Wan2.2-TI2V-5B]]: 使用其作为视频生成 Backbone
- [[VAE]]: 潜空间记忆的底层表示空间
- [[ControlNet]]: 条件注入的架构范式

### 对比

- [[Spatia]]: 最近的 RGB 点云基线，WorldScore 69.73 vs Mirage 70.36
- [[Voyager]]: 另一 RGB 点云方法，RealEstate10K SSIM 0.636 vs Mirage 0.779
- [[ViewCrafter]]: 早期新视角合成方法
- [[FlexWorld]]: 相机可控视频生成方法

### 方法相关

- [[Latent Spatial Memory]]: 本文核心贡献
- [[z-Buffering]]: 潜空间可见性判断的核心算法
- [[LoRA]]: 轻量微调策略
- [[单目深度估计]]: 反投影的深度来源
- [[SAM3]]: 动态目标分割工具
- [[Qwen3-VL]]: 动态目标语义识别

### 数据集相关

- [[RealEstate10K]]: 主要训练和测试数据集
- [[WorldScore]]: 综合评测 benchmark

---

## 速查卡片

> [!summary] Mirage: Latent Spatial Memory for Video World Models
> - **核心**: 将三维场景信息缓存在 VAE 潜空间而非 RGB 点云，消除"光栅化→重编码"瓶颈
> - **方法**: 深度引导反投影构建潜缓存 + z-buffering 直接潜空间读出 + 动态目标过滤
> - **结果**: 10.57× 速度提升，55× 内存压缩，WorldScore SOTA（70.36），RealEstate10K SSIM 0.779
> - **代码**: https://github.com/microsoft/LatentSpatialMemory

---

*笔记创建时间: 2026-06-10*

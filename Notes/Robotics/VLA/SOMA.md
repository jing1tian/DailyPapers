---
title: "Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action"
method_name: "SOMA"
authors: [Pengteng Li, Weiyu Guo, He Zhang, Tiefu Cai, Xiao He, Yandong Guo, Hui Xiong]
year: 2026
venue: ICML 2026
tags: [vla, spatial-memory, out-of-vision-manipulation, multi-view, partial-observability]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.22283
created: 2026-05-23
---

# 论文笔记：SOMA — Spatial Memory for Out-of-Vision Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未披露（通讯作者 Hui Xiong） |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22283) / Code coming soon |

---

## 一句话总结

> SOMA 为 VLA 模型引入持久性空间记忆，通过多视角扫描构建全局语义-几何表示，使机器人能操作相机视野之外的物体。

---

## 核心贡献

1. **Out-of-Vision（OOV）问题的系统性定义与基准**：构建了 5 个真实场景的 OOV pick-and-place 任务，涵盖单臂、双物体、双臂协作场景，填补了 VLA 在部分可观测操作评测上的空白。
2. **持久性空间记忆框架**：提出三阶段记忆机制——空间记忆构建（Spatial Memory Construction）、动态记忆精炼（Dynamic Memory Refinement）、上下文记忆检索（Contextual Memory Retrieval），将多视角观测聚合为可持久使用的语义-几何 token。
3. **指令对齐的记忆检索**：通过 [[Cross-Attention|交叉注意力]] 将语言指令与空间记忆 token 对齐，选择性激活与任务相关的空间区域，而非简单地将所有记忆灌入策略网络。

---

## 问题背景

### 要解决的问题

现有 [[VLA|视觉-语言-动作模型]] 基于当前帧进行 reactive 决策，无法处理"视野之外"（Out-of-Vision, OOV）的目标物体——即目标在任务执行初始阶段不在相机视锥范围内，需要机器人主动探索后才能抓取。

### 现有方法的局限

- **VLA 模型（π₀、OpenVLA、GR00T-N1.5 等）**：本质上是视野依赖（view-dependent）的，缺乏跨时间步保留和复用空间信息的机制。
- **主动感知方法（如 MemoryVLA、MemER）**：存储历史观测，但不具备 3D 空间几何先验，且多数方法是 instruction-agnostic 的，无法根据当前任务指令选择性地提取相关记忆。
- **简单的 scan + VLA 组合（如 Scan+GR00T）**：在扫描阶段收集图像，但策略本身无法利用扫描信息，成功率仅与无扫描持平（18.5% vs 19.8%）。

### 本文的动机

机器人执行任务前可以通过主动扫描场景建立一张"空间地图"，将各视角的检测结果融合为统一的 3D 语义表示并在任务执行全程维护更新，从而在视野被遮挡或物体移出视锥后仍能精准定位目标，类似人类对工作台的空间认知。

---

## 方法详解

### 模型架构

SOMA 在 [[GR00T N1.5]] VLA 基础上附加一个**空间记忆模块**，整体采用"感知扫描 → 记忆构建 → 策略执行"三阶段流程：

- **输入**：扫描视频帧序列（任务前）+ 当前观测帧 $o_t$（执行中）+ 语言指令 $l$
- **空间感知工具链**：[[YOLO-World]]（开放词汇检测）+ [[DINOv2]]（外观特征）+ [[VGGT]]（几何先验）
- **记忆 token**：每个实例 $k$ 对应一个记忆向量 $\mathbf{m}_k$，融合语义与位置信息
- **策略骨干**：GR00T-N1.5（保持冻结或微调，注入记忆 token）
- **输出**：机器人动作 $a_{t:t+k}$（Action Chunking）

### 核心模块

#### 模块 1：空间记忆构建（Spatial Memory Construction）

**设计动机**：利用 [[多视角几何]] 将分散的单帧检测聚合为全局一致的 3D 实例表示。

**具体实现**：

1. 对扫描视频均匀采样，逐帧用 YOLO-World 检测目标，得到 2D bounding box 与类别信息。
2. 用 DINOv2 提取每个检测框的外观特征 $\mathbf{f}_k$（instance-level embedding）。
3. 用 VGGT 将 2D 检测 lift 到全局坐标系，得到 3D 空间位置编码 $\mathbf{p}_k$。
4. 跨视角实例关联：用联合外观-几何相似度匹配不同帧中的同一物体，合并为统一记忆 token。

初始记忆 token 由语义特征与位置编码相加构成：

$$
\mathbf{m}_k^0 = \Phi_{\text{mem}}(\mathbf{f}_k) + \mathbf{p}_k
$$

#### 模块 2：动态记忆精炼（Dynamic Memory Refinement）

**设计动机**：任务执行中视角变化会带来新的观测，用[[指数移动平均|EMA]]式更新维护记忆的时序一致性，同时用自适应融合系数（语义相似度 × 融合分数）防止无关观测污染记忆。

**具体实现**：

- 语义相似度打分：用 MLP $\Phi_{\text{sim}}$ 判断新观测 $\mathbf{m}_j^t$ 与历史记忆 $\mathbf{m}_k^{t-1}$ 的匹配度
- 融合打分：用 MLP $\Phi_{\text{fuse}}$ 决定更新权重
- 自适应融合系数：$\alpha_{kj}^t = g_{kj}^t \cdot s_{kj}^t$（相似度与融合分数的乘积）
- [[指数移动平均|EMA]] 更新：$\mathbf{m}_k^t = \alpha_{kj}^t \mathbf{m}_j^t + (1 - \alpha_{kj}^t) \mathbf{m}_k^{t-1}$

#### 模块 3：上下文记忆检索（Contextual Memory Retrieval）

**设计动机**：并非所有空间记忆都与当前指令相关，通过 [[Cross-Attention|交叉注意力]] 实现 instruction-grounded 的记忆激活，降低无关记忆对策略的干扰。

**具体实现**：

- Query $Q$：来自 VLA 的视觉-语言 embedding（指令条件化）
- Key/Value $K, V$：空间记忆 token 集合 $\{\mathbf{m}_k\}$
- 输出 $X_{\text{boost}}$ 注入 VLA 骨干，增强策略对 OOV 目标的感知

---

## 关键公式

### 公式 1：[[空间记忆|初始记忆 Token 构建]]

$$
\mathbf{m}_k^0 = \Phi_{\text{mem}}(\mathbf{f}_k) + \mathbf{p}_k
$$

**含义**：将实例外观特征经 MLP 映射后与 3D 位置编码相加，得到融合语义与空间位置的记忆向量。

**符号说明**：
- $\mathbf{m}_k^0$：实例 $k$ 的初始记忆 token
- $\Phi_{\text{mem}}(\cdot)$：语义特征映射 MLP
- $\mathbf{f}_k$：DINOv2 提取的实例外观特征
- $\mathbf{p}_k$：VGGT 估计的 3D 位置编码

### 公式 2：[[指数移动平均|动态记忆更新（自适应 EMA）]]

$$
s_{kj}^t = \sigma\!\left(\Phi_{\text{sim}}\!\left([\mathbf{m}_k^{t-1} - \mathbf{m}_j^t]\right)\right)
$$

$$
g_{kj}^t = \sigma\!\left(\Phi_{\text{fuse}}\!\left([\mathbf{m}_k^{t-1},\, \mathbf{m}_j^t]\right)\right)
$$

$$
\alpha_{kj}^t = g_{kj}^t \cdot s_{kj}^t
$$

$$
\mathbf{m}_k^t = \alpha_{kj}^t\, \mathbf{m}_j^t + (1 - \alpha_{kj}^t)\, \mathbf{m}_k^{t-1}
$$

**含义**：用语义相似度 $s$ 与融合分数 $g$ 共同决定新观测对历史记忆的更新比例，相似且高置信的观测会更大幅度刷新记忆，防止噪声干扰。

**符号说明**：
- $s_{kj}^t$：语义相似度分数（取值 $[0,1]$）
- $g_{kj}^t$：融合分数（取值 $[0,1]$）
- $\alpha_{kj}^t$：自适应更新系数（二者乘积）
- $\sigma(\cdot)$：Sigmoid 激活函数
- $\Phi_{\text{sim}}, \Phi_{\text{fuse}}$：两个独立的轻量 MLP
- $\mathbf{m}_j^t$：当前时刻新观测的记忆 token

### 公式 3：[[Cross-Attention|上下文记忆检索（交叉注意力）]]

$$
X_{\text{boost}} = \left\{x'_i\right\}_{i=1}^{N_q} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{C}}\right)V
$$

**含义**：以 VLA 视觉-语言 embedding 为 Query，空间记忆 token 为 Key/Value，通过交叉注意力选择性地提取与当前指令相关的空间信息，注入策略骨干。

**符号说明**：
- $Q$：来自 VLA 的指令条件化 query，维度 $N_q \times C$
- $K, V$：空间记忆 token 集合，维度 $N_m \times C$
- $C$：特征通道维度（缩放因子）
- $X_{\text{boost}}$：增强后的视觉特征，注回 VLA backbone

---

## 关键图表

### Figure 1：OOV 问题示意

![Figure 1 - OOV limitation in VLA](https://arxiv.org/html/2605.22283v1/x1.png)

**说明**：展示了现有 [[VLA]] 模型在 Out-of-Vision 场景下的失效模式——当目标物体位于相机视锥之外时，反应式策略无法定位目标，只能进行大量低效的视角搜索。

### Figure 2：SOMA 框架整体架构

![Figure 2 - SOMA framework overview](https://arxiv.org/html/2605.22283v1/x2.png)

**说明**：SOMA 完整流程图，展示三大模块的数据流：扫描视频经空间记忆构建模块生成初始记忆 token $\{\mathbf{m}_k^0\}$；执行阶段通过动态精炼持续更新；上下文检索模块将记忆注入 [[GR00T N1.5]] 骨干，输出动作指令。

### Figure 3：真实世界基准任务设置

![Figure 3 - Real world benchmark](https://arxiv.org/html/2605.22283v1/x3.png)

**说明**：五个递进式 OOV 任务的场景布局：Task 1-3 为单臂单目标（不同遮挡程度），Task 4 为单臂双目标序列操作，Task 5 为双臂协调任务，全部使用人形机器人（双 7-DoF Realman ZM73 臂 + 2-DoF 电动头部）。

### Figure 4：真实世界任务成功率对比

![Figure 4 - Real world success rate comparison](https://arxiv.org/html/2605.22283v1/x4.png)

**说明**：SOMA 在全部 5 个 OOV 任务上均大幅领先 GR00T-N1.5 基线，整体成功率从 ~18-20% 提升至 ~28%，且随任务复杂度上升优势更明显。

### Figure 5：任务执行示例

![Figure 5 - Task execution examples](https://arxiv.org/html/2605.22283v1/x5.png)

**说明**：定性展示 SOMA 在 5 个真实任务中的执行轨迹——目标快速定位、极少视角修正次数、近乎一次成功抓取（near one-shot grasping）。

### Figure 6：机器人硬件平台

![Figure 6 - Robot hardware](https://arxiv.org/html/2605.22283v1/x6.png)

**说明**：自研人形机器人平台：双 7-DoF Realman ZM73 机械臂、2-DoF 电动头部（搭载 Intel RealSense D435 RGB-D 相机）、NVIDIA Jetson AGX Orin 64GB 板载计算。

### Figure 7：VR 遥操作系统

![Figure 7 - VR teleoperation system](https://arxiv.org/html/2605.22283v1/img/VR.png)

**说明**：使用 Meta Quest 3 构建的 VR 遥操作数据采集系统，用于收集每任务 400 条专家示范，并经帧级质量过滤（300ms 时延阈值、2.5 帧/25FPS 对齐阈值）。

### Table 1：行为分析指标（真实任务）

| 指标 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------|--------|--------|--------|--------|--------|
| **首次注视时间 (s) ↓** | | | | | |
| GR00T-N1.5 | 7.6 | 21.0 | 14.8 | 10.9 | 11.5 |
| SOMA | 4.2 ↓45% | 12.7 ↓40% | 8.2 ↓45% | 4.9 ↓55% | 4.7 ↓59% |
| **头部搜索路径长度 (deg) ↓** | | | | | |
| GR00T-N1.5 | 50.5 | 51.0 | 83.8 | 109.2 | 164.0 |
| SOMA | 27.8 ↓45% | 28.1 ↓45% | 50.3 ↓40% | 54.6 ↓50% | 70.4 ↓57% |
| **视角修正次数 ↓** | | | | | |
| GR00T-N1.5 | 1.6 | 1.9 | 1.4 | 3.4 | 5.3 |
| SOMA | 0.9 ↓44% | 1.1 ↓42% | 0.8 ↓43% | 1.7 ↓50% | 2.3 ↓57% |
| **抓取尝试次数 ↓** | | | | | |
| GR00T-N1.5 | 1.8 | 2.0 | 1.7 | 2.4 | 3.7 |
| SOMA | 1.0 ↓44% | 1.2 ↓40% | 1.0 ↓41% | 1.2 ↓50% | 1.6 ↓57% |
| **抓取耗时 (s) ↓** | | | | | |
| GR00T-N1.5 | 58.0 | 30.0 | 50.0 | 65.5 | 36.5 |
| SOMA | 32.3 ↓44% | 16.8 ↓44% | 29.7 ↓41% | 30.4 ↓54% | 14.6 ↓60% |

**关键发现**：SOMA 在所有行为指标上实现 40–60% 的一致性提升，说明空间记忆从根本上改变了机器人的探索策略——从盲目搜索变为目标直达。

### Table 2：扫描与空间记忆消融

| 配置 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | **平均 SR (%)** |
|------|--------|--------|--------|--------|--------|----------------|
| Scan + GR00T | 19.0 | 22.0 | 16.0 | 25.0 | 10.5 | 18.5 |
| No-Scan SOMA | 20.0 | 24.0 | 17.5 | 26.0 | 11.7 | 19.8 |
| Scan-only SOMA | 25.0 | 29.0 | 22.5 | 30.0 | 14.2 | 24.1 |
| **Full SOMA** | **30.0** | **35.0** | **27.5** | **32.5** | **16.7** | **28.3** |

**关键发现**：仅添加扫描（Scan+GR00T）几乎没有帮助，说明 VLA 本身无法利用历史帧；SOMA 的记忆机制才是提升的关键，且扫描与记忆互为补充，缺一不可。

### Table 3：RoboCasa Tabletop GR1 基准（300 条示范）

| 任务类别 | Diffusion Policy | StarVLA | GR00T-N1.5 | **SOMA** |
|----------|-----------------|---------|------------|---------|
| Container Interaction | 54.2 | 1.8 | 35.7 | **53.3** |
| Cooking Preparation | 35.3 | 25.8 | 42.8 | **48.4** |
| Tabletop Serving | 28.4 | 24.0 | 48.5 | **54.0** |
| Dish Transfer | 39.0 | 21.5 | 51.0 | 55.0 |
| Tray Organization | 39.2 | 27.9 | 43.6 | **48.8** |
| **平均** | **39.2** | **20.2** | **44.3** | **52.0** |

**关键发现**：SOMA 在标准 in-distribution 任务上也超越 GR00T-N1.5 基线（52.0% vs 44.3%），说明空间记忆对常规操作同样有益。

### Table 4a：SimplerEnv Visual Matching

| 模型 | Pick Coke Can | Move Near | Open/Close Drawer | **平均** |
|------|---------------|-----------|-------------------|---------|
| OpenVLA-OFT | 72.3 | 69.6 | 47.2 | 63.0 |
| GR00T-N1.5 | 47.0 | 70.0 | 18.1 | 45.0 |
| **SOMA** | **85.0** | **73.0** | 31.5 | **63.2** |

### Table 4b：SimplerEnv Variant Aggregation

| 模型 | Pick Coke Can | Move Near | Open/Close Drawer | **平均** |
|------|---------------|-----------|-------------------|---------|
| RoboVLM | 75.6 | 60.0 | 10.6 | 51.3 |
| GR00T-N1.5 | 46.7 | 62.9 | 17.5 | 42.4 |
| **SOMA** | 55.5 | **76.6** | 25.4 | **52.5** |

**关键发现**：SimplerEnv 上 SOMA 达到 SOTA（63.2% / 52.5%），在 Pick Coke Can 任务上领先尤为显著（85.0%），但 Open/Close Drawer 任务上弱于 OpenVLA-OFT，提示空间记忆对精细铰接操作的帮助有限。

### Table 5：记忆组件消融（GR1 Benchmark）

| 消融配置 | 几何线索 | 目标语义 | 动态更新 | **整体成功率 (%)** |
|----------|----------|----------|----------|-------------------|
| w/o 几何线索 | ✗ | ✓ | ✓ | 45.1 |
| w/o 目标语义 | ✓ | ✗ | ✓ | 43.7 |
| w/o 动态更新 | ✓ | ✓ | ✗ | 41.5 |
| **Full SOMA** | ✓ | ✓ | ✓ | **49.3** |

**关键发现**：三个组件均不可缺，其中动态更新的贡献最大（去掉后下降 7.8pp），几何线索次之（4.2pp），目标语义第三（5.6pp）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自采真实 OOV 数据集 | 每任务 400 条专家示范（VR 遥操作） | 含 5 个 OOV pick-and-place 任务，帧级质量过滤 | 训练 + 评测 |
| RoboCasa Tabletop GR1 | 30 / 100 / 300 条 | 5 个桌面操作类别，GR1 人形机器人 | 测试泛化性 |
| SimplerEnv | 标准评测集 | Visual Matching + Variant Aggregation | 仿真评测 |

### 实现细节

- **VLA 骨干**：[[GR00T N1.5]]（基础模型，接入空间记忆模块）
- **视觉检测**：[[YOLO-World]]（开放词汇，无需重训练）
- **外观特征**：[[DINOv2]]（实例级 embedding）
- **3D 几何先验**：[[VGGT]]（单目多视角几何估计）
- **记忆更新 MLP**：$\Phi_{\text{sim}}$、$\Phi_{\text{fuse}}$、$\Phi_{\text{mem}}$（轻量 MLP）
- **硬件（真实机器人）**：双 7-DoF Realman ZM73 + 2-DoF 电动头部 + Intel RealSense D435 + NVIDIA Jetson AGX Orin 64GB
- **数据采集**：Meta Quest 3 VR 遥操作系统，25 FPS，质量阈值：时延 ≥300ms、对齐偏差 ≥2.5 帧
- **代码**：尚未开源（"Code will be released soon"）

### 可视化结果

SOMA 在真实任务执行中展现出明显不同于 baseline 的行为模式：
- **目标快速注视**（首次注视时间减少 40–59%）：扫描后即知目标位置，无需盲目转头搜索
- **近一次成功抓取**（抓取尝试次数 ≈1）：空间记忆提供精准的 3D 位置，减少校正迭代

---

## 批判性思考

### 优点

1. **问题定义清晰**：OOV 是真实操作场景中被长期忽视的痛点，文章对此做了系统性定义与基准构建，具有较高的实用价值。
2. **模块化设计**：三阶段记忆框架（构建→精炼→检索）逻辑清晰，每个组件均有消融验证，且检索模块的 instruction-grounding 设计比简单全量注入更优雅。
3. **行为指标多维度**：除成功率外还报告了首次注视时间、搜索路径、修正次数等行为指标，对分析系统机制价值更高。

### 局限性

1. **VGGT 位姿漂移**：论文 Appendix 承认在较长操作序列中 VGGT 的深度估计存在累积漂移，影响 3D 记忆精度，限制了更长时域任务的适用性。
2. **依赖扫描阶段**：需要任务前的主动扫描过程，若场景动态变化（如他人移动物体）则记忆可能过时，缺乏主动重扫描机制。
3. **Open/Close Drawer 任务弱势**：在需要精细铰接操作的子任务（SimplerEnv 抽屉开关）上表现不及部分 baseline，空间记忆对精细操作的帮助有限。
4. **计算开销未报告**：未披露空间记忆模块引入的额外推理延迟，对实时性要求高的场景不确定是否实用。

### 潜在改进方向

1. **与场景完整重建结合**：引入 3DGS 或 NeRF 实现更精准的场景理解，替代 VGGT 几何估计。
2. **动态环境适应**：加入变化检测机制，在检测到场景变化时触发局部记忆更新，而非依赖初始扫描。
3. **安全感知控制**：论文提及可与安全约束（CBF）结合，将记忆用于障碍感知和碰撞规避。

### 可复现性评估

- [ ] 代码开源（尚未）
- [ ] 预训练模型（未提供）
- [x] 训练细节基本完整（VR 采集、过滤标准有详细描述）
- [ ] 真实数据集可获取（未开放）

---

## 关联笔记

### 基于

- [[GR00T N1.5]]：VLA 骨干，SOMA 在其基础上附加空间记忆模块
- [[VGGT]]：提供单目多视角几何先验，支撑 3D 位置编码
- [[DINOv2]]：实例外观特征提取器
- [[YOLO-World]]：开放词汇目标检测，支持任意类别的场景扫描

### 对比

- [[GR00T N1.5]]：主要对比基线，展示空间记忆带来的提升
- [[Diffusion Policy|扩散策略]]：RoboCasa 基准对比项之一
- [[TraceVLA]]：SimplerEnv 对比项，基于轨迹可视化的 VLA

### 方法相关

- [[Cross-Attention|交叉注意力]]：上下文记忆检索的核心机制
- [[指数移动平均]]：动态记忆精炼的更新方式
- [[空间记忆]]：本文提出的核心概念，需新建概念笔记
- [[部分可观测性]]：OOV 问题的形式化背景概念
- [[多视角几何]]：跨视角实例关联的理论基础

### 硬件/数据相关

- [[Realman ZM73]]：双臂人形机器人平台使用的机械臂
- [[Intel RealSense D435]]：机器人头部 RGB-D 相机
- [[Meta Quest 3]]：VR 遥操作数据采集工具

---

## 速查卡片

> [!summary] SOMA: Spatial Memory for Out-of-Vision Manipulation
> - **核心**：为 VLA 引入持久性空间记忆，解决相机视野外目标的操作问题
> - **方法**：扫描构建 3D 语义记忆 → EMA 动态精炼 → 交叉注意力检索注入策略
> - **结果**：真实 OOV 任务整体成功率 28.3%（vs 基线 18.5%），行为指标减少 40–60%；SimplerEnv SOTA 63.2%
> - **代码**：Coming soon (https://arxiv.org/abs/2605.22283)

---

*笔记创建时间: 2026-05-23*

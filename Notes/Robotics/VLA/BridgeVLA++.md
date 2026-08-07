---
title: "BridgeVLA++: A Data-Efficient, Generalizable, and Memory-Augmented Vision-Language-Action Framework for 3D Manipulation"
method_name: "BridgeVLAPP"
authors: [Peiyan Li, Yuze Zhu, Yixiang Chen, Qisen Ma, Yuan Xu, Jiabing Yang, He Guan, Yan Huang, Hongtao Wu, Xiao Ma, Tao Kong, Liang Wang, Tieniu Tan]
year: 2026
venue: arXiv (under review TPAMI)
tags: [vla, 3d-manipulation, memory-augmented, heatmap-action, point-cloud, generalization, bimanual]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.05042
created: 2026-08-07
---

# 论文笔记：BridgeVLA++

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Chinese Academy of Sciences (CASIA), ByteDance |
| 日期 | August 2026 |
| 项目主页 | [bridgevla-plus.github.io](https://bridgevla-plus.github.io/) |
| 对比基线 | [[SAM2Act]], [[RVT-2]], [[LingBot-VLA-2.0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.05042) / Code (TBD) |

---

## 一句话总结

> BridgeVLA++ 在 BridgeVLA 的统一二维热图框架基础上，引入时空记忆模块，以极低数据量（10 次示范）在多平台 3D 操纵、双臂操纵及记忆依赖任务上达到 SOTA。

---

## 核心贡献

1. **统一 2D 热图动作接口**：将[[正交投影多视图|三视图正交投影]]与 VLM 的二维预训练空间对齐，避免离散化 token 造成的精度损失，在 RLBench 上达到 90.5%
2. **时空记忆架构（MemoryBridgeVLA++）**：[[时空记忆|时间记忆]]存储锚点帧与关键帧序列，[[空间记忆]]在细化阶段提供遮挡鲁棒的参考几何，RMBench 上从 18.9% 跃升至 96.0%
3. **极低数据效率**：Franka 真实机器人上仅需 3 次示范即达 95.4%，10 次达 96.2%，超越 50 次示范的 SpatialVLA

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在 3D 操纵中面临三重挑战：
1. **数据饥渴**：需大量真实机器人示范，部署成本极高
2. **分布外泛化差**：对光照、纹理、颜色变化鲁棒性不足
3. **记忆依赖任务失败**：需要跨帧推理（如先看后操作）时完全失效

### 现有方法的局限

- [[PerAct]]、[[Act3D]]、[[RVT-2]]：依赖离散化 token 预测 3D 坐标，精度受词表大小制约
- [[Diffusion Policy]]：在记忆依赖场景下无上下文建模
- MemoryWAM：有记忆但依赖大规模预训练数据

### 本文的动机

[[PaliGemma]] 等 VLM 在二维空间预训练，直接输入 3D 坐标破坏了预训练对齐。将点云渲染为[[正交投影多视图|正交投影图]]，让 VLM 的"视觉 token"与动作位置自然对应，从而保留预训练能力的同时实现精确 3D 定位。

---

## 方法详解

### 模型架构

**BridgeVLA** 采用 **VLM + 热图解码** 架构：

- **输入**：语言指令 $l$ + 点云 $\mathbf{P}_t$ 渲染的[[正交投影多视图|三视图正交投影]]（top / front / right）
- **Backbone**：[[PaliGemma]]（SigLIP 视觉编码器 + Gemma 语言解码器），总参数 ~2.92B
- **核心模块**：[[热图动作解码|粗阶热图]] + [[粗到细动作精化|细化阶段]]
- **输出**：端效器 3D 位置热图 + 旋转 + 夹爪开合 + 碰撞标志
- **总参数**（BridgeVLA++）：2.92B + 269.77M = ~3.19B

**BridgeVLA++** 在此基础上注入[[时空记忆]]模块：
- [[时间记忆]] $\mathcal{T}_t$ 作用于粗阶特征 $\mathbf{Z}_t^c$
- [[空间记忆]] $\mathcal{S}_t$ 作用于细阶特征 $\mathbf{Z}_t^f$

![Figure 1: Overview](https://arxiv.org/html/2608.05042v1/x1.png)

**说明**：BridgeVLA++ 整体框架。点云经[[正交投影多视图|正交投影]]变为三视图图像，VLM 在 2D 空间预测热图，再反投影到 3D 端效器位置。BridgeVLA++ 在粗/细阶段分别注入时间和空间记忆。

---

### 核心模块

#### 模块 1：正交投影多视图表示

**设计动机**：直接将 [[点云]] 传入 VLM 需要额外的 3D 编码器，且破坏 VLM 的 2D 预训练分布。[[正交投影多视图|正交投影]]将点云渲染为 top/front/right 三张灰度深度图 + RGB 图像，完全在 VLM 原有的视觉 token 空间内操作。

**具体实现**：
- 以[[正交投影多视图|正交视角]]分别渲染深度和颜色，拼接为多通道图
- 粗阶：全场景低分辨率三视图
- 细阶：对当前预测点 $\hat{\mathbf{x}}_t^c$ 做局部 Zoom，高分辨率三视图

#### 模块 2：预训练对齐（RoboPoint 热图预训练）

**设计动机**：直接从 scratch 学习热图预测收敛慢、泛化差。用[[RoboPoint]]数据集（120K 样本）做语言条件下的 2D 目标定位预训练，激活 VLM 的空间理解能力。

**具体实现**：
- Ground truth 热图用**截断高斯**构建（见公式4–7）
- 损失：[[交叉熵损失|像素级交叉熵]]
- 预训练后微调时 VLM 权重可端到端更新

#### 模块 3：时间记忆 $\mathcal{T}_t$

**设计动机**：记忆依赖任务（如"先观察再操作"）需要跨帧推理，单帧输入完全失效。

**具体实现**：
$$
\mathcal{T}_t = \left(\mathbf{A}_0,\; \mathcal{H}_t^{\text{nbr}},\; \mathcal{H}_t^{\text{sub}}\right)
$$

- $\mathbf{A}_0$：Episode 开始时的**锚点视图**，提供不变的全局参考
- $\mathcal{H}_t^{\text{nbr}}$：最近 $n=2$ 个关键帧（相邻历史）
- $\mathcal{H}_t^{\text{sub}}$：**自适应子目标帧**，由选择器 $D_\phi$ 决定是否保留

自适应选择模块 $D_\phi$ 判断当前观测是否提供了历史帧之外的新信息，若否则放弃，避免冗余 token。

时间记忆注入：[[交叉注意力]]

$$
\tilde{\mathbf{Z}}_t^c = F_{\text{temp}}\!\left(\mathbf{Z}_t^c,\; \mathcal{T}_t\right)
$$

#### 模块 4：空间记忆 $\mathcal{S}_t$

**设计动机**：细化阶段需要精确定位，但夹爪或已拾起的物体会遮挡目标区域。

**具体实现**：
- 在 Episode 开始时存储初始点云 $\mathbf{P}_0$（交互前无遮挡）
- 推理时以**相同 Zoom 配置**重渲染 $\mathbf{P}_0$，提供无遮挡的参考几何

![Figure 3: Spatial Memory Occlusion](https://arxiv.org/html/2608.05042v1/x3.png)

**说明**：左为当前被遮挡的细化裁剪图，右为空间记忆提供的同区域无遮挡参考图（初始点云重渲染）。

$$
\mathcal{S}_t = \Phi\!\left(\text{Render}\!\left(\text{Zoom}(\mathbf{P}_0;\, \hat{\mathbf{x}}_t^c)\right)\right)
$$

$$
\tilde{\mathbf{Z}}_t^f = F_{\text{spa}}\!\left(\mathbf{Z}_t^f,\; \mathcal{S}_t\right)
$$

其中 $\Phi(\cdot)$ 为 VLM 视觉编码器，$\hat{\mathbf{x}}_t^c$ 为粗阶预测位置。

#### 模块 5：双臂扩展

**具体实现**：
- 共享 VLM Backbone 和记忆模块（场景级表征共享）
- 独立的 Convex-Upsampling 和 Action Head（各臂独立细化）

$$
\mathbf{a}_t^{\text{bi}} = \left(\mathbf{a}_t^{\text{left}},\; \mathbf{a}_t^{\text{right}}\right)
$$

---

## 关键公式

### 公式 1：[[动作表示|动作元组]]

$$
\mathbf{a}_t = \left(\mathbf{x}_t,\; \mathbf{R}_t,\; g_t,\; c_t\right)
$$

**含义**：机器人动作由端效器 3D 位置 $\mathbf{x}_t$、旋转矩阵 $\mathbf{R}_t$、夹爪开合标志 $g_t$、碰撞标志 $c_t$ 组成

**符号说明**：
- $\mathbf{x}_t \in \mathbb{R}^3$：端效器三维位置
- $\mathbf{R}_t \in SO(3)$：旋转，以离散化欧拉角表示
- $g_t \in \{0,1\}$：夹爪开合
- $c_t \in \{0,1\}$：是否碰撞

---

### 公式 2-3：[[热图动作解码|截断高斯热图构建]]

$$
H_i^{\text{gt}}(\mathbf{x}) = \begin{cases} p_i(\mathbf{x}) & \text{if } p_i(\mathbf{x}) \geq p_{\min} \\ 0 & \text{otherwise} \end{cases}
$$

$$
p_i(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x} - \hat{\mathbf{x}}_i\|_2^2}{2\sigma^2}\right)
$$

**含义**：以目标位置 $\hat{\mathbf{x}}_i$ 为中心构建截断高斯分布作为监督热图，截断阈值 $p_{\min}$ 防止远处像素干扰

**符号说明**：
- $\hat{\mathbf{x}}_i$：第 $i$ 个目标的真实 3D 位置
- $\sigma$：高斯宽度超参数
- $p_{\min}$：截断阈值

---

### 公式 4-5：[[热图动作解码|多目标热图归一化]]

$$
H_{\text{avg}}(\mathbf{x}) = \frac{1}{N_{\text{obj}}} \sum_{i=1}^{N_{\text{obj}}} H_i^{\text{gt}}(\mathbf{x})
$$

$$
H^{\text{gt}}(\mathbf{x}) = \frac{H_{\text{avg}}(\mathbf{x})}{\sum_{\mathbf{x}' \in \Omega} H_{\text{avg}}(\mathbf{x}')}
$$

**含义**：多目标场景下对各目标热图平均后归一化为概率分布

**符号说明**：
- $N_{\text{obj}}$：场景中目标数量
- $\Omega$：图像像素域

---

### 公式 6：[[交叉熵损失|预训练对齐损失]]

$$
\mathcal{L}_{\text{pre}} = -\sum_{\mathbf{x} \in \Omega} H^{\text{gt}}(\mathbf{x}) \log \hat{H}(\mathbf{x})
$$

**含义**：最小化预测热图 $\hat{H}$ 与 GT 热图 $H^{\text{gt}}$ 之间的交叉熵，实现语言条件目标定位预训练

**符号说明**：
- $\hat{H}(\mathbf{x})$：模型预测的归一化热图
- $H^{\text{gt}}(\mathbf{x})$：截断高斯 GT 热图

---

### 公式 7-8：[[热图动作解码|多视图位移聚合]]

$$
s_t(\mathbf{x}) = \sum_{v=1}^{V} \hat{H}_{t,v}\!\left(\Pi_v(\mathbf{x})\right), \quad V = 3
$$

$$
\hat{\mathbf{x}}_t = \arg\max_{\mathbf{x}}\; s_t(\mathbf{x})
$$

**含义**：将三个正交视图的热图响应投影到 3D 空间并聚合，argmax 得到预测的端效器位置

**符号说明**：
- $\Pi_v(\mathbf{x})$：将 3D 点 $\mathbf{x}$ 投影到第 $v$ 个正交视图的 2D 坐标
- $\hat{H}_{t,v}$：第 $v$ 个视图的预测热图
- $V = 3$：top、front、right 三视图

---

### 公式 9：[[粗到细动作精化|微调总损失]]

$$
\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{trans}} + \mathcal{L}_{\text{rot}} + \mathcal{L}_{\text{gripper}} + \mathcal{L}_{\text{collision}}
$$

**含义**：微调阶段联合优化平移、旋转、夹爪、碰撞四项预测

---

### 公式 10：[[时空记忆|记忆条件策略]]

$$
\mathbf{a}_t = \pi_M\!\left(\mathbf{o}_t,\; l,\; \mathcal{M}_t\right), \quad \mathcal{M}_t = \left(\mathcal{T}_t,\; \mathcal{S}_t\right)
$$

**含义**：BridgeVLA++ 的策略同时以当前观测、语言指令和时空记忆为条件

**符号说明**：
- $\mathcal{T}_t$：时间记忆（锚点帧 + 邻近历史帧 + 子目标帧）
- $\mathcal{S}_t$：空间记忆（初始点云重渲染）

---

### 公式 11：[[时空记忆|空间记忆构建]]

$$
\mathcal{S}_t = \Phi\!\left(\text{Render}\!\left(\text{Zoom}(\mathbf{P}_0;\; \hat{\mathbf{x}}_t^c)\right)\right)
$$

**含义**：对初始点云 $\mathbf{P}_0$ 按粗阶预测位置 Zoom 后重渲染，经 VLM 编码得到空间记忆 token

**符号说明**：
- $\mathbf{P}_0$：Episode 开始时记录的初始彩色点云
- $\hat{\mathbf{x}}_t^c$：粗阶预测的 3D 位置
- $\Phi$：VLM 视觉编码器

---

### 公式 12：完整训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{base}} + \lambda_{\text{check}}\, \mathcal{L}_{\text{check}}
$$

**含义**：在基础操纵损失上加入 Checkpoint 一致性正则项

**符号说明**：
- $\lambda_{\text{check}}$：正则权重
- $\mathcal{L}_{\text{check}}$：记忆模块的一致性约束损失

---

## 关键图表

### Figure 1: Overview / 系统概览

![BridgeVLA++ Overview](https://arxiv.org/html/2608.05042v1/x1.png)

**说明**：BridgeVLA++ 整体概览。点云经三视图[[正交投影多视图|正交投影]]输入 VLM，粗阶预测 3D 热图定位，细阶局部精化；BridgeVLA++ 在粗/细阶分别注入[[时间记忆]]和[[空间记忆]]。右侧 Radar Chart 展示在多个 benchmark 上的综合性能。

### Figure 2: Model Architecture / 模型架构

![BridgeVLA++ Architecture](https://arxiv.org/html/2608.05042v1/x2.png)

**说明**：上方为 BridgeVLA 两阶段训练流程（RoboPoint 预训练 → 3D 操纵微调）。下方展示 BridgeVLA++ 的时间记忆注入（Cross-Attention 注入粗阶 token）和空间记忆注入（重渲染初始点云注入细阶 token）的详细结构。

### Figure 3: Spatial Memory Occlusion Robustness / 空间记忆遮挡鲁棒性

![Spatial Memory](https://arxiv.org/html/2608.05042v1/x3.png)

**说明**：左图为当前细化阶段的裁剪图，夹爪和被操纵物体遮挡了目标区域；右图为[[空间记忆]]提供的同一区域无遮挡初始几何，模型可以"看到"被遮挡的目标。

### Figure 4: Real-Robot Evaluation Setup / 真实机器人评估

![Real Robot Setup](https://arxiv.org/html/2608.05042v1/x4.png)

**说明**：左上为 Franka Research 3（7-DoF）+ ZED 2i 立体相机的通用操纵平台；右侧为 Dobot CR5A 记忆依赖任务平台。两个平台均只用 10 次示范训练，验证跨平台可迁移性。

### Project Teaser: 整体效果展示

![Teaser](https://bridgevla-plus.github.io/static/images/paper/teaser.png)

**说明**：项目主页 Teaser 图，展示多视图热图对齐思想、记忆架构的雷达图综合对比以及真实机器人泛化设置。

---

### Table 1: RLBench 18 任务成功率及消融（TABLE I）

| Method | Avg SR (%) | Avg Rank | Close Jar | Drag Stick | Insert Peg | Meat off Grill | Open Drawer | Place Cups | Place Wine | Push Buttons |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PerAct | 49.4 | 11.33 | 55.2 | 89.6 | 5.6 | 70.4 | 88.0 | 2.4 | 44.8 | 92.8 |
| Act3D | 65.0 | 9.17 | 92.0 | 92.0 | 27.0 | 94.0 | 93.0 | 3.0 | 80.0 | 99.0 |
| RVT | 62.9 | 9.08 | 52.0 | 99.2 | 11.2 | 88.0 | 71.2 | 4.0 | 91.0 | 100.0 |
| 3D Diffuser Actor | 81.3 | 6.19 | 96.0 | 100.0 | 65.6 | 96.8 | 89.6 | 24.0 | 93.6 | 98.4 |
| RVT-2 | 81.4 | 5.97 | 100.0 | 99.0 | 40.0 | 99.0 | 74.0 | 38.0 | 95.0 | 100.0 |
| SAM2Act | 86.8 | 5.47 | 99.0 | 99.0 | 84.0 | 98.0 | 83.0 | 47.0 | 93.0 | 100.0 |
| **BridgeVLA** | **90.5** | **4.75** | **100.0** | 97.6 | **91.2** | **100.0** | **99.2** | **58.4** | 89.6 | **100.0** |
| w/ discretized rot | 88.2 | 4.86 | 100.0 | 100.0 | 88.0 | 100.0 | 100.0 | 58.4 | 88.0 | 98.4 |
| w/o heatmap decoding | 31.4 | 12.78 | 49.3 | 65.3 | 0.0 | 81.3 | 74.7 | 1.3 | 32.0 | 54.7 |
| w/ 3D position input | 56.2 | 10.14 | 96.0 | 58.7 | 26.7 | 96.0 | 97.3 | 14.7 | 81.3 | 86.7 |
| **BridgeVLA++** | **93.7** | **3.64** | **100.0** | **98.4** | **99.2** | **100.0** | **99.2** | **76.8** | **95.2** | **100.0** |
| w/o spatial memory | 92.0 | 3.81 | 100.0 | 100.0 | 95.2 | 100.0 | 99.2 | 74.4 | 78.4 | 100.0 |
| w/o temporal memory | 91.9 | 3.81 | 100.0 | 99.2 | 82.4 | 100.0 | 97.6 | 57.6 | 90.4 | 100.0 |

| Method | Put Cupboard | Put Drawer | Put Safe | Screw Bulb | Slide Block | Sort Shape | Stack Blocks | Stack Cups | Sweep Dustpan | Turn Tap |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PerAct | 28.0 | 51.2 | 84.0 | 17.6 | 74.0 | 16.8 | 26.4 | 2.4 | 52.0 | 88.0 |
| Act3D | 51.0 | 90.0 | 95.0 | 47.0 | 93.0 | 8.0 | 12.0 | 9.0 | 92.0 | 94.0 |
| RVT | 49.6 | 88.0 | 91.2 | 48.0 | 81.6 | 36.0 | 28.8 | 26.4 | 72.0 | 93.6 |
| 3D Diffuser Actor | 85.6 | 96.0 | 97.6 | 82.4 | 97.6 | 44.0 | 68.3 | 47.2 | 84.0 | 99.2 |
| RVT-2 | 66.0 | 96.0 | 96.0 | 88.0 | 92.0 | 35.0 | 80.0 | 69.0 | 100.0 | 99.0 |
| SAM2Act | 75.0 | 99.0 | 98.0 | 89.0 | 86.0 | 64.0 | 76.0 | 78.0 | 99.0 | 96.0 |
| **BridgeVLA** | **91.2** | **96.0** | 95.2 | **93.6** | **95.2** | 55.2 | **84.8** | **88.8** | **100.0** | 92.8 |
| w/ discretized rot | 73.6 | 99.2 | 99.2 | 87.2 | 96.0 | 60.8 | 76.8 | 81.6 | 87.2 | 92.8 |
| w/o heatmap decoding | 5.3 | 0.0 | 58.7 | 2.7 | 64.0 | 4.0 | 0.0 | 0.0 | 32.0 | 40.0 |
| w/ 3D position input | 10.7 | 78.7 | 97.3 | 16.0 | 72.0 | 21.3 | 17.3 | 4.0 | 53.3 | 84.0 |
| **BridgeVLA++** | **92.0** | **99.2** | 92.8 | **95.2** | **96.0** | **72.0** | **85.6** | **98.4** | **97.6** | **89.6** |
| w/o spatial memory | 90.4 | 91.2 | 96.0 | 95.2 | 97.6 | 60.8 | 91.2 | 92.8 | 99.2 | 95.2 |
| w/o temporal memory | 88.8 | 98.4 | 92.8 | 96.0 | 100.0 | 73.6 | 88.8 | 93.6 | 100.0 | 94.4 |

**关键发现**：
- 去除热图解码（w/o heatmap decoding）：90.5% → 31.4%，灾难性失效，证明[[热图动作解码]]是核心
- 去除 2D 预训练对齐（w/ 3D position input）：90.5% → 56.2%，严重退化，证明 RoboPoint 预训练不可缺
- 去除空间记忆：93.7% → 92.0%，无遮挡任务影响有限，遮挡任务明显退化
- 去除时间记忆：93.7% → 91.9%，记忆依赖任务（Insert Peg、Place Cups）退化显著

---

### Table 2: RMBench 双臂记忆依赖任务（TABLE II）

| Method | M(1) Avg (%) | Observe Pick | Rearrange | Put Back | Swap Block | Swap Blocks | Battery T | M(n) Avg (%) | Blocks Ranking | Cover Blocks | Press Button | Overall (%) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DP | 5.8 | 1 | 0 | 0 | 11 | 20 | 6.4 | 10 | 10 | 0 | 0 | 5.0 |
| ACT | 5.9 | 1 | 29 | 0 | 2 | 2 | 6.8 | 19 | 0 | 0 | 0 | 4.8 |
| π₀.₅ | 10.4 | 9 | 13 | 11 | 24 | 15 | 14.4 | 16 | 6 | 0 | 0 | 5.5 |
| X-VLA | 9.8 | 9 | 13 | 18 | 16 | 3 | 11.8 | 26 | 1 | 2 | 0 | 7.3 |
| Mem-0 | 42.0 | 4 | 89 | 90 | 67 | 14 | 52.8 | 28 | 18 | 68 | 0 | 28.5 |
| Fast-WAM | 5.9 | 0 | 0 | 0 | 0 | 7 | 1.4 | 20 | 26 | 0 | 0 | 11.5 |
| LingBot-VA | 78.2 | 13 | 100 | 100 | 99 | 88 | 80.0 | 41 | 100 | 79 | 84 | 76.0 |
| MemoryWAM | 83.0 | 27 | 100 | 100 | 100 | 94 | 84.2 | 41 | 100 | 98 | 87 | 81.5 |
| **BridgeVLA** | **18.9** | 75 | 0 | 1 | 11 | 8 | 19.0 | 72 | 0 | 3 | 0 | 18.8 |
| **BridgeVLA++** | **96.0** | **81** | **100** | **100** | **99** | **96** | **95.2** | **96** | **100** | **99** | **93** | **97.0** |

**关键发现**：BridgeVLA（无记忆）在 RMBench 上仅 18.9%，而 BridgeVLA++ 达 96.0%，超越 MemoryWAM（81.5%）14.5 个百分点。说明[[时空记忆]]对跨帧推理任务是决定性的。

---

### Table 3: COLOSSEUM 泛化能力（TABLE III）

| Method | Avg SR (%) | Avg Rank | All Perturb | MO Color | RO Color | MO Texture | RO Texture | MO Size | RO Size | Light Color | Table Color | Table Texture | BG Distractor | Camera | RLBench | Pose |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PerAct | 27.9 | 4.71 | 7.2 | 24.0 | 29.2 | 28.8 | 17.7 | 35.6 | 29.3 | 29.1 | 30.4 | 23.2 | 27.1 | 33.5 | 39.4 | 36.3 |
| RVT | 35.4 | 4.29 | 6.4 | 26.0 | 31.3 | 44.8 | 41.1 | 35.3 | 40.5 | 34.0 | 30.0 | 45.2 | 18.8 | 46.4 | 53.4 | 42.2 |
| RVT-2 | 56.7 | 2.86 | 15.6 | 53.0 | 54.6 | 59.7 | 56.7 | 60.9 | 53.4 | 58.0 | 62.6 | 56.6 | 60.8 | 68.7 | 68.8 | 64.4 |
| **BridgeVLA** | **64.0** | **1.50** | **18.7** | **60.5** | **63.8** | **63.5** | **68.4** | **69.3** | **61.7** | **69.7** | **75.7** | **71.3** | 51.8 | **74.8** | **73.1** | **73.8** |
| **BridgeVLA++** | **65.2** | 1.64 | **38.9** | 68.7 | 62.7 | 65.7 | 65.5 | 71.5 | 62.0 | 68.2 | 71.5 | 69.2 | **61.6** | 69.5 | 68.5 | 68.7 |

**关键发现**：BridgeVLA 在 COLOSSEUM 12 项干扰轴中的大多数上超越 RVT-2（~7.3 个百分点）。BridgeVLA++ 在背景干扰（BG Distractor）和全干扰（All Perturb: 18.7→38.9）上有显著提升。

---

### Table 4: 真实机器人 Franka（10 demos, 13 任务，TABLE IV 部分）

| Method | Avg SR (%) | Soda Can (Bottom) | Giraffe Drawer | Red Block Blue Plate | Press Sanitizer | RedBull Top | RedBull Bottom |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SpatialVLA (50 demos) | 28.5 | 1/10 | 1/10 | 5/10 | 6/10 | 3/10 | 1/10 |
| SpatialVLA (10 demos) | 3.1 | 0/10 | 0/10 | 0/10 | 2/10 | 0/10 | 0/10 |
| π₀.₅ | 20.0 | 2/10 | 1/10 | 4/10 | 4/10 | 1/10 | 1/10 |
| ACT | 21.5 | 2/10 | 2/10 | 3/10 | 2/10 | 3/10 | 1/10 |
| RVT-2 | 90.0 | 10/10 | 8/10 | 8/10 | 10/10 | 9/10 | 10/10 |
| **BridgeVLA (3 demos)** | **95.4** | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 |
| **BridgeVLA (10 demos)** | **96.2** | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 |

**关键发现**：BridgeVLA 仅用 3 次示范达 95.4%，超越使用 50 次示范的 SpatialVLA（28.5%）近 67 个百分点，展现极强的数据效率。

---

## 实验

### 数据集与 Benchmark

| 数据集 / Benchmark | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboPoint]] | 120K 样本 | 语言条件 2D 目标定位 | 预训练 |
| [[RLBench]] | 18 任务，100 demo/task | 仿真操纵，多种精细任务 | 训练 + 测试 |
| [[COLOSSEUM]] | 14 扰动轴 | 系统性视觉泛化评估 | 测试（泛化） |
| [[GemBench]] | 4 泛化级别 | 新物体-技能组合泛化 | 测试（新组合） |
| [[RMBench]] | 9 双臂任务 | 记忆依赖、双臂 | 测试（记忆） |
| [[MemoryBench]] | 3 任务 | 单臂记忆依赖 | 测试（记忆） |

### 实现细节

- **Backbone**：[[PaliGemma]]（SigLIP-400M 视觉编码器 + Gemma-2B 语言模型），总参数 ~2.92B
- **视图数量**：3（top / front / right 正交投影）
- **历史帧数**：$n = 2$（时间记忆相邻帧）
- **参数增量**：BridgeVLA++ 新增 269.77M（时间记忆 168M + 空间记忆 84M）
- **推理延迟**：BridgeVLA 0.35s / step → BridgeVLA++ 0.57s / step
- **训练数据**：每任务 100 条 demo（仿真），真实 3-10 条 demo

### GemBench 和 MemoryBench 结果

- **GemBench**：BridgeVLA 达 **50.0%** avg SR（Appendix Table XI 详细数据）
- **MemoryBench**（单臂）：BridgeVLA++ 达 **99.7 ± 0.3%** avg SR

---

## 批判性思考

### 优点

1. **概念优雅**：将 3D 操纵"降维"到 VLM 原生 2D 空间，热图接口既精确又与预训练兼容
2. **数据效率极高**：3 次真实示范即达工业级精度，对实际部署极具价值
3. **记忆架构轻量**：不改动 VLM 核心结构，以 9.2% 参数代价在记忆任务上翻 5 倍

### 局限性

1. **推理延迟上升**：0.35s → 0.57s，在需要快速响应的任务中可能不可接受
2. **严重遮挡仍失败**：即使有空间记忆，极端遮挡场景仍是主要失败模式
3. **TABLE IV 数据未完全公开**：论文中提及 13 个任务，但公开 HTML 版本只展示部分

### 潜在改进方向

1. 探索更高效的记忆压缩方案（如 KV cache 共享）降低推理延迟
2. 引入 NeRF / 3DGS 做更精确的遮挡感知几何推断
3. 扩展到需要工具使用的复杂多步骤规划

### 可复现性评估

- [ ] 代码开源（项目页有链接，截至笔记日期未完全公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（Appendix 有详细描述）
- [x] 数据集可获取（RLBench、COLOSSEUM 等均公开）

---

## 关联笔记

### 基于

- [[BridgeVLA]]（原始论文）：提出热图 + 正交投影的基础框架
- [[PaliGemma]]：视觉语言模型 Backbone

### 对比

- [[RVT-2]]：多视图 VLA baseline，BridgeVLA 在 COLOSSEUM 超越 7.3%
- [[SAM2Act]]：RLBench SOTA baseline，BridgeVLA 超越 3.7%
- [[LingBot-VLA-2.0]]：RMBench 强基线，BridgeVLA++ 超越 21 个百分点
- [[MemoryWAM]]：记忆增强 WAM，BridgeVLA++ 超越 14.5 个百分点

### 方法相关

- [[热图动作解码]]：核心动作接口
- [[正交投影多视图]]：3D→2D 转换方案
- [[时空记忆]]：BridgeVLA++ 的核心扩展
- [[粗到细动作精化]]：两阶段定位策略

### 数据集相关

- [[RoboPoint]]：预训练数据集
- [[RLBench]]：主要训练/测试 benchmark
- [[COLOSSEUM]]：泛化能力评估
- [[GemBench]]：新组合泛化评估
- [[RMBench]]：记忆依赖双臂任务

---

## 速查卡片

> [!summary] BridgeVLA++
> - **核心**：VLM + 三视图正交投影热图 + 时空记忆，解决 3D 操纵的数据效率、泛化与记忆依赖三重挑战
> - **方法**：点云→三视图正交图→VLM 热图预测→多视图聚合→3D 端效器位置；时间记忆（锚点+历史+子目标帧）注入粗阶，空间记忆（初始点云重渲染）注入细阶
> - **结果**：RLBench 93.7%、RMBench 96.0%（原 18.9%→96.0%）、Franka 3-demo 95.4%
> - **代码**：[bridgevla-plus.github.io](https://bridgevla-plus.github.io/)

---

*笔记创建时间: 2026-08-07*

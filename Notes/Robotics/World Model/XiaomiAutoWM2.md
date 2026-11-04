---
title: "Xiaomi Auto World Model 2.0: Causal World Models for Real-World Driving and Embodied Intelligence"
method_name: "XiaomiAutoWM2"
authors: [Zhenxin Zhu, Zhijian Huang, Cheng Chi, Yuru Wang, Jianhua Wu, Jiayi Guan, Yonggang Li, Mingfei Tu, JiaKui Hu, Tao Xu, Xianqi Wang, Hao Li, Yuechuan Pu, Qizeng Jia, Ruiling Yang, Hongxiao Yu, Xuhan Sheng, Jian Mao, Ruiqiang Zhang, Yu Zhu, Guang Chen, Hangjun Ye, Bing Wang]
year: 2026
venue: arXiv
tags: [world-model, autonomous-driving, embodied-ai, 3d-gaussian-splatting, video-generation, causal-attention]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.01677v1
created: 2026-11-04
---

# 论文笔记：Xiaomi Auto World Model 2.0

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Xiaomi Auto |
| 日期 | November 2026 |
| 项目主页 | [xiaomiautoworldmodel.github.io/CausalWorldModel](https://xiaomiautoworldmodel.github.io/CausalWorldModel/) |
| 对比基线 | [[GigaWorld1]] / [[Cosmos3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.01677) / [PDF](https://arxiv.org/pdf/2611.01677) |

---

## 一句话总结

> 在 Wan2.2-5B 基础上，提出统一流式因果世界模型，以 [[View-Temporal Attention]] + [[GLA|Gated Linear Attention]] 实现高效多视角生成，以 [[WorldRec|WorldRec 2.0]] 提供 3D 高斯先验，同时支持自动驾驶闭环仿真、闭环训练（RL）和世界动作模型三大应用。

---

## 核心贡献

1. **流式因果架构**: 结合 [[View-Temporal Attention]]（多视角时间注意力）与 [[GLA]]（门控线性注意力），将 DiT 推理时延从 6.48 s/chunk 压缩至 1.74 s/chunk（3.7×加速），时序 KV cache 减少 97%+
2. **WorldRec 2.0**: 单次前向同时重建静态和动态 [[3DGS|3D 高斯]]，渲染几何先验作为 World Control 注入生成管线，支持长程一致性
3. **三大 Physical AI 应用**: 闭环仿真（PAI-Bench-G SOTA）、基于 [[PPO]] 的闭环强化训练（NavSim SOTA）、以及融合世界模型的 [[WAM|World Action Model]]（LIBERO-Plus 75.0% 无需具身预训练）

---

## 问题背景

### 要解决的问题

现有世界模型大多仅支持单视角、短时序、单一领域（自动驾驶或机器人），无法满足 Physical AI 对**多视角**（最多 7 路）、**长程**（45 s+）、**跨域**（通用视频 + 驾驶 + 机器人）生成的需求，且推理速度无法支持实时闭环仿真。

### 现有方法的局限

- DriveDreamer、MagicDrive-V2 等：仅支持固定视角配置，无法灵活扩展到 1–7 路
- Self-Forcing / LongLive：长视频生成质量下降，FID/FVD 均弱于本方法
- 现有 WAM 方法（Fast-WAM、4D-WAM）：操作任务成功率偏低（<72%），或依赖具身预训练

### 本文的动机

以 [[Wan2.2]] 5B 视频扩散基座为起点，通过架构改造（[[View-Temporal Attention]] + [[GLA]] + [[Fused-PRoPE]]）实现流式多视角因果生成，同时用 [[WorldRec|WorldRec 2.0]] 将 3D 几何先验显式注入，从而兼顾质量、速度与几何一致性。

---

## 方法详解

### 模型架构

XiaomiAutoWM2 采用 **流式因果 DiT** 架构，基座为 Wan2.2-5B：

- **输入**: 多视角图像块（1–7 路） + 世界控制（相机轨迹、布局、动作信号、3D 高斯渲染）+ 文本条件
- **Backbone**: [[Wan2.2]] 5B 视频扩散 Transformer
- **核心模块**:
  - [[View-Temporal Attention]]（VTA）：时序与视角注意力交替层，视角数增加时计算量线性增长而非二次
  - [[GLA|Gated Linear Attention]]（GLA）：替换时序层的 Softmax 注意力，配合周期性 Softmax anchor，减少 97%+ KV cache
  - [[Fused-PRoPE]]：融合相机内外参的投影式 RoPE，统一单视角和多视角条件
  - [[WorldRec|WorldRec 2.0]]：3D 高斯重建模块，提供几何先验作为 World Control
- **输出**: 时序一致的多视角视频序列（chunk 形式流式生成）
- **总参数**: ~5B（继承 Wan2.2-5B）

### 核心模块

#### 模块 1：View-Temporal Attention（VTA）

**设计动机**: 全局时空注意力在多视角场景下复杂度为 $O((T \cdot V)^2)$，[[View-Temporal Attention]] 将其拆分为时序层 + 视角层交替，降至 $O(T^2 \cdot V + V^2 \cdot T)$。

**具体实现**:
- 奇数 Transformer 层：仅对时序维度做 [[GLA]]（流式因果，支持递增推理）
- 偶数 Transformer 层：仅对视角维度做完整 Softmax 注意力（视角间无因果约束）
- 视角层引入 [[Fused-PRoPE]] 注入多视角相机几何信息

#### 模块 2：Gated Linear Attention in Temporal Layers

**设计动机**: 长程时序依赖需要大 KV cache，[[GLA]] 以固定大小状态矩阵近似历史，配合门控机制平衡记忆容量与遗忘。

**具体实现**:
- 每头维护 $d \times d$ 状态矩阵（双向线性注意力）
- 每隔若干 token 插入 "Softmax anchor"，锚定当前局部精确注意力，防止误差累积
- 推理时仅保留当前 chunk 的状态，cache 从 $O(T \cdot d)$ 降至 $O(d^2)$

#### 模块 3：WorldRec 2.0

**设计动机**: 长程生成中场景几何一致性难以仅靠像素级自回归维持，[[WorldRec]] 显式重建 3D 高斯场景记忆并渲染为先验，注入 World Control。

**具体实现**:
- 利用 DUSt3R/MASt3R 类框架初始化点云，配合寄存器交互（compress → communicate → broadcast）生成实例级动态高斯
- 单次前向：静态背景高斯 + 动态目标高斯同时输出
- 渲染深度图/语义图/光流拼接入 World Control 通道（Eq. 7）
- 自 WorldRec 1.0 升级：动态 D-RMSE 从 6.07 降至 3.99（Waymo）

---

## 关键公式

### 公式 1：[[Autoregressive Generation|因果自回归分解]]

$$
p(\mathbf{x}) = \prod_{k=1}^{K} p\!\left(\mathbf{x}^{(k)} \mid \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(k-1)},\ \mathbf{w},\ \mathbf{c},\ \mathbf{x}^{(0)},\ \mathbf{t}\right)
$$

**含义**: 将视频分解为 $K$ 个 chunk，每个 chunk 以前序 chunk、世界控制 $\mathbf{w}$、相机位姿 $\mathbf{c}$、首帧 $\mathbf{x}^{(0)}$ 和文本 $\mathbf{t}$ 为条件自回归生成。

**符号说明**:
- $\mathbf{x}^{(k)}$：第 $k$ 个视频 chunk
- $\mathbf{w}$：World Control（布局、动作、3D 渲染先验）
- $\mathbf{c}$：相机位姿序列
- $\mathbf{t}$：文本描述

---

### 公式 2：[[GLA|门控双向线性注意力]]

$$
\text{GLA}(Q, K, V) = \sigma(G) \odot \left( Q \cdot \left( \sum_{j \le i} K_j^\top V_j \right) \right)
$$

**含义**: 以门控矩阵 $\sigma(G)$ 控制历史状态的激活量，每头维护 $d \times d$ 状态，实现 $O(1)$ 时序 cache 的高效递归推理。

**符号说明**:
- $G$：可学习门控参数
- $\sigma$：Sigmoid 激活
- $Q, K, V$：query/key/value 投影
- $d$：头维度，状态矩阵大小 $d \times d$

---

### 公式 3：[[Fused-PRoPE|融合投影式 RoPE]]

$$
P_i = K_f \cdot E_i
$$

**含义**: 以相机内参矩阵 $K_f$（统一焦距）和外参 $E_i$（视角 $i$ 的位姿）构造投影矩阵，注入视角层 RoPE，使模型感知多视角几何关系。

**符号说明**:
- $P_i$：视角 $i$ 的投影矩阵
- $K_f$：统一相机内参矩阵（以统一焦距近似不同相机）
- $E_i$：视角 $i$ 的外参（world-to-camera 变换）

---

### 公式 4：[[Consistency Distillation|一致性蒸馏损失]]

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{\mathbf{x}, t}\left[ \left\| f_\theta(\mathbf{x}_t, t) - f_{\hat{\theta}}(\mathbf{x}_{t-\Delta t}, t-\Delta t) \right\|^2 \right]
$$

**含义**: 要求学生模型在相邻噪声水平的输出保持一致，以 EMA 教师网络 $f_{\hat{\theta}}$ 为目标，实现少步推理蒸馏。

**符号说明**:
- $f_\theta$：学生模型
- $f_{\hat{\theta}}$：EMA 教师模型（指数移动平均）
- $\mathbf{x}_t$：噪声水平 $t$ 的样本
- $\Delta t$：噪声水平步长

---

### 公式 5：动态实例 Query 初始化

$$
q_j^{(i)} = \text{Linear}\!\left(\text{PE}\!\left(T_{\text{world} \to i} \cdot p_j\right)\right)
$$

**含义**: 将动态实例的世界坐标系 3D 位置 $p_j$ 变换到视角 $i$ 的相机坐标系，再编码为 Query，实现跨视角动态实例对齐。

**符号说明**:
- $p_j$：第 $j$ 个动态实例在世界坐标系的 3D 中心位置
- $T_{\text{world} \to i}$：世界坐标系到视角 $i$ 相机坐标系的变换矩阵
- $\text{PE}(\cdot)$：位置编码函数

---

### 公式 6：WorldRec 寄存器交互（Compress → Communicate → Broadcast）

$$
\mathbf{r}' = \text{Broadcast}\!\left(\text{Communicate}\!\left(\text{Compress}(\mathbf{f}_{\text{patch}}, \mathbf{r})\right)\right)
$$

**含义**: 将局部 patch 特征 $\mathbf{f}_{\text{patch}}$ 压缩进寄存器 $\mathbf{r}$，经全局寄存器间通信后广播，实现跨视角跨时序的全局场景信息聚合。

**符号说明**:
- $\mathbf{r}$：寄存器 token（全局场景记忆载体）
- $\mathbf{f}_{\text{patch}}$：局部 patch 视觉特征
- $\text{Compress/Communicate/Broadcast}$：三阶段注意力聚合操作

---

### 公式 7：几何先验渲染注入

$$
\mathbf{w}_t = \text{Cat}\!\left[\text{Render}_{\text{depth}}(\mathcal{G}_t),\ \text{Render}_{\text{sem}}(\mathcal{G}_t),\ \text{Render}_{\text{flow}}(\mathcal{G}_t)\right]
$$

**含义**: 将 WorldRec 2.0 重建的 3D 高斯场景 $\mathcal{G}_t$ 渲染为深度、语义分割和光流三通道，拼接后作为 World Control 注入生成模型。

**符号说明**:
- $\mathcal{G}_t$：时刻 $t$ 的 3D 高斯场景（静态 + 动态）
- $\text{Render}_{\text{depth/sem/flow}}$：可微高斯光栅化渲染器

---

### 公式 8：[[Flow-Matching|Flow Matching]] Stage I 训练目标

$$
\mathbf{x}_\tau = (1-\tau)\mathbf{x}_0 + \tau \boldsymbol{\epsilon}, \quad \mathbf{v}^* = \boldsymbol{\epsilon} - \mathbf{x}_0, \quad \mathcal{L} = \mathbb{E}_{\tau, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\| f_\theta(\mathbf{x}_\tau, \tau) - \mathbf{v}^* \|^2\right]
$$

**含义**: 以线性插值构造 $[0,1]$ 时间轴上的 ODE 轨迹，模型预测速度场 $\mathbf{v}^*$，Stage I 预训练通用视频生成能力。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$：Flow Matching 时间步
- $\mathbf{x}_0$：干净样本
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)$：标准高斯噪声
- $\mathbf{v}^*$：恒定速度场目标

---

## 关键图表

### Figure 1：Teaser — 统一因果世界模型

![Figure 1 Teaser](https://arxiv.org/html/2611.01677v1/teaser.png)

**说明**: XiaomiAutoWM2 以统一的流式因果架构同时覆盖通用视频、自动驾驶和机器人三个域，输入多视角图像 + World Control，输出时序一致的多路视频，以 [[WorldRec|WorldRec 2.0]] 提供 3D 记忆支撑长程生成。

---

### Figure 2：数据流水线

![Figure 2 Data Pipeline](https://arxiv.org/html/2611.01677v1/data_pipeline_v1016.png)

**说明**: 多源数据（通用视频、驾驶多路相机、机器人操作）经过采集→过滤→打分→自动标注（字幕 + 相机位姿 + 条件准备）流水线，构建覆盖 Physical World 和 Physical Causal 的异构数据集。

---

### Figure 3：数据集分布分析（a-d）

**说明**: (a) 各来源样本数量分布；(b) 字幕 token 数分布；(c) 通用视频场景分布；(d) 驾驶数据场景分布（包含长尾天气/时间段覆盖）。展示训练数据的多样性与覆盖均衡性。

---

### Figure 4：整体架构与三阶段训练概览

![Figure 4 Main Architecture](https://arxiv.org/html/2611.01677v1/main.png)

**说明**: 左侧为 Causal DiT 架构，[[View-Temporal Attention]] 层与 [[GLA]] 层交替排列；右侧为三阶段训练：Stage I（通用视频预训练，[[Flow-Matching]]）→ Stage II（多视角配置中训练，渐进式条件课程）→ Stage III（后训练，[[Consistency Distillation]] + [[Streaming DMD]]）。

---

### Figure 5：WorldRec 2.0 点对齐交互与 3D 场景记忆

![Figure 5 WorldRec2](https://arxiv.org/html/2611.01677v1/WorldRec2_MAIN.png)

**说明**: WorldRec 2.0 以寄存器（Compress → Communicate → Broadcast）实现跨视角全局聚合，单次前向输出静态背景高斯 + 动态目标高斯，对比 WorldRec 1.0 新增动态实例解耦，动态 D-RMSE 从 6.07 降至 3.99（Waymo）。

---

### Figure 6：长尾角落案例

![Figure 6 Corner Cases](https://arxiv.org/html/2611.01677v1/5_1_corner.png)

**说明**: 展示模型对长尾驾驶场景的生成质量，包括城市道路中的马匹、夜间降雪等极端情况，说明数据覆盖和模型鲁棒性。

---

### Figure 7：物理因果事件生成

![Figure 7 Physical Causality](https://arxiv.org/html/2611.01677v1/5_1_physical.png)

**说明**: 展示模型对物理因果序列的建模能力，包括落石规避和追尾碰撞等事件，验证 "Physical Causal Data" 训练数据分支的有效性。

---

### Figure 8：深度预测与指称分割可视化

![Figure 8 Visual Tasks](https://arxiv.org/html/2611.01677v1/5_1_2_visual.png)

**说明**: 以生成视频为输入，模型可直接预测深度图（AbsRel 0.058，δ1 96.3%，Table 6 最优）和指称分割结果，说明 World Control 信号中包含丰富几何语义信息。

---

### Figure 9：单视角与多视角相机位姿控制

![Figure 9 Camera Control](https://arxiv.org/html/2611.01677v1/5_1_3_visual.png)

**说明**: 展示 [[Fused-PRoPE]] 相机控制的效果，支持从单路到多路（最多 7 路）的位姿精确跟随，WorldScore 86.35（Table 7 最优）。

---

### Figure 10：7 路 45 秒长程驾驶视频生成

![Figure 10 45s Generation](https://arxiv.org/html/2611.01677v1/simulation_midata_45s.png)

**说明**: 利用 [[WorldRec|WorldRec 2.0]] 3D 记忆支撑，XiaomiAutoWM2 可生成 7 路视角一致、时序连续的 45 秒驾驶视频，验证长程生成能力。

---

### Figure 11：新轨迹泛化生成

![Figure 11 Novel Trajectories](https://arxiv.org/html/2611.01677v1/simulation_midata_3traj.png)

**说明**: 给定同一起始场景，沿三条不同规划轨迹生成对应视频，演示闭环仿真的轨迹泛化能力，支持 Counterfactual 场景探索。

---

### Figure 12：机器人世界控制与多视角输出

![Figure 12 Robot Simulation](https://arxiv.org/html/2611.01677v1/simulation_robot.png)

**说明**: 机器人操作场景下，以末端执行器轨迹为 World Control，模型生成多视角机械臂操作视频，支持 [[WAM|World Action Model]] 应用。

---

### Figure 13：闭环强化训练框架

![Figure 13 WM-RL Framework](https://arxiv.org/html/2611.01677v1/WM_RL_framework.png)

**说明**: 以 XiaomiAutoWM2 为仿真器，在其中用 [[PPO]] 训练端到端自动驾驶策略。奖励函数综合进度、可驾驶区域、TTC、碰撞、偏离道路和舒适度六项，训练后 NavSim 碰撞率降至 0.4%（Table 10 最优）。

---

### Figure 14：闭环仿真 — 无信号路口场景对比

| Baseline | Ours |
|----------|------|
| ![Figure 14a Baseline](https://arxiv.org/html/2611.01677v1/clossed_loop_709_scene_dark_text_new_baseline.png) | ![Figure 14b Ours](https://arxiv.org/html/2611.01677v1/clossed_loop_709_scene_dark_ours.png) |

**说明**: 无信号路口场景，Baseline 策略（左）行驶失败，本方法（右）成功通过，展示世界模型提升后的策略鲁棒性。

---

### Figure 15：闭环仿真 — 停靠车辆遮挡场景对比

| Baseline | Ours |
|----------|------|
| ![Figure 15a Baseline](https://arxiv.org/html/2611.01677v1/clossed_loop_13_scene_dark_text_new_baseline.png) | ![Figure 15b Ours](https://arxiv.org/html/2611.01677v1/clossed_loop_13_scene_dark_ours.png) |

**说明**: 停靠车辆形成遮挡，Baseline 策略发生碰撞，本方法预判并安全绕过，体现 [[WorldRec|WorldRec 2.0]] 3D 感知带来的空间推理优势。

---

### Figure 16：World Action Model 架构

![Figure 16 WAM Architecture](https://arxiv.org/html/2611.01677v1/figures/WAM_MAIN.jpg)

**说明**: WAM 模块将 XiaomiAutoWM2 的未来视频预测能力与动作策略解码器联合，模型既生成未来帧用于价值估计，又直接解码动作，实现感知→预测→决策一体化。

---

### Figure 17：WAM 候选轨迹评估

![Figure 17 WAM Trajectories](https://arxiv.org/html/2611.01677v1/figures/WAM_TRAJ.jpg)

**说明**: 给定多条候选轨迹，WAM 以世界模型展开各轨迹对应的未来视频，依据生成质量和价值估计排序，选择最优轨迹执行，实现模型预测控制（MPC）风格规划。

---

### Figure 18：Waymo 和 nuScenes 重建结果

![Figure 18 WorldRec2 Waymo](https://arxiv.org/html/2611.01677v1/figures/WorldRec2_waymo.jpg)

**说明**: WorldRec 2.0 在 Waymo 上动态 PSNR 25.81、SSIM 0.773，在 nuScenes 零样本 PSNR 28.37，细节与深度精度均优于 WorldRec 1.0 基线（Table 13-14）。

---

### Figure 19：MiData 7 路重建

![Figure 19 WorldRec2 MiData](https://arxiv.org/html/2611.01677v1/figures/WorldRec2_MIDATA1.jpg)

**说明**: 在小米自有数据 MiData 上展示 7 路相机的完整场景重建，包括前视、侧视和后视摄像头，验证 [[WorldRec|WorldRec 2.0]] 对多相机配置的支持。

---

### Figure 20：MiData 新视角合成

![Figure 20 WorldRec2 MiData2](https://arxiv.org/html/2611.01677v1/figures/WorldRec2_MIDATA2.jpg)

**说明**: 以重建的 3D 高斯场景为基础，合成训练中未见的新视角图像，验证几何重建的精度与视角泛化能力。

---

## 关键图表（表格）

### Table 1：世界模型能力对比

| 方法 | 时长 | 视角数 | 控制类型 | 领域 | 应用 | 实时 |
|------|------|--------|----------|------|------|------|
| Vista | 短 | 1 | 相机 | 驾驶 | 仿真 | 否 |
| DriveDreamer | 短 | 6 | 相机+动作 | 驾驶 | 仿真 | 否 |
| MagicDrive-V2 | 短 | 6 | 布局+相机 | 驾驶 | 仿真 | 否 |
| **XiaomiAutoWM2 (Ours)** | **长** | **1–7** | **相机+动作+布局** | **驾驶+机器人** | **仿真+训练+WAM** | **是** |

**说明**: 本方法是唯一同时支持灵活视角数（1–7）、长程视频、跨域（驾驶+机器人）和实时推理的系统。

---

### Table 2：三阶段训练课程

| 阶段 | 名称 | 数据 | 目标 |
|------|------|------|------|
| Stage I | 预训练 | 通用视频 + 驾驶 + 机器人 | Flow Matching 视频生成 |
| Stage II | 中训练 | 渐进式条件课程（1→7 路）| 多视角配置适应 + 条件注入 |
| Stage III | 后训练 | 高质量子集 | Consistency Distillation + Streaming DMD 少步推理 |

---

### Table 3：注意力设计消融（延迟 / FID）

| 设计 | 延迟（s/chunk）| FID |
|------|--------------|-----|
| 完整 Softmax 注意力（FSA） | 6.481 | — |
| View-Temporal Attention（VTA） | 2.299 | 37.24 |
| **VTA + GLA（Ours）** | **1.742** | **38.08** |

**关键发现**: GLA 在 FID 仅增加 0.84 的代价下，将推理延迟从 2.299 s 进一步压缩至 1.742 s（约 24%），总体较 FSA 提速 3.72×。

---

### Table 4：PAI-Bench-G 自动驾驶子集

| 方法 | AV-Quality ↑ |
|------|-------------|
| 其他方法（最高）| — |
| **XiaomiAutoWM2 (Ours)** | **78.0** |

**说明**: 本方法在 PAI-Bench-G 自动驾驶质量评测中得分最高。

---

### Table 5：PAI-Bench-G 机器人子集

| 方法 | Robot-Quality ↑ |
|------|----------------|
| 其他方法（最高）| — |
| **XiaomiAutoWM2 (Ours)** | **77.1** |

**说明**: 在机器人操作视频生成质量上也取得最优成绩，验证跨域统一建模的有效性。

---

### Table 6：深度预测（nuScenes / KITTI）

| 方法 | AbsRel ↓ | δ1 ↑ |
|------|---------|------|
| 其他方法 | > 0.058 | < 96.3 |
| **XiaomiAutoWM2 (Ours)** | **0.058** | **96.3** |

**说明**: 世界模型隐式学到了准确的场景几何，无需专门深度监督即可实现 SOTA 深度预测。

---

### Table 7：相机可控性（CamTrol Benchmark）

| 方法 | WorldScore | Combined | Average |
|------|-----------|----------|---------|
| 其他方法 | < 86.35 | < 70.08 | < 78.22 |
| **XiaomiAutoWM2 (Ours)** | **86.35** | **70.08** | **78.22** |

**说明**: [[Fused-PRoPE]] 在相机可控性评测上全面超越基线，平均分 78.22 领先所有对比方法。

---

### Table 8：nuScenes 闭环视频生成（FID ↓ / FVD ↓）

**短视频（<30 帧）**:

| 方法 | 视角数 | FID ↓ | FVD ↓ |
|------|--------|-------|-------|
| Vista | 1 | 6.90 | 89.40 |
| DriveDreamer | 6 | 14.90 | 340.80 |
| MagicDrive-V2 | 6 | 20.91 | 94.84 |
| **Ours** | **1** | **12.09** | **135.13** |
| **Ours** | **6** | **19.62** | **171.73** |

**长视频（211 帧）**:

| 方法 | FID ↓ | FVD ↓ |
|------|-------|-------|
| Self-Forcing | 41.53 | 161.00 |
| Self-Forcing++ | 28.84 | 147.57 |
| LongLive | 29.05 | 161.41 |
| Causal Forcing++ | 25.05 | 126.86 |
| **Ours (1-view)** | **25.57** | **125.74** |
| **Ours (6-view)** | 58.90 | 203.76 |

**说明**: 1 路长视频达到最优 FVD（125.74）和次优 FID（25.57），6 路长视频因视角一致性约束导致 FID 上升，但空间布局更完整。

---

### Table 9：TriWorldBench 具身仿真（TWB-Score ↑）

| 方法 | TWB-Score ↑ | 轨迹准确率 ↑ |
|------|------------|-----------|
| DreamDojo | 51.72 | — |
| Fysiverse-Video | 63.62 | — |
| WoVR_Plus | 65.39 | — |
| BWM | 65.54 | — |
| CausalWM | 66.04 | — |
| **Ours** | 64.79 | **47.25** |

**说明**: 本方法在轨迹准确率上排名第一（47.25），总体 TWB-Score 位居前列。

---

### Table 10：NavSim 闭环强化训练结果

| 方法 | CR(%) ↓ | SCR(%) ↑ | OR(%) ↓ | TTC(s) ↑ | EP(%) ↑ | HDR(%) ↑ |
|------|---------|---------|---------|---------|---------|---------|
| DrivoR-E2E | 1.4 | 80.3 | 0.0 | 0.504 | 81.0 | 99.96 |
| ST-JEPA-E2E | 1.2 | 95.2 | 0.0 | 0.614 | 84.7 | 99.93 |
| **E2E-WM-RL (Ours)** | **0.4** | **96.3** | **0.0** | **0.740** | 84.5 | **99.99** |

**说明**: [[PPO]] 在世界模型仿真器中训练，碰撞率（CR）降至 0.4%，成功率（SCR）96.3%，TTC 安全余量 0.740 s，均为 SOTA。

---

### Table 11：NavSim 规划（WAM 应用）

| 方法 | DAC | EP | PDMS ↑ |
|------|-----|-----|--------|
| Base（无 WM）| 98.7 | 92.8 | 94.9 |
| **+ CausalWM（Ours）** | **99.0** | **93.6** | **95.3** |
| 外部最优方法 | — | — | 94.8 |

**说明**: 世界模型辅助规划使 PDMS 从 94.9 提升至 95.3，超过所有外部方法（次优 94.8）。

---

### Table 12：LIBERO-Plus 操作任务（成功率 ↑）

| 方法 | 具身预训练 | 平均成功率 ↑ |
|------|-----------|------------|
| Fast-WAM | 否 | 49.9% |
| Fast-WAM-Joint | 否 | 62.2% |
| 4D-WAM | 是 | 71.0% |
| GAM | 是 | **85.5%** |
| **XiaomiAutoWM2 WAM (Ours)** | **否** | **75.0%** |

**说明**: 无需具身预训练情况下达到 75.0%，超过所有无预训练基线，接近使用具身预训练的 4D-WAM（71.0%），与 GAM（85.5%）的差距来自预训练数据优势。

---

### Table 13：Waymo 重建结果（WorldRec 2.0）

| 方法 | 动态 PSNR ↑ | 动态 SSIM ↑ | 动态 D-RMSE ↓ | 全图 PSNR ↑ | 全图 SSIM ↑ | 全图 D-RMSE ↓ |
|------|------------|------------|--------------|------------|------------|--------------|
| WorldRec 1.0 | — | — | 6.07 | — | — | — |
| **WorldRec 2.0 (Ours)** | **25.81** | **0.773** | **3.99** | **29.88** | **0.865** | **2.89** |

**说明**: 动态 D-RMSE 从 6.07 降至 3.99（-34%），证明动态实例解耦设计的有效性。

---

### Table 14：跨数据集零样本重建

| 目标数据集 | 零样本 PSNR ↑ | 微调 PSNR ↑ |
|-----------|-------------|-----------|
| nuScenes | 28.37 | 29.22 |
| Argoverse 2 | 26.49 | 28.63 |

**说明**: WorldRec 2.0 对未见数据集具有良好泛化性，零样本 PSNR 均在合理范围，微调后进一步提升。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 通用视频（内部）| 大规模 | 多场景视频，字幕+条件标注 | Stage I 预训练 |
| MiData（小米驾驶数据）| 大规模 | 7 路相机，含长尾场景 | 驾驶生成训练/评测 |
| nuScenes | 1000 场景 | 6 路相机，标准 AV benchmark | 视频生成/深度/控制评测 |
| Waymo | 1000 场景 | 高精激光雷达，标准 AV benchmark | WorldRec 重建评测 |
| Argoverse 2 | — | 多传感器 | WorldRec 跨数据集泛化 |
| NavSim | — | 闭环驾驶仿真 | 强化训练/WAM 规划评测 |
| LIBERO-Plus | — | 机器人操作，6 任务套件 | WAM 操作评测 |
| TriWorldBench | — | 具身世界模型综合 benchmark | 闭环仿真评测 |

### 实现细节

- **Backbone**: Wan2.2-5B（视频扩散 Transformer）
- **注意力设计**: [[View-Temporal Attention]] + [[GLA]] 交替（偶数层 VTA，奇数层 GLA）
- **位置编码**: [[Fused-PRoPE]]（视角层）+ 时序 RoPE（时序层）
- **后训练**: [[Consistency Distillation]] + [[Streaming DMD]] 两步蒸馏
- **WorldRec 2.0**: 单次前向，静态 + 动态高斯联合重建
- **RL 训练**: [[PPO]]，6 项奖励（进度/可驾驶区域/TTC/碰撞/偏离道路/舒适度）

---

## 批判性思考

### 优点

1. **架构设计系统性强**: [[View-Temporal Attention]] + [[GLA]] 组合从理论和实验均证明有效，消融清晰（Table 3）
2. **应用覆盖全面**: 单一模型覆盖仿真、RL 训练、WAM 三个方向，工业落地意义大
3. **WorldRec 2.0 实用性高**: 显式 3D 高斯记忆解决了纯隐式模型的长程一致性问题

### 局限性

1. **6 路长视频 FID 偏高**: Table 8 显示多视角长视频 FID 58.90 远高于 1 路，视角一致性与单视图质量存在 trade-off
2. **与 GAM 的 LIBERO 差距**: 无具身预训练时（75.0%）与有预训练的 GAM（85.5%）仍有 10% 差距，说明数据质量仍是瓶颈
3. **实时性条件苛刻**: "实时"依赖 GLA 的 1.74 s/chunk，实际端到端延迟（包含 WorldRec 渲染）未明确报告

### 潜在改进方向

1. 探索 [[View-Temporal Attention]] 中视角层的 GLA 化，进一步压缩多视角推理延迟
2. 引入具身预训练数据以缩小 WAM 与 GAM 的操作差距
3. WorldRec 2.0 与在线 SLAM 结合，实现真正实时 3D 记忆更新

### 可复现性评估

- [ ] 代码开源（暂未发布）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节基本完整（三阶段课程、超参数部分披露）
- [x] 数据集可获取（nuScenes/Waymo/NavSim/LIBERO-Plus 均公开）

---

## 关联笔记

### 基于

- [[Wan2.2]]: 基座视频扩散 Transformer（5B 参数）
- [[PRoPE]]: 投影式 RoPE，本文 Fused-PRoPE 的设计基础
- [[DMD2]]: Streaming DMD 的前驱蒸馏方法
- [[WorldRec]]: WorldRec 1.0 → 2.0 升级路线

### 对比

- [[GigaWorld1]]: 多视角长程驾驶世界模型对比
- [[Cosmos3]]: 通用物理 AI 世界模型对比
- [[Vista]]: 1 路驾驶世界模型基线
- [[MagicDrive-V2]]: 6 路驾驶条件生成基线

### 方法相关

- [[View-Temporal Attention]]: 核心注意力设计
- [[GLA]]: 门控线性注意力，时序高效化
- [[Fused-PRoPE]]: 多视角相机位置编码
- [[WorldRec]]: 3D 高斯场景记忆
- [[Consistency Distillation]]: 后训练少步推理
- [[Streaming DMD]]: 流式蒸馏推理
- [[Flow-Matching]]: Stage I 训练目标
- [[PPO]]: 闭环强化训练算法
- [[WAM|World Action Model]]: 世界动作模型应用

### 硬件/数据相关

- [[nuScenes]]: 评测数据集
- [[Waymo Open Dataset]]: WorldRec 评测
- [[LIBERO-Plus]]: 操作任务 benchmark

---

## 速查卡片

> [!summary] Xiaomi Auto World Model 2.0
> - **核心**: 统一流式因果世界模型，VTA+GLA 实现 3.7× 推理加速，WorldRec 2.0 提供 3D 高斯先验
> - **方法**: View-Temporal Attention + Gated Linear Attention + Fused-PRoPE + WorldRec 2.0；三阶段训练（Flow Matching → 多视角课程 → CD+Streaming DMD）
> - **结果**: PAI-Bench-G AV/Robot SOTA，NavSim CR 0.4%（最优），LIBERO-Plus 75.0%（无需具身预训练）
> - **代码**: 暂未开源（项目主页：https://xiaomiautoworldmodel.github.io/CausalWorldModel/）

---

*笔记创建时间: 2026-11-04*

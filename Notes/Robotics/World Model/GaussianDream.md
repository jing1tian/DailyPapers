---
title: "GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation"
method_name: "GaussianDream"
authors: [Zijian Zhang, Yuqing Jiang, Qian Cheng, Si Liu, Ding Zhao, Ping Luo, Weitao Zhou, Haibao Yu]
year: 2026
venue: arXiv
tags: [3d-gaussian-world-model, vla, robot-manipulation, gaussian-splatting, world-model, feed-forward, scene-flow]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.20752
created: 2026-05-22
---

# 论文笔记：GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tuojing Intelligence, UCAS, Institute of Automation CAS, Tsinghua University, Beihang University, CMU, University of Hong Kong |
| 日期 | May 2026 |
| 项目主页 | — |
| 对比基线 | [[pi0]], [[SpatialVLA]], [[VLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.20752) / [Code](https://github.com/TuojingAI/GaussianDream) |

---

## 一句话总结

> GaussianDream 将 [[3D Gaussian Splatting]] 世界模型以"即插即用"的方式耦合到 VLA 策略训练中，训练期间重建当前帧并预测未来帧的 3D 高斯状态，推理时仅用紧凑前缀条件化动作生成，无需测试时解码，实现轻量高效的在线控制。

---

## 核心贡献

1. **3D 空间锚定**: 用 [[3D Gaussian Splatting]] 重建当前场景，将 2D 像素级 VLA 提升到结构化 3D 空间表示，克服 VLM 的几何欠定问题。
2. **密集多模态监督**: 从机器人演示轨迹中自动挖掘 RGB 渲染、深度图和伪 3D 场景流监督信号，无需额外人工标注，解决稀疏动作模仿目标的低利用率问题。
3. **非对称训练-推理设计**: 训练时全量重建+预测（多解码头），推理时丢弃辅助头，只保留 GaussianDream 前缀（prefix）条件化动作生成，实现"轻量在线控制流水线"。

---

## 问题背景

### 要解决的问题

当前 [[VLA]]（视觉-语言-动作）策略存在三大核心瓶颈，限制了机器人操作的精度和泛化能力。

### 现有方法的局限

1. **空间/几何欠定（Spatial underspecification）**: 预训练 VLM 在 2D 像素上运行，3D 几何结构仅被隐式编码，缺乏对物体形状、位置和遮挡的显式建模。
2. **稀疏监督（Sparse supervision）**: 标准动作模仿目标仅对离散动作标签回归，忽略了演示视频中丰富的密集视觉信号（RGB、深度、运动场）。
3. **未来预测缺失（Limited future emulation）**: 大多数 VLA 缺乏预判交互后环境演化的机制，无法形成前瞻性的动作规划。

现有 3D 增强策略（如 GeoVLA、StereoVLA、VLA-4D）主要用几何信息锚定当前场景，缺乏显式的未来状态演化；基于 RGB/潜空间的世界模型（DreamZero、Cosmos Policy）则在非结构化空间预测，缺少 3D 几何约束。

### 本文的动机

通过将 [[3D Gaussian Splatting]] 的可微渲染能力与 VLA 训练流程结合，在训练期间强制一个紧凑的时空前缀（GaussianDream prefix）同时解码为可渲染的当前 3D 高斯状态和预测的未来 3D 高斯状态，从而将几何约束、密集视觉监督和未来预测三者统一在单一框架中。推理时前缀已隐式编码了这些几何先验，无需额外解码开销。

---

## 方法详解

### 模型架构

GaussianDream 采用**非对称训练-推理（Asymmetric Train-Inference）**架构：

- **输入**: 语言指令 $l$ + 当前多视角观测 $o_t$ + 时序历史帧 $o_{t-K:t}$ + 机器人状态 $s_t$ + 可学习查询 $Q_{GD}$
- **Backbone**: [[VGGT]]（冻结），提取多尺度 3D 感知 patch 特征
- **核心模块**: [[Temporal Gaussian Evolution|TGE 时序高斯演化模块]]（12 层注意力块）
- **输出（训练）**: 当前 3D 高斯集合 $G_t$ + 未来高斯预测 $\hat{G}_{t+\Delta}$ + 动作 $a_t$
- **输出（推理）**: 仅动作 $a_t$（辅助解码头丢弃）

### 核心模块

#### 模块1: GaussianDream 编码器（Temporal Gaussian Evolution, TGE）

**设计动机**: 利用 [[VGGT]] 的 3D 感知特征与可学习高斯查询的交互，建立时序感知的紧凑前缀表示。

**具体实现**:
- 从时序观测帧 $\{t-10, t-5, t\}$ 中，用冻结的 [[VGGT]] 提取多尺度 patch 特征
- 将每帧特征池化到 $32\times32$ 网格并投影成时序 token $\{P_{t-K:t}^{(m)}\}$
- TGE 包含 **12 个注意力块（8 头）**，每块先做帧内 [[Cross-Attention|自注意力]]（frame-wise token interaction），再做 token 级时间注意力（temporal attention），最后 MLP
- 维度：$2048 \to 512$（投影）$\to$ TGE $\to 2048$（投影回）
- 取当前帧输出作为 GaussianDream 前缀 $Z_t^{GD} \in \mathbb{R}^{1024 \times 2048}$

#### 模块2: 静态高斯头（Current Gaussian Reconstruction）

**设计动机**: 将前缀重塑为空间特征图，用轻量解码头预测可渲染的 [[3D Gaussian Splatting]] 场景。

**具体实现**:
- 将 1024-token 前缀 reshape 为 $32\times32$ 潜在网格，逐步上采样到 $256\times256$ 特征图 $F_t^G \in \mathbb{R}^{256\times256\times128}$
- 三个轻量预测头：
  - **几何头** $H_{geo}$：预测旋转、缩放、不透明度和深度 $D_t$
  - **外观头** $H_{app}$：预测 1 阶[[球谐函数|球谐系数]]（degree-1 spherical harmonics）
  - **深度头**：输出深度图用于中心点定位
- 组合为高斯集合 $G_t = \{(\mu_i^t, \theta_i^t)\}_{i=1}^{N_t}$

#### 模块3: 动态高斯头（Future Gaussian Prediction）

**设计动机**: 利用前缀中的时序运动线索，预测 horizon-conditioned 的未来帧 3D 高斯位移，实现前瞻性状态预测。

**具体实现**:
- 以**中心点位移**（center displacement）为预测目标，避免对高斯几何属性全面重预测
- 速度预测头 $H_{vel}$ 接收预测专用特征和 horizon 嵌入 $e_\Delta$，输出 $\nu_t^{(\Delta)}$
- 缩放因子 $\alpha_\Delta$ 调制位移幅度，更新中心 $\hat{\mu}_i^{t+\Delta} = \mu_i^t + \Delta x_i^{(\Delta)}$
- 保持几何属性 $\theta_i^t$ 不变，即 $\hat{G}_{t+\Delta} = \{(\hat{\mu}_i^{t+\Delta}, \theta_i^t)\}_{i=1}^{N_t}$

#### 模块4: VLA 动作策略

**设计动机**: 以 GaussianDream 前缀作为额外条件，用 [[Flow Matching]] 生成动作。

**具体实现**:
- 基于 [[pi0|π₀]] 的 [[Flow Matching]] 策略
- 动作生成以 $Z_t^{GD}$ 为前缀条件，与 $o_t, l, s_t$ 共同输入
- 推理时辅助头已丢弃，前缀直接参与动作生成，无渲染开销

---

## 关键公式

### 公式1: [[Temporal Gaussian Evolution|GaussianDream 前缀生成]]

$$
Z_t^{GD} = F_\omega(o_{t-K:t},\, Q_{GD})
$$

**含义**: TGE 编码器 $F_\omega$ 从时序观测和可学习查询生成 1024-token 的紧凑时空前缀。

**符号说明**:
- $Z_t^{GD} \in \mathbb{R}^{1024 \times 2048}$: GaussianDream 前缀
- $o_{t-K:t}$: 时序历史多视角观测帧
- $Q_{GD}$: 可学习高斯查询 token

### 公式2: [[3D Gaussian Splatting|双解码（仅训练期）]]

$$
G_t = R_\phi(Z_t^{GD},\, o_t), \quad \hat{G}_{t+\Delta} = D_\psi(G_t,\, Z_t^{GD},\, \Delta)
$$

**含义**: 训练期间前缀同时解码为当前帧重建高斯 $G_t$ 和未来帧预测高斯 $\hat{G}_{t+\Delta}$；推理时两个解码器均丢弃。

**符号说明**:
- $R_\phi$: 静态高斯重建解码器
- $D_\psi$: 动态高斯预测解码器
- $\Delta$: 预测 horizon（帧数偏移）

### 公式3: [[Flow Matching|动作生成]]

$$
a_t = \pi_\theta(o_t,\, l,\, s_t;\, Z_t^{GD})
$$

**含义**: 策略 $\pi_\theta$ 以 GaussianDream 前缀为条件，从当前观测、语言指令和机器人状态生成动作。

**符号说明**:
- $a_t$: 预测动作
- $l$: 语言指令
- $s_t$: 机器人状态

### 公式4: [[VGGT|时序 Token 特征池化]]

$$
P_{t-K:t}^{(m)} = W_m \cdot P_{32\times32}\!\left(E_{\text{VGGT}}^{(m)}(o_{t-K:t})\right)
$$

**含义**: 对每个时序帧，VGGT 的第 $m$ 层特征被池化到 $32\times32$ 空间网格后投影，形成 TGE 的输入时序 token。

**符号说明**:
- $E_{\text{VGGT}}^{(m)}$: VGGT 第 $m$ 尺度的特征提取器
- $P_{32\times32}$: 空间池化到 32×32 网格
- $W_m$: 可学习线性投影

### 公式5: [[Temporal Gaussian Evolution|TGE 前缀精炼]]

$$
Z_t^{GD} = \text{Proj}_{512 \to 2048}\!\left[\text{TGE}\!\left(\text{Proj}_{2048 \to 512}(Q_{GD}),\; \{P_{t-K:t}^{(m)}\}\right)\right]_t
$$

**含义**: 查询先降维到 512，经 TGE 与时序 token 交互后升维回 2048；取当前帧 $t$ 对应输出作为 GaussianDream 前缀。

**符号说明**:
- $\text{Proj}_{2048 \to 512}$: 降维线性投影
- $[\cdot]_t$: 取当前帧 $t$ 的输出 token

### 公式6: [[3D Gaussian Splatting|高斯特征图解码]]

$$
F_t^G = B_G\!\left(\text{Grid}(Z_t^{GD})\right), \quad F_t^G \in \mathbb{R}^{256\times256\times128}
$$

**含义**: 将 1024-token 前缀 reshape 为 $32\times32$ 网格，经上采样解码器 $B_G$ 生成 $256\times256$ 空间特征图，供几何/外观头预测。

**符号说明**:
- $B_G$: 高斯特征图上采样解码器
- $\text{Grid}(\cdot)$: 将 token 序列重排为 $32\times32$ 空间网格

### 公式7: [[3D Gaussian Splatting|高斯几何与外观预测]]

$$
D_t,\; \Theta_t^{geo} = H_{geo}(F_t^G), \quad \Theta_t^{app} = H_{app}(F_t^G,\, o_t)
$$

$$
G_t = \mathcal{A}(D_t,\, \Theta_t) = \{(\mu_i^t,\, \theta_i^t)\}_{i=1}^{N_t}
$$

**含义**: 几何头预测深度图和几何属性（旋转、缩放、不透明度）；外观头预测球谐系数；组合后得到完整高斯集合。

**符号说明**:
- $D_t$: 深度图，用于计算高斯中心 $\mu_i^t$
- $\Theta_t^{geo}$: 几何属性（旋转 $r$、缩放 $s$、不透明度 $\alpha$）
- $\Theta_t^{app}$: 1 阶球谐系数（外观属性）
- $\mathcal{A}$: 高斯组装函数

### 公式8: [[场景流|未来高斯速度与位置预测]]

$$
\nu_t^{(\Delta)} = H_{vel}\!\left(B_{pred}(Z_t^{GD}),\; e_\Delta\right), \quad
\Delta X_t^{(\Delta)} = \alpha_\Delta \nu_t^{(\Delta)}
$$

$$
\hat{\mu}_i^{t+\Delta} = \mu_i^t + \Delta x_i^{(\Delta)}, \quad
\hat{G}_{t+\Delta} = \{(\hat{\mu}_i^{t+\Delta},\; \theta_i^t)\}_{i=1}^{N_t}
$$

**含义**: 速度预测头在 horizon 嵌入 $e_\Delta$ 条件下预测高斯中心位移，缩放后加到当前中心，外观和几何属性保持不变。

**符号说明**:
- $\nu_t^{(\Delta)}$: 预测的高斯中心速度场
- $e_\Delta$: horizon 嵌入（编码预测步长 $\Delta$）
- $\alpha_\Delta$: horizon 相关的位移缩放因子
- $B_{pred}$: 预测专用特征解码器

### 公式9: [[World Model|Stage I 联合预训练损失]]

$$
\mathcal{L}_{GD} = \Bigl[\lambda_{cur}^{depth} \mathcal{L}_{cur}^{depth} + \lambda_{cur}^{render} \mathcal{L}_{cur}^{render}\Bigr]
+ \sum_{\Delta \in \mathcal{H}} w_\Delta \Bigl[\lambda_{depth} \mathcal{L}_{depth}^{(\Delta)} + \lambda_{render} \mathcal{L}_{render}^{(\Delta)} + \lambda_{flow} \mathcal{L}_{flow}^{(\Delta)}\Bigr]
$$

**含义**: 对当前帧用深度和 RGB 渲染损失；对每个预测 horizon $\Delta$，同时施加深度、RGB 渲染和伪 3D 场景流监督，权重 $w_\Delta$ 随 horizon 衰减。

**符号说明**:
- $\mathcal{L}_{cur}^{depth}$: 当前帧深度重建损失
- $\mathcal{L}_{cur}^{render}$: 当前帧 RGB 渲染损失
- $\mathcal{L}_{depth}^{(\Delta)}, \mathcal{L}_{render}^{(\Delta)}$: 未来帧深度/渲染损失
- $\mathcal{L}_{flow}^{(\Delta)}$: 伪 3D 场景流监督损失
- $\mathcal{H}$: 预测 horizon 集合
- $w_\Delta$: horizon 权重

### 公式10: [[Flow Matching|Stage II 动作学习损失]]

$$
\mathcal{L}_{act} = \mathbb{E}_{\tau,\varepsilon,a_t^*}\!\left[\left\|v_\theta\!\left(\tau\varepsilon + (1-\tau)a_t^*,\; c_t,\; \tau\right) - (\varepsilon - a_t^*)\right\|_2^2\right]
$$

$$
\mathcal{L} = \mathcal{L}_{act} + \lambda_{GD} \mathcal{L}_{GD}
$$

**含义**: 流匹配动作目标，预测噪声方向场；联合优化时 $\mathcal{L}_{GD}$ 继续提供 3D 几何监督。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流时间步（插值比例）
- $\varepsilon$: 噪声
- $a_t^*$: 专家动作
- $v_\theta$: 速度场网络
- $c_t$: 条件（包含 $Z_t^{GD}, o_t, l, s_t$）
- $\lambda_{GD}$: GD 损失权重

---

## 关键图表

### Figure 1: 操作策略范式对比

![Figure 1](https://arxiv.org/html/2605.20752v1/assets/comparation_3.png)

**说明**: 比较四类机器人操作策略范式——2D 策略（纯像素）、3D 增强策略（几何锚定当前帧）、视频/潜空间世界模型（非结构化预测）以及 GaussianDream（结构化 3D 高斯重建+预测+高效前缀推理）。GaussianDream 是唯一同时具备结构化 3D 表示、未来预测和轻量推理的方案。

### Figure 2: GaussianDream 框架总览

![Figure 2](https://arxiv.org/html/2605.20752v1/assets/framework_final_v.drawio.png)

**说明**: 展示完整的训练与推理流程。(a) 输入：语言指令、机器人状态、当前观测、时序历史帧、可学习查询。(b) GaussianDream 编码器（TGE）：从时序帧提取 VGGT 特征，与查询交互生成前缀。(c) 解码器：静态高斯头重建当前帧，动态高斯头预测未来帧。(d) RGB/深度渲染监督当前和未来高斯。(e) 动作策略以前缀为条件生成动作。(f) TGE 模块内部：交替执行帧内自注意力、时间注意力和 MLP。

### Figure 3: 真实机器人评估场景

![Figure 3](https://arxiv.org/html/2605.20752v1/assets/piper_setup_2.drawio.png)

**说明**: 展示四类真实机器人任务场景：属性区分抓取（Scene-A）、空间关系操作（Scene-B）、堆叠/拆叠（Scene-C）、长 horizon 多步骤任务（Scene-D）。平台使用 follower 机械臂 + agent 视角相机 + 腕部相机。

### Figure 4: 深度渲染可视化

![Figure 4](https://arxiv.org/html/2605.20752v1/assets/q_vis.drawio_final_2.png)

**说明**: 对比真实深度序列（行 1、3）与 GaussianDream 重建/预测结果（行 2、4）。每序列第一帧为当前帧，后五帧为连续未来帧。结果表明预测深度渲染在物体布局上保持一致，时序上连贯。

### Figure 5: 伪深度与场景流可视化

![Figure 5](https://arxiv.org/html/2605.20752v1/assets/LIBERO_Robocasa_flow_depth.drawio_compressed.png)

**说明**: 展示在 LIBERO 和 RoboCasa 演示数据上自动提取的伪深度图和场景流场。前两行为 LIBERO 的深度图和光流场，后两行为 RoboCasa。这些伪标签用于构造未来高斯预测的 3D 场景流监督目标，无需人工标注。

### Figure 6: 真实机器人硬件配置

![Figure 6](https://arxiv.org/html/2605.20752v1/assets/hardware_compressed.png)

**说明**: 展示评估平台：策略控制的 follower 机械臂（安装在移动底盘）+ agent 视角摄像头（全局观测）+ 腕部摄像头（末端局部观测）。演示采集时使用 leader 机械臂遥操作，评估时 GaussianDream 直接从语言指令和视觉观测控制 follower 臂。

### Figure 7: 真实机器人视觉观测

![Figure 7](https://arxiv.org/html/2605.20752v1/assets/piper_vis.drawio.png)

**说明**: 展示 agent 视角相机（提供工作空间全局布局）和腕部相机（提供末端执行器局部反馈）的互补视觉输入。

### Figure 8: 推理轨迹平滑度对比（附录）

![Figure 8](https://arxiv.org/html/2605.20752v1/assets/appendix_smooth.png)

**说明**: 与 π₀.₅ 基线相比，GaussianDream 以 GaussianDream 前缀为条件，生成更平滑的执行轨迹，印证了 3D 几何前缀对动作生成的稳定化作用。

### Figure 9: 重建与预测额外可视化（附录）

![Figure 9](https://arxiv.org/html/2605.20752v1/assets/LIBERO_appendix_vis.drawio.png)

**说明**: 更多当前高斯重建和未来高斯预测的定性对比示例，包含观测帧、伪深度目标、重建当前几何和预测未来高斯状态。展示 GaussianDream 同时捕获当前场景布局和短 horizon 几何变化的能力。

---

### Table 1: LIBERO 基准成功率（%）

| 方法 | Spatial | Object | Goal | Long | **Average** |
|------|---------|--------|------|------|-------------|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| GeoPredict | 98.0 | 98.2 | 95.7 | 94.0 | 96.5 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| **GaussianDream** | **99.0** | **99.6** | **99.0** | 96.0 | **98.4** |

**关键发现**: GaussianDream 在 Spatial、Object、Goal 三项达到 SOTA，4 项平均以 98.4% 超越 π₀.₅（96.7%）和 GeoPredict（96.5%），仅在 Long 任务被 LingBot-VA 略超（96.0% vs 98.5%）。

### Table 2: RoboCasa Human-50 成功率（%）

| 方法 | Pick&Place | Doors/Drawers | Others | **Average** |
|------|------------|---------------|--------|-------------|
| π₀ | 13.95 | 53.1 | 58.5 | 42.4 |
| π₀.₅ | 36.0 | 46.5 | 39.5 | 40.1 |
| GeoPredict | 22.7 | 75.1 | 62.4 | 52.4 |
| **GaussianDream** | **43.8** | 65.2 | 52.0 | **52.6** |

**关键发现**: GaussianDream 在长 horizon 厨房任务中以 52.6% 平均成功率领先，尤其在 Pick&Place 类任务（43.8%）显著优于所有基线，但在 Doors/Drawers 类任务被 GeoPredict 超越（65.2% vs 75.1%）。

### Table 3: 真实机器人成功率（%）

| 方法 | Scene-A（属性） | Scene-B（空间） | Scene-C（堆叠） | Scene-D（长 horizon） | **Average** |
|------|----------------|----------------|----------------|----------------------|-------------|
| π₀.₅ | 42.5 | 50.0 | 25.0 | 20.0 | 34.4 |
| **GaussianDream** | **55.0** | **70.0** | **35.0** | **40.0** | **50.0** |

**关键发现**: 在真实机器人上全面超越 π₀.₅，整体成功率从 34.4% 提升至 50.0%（+15.6pp），其中 Scene-D（长 horizon）提升最显著（+20pp），验证了 3D 前缀对复杂任务的实际收益。

### Table 4: 消融实验（LIBERO 平均成功率 %）

| Current Recon | Future Pred | RGB Rendering | Depth Sup | LIBERO Avg |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | ✗ | 97.0 |
| ✓ | ✗ | ✓ | ✓ | 97.3 |
| ✓ | ✓ | ✗ | ✓ | 97.5 |
| ✓ | ✓ | ✓ | ✗ | 97.2 |
| ✓ | ✓ | ✓ | ✓ | **98.4** |

**关键发现**: 各组件提供互补收益——仅当前重建基线 97.0%，加入未来预测贡献 +0.5%，RGB 渲染与深度监督缺一不可（移除任一均下降至 97.2%-97.5%），完整模型达到 98.4%。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 50 demos/任务，4 个协议（Spatial/Object/Goal/Long） | 桌面操作，多任务 benchmark | 仿真训练+测试 |
| RoboCasa Human-50 | 50 trials，24 个厨房任务，5 个场景 | 长 horizon，家庭环境，复杂任务组合 | 仿真训练+测试 |
| 真实机器人数据 | 遥操作采集，4 类任务场景 | 属性区分、空间关系、堆叠、长 horizon | 真实机器人评估 |

### 实现细节

- **Backbone**: [[VGGT]]（冻结）提取时序 3D 感知特征
- **TGE 模块**: 12 层注意力块，8 头，维度 512
- **GaussianDream 前缀**: 1024 tokens × 2048 dim
- **高斯特征图**: $256\times256\times128$
- **训练阶段**: Stage I（GD 预训练）+ Stage II（策略联合优化）
- **基础策略**: 基于 [[pi0|π₀]] 的 [[Flow Matching]] 框架
- **训练细节**: 论文未公开具体 batch size、学习率、GPU 型号

### 可视化结果

深度渲染结果（Figure 4）表明预测的高斯深度图在物体布局上与真实值高度一致，且在时序上保持连贯性，未出现漂移或坍塌。伪深度和场景流可视化（Figure 5）确认了自动提取的伪标签质量可靠，为未来预测提供有效的几何监督信号。真实机器人轨迹（Figure 8）相比 π₀.₅ 更平滑，印证了 3D 前缀对动作生成的稳定化效果。

---

## 批判性思考

### 优点

1. **非对称设计优雅**: 训练时引入重量级 3D 重建/预测头，推理时完全丢弃，前缀已隐式蒸馏了几何先验，几乎无推理开销。
2. **无需额外标注**: 伪深度和场景流从演示轨迹自动提取，不依赖特殊传感器或人工标注，工程可行性高。
3. **模块化即插即用**: 设计为 VLA 策略的"插件"，可接驳不同基础策略（论文中对接 π₀）。
4. **全面验证**: 同时在仿真（LIBERO、RoboCasa）和真实机器人上验证，有一定说服力。

### 局限性

1. **长 horizon 任务相对弱**: LIBERO Long 子任务（96.0%）被 LingBot-VA（98.5%）超越，RoboCasa Doors/Drawers 也被 GeoPredict 超越，说明 3D 前缀在需要精确力控的开关类任务上收益有限。
2. **实现细节不透明**: 论文未公开 batch size、学习率、GPU 配置、训练步数等关键复现细节，给复现带来障碍。
3. **伪标签质量依赖**: 深度和场景流监督基于伪标签，在纹理稀疏或快速运动场景中可能引入噪声。
4. **单臂评估**: 真实机器人实验仅限单臂操作，未验证双臂或灵巧手场景。

### 潜在改进方向

1. 探索更高 horizon 预测（当前约 5-10 帧），结合语言条件化的长 horizon 规划。
2. 将精确深度传感器（RealSense 等）替换伪深度标签，在纹理稀疏场景提升监督质量。
3. 扩展至双臂协作操作，验证多臂场景下的 3D 世界模型收益。

### 可复现性评估

- [x] 代码开源（https://github.com/TuojingAI/GaussianDream）
- [ ] 预训练模型（未明确提供）
- [ ] 训练细节完整（batch size、LR 等未公开）
- [x] 数据集可获取（LIBERO、RoboCasa 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[pi0|π₀]]: 基础 VLA 策略框架，GaussianDream 在其上构建动作生成策略
- [[VGGT]]: 冻结的 3D 感知视觉编码器，提供多尺度 patch 特征
- [[3D Gaussian Splatting]]: 核心 3D 场景表示方式，支持可微渲染

### 对比

- [[SpatialVLA]]: 3D 增强 VLA，使用几何信息锚定当前场景但无未来预测
- [[VLA]]: 标准 2D VLA 基线（π₀、π₀.₅）
- [[World Model]]: 视频/潜空间世界模型（DreamZero、Cosmos Policy）

### 方法相关

- [[Temporal Gaussian Evolution]]: 核心时序交互模块（本文新提出）
- [[Flow Matching]]: 动作生成的概率框架
- [[场景流]]: 3D 场景运动表示，用于未来预测的伪标签
- [[球谐函数]]: 高斯外观表示的基函数（1 阶）

### 硬件/数据相关

- [[LIBERO]]: 仿真操作 benchmark
- [[RoboCasa]]: 长 horizon 厨房任务 benchmark

---

## 速查卡片

> [!summary] GaussianDream (2026)
> - **核心**: 将 3D 高斯世界模型以即插即用方式耦合到 VLA 训练，推理时仅用紧凑前缀
> - **方法**: VGGT 编码 + TGE 模块 + 静态/动态高斯双解码头 + Flow Matching 策略
> - **结果**: LIBERO 98.4%、RoboCasa Human-50 52.6%、真实机器人 50.0%
> - **代码**: https://github.com/TuojingAI/GaussianDream

---

*笔记创建时间: 2026-05-22*

---
title: "LAWM-3D: Learning 3D-Aware Latent Actions from Human Videos for Generalizable Robot World Models"
method_name: "LAWM-3D"
authors: [Jiarui Yang, Jiale Zhang, Jiawei Li, Hang Guo, Wen Huang, Jinpeng Wang, Peidong Liu, Shu-Tao Xia]
year: 2026
venue: arXiv
tags: [world-model, latent-action, multi-view-learning, 3d-aware, robot-learning, depth-estimation, human-video-pretraining]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.05706v1
created: 2026-08-08
---

# 论文笔记：LAWM-3D

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, Zhejiang University |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[DreamDojo]]、[[Cosmos-Predict-2.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.05706) / Code N/A |

---

## 一句话总结

> 通过多视角不变 tokenization、[[VGGT]] 几何对齐约束和 RGB-D 联合重建三个互补组件，从无标注人类视频中学习 3D 感知的 [[LAM|隐动作表示]]，显著提升机器人世界模型的生成质量与泛化能力。

---

## 核心贡献

1. **多视角不变动作 Tokenization**: 使用随机视角遮盖训练 spatio-temporal [[Transformer]] encoder，强制模型从不完整观测中推断视角无关的隐动作，解决跨摄像头外观差异问题。
2. **非注入式 RGB-D 联合重建**: Decoder 同时预测未来 RGB 帧和深度图，深度监督提供视角一致的几何线索，避免隐动作退化为未来帧的压缩表示（appearance leakage）。
3. **VGGT 几何表示对齐**: 在 encoder 中间层（第 6、11 层）与预训练 [[VGGT]] 的特征进行角度+尺度双重对齐，显式注入 3D 几何知识而不影响动作建模容量。

---

## 问题背景

### 要解决的问题

现有 [[LAM|Latent Action Model]] 从单视角视频学习隐动作，无法捕获真实的 3D 运动语义。简单将多视角视频引入 LAM 训练并不能自动产生 3D 感知的动作表示，因为两个关键问题：
1. **Future-frame appearance leakage**：encoder 倾向于将未来帧的外观信息编码到隐动作中，而非真实运动
2. **Inter-camera appearance discrepancy**：不同摄像头的光照、视角变化导致跨视角特征不一致

### 现有方法的局限

- **LAPA、UniLACT、CoMo、DreamDojo** 等 LAM 方法均基于单视角视频，隐动作缺乏 3D 几何信息
- 简单多视角扩展（消融 Case 1）反而导致性能下降（LAM PSNR 从 29.31 → 28.97），因为 appearance leakage 被放大
- 基于 2D 语义特征（[[DINOv2]]）的对齐不如基于 3D 几何的对齐有效

### 本文的动机

人类视频天然包含多视角同步记录（Ego-Exo 设置），且 3D 基础模型（[[VGGT]]、[[DUSt3R]]）已能从单/多视角图像重建精确几何。将几何先验蒸馏进 LAM 的 encoder，同时通过 RGB-D 重建提供隐式 3D 监督，可以从根本上解决 appearance leakage 和视角一致性问题。

---

## 方法详解

### 模型架构

LAWM-3D 采用三阶段训练框架：

- **输入**: 多视角同步观测帧 $\{f^v_t\}_{v=1}^V$（最多 7 个视角），分辨率 320×240（LAM）/ 640×480（世界模型）
- **Stage 1 — LAM 预训练**: 在人类视频上学习 3D 感知隐动作
- **Stage 2 — 世界模型预训练**: 以 [[Cosmos-Predict-2.5]] 为 backbone，用学到的隐动作条件生成
- **Stage 3 — 机器人微调**: 在机器人交互数据上做动作空间对齐

### 核心模块

#### 模块 1：多视角不变动作 Tokenization

**设计动机**: 利用[[Transformer]]的 spatio-temporal 注意力 + 随机视角遮盖，强制 encoder 从任意视角子集推断一致动作表示

**具体实现**:
- 输入帧划分为 16×16 patch，转换为 token
- Token 与 action token 拼接，添加 view embedding 和 [[RoPE|Rotary Positional Encoding]]
- Encoder：L=24 层，交替进行 spatial 和 temporal 自注意力
- 隐动作维度：32
- 训练时随机遮盖部分视角，推理时 mean pooling 聚合多视角动作（效果最优，见 Table 5）

#### 模块 2：非注入式 RGB-D 联合重建

**设计动机**: 深度图是视角一致的几何信号，对外观变化不敏感；通过重建目标施加 3D 约束，而非直接将深度注入 encoder（防止 encoder 走捷径）

**具体实现**:
- Decoder 联合预测未来 RGB 帧 $\hat{f}^v_{t+1}$ 和深度图 $\hat{d}^v_{t+1}$
- 深度标注来自离线预计算（[[DepthPro]] 或类似模型）
- 重建损失基于 [[β-VAE]] 框架（见公式部分）
- "非注入"指深度信息仅在重建端（decoder）引入，encoder 输入保持纯 RGB

#### 模块 3：VGGT 几何表示对齐

**设计动机**: 直接对齐 encoder 的中间特征与 [[VGGT]] 提取的 3D 几何特征，将外部 3D 先验蒸馏进动作表示

**具体实现**:
- 对 encoder 第 6、11 层的 spatial token $h_{k,v,p}$ 进行双路对齐
- **Angular loss**：确保方向一致（余弦相似度最大化）
- **Scale loss**：通过额外 MLP $g_\psi$ 预测特征幅度，确保量级一致
- [[VGGT]] 作为 3D 基础模型冻结，提供目标特征 $y_{k,v,p}$
- 对比实验（Table 8）表明 [[VGGT]] 优于 [[DUSt3R]]（+0.25 PSNR）和 [[DINOv2]]（+0.78 PSNR）

---

## 关键公式

### 公式 1：[[β-VAE|β-VAE 重建损失]] $\mathcal{L}_{\text{Rec}}$

$$
\mathcal{L}_{\text{Rec}}(f^v_{t+1}) = \mathbb{E}_{q_\phi(\hat{\mathbf{a}}|f^v_{t:t+1})}\left[\log p_\theta(f^v_{t+1}|\mathbf{a},f^v_t)\right] - \beta \, D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{a}|f^v_{t:t+1})\,\|\,p(\mathbf{a})\right)
$$

**含义**: 在 β-VAE 框架下，最大化重建对数似然同时最小化后验分布与标准正态先验的 KL 散度，β 控制正则化强度

**符号说明**:
- $f^v_t, f^v_{t+1}$: 视角 $v$ 下时刻 $t$、$t+1$ 的观测帧
- $q_\phi$: encoder（推理网络），参数为 $\phi$
- $p_\theta$: decoder（生成网络），参数为 $\theta$
- $\mathbf{a}$: 隐动作向量（32 维）
- $\beta$: KL 散度权重系数
- $D_{\mathrm{KL}}$: KL 散度，约束 $\mathbf{a}$ 的分布贴近标准正态

### 公式 2：[[几何角度对齐损失|Angular Alignment Loss]] $\mathcal{L}_{\text{Angular}}$

$$
\mathcal{L}_{\text{Angular}} = -\frac{1}{KVP}\sum_{k,v,p}\cos\!\left(y_{k,v,p},\, f_\phi(h_{k,v,p})\right)
$$

**含义**: 通过余弦相似度最大化，使 encoder 中间层特征的**方向**与 [[VGGT]] 3D 特征对齐

**符号说明**:
- $K$: 对齐的层数（本文选 $K=2$，对应第 6、11 层）
- $V$: 视角数量
- $P$: 每帧的 patch token 数量
- $y_{k,v,p}$: [[VGGT]] 提取的第 $k$ 层、视角 $v$、位置 $p$ 处的几何特征（目标）
- $f_\phi(h_{k,v,p})$: encoder 第 $k$ 层对应位置的中间特征（经过线性投影）
- $\cos(\cdot,\cdot)$: 余弦相似度

### 公式 3：[[几何尺度对齐损失|Scale Alignment Loss]] $\mathcal{L}_{\text{Scale}}$

$$
\mathcal{L}_{\text{Scale}} = \frac{1}{KVP}\sum_{k,v,p}\left\|g_\psi\!\left(\frac{f_\phi(h_{k,v,p})}{\|f_\phi(h_{k,v,p})\|_2}\right) - y_{k,v,p}\right\|^2_2
$$

**含义**: 在方向归一化后，通过可学习 MLP $g_\psi$ 预测特征幅度（尺度），使 encoder 特征的**量级**也与 [[VGGT]] 对齐

**符号说明**:
- $g_\psi$: 可学习的尺度预测 MLP，参数为 $\psi$
- $\|\cdot\|_2$: L2 范数
- 分子 $f_\phi(h_{k,v,p})/\|f_\phi(h_{k,v,p})\|_2$: encoder 特征的单位方向向量

### 公式 4：[[LAWM-3D 总损失|LAM 总目标函数]] $\mathcal{L}_{\text{LAM}}$

$$
\mathcal{L}_{\text{LAM}} = \mathcal{L}_{\text{Rec}} + \lambda_{\text{Angular}}\,\mathcal{L}_{\text{Angular}} + \lambda_{\text{Scale}}\,\mathcal{L}_{\text{Scale}}
$$

**含义**: 三个互补损失的加权求和，重建损失确保动作可控生成，两个对齐损失将 3D 几何先验注入隐动作空间

**符号说明**:
- $\lambda_{\text{Angular}} = 0.5$: 角度损失权重（消融 Table 7 最优值）
- $\lambda_{\text{Scale}} = 0.05$: 尺度损失权重（消融 Table 7 最优值）

---

## 关键图表

### Figure 1：Framework Overview / 框架概览

![Figure 1](https://arxiv.org/html/2608.05706v1/x1.png)

**说明**: LAWM-3D 整体框架。左侧多视角和单视角人类视频输入 LAM，学习 3D 感知隐动作；右侧隐动作条件世界模型在人类视频上预训练后，微调至机器人数据。三个核心组件（多视角 tokenization、VGGT 对齐、RGB-D 重建）通过箭头标注。

### Figure 2：LAM Architecture / 模型架构

![Figure 2](https://arxiv.org/html/2608.05706v1/x2.png)

**说明**: 3D 感知 [[LAM]] 的详细架构。左：spatio-temporal [[Transformer]] encoder 结合随机视角遮盖；中：[[VGGT]] 对中间层做几何特征对齐；右：decoder 联合重建 RGB 和深度图。

### Figure 3：Depth Prediction Visualization / 深度预测对比

![Figure 3](https://arxiv.org/html/2608.05706v1/x3.png)

**说明**: 自我视角深度预测的定性对比。LAWM-3D 表现出更强的深度理解和对大幅运动的准确预测，baseline（DreamDojo-D）在运动区域出现明显失真。

### Figure 4：World Model Rollouts / 世界模型展开

![Figure 4](https://arxiv.org/html/2608.05706v1/x4.png)

**说明**: WorldArena benchmark 上机器人操作任务的世界模型展开可视化。LAWM-3D 在物体交互区域表现出更高的物理合理性。

### Figure 5：Real-world Comparison / 真实场景对比

![Figure 5](https://arxiv.org/html/2608.05706v1/x5.png)

**说明**: Ground truth、DreamDojo 与 LAWM-3D 的真实场景展开对比。本方法在空间定位和物体交互上展现更强的物理可信度。

### Figure 6：Pretraining Strategy Impact / 预训练策略影响

![Figure 6](https://arxiv.org/html/2608.05706v1/Figures/datasets.png)

**说明**: 不同预训练策略对世界模型性能的影响曲线，包含平均归一化得分和 PSNR 收敛曲线，说明三阶段训练管线的合理性。

### Figure 7：Policy Evaluation Agreement / 策略评估一致性

![Figure 7](https://arxiv.org/html/2608.05706v1/Figures/policy.png)

**说明**: 世界模型想象回报（imagined returns）与真实环境表现的一致性分析，Spearman 秩相关系数 $\rho=0.952$，Regret@1 = 0.040。

### Figure 8：Online Policy Selection / 在线策略选择

![Figure 8](https://arxiv.org/html/2608.05706v1/Figures/planning.png)

**说明**: 不同策略选择方法的闭环成功率对比。自适应选择方法（利用多步 imagination）将成功率从 53% 提升到 70%（+17 个百分点绝对增益）。

### Figure 9：Long Rollout Comparison / 长时间展开对比

![Figure 9](https://arxiv.org/html/2608.05706v1/x6.png)

**说明**: 长时间真实场景展开对比，LAWM-3D 保持更高视觉质量和更好的动作跟随能力，DreamDojo 在长时间出现累积误差。

### Figure 10：Cross-View Latent Transfer / 跨视角隐动作迁移

![Figure 10](https://arxiv.org/html/2608.05706v1/x7.png)

**说明**: 从一个视角提取的隐动作可成功预测另一视角的未来观测，验证 LAWM-3D 的隐动作是视角无关的运动描述符，而非视角特定外观编码。

### Figure 11：Representation Similarity Matrix / 表示相似度矩阵

![Figure 11](https://arxiv.org/html/2608.05706v1/Figures/cross.png)

**说明**: 100 个人类视频 clip 的隐表示余弦相似度矩阵，LAWM-3D 在类内相似度（同类动作）显著高于类间，体现出良好的动作语义聚类。

### Figure 12：Mutual Information Analysis / 互信息分析

![Figure 12](https://arxiv.org/html/2608.05706v1/Figures/mu.png)

**说明**: 学到的隐动作与真实机器人动作的[[互信息]]分析，使用 KSG、Barber-Agakov、[[MINE]] 三个独立估计器，LAWM-3D 均超越 MVP-LAM baseline 约 15%。

### Figure 13：Linear Probing Results / 线性探测结果

![Figure 13](https://arxiv.org/html/2608.05706v1/Figures/li.png)

**说明**: [[线性探测|Linear Probing]] 评估隐动作的动作可恢复性，在 Bridge V2 和 LIBERO（分布外）benchmark 上均取得最低归一化 MSE，长时间任务优势尤为突出。

### Figure 14：VGGT Point Cloud / VGGT 点云重建

![Figure 14](https://arxiv.org/html/2608.05706v1/x8.png)

**说明**: [[VGGT]] 从多视角输入重建的点云示意，说明其提供的 3D 几何特征的质量与准确性，为 LAWM-3D 的几何对齐监督提供基础。

---

### Table 1：LAM 定量对比

| Method | EgoExo4D RGB PSNR↑ | EgoExo4D RGB SSIM↑ | EgoExo4D Depth PSNR↑ | EgoExo4D Depth SSIM↑ | Assembly101 RGB PSNR↑ | Assembly101 RGB SSIM↑ | Assembly101 Depth PSNR↑ | Assembly101 Depth SSIM↑ |
|--------|---|---|---|---|---|---|---|---|
| LAPA | 30.75 | 0.842 | / | / | 31.26 | 0.867 | / | / |
| UniLACT | 30.97 | 0.847 | 32.58 | 0.880 | 31.66 | 0.875 | 33.91 | 0.911 |
| CoMo | 31.85 | 0.864 | / | / | 32.96 | 0.891 | / | / |
| DreamDojo | 32.91 | 0.901 | / | / | 33.43 | 0.922 | / | / |
| DreamDojo-D | 32.56 | 0.893 | 33.54 | 0.907 | 33.12 | 0.907 | 34.35 | 0.921 |
| **LAWM-3D** | **33.38** | **0.936** | **35.62** | **0.939** | **35.76** | **0.944** | **35.85** | **0.948** |

**关键发现**: LAWM-3D 在所有指标上全面超越 baseline，深度 PSNR 提升尤为显著（+2.08 vs DreamDojo-D on EgoExo4D），验证三个组件协同带来的 3D 理解提升。

---

### Table 2：视频质量综合评估（16 指标 × 6 维度）

| Method | 图像质量 | 美学质量 | JEPA相似 | 动态程度 | 光流分 | 运动平滑 | 主体一致 | 背景一致 | 光度一致 | 交互质量 | 轨迹精度 | 深度精度 | 透视感 | 指令跟随 | 语义对齐 | 动作跟随 |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GigaWorld | 0.463 | 0.392 | 0.437 | 0.614 | 0.315 | 0.778 | 0.734 | 0.832 | 0.179 | 0.529 | 0.159 | 0.618 | 0.760 | 0.607 | 0.851 | 0.116 |
| Genie | 0.229 | 0.326 | 0.334 | 0.674 | 0.090 | 0.694 | 0.770 | 0.883 | 0.205 | 0.205 | 0.074 | 0.851 | 0.520 | 0.208 | 0.855 | 0.018 |
| RoboMaster | 0.353 | 0.405 | 0.302 | 0.658 | 0.153 | 0.700 | 0.810 | 0.903 | 0.341 | 0.533 | 0.122 | 0.809 | 0.738 | 0.569 | 0.846 | 0.082 |
| Cosmos 2.5 | 0.440 | 0.355 | 0.903 | 0.592 | 0.260 | 0.735 | 0.804 | 0.895 | 0.355 | 0.541 | 0.294 | 0.862 | 0.763 | 0.611 | 0.857 | 0.070 |
| DreamDojo | 0.476 | 0.411 | 0.915 | 0.603 | 0.304 | 0.771 | 0.812 | 0.909 | 0.356 | 0.558 | 0.322 | 0.873 | 0.795 | 0.680 | 0.883 | 0.087 |
| WoW | 0.460 | 0.381 | 0.744 | 0.454 | 0.271 | 0.781 | 0.812 | 0.893 | 0.221 | 0.538 | 0.212 | 0.727 | 0.749 | 0.573 | 0.881 | 0.049 |
| IRASim | 0.347 | 0.365 | 0.916 | 0.419 | 0.211 | 0.692 | 0.813 | 0.898 | 0.350 | 0.567 | 0.265 | 0.870 | 0.782 | 0.648 | 0.859 | 0.059 |
| **LAWM-3D** | **0.491** | **0.397** | **0.920** | **0.618** | **0.320** | **0.789** | **0.815** | **0.913** | **0.361** | **0.578** | **0.350** | **0.889** | **0.805** | **0.731** | **0.892** | **0.095** |

**关键发现**: LAWM-3D 在 16 个指标中 15 个达到 SOTA，深度精度（0.889 vs 0.873）和动作跟随（0.095 vs 0.087）提升最为突出，验证 3D 感知隐动作的价值。

---

### Table 3：分布外泛化评估（OOD 场景）

| Method | In-Lab PSNR↑ | In-Lab SSIM↑ | In-Lab LPIPS↓ | EgoDex PSNR↑ | EgoDex SSIM↑ | EgoDex LPIPS↓ | DH-HV PSNR↑ | DH-HV SSIM↑ | DH-HV LPIPS↓ |
|--------|---|---|---|---|---|---|---|---|---|
| Cosmos-Predict2.5 | 20.576 | 0.774 | 0.222 | 19.952 | 0.787 | 0.219 | 18.274 | 0.754 | 0.236 |
| DreamDojo-2B | 21.114 | 0.774 | 0.222 | 20.411 | 0.775 | 0.226 | 18.813 | 0.747 | 0.238 |
| DreamDojo-14B | 21.413 | 0.788 | 0.208 | 20.525 | 0.787 | 0.213 | 18.924 | 0.751 | 0.228 |
| **LAWM-3D (2B)** | **21.465** | **0.792** | **0.207** | **20.542** | **0.796** | **0.197** | **19.034** | **0.777** | **0.211** |

**关键发现**: LAWM-3D 2B 模型超越参数量 7× 的 DreamDojo-14B，尤其在 LPIPS（感知质量）上全面领先，体现 3D 感知隐动作对泛化的价值。

---

### Table 4：消融实验 — 各组件贡献

| 配置 | 多视角 | 几何对齐 | 深度重建 | LAM PSNR↑ | LAM SSIM↑ | WM PSNR↑ | WM SSIM↑ |
|------|:---:|:---:|:---:|---|---|---|---|
| Baseline | | | | 29.31 | 0.886 | 21.22 | 0.782 |
| Case 1 (仅多视角) | ✓ | | | 28.97 | 0.872 | 20.53 | 0.711 |
| Case 2 (仅几何) | | ✓ | | 29.84 | 0.906 | 21.36 | 0.789 |
| Case 3 (几何+深度) | | ✓ | ✓ | 29.69 | 0.912 | 21.42 | 0.788 |
| Case 4 (多视角+几何) | ✓ | ✓ | | 29.92 | 0.919 | 21.64 | 0.793 |
| **Default (Full)** | ✓ | ✓ | ✓ | **30.56** | **0.933** | **21.68** | **0.797** |

**关键发现**: 仅加入多视角（Case 1）反而比 baseline **下降** 0.34 PSNR，证明几何对齐是抑制 appearance leakage 的必要条件；三个组件互补，Full 模型效果最优。

---

### Table 5：多视角聚合策略对比

| 聚合策略 | RGB PSNR↑ | RGB SSIM↑ | Depth PSNR↑ | Depth SSIM↑ |
|----------|---|---|---|---|
| Max Pooling | 33.08 | 0.924 | 35.84 | 0.931 |
| Attention Pooling | 33.34 | 0.929 | 36.21 | 0.939 |
| Concatenation | 33.29 | 0.928 | 36.08 | 0.937 |
| **Mean Pooling** | **33.56** | **0.933** | **36.58** | **0.945** |

**关键发现**: 简单 Mean Pooling 效果最优，说明各视角动作表示质量均衡，无需加权或注意力聚合。

---

### Table 6：VGGT 对齐层选择分析

| 对齐层 | RGB PSNR↑ | RGB SSIM↑ |
|--------|---|---|
| 无对齐（仅重建） | 29.31 | 0.886 |
| [1,6]（浅层） | 29.74 | 0.905 |
| [19,24]（深层） | 29.96 | 0.917 |
| [6,8]（窄范围） | 30.15 | 0.922 |
| **[6,11]（本文）** | **30.56** | **0.933** |
| [6,16]（宽范围） | 30.38 | 0.927 |
| [1,24]（全层） | 30.02 | 0.918 |

**关键发现**: 第 6-11 层（中间层）对齐效果最优，过浅（底层语义不足）或过深（过于任务特定）均次优；全层对齐反而不如中间层，体现几何监督的"适量"原则。

---

### Table 7：对齐损失权重灵敏度

| $\lambda_{\text{Angular}}$ | $\lambda_{\text{Scale}}$ | RGB PSNR↑ | RGB SSIM↑ |
|---|---|---|---|
| 0.0 | 0.0 | 29.31 | 0.886 |
| 0.1 | 0.05 | 30.18 | 0.921 |
| **0.5** | **0.05** | **30.56** | **0.933** |
| 1.0 | 0.05 | 30.37 | 0.926 |
| 0.5 | 0.01 | 30.29 | 0.924 |
| 0.5 | 0.1 | 30.33 | 0.927 |
| 0.5 | 0.5 | 29.95 | 0.915 |

**关键发现**: 模型对权重具有较好的鲁棒性，$\lambda_{\text{Angular}}=0.5, \lambda_{\text{Scale}}=0.05$ 为最优组合，$\lambda_{\text{Scale}}$ 过大（0.5）会略微损害性能。

---

### Table 8：3D 基础模型对比

| 对齐目标 | RGB PSNR↑ | RGB SSIM↑ |
|----------|---|---|
| 无对齐 | 29.31 | 0.886 |
| DINOv2（2D 语义） | 29.78 | 0.907 |
| Depth-Anything-V2 | 30.02 | 0.916 |
| [[DUSt3R]] | 30.31 | 0.926 |
| **[[VGGT]]（本文）** | **30.56** | **0.933** |

**关键发现**: 多视角几何模型（[[DUSt3R]]、[[VGGT]]）显著优于 2D 语义模型（[[DINOv2]]），说明真正的 3D 几何监督不可或缺；[[VGGT]] 优于 [[DUSt3R]] +0.25 PSNR，可能因其更强的泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Ego-Exo|Ego-Exo4D]] | 1,286 小时，740 名参与者 | 同步 Ego+Exo 多视角 | LAM 预训练（多视角） |
| [[Assembly101]] | 4,321 视频，513 小时 | 程序性装配任务，多视角 | LAM 预训练（多视角） |
| [[EgoDex]] | 829 小时，~90M 帧，338k 示范 | 单视角手部操作 | LAM 预训练（单视角） |
| AgiBot-World | — | 机器人操作数据 | 世界模型预训练 |
| RT-1 / DROID | — | 机器人多样场景 | 世界模型预训练 |
| GR1_robot | — | 机器人微调数据 | Stage 3 微调 |

### 实现细节

**LAM 预训练（Stage 1）:**
- 优化器：AdamW，学习率 2.5×10⁻⁵，weight decay 0.01
- Batch size：128，迭代次数：600k
- 最大视角数：7，输入分辨率：320×240

**世界模型预训练（Stage 2）:**
- Backbone：[[Cosmos-Predict-2.5]]
- 优化器：AdamW，学习率 1.6×10⁻⁴
- Batch size：256，迭代次数：300k，附加 temporal consistency loss

**Stage 3 微调:**
- 在机器人交互数据上做动作空间对齐微调

### 可视化结果

定性对比（Figure 3-5, 9）表明 LAWM-3D 在以下场景上优势明显：
- 物体接触区域的深度一致性
- 大幅运动下的帧间连贯性
- 长时间展开的累积误差控制

---

## 批判性思考

### 优点
1. **消融设计严谨**：Case 1（仅多视角退化）清楚证明几何监督的必要性，逻辑链条清晰
2. **下游应用完整**：不仅报告生成指标，还验证了世界模型用于策略评估和在线策略选择的实际价值（+17pp 成功率）
3. **参数效率高**：2B 参数量超越 14B DreamDojo，体现隐动作质量的杠杆效应

### 局限性
1. **依赖高质量 3D 基础模型**：需要 [[VGGT]] 冻结提取特征，若 VGGT 在特定场景失效，对齐质量无法保证
2. **多视角数据需求**：LAM 预训练需要同步多摄像头数据，限制了数据来源的多样性
3. **离线深度监督**：依赖预计算深度图，在线实时深度不可用时训练流程受阻
4. **评估局限于操作任务**：所有实验集中在手部操作，足式运动、导航等场景的迁移能力未验证
5. **高预训练计算成本**：三阶段训练对硬件资源要求高，阻碍快速迭代

### 潜在改进方向
1. 在线深度估计替代离线预计算深度图
2. 将几何对齐扩展到视频时序维度（当前仅在 patch 空间对齐）
3. 探索更轻量级的 3D 基础模型替代 VGGT 以降低推理开销

### 可复现性评估
- [ ] 代码开源（论文提交时未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、数据集均有描述）
- [x] 数据集可获取（Ego-Exo4D、Assembly101、EgoDex 均为公开数据）

---

## 关联笔记

### 基于
- [[LAM]]: 隐动作模型基础框架
- [[Cosmos-Predict-2.5]]: 世界模型 backbone
- [[VGGT]]: 3D 几何特征提取的教师模型
- [[β-VAE]]: 隐动作学习的变分框架

### 对比
- [[DreamDojo]]: 同类 LAM-based 世界模型，最强 baseline
- [[DUSt3R]]: 替代 3D 对齐目标（LAWM-3D -0.25 PSNR）
- [[DINOv2]]: 替代 2D 语义对齐目标（差距 0.78 PSNR）

### 方法相关
- [[Latent-Action]]: 隐动作表示核心概念
- [[RoPE]]: Encoder 使用的位置编码
- [[互信息]]: 隐动作质量评估方法
- [[线性探测]]: 动作可恢复性评估方法

### 数据相关
- [[Ego-Exo|Ego-Exo4D]]: 主要多视角预训练数据集
- [[EgoDex]]: 单视角预训练数据集
- [[Assembly101]]: 多视角装配任务数据集

---

## 速查卡片

> [!summary] LAWM-3D (2026)
> - **核心**: 从人类多视角视频学习 3D 感知隐动作，提升机器人世界模型泛化
> - **方法**: 多视角不变 tokenization + VGGT 几何对齐 + 非注入式 RGB-D 重建
> - **结果**: 2B 模型超越 14B DreamDojo，OOD 场景全面领先；策略选择成功率 +17pp
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-08*

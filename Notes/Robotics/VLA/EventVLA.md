---
title: "Event-VLA: Action-Conditioned Event Fusion for Robust Vision-Language-Action Model"
method_name: "EventVLA"
authors: [Jiaxin Liu, Xun Xu, Zhenhao Zhang, Hanqing Wang, Ruiqi Chen, Shi Chang, Weiyu Guo, Laurent Kneip]
year: 2026
venue: arXiv
tags: [vla, event-camera, robustness, multi-modal-fusion, robot-manipulation, low-light, action-conditioned]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.29384
created: 2026-07-01
---

# 论文笔记：Event-VLA: Action-Conditioned Event Fusion for Robust Vision-Language-Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未注明（提交于 arXiv） |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[OpenVLA-OFT]], [[π0]], [[MM-ACT]], [[ResVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.29384) / Code: 未公开 |

---

## 一句话总结

> Event-VLA 通过动作条件化的门控跨注意力机制，将事件相机流融入预训练 [[VLA]] 模型，在弱光等视觉退化条件下大幅提升机器人操控鲁棒性，且几乎不增加推理延迟。

---

## 核心贡献

1. **PREI 事件表征**: 将异步事件流压缩为三通道物理残差图（瞬时、显著、持续），相比传统时间面提供更稳定、更具结构性的表征
2. **动作条件化事件接口**: 通过可学习动作查询从 VLA 推理过程中提取任务相关语义，再以门控跨注意力选择性地将事件 token 注入动作路径，完整保留预训练语言-视觉先验
3. **LIBERO-Cross 基准**: 提出含三级受控光照退化（LL-Mild/Dark/Severe）的新 benchmark，及 RGB-to-Event 模拟器，支撑系统评估

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型假设"光照良好的操控场景、稳定成像条件和高质量 RGB 输入"。在弱光、传感器噪声或动态模糊下，RGB 视觉严重退化导致动作预测失败，而现实工业/家庭场景中照明条件往往不可控。

### 现有方法的局限

- RGB 专有 VLA（如 [[OpenVLA]]、[[π0]]、[[OpenVLA-OFT]]）对光照变化极敏感，在 LL-Severe 条件下成功率从 96% 骤降至 62%
- 直接将事件 token 全局融入 backbone，会破坏预训练语言-视觉语义空间，引入训练崩溃或遗忘问题
- 既有事件-RGB 融合工作将事件作为全局融合模态而非动作导向的信息源

### 本文的动机

[[事件相机]]（Event Camera）具有极高动态范围（>120 dB）和微秒级时间分辨率，在强度信息丢失时仍能保留边缘与运动线索。关键洞见：事件流提供的是**动作时刻的物理残差**，最应注入的是动作路径而非全局语义路径——因此设计"动作条件化"的局部注入机制，而非全局融合。

---

## 方法详解

### 模型架构

Event-VLA 采用**插件式融合**架构，在冻结的预训练 [[VLA|Prismatic-7B VLA]] backbone（初始化自 [[OpenVLA]] 权重）之外增加事件分支：

- **输入**: 语言指令 $\ell$ + RGB 观测 $I_t$ + 本体状态 $s_t$ + 事件历史 $\mathcal{S}_{t-H:t}$
- **Backbone**: [[Prismatic-7B]]（保持冻结语义处理）
- **核心模块**: [[PREI]] 编码器 + [[门控跨注意力|Action-Conditioned Gated Cross-Attention]] 融合接口
- **输出**: [[Action Chunking|动作块]] $\hat{A}_t = [\hat{a}_t, \ldots, \hat{a}_{t+K-1}] \in \mathbb{R}^{K \times d_a}$
- **额外延迟**: 仅 +2.157 ms

### 核心模块

#### 模块1：PREI（Physical Residual Event Integration）

**设计动机**: 利用 [[事件相机]] 的高时间分辨率捕捉 RGB 退化时仍存在的动态边缘和运动信息，同时提供比 [[时间面|Time Surface]] 更稳定的结构性表征。

**具体实现**:

1. 以指数衰减计算像素 $u$ 处的活动图：

$$
A_\tau(u) = \sum_{i:(x_i,y_i)=u} \exp\left(-\frac{t - \tau_i}{\tau}\right)
$$

2. 三通道分解：

$$
E_t^{\mathrm{prei}} = [E_t^{\mathrm{ins}},\, E_t^{\mathrm{sal}},\, E_t^{\mathrm{per}}] \in [0,1]^{H_{\mathrm{img}} \times W_{\mathrm{img}} \times 3}
$$

- $E_t^{\mathrm{ins}}$（**瞬时通道**）：短衰减常数 $\tau_\mathrm{ins}$，捕捉机器人-物体最近运动
- $E_t^{\mathrm{sal}}$（**显著通道**）：用平滑背景归一化突出局部显著活动
- $E_t^{\mathrm{per}}$（**持续通道**）：RGB 退化时保留的短期轮廓（基于事件计数）

3. [[特征蒸馏]] 训练事件编码器 $\Phi_e$，对齐到 VLA 视觉教师特征空间：

$$
Y_t^e = \Phi_e(E_t^{\mathrm{prei}}) = [g_t^e;\, P_t^e]
$$

其中 $g_t^e$ 为全局 token，$P_t^e$ 为补丁 token。

#### 模块2：动作条件化事件接口（Action-Conditioned Event Interface）

**设计动机**: 保留预训练 VLA 全局语义，将事件信息仅注入动作路径，避免全局语义污染。

**输入 token 组合**:

$$
X_t = [Z_t^v;\; z_t^s;\; Z^\ell;\; Z_0^a;\; Q_0^c;\; Q_0^a;\; Q_0^e]
$$

- $Z_t^v$：RGB token
- $z_t^s$：本体状态 token
- $Z^\ell$：语言 token
- $Z_0^a$：动作占位 token
- $Q_0^c, Q_0^a, Q_0^e$：可学习的**公共/动作/事件查询** token

事件 token $Y_t^e$ **不进入 backbone 自注意力**，始终保持在外部。

**Post-backbone 门控跨注意力融合**:

$$
Z_t^{\mathrm{attn}} = \mathrm{CrossAttn}(W_q Z_t^{\mathrm{lm}},\; W_k(Y_t^e + T_{\mathrm{pos}}),\; W_v(Y_t^e + T_{\mathrm{pos}}))
$$

$$
\Gamma_t = \sigma(f_{\mathrm{gate}}(\cdot)), \qquad Z_t^{\mathrm{fused}} = \mathrm{Norm}(Z_t^{\mathrm{lm}} + \Gamma_t \odot Z_t^{\mathrm{attn}})
$$

门控 $\Gamma_t \in [0,1]$ 自适应控制事件信息融合强度。

**查询路由策略**:
- **公共+动作查询** → 路由动作相关 token 至动作预测头
- **公共+事件查询** → 路由事件相关 token 至辅助预测头

#### 模块3：训练正则化辅助目标

**未来 PREI 预测损失** $\mathcal{L}_\mathrm{evt}$: 预测下一帧 PREI，正则化事件表征保留。内容掩码限制计算到有效区域。

**一阶导数一致性损失** $\mathcal{L}_\mathrm{deriv}$: 通过空间梯度强制边缘保留。

---

## 关键公式

### 公式1：[[VLA|标准 VLA 动作预测]]

$$
\hat{A}_t = \pi_\theta(I_t, s_t, \ell), \quad \hat{A}_t = [\hat{a}_t, \ldots, \hat{a}_{t+K-1}] \in \mathbb{R}^{K \times d_a}
$$

**含义**: 标准 VLA 从 RGB 观测 $I_t$、本体状态 $s_t$ 和语言指令 $\ell$ 预测 $K$ 步动作块。

**符号说明**:
- $\pi_\theta$：参数为 $\theta$ 的 VLA 策略
- $K$：动作块长度（实验中 $K=8$）
- $d_a$：动作维度

### 公式2：[[Event Camera|事件增强策略]]

$$
\hat{A}_t = \pi_{\theta,\phi}(I_t, s_t, \ell, \mathcal{S}_{t-H:t})
$$

**含义**: 在标准 VLA 输入基础上引入事件历史流 $\mathcal{S}_{t-H:t}$，$\phi$ 为事件分支参数。

**符号说明**:
- $\mathcal{S}_{t-H:t}$：过去 $H$ 帧内的异步事件流
- $e_i = (x_i, y_i, \tau_i, p_i)$：单个事件（位置、时间、极性）

### 公式3：[[PREI|活动图计算]]

$$
A_\tau(u) = \sum_{i:(x_i,y_i)=u} \exp\left(-\frac{t - \tau_i}{\tau}\right)
$$

**含义**: 用指数衰减加权计算像素 $u$ 的活动强度，衰减常数 $\tau$ 控制时间窗口长短。

**符号说明**:
- $\tau$：衰减常数（不同通道取不同值）
- $\tau_i$：第 $i$ 个事件的时间戳
- $u$：像素坐标

### 公式4：[[PREI|PREI 三通道分解]]

$$
E_t^{\mathrm{prei}} = [E_t^{\mathrm{ins}},\, E_t^{\mathrm{sal}},\, E_t^{\mathrm{per}}] \in [0,1]^{H_{\mathrm{img}} \times W_{\mathrm{img}} \times 3}
$$

**含义**: 将事件流压缩为三通道残差图，覆盖瞬时、显著、持续三种时间尺度的运动信息。

**符号说明**:
- $E_t^{\mathrm{ins}}$：瞬时通道，短衰减
- $E_t^{\mathrm{sal}}$：显著通道，局部归一化
- $E_t^{\mathrm{per}}$：持续通道，累积计数

### 公式5：[[特征蒸馏|事件编码]]

$$
Y_t^e = \Phi_e(E_t^{\mathrm{prei}}) = [g_t^e;\, P_t^e]
$$

**含义**: 事件编码器 $\Phi_e$ 将 PREI 图编码为 VLA 兼容的 token 序列（全局 token + 补丁 token）。

### 公式6：[[Transformer|输入 Token 组合]]

$$
X_t = [Z_t^v;\; z_t^s;\; Z^\ell;\; Z_0^a;\; Q_0^c;\; Q_0^a;\; Q_0^e]
$$

**含义**: VLA backbone 的完整输入 token 序列，包含 RGB、状态、语言、动作占位和三类可学习查询。

### 公式7：[[门控跨注意力|跨注意力事件融合]]

$$
Z_t^{\mathrm{attn}} = \mathrm{CrossAttn}(W_q Z_t^{\mathrm{lm}},\; W_k(Y_t^e + T_{\mathrm{pos}}),\; W_v(Y_t^e + T_{\mathrm{pos}}))
$$

**含义**: 以 backbone 输出 $Z_t^{\mathrm{lm}}$ 为 Query，以事件 token（加位置编码）为 Key/Value 进行跨注意力。

**符号说明**:
- $W_q, W_k, W_v$：投影矩阵
- $T_{\mathrm{pos}}$：位置编码

### 公式8：[[门控跨注意力|门控融合]]

$$
\Gamma_t = \sigma(f_{\mathrm{gate}}(\cdot)), \qquad Z_t^{\mathrm{fused}} = \mathrm{Norm}(Z_t^{\mathrm{lm}} + \Gamma_t \odot Z_t^{\mathrm{attn}})
$$

**含义**: 门控 $\Gamma_t$ 由 sigmoid 函数生成，自适应控制事件融合强度；残差加和后层归一化。

**符号说明**:
- $\sigma$：Sigmoid 激活函数
- $\odot$：逐元素乘法
- $\Gamma_t \in [0,1]^d$：融合门控向量

### 公式9：[[Action Chunking|动作 L1 损失]]

$$
\mathcal{L}_{\mathrm{act}} = \frac{1}{K} \sum_{k=0}^{K-1} \|\hat{a}_{t+k} - a_{t+k}\|_1
$$

**含义**: 主损失函数，对 $K$ 步动作块的 L1 距离求平均。

**符号说明**:
- $\hat{a}_{t+k}$：预测动作
- $a_{t+k}$：真实动作
- $K=8$：动作块长度

### 公式10：[[多任务学习|完整训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{evt}} \mathcal{L}_{\mathrm{evt}} + \lambda_{\mathrm{deriv}} \mathcal{L}_{\mathrm{deriv}}
$$

**含义**: 主任务损失加两个辅助正则损失的加权组合。

**符号说明**:
- $\lambda_{\mathrm{evt}} = 0.1$：未来 PREI 预测损失权重
- $\lambda_{\mathrm{deriv}} = 0.3$：导数一致性损失权重
- $\mathcal{L}_{\mathrm{evt}}$：未来 PREI 预测损失
- $\mathcal{L}_{\mathrm{deriv}}$：空间梯度一致性损失

---

## 关键图表

### Figure 1：接口方案对比

![Figure 1](https://arxiv.org/html/2606.29384v1/x1.png)

**说明**: 对比三种事件-VLA 接口方案。在视觉退化时，事件流保留运动和边缘线索而 RGB 退化。动作查询路由（本文方法）比直接 token 融合和 adapter 融合更有效地将事件信息注入动作路径。

### Figure 2：Event-VLA 整体架构

![Figure 2](https://arxiv.org/html/2606.29384v1/x2.png)

**说明**: 事件流首先压缩为 [[PREI]] 残差图并编码为事件 token；预训练 VLA 用 RGB-语言-本体状态构建语义上下文与查询 token；事件残差通过[[门控跨注意力|动作条件化门控跨注意力接口]]选择性注入动作 slot。

### Figure 3：真实场景部署

![Figure 3](https://arxiv.org/html/2606.29384v1/x3.png)

**说明**: Franka Research 3 机器人真实部署设置，展示腕部 ZED 相机、外置 Orbbec 相机和 DAVIS [[事件相机]] 的配置，以及视觉退化条件下同时获取 RGB 和事件观测的任务示例。

### Figure 4：时间面 vs PREI 定性对比

![Figure 4](https://arxiv.org/html/2606.29384v1/x4.png)

**说明**: 在弱光和正常光条件下，PREI 提供比[[时间面|时间面（Time Surface）]]更稳定、更具结构性的事件表征。

### Figure 5：特征蒸馏训练数据

![Figure 5](https://arxiv.org/html/2606.29384v1/x5.png)

**说明**: [[特征蒸馏]]所用 RGB-事件对数据。左：来自 N-ImageNet 的真实 RGB-事件对；右：用 v2e 从 LIBERO 视频合成的 RGB-事件对。

### Figure 6：RGB-to-Event 模拟器输出

![Figure 6](https://arxiv.org/html/2606.29384v1/x6.png)

**说明**: 在未见 LIBERO 轨迹上的定性可视化。从左到右：输入 RGB 帧、v2e 生成的 PREI 目标、模拟器预测 PREI、v2e 生成的时间面目标、模拟器预测时间面。PREI 质量显著优于时间面。

### Table 1：LIBERO 正常光照性能对比

| 方法 | Spatial | Object | Goal | Long | **Avg.** |
|------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 | 95.4 |
| ResVLA | 96.8 | 98.6 | 97.4 | 92.4 | 96.3 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 93.0 | 96.3 |
| Ours w/o event | 94.4 | 99.2 | 96.8 | 94.4 | 96.2 |
| **Ours** | **94.2** | **99.4** | **97.4** | **94.8** | **96.5** |

**关键发现**: Event-VLA 在正常光照下与最强 RGB-only 基线持平（96.5% vs 96.3%），事件融合不损害原始性能。

### Table 2：LIBERO-Cross 渐进退化性能

| 退化级别 | 方法 | Spatial | Object | Goal | Long | **Avg.** |
|---------|------|---------|--------|------|------|---------|
| LL-Mild | π₀ | 94.0 | 96.2 | 93.6 | 82.8 | 91.7 |
| | OpenVLA-OFT | 93.2 | 97.4 | 96.4 | 88.6 | 93.9 |
| | MM-ACT | 95.8 | 99.2 | 95.6 | 92.8 | 95.9 |
| | **Ours** | **94.4** | **99.4** | **99.0** | **91.6** | **96.1** |
| LL-Dark | π₀ | 89.6 | 94.4 | 91.4 | 81.6 | 89.3 |
| | OpenVLA-OFT | 91.6 | 95.8 | 93.2 | 85.8 | 91.6 |
| | MM-ACT | 94.4 | 97.6 | 96.8 | 83.2 | 93.0 |
| | **Ours** | **94.2** | **98.8** | **98.6** | **94.6** | **96.5** |
| LL-Severe | π₀ | 62.6 | 76.2 | 52.6 | 56.4 | 61.9 |
| | OpenVLA-OFT | 51.4 | 74.6 | 60.4 | 58.2 | 61.2 |
| | MM-ACT | 58.2 | 76.8 | 65.4 | 78.2 | 69.6 |
| | **Ours** | **95.6** | **97.2** | **97.8** | **92.0** | **95.6** |

**关键发现**: 在最严重退化（LL-Severe）条件下，基线方法从 ~96% 骤降至 62-70%，而 Event-VLA 维持 **95.6%**，提升幅度高达 **+25.9%**（相比 π₀）。

### Table 3：消融实验（LL-Severe）

| 类型 | 变体 | SR (%) | Δ延迟 (ms) |
|------|------|--------|------------|
| 表征 | No event | 60.6 | +0 |
| | Time surface | 91.2 | +2.157 |
| | **PREI** | **95.6** | **+2.157** |
| 接口 | Unified enc. | 95.1 | +62.874 |
| | RGB/event adapter | 94.2 | +1.909 |
| | **Query routing** | **95.6** | **+2.157** |
| 查询 | w/o common | 94.5 | +3.708 |
| | w/o event | 95.2 | +1.096 |
| | **Full queries** | **95.6** | **+2.157** |
| 正则 | None | 94.8 | – |
| | w/o mask | 95.1 | – |
| | **Full objective** | **95.6** | – |

**关键发现**: PREI 表征 > 时间面（+4.4%），Query routing 接口在 延迟与精度上均最优，统一编码虽精度接近但延迟高出 **29×**。

### Table 4：真实 Franka 机器人部署

| 方法 | Normal | Low-Light | Near-Dark | Avg. |
|------|--------|-----------|-----------|------|
| π₀ | 75.0 | 55.0 | 15.0 | 48.3 |
| OpenVLA-OFT | 70.0 | 57.5 | 12.5 | 46.7 |
| Ours w/o queries | 70 | 65 | 45 | 60 |
| **Ours** | **72.5** | **70.0** | **52.5** | **65.0** |

**关键发现**: 真实场景 Near-Dark 条件下，基线接近失效（12.5-15%），Event-VLA 维持 **52.5%**，绝对提升 **+37.5-40%**。

### Table 5：特征蒸馏质量

| 事件表征 | MSE ↓ | R@1 (RGB→Event) ↑ | R@1 (Event→RGB) ↑ | Acc@5 ↑ | CKA ↑ |
|---------|-------|------------------|------------------|---------|-------|
| Time Surface | 29.202 | 73.55 | 67.77 | 91.68 | 94.15 |
| **PREI** | **26.870** | **81.36** | **75.91** | **94.25** | **95.33** |

**关键发现**: PREI 在所有特征对齐指标上均优于时间面，验证其更适合与 VLA 视觉特征空间对齐。

### Table 6：RGB-to-Event 模拟器验证

| 表征 | MAE ↓ | RMSE ↓ | PSNR ↑ | MS-SSIM ↑ | Edge-L1 ↓ |
|-----|-------|--------|--------|-----------|----------|
| Time surface | 0.028 | 0.085 | 21.483 | 0.780 | 0.101 |
| **PREI** | **0.008** | **0.043** | **27.858** | **0.897** | **0.031** |

**关键发现**: 模拟器在 PREI 表征上重建精度显著优于时间面（PSNR 提升 6.4 dB），为仿真训练提供高质量事件数据。

### Table 7：下游验证（LL-Severe）

| 事件表征 | Spatial | Object | Goal | Long | Avg. |
|---------|---------|--------|------|------|------|
| Time surface | 90.4 | 92.6 | 93.2 | 88.4 | 91.15 |
| **PREI** | **95.6** | **97.2** | **97.8** | **92.0** | **95.65** |

### Table 8：LIBERO-Cross 退化参数

| 级别 | 曝光降低 | SNR @18% gray | 运动模糊 |
|-----|---------|---------------|---------|
| LL-Mild | 2.32 EV / −6.9 dB | 15.5 dB | 1 px |
| LL-Dark | 3.24 EV / −9.7 dB | −0.2 dB | 2 px |
| LL-Severe | 4.78 EV / −14.4 dB | −11.9 dB | 3 px |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 个子任务套件 | 仿真操控 benchmark | 正常光照评估 |
| LIBERO-Cross（新） | 4 × 3 退化级别 | 受控光照退化 | 退化鲁棒性评估 |
| N-ImageNet | 真实 RGB-事件对 | 来自 DVS 相机 | 特征蒸馏训练 |
| LIBERO (v2e 合成) | LIBERO 轨迹 | v2e 合成事件 | 模拟器训练 |
| Franka Research 3 | 4 任务 × 3 光照 × 10 轮 | 真实 DAVIS 事件相机 | 真实场景验证 |

### 实现细节

- **Backbone**: [[Prismatic-7B]]（初始化自 [[OpenVLA]] 权重，冻结）
- **事件编码器**: 与 VLA 视觉编码器同架构
- **动作块长度**: $K = 8$
- **Batch Size**: 64
- **训练步数**: ~140k steps（每个子 benchmark 独立训练）
- **硬件**: 8× H100 GPU
- **损失权重**: $\lambda_{\mathrm{evt}} = 0.1$，$\lambda_{\mathrm{deriv}} = 0.3$
- **真实机器人**: Franka Research 3 + ZED 腕部相机 + Orbbec 外置相机 + DAVIS 事件相机

### 可视化结果

- 在 LL-Severe 条件下，RGB 观测几乎变为纯噪声，而 PREI 仍清晰呈现物体轮廓和机器人末端运动轨迹
- 时间面在弱光下结构紊乱，PREI 三通道各司其职保持稳定
- 真实场景中 Near-Dark 条件下，RGB 帧肉眼几乎不可分辨，DAVIS 事件相机依然捕捉到清晰运动信号

---

## 批判性思考

### 优点

1. **优雅的架构设计**: 动作查询路由方案将事件信息精准注入动作路径，以极小延迟代价（+2 ms）实现显著性能提升
2. **极强的鲁棒性提升**: LL-Severe 下 +25.9% 的绝对增益，且正常光照无损耗，实用价值突出
3. **PREI 表征简洁有效**: 三通道物理直觉清晰，特征蒸馏使事件编码器与 VLA 语义空间对齐

### 局限性

1. **依赖 RGB-to-Event 模拟器**: 训练数据来自仿真而非真实事件流，domain gap 可能限制泛化
2. **仅针对光照退化**: 未验证运动模糊、遮挡、视角变化等其他退化类型
3. **硬件依赖**: 需要额外 [[事件相机]]（DAVIS），增加部署成本和系统复杂度
4. **真实评估规模有限**: 仅 4 个任务、单机器人，泛化性有待扩展验证

### 潜在改进方向

1. 扩展 LIBERO-Cross 至更多退化类型（雾、遮挡、快速运动），验证方法通用性
2. 研究直接从真实事件流训练的方案，减少对模拟器的依赖
3. 探索将事件信息用于 VLA 奖励信号或在线微调，提升零样本泛化

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、硬件均在论文中说明）
- [x] 数据集可获取（LIBERO 公开，N-ImageNet 公开；LIBERO-Cross 为自建）

---

## 关联笔记

### 基于

- [[OpenVLA]]: 使用其 Prismatic-7B 权重初始化，Event-VLA 以此为 backbone
- [[事件相机]]: 核心传感模态，提供高动态范围视觉信号
- [[LIBERO]]: 主评测 benchmark，扩展为 LIBERO-Cross

### 对比

- [[OpenVLA-OFT]]: RGB-only 微调基线，LL-Severe 下性能骤降
- [[π0]]: 扩散策略 VLA，同样对光照敏感
- [[MM-ACT]]: 多模态动作生成，比其他 RGB-only 方法稍强但仍无法应对 LL-Severe

### 方法相关

- [[PREI]]: 物理残差事件集成，本文提出的核心事件表征
- [[门控跨注意力]]: 事件融合的核心机制
- [[特征蒸馏]]: 训练事件编码器与 VLA 视觉空间对齐
- [[Action Chunking]]: VLA 预测的动作块输出形式
- [[时间面|Time Surface]]: PREI 的对比基线事件表征

### 硬件/数据相关

- [[DAVIS 事件相机]]: 真实部署所用传感器
- [[N-ImageNet]]: 特征蒸馏训练集（真实 RGB-事件对）

---

## 速查卡片

> [!summary] Event-VLA (arXiv 2026)
> - **核心**: 事件相机 + 动作条件化门控跨注意力，让 VLA 在弱光下鲁棒
> - **方法**: PREI 三通道事件表征 + 查询路由接口（冻结 VLA backbone，仅融合事件）
> - **结果**: LL-Severe 维持 95.6%（基线骤降至 62-70%），真实机器人 Near-Dark +37.5%
> - **代码**: 未公开

---

*笔记创建时间: 2026-07-01*

---
title: "GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving"
method_name: "GuidedWAM"
authors: [Xiyan Su, Finn Rasmus Schäfer, Hojun Lim, Frank Diermeyer, Johannes Betz, Markus Lienkamp]
year: 2026
venue: arXiv
tags: [world-action-model, autonomous-driving, flow-matching, energy-guidance, sampling-time-guidance, navsim]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.07912
created: 2026-11-07
---

# 论文笔记：GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Technical University of Munich |
| 日期 | November 2026 |
| 项目主页 | 无 |
| 对比基线 | [[DriveWAM]] / [[DriveLaW]] / [[SimWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.07912) / Code: 未发布 |

---

## 一句话总结

> GuidedWAM 在不重新训练冻结 [[世界动作模型（WAM）]] 的前提下，通过在采样时注入可微能量梯度，将 NAVSIM v2 hard split 的 EPDMS 提升 1.8–3.6 分。

---

## 核心贡献

1. **Jacobian-free 能量引导**: 将驾驶目标建模为作用于单步干净估计的可微能量，省去对冻结模型的反向传播，仅需前向推理
2. **三重稳定化机制**: 梯形引导调度（Guidance Scheduling）+ 单能量梯度归一化（Per-energy Normalization）+ 速度有界修正（Velocity-bounded Correction），防止采样崩溃
3. **模型无关框架**: 在三个公开发布的冻结 [[世界动作模型（WAM）]] 上验证，推理时间增加不超过 10%

---

## 问题背景

### 要解决的问题

[[世界动作模型（WAM）]] 将驾驶策略和视觉世界模型合并为一个生成网络，训练后其驾驶目标固定。若要加入新的驾驶目标（如可行驶区域约束），通常需要完整重新训练。

### 现有方法的局限

- **重新训练代价大**: 修改目标需要完整训练循环
- **后处理投影（Post-hoc Projection）**: 在采样完成后对轨迹做硬约束投影，会引起突变（sudden lateral change），牺牲舒适度
- **Classifier Guidance**: 需要对冻结模型做 Jacobian 计算，计算代价高（DriveWAM 上高达 72.5 ms/step vs Jacobian-free 1.9 ms/step）

### 本文的动机

在 [[Flow Matching|flow matching]] 采样的中间步可以访问单步干净估计 $\hat{\tau}_0$，能量梯度可以直接作用于此，不需要完整 Jacobian，从而以极小代价注入新目标。

---

## 方法详解

### 模型架构

GuidedWAM 是一个**采样时引导框架**，不修改任何模型权重：

- **基础模型**: 任意冻结的 [[世界动作模型（WAM）]]（[[DriveWAM]]、[[SimWAM]]、[[DriveLaW]]）
- **引导信号**: 可微能量函数 $\mathcal{E}(\hat{\tau}_0)$，作用于解码后的干净轨迹（每个路径点的位置和朝向）
- **注入位置**: [[Flow Matching]] 采样循环的每个去噪步
- **推理**: 采样从 $t=1$（纯噪声）积分到 $t=0$（干净轨迹），每步修改速度场

### 核心模块

#### 模块1: 单步干净估计（One-step Clean Estimate）

**设计动机**: 利用 [[Flow Matching]] 在每个去噪步的结构，获得无需额外模型调用的干净轨迹近似。

**具体实现**:
- 中间噪声状态 $\tau_t = (1-t)\tau_0 + t\varepsilon$
- 利用网络预测直接估计干净轨迹（见 Eq.2）
- 在估计上评估能量并反传梯度

#### 模块2: Jacobian-free 引导（无需 Jacobian 的速度更新）

**设计动机**: 完整 Jacobian $J_t = I - t \frac{\partial v_\theta}{\partial \tau_t}$ 需要对冻结模型反向传播，代价极高（memory 和时间均显著增加）。

**具体实现**:
- 将 $J_t$ 以单位矩阵替代（即 $J_t \approx I$）
- 能量梯度 $\nabla_{\hat{\tau}_0}\mathcal{E}$ 直接加到速度场
- 实验表明这在大多数模型上与完整 Jacobian 形式等效（Table 5）

#### 模块3: 稳定化三重机制

**设计动机**: 规范化调度 $\lambda^*(t) = \gamma \cdot t/(1-t)$ 在 $t \to 1$ 时发散，直接使用会导致采样不稳定。

**具体实现**:
- 梯形调度窗口（Trapezoidal Window）: 在噪声水平中段最大，两侧线性 ramp
- 单能量归一化: 消除不同能量尺度的影响（见 Eq.7）
- 速度有界修正: 以参考速度范数限制修正幅度（见 Eq.8）

---

## 关键公式

### 公式1: [[Flow Matching|速度-Score 等价关系]]

$$
v(\tau_t, t) = -\frac{\tau_t + t \cdot s_t(\tau_t)}{1-t}
$$

**含义**: flow matching 中速度场与 score 函数的等价关系，Score 的加性修正 $\Delta s$ 对应速度修正 $-\frac{t}{1-t}\Delta s$

**符号说明**:
- $v(\tau_t, t)$: 速度场（flow matching 的核心预测量）
- $s_t(\tau_t)$: score 函数 $\nabla_{\tau_t} \log p_t(\tau_t)$
- $\tau_t$: $t$ 时刻的噪声轨迹状态

### 公式2: [[Energy Guidance|单步干净估计]]

$$
\hat{\tau}_0(\tau_t) = \tau_t - t \cdot v_\theta(\tau_t, t, X)
$$

**含义**: 利用冻结模型 $v_\theta$ 的单次前向推理，从当前噪声状态直接估计干净轨迹，是 Jacobian-free 引导的关键中间量

**符号说明**:
- $\hat{\tau}_0$: 单步干净轨迹估计
- $v_\theta$: 冻结的 WAM 速度网络
- $X$: 条件输入（相机观测、地图等）

### 公式3: [[Energy Guidance|目标倾斜分布]]

$$
p_\gamma(\tau_0 \mid X) \propto p(\tau_0 \mid X) \cdot e^{-\gamma \mathcal{E}(\tau_0)}
$$

**含义**: 通过能量函数 $\mathcal{E}$ 对原始模型分布施加软约束，使高能量（违规）轨迹概率降低

**符号说明**:
- $\gamma$: 引导强度超参数
- $\mathcal{E}(\tau_0)$: 驾驶目标能量函数（越低越好）
- $p(\tau_0 \mid X)$: 冻结模型的原始分布

### 公式4: [[Energy Guidance|条件边际 Score]]

$$
s_t^\gamma(\tau_t) = s_t(\tau_t) + \nabla_{\tau_t} \log \mathbb{E}_{p(\tau_0 \mid \tau_t, X)}\big[e^{-\gamma \mathcal{E}(\tau_0)}\big]
$$

**含义**: 引导后的 Score 函数理论形式，期望项难以直接计算，需要用单步干净估计近似

**符号说明**:
- $s_t^\gamma$: 引导后的 score
- $p(\tau_0 \mid \tau_t, X)$: 给定 $\tau_t$ 的后验分布（以 $\hat{\tau}_0$ 近似）

### 公式5: [[Energy Guidance|完整 Jacobian 梯度]]

$$
\nabla_{\tau_t} \mathcal{E}(\hat{\tau}_0) = J_t^\top \nabla_{\hat{\tau}_0} \mathcal{E}(\hat{\tau}_0), \quad J_t = I - t \cdot \frac{\partial v_\theta}{\partial \tau_t}
$$

**含义**: 通过链式法则将能量对干净估计的梯度反传回噪声状态，完整形式需要对冻结模型求 Jacobian

**符号说明**:
- $J_t$: Jacobian 矩阵，涉及对 $v_\theta$ 的反向传播
- $\nabla_{\hat{\tau}_0} \mathcal{E}$: 能量对干净估计的梯度

### 公式6: [[Energy Guidance|Jacobian-free 速度更新]]

$$
v' = v_\theta + \lambda(t) \cdot \nabla_{\hat{\tau}_0} \mathcal{E}(\hat{\tau}_0), \qquad \lambda^*(t) = \gamma \cdot \frac{t}{1-t}
$$

**含义**: 以 $J_t \approx I$ 简化后的引导速度更新公式，能量梯度直接加到速度场；规范调度 $\lambda^*$ 随噪声水平 $t$ 增大而增大

**符号说明**:
- $\lambda(t)$: 引导强度调度函数，实际使用梯形窗口替代 $\lambda^*$
- $\nabla_{\hat{\tau}_0} \mathcal{E}$: 对干净估计的能量梯度（仅需前向传播）

### 公式7: [[Energy Guidance|多能量归一化加权组合]]

$$
\Delta v = \sum_{k=1}^{K} w_k \cdot \frac{\nabla_{\hat{\tau}_0}\mathcal{E}_k(\hat{\tau}_0)}{\lVert \nabla_{\hat{\tau}_0}\mathcal{E}_k(\hat{\tau}_0) \rVert_2}
$$

**含义**: 多个驾驶目标能量的归一化加权组合，消除不同能量尺度差异，$\lambda(t)$ 携带整体强度

**符号说明**:
- $K$: 能量函数数量
- $w_k$: 第 $k$ 个能量的相对权重
- $\lVert \cdot \rVert_2$: L2 范数归一化，使各能量贡献可比

### 公式8: [[Energy Guidance|速度有界修正]]

$$
\Delta v \leftarrow \min\!\Big(1,\; \frac{c \cdot \rho}{\lambda(t) \lVert \Delta v\rVert_2}\Big) \Delta v, \qquad \rho = \lVert v_\theta(\tau_t) \rVert_2
$$

**含义**: 以参考速度范数 $\rho$ 限制修正幅度，防止引导步过大破坏采样轨迹；$\rho$ 在第一个引导步记录后固定

**符号说明**:
- $c$: 跨模型共享的比例系数
- $\rho$: 第一步引导时记录的参考速度范数（固定不更新）

---

## 关键图表

### Figure 1: Guidance vs Post-hoc Projection / 引导与后处理投影对比

![Figure 1](https://arxiv.org/html/2611.07912v1/fig_guidance_vs_projection_6333f19e.png)

**说明**: 在一个 NAVSIM v2 hard-split 场景中对比 [[Energy Guidance]] 与后处理投影的轨迹行为。左：前视摄像头投影三条轨迹；右：BEV 视角下的可行驶区域能量场。
- 灰色（Stock plan）: 偏离可行驶区域
- 蓝色（后处理投影）: 在违规点突发横向修正（急刹式修正）
- 橙色（GuidedWAM）: 整段轨迹平滑修正，到达相同评分

### Figure 2: EPDMS vs Guidance Strength / 引导强度扫描

*图片 URL 暂不可得，详见 arXiv HTML*

**说明**: 三个模型（DriveWAM、SimWAM、DriveLaW）在单相机可行驶区域能量下，EPDMS 随引导强度 $\lambda$（对数坐标轴）的变化曲线。虚线为各模型 stock 基准；空心标记为 Table 1 中汇报的运行点。各模型最优 $\lambda$ 差异较大（DriveWAM: $\lambda=4$，SimWAM 和 DriveLaW: $\lambda=0.05$）。

### Figure 3: Window Shape vs Guidance Strength / 调度窗口形状消融

*图片 URL 暂不可得，详见 arXiv HTML*

**说明**: 在 DriveWAM 的 40 个 DAC 失败场景中，不同调度窗口形状与引导强度组合的对比。
- 上图: Failure rate（相对 stock 计划抖动超过 3 倍的场景比例）
- 下图: 相对 stock 计划的 ADE/FDE
运行强度 $\lambda=4$。

### Table 1: NAVSIM v2 Hard Split 主要结果

| Method (Venue) | Stage | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TransFuser (TPAMI'23) | S1 | 96.2 | 79.5 | 99.1 | 99.5 | 84.1 | 95.1 | 94.2 | 97.5 | 79.1 | 25.1±0.0 |
| | S2 | 77.7 | 70.2 | 84.2 | 98.0 | 85.1 | 75.6 | 45.4 | 95.7 | 75.9 | — |
| DiffusionDrive (CVPR'25) | S1 | 96.8 | 86.0 | 98.8 | 99.3 | 84.0 | 95.8 | 96.7 | 97.6 | 79.6 | 27.5±0.0 |
| | S2 | 80.1 | 72.8 | 84.4 | 98.4 | 85.9 | 76.6 | 46.4 | 96.3 | 72.8 | — |
| GuideFlow (CVPR'26) | S1 | 96.6 | 80.5 | 96.3 | 99.3 | 82.3 | 94.9 | 91.5 | 97.7 | 67.8 | 27.1±0.0 |
| | S2 | 87.3 | 76.7 | 88.8 | 99.2 | 84.3 | 85.1 | 49.7 | 93.1 | 44.5 | — |
| ReCogDrive (ICLR'26) | S1 | 96.4 | 78.9 | 98.7 | 99.8 | 82.6 | 95.6 | 94.4 | 97.6 | 74.2 | 25.7±0.0 |
| | S2 | 80.2 | 65.0 | 82.4 | 98.7 | 85.2 | 76.9 | 43.8 | 96.6 | 71.8 | — |
| SGDrive (CVPR'26) | S1 | 95.8 | 87.6 | 97.8 | 99.8 | 84.4 | 94.7 | 92.9 | 97.8 | 28.9 | 25.5±0.0 |
| | S2 | 79.4 | 65.4 | 79.1 | 98.9 | 88.9 | 75.3 | 42.7 | 96.4 | 29.6 | — |
| DriveFine (ECCV'26) | S1 | 97.6 | 90.0 | 99.1 | 99.3 | 84.9 | 96.7 | 97.3 | 97.6 | 72.0 | 30.5±0.0 |
| | S2 | 82.1 | 71.3 | 84.8 | 98.4 | 88.1 | 74.3 | 47.2 | 96.8 | 72.8 | — |
| DriveVLA-M0 (MM'26) | S1 | 98.9 | 96.2 | 99.7 | 100.0 | 73.8 | 99.1 | 94.2 | 97.3 | 64.4 | 47.0±0.0 |
| | S2 | 91.1 | 89.5 | 93.1 | 98.6 | 65.5 | 89.2 | 52.7 | 98.6 | 72.1 | — |
| DriveLaW (CVPR'26) | S1 | 97.1 | 88.4 | 99.2 | 99.6 | 84.4 | 97.1 | 96.4 | 97.8 | 66.7 | 30.0±0.0 |
| | S2 | 82.3 | 67.4 | 83.8 | 98.1 | 84.7 | 78.6 | 46.4 | 96.4 | 57.9 | — |
| **DriveLaW + EG (ours)** | S1 | 96.6±0.1 | 94.9±0.2 | 99.1±0.1 | 99.6±0.0 | 84.1±0.1 | 95.7±0.1 | 95.0±0.5 | 97.8±0.0 | 61.9±4.1 | **33.6±0.4** |
| | S2 | 82.7±0.3 | 71.5±0.4 | 83.8±0.1 | 98.1±0.0 | 84.5±0.1 | 79.7±0.3 | 45.0±0.4 | 96.6±0.0 | 57.8±0.6 | — |
| DriveWAM (arXiv'26) | S1 | 97.2 | 92.7 | 99.6 | 99.3 | 84.6 | 95.8 | 98.2 | 97.8 | 81.3 | 35.9±0.0 |
| | S2 | 81.9 | 76.4 | 86.1 | 98.2 | 87.0 | 77.4 | 50.5 | 95.0 | 73.0 | — |
| **DriveWAM + EG (ours)** | S1 | 97.0±0.2 | 95.9±0.1 | 99.5±0.1 | 99.3±0.0 | 84.6±0.2 | 96.6±0.1 | 97.2±0.1 | 97.8±0.0 | 79.9±0.7 | **37.8±0.8** |
| | S2 | 82.3±0.3 | 77.6±0.5 | 85.3±0.9 | 98.6±0.2 | 87.4±0.3 | 77.5±0.4 | 48.0±0.7 | 95.2±0.2 | 71.4±0.8 | — |
| SimWAM (arXiv'26) | S1 | 97.8 | 92.4 | 99.6 | 99.6 | 83.8 | 96.2 | 97.8 | 97.8 | 76.0 | 37.8±0.0 |
| | S2 | 82.1 | 78.5 | 87.5 | 98.4 | 86.2 | 78.7 | 50.0 | 96.4 | 72.8 | — |
| **SimWAM + EG (ours)** | S1 | 97.7±0.1 | 95.7±0.6 | 99.5±0.1 | 99.6±0.0 | 83.9±0.1 | 96.5±0.3 | 98.2±0.4 | 97.8±0.0 | 76.6±1.4 | **39.6±0.5** |
| | S2 | 82.1±0.3 | 79.8±0.3 | 87.7±0.1 | 98.4±0.0 | 86.5±0.1 | 78.6±0.1 | 50.7±0.1 | 96.2±0.2 | 70.4±1.3 | — |

**说明**: S1=真实场景，S2=合成场景。EG = Energy Guidance（本文方法）。DAC 显著提升说明引导有效减少可行驶区域违规。碰撞指标（NC）无显著提升。

### Table 2: 引导与后处理投影对比（特权信息下）

| Model | Method | EPDMS↑ | Δ | Off-road↓ | EC↑ |
|---|---|---|---|---|---|
| DriveLaW | stock | 30.0 | — | 26.0 | 57.9 |
| DriveLaW | guidance | 34.8 | +4.8 | 22.7 | **58.0** |
| DriveLaW | projection | **37.7** | +7.7 | **16.6** | 51.2 |
| DriveWAM | stock | 35.9 | — | 20.9 | **73.0** |
| DriveWAM | guidance | 39.2 | +3.3 | 18.5 | 71.4 |
| DriveWAM | projection | **40.3** | +4.3 | **15.2** | 66.0 |
| SimWAM | stock | 37.8 | — | 18.7 | **72.8** |
| SimWAM | guidance | 39.2 | +1.5 | 18.0 | 71.6 |
| SimWAM | projection | **41.0** | +3.2 | **14.0** | 65.3 |

**关键发现**: 投影（projection）EPDMS 更高但舒适度（EC）损失显著（下降 5–7 点），引导（guidance）在 EPDMS 和 EC 之间取得更好平衡

### Table 3: 能量组合消融（特权信息下的 EPDMS）

LK=车道保持，DA=可行驶区域，DD=行驶方向

| Model | stock | LK | DA | DD | LK+DA | DA+DD | LK+DA+DD |
|---|---|---|---|---|---|---|---|
| DriveLaW | 30.0 | 31.1 | 34.0 | 30.6 | 33.3 | **34.8** | 33.0 |
| DriveWAM | 35.9 | 37.5 | 39.2 | 36.4 | 39.5 | **40.0** | 39.6 |
| SimWAM | 37.8 | 37.9 | 39.2 | 37.8 | 38.2 | **39.6** | 38.9 |

**关键发现**: DA（可行驶区域能量）单独效果最强；DA+DD 组合效果最优；三能量组合反而略有下降（可能存在能量冲突）

### Table 4: 交叉拟合选择 vs 完整数据集最优

| Model | Full-split best | Cross-fitted | 95% 置信区间 |
|---|---|---|---|
| DriveLaW | 34.1 | 33.5 | [+1.9, +6.3] |
| DriveWAM | 38.2 | 37.7 | [+0.5, +4.2] |
| SimWAM | 40.0 | 40.0 | [+0.8, +3.8] |

**关键发现**: 交叉拟合选择（避免过拟合评估集）与全数据最优接近，说明超参选择泛化性好

### Table 5: Jacobian-free 与完整 Jacobian 形式对比

| Model | Form | EPDMS↑ | Guidance/step↓ | Forward/step↓ | Memory↓ |
|---|---|---|---|---|---|
| DriveLaW | Jacobian-free | 34.1 | 0.1 ms | 48 ms | 8.2 GB |
| DriveLaW | Jacobian | 29.9 | 50.4 ms | 42.4 ms | 9.8 GB |
| DriveWAM | Jacobian-free | 37.7 | 1.9 ms | 36 ms | 13.7 GB |
| DriveWAM | Jacobian | 36.3 | 72.5 ms | 35.9 ms | 14.3 GB |
| SimWAM | Jacobian-free | 40.0 | 1.4 ms | 16.3 ms | 12.7 GB |
| SimWAM | Jacobian | 40.0 | 14.3 ms | 16.2 ms | 12.7 GB |

测试环境: NVIDIA RTX 4090。Memory 为采样期间峰值显存。

**关键发现**: Jacobian-free 推理时间仅增加 0.1–1.9 ms（不超过 10%），而完整 Jacobian 增加 14–72 ms；DriveLaW 完整 Jacobian 形式 EPDMS 甚至低于基线（梯度不稳定）

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[NAVSIM]] v2 hard split | S1（真实）+ S2（合成） | 仅含困难场景，含可行驶区域违规等边缘案例 | 评估（无训练） |

### 实现细节

- **基础模型**: 使用公开发布的冻结检查点（DriveWAM、SimWAM、DriveLaW）
- **硬件**: NVIDIA RTX 4090
- **超参**: 引导强度 $\lambda$ 通过 3 折交叉验证在验证集上选取
- **能量**（主要结果）: 单相机可行驶区域能量（camera-only，不使用特权地图信息）
- **种子**: 结果取 3 个采样种子平均 ± 标准差
- **比例系数**: $c=0.4$（跨模型共享）

### 可视化结果

Figure 1 展示的典型场景中，GuidedWAM 产生的引导轨迹（橙色）相比后处理投影更平滑，轨迹从整体上向可行驶区域偏移，而非在最后时刻急剧修正。

---

## 批判性思考

### 优点

1. **无训练代价**: 完全不需要重新训练冻结模型，可即插即用添加新驾驶目标
2. **计算效率高**: 推理时间增加 <10%，仅需前向传播
3. **泛化性好**: 在三个架构差异显著的 WAM 上均有效，框架模型无关

### 局限性

1. **开环评估**: 仅在 NAVSIM 开环 benchmark 上验证，未测试闭环实车或仿真
2. **软约束无硬保证**: 基于梯度的引导是软约束，无法保证轨迹绝对可行驶
3. **加权求和组合**: 当多个能量存在冲突时（如 LK+DA+DD 组合下降），无法自动解决
4. **仅支持确定性采样**: 不适用于随机采样（stochastic samplers）
5. **碰撞指标无提升**: DAC 显著提升但 NC 无改善，说明可行驶约束与碰撞避免是不同层次问题

### 潜在改进方向

1. 探索闭环评估（CARLA、nuPlan 等）中的实际收益
2. 设计基于 [[Control Barrier Function|CBF]] 的硬约束组合机制
3. 自适应能量权重（基于场景检测动态切换）

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（依赖第三方 WAM 检查点）
- [x] 训练细节完整（方法细节详尽）
- [x] 数据集可获取（NAVSIM v2 公开）

---

## 关联笔记

### 基于

- [[DriveWAM]]: 被引导的 WAM 之一，flow matching 双分支架构
- [[SimWAM]]: 被引导的 WAM 之一，视频专家 + 轻量动作专家
- [[DriveLaW]]: 被引导的 WAM 之一，从视频生成器潜空间规划
- [[Flow-Matching]]: GuidedWAM 的基础生成框架
- [[Classifier-Free Guidance]]: 相关控制方法，GuidedWAM 是其 flow matching 下的能量引导变体

### 对比

- [[GuideFlow]]: CVPR'26 classifier guidance 方法，需要条件/无条件双模型
- DiffusionDrive: CVPR'25 基于扩散的驾驶规划方法
- 后处理投影（Post-hoc Projection）: 采样后硬约束，EPDMS 更高但舒适度差

### 方法相关

- [[Energy Guidance]]: 本文核心方法
- [[Flow-Matching]]: 基础生成框架
- [[世界动作模型（WAM）]]: 被引导的基础模型类型

### 硬件/数据相关

- [[NAVSIM]]: 评估 benchmark，v2 hard split

---

## 速查卡片

> [!summary] GuidedWAM (arXiv 2026)
> - **核心**: 采样时注入可微能量梯度，无需重训冻结 WAM
> - **方法**: Jacobian-free 速度更新 + 梯形调度 + 梯度归一化 + 速度有界修正
> - **结果**: NAVSIM v2 hard split EPDMS +1.8 ~ +3.6，推理时间增加 <10%
> - **代码**: 未发布

---

*笔记创建时间: 2026-11-07*

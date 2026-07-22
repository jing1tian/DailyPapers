---
title: "Test-Time Scaling for World Action Models via Zero-Shot Geometric Verification"
method_name: "GeoBoN"
authors: [Zesen Zhao, Minkyoung Cho, Hui Shen, Boyuan Zheng, Kunxiao Gao, Yulong Cao, Z. Morley Mao]
year: 2026
venue: CVPR 2026 EAI Workshop (Extended)
tags: [test-time-scaling, world-action-model, best-of-n, geometric-verification, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.17454v1
created: 2026-07-22
---

# 论文笔记：Test-Time Scaling for World Action Models via Zero-Shot Geometric Verification

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Michigan |
| 日期 | July 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Cosmos-Policy]]、[[Motus]]、[[LingBot-VA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.17454) |

---

## 一句话总结

> 利用跨视角深度重投影一致性对 WAM 候选动作排序，无需训练即可在多个机器人操作基准上提升成功率，并通过动作-未来门控大幅降低额外采样开销。

---

## 核心贡献

1. **跨视角几何评分器（GeoBoN）**: 用冻结 [[VGGT]] 几何基础模型计算主摄与腕摄深度重投影不一致度，作为无需训练、模型无关的候选动作排序信号。
2. **动作-未来门控（Action-Future Gate）**: 通过计算预测末端执行器位移与 [[光流 (Optical Flow)|光流]] 运动的余弦一致性，仅在检测到不一致时触发额外采样，恢复约 75% 性能增益同时只在 26% 决策点额外采样。
3. **扩展失败分析**: 识别出"假低分选择（False Low-Score Selection）"机制，解释了 [[Best-of-N]] 候选池扩大时性能饱和的根本原因。

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model（WAM）]] 联合预测未来视觉观测与动作块，在执行前暴露了候选动作及其预测的视觉后果。如何在不额外训练的情况下利用这些丰富的中间表示对候选动作进行选择，是提升 WAM 推理质量的核心问题。

### 现有方法的局限

- **模型特定值头（Value Head）**：如 [[Cosmos-Policy]] 内置的值头，需要特定模型的附加训练，无法迁移到其他 WAM 骨干网络。
- **基于置信度的排序**：直接使用几何模型的置信度分数排序，在实验中表现不稳定，会在部分 benchmark 上降低在线成功率。
- **未来共识（Future-Consensus）**：同期工作，通过多视角一致性测量排序，但在 RoboCasa 上（−2.3%～−5.1%）表现劣于 GeoBoN。

### 本文的动机

WAM 既然预测了主摄和腕摄的未来画面，这两路视角必须在几何上相互一致——若从腕摄投影的 3D 点与主摄直接预测的深度不吻合，说明该 rollout 内部不一致，不值得执行。这一物理约束来自外部冻结的几何基础模型（[[VGGT]]），完全无需额外训练。

---

## 方法详解

### 整体框架

GeoBoN 是一个两阶段推理框架，工作在任意 [[WAM]] 骨干之上：

- **输入**：语言指令 $l$ + 多视角观测 $\{o^{\text{pri}}, o^{\text{wri}}\}$
- **阶段 1 — 动作-未来门控（Section 3.1）**：评估初始 rollout 是否自洽，若一致则直接执行，若不一致则触发额外采样
- **阶段 2 — 跨视角几何评分器（Section 3.2）**：对所有候选 rollout 用 [[VGGT]] 评分，选择深度重投影误差最小者执行
- **输出**：最优候选 rollout 的动作块 $\mathbf{a}_{1:H}$

### 核心模块

#### 模块 1：动作-未来门控（Action-Future Gate）

**设计动机**：利用 [[光流 (Optical Flow)|光流]] 与投影动作位移的余弦一致性，以低开销判断初始 rollout 是否可信，避免对所有输入都运行代价高昂的几何评分。

**具体实现**：
- 对 WAM 初始 rollout $\tau_1 = (\hat{I}_1^{\text{pri}}, \hat{I}_1^{\text{wri}}, \mathbf{a}_{1,1:H})$，计算末端执行器在主摄像素坐标系下的位移 $\Delta\mathbf{u}_{1,r}$
- 用 Farneback 算法计算末端执行器轨迹胶囊区域 $\mathcal{M}_{1,r}$（半径 $\rho_{\text{cap}}=30$ 像素）内的平均 [[光流 (Optical Flow)|光流]] $\bar{\mathbf{f}}_{1,r}$
- 计算余弦一致性 $c_{1,r}$；若 $c_{1,r} < \tau_{\text{gate}} = -0.2$，触发额外采样 $N_{\text{max}}-1$ 个 rollout

#### 模块 2：跨视角几何评分器（Cross-View Geometric Evaluator）

**设计动机**：[[跨视角深度重投影|跨视角深度重投影]] 一致性是一个纯几何约束，不依赖任务标签或真实观测，用冻结 [[VGGT]] 计算可量化的三维场景一致性得分。

**具体实现**：
- 对每个候选 rollout $\tau_i$，用冻结的 VGGT-Ω 同时处理预测的主摄图像 $\hat{I}_i^{\text{pri}}$ 和腕摄图像 $\hat{I}_i^{\text{wri}}$
- 获取两路深度图：直接预测的主摄深度 $d_i^{\text{vggt}}(p)$ 和从腕摄视角 3D 点投影到主摄的深度 $d_i^{\text{proj}}(p)$
- 在有效像素集 $\Omega_i$（正深度且置信度 $> \gamma_{\text{conf}} = 0.5$）上计算对数比绝对误差 $e_{\text{depth}}(\tau_i)$
- 选择 $e_{\text{depth}}$ 最小的候选执行

#### 模块 3：推理流程（Inference Procedure）

三种操作模式：
1. **基线**：直接执行初始 rollout $\tau_1$，不额外采样
2. **固定预算 GeoBoN**：始终采样 $N$ 个候选，全部经几何评分后选最优
3. **门控 GeoBoN**：先用动作-未来门控；若一致则执行 $\tau_1$；若不一致则采样 $N_{\text{max}}-1$ 个额外 rollout 并应用几何评分

---

## 关键公式

### 公式 1：[[WAM|Rollout 表示]]

$$
\tau_i = \left(\hat{I}_i^{\text{pri}},\; \hat{I}_i^{\text{wri}},\; \mathbf{a}_{i,1:H}\right)
$$

**含义**：WAM 第 $i$ 个 rollout 由预测的主摄未来帧、腕摄未来帧和动作序列三元组构成。

**符号说明**：
- $\hat{I}_i^{\text{pri}}$：第 $i$ 个 rollout 预测的主摄（primary）未来图像
- $\hat{I}_i^{\text{wri}}$：第 $i$ 个 rollout 预测的腕摄（wrist）未来图像
- $\mathbf{a}_{i,1:H}$：预测的 $H$ 步动作序列

---

### 公式 2：[[光流 (Optical Flow)|末端执行器投影位移]]

$$
\Delta\mathbf{u}_{1,r} = \pi_{\text{pri}}\!\left(\mathbf{x}_{1,r,H}\right) - \pi_{\text{pri}}\!\left(\mathbf{x}_{r,0}\right)
$$

**含义**：将末端执行器的三维位置投影到主摄像素坐标系，得到 $H$ 步后的期望像素位移。

**符号说明**：
- $\pi_{\text{pri}}(\cdot)$：将三维末端执行器位置投影到主摄像素坐标的投影函数
- $\mathbf{x}_{1,r,H}$：第 $r$ 个 rollout 第 $H$ 帧的末端执行器三维位置
- $\mathbf{x}_{r,0}$：当前帧末端执行器三维位置（初始帧）

---

### 公式 3：[[光流 (Optical Flow)|胶囊区域平均光流]]

$$
\bar{\mathbf{f}}_{1,r} = \frac{1}{|\mathcal{M}_{1,r}|} \sum_{p \in \mathcal{M}_{1,r}} F_1(p)
$$

**含义**：计算末端执行器轨迹胶囊掩码区域内的平均光流向量，作为视频预测中实际视觉运动的代理。

**符号说明**：
- $\mathcal{M}_{1,r}$：以末端执行器轨迹为轴线、半径 $\rho_{\text{cap}}=30$ 像素的胶囊掩码
- $F_1(p)$：主摄视频预测第一帧像素 $p$ 处的光流向量
- $|\mathcal{M}_{1,r}|$：掩码内有效像素数量

---

### 公式 4：[[Best-of-N|动作-未来余弦一致性]]

$$
c_{1,r} = \frac{\bar{\mathbf{f}}_{1,r}^\top \Delta\mathbf{u}_{1,r}}{\|\bar{\mathbf{f}}_{1,r}\|_2 \;\|\Delta\mathbf{u}_{1,r}\|_2 + \varepsilon}
$$

**含义**：度量预测的视觉运动（光流）与动作隐含的位移是否方向一致；低于阈值 $\tau_{\text{gate}} = -0.2$ 时触发额外采样。

**符号说明**：
- $\bar{\mathbf{f}}_{1,r}$：胶囊区域平均光流向量（见公式 3）
- $\Delta\mathbf{u}_{1,r}$：投影的末端执行器像素位移（见公式 2）
- $\varepsilon$：数值稳定小量
- $\tau_{\text{gate}} = -0.2$：门控触发阈值（全局固定，跨 benchmark）

---

### 公式 5：[[跨视角深度重投影|跨视角深度重投影误差]]

$$
e_{\text{depth}}(\tau_i) = \frac{1}{|\Omega_i|} \sum_{p \in \Omega_i} \left|\log\frac{d_i^{\text{proj}}(p)}{d_i^{\text{vggt}}(p)}\right|
$$

**含义**：通过对数比绝对误差量化两路深度估计的一致性，排除绝对深度尺度影响；该值越小说明候选 rollout 的想象未来在三维几何上越自洽。

**符号说明**：
- $d_i^{\text{vggt}}(p)$：[[VGGT]] 直接从主摄预测图像估计的像素 $p$ 处深度
- $d_i^{\text{proj}}(p)$：将腕摄视角的 3D 点云投影到主摄后得到的像素 $p$ 处深度
- $\Omega_i$：有效像素集（要求两路深度均为正值且 VGGT 置信度 $> \gamma_{\text{conf}} = 0.5$）
- $|\Omega_i|$：有效像素数量

---

## 关键图表

### Figure 1：GeoBoN 整体框架

![Figure 1](https://arxiv.org/html/2607.17454v1/x1.png)

**说明**：左侧展示动作-未来门控流程：Farneback [[光流 (Optical Flow)|光流]] 计算胶囊区域平均运动，与投影末端执行器位移做余弦对比；一致则直接执行，不一致则触发右侧 GeoBoN 几何评分流程。右侧展示用冻结 [[VGGT]] 对 $N$ 个候选 rollout 计算[[跨视角深度重投影|跨视角深度重投影误差]]，选取误差最小者执行。

---

### Figure 2：候选池大小与假低分选择率

![Figure 2](https://arxiv.org/html/2607.17454v1/x2.png)

**说明**：假低分选择率（False Low-Score Selection Rate）随候选数 $N$ 升高而急剧上升（N=2 时 7%，N=8 时 12%，N=16 时 32%），解释了固定预算 GeoBoN 在 $N=16$ 时性能不再提升甚至略降的"多重比较效应"（multiple-comparisons effect）。

---

### Figure 3：假低分选择示例

![Figure 3](https://arxiv.org/html/2607.17454v1/x3.png)

**说明**：展示了典型假低分选择案例：两个候选 rollout 视觉外观几乎相同（[[LPIPS]] < 0.02），但 [[跨视角深度重投影|深度重投影误差]] 差异悬殊，导致几何评分器错误地将偶发的离群值候选选为最优。

---

### Table 1：固定预算 GeoBoN 成功率（%）

| Benchmark / WAM | 基线 | N=2 | N=4 | N=8 | N=16 | Δ (N=8 vs 基线, 95% CI) |
|---|---|---|---|---|---|---|
| RoboCasa Door/Drawer / Cosmos Policy | 89.8 | 88.7 | 90.2 | 90.8 | 91.2 | +1.0 [−2.6, +4.6] |
| RoboCasa Pick-and-Place / Cosmos Policy | 47.2 | 49.2 | 47.9 | 49.0 | 48.1 | +1.8 [−2.1, +7.7] |
| RoboCasa Appliance Control / Cosmos Policy | 79.9 | 79.3 | 81.0 | 82.6 | 82.3 | +2.7 [−2.0, +7.4] |
| RoboCasa Coffee Making / Cosmos Policy | 38.3 | 38.7 | 38.5 | 42.0 | 42.9 | +3.7 [−3.0, +13.0] |
| **RoboCasa 平均 / Cosmos Policy** | 66.3 | 66.5 | 66.9 | **68.4** | 68.2 | **+2.1 [+0.6, +7.3]** |
| RoboCasa Door/Drawer / X-WAM | 96.7 | 89.7 | 92.7 | 92.1 | 88.7 | −4.6 [−6.9, −2.9] |
| RoboCasa Pick-and-Place / X-WAM | 68.8 | 70.5 | 71.0 | 70.4 | 73.2 | +1.6 [−6.6, +9.8] |
| RoboCasa Appliance Control / X-WAM | 84.3 | 84.4 | 81.0 | 87.5 | 83.4 | +3.2 [−2.3, +8.8] |
| RoboCasa Coffee Making / X-WAM | 73.3 | 86.3 | 87.7 | 83.7 | 92.0 | +10.4 [−2.8, +24.5] |
| **RoboCasa 平均 / X-WAM** | 80.8 | 81.3 | 81.4 | **82.5** | 82.4 | **+1.7 [+0.3, +2.6]** |
| **LIBERO Long / Cosmos Policy** | 97.5 | 98.2 | 98.6 | **99.3** | 98.9 | **+1.8 [+0.4, +2.2]** |
| **LIBERO Long / LingBotVA** | 97.2 | 97.8 | 98.0 | **98.3** | 99.1 | **+1.1 [+0.2, +1.5]** |
| **RoboTwin 2.0 / Motus** | 87.8 | 88.3 | 88.5 | **89.9** | 89.5 | **+2.1 [+0.2, +3.4]** |

**表格说明**：GeoBoN 在 5 个 benchmark-WAM 配置中以 N=8 达到最佳；X-WAM 在 Door/Drawer 子任务上出现下降（−4.6%），可能与该子任务高基线导致已无多少提升空间有关。

---

### Table 2：门控 GeoBoN（$N_{\text{max}}=8$）效率对比

| Benchmark / WAM | 模式 | 成功率 (%) | 满分恢复率 (%) | BoN 触发率 (%) | 平均延迟 (s) |
|---|---|---|---|---|---|
| RoboCasa / Cosmos Policy | 基线 | 66.3 | 0.0 | 0 | 0.90 |
| RoboCasa / Cosmos Policy | **Gated GeoBoN** | **67.9** | **76.2** | **24.7** | **1.29** |
| RoboCasa / Cosmos Policy | GeoBoN (固定) | 68.4 | 100.0 | 100 | 3.65 |
| RoboCasa / X-WAM | 基线 | 80.8 | 0.0 | 0 | 2.68 |
| RoboCasa / X-WAM | **Gated GeoBoN** | **82.1** | **76.5** | **25.2** | **3.11** |
| RoboCasa / X-WAM | GeoBoN (固定) | 82.5 | 100.0 | 100 | 9.67 |
| LIBERO Long / Cosmos Policy | 基线 | 97.5 | 0.0 | 0 | 0.90 |
| LIBERO Long / Cosmos Policy | **Gated GeoBoN** | **98.8** | **72.2** | **14.2** | **0.96** |
| LIBERO Long / Cosmos Policy | GeoBoN (固定) | 99.3 | 100.0 | 100 | 3.66 |
| LIBERO Long / LingBotVA | 基线 | 97.2 | 0.0 | 0 | 2.27 |
| LIBERO Long / LingBotVA | **Gated GeoBoN** | **97.9** | **63.6** | **34.5** | **2.83** |
| LIBERO Long / LingBotVA | GeoBoN (固定) | 98.3 | 100.0 | 100 | 3.86 |
| RoboTwin 2.0 / Motus | 基线 | 87.8 | 0.0 | 0 | 2.29 |
| RoboTwin 2.0 / Motus | **Gated GeoBoN** | **89.6** | **85.7** | **32.2** | **2.83** |
| RoboTwin 2.0 / Motus | GeoBoN (固定) | 89.9 | 100.0 | 100 | 3.86 |

**表格说明**：门控 GeoBoN 平均恢复 74.8% 的固定预算增益，仅在 26.2% 的决策点触发额外采样，延迟远低于全量 GeoBoN。

---

### Table 3：选择器消融（Error Recovery & 在线成功率）

| Benchmark-WAM | 误差恢复率 (%) | | | | 在线成功率 (%) | | | |
|---|---|---|---|---|---|---|---|---|
| | 基线 | VGGT Conf. | 未来共识 | **GeoBoN** | 基线 | VGGT Conf. | 未来共识 | **GeoBoN** |
| RoboCasa / Cosmos Policy | 0 | 7.3 | 6.9 | **7.6** | 66.3 | 65.9 | 67.1 | **68.4** |
| RoboCasa / X-WAM | 0 | 4.0 | 8.6 | **6.9** | 80.8 | 77.5 | 77.4 | **82.5** |
| LIBERO Long / LingBotVA | 0 | 11.4 | 21.3 | **22.0** | 97.2 | 96.3 | 95.3 | **98.3** |
| RoboTwin 2.0 / Motus | 0 | 1.5 | 7.6 | **8.3** | 87.8 | 88.1 | 90.5 | **89.9** |

**表格说明**：GeoBoN 是所有选择器中最稳定的，在全部 4 个配置中提升在线成功率；VGGT 置信度排序不稳定，在 RoboCasa/X-WAM 上导致 −3.3% 的显著下降。

---

### Table 4：等预算门控诊断（$N=8$）

| Benchmark-WAM | 触发率 (%) | GeoBoN 帮助率 (%) | | Δ |
|---|---|---|---|---|
| | | 随机触发 | AF 门控 | |
| RoboCasa / X-WAM | 25.2 | 43.2 | **67.2** | **+24.0** |
| RoboCasa / Cosmos Policy | 24.7 | 43.2 | **63.5** | **+20.3** |
| LIBERO / LingBotVA | 34.5 | 47.8 | **69.4** | **+21.6** |
| RoboTwin / Motus | 32.2 | 71.7 | 75.8 | +4.1 |

**表格说明**：动作-未来门控在相同触发率下显著优于随机触发（RoboCasa 提升 +20～24%），证明门控信号本身具有信息量，而非仅起预算缩减作用；RoboTwin 域下增益较小（+4.1%）。

---

### Table 5：与 Cosmos Policy 值头对比

| Benchmark | 选择器 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|
| RoboCasa | 值头（Value Head）| 65.2 | 65.5 | 66.1 | 65.9 |
| RoboCasa | **GeoBoN** | **66.5** | **66.9** | **68.4** | **68.2** |
| LIBERO Long | 值头（Value Head）| 97.1 | 97.9 | 98.1 | 97.6 |
| LIBERO Long | **GeoBoN** | **98.2** | **98.6** | **99.3** | **98.9** |

**表格说明**：无训练的 GeoBoN 在 RoboCasa 上超越 Cosmos Policy 自身值头 +1.3～+2.3 百分点，在 LIBERO Long 上超越 +0.7～+1.3 百分点，且不受限于特定 WAM 骨干网络。

---

## 实验

### 基准与评估协议

| Benchmark | 任务类型 | 评估规模 | 领域随机化 |
|---|---|---|---|
| [[RoboCasa]] | 厨房操作（4 组任务） | 每任务 10 rollouts × 4 seeds | 否 |
| [[LIBERO]] Long | 长时序任务序列 | 每任务 50 rollouts × 4 seeds | 否 |
| [[RoboTwin 2.0]] | 双臂操作 | 每任务 10 rollouts × 4 seeds | 是 |

### 实现细节

- **WAM 骨干**：[[Cosmos-Policy]]、X-WAM、[[LingBot-VA|LingBotVA]]、[[Motus]]（均直接使用发布版本，无额外训练）
- **几何模型**：冻结的 VGGT-Ω（[[VGGT]] 变体，支持任意多视角输入）
- **光流算法**：Farneback 算法
- **固定超参数**（所有实验统一，无数据集特定调参）：

| 超参 | 值 | 说明 |
|---|---|---|
| 空闲阈值 $\delta_{\text{idle}}$ | 1 cm | 过滤末端执行器几乎不动的帧 |
| 门控阈值 $\tau_{\text{gate}}$ | −0.2 | 余弦一致性触发阈值 |
| 胶囊半径 $\rho_{\text{cap}}$ | 30 px | 末端执行器掩码区域半径 |
| VGGT 置信度阈值 $\gamma_{\text{conf}}$ | 0.5 | 有效像素筛选 |

- **硬件**：H200 和 RTX Pro 6000 GPU

### 定量结果摘要

- 5 个 benchmark-WAM 配置在 N=8 时均取得正向收益（+1.1%～+2.1%）
- 门控 GeoBoN 平均触发率 26.2%，平均恢复固定预算增益的 74.8%
- 无训练 GeoBoN 在 RoboCasa 上超越有训练值头 +1.3～+2.3pp

---

## 批判性思考

### 优点

1. **训练无关**：GeoBoN 完全不依赖任务标签或真实观测，跨 WAM 骨干通用
2. **信号物理直觉清晰**：跨视角深度一致性是硬约束，不是学习来的启发式
3. **门控设计实用**：Gated GeoBoN 以极小延迟代价（基线 2-4× 延迟 vs 固定预算 10× 延迟）恢复大部分增益
4. **全面的失败分析**：显式识别"假低分选择"现象，为未来改进指明方向

### 局限性

1. **性能提升幅度有限**：绝对成功率提升在 1-2% 量级，对于高难度任务提升更小（置信区间宽）
2. **假低分选择问题未解决**：N=16 时假选择率达 32%，Paper 识别了问题但未提供解决方案
3. **依赖 WAM 多视角预测**：对于不预测腕摄视角的 WAM 架构，方法无法直接应用
4. **门控在 RoboTwin 下信号较弱**：AF 门控在 RoboTwin 域的信息量有限（Δ 仅 +4.1%），可能与任务特性有关
5. **VGGT 推理开销**：N=8 固定预算版本延迟达到基线的 4× (Cosmos Policy) 至 3.6× (X-WAM)

### 潜在改进方向

1. **解决假低分选择**：引入对深度一致性离群值的鲁棒统计量（如 Huber 损失替代 $|{\log}|$）
2. **更好的门控设计**：探索除光流余弦之外的一致性信号，尤其对 RoboTwin 类域
3. **结合值头与几何评分**：两者互补（值头偏任务相关，几何评分偏物理约束），可能组合更优

### 可复现性评估

- [ ] 代码开源（论文中未提及开源计划）
- [ ] 预训练模型（依赖外部 WAM 模型，自身无需训练）
- [x] 训练细节完整（无训练，超参完整给出）
- [x] 数据集可获取（RoboCasa / LIBERO / RoboTwin 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[WAM|World Action Model (WAM)]]: GeoBoN 工作在任意 WAM 骨干之上，依赖 WAM 的多视角未来预测
- [[VGGT]]: 核心几何评分工具，用于计算跨视角深度一致性

### 对比

- [[Cosmos-Policy]]: 主要实验骨干之一，也提供了内置值头对比基线
- [[Motus]]: 第三个实验骨干（RoboTwin 2.0）
- [[LingBot-VA]]: 第四个实验骨干（LIBERO Long）

### 方法相关

- [[光流 (Optical Flow)|光流]]: 动作-未来门控的核心信号
- [[跨视角深度重投影]]: GeoBoN 几何评分器的核心度量
- [[Best-of-N]]: GeoBoN 本质是几何引导的 Best-of-N 采样

### 硬件/数据相关

- [[RoboCasa]]: 主实验 benchmark（厨房操作，4 类任务组）
- [[LIBERO]]: 长序列任务 benchmark
- [[RoboTwin 2.0]]: 双臂操作 benchmark，含领域随机化

---

## 速查卡片

> [!summary] GeoBoN：几何引导的 WAM 测试时最优候选选择
> - **核心**: 用冻结 VGGT 的跨视角深度重投影误差排序 WAM 候选 rollout，无需训练
> - **方法**: 动作-未来门控（光流余弦一致性）+ 几何 Best-of-N 选择
> - **结果**: N=8 时 5 个配置均正向提升（+1.1～+2.1%），门控版以 26% 触发率恢复 75% 增益
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-22*

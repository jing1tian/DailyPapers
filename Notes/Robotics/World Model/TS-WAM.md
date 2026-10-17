---
title: "TS-WAM: Temporal Latent Actions and World-Action Coupling for Vision-Language Robot Control"
method_name: "TS-WAM"
authors: [Menghao Yin, Jiaju Yin, Xiyue Dong, Lixin Xu, Zhenhui Zhang, Ke Zhang, Yating Feng, Renjing Xu]
year: 2026
venue: arXiv
tags: [world-action-model, jepa, vla, teacher-student, latent-action, self-supervised-learning, flow-matching, ood-generalization, ego-video]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.21922v1
created: 2026-10-17
---

# 论文笔记：TS-WAM — Temporal Latent Actions and World-Action Coupling for Vision-Language Robot Control

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Hong Kong University of Science and Technology (Guangzhou) |
| 日期 | October 2026 |
| 项目主页 | N/A |
| 对比基线 | [[π0]] · [[π0.5]] · [[VLA-JEPA]] · [[UniVLA]] · [[LAPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.21922) / [HTML](https://arxiv.org/html/2610.21922v1) |

---

## 一句话总结

> TS-WAM 是一个轻量端到端的 [[World Action Model|世界-动作模型]]，通过 [[EMA]] 教师分支（[[vlm-jepa]]）学习时序潜在动作，并用 [[delta-lwm]] 将潜在动作与视觉特征变化耦合，驱动 [[Flow Matching|流匹配]] 动作生成，推理时去掉教师分支，在 LIBERO-Plus OOD 基准以仅 **1.2B** 参数达到 **75.7%**，超越所有同类方法。

---

## 核心贡献

1. **[[vlm-jepa]]（时序潜在动作学习）**: 利用 [[EMA]] 目标分支实现自监督时序对齐，无需外部教师模型或独立预训练阶段，直接从机器人演示与无标注人类视频中学习潜在动作表示。
2. **[[delta-lwm]]（潜在世界-动作耦合）**: 设计特征增量预测器 $D_\psi$ 预测 [[DINOv3]] 特征变化 $\Delta F_t$，引入 Ranking Loss 确保动作条件对未来视觉预测的真实依赖，形成"潜在动作→视觉预测→动作生成"的完整闭环。
3. **端到端联合训练**: 机器人 clip 与 ego（无动作）clip 在同一框架内联合更新，推理时仅保留在线分支，无额外计算开销，支持 30 Hz 在消费级 RTX 4090 上实时运行。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 策略仅从含动作标注的机器人演示中学习，缺乏对未来视觉状态的显式预测，导致：
- 无法利用大量**无动作标注**的人类 ego 视频
- 对视觉分布偏移（相机角度、光照、背景）泛化能力弱

### 现有方法的局限

- **[[LAPA]]**：将潜在动作模型作为独立预训练阶段，产生多余的检查点，且不能保证潜在表示对动作解码器有帮助，没有端到端耦合。
- **[[VLA-JEPA]]**：基于 JEPA 的 VLA 方法，已探索 JEPA 与 VLA 的结合，但缺乏显式的视觉特征变化预测耦合机制。
- **其他 WAM 方法**（如 [[DINO-WM]]）：通常需要分离预训练，推理时仍需世界模型分支，计算开销高。
- **[[IDM|逆动力学模型（IDM）]]** 方法：从输入当前帧+未来帧推断动作，导致策略对未来帧产生依赖，无法在推理时独立运行。

### 本文的动机

通过非对称设计——在线 VLM 只看当前观测，[[EMA]] VLM（教师）同时看当前和未来帧——避免推理时对未来帧的依赖，同时利用教师分支的丰富时序信号。耦合 [[DINOv3]] 特征差分预测进一步将潜在动作与可验证的视觉变化绑定，形成可学习的内部一致性约束。

---

## 方法详解

### 模型架构

TS-WAM 采用**非对称教师-学生 [[World Action Model|世界-动作模型]]** 架构：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$（训练时额外需要未来观测 $o_{t+1}$）
- **在线 VLM（学生）**: 只看当前帧，通过可学习查询令牌 $Q_Y$ 预测潜在动作 $\hat{z}_t$
- **EMA VLM（教师）**: 看当前+未来帧，产生停止梯度目标 $z_t^Y$；推理时**完全移除**
- **视觉编码器**: 冻结的 [[DINOv3]] 提取空间特征网格 $F_t$
- **核心模块**: [[vlm-jepa]] 用于时序潜在动作 + [[delta-lwm]] 用于视觉特征耦合
- **动作头**: [[Flow Matching|流匹配]] 动作专家，以当前/预测未来特征和 VLM 上下文为条件
- **总参数**: **1.2B**（推理时，教师分支去除）
- **推理频率**: 4× A100 训练，单卡 RTX 4090 推理，30 Hz

### 核心模块

#### 模块 1：[[vlm-jepa]]（VLM 级时序 JEPA）

**设计动机**: 传统 [[JEPA]] 在图像 patch 级操作；vlm-jepa 将其提升到 **VLM 语义级别**，利用语言指令为潜在动作提供任务感知的归纳偏置，同时通过 [[EMA]] 自监督消除对外部教师的依赖。

**具体实现**:
- 在线分支：将当前视觉令牌 $X_t$ 和可学习动作查询 $Q_Y$（维度 $Q_a \times d_z$）输入 Transformer 预测器 $P_\phi$，输出潜在动作 $\hat{z}_t$
- 教师分支：参数由 [[EMA]] 更新，处理当前+未来帧，提取 [[Stop-Gradient|停止梯度]] 目标 $z_t^Y$，经 LayerNorm 归一化为 $\bar{z}_t^Y$
- 时序损失：L1 距离 $\mathcal{L}_{\text{temp}} = \|\hat{z}_t - \bar{z}_t^Y\|_1$

#### 模块 2：[[delta-lwm]]（增量潜在世界模型）

**设计动机**: 仅有潜在动作不足以保证其携带真实的视觉因果信息；通过让解码器预测 [[DINOv3]] 特征增量，并引入 [[Ranking Loss|批次不匹配排名损失]] 强制解码器依赖匹配的潜在动作，形成视觉-动作一致性约束。

**具体实现**:
- 冻结 [[DINOv3]] 编码器提取空间特征网格 $F_t, F_{t+1}$
- 解码器 $D_\psi$ 以 $F_t$ 和 $\hat{z}_t$ 为输入，预测特征增量 $\widehat{\Delta F}_t$
- 世界损失同时使用教师和学生路径（双路监督）
- Ranking Loss：以 batch 内置换的潜在动作 $\tilde{z}_t$ 作为负样本，边距 $\mu$ 约束，确保匹配 $\hat{z}_t$ 比乱序 $\tilde{z}_t$ 产生更好的预测

#### 模块 3：[[Flow Matching|流匹配]] 动作头

**设计动机**: [[Flow Matching]] 相比扩散模型采样效率更高，且可以无缝接受多模态条件输入。

**具体实现**:
- 条件向量：$c_t = (F_t,\ F_{t+1}^{\text{cond}},\ h_t^{\text{vlm}},\ s_t)$（当前特征、预测/真实未来特征混合、VLM 隐状态、本体感知）
- 未来特征混合：以比率 $r_t$ 在预测特征 $\hat{F}_{t+1}$ 与真实特征 $F_{t+1}$ 之间插值，加速收敛
- 推理时完全使用预测的 $\hat{F}_{t+1}$

---

## 关键公式

### 公式 1：[[vlm-jepa|在线潜在动作预测]]

$$
\hat{z}_t = P_\phi(X_t, Q_Y), \quad \hat{z}_t \in \mathbb{R}^{Q_a \times d_z}
$$

**含义**: 在线分支（学生 VLM）从当前视觉令牌 $X_t$ 和可学习查询 $Q_Y$ 中预测潜在动作，形状为 $(Q_a, d_z)$。

**符号说明**:
- $X_t$：当前帧视觉令牌
- $Q_Y$：可学习动作查询令牌，维度 $Q_a \times d_z$
- $P_\phi$：Transformer 预测器（在线分支参数 $\phi$）
- $Q_a$：潜在动作令牌数；$d_z$：潜在动作维度

### 公式 2：[[vlm-jepa|预测器显式瓶颈]]

$$
\hat{z}_t = W_o \operatorname{LN}\!\left(\operatorname{Tr}_\phi\!\left[W_x X_t + E_x \;;\; W_q Q_Y + E_y\right]_{\text{target}}\right)
$$

**含义**: 在线预测器通过线性投影 + 位置编码将视觉令牌和查询令牌融合，经 Transformer 处理后输出经 LayerNorm 规范化的潜在动作。

**符号说明**:
- $W_x, W_q$：线性投影矩阵；$E_x, E_y$：位置编码
- $\operatorname{Tr}_\phi$：在线 Transformer 预测器；$W_o$：输出投影
- $\operatorname{LN}$：LayerNorm；`target`：仅保留对应查询令牌的输出

### 公式 3：[[EMA|EMA 教师更新]]

$$
\xi \leftarrow m\xi + (1-m)\theta_Y
$$

**含义**: 教师分支参数 $\xi$ 以动量 $m$ 缓慢跟踪学生参数 $\theta_Y$，提供稳定的停止梯度目标，避免崩溃。

**符号说明**:
- $\xi$：教师分支参数（[[Stop-Gradient|不接受反向传播]]）
- $m$：EMA 动量（接近 1，如 0.996）
- $\theta_Y$：学生 VLM 参数

### 公式 4：[[vlm-jepa|时序对齐损失]]

$$
\bar{z}_t^Y = \operatorname{LN}_z\!\left(\operatorname{sg}(z_t^Y)\right), \quad \mathcal{L}_{\text{temp}} = \left\|\hat{z}_t - \bar{z}_t^Y\right\|_1
$$

**含义**: 学生预测的潜在动作与 [[Stop-Gradient|停止梯度]] 的教师潜在动作之间的 L1 距离，驱动学生学习捕获帧间时序变化的潜在表示。

**符号说明**:
- $\operatorname{sg}(\cdot)$：[[Stop-Gradient]] 算子，阻止梯度流向教师
- $\operatorname{LN}_z$：对教师输出做 LayerNorm 归一化
- $z_t^Y$：教师分支潜在动作；$\hat{z}_t$：学生预测潜在动作

### 公式 5：[[delta-lwm|特征增量目标与预测]]

$$
\Delta F_t = F_{t+1} - F_t, \quad \widehat{\Delta F}_t = D_\psi(F_t, \hat{z}_t)
$$

**含义**: 将未来-当前 [[DINOv3]] 特征差作为预测目标，解码器 $D_\psi$ 以当前特征网格和潜在动作为条件预测该增量。

**符号说明**:
- $F_t, F_{t+1}$：[[DINOv3]] 冻结编码器提取的当前/未来空间特征网格
- $\Delta F_t$：真实特征增量（监督目标）
- $D_\psi$：增量解码器（从零训练）；$\psi$：解码器参数

### 公式 6：[[delta-lwm|世界模型损失（双路）]]

$$
\mathcal{L}_{\text{world}} = \left\|D_\psi(F_t, z_t^Y) - \Delta F_t\right\|_2^2 + \left\|D_\psi(F_t, \hat{z}_t) - \Delta F_t\right\|_2^2
$$

**含义**: 同时用教师潜在动作 $z_t^Y$ 和学生预测 $\hat{z}_t$ 训练解码器，确保两种潜在动作都与视觉变化信息对齐。

**符号说明**:
- 第一项：教师路径，使用教师 VLM 输出的潜在动作
- 第二项：学生路径，使用在线预测的潜在动作
- $\|\cdot\|_2^2$：L2 范数平方（均方误差）

### 公式 7：[[delta-lwm|预测未来特征]]

$$
\hat{F}_{t+1} = F_t + \widehat{\Delta F}_t
$$

**含义**: 以当前特征为基础，叠加预测的特征增量，得到未来时刻的预测视觉特征，用于条件化动作生成。

**符号说明**:
- $F_t$：当前帧 [[DINOv3]] 特征（冻结编码器输出）
- $\widehat{\Delta F}_t$：解码器预测的特征增量
- $\hat{F}_{t+1}$：预测的未来特征（推理时用于替代真实 $F_{t+1}$）

### 公式 8：[[Ranking Loss|批次不匹配排名损失]]

$$
\mathcal{L}_{\text{dep}} = \left[\mu + \left\|D_\psi(F_t, \hat{z}_t) - \Delta F_t\right\|_2^2 - \left\|D_\psi(F_t, \tilde{z}_t) - \Delta F_t\right\|_2^2\right]_+
$$

**含义**: 将 batch 内置换的潜在动作 $\tilde{z}_t$ 作为负样本，以边距 $\mu$ 约束，迫使解码器在使用正确匹配的潜在动作时产生更低的预测误差，从而强制解码器真实依赖潜在动作的内容。

**符号说明**:
- $\tilde{z}_t$：batch 内随机置换后的潜在动作（错误配对的负样本）
- $\mu$：边距超参数
- $[\cdot]_+$：hinge 函数（ReLU）

### 公式 9：[[Flow Matching|流匹配插值与动作损失]]

$$
a_\tau = (1-\tau)\epsilon + \tau a, \quad u(a, \epsilon, \tau) = a - \epsilon
$$

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{\tau, \epsilon}\!\left\|v_\theta(a_\tau, c_t) - u(a, \epsilon, \tau)\right\|_2^2
$$

**含义**: 在噪声 $\epsilon$ 和真实动作 $a$ 之间以时间步 $\tau$ 线性插值，训练速度场 $v_\theta$ 拟合直线流方向，推理时通过 ODE 积分生成动作。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$：流时间步，均匀采样
- $\epsilon \sim \mathcal{N}(0,I)$：标准高斯噪声
- $a_\tau$：插值中间动作；$u$：流方向目标（直线）
- $v_\theta$：速度场网络；$c_t$：多模态条件向量

### 公式 10：[[delta-lwm|混合未来特征条件]]

$$
F_{t+1}^{\text{cond}} = r_t \hat{F}_{t+1} + (1-r_t) F_{t+1}, \quad c_t = (F_t,\ F_{t+1}^{\text{cond}},\ h_t^{\text{vlm}},\ s_t)
$$

**含义**: 训练时以混合比率 $r_t$ 在预测特征与真实未来特征之间插值，帮助动作头适应测试时完全使用预测特征的情况，缓解分布偏移。

**符号说明**:
- $r_t$：混合比率（训练过程中逐渐增大）
- $h_t^{\text{vlm}}$：在线 VLM 隐状态（语言+视觉语义）
- $s_t$：本体感知状态（关节角度等）

### 公式 11：机器人 clip 总损失

$$
\mathcal{L}_{\text{robot}} = \mathcal{L}_{\text{flow}} + \lambda_T \mathcal{L}_{\text{temp}} + \lambda_W \mathcal{L}_{\text{world}}
$$

**含义**: 有动作标注的机器人演示 clip 的完整训练目标，三项损失联合优化流匹配动作生成、时序潜在对齐和视觉世界耦合。

**符号说明**:
- $\lambda_T$：时序损失权重；$\lambda_W$：世界损失权重
- $\mathcal{L}_{\text{flow}}$：[[Flow Matching|流匹配]] 动作损失
- $\mathcal{L}_{\text{temp}}$：[[vlm-jepa]] 时序对齐损失
- $\mathcal{L}_{\text{world}}$：[[delta-lwm]] 视觉预测损失

### 公式 12：Ego clip 总损失

$$
\mathcal{L}_{\text{ego}} = \lambda_T \mathcal{L}_{\text{temp}} + \lambda_W \mathcal{L}_{\text{world}}
$$

**含义**: 无动作标注的人类 ego 视频 clip 的训练目标，仅优化时序表示学习和视觉预测，使模型从人类视频中获益。

**符号说明**:
- 去掉 $\mathcal{L}_{\text{flow}}$（无动作标注，不计算动作损失）
- 其余符号同公式 11

---

## 关键图表

### Figure 1：IDM vs JEPA 世界-动作模型对比

![Figure 1](https://arxiv.org/html/2610.21922v1/figures/teaser-1-cropped.png)

**说明**: 对比传统基于 [[IDM|逆动力学模型（IDM）]] 的世界-动作模型与 TS-WAM 的 [[JEPA]] 范式。IDM 方法在推理时需要依赖未来帧，而 TS-WAM 通过 [[EMA]] 教师分支在训练时引入未来信息，推理时仅需当前观测。

### Figure 2：TS-WAM 整体架构

![Figure 2](https://arxiv.org/html/2610.21922v1/figures/overview-1.png)

**说明**: TS-WAM 完整架构概览。左侧在线分支（学生 VLM）处理当前观测，右侧 EMA 教师分支训练时提供时序目标（推理时移除），[[DINOv3]] 特征增量预测器连接潜在动作与视觉预测，[[Flow Matching|流匹配]] 动作头生成最终动作。

### Figure 3：vlm-jepa 模块架构

![Figure 3](https://arxiv.org/html/2610.21922v1/figures/vlm-jepa-1.jpg)

**说明**: [[vlm-jepa]] 的详细结构。在线 VLM 通过 Transformer 预测器从视觉令牌和可学习查询令牌预测潜在动作；[[EMA]] VLM 处理未来帧产生 [[Stop-Gradient|停止梯度]] 目标；L1 时序损失驱动学生拟合教师表示。

### Figure 4：潜在动作表示多样性分析

![Figure 4](https://arxiv.org/html/2610.21922v1/figures/la_representation_diversity.png)

**说明**: 对学生和教师分支输出的潜在动作进行 PCA 可视化，展示不同任务下潜在动作的分布分离情况，验证 [[vlm-jepa]] 学到了有意义的任务感知时序表示。

### Figure 5：潜在动作条件化对视觉预测的影响

![Figure 5](https://arxiv.org/html/2610.21922v1/figures/la_conditioning_effect.png)

**说明**: 在正常潜在动作、均值潜在动作和零潜在动作三种条件下可视化 [[delta-lwm]] 的特征变化预测结果，证明解码器真实依赖潜在动作内容（[[Ranking Loss]] 的有效性）。

### Figure 6：LIBERO 消融实验

![Figure 6](https://arxiv.org/html/2610.21922v1/figures/libero_ablation.png)

**说明**: LIBERO 四个子集上的消融实验结果。移除 [[vlm-jepa]] 导致平均成功率从 97.00% 下降到 92.35%（最大损失），移除人类视频下降到 93.35%，移除预测 $\Delta F$ 条件下降约 3.0 点。

### Figure 7：LIBERO-Plus 消融实验

![Figure 7](https://arxiv.org/html/2610.21922v1/figures/libero_plus_ablation.png)

**说明**: LIBERO-Plus OOD 场景下的消融实验。整体成功率从 75.7% 降至：去掉 [[vlm-jepa]] 61.0%，去掉人类视频 64.8%，去掉预测 $\Delta F$ 条件 70.8%，一致说明各组件对 OOD 泛化的贡献。

### Figure 8：真实机器人部署（RevoTron 双臂）

![Figure 8](https://arxiv.org/html/2610.21922v1/figures/realworld-1-cropped.png)

**说明**: TS-WAM 在 RevoTron 双臂机器人上执行 Drawer-Long 任务，分别在 IND（分布内）和三种 OOD 条件（光照变化 OOD-LC、物体布局 OOD-OL、背景纹理 OOD-BT）下的执行过程，直观展示其 OOD 鲁棒性。

### Table 1：LIBERO 域内成功率（%）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| [[π0]] | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| [[π0-FAST]] | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| [[π0.5]] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| [[LAPA]] | 55.4 | 58.8 | 74.6 | 73.8 | 65.7 |
| Mantis | 98.8 | 99.2 | 94.4 | 94.2 | 96.7 |
| [[UniVLA]] | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| [[VLA-JEPA]] | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |
| **TS-WAM** | **97.2** | **99.8** | **97.6** | **93.4** | **97.0** |

**关键发现**: TS-WAM 以 1.2B 参数在 LIBERO 平均成功率上超越所有基线（97.0%），Object 子集 99.8% 为最高分。

### Table 2：SimplerEnv 成功率（%）

| Method | Spoon (VM) | Carrot (VM) | Pick (VA) | Move (VA) | Avg. |
|--------|-----------|------------|----------|----------|------|
| RT-1-X | 0.0 | 4.2 | 49.0 | 32.3 | 21.38 |
| Octo-Base | 15.8 | 12.5 | 0.6 | 3.1 | 8.00 |
| [[OpenVLA]] | 4.2 | 0.0 | 60.8 | 67.7 | 33.18 |
| Magma | 37.5 | 31.0 | 68.8 | 65.7 | 50.75 |
| [[π0]] | 29.1 | 0.0 | 75.2 | 63.7 | 42.00 |
| π0-FAST | 29.1 | 21.9 | 77.6 | 68.2 | 49.20 |
| RoboVLMs | 45.8 | 20.8 | 77.3 | 61.7 | 51.40 |
| TraceVLA | 12.5 | 16.6 | 64.4 | 60.6 | 38.53 |
| **TS-WAM** | **62.5** | **37.5** | **80.33** | **75.0** | **63.83** |

**关键发现**: TS-WAM 在 [[SimplerEnv]] 平均成功率 63.83%，超越次优方法 RoboVLMs（51.40%）达 **+12.43 个百分点**，Spoon（Visual Matching）任务提升最为显著（+16.7pp）。

### Table 3：LIBERO-Plus 零样本泛化成功率（%）

| Method | Params | Camera | Robot | Language | Light | Background | Noise | Layout | Overall |
|--------|--------|--------|-------|---------|-------|-----------|-------|--------|---------|
| WorldVLA | 7B | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| Mantis | 5.8B | 15.7 | 41.8 | 45.9 | 45.1 | 28.9 | 39.2 | 62.5 | 39.8 |
| [[UniVLA]] | 7B | 4.3 | 50.3 | 71.8 | 59.1 | 80.0 | 25.3 | 34.3 | 41.5 |
| [[π0]] | 3.3B | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| [[VLA-JEPA]] | 2B | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |
| Spatial Forcing | 7B | 20.1 | 13.4 | 40.9 | 29.1 | 33.4 | 25.7 | 39.3 | 28.8 |
| Fast-WAM | 6B | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.4 |
| **TS-WAM** | **1.2B** | **50.5** | **60.1** | **77.5** | **94.4** | **91.5** | **76.1** | **79.6** | **75.7** |

**关键发现**: TS-WAM 以最小参数量（**1.2B**，相比 VLA-JEPA 2B、π0 3.3B）在 LIBERO-Plus Overall 达到 75.7%，超越次优 [[VLA-JEPA]] **+12.8pp**，在光照（94.4%）和背景（91.5%）扰动下尤为突出。

### Table 4：真实机器人成功率（%，每任务每条件 20 次试验）

| Method | Condition | Stack Bowls | Drawer-Long | Item Sorting | Drawer-Long² | Overall |
|--------|-----------|------------|------------|-------------|-------------|---------|
| [[π0.5]] | IND | 75 | 40 | 60 | 45 | 55.00 |
| [[π0.5]] | OOD-LC | 60 | 20 | 50 | 35 | 41.25 |
| [[π0.5]] | OOD-OL | 45 | 20 | 40 | 25 | 32.50 |
| [[π0.5]] | OOD-BT | 40 | 25 | 30 | 30 | 31.25 |
| **TS-WAM** | IND | **85** | **65** | **80** | **60** | **72.50** |
| **TS-WAM** | OOD-LC | **80** | **55** | **75** | **60** | **67.50** |
| **TS-WAM** | OOD-OL | **80** | **45** | **70** | **50** | **61.25** |
| **TS-WAM** | OOD-BT | **70** | **45** | **70** | **55** | **60.00** |

注：OOD-LC = 光照变化，OOD-OL = 物体布局，OOD-BT = 背景纹理；Franka Research 3（Stack Bowls、Drawer-Long）和 RevoTron（Item Sorting、Drawer-Long）

**关键发现**: TS-WAM 在真实机器人 IND 场景比 [[π0.5]] 高 **+17.5pp**（72.50% vs 55.00%），更重要的是 IND→OOD 平均下降仅 **9.58pp**，而 π0.5 下降 **20.00pp**，证明 TS-WAM 的 OOD 鲁棒性优势在真实世界中同样成立。

### Table 5：消融实验汇总

| 配置 | LIBERO Avg. | LIBERO-Plus Overall |
|------|------------|-------------------|
| Full TS-WAM | 97.00% | 75.7% |
| w/o [[vlm-jepa]] | 92.35% | 61.0% |
| w/o human video | 93.35% | 64.8% |
| w/o Pred $\Delta F$ cond. | ~94.0% | 70.8% |

**关键发现**: [[vlm-jepa]] 模块贡献最大（LIBERO -4.65pp，LIBERO-Plus -14.7pp），人类视频贡献次之，预测 $\Delta F$ 条件主要改善 OOD 场景。

---

## 实验

### 数据集

| 数据集 / 环境 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| [[LIBERO]] | 500 任务（4 子集） | 桌面操作标准 benchmark | 域内训练 + 测试 |
| [[LIBERO-Plus]] | 同 LIBERO + 7 种扰动 | 相机/机械臂/语言/光照/背景/噪声/布局 | OOD 零样本泛化测试 |
| [[SimplerEnv]] | WindowX + Google Robot | 跨机器人迁移 | 跨平台泛化测试 |
| 真实世界（Franka R3 + RevoTron） | 4 任务 × 4 条件 | 真实硬件，IND + 3×OOD | 真实部署验证 |
| 人类 ego 视频 | 未公开具体数量 | 无动作标注，仅视觉时序监督 | [[delta-lwm]] + [[vlm-jepa]] 训练 |

### 实现细节

- **视觉编码器**: 冻结 [[DINOv3]]（空间特征网格，不参与梯度更新）
- **训练硬件**: 4× NVIDIA A100
- **推理硬件**: 单卡 NVIDIA RTX 4090，**30 Hz** 实时推理
- **模型规模**: 1.2B 参数（推理时，不含教师分支）
- **超参数**: $\lambda_T$、$\lambda_W$、边距 $\mu$、EMA 动量 $m$、$Q_a$、$d_z$ 均未在论文正文中公开

### 可视化结果

- **Figure 4**（潜在动作 PCA）: 不同任务下潜在动作分布清晰分离，证明 [[vlm-jepa]] 学到任务感知表示
- **Figure 5**（latent conditioning 效果）: 不同潜在动作导致不同的特征变化预测，[[Ranking Loss]] 有效保证依赖性
- **Figure 8**（真实机器人）: RevoTron 在光照/布局/背景变化下仍能稳定完成 Drawer-Long 任务

---

## 批判性思考

### 优点

1. **极高参数效率**: 以 1.2B 参数在 LIBERO-Plus 超越所有 2B~7B 参数的对比方法，参数利用效率突出。
2. **推理零额外开销**: EMA 教师分支仅存在于训练时，推理时完全移除，不引入任何延迟；30 Hz 在消费级 GPU 上实时运行。
3. **人类视频协同训练**: ego clip 与 robot clip 统一框架内同步训练，真正实现从无标注视频中受益，消融验证 LIBERO-Plus OOD 提升 10.9pp。
4. **Ranking Loss 约束**: 强制解码器依赖匹配潜在动作，防止世界模型"走捷径"忽略动作条件，设计简洁有效。

### 局限性

1. **关键超参数未公开**: EMA 动量 $m$、损失权重 $\lambda_T$/$\lambda_W$、边距 $\mu$、潜在动作维度等均未公开，复现难度较高。
2. **训练内存开销**: EMA 教师分支需要维护两份 VLM 参数，增加训练显存需求（论文承认此局限）。
3. **收敛速度慢**: 动作头以预测未来特征为条件使收敛速度变慢（论文明确指出），混合训练缓解但未完全解决。
4. **视频数据来源不透明**: 人类 ego 视频的规模、来源和预处理未详细说明，难以评估数据贡献的上界。

### 潜在改进方向

1. **语言条件化潜在动作**: 当前潜在动作目标仅由视觉时序决定，引入语言条件化可改善语言扰动场景（LIBERO-Plus 语言扰动 77.5% 仍有提升空间）。
2. **在线 EMA 蒸馏到推理端**: 探索将教师知识蒸馏到学生，进一步压缩模型或提升推理质量。
3. **多步未来预测**: 当前仅预测一步特征增量，扩展到多步滚动预测可能进一步提升长时序任务（LIBERO-Long）性能。

### 可复现性评估

- [ ] 代码开源（论文未提供代码链接）
- [ ] 预训练模型（未提供）
- [ ] 训练细节完整（关键超参数未公开，**部分**）
- [x] 数据集可获取（LIBERO、SimplerEnv 等均为公开 benchmark）

---

## 关联笔记

### 基于

- [[JEPA]]: 联合嵌入预测架构的基础范式，[[vlm-jepa]] 的理论来源
- [[V-JEPA]]: 视频 JEPA，TS-WAM 将其思想迁移到 VLM 级别
- [[EMA]]: 教师分支的核心更新机制，来自 [[BYOL]]/[[MoCo]] 等自监督方法
- [[Flow Matching]]: 动作生成头采用的生成模型
- [[DINOv3]]: 冻结视觉编码器，提供稳定的空间特征表示
- [[World Action Model]]: TS-WAM 所属的整体范式

### 对比

- [[VLA-JEPA]]: 最直接的对比方法，同为 JEPA + VLA，但 TS-WAM 增加了显式特征耦合
- [[π0.5]]: 大型预训练 VLA 基线，TS-WAM 以更少参数超越
- [[LAPA]]: 分离预训练的潜在动作方法，本文端到端设计的对比起点
- [[UniVLA]]: 另一个利用视频预训练的 VLA 方法
- [[DINO-WM]]: 基于 DINO 特征的世界模型方法

### 方法相关

- [[vlm-jepa]]: 本文提出的 VLM 级时序 JEPA 模块
- [[delta-lwm]]: 本文提出的增量潜在世界模型
- [[Ranking Loss]]: 保证潜在动作条件依赖性的对比损失
- [[Stop-Gradient]]: 教师分支梯度截断
- [[Action Chunking]]: 动作块生成方式
- [[Latent Action]]: 潜在动作表示的基础概念

### 硬件/数据相关

- [[LIBERO]]: 域内训练与测试 benchmark
- [[LIBERO-Plus]]: OOD 扰动评估 benchmark
- [[SimplerEnv]]: 跨机器人迁移 benchmark
- [[Franka Research 3]]: 真实机器人平台之一

---

## 速查卡片

> [!summary] TS-WAM (2026)
> - **核心**: [[EMA]] 教师分支（[[vlm-jepa]]）提供时序潜在动作目标，[[delta-lwm]] 将潜在动作与 DINOv3 特征增量耦合，推理时去除教师分支
> - **方法**: vlm-jepa（L1 时序对齐）+ delta-lwm（特征增量预测 + Ranking Loss）+ Flow Matching 动作头
> - **结果**: LIBERO 97.0%、LIBERO-Plus OOD **75.7%**（1.2B，SOTA）、SimplerEnv 63.83%、真实机器人 OOD 62.92%
> - **代码**: 未公开

---

*笔记创建时间: 2026-10-17*

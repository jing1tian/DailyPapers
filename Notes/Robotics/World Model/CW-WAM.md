---
title: "CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation"
method_name: "CW-WAM"
authors: [Houzhu Ding, Xinda Qi, Luqi Wang, Ming Yang]
year: 2026
venue: arXiv
tags: [world-action-model, contact-wrench, force-sensitive-manipulation, admittance-control, slow-fast-control]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.30117v1
created: 2026-10-23
---

# 论文笔记：CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Ant Group |
| 日期 | October 2026 |
| 项目主页 | [cw-wam-project.github.io](https://cw-wam-project.github.io/) |
| 对比基线 | [[世界动作模型（WAM）]] / [[DreamZero]] / [[FAWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.30117) |

---

## 一句话总结

> CW-WAM 是一个联合预测视频、运动和接触力矩的世界动作模型，通过[[Pair-Causal Attention|Pair-Causal 注意力]]和[[长程力矩记忆（LTM）|长程力矩记忆]]建模结构化运动-力矩历史，再以预测力矩驱动[[导纳控制（Admittance Control）|导纳控制器]]实现力敏操作，将方形插销成功率从 25% 提升至 70%。

---

## 核心贡献

1. **结构化运动-力矩建模**: 提出带有配对历史、独立输出头和模态专属损失的[[Contact-Wrench|接触力矩]]联合预测框架，引入[[Pair-Causal Attention|Pair-Causal 注意力]]和[[长程力矩记忆（LTM）|长程力矩记忆]]，位置误差比 WAM+Wrench 降低 5.58 mm。
2. **预测力矩作为导纳参考**: 将预测力矩直接用作 20 Hz [[导纳控制（Admittance Control）|导纳控制器]]的力参考，无需额外的校正策略，力参考 RMSE 在插销任务从 4.8 N 降至 2.7 N。
3. **慢-快运动-力控制系统**: 设计[[慢-快执行系统（Slow-Fast）|慢-快执行系统]]，20 Hz 导纳控制在 WAM 推理查询间持续运行，通过 C² 样条桥接平滑动作块，1 kHz 关节阻抗控制器跟踪轨迹。

---

## 问题背景

### 要解决的问题

力敏操作（方形插销插入、接口插拔、擦拭）依赖接触和载荷信息，而 RGB 观测无法直接提供这些信息。现有[[世界动作模型（WAM）]]仅预测视频和运动，无法感知接触力，导致在精密接触任务中失败率高。

### 现有方法的局限

- **纯视觉 WAM**（[[DreamZero]]等）：缺乏力信息，无法感知插入阻力或载荷变化。
- **简单力矩拼接**（WAM + Wrench）：直接池化历史力矩、拼接 16D 动作-力矩向量，忽略运动命令与力矩的时序配对关系，也缺乏长程历史。
- **[[FAWAM]]、[[Facet-0]]**：联合预测动作和力矩，但无结构化配对机制，且缺乏将预测力矩直接用于力控的路径。
- **[[ForceVLA]]、Reactive Diffusion Policy**：依赖触觉/力反馈，但不预测未来力矩，无法前馈。

### 本文的动机

混合运动-力控制理论指出，操作任务存在力和运动的互补约束。若 WAM 能联合预测时间对齐的运动块和力矩块，则预测的力矩可直接作为导纳控制器的期望参考，实现无需额外策略的力敏控制。

---

## 方法详解

### 模型架构

**CW-WAM** 以 [[DreamZero]]-style [[世界动作模型（WAM）]] 为骨干（[[Wan2.2-TI2V-5B]] DiT），联合预测视频潜变量、运动块和接触力矩块：

- **输入**: RGB 观测 $V_t$（3 路相机，160×320 合成图像）+ 机器人状态 $s_t$（MLP 编码）+ 语言 $\ell$（OpenCLIP + UMT5）+ 近期运动-力矩历史 $\mathcal{H}_t^{AW}$ + 长程力矩记忆 $\mathcal{H}_t^W$
- **Backbone**: [[Wan2.2-TI2V-5B]] DiT
- **核心模块**: [[Pair-Causal Attention|Pair-Causal 注意力]]保证运动-力矩时序因果性；[[长程力矩记忆（LTM）]]提供 10 s 历史
- **输出**: 视频潜变量 $Z_t^+$（2 帧）+ 运动块 $A_t = (a_{t,i})_{i=1}^{24}$（10D 每步：位置 3D + 连续 6D 旋转 + 夹爪 1D）+ 力矩块 $W_t = (\hat{w}_{t,i})_{i=1}^{24}$（6D，力 3D + 力矩 3D）
- **推理频率**: 每次查询约 0.5–0.6 s（RTX 5090，4 步 Euler），生成 1.6 s 的 24 步预测

### 核心模块

#### 模块 1: 统一交互模块（Interaction Module）

**设计动机**: 利用运动命令与接触力矩之间的时序配对关系（每个运动命令对应一段内的高频力矩采样）实现结构化历史编码。

**具体实现**:
- 近期历史覆盖 $H_s = 24$ 个间隔，频率 $f_h = 15$ Hz（1.6 s），每间隔最多 $K_w = 16$ 个高频力矩采样（150 Hz / 15 Hz ≈ 10，含 padding）
- **时序力矩编码器**：将每个样本的 6D 力矩值、相对时间、采样间隔编码为 token，validity mask 处理 padding
- 历史序列：

$$
\mathcal{H}_t^{AW} = [A_1^-, W_1^-, \ldots, A_{H_s}^-, W_{H_s}^-]
$$

其中 $A_i^-$ 为第 $i$ 间隔的运动命令 token，$W_i^-$ 为对应的力矩 token。

#### 模块 2: Pair-Causal Attention

**设计动机**: 未来预测中运动块 $A_i^+$ 需要在力矩反馈 $W_i^+$ 之前生成（运动先规划，力矩随之产生），同时需要参考过去的运动-力矩对。

**具体实现**: 对未来 token 序列 $[A_1^+, W_1^+, \ldots, A_H^+, W_H^+]$ 应用[[Pair-Causal Attention|Pair-Causal 下三角掩码]]：
- $W_i^+$ 可以 attend $A_i^+$（同对内：力矩看运动）
- $A_i^+$ **不能** attend $W_i^+$（运动不看同对未来力矩）
- 所有 token 可以 attend 早期对（双向历史访问）
- 所有未来 token 联合去噪（不是自回归）

#### 模块 3: 长程力矩记忆（LTM）

**设计动机**: 载荷感知任务（如弹簧秤搬运）需要持续的承重张力历史，近期 1.6 s 不够。

**具体实现**: 额外提供 $H_\ell = 150$ 个只含力矩的历史间隔（10 s），组成 $\mathcal{H}_t^W$，与 $\mathcal{H}_t^{AW}$ 并联输入。LTM 将插销插入成功率从 7/20 提升至 14/20。

#### 模块 4: 导纳控制器

**设计动机**: 利用预测力矩作为前馈参考，在 WAM 慢速查询间持续 20 Hz 修正末端位姿，实现力敏跟踪。

**具体实现**:

$$
e_w = \hat{w}(\varphi^w) - w^{\text{meas}}
$$

$$
\begin{aligned}
M\Delta\ddot{x} + C\Delta\dot{x} + K_a\Delta x &= -S(e_w + k_I \eta) \\
\dot{\eta} &= e_w - \lambda\eta
\end{aligned}
$$

- 半隐式 Euler 积分，频率 20 Hz，有加速度/速度/位移限幅
- $\varphi^w$：进度对齐的力矩选取（由当前 TCP 位置匹配，不后退）

---

## 关键公式

### 公式 1: [[世界动作模型（WAM）|CW-WAM 联合条件分布]]

$$
p_\theta(Z_t^+, A_t, W_t \mid V_t, s_t, \ell, \mathcal{H}_t^{AW}, \mathcal{H}_t^W)
$$

**含义**: CW-WAM 学习在 RGB 观测、机器人状态、语言指令及两类历史条件下联合生成视频潜变量、运动块和力矩块的分布。

**符号说明**:
- $V_t$: 当前 RGB 观测（3 路相机合成）
- $s_t$: 机器人状态（TCP 位姿 + 夹爪）
- $\ell$: 语言任务指令
- $\mathcal{H}_t^{AW}$: 近期 1.6 s 运动-力矩配对历史
- $\mathcal{H}_t^W$: 长程 10 s 力矩记忆（LTM）
- $Z_t^+$: 未来 2 帧视频潜变量
- $A_t$: 24 步运动块（每步 10D）
- $W_t$: 24 步力矩块（每步 6D，补偿重力后的腕力/力矩）

### 公式 2: [[Flow-Matching|Flow Matching 插值与速度目标]]

$$
x_\sigma^m = (1 - \sigma)x^m + \sigma\varepsilon^m, \quad u^m = \varepsilon^m - x^m
$$

**含义**: 在噪声水平 $\sigma$ 下对模态 $m$（视频/运动/力矩）进行线性插值，速度目标 $u^m$ 为噪声与数据的差值。

**符号说明**:
- $m \in \{V, A, W\}$: 模态（视频/运动/力矩）
- $\sigma \in [0,1]$: 噪声水平（每 block 共享）
- $x^m$: 干净数据
- $\varepsilon^m$: 标准高斯噪声
- $u^m$: flow matching 速度目标

### 公式 3: [[Flow-Matching|多模态训练损失]]

$$
\mathcal{L} = \sum_{m \in \{V, A, W\}} \lambda_m \, \mathbb{E}\!\left[\omega(\sigma) \, \text{MSE}_{M_m}(v_\theta^m, u^m)\right]
$$

**含义**: 对视频、运动、力矩三路模态分别计算加权 MSE，通过 valid-channel 归一化处理 padding，三路权重 $\lambda_m = 1$。

**符号说明**:
- $v_\theta^m$: 网络对模态 $m$ 的速度预测
- $\omega(\sigma)$: 噪声水平加权函数
- $M_m$: 模态 $m$ 的有效通道掩码（去除 padding）
- $\lambda_m = 1$: 各模态等权重

### 公式 4: [[导纳控制（Admittance Control）|力矩误差与导纳动力学]]

$$
e_w = \hat{w}(\varphi^w) - w^{\text{meas}}
$$

$$
\begin{aligned}
M\Delta\ddot{x} + C\Delta\dot{x} + K_a\Delta x &= -S(e_w + k_I \eta) \\
\dot{\eta} &= e_w - \lambda\eta
\end{aligned}
$$

**含义**: 力矩误差 $e_w$ 驱动虚拟质量-阻尼-弹簧系统，产生末端位姿修正量 $\Delta x$；积分项 $\eta$ 消除稳态误差。

**符号说明**:
- $\hat{w}(\varphi^w)$: 进度对齐选取的预测力矩参考
- $w^{\text{meas}}$: 实测腕力矩（经偏置补偿和重力补偿）
- $M, C, K_a$: 虚拟惯量、阻尼、刚度矩阵（对角）
- $S = \text{blkdiag}(s_T, s_R)$: 反馈权重矩阵（调节各方向敏感度）
- $\eta$: 积分力矩误差（指数衰减）
- $k_I$: 积分增益（s⁻¹）；$\lambda$: 衰减率（s⁻¹）

### 公式 5: [[导纳控制（Admittance Control）|修正位姿合成]]

$$
p^{\text{cmd}} = p^{\text{nom}} + R^{\text{nom}}\Delta p
$$

$$
R^{\text{cmd}} = R^{\text{nom}} \exp([\Delta\theta]_\times)
$$

**含义**: 将导纳输出的平移和旋转修正量叠加到名义位姿上，得到最终指令位姿。

**符号说明**:
- $p^{\text{nom}}, R^{\text{nom}}$: WAM 预测的名义位置和姿态
- $\Delta p, \Delta\theta$: 导纳控制器输出的平移/旋转修正量（基坐标系）
- $[\Delta\theta]_\times$: $\Delta\theta$ 的反对称矩阵（李代数到旋转矩阵的 exp map）

---

## 关键图表

### Figure 1: Teaser / 系统概览

![Figure 1: CW-WAM Teaser](https://arxiv.org/html/2610.30117v1/fig01_cw_wam_teaser_short.png)

**说明**: CW-WAM 使用过去运动命令和腕部力矩预测未来运动和接触力矩。右侧展示方形插销插入任务，成功率从 WAM 的 25% 提升至 CW-WAM 的 70%。

### Figure 2: 模型架构

![Figure 2: Model Architecture](https://arxiv.org/html/2610.30117v1/figure1-HD.png)

**说明**: 展示联合视频-运动-力矩预测的完整架构。输入包括 RGB 图像（通过 [[Wan2.2-TI2V-5B]] DiT 的视频 VAE 编码）、机器人状态（MLP）、语言（OpenCLIP + UMT5），以及配对的运动-力矩历史和 LTM。[[Pair-Causal Attention|Pair-Causal 注意力掩码]]控制未来动作-力矩 token 间的因果关系；视频路径和动作/力矩路径各有独立的投影头和损失。

### Figure 3: 慢-快运动-力控制

![Figure 3: Slow-Fast Controller](https://arxiv.org/html/2610.30117v1/fig02_slow_fast_controller_v7.png)

**说明**: 展示[[慢-快执行系统（Slow-Fast）|慢-快执行架构]]。WAM 以 ~1.5 Hz 查询生成 24 步名义轨迹（慢路径）；20 Hz [[导纳控制（Admittance Control）|导纳控制器]]实时修正位姿（快路径）；每 50 ms 重规划 0.5 s 的分段五次多项式关节轨迹；1 kHz 关节阻抗控制器跟踪。C² 桥接连续相邻动作块。

### Figure 4: 任务完成结果

![Figure 4: Task Outcomes](https://cw-wam-project.github.io/assets/images/results.webp)

**说明**: 20 次试验的任务完成情况（插销/接口插拔的插入和释放成功数）及执行质量指标（位置 RMSE、力参考 RMSE）。CW-WAM（含 LTM）的适配器插入达 19/20、释放 16/20，插销插入 14/20、释放 11/20，显著优于 WAM+Wrench（适配器插入 9/20，释放 0/20）。

### Figure 5: 接触力曲线

![Figure 5: Contact Force Profiles](https://arxiv.org/html/2610.30117v1/fig05_contact_force_profiles.png)

**说明**: 各任务的实测接触力时序曲线，与示范中位线对比。CW-WAM 的力变化（force variation）在插销插入中为 0.94 N（演示中位 1.39 N，无跟踪 1.86 N），在擦拭中为 0.85 N（演示 0.92 N，无跟踪 1.11 N），接触力更平稳。

### Figure 6: 载荷自适应抬升

![Figure 6: Load-Dependent Lifting](https://arxiv.org/html/2610.30117v1/fig06_load_dependent_lifting.png)

**说明**: 弹簧秤搬运任务的峰值抬升高度。CW-WAM 轻箱（30 g）423.6 mm、重箱（830 g）481.8 mm，与演示中位（420.9 / 492.7 mm）一致。WAM 无法区分载荷（464.4 / 447.3 mm），重箱反而抬得更低；WAM+Wrench 差值（10.4 / 24.8 mm）也大于 CW-WAM（2.7 / 10.9 mm）。

### Figure 7: 力矩反馈跟踪

![Figure 7: Wrench Feedback Tracking](https://cw-wam-project.github.io/assets/images/wrench-feedback.webp)

**说明**: 三种条件（无导纳/演示力参考导纳/预测力参考导纳）下，插销插入和擦拭的力/力矩跟踪曲线及力变化汇总。使用预测力参考后，插销插入 RMSE 从 4.8 N → 2.7 N，擦拭从 2.0 N → 1.3 N；成功率无变化，但力控质量显著提升。

### Table I: 导纳增益设置

| Task | 反馈权重 | M | C | K_a | k_I (s⁻¹) | λ (s⁻¹) |
|------|---------|---|---|-----|-----------|---------|
| Peg insertion | $s_T = \text{diag}(1,1,0.05)$ | 5 | 75 | 200 | 0.5 | 0.5 |
| | $s_R = \text{diag}(1,1,0.2)$ | 0.1 | 1.5 | 4 | 0.5 | 0.5 |
| Adapter insertion | $s_T = \text{diag}(1,1,0.5)$ | 5 | 75 | 200 | 0.1 | 1 |
| | $s_R = \text{diag}(1,1,1)$ | 0.1 | 1.5 | 4 | 0.1 | 1 |
| Whiteboard wiping | $s_T = \text{diag}(0,0,1)$ | 3 | 34.64 | 100 | 0.1 | 1 |
| | $s_R = \text{diag}(0,0,0)$ | — | — | — | — | — |

**说明**: 插销和接口插拔任务在 XY 方向有较低导纳（$s_T$ 的前两项小），Z 方向更敏感；擦拭任务只在 Z 方向施加力参考跟踪。增益由人工调节。

### Table II: 开环预测消融（373 个 24 步窗口，22 个测试 episode）

| 模型 | e_p [mm] ↓ | e_R [°] ↓ | e_F [N] ↓ | e_I [N·s] ↓ |
|------|------------|-----------|-----------|-------------|
| WAM | 26.97 | 2.85 | — | — |
| WAM + Wrench (padded norm) | 26.32 | 2.76 | 3.15 | 2.33 |
| WAM + Wrench (valid norm) | 25.50 | 2.80 | 3.03 | 2.21 |
| CW-WAM w/o LTM and PC | 22.88 | 2.25 | 2.92 | 2.11 |
| CW-WAM w/o LTM | **19.92** | **2.02** | 2.84 | 2.17 |
| CW-WAM (full) | 20.69 | 2.07 | **2.79** | **2.01** |

**关键发现**: 交互模块（配对历史 + 独立流 + PC 注意力）相比同等归一化的 WAM+Wrench 降低位置误差 5.58 mm（95% CI [3.65, 7.84]）；[[Pair-Causal Attention|Pair-Causal 注意力]]单独贡献 2.96 mm（95% CI [2.05, 3.89]）；LTM 主要改善力矩预测（$e_F$、$e_I$）。

---

## 实验

### 数据集

| 任务 | 演示数 | 特点 | 评价指标 |
|------|--------|------|---------|
| 方形插销插入（1 mm 间隙） | 213 | 38 mm 插销，40 mm 孔，约 50 mm 插入深度 | 插入/释放后保持成功率 |
| 电源适配器插拔 | 97 | 松放适配器抓取后插入固定底座 | 插入/释放成功率 |
| 白板擦拭 | 64 | 拾取橡皮擦，清除标记痕迹 | 去除率（简单/复杂图案） |
| 弹簧秤搬运 | 76 | 30 g / 830 g 相同外观箱子，需高抬才能越过边缘 | 峰值抬升高度 |

**总计**: 450 次双臂遥操作演示（7-DoF Franka + LinkTouch 六轴 F/T 传感器 150 Hz，3 RealSense 相机，15 Hz 记录）。

### 实现细节

- **Backbone**: [[Wan2.2-TI2V-5B]] DiT（视频 VAE、OpenCLIP、UMT5 冻结）
- **优化器**: AdamW，峰值学习率 1e-5，权重衰减 1e-5
- **Batch Size**: 全局 8，任务均衡采样
- **训练步数**: 50,000 步，5% warmup + cosine decay
- **硬件**: 8× NVIDIA H20 GPU（训练），NVIDIA RTX 5090（推理，~0.5–0.6 s/query，4 步 Euler）
- **历史设置**: $H_s = H = 24$，$H_\ell = 150$，$f_h = 15$ Hz，$K_w = 16$

### 可视化结果

- 成功插销试验中，CW-WAM 的 TCP 位置 RMSE 为 2.8 mm（WAM+Wrench 为 12.5 mm），力参考 RMSE 为 4.8 N（WAM+Wrench 为 15.2 N），轨迹显著更稳。
- 擦拭任务中所有方法均可完成简单图案（20/20），CW-WAM 在 6 种复杂图案上平均去除率 85.9%。
- 弹簧秤任务：WAM 将重箱抬得比轻箱更低（无载荷感知），CW-WAM 通过 LTM 正确区分重量。

---

## 批判性思考

### 优点

1. **结构化建模**: Pair-Causal 注意力将运动-力矩因果关系显式编码，比简单拼接更符合物理直觉，位置误差改善显著。
2. **无缝力控集成**: 预测力矩直接作为导纳参考，无需独立的力控策略，架构简洁且工程可行。
3. **LTM 对载荷自适应至关重要**: 10 s 力矩记忆的消融显示，短期历史对需要持续张力感知的任务远远不够。

### 局限性

1. **任务域窄**: 仅在 4 个任务、450 次演示内评估，未验证对新物体和新任务的泛化能力。
2. **增益手工调节**: 导纳控制器参数（$M, C, K_a, S, k_I, \lambda$）均手动设置，不同任务需重新调节，缺乏自适应性。
3. **接触方向预定义**: 导纳反馈方向（$S$ 矩阵）是固定的，无法自动适应未知接触方向。

### 潜在改进方向

1. **学习控制器参数**: 端到端学习 $S$、接触帧、阻抗参数，减少手工调节。
2. **闭环预测引导**: 将力矩误差反馈融入 WAM 的下一次查询条件（目前预测和控制仍是开环叠加）。
3. **跨任务泛化**: 扩展到更多种类的插入、装配和接触密集型操作，验证方法通用性。

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（学习率、批大小、步数均有报告）
- [ ] 数据集可获取（私有遥操作数据）

---

## 关联笔记

### 基于

- [[DreamZero]]: WAM 骨干 + Wan2.2-TI2V-5B DiT，CW-WAM 直接扩展其架构
- [[Wan2.2-TI2V-5B]]: 视频生成 backbone
- [[Flow-Matching]]: 训练范式，线性插值 + 速度目标

### 对比

- [[世界动作模型（WAM）]]: 无力矩预测，作为 baseline
- [[FAWAM]]: 也联合预测动作和力矩，但无配对结构和导纳控制路径
- [[ForceVLA]]: 依赖触觉反馈的力感知策略

### 方法相关

- [[Contact-Wrench]]: 核心预测对象（F/T 传感器输出）
- [[Pair-Causal Attention]]: 新提出的运动-力矩因果注意力机制
- [[长程力矩记忆（LTM）]]: 10 s 力矩历史记忆模块
- [[导纳控制（Admittance Control）]]: 将预测力矩转化为位姿修正
- [[慢-快执行系统（Slow-Fast）]]: WAM 慢查询 + 导纳快控制的分层架构
- [[Action Diffusion]]: 动作 chunk 执行框架的前身

### 硬件/数据相关

- [[LinkTouch 六轴 F/T 传感器]]: 腕部力矩传感（150 Hz，10 Hz 低通滤波后使用）
- [[Franka Emika]]: 7-DoF 机械臂平台

---

## 速查卡片

> [!summary] CW-WAM (Ding et al., 2026)
> - **核心**: 联合预测运动+接触力矩，用预测力矩驱动 20 Hz 导纳控制器
> - **方法**: Pair-Causal 注意力 + 长程力矩记忆（LTM）+ 慢-快控制系统
> - **结果**: 插销成功率 25%→70%，适配器 40%→95%，力参考 RMSE 4.8→2.7 N
> - **代码**: 未公开（项目主页 [cw-wam-project.github.io](https://cw-wam-project.github.io)）

---

*笔记创建时间: 2026-10-23*

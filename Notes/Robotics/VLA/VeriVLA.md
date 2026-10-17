---
title: "VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models"
method_name: "VeriVLA"
authors: [Xueli Sun, Yuxiang Lu, Yantong Liu, Xianzhe Fan, Xi Chen, Zhe Liu, Hengshuang Zhao]
year: 2026
venue: arXiv
tags: [vla, test-time-verification, world-model, flow-matching, latent-jepa, robot-manipulation, progress-estimation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.21985v1
created: 2026-10-17
---

# 论文笔记：VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The University of Hong Kong |
| 日期 | October 2026 |
| 项目主页 | [shirley430316.github.io/projects/verivla](https://shirley430316.github.io/projects/verivla/) |
| 对比基线 | [[RoboMonkey]], [[GPC-RANK]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.21985) / Code (未公开) |

---

## 一句话总结

> 针对 Flow-based [[VLA（视觉-语言-动作模型）|VLA]] 的随机性问题，提出双轴无训练触发器识别关键状态，并利用基于 [[Joint Embedding Predictive Architecture|JEPA]] 的潜在世界模型在候选动作块中选出进度最优的一个，在真实场景平均成功率从 50.8% 提升至 65.5%。

---

## 核心贡献

1. **双轴无训练触发器（Two-Axis Trigger）**: 通过时序变化（$C_1$）和跨样本方差（$C_2$）两个互补维度识别需要额外采样验证的关键状态，无需任何训练。
2. **JEPA 潜在世界模型验证器**: 利用冻结的 [[VJEPA-2|V-JEPA 2]] 编码器 + 动作条件自回归预测器，联合预测未来潜在特征和 [[任务进度|Delta Progress]]，在潜在空间完成候选动作排序。
3. **端到端低延迟流水线**: 通过 KV 缓存复用、并行计算和 PyTorch 预编译，将每次触发验证的额外延迟压缩至 326 ms（单卡 RTX 4090），支持实时双臂操作。

---

## 问题背景

### 要解决的问题

[[Flow Matching|Flow-based]] [[VLA（视觉-语言-动作模型）|VLA]] 策略（如 [[π₀.₅]]）通过噪声种子 $\epsilon$ 采样 [[动作分块|动作块]]，不同种子产出质量差异显著。策略正常部署只使用单次采样，却无法评估质量——一个低质量动作块在精细操作（如插管、移液）中可能引起不可逆失败。

### 现有方法的局限

- **VLM 验证器**（如 RoboMonkey）：依赖 VLM 对图像的静态评分，缺乏对物理动态的建模，无法区分精细操作的微妙状态差异。
- **像素级世界模型**（如 GPC-RANK）：先做像素生成再评估，计算成本高且生成质量在测试分布外退化。
- **常开验证**（Always-on）：忽略策略方差的非均匀性，浪费计算资源——失败更集中在接触富集（contact-rich）的片段，而非整个轨迹。

### 本文的动机

- 策略的方差在轨迹中非均匀分布，抓取/释放等关键操作点方差最大（Figure 1）。
- 验证本质上是低维标量判断（好/坏），不需要解高维像素生成问题，[[Latent World Dynamics|潜在动态建模]] 足够。
- 双轴触发器的两个维度信号互补：时序维度捕捉操作复杂性，跨样本维度捕捉策略不确定性。

---

## 方法详解

### 模型架构

VeriVLA 由两个核心模块构成，整体流程如下：

- **输入**: 观测 $o_t$ + 语言指令 $l$ + 本体状态 $s_t$，基础策略 $\pi_\theta$（[[π₀.₅]] / Hy-VLA）
- **触发阶段**: [[动作分块|动作块]] $a^{(1)}, a^{(2)}$ → Two-Axis Trigger 判断是否需要验证
- **验证阶段**: 扩展采样至 $K=10$ 个候选 → [[VJEPA-2|V-JEPA 2]] 编码 + 动作条件预测器 → Delta Progress 排序
- **执行**: 直接执行（未触发）或执行最高进度候选（已触发）

### 核心模块

#### 模块 1：双轴无训练触发器（Two-Axis Trigger）

**设计动机**: 仅在真正关键、高不确定性的状态下激活验证，节省计算资源。

**具体实现**:

先对每个动作块按平均动作幅度归一化：

$$
\tilde{a}^{(n)}_h = \frac{a^{(n)}_h}{\frac{1}{H}\sum_{h'=1}^{H}\|a^{(n)}_{h'}\|}
$$

- **时序项** $C_1$（捕捉块内时序复杂性，反映接触富集程度）：

$$
C_1 = \frac{1}{k} \sum_{n=1}^{k} \frac{1}{H-1} \sum_{h=1}^{H-1} \left\| \tilde{a}^{(n)}_{h+1} - \tilde{a}^{(n)}_h \right\|^2
$$

- **跨样本项** $C_2$（捕捉多样本间不一致性，反映策略不确定度）：

$$
C_2 = \frac{1}{H} \sum_{h=1}^{H} \frac{1}{k} \sum_{n=1}^{k} \left\| a^{(n)}_h - \bar{a}_h \right\|^2, \quad \bar{a}_h = \frac{1}{k}\sum_{n=1}^{k} a^{(n)}_h
$$

- **触发分数**（两项取百分位后取最大）：

$$
\tau(s) = \max\left(\tau_{temp}(s),\; \tau_{cross}(s)\right)
$$

当 $\tau(s) > \tau_0$（唯一超参）时触发验证。百分位通过部署前滚出基础策略收集参考分布进行校准（无梯度）。

#### 模块 2：JEPA 潜在世界模型验证器

**设计动机**: 在潜在特征空间预测动作效果和任务进度，避免像素重建，计算高效。

**架构**:
- **视觉编码器**: 冻结的 [[VJEPA-2|V-JEPA 2]]（预训练视频理解模型）
- **具身编码器**: 任务特定的状态和动作编码器（可投影到预测器嵌入空间）
- **动作条件预测器**: 自回归 Transformer，以前序预测潜在特征和当前动作块为条件，逐 sub-chunk 滚出
- **任务 Query Token**: 可学习 token，专门条件化 Delta Progress 预测
- **注意力掩码**: 将动态预测与 Query Token 解耦（Figure 3c），动力学模型保持任务无关

**多步骤滚出**: 动作块（horizon $H$）被分为 $M$ 个 sub-chunk，每步预测一个潜在特征，提供更密集监督并限制误差累积。

### 训练细节

**Delta Progress 定义（Eq. 3）**:

$$
\Delta p(a_{t:t+H}) = p(s_{t+H}) - p(s_t)
$$

其中绝对进度标签 $p(s)$ 由 Robometer 提供（任务完成度 0–1）。

**训练损失**:

**进度损失（Eq. 8）**——Huber 回归 + [[Pairwise Ranking Loss|对比排序损失]]：

$$
\mathcal{L}_{prog} = \mathcal{L}_{Huber}\!\left(\Delta\hat{p},\, \Delta p\right) + \lambda_{rank} \sum_{\substack{(i,j):\\ \Delta p^i > \Delta p^j}} \max\!\left(0,\; \Delta\hat{p}^j - \Delta\hat{p}^i + \alpha\right)
$$

**含义**: Huber 项保证回归精度，排序项保证正确地对同任务候选对进行排序。

**符号说明**:
- $\Delta\hat{p}$: 预测的 delta 进度
- $\Delta p$: Robometer 给出的真实 delta 进度
- $\lambda_{rank}$: 排序损失权重（实验中取 1）
- $\alpha$: 排序间隔（margin）

**视觉/潜在预测损失（Eq. 9）**——逐 sub-chunk L1 损失：

$$
\mathcal{L}_{vis} = \frac{1}{M} \sum_{m=1}^{M} \left\| \hat{z}_m - z_m \right\|_1
$$

**含义**: 让预测器学习动作条件下的潜在特征动态，提供密集动力学监督。

**符号说明**:
- $\hat{z}_m$: 第 $m$ 步的预测潜在特征
- $z_m$: V-JEPA 2 编码的目标潜在特征
- $M$: sub-chunk 数量（= 5）

**总损失（Eq. 10）**：

$$
\mathcal{L} = \mathcal{L}_{prog} + \lambda_{vis} \cdot \mathcal{L}_{vis}
$$

**含义**: 进度预测为主要目标，潜在动力学为辅助目标。

**符号说明**:
- $\lambda_{vis}$: 视觉损失权重（实验中取 1）

训练同时使用 [[teacher forcing|Teacher Forcing]] 和 [[Self-forcing|Self-Forcing]]（自回归推理时仅用 Self-Forcing）。预测器在 DROID 数据集（23k 条轨迹）上预训练，再在任务特定数据上微调。

**验证时动作选择（Eq. 4）**:

$$
\hat{a}^* = \arg\max_{n \in [K]} \Delta\hat{p}^{(n)}
$$

---

## 关键图表

### Figure 1: 策略方差与触发器直觉

![Figure 1](https://arxiv.org/html/2610.21985v1/intuition.png)

**说明**: LIBERO 任务中策略方差分布非均匀，失败集中在 grasp/release 等关键接触时刻。VeriVLA 的触发分数（右侧）在这些关键时刻产生明显峰值，验证了双轴触发设计的合理性。

### Figure 2: 范式对比

![Figure 2](https://arxiv.org/html/2610.21985v1/concept_comp.png)

**说明**: 对比 VLM 验证器（RoboMonkey）、像素级世界模型（GPC-RANK）和 VeriVLA 的潜在世界模型验证范式。VeriVLA 跳过像素生成，直接在潜在空间估计动作效果，兼顾效率和物理建模能力。

### Figure 3: VeriVLA 整体流程

![Figure 3](https://arxiv.org/html/2610.21985v1/method.png)

**说明**: (a) 双轴触发器根据 $C_1$/$C_2$ 决定是否触发验证；(b) 触发后，$K=10$ 个候选块经验证器打分，执行最高分候选；(c) 注意力掩码将 Delta Progress Query Token 与潜在预测解耦，保持动力学模型任务无关。

### Figure 4: 真实机器人实验平台

![Figure 4](https://arxiv.org/html/2610.21985v1/real_world.png)

**说明**: AgileX PiperX 双臂机器人，配备 RealSense D455 前置相机和两个 D435 手腕相机。三个任务：笔插入（pen insertion）、移液（pipetting）和苹果摆放（apple placement），均涉及精细接触操作。

### Figure 5: 延迟分析

![Figure 5](https://shirley430316.github.io/projects/verivla/assets/latency-a.png)

**说明**: (a) 各优化阶段的每决策验证延迟——KV 缓存复用（O1）带来最大提速；(b) 三个真实任务的触发率和平均每块附加延迟，笔插入触发率最高（操作最复杂），移液次之。

### Figure 6: RoboTwin 触发阈值扫描

> 注：Figure 6 在 arXiv HTML 中未提取到图片 URL，内容为：触发阈值 $\tau_0$ 的扫描曲线，展示 VeriVLA 在不同触发率下均优于基线，常开（always-on，阈值=0）时性能最高，触发器可在保持大部分增益的同时显著降低平均延迟。

### Figure 7: LIBERO 关键时刻可视化（附录）

![Figure 7](https://arxiv.org/html/2610.21985v1/Figures_appendix/events_r1_task0.png)

**说明**: LIBERO 轨迹中，Critical Index（触发分数）在 grasp 和 place 等接触事件附近出现峰值，与基于仿真器接触状态构建的 critical label 吻合。

### Figure 8: RoboTwin 关键时刻可视化（附录）

![Figure 8](https://arxiv.org/html/2610.21985v1/Figures_appendix/_contact_crit_success_seed24_curve34.png)

**说明**: RoboTwin 轨迹中，触发分数（Critical Index）与基于接触翻转事件构建的 critical label 高度一致，验证了触发器的物理意义。

### Table 1: 真实机器人实验结果（完成度得分 %）

| 方法 | Rank Loss ↓ | Huber Loss ↓ | 笔插入 | 移液 | 苹果摆放 | 平均 |
|------|------------|-------------|--------|------|----------|------|
| π0.5 基线 | — | — | 45.0 | 52.5 | 55.0 | 50.8 |
| + RoboMonkey | 1.19 | 0.37 | 49.0 | 55.5 | 63.5 | 56.0 |
| + GPC-RANK | 1.18 | 0.30 | 46.0 | 54.5 | 58.0 | 52.8 |
| **+ VeriVLA** | **1.09** | **0.25** | **60.0** | **63.5** | **73.0** | **65.5** |

**关键发现**: VeriVLA 平均提升 14.7 分，远超 RoboMonkey（+5.2）和 GPC-RANK（+2.0）。RoboMonkey 在粗定位任务（苹果摆放）表现尚可，但精细操作（笔插入）收效甚微；GPC-RANK 受测试时生成质量下降影响较大。

### Table 2: RoboTwin 2.0 仿真实验（成功率 %，常开验证）

**Clean 场景**:

| 方法 | Mug | Switch | Seal | Mouse | Scan | 平均 |
|------|-----|--------|------|-------|------|------|
| Hy-VLA 基线 | 38.0 | 44.0 | 74.0 | 88.0 | 88.0 | 66.4 |
| + RoboMonkey | 34.0 | 48.0 | 70.0 | 88.0 | 86.0 | 65.2 |
| + GPC-RANK | 42.0 | 50.0 | 78.0 | 88.0 | 86.0 | 68.8 |
| **+ VeriVLA** | **48.0** | **50.0** | **82.0** | **96.0** | **92.0** | **73.6** |

**Clean2Random（训练干净场景，测试随机化场景）**:

| 方法 | Mug | Switch | Seal | Mouse | Scan | 平均 |
|------|-----|--------|------|-------|------|------|
| Hy-VLA 基线 | 16.0 | 50.0 | 72.0 | 80.0 | 96.0 | 62.8 |
| + RoboMonkey | 22.0 | 44.0 | 78.0 | 86.0 | 94.0 | 64.8 |
| + GPC-RANK | 22.0 | 50.0 | 80.0 | 82.0 | 90.0 | 64.8 |
| **+ VeriVLA** | **26.0** | **56.0** | **78.0** | **92.0** | **96.0** | **69.6** |

**关键发现**: Clean 场景提升 7.2 分，Clean2Random 提升 6.8 分，表明潜在世界模型在分布外场景仍有一定泛化性。

### Table 3: LIBERO 实验（成功率 %）

**π0.5 基础策略**:

| 方法 | Spatial | Object | Goal | LIBERO-10 | 平均 |
|------|---------|--------|------|-----------|------|
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| + RoboMonkey | 98.6 | 99.0 | 97.6 | 94.0 | 97.3 |
| + GPC-RANK | 97.6 | 96.8 | 97.8 | 89.0 | 95.3 |
| **+ VeriVLA** | **98.8** | **98.8** | **98.6** | **94.0** | **97.6** |

**StarVLA-GR00T 基础策略**:

| 方法 | Spatial | Object | Goal | LIBERO-10 | 平均 |
|------|---------|--------|------|-----------|------|
| StarVLA-GR00T | 98.0 | 99.4 | 96.4 | 91.0 | 96.2 |
| + RoboMonkey | 98.2 | 99.4 | 96.0 | 91.6 | 96.3 |
| + GPC-RANK | 98.6 | 99.2 | 96.6 | 92.4 | 96.7 |
| **+ VeriVLA** | **98.4** | **99.0** | **97.8** | **93.6** | **97.2** |

**关键发现**: LIBERO 基础成功率已较高，最大增益在最不饱和的 LIBERO-10 子集（+1.6）。验证器在两个不同基础策略上均有效，证明了跨策略泛化性。

### Table 4: 消融实验（真实机器人，完成度得分 %）

| 方法 | Rank Loss ↓ | Huber Loss ↓ | 笔插入 | 移液 | 苹果摆放 | 平均 |
|------|------------|-------------|--------|------|----------|------|
| **VeriVLA（完整）** | **1.09** | **0.25** | **60.0** | **63.5** | **73.0** | **65.5** |
| w/o 世界模型 | 1.21 | 0.26 | 52.0 | 55.5 | 58.0 | 55.2 |
| w/o 预训练 | 1.17 | 0.25 | 54.0 | 61.0 | 63.5 | 59.5 |

**关键发现**: 去除世界模型（仅用进度回归 Transformer）下降 10.3 分，说明密集动力学监督不可缺少；去除 DROID 预训练下降 6.0 分，苹果摆放任务（需更多泛化）损失最大，移液（几乎固定瓶位）损失最小。

### Table 5: 触发器对比分析

| 方法 | Status AUC ↑ | Variance Corr. ↑ | k / 附加延迟 (ms) ↓ |
|------|-------------|-----------------|---------------------|
| 随机触发 | 0.487 | — | — |
| DINO + MLP（有监督） | 0.702 | 0.630 | 0 / 0 |
| ACE（动作块熵） | 0.588 | 0.716 | 2 / 14 |
| 仅 $T_{temp}$ | 0.704 | 0.789 | 1 / 0 |
| 仅 $T_{cross}$ | 0.694 | 0.877 | 2 / 14 |
| **VeriVLA 双轴** | **0.727** | **0.887** | **2 / 14** |

**关键发现**: 双轴触发优于任一单轴；时序轴在状态识别（Status AUC）上更优，跨样本轴在方差相关性上更优，二者信号互补。双轴触发以相同计算量（2 次采样）超越有监督的 DINO+MLP 方法。

### Table 6: 触发器类型对比（附录）

| 方法 | 触发类型 | 世界建模 | 阶段数 | 预测目标 |
|------|---------|---------|--------|---------|
| RoboMonkey | 视觉/文本 | 无 | 多阶段 | 成功/失败 |
| GPC-RANK | 基于图像 | 像素级 | 多阶段 | 排序分数 |
| DreamAvoid | 基于图像 | 像素级 | 多阶段 | 避免失败 |
| **VeriVLA** | **动作（无训练）** | **潜在** | **单阶段** | **潜在 Delta Progress** |

**关键发现**: VeriVLA 是唯一同时具备动作触发、潜在世界建模、单阶段验证和潜在 Delta Progress 预测的方法。

### Table 7: 训练超参（附录）

| 超参 | 值 |
|------|-----|
| 学习率 | 5×10⁻⁴ |
| 权重衰减 | 0.04 |
| Batch size | 32 |
| 优化器 | AdamW |
| JEPA 损失权重 | 1 |
| 排序损失权重 | 1 |
| 输入裁剪 | 256×256 |
| Patch 大小 | 16 |
| Tubelet | 2 |
| Sub-chunk 数量 M | 5 |
| 硬件 | 1× NVIDIA H800 (80 GB) |

### Table 8: 真实机器人任务完成度评分标准（附录）

| 任务 | 评分点 | 分值 |
|------|--------|------|
| 笔插入 | 抓取笔 | 0.5 |
| | 插入瓶中 | 0.5 |
| 移液 | 右手移至右瓶 | 0.25 |
| | 左手吸取 | 0.25 |
| | 右手移至左瓶 | 0.25 |
| | 左手注入 | 0.25 |
| 苹果摆放 | 抓取苹果 | 0.5 |
| | 放入容器（一次成功）| 0.5 |
| | 放入容器（多次尝试）| 0.25 |

### Table 9: 计算成本（附录）

| 组件 | 延迟 (ms) |
|------|----------|
| VLA 策略 | 82 |
| 双轴触发器（k=2） | 14 |
| 额外采样（8 个候选） | 80 |
| 验证器前向传播（10 候选） | 246 |
| **总计（触发器未激活）** | **96** |
| **总计（触发器激活）** | **422** |
| 验证额外延迟 | 326 |

**显存**: VLA 策略 9.8 GB，验证器 8.7 GB，合计 18.5 GB。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人任务（3 tasks） | 50 示范/任务 | 精细双臂操作 | 微调基础策略 + 训练验证器 |
| DROID | 23k 轨迹 | 多样真实场景 | 预训练验证器预测器 |
| RoboTwin 2.0 | 5 任务 × Clean/Random | 双臂仿真基准 | 仿真评估 |
| LIBERO | 4 套件 × 10 任务 = 40 任务 | 桌面操作仿真 | 仿真评估 |

### 实现细节

- **基础策略**: [[π₀.₅]]（真实机器人 + LIBERO）/ Hy-Embodied-0.5-VLA（RoboTwin）
- **视觉编码器**: 冻结的 [[VJEPA-2|V-JEPA 2]]
- **优化器**: [[AdamW]]，lr=5×10⁻⁴，weight decay=0.04
- **Batch Size**: 32
- **训练硬件**: NVIDIA H800 (80 GB)
- **推理硬件**: RTX 4090（单卡）
- **进度标签来源**: Robometer（闭环回放评估工具）
- **触发器校准**: 部署前在任务环境中滚出基础策略收集参考分布（无梯度）

### 可视化结果

真实机器人实验中：
- **笔插入**：基础策略经常在插入阶段失败（需精确对齐），VeriVLA 通过选择更稳定的抓取姿态显著改善。
- **移液**：多步骤操作（4 个评分阶段），验证在吸取步骤效果最明显。
- **苹果摆放**：任务相对宽松，RoboMonkey 也有收益，VeriVLA 进一步提升摆放精度。

---

## 批判性思考

### 优点

1. **无训练触发器**: 双轴触发器不需要任何额外训练，迁移成本低，可即插即用于任何 flow-based VLA。
2. **潜在验证高效**: 跳过像素生成，直接在潜在空间评估，326 ms 额外延迟对大多数真实场景可接受。
3. **跨策略/跨场景泛化**: 在两种不同基础策略（π0.5、StarVLA-GR00T）和 Clean2Random 分布迁移上均有效。
4. **模块化设计**: 验证器一个实例可覆盖同机器人平台的所有任务，无需逐任务训练。

### 局限性

1. **采样多样性有限**: 重复采样不同噪声种子的多样性仍然受限于策略本身的表达能力，当策略在关键状态完全崩溃时验证器也无法选出好的动作。
2. **冻结视觉编码器**: V-JEPA 2 并非针对操作进度优化，可能错过细微的接触状态变化。
3. **需要 Robometer 标签**: 训练验证器需要闭环回放来收集进度标签，对新任务的数据采集有一定成本。
4. **验证器需要任务特定微调**: 尽管预测器在 DROID 上预训练，仍需针对目标任务微调。

### 潜在改进方向

1. **联合学习编码器和预测器**: 将进度估计作为优化目标，在 [[Joint Embedding Predictive Architecture|JEPA]] 框架内端到端学习，使表征更适配操作进度判断。
2. **探索更多样的采样策略**: 除随机噪声种子外，引入结构化扰动（如温度调整、分布引导）扩展候选多样性。
3. **触发器自适应阈值**: 基于任务历史动态调整 $\tau_0$，减少手动校准。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（附录 B）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均公开）

---

## 关联笔记

### 基于

- [[π₀.₅]]: 真实机器人实验的基础 flow-based VLA 策略
- [[VJEPA-2]]: 提供冻结的视觉特征编码，潜在世界模型的骨干
- [[Joint Embedding Predictive Architecture|JEPA]]: 验证器的核心架构范式
- [[Flow Matching]]: VeriVLA 目标的基础策略类型

### 对比

- [[RoboMonkey]]: VLM-based 验证器基线，缺乏物理动态建模
- [[GPC-RANK]]: 像素级世界模型验证器基线，计算成本高且分布外退化

### 方法相关

- [[动作分块]]: 验证器的评估单元
- [[任务进度]]: Delta Progress 是验证的核心信号
- [[Latent World Dynamics]]: 本文验证器的潜在动力学预测组件
- [[Pairwise Ranking Loss]]: 排序损失训练验证器
- [[Best-of-N]]: 验证框架的核心采样-选择范式
- [[测试时缩放]]: 测试时扩大计算量的宏观范式
- [[KV 缓存]]: 验证器延迟优化的关键技术

### 硬件/数据相关

- [[AgileX PiperX]]: 真实机器人实验平台（双臂）
- [[RoboTwin 2.0]]: 仿真评估基准
- [[LIBERO]]: 仿真评估基准

---

## 速查卡片

> [!summary] VeriVLA
> - **核心**: 自适应测试时验证框架，解决 flow-based VLA 随机采样质量不稳定问题
> - **方法**: 双轴无训练触发器（时序+跨样本）+ JEPA 潜在世界模型验证器（Delta Progress 排序）
> - **结果**: 真实场景 50.8% → 65.5%（+14.7%），RoboTwin 66.4% → 73.6%，326 ms 额外延迟
> - **代码**: 暂未开源（项目主页：shirley430316.github.io/projects/verivla）

---

*笔记创建时间: 2026-10-17*

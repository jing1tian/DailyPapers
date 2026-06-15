---
title: "FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation"
method_name: "FAWAM"
authors: [Haotian He, Zeyu Yan, Qipeng Liu, Ning Guo, Wenzhao Lian]
year: 2026
venue: arXiv
tags: [force-aware-manipulation, contact-rich-manipulation, robot-policy, closed-loop-control, world-action-model]
zotero_collection: Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2606.08555
created: 2026-06-10
---

# 论文笔记：FAWAM: Force-Aware World Action Models for Closed-Loop Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未注明（arXiv 投稿） |
| 日期 | June 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Pi05]]、[[ForceVLA]]、[[FACTR]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.08555) / Code 暂未开放 |

---

## 一句话总结

> FAWAM 在感知、预测、执行三个层次系统性地融合力/力矩信号，通过联合预测动作与接触力轨迹、在线残差修正，将接触密集型操作成功率提升至 85%。

---

## 核心贡献

1. **力感知动作生成（Force-Envisioned Action Model）**: 将历史 6 轴力/力矩信号编码为接触特征，通过 [[AdaLN]] 调制层将力信息注入动作生成，同时联合预测未来力轨迹以显式建模接触演化
2. **力引导残差修正（Force-Guided Residual Corrector）**: 以预测力轨迹为参考，实时比较测量值与预测值，以 10 Hz 频率生成残差动作修正，实现闭环力反馈控制
3. **三阶段力集成框架**: 在感知（历史力编码）、预测（联合力/动作预测）、执行（在线残差修正）三个层次上系统性地利用力信号，与仅将力作为附加观测的方法相比有本质区别

---

## 问题背景

### 要解决的问题

接触密集型操作任务（如擦白板、削黄瓜、旋转箱体、擦花瓶）需要机器人对接触力进行精确感知和控制。现有视觉-语言-动作（[[VLA]]）模型主要依赖视觉信号，在面对接触变化时缺乏有效的力反馈机制。

### 现有方法的局限

- **纯视觉方法**（如 [[Pi05]]、GE-Act）：无法感知接触力，遭遇意外接触偏差时无法修正，平均成功率约 48%
- **力作为附加观测**（ForceVLA、TA-VLA）：仅将力信号拼接为额外输入，未能充分利用力信息预测未来接触演化，成功率约 64%
- **缺乏闭环力反馈**：现有方法在执行时无法利用预测力轨迹与实测值的偏差进行实时修正

### 本文的动机

力信号在接触感知、接触演化预测、执行修正三个层次各有独特价值。作者认为系统性地在每个层次利用力信息，而非仅在某一环节引入，才能真正发挥力传感器的潜力，实现鲁棒的接触密集型操作。

---

## 方法详解

### 模型架构

FAWAM 采用两阶段训练的 **[[World Action Model]]** 架构：

- **输入**: 语言指令 $l$ + 多视角图像观测 $I_{t-k:t}$ + 历史力/力矩信号 $\mathbf{w}_{t-H_f+1:t}$ + 本体感知状态 $\mathbf{q}_t$
- **Stage 1 Backbone**: 视频生成模型，通过 [[Rectified Flow]] 框架预测未来视觉帧
- **Stage 2**: 力感知动作解码器，在冻结视频模型基础上联合预测动作块与力轨迹
- **在线模块**: 力引导残差修正器（三层 [[MLP]]），以 10 Hz 运行
- **输出**: [[Action Chunking|动作块]] $\mathbf{a}_{t+1:t+H_p}$ + 修正残差 $\Delta\mathbf{a}^{res}$

### 核心模块

#### 模块 1：Force-Envisioned Action Model（力感知动作模型）

**设计动机**: 单纯将力拼接为观测向量会因多模态信息相互干扰而效果有限；通过 [[AdaLN]] 调制层可以让力信号"软性"地调整视频模型中每个 Transformer block 的归一化参数，实现深层次的力-视觉-动作融合。

**具体实现**:

1. 轻量级 MLP 编码历史力序列为接触特征 $\mathbf{c}_t$
2. 每个 Transformer block 的 [[AdaLN]] 参数（shift, scale, gate）由 $\mathbf{c}_t$ 进行残差调制
3. 零初始化调制层（[[Zero Initialization]]），训练初期保持基础模型行为，逐步学习力条件
4. 解码器同时输出动作速度场 $\hat{v}_t^A$ 和力/力矩速度场 $\hat{v}_t^W$，实现联合预测

#### 模块 2：Force-Guided Residual Corrector（力引导残差修正器）

**设计动机**: 基础策略预测的力轨迹提供了"期望接触参考"，执行时若测量力与预测力出现偏差，说明发生了意外接触变化，应立即触发修正。借鉴人类遥操作中的干预信号学习残差修正。

**具体实现**:

- 输入：力跟踪误差 $\mathbf{e}^w$、接触特征 $\mathbf{c}_t$、待执行基础动作 $\mathbf{a}^{base}$、预测力轨迹 $\mathbf{w}^{pred}$、本体感知 $\mathbf{q}_t$
- 输出：残差动作 $\Delta\mathbf{a}^{res}$ + 介入门控 $g_t^{res}$（标量）
- 二值门控阈值 $\tau = 0.8$（保守设置，防止对正常执行的不必要干扰）
- 使用 **FACTR** 力反馈遥操作系统采集 20 条人类干预 episode 作为训练数据
- 50% 干预 + 50% 正常执行的平衡采样，保留基础策略能力的同时学习修正

---

## 关键公式

### 公式 1：[[Force Feature Encoding|接触特征编码]]

$$
\mathbf{c}_t = \text{Enc}_f(\mathbf{w}_{t-H_f+1:t})
$$

**含义**: 用轻量 MLP 将历史 $H_f$ 步力/力矩信号编码为紧凑的接触特征向量

**符号说明**:
- $\mathbf{c}_t \in \mathbb{R}^d$: 时刻 $t$ 的接触特征
- $\mathbf{w}_t \in \mathbb{R}^6$: 腕部 6 轴力/力矩测量值（三维力 + 三维力矩）
- $\text{Enc}_f$: 轻量级 MLP 编码器
- $H_f$: 力历史窗口长度

### 公式 2：[[AdaLN|力条件化 AdaLN 参数调制]]

$$
\bar{\mathbf{u}}_t^{(i)} = \mathbf{u}_t^{(i)} + \text{MLP}_f^{(i)}(\mathbf{c}_t)
$$

**含义**: 将力特征通过残差方式调制每个 Transformer block 的 AdaLN 归一化参数，实现力对动作生成的深层次影响

**符号说明**:
- $\bar{\mathbf{u}}_t^{(i)}$: 第 $i$ 个 block 调制后的 AdaLN 参数（包含 shift、scale、gate）
- $\mathbf{u}_t^{(i)}$: 原始 AdaLN 参数（由视觉/语言条件决定）
- $\text{MLP}_f^{(i)}$: 第 $i$ 个 block 专属的力条件化 MLP

### 公式 3：[[Rectified Flow|Stage 1 视频生成损失]]

$$
\mathcal{L}_{stage1} = \mathbb{E}\left[\left\|v_\psi^O(\hat{O}_t^\alpha, \alpha, L_t, I_{t-k:t}) - (O_t - \epsilon^O)\right\|_2^2\right]
$$

**含义**: 通过 [[Rectified Flow]] 框架训练视频生成模型，预测从噪声到干净帧的速度场

**符号说明**:
- $v_\psi^O$: 视频生成模型（velocity field）
- $\hat{O}_t^\alpha = \alpha \cdot O_t + (1-\alpha) \cdot \epsilon^O$: 插值后的视觉潜码（$\alpha \in [0,1]$）
- $L_t$: 语言指令
- $I_{t-k:t}$: 历史视觉潜码
- $\epsilon^O$: 视觉噪声

### 公式 4：[[Flow Matching|Stage 2 动作与力联合预测损失]]

$$
\mathcal{L}_{action} = \mathbb{E}\left[\left\|\hat{v}_t^A - (A_t - \epsilon^A)\right\|_2^2\right]
$$

$$
\mathcal{L}_{force} = \mathbb{E}\left[\left\|\hat{v}_t^W - (W_t - \epsilon^W)\right\|_2^2\right]
$$

**含义**: 同时监督动作轨迹速度场和力轨迹速度场的预测，使模型联合学习动作与接触演化

**符号说明**:
- $\hat{v}_t^A$: 预测的动作速度场
- $\hat{v}_t^W$: 预测的力/力矩速度场
- $A_t$: 动作轨迹目标
- $W_t$: 力轨迹目标
- $\epsilon^A, \epsilon^W$: 对应噪声项

### 公式 5：[[Wrench Tracking Error|力跟踪误差]]

$$
\mathbf{e}_{t-H_e+1:t}^w = \mathbf{w}_{t-H_e+1:t}^{meas} - \mathbf{w}_{t-H_e+1:t}^{pred}
$$

**含义**: 计算历史窗口内实测力与预测力的偏差，作为是否需要残差修正的关键信号

**符号说明**:
- $\mathbf{e}^w$: 力跟踪误差序列
- $\mathbf{w}^{meas}$: 实时测量的力/力矩值
- $\mathbf{w}^{pred}$: 基础策略预测的力轨迹
- $H_e$: 误差历史窗口长度

### 公式 6：[[Residual Policy|残差修正器输出]]

$$
\Delta\mathbf{a}_{t+1:t+H_r}^{res},\; g_t^{res} = \text{Res}_\phi\left(\mathbf{e}_{t-H_e+1:t}^w,\; \mathbf{c}_t,\; \mathbf{a}_{t+1:t+H_p}^{base},\; \mathbf{w}_{t+1:t+H_p}^{pred},\; \mathbf{q}_t\right)
$$

**含义**: 三层 MLP 根据力跟踪误差、接触特征、基础动作等输入，预测残差修正量和介入门控信号

**符号说明**:
- $\Delta\mathbf{a}^{res}$: 残差动作修正（叠加到基础动作上）
- $g_t^{res} \in [0,1]$: 残差门控值（标量，通过 sigmoid 输出）
- $\text{Res}_\phi$: 三层 MLP 残差修正器
- $\mathbf{q}_t$: 本体感知状态（关节角度等）
- $H_r$: 残差修正的动作块长度

### 公式 7：[[Closed-Loop Control|修正后的动作]]

$$
\mathbf{a}_{t+1:t+H_r}^{cor} = \mathbf{a}_{t+1:t+H_r}^{base} + \kappa_t \cdot \Delta\mathbf{a}_{t+1:t+H_r}^{res}
$$

**含义**: 通过二值门控决定是否应用残差修正，仅在检测到明显力偏差时才修正基础动作

**符号说明**:
- $\mathbf{a}^{cor}$: 修正后的动作序列
- $\mathbf{a}^{base}$: 基础策略输出的动作块
- $\kappa_t \in \{0, 1\}$: 二值激活掩码，$\kappa_t = 1$ 当 $g_t^{res} > \tau$（$\tau = 0.8$）

### 公式 8：[[Residual Policy|残差修正器训练损失]]

$$
\mathcal{L}_{res} = \mathbb{E}\left[\left\|\Delta\mathbf{a}^{res} - \Delta\mathbf{a}^{gt}\right\|_2^2 + \lambda_{gate} \cdot \text{BCE}(g_t^{res},\; y_t)\right]
$$

**含义**: 联合优化残差动作精度（MSE）和介入时机判断（BCE），使修正器既能给出正确的修正量，又能在合适时机触发

**符号说明**:
- $\Delta\mathbf{a}^{gt}$: 人类干预提供的 ground-truth 残差修正
- $y_t \in \{0, 1\}$: 是否发生干预的标签
- $\lambda_{gate}$: 门控损失权重
- $\text{BCE}$: 二元交叉熵损失

---

## 关键图表

### Figure 1：动机对比图

![Figure 1](https://arxiv.org/html/2606.08555v1/x1.png)

**说明**: 对比"仅力观测条件化"（左）与 FAWAM 三层次力集成（右）在执行中途出现接触偏差时的表现。左侧方法无法纠正接触偏差导致失败，右侧 FAWAM 通过力引导残差修正恢复正确接触，任务成功。

### Figure 2：FAWAM 整体框架

![Figure 2](https://arxiv.org/html/2606.08555v1/x2.png)

**说明**: FAWAM 的完整架构。(a) [[Force-Envisioned Action Model]] 将视觉、语言、力信号通过 [[AdaLN]] 融合，联合输出动作块和未来力轨迹；(b) [[Force-Guided Residual Corrector]] 实时对比测量力与预测力，在检测到偏差时输出残差修正动作。

### Figure 3：接触密集型评测任务

![Figure 3](https://arxiv.org/html/2606.08555v1/x3.png)

**说明**: 四个评测任务——擦白板（Erase Board，倾斜角变化）、削黄瓜（Peel Cucumber，高度变化）、旋转箱体（Pivot Box，位置变化）、擦花瓶（Wipe Vase，曲面形变）。每个任务通过控制物理参数变化来测试力控制鲁棒性。

### Figure 5：扰动恢复对比

![Figure 5](https://arxiv.org/html/2606.08555v1/x4.png)

**说明**: 在执行过程中施加物理扰动（倾斜白板、改变高度等）时的轨迹对比。无残差修正的模型（上）无法恢复稳定接触；FAWAM（下）检测到力偏差后触发残差修正，成功恢复接触并完成任务。

### Figure 6：实验环境

![Figure 6](https://arxiv.org/html/2606.08555v1/fig/experimental_setup.png)

**说明**: 实验平台配置——[[Franka Research 3]] 机械臂执行任务，[[FACTR]] 力反馈遥操系统（Leader 臂）采集人类干预数据，三台 [[Intel RealSense D435]] 相机提供多视角视觉观测，ATI Axia80-M8 力/力矩传感器安装于腕部。

### Figure 15：残差激活阈值 τ 的敏感性分析

![Figure 15](https://arxiv.org/html/2606.08555v1/fig/tau_success_rate.png)

**说明**: 不同阈值 $\tau$ 下各任务的成功率。$\tau = 0.8$ 在所有任务中取得最佳平均性能，过低的阈值导致频繁不必要的修正干扰正常执行，过高则在关键时刻错过修正时机。

### Table 1：主要实验结果对比

| 方法 | Erase Board | Peel Cucumber | Pivot Box | Wipe Vase | 平均成功率 |
|------|------------|---------------|-----------|-----------|-----------|
| [[Pi05]] | 9/20 | 8/20 | 10/20 | 11/20 | 47.50% |
| GE-Act | 9/20 | 10/20 | 9/20 | 11/20 | 48.75% |
| [[ForceVLA]] | 11/20 | 14/20 | 13/20 | 13/20 | 63.75% |
| TA-VLA | 10/20 | 17/20 | 14/20 | 10/20 | 63.75% |
| FAWAM w/o Res | 14/20 | 15/20 | 16/20 | 14/20 | 73.75% |
| **FAWAM** | **15/20** | **19/20** | **17/20** | **17/20** | **85.00%** |

**关键发现**: FAWAM 比视觉-only 基线高 36.25%，比现有最优力感知方法高 21.25%；残差修正模块单独贡献约 11.25% 的提升（73.75% → 85%）。

### Table 2：消融实验结果

| 配置 | 平均成功率 | 说明 |
|------|-----------|------|
| 基础（视觉-only）| ~48.75% | 无任何力集成 |
| +力观测编码 | ~56.25% | 仅感知层：+7.5% |
| +力预测（联合） | ~73.75% | 感知+预测层：+17.5% |
| +残差修正 | **85.00%** | 三层完整集成：+11.25% |

**关键发现**: 三个力集成层次相互补充，每个层次均有显著贡献，联合使用效果最优；力预测（+17.5%）的贡献大于纯力观测（+7.5%），说明未来力建模比历史力感知更关键。

### Table 3：扰动鲁棒性测试

| 任务 | 无残差修正（/5） | 有残差修正（/5） |
|------|----------------|----------------|
| Erase Board | 0-2 | 3-5 |
| Peel Cucumber | 0-2 | 3-5 |
| Pivot Box | 0-2 | 3-5 |
| Wipe Vase | 0-2 | 3-5 |

**关键发现**: 在执行中施加物理扰动的情况下，残差修正模块使所有任务的恢复成功率从接近 0 提升至 60-100%，证明了闭环力反馈对于鲁棒性的决定性作用。

---

## 实验

### 数据集

| 数据 | 规模 | 特点 | 用途 |
|------|------|------|------|
| 操作演示 | 90 条 | 四种接触密集型任务，含视觉+力信号同步 | 基础策略训练（Stage 1+2） |
| 人类干预 | 20 条 | [[FACTR]] 力反馈遥操采集，含干预/非干预样本 | 残差修正器训练 |
| 测试 | 20 次/任务 | 含受控物理扰动变化 | 评估 |

### 实现细节

- **硬件平台**: [[Franka Research 3]] 机械臂 + ATI Axia80-M8 六轴力/力矩传感器（腕部）
- **视觉传感**: 三台 [[Intel RealSense D435]] 相机，30 Hz 视频流
- **力采样**: 120 Hz，经 10 Hz 截止频率二阶 Butterworth 低通滤波
- **执行频率**: 基础策略 5 Hz，残差修正器 10 Hz
- **Stage 1 训练**: 8× NVIDIA A100，30,000 步，学习率 $3\times10^{-5}$
- **Stage 2 训练**: 8× A100，30,000 步，学习率 $1.5\times10^{-4}$
- **残差修正器**: 500 epochs，平衡正负样本（50% 干预 + 50% 正常）
- **动作表示**: [[Action Chunking]]，块长度 $H_p$，末端执行器 pose + 夹爪开合

### 可视化结果

四种任务的力引导修正均显示出一致规律：当末端执行器接触条件改变（白板倾斜、台面高度变化、箱体位置偏移）时，测量力与预测力出现误差峰值，触发残差修正，机器人在 1-2 个控制周期内恢复正确接触，并维持稳定执行直至任务完成。

---

## 批判性思考

### 优点

1. **系统性三层次力集成**: 区别于"力作为附加观测"的浅层处理，在感知、预测、执行三个维度均引入力信号，形成完整的力感知控制闭环
2. **数据效率高**: 仅用 90 条演示（基础策略）+ 20 条干预（残差修正器）即取得 85% 成功率，训练数据规模合理
3. **模块化设计**: 残差修正器独立于基础策略，可轻松叠加到其他 WAM/VLA 模型上；零初始化设计确保训练稳定

### 局限性

1. **力信号未进入视频生成阶段**: 作者承认 Stage 1 视频模型未融合力信号，原因是缺乏大规模同步力/视频数据集。这一缺失限制了视频模型对接触演化的长程预测能力
2. **干预数据多样性有限**: 仅 20 条人类干预 episode 可能无法覆盖所有失败模式，在分布外扰动场景下修正效果可能下降
3. **仅末端腕部力信息**: 6 轴力/力矩传感器提供的是末端合力，无法感知分布式接触；密集触觉传感（如 GelSight）可提供更精细的接触位置与压力分布信息
4. **任务场景较为单一**: 四个任务均在桌面固定环境中测试，未验证在移动底座、双臂协同等更复杂场景下的泛化能力

### 潜在改进方向

1. 构建大规模同步力/视频数据集，将力信号引入 Stage 1 视频生成，实现接触感知的视觉预测
2. 结合密集触觉传感（如 [[GelSight]]）提供接触分布信息，替代或补充腕部力/力矩传感器
3. 探索在线学习机制，使残差修正器在部署阶段持续适应新的扰动模式

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文中有详细超参数和数据配置）
- [ ] 数据集可获取（演示数据和干预数据均需自行采集）

---

## 关联笔记

### 基于

- [[Pi05]]: 基础 World Action Model 框架，FAWAM 在其视频生成架构上添加力感知模块
- [[FACTR]]: 力反馈遥操系统，用于采集 FAWAM 残差修正器的训练数据

### 对比

- [[ForceVLA]]: 将力作为附加观测的代表方法，FAWAM 比其高 21.25%
- [[Pi05]]: 视觉-only VLA 基线，FAWAM 比其高 36.25%

### 方法相关

- [[AdaLN]]: 力条件化调制的核心组件
- [[Rectified Flow]]: Stage 1/2 训练的流匹配框架
- [[Action Chunking]]: 动作块预测策略
- [[Residual Policy]]: 残差修正器的设计范式
- [[Closed-Loop Control]]: 力引导在线修正的控制论基础
- [[World Action Model]]: FAWAM 所属的模型类别

### 硬件/数据相关

- [[Franka Research 3]]: 实验平台机械臂
- [[Intel RealSense D435]]: 多视角视觉传感器
- [[FACTR]]: 力反馈遥操系统（同时也作为对比基线）

---

## 速查卡片

> [!summary] FAWAM: Force-Aware World Action Models
> - **核心**: 在感知、预测、执行三层次系统性融合力/力矩信号，实现接触密集型操作的闭环力控制
> - **方法**: 力条件化 AdaLN（感知）+ 联合力/动作预测（预测）+ 力误差触发残差修正（执行）
> - **结果**: 85% 平均成功率，+36.25% vs 视觉-only，+21.25% vs 现有力感知方法；扰动恢复成功率接近 100%
> - **代码**: 暂未开放

---

*笔记创建时间: 2026-06-10*

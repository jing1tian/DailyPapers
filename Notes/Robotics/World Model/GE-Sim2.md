---
title: "GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation"
method_name: "GE-Sim2"
authors: [Boxiang Qiu, Liliang Chen, Yue Liao, Nan Wang, Lintao Wang, Jiayi Luo, Wenzhi Zhao, Shengcong Chen, Di Chen, Ye Li, Chen Gao, Shuicheng Yan, Si Liu, Maoqing Yao, Guanghui Ren]
year: 2026
venue: arXiv
tags: [video-world-model, robotic-manipulation, closed-loop-simulation, proprioceptive-state, reward-model]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.27491
created: 2026-05-29
---

# 论文笔记：GE-Sim 2.0

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beihang University, Sea AI Lab, Meituan |
| 日期 | May 2026 |
| 项目主页 | [ge-sim-v2.github.io](https://ge-sim-v2.github.io/) |
| 对比基线 | [[Ctrl-World]], [[DreamDojo]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.27491) |

---

## 一句话总结

> GE-Sim 2.0 将 [[Action-Conditioned World Model|动作条件视频生成]] 扩展为完整闭环仿真器，通过增加本体感知状态预测模块和专用奖励判别器，使机器人策略可在虚拟视频世界中完整闭环评估并提升真实成功率。

---

## 核心贡献

1. **Vision Expert（视觉专家）**: 基于 2B 参数 [[Diffusion Transformer|扩散变换器]] 的多视角动作条件视频生成，支持长时程稀疏记忆，在 WorldArena 排行榜上取得最优综合分数。
2. **Proprioceptive State Expert（本体感知状态专家）**: 与视觉专家并行的轻量变换器分支，从视频特征解码 16 维关节状态，使策略可直接在世界模型输出上闭环运行。
3. **World Judge（世界裁判）**: 专门训练的视觉语言奖励模型，逐帧预测任务成功信号，在真实机器人对齐上远超通用 VLM 基线，支持奖励过滤行为克隆提升策略性能。
4. **加速框架**: 结合 [[Distribution Matching Distillation|DMD2]] 步骤蒸馏和随机步幅训练，在单张 H100 上实现 25 帧 2.3 秒生成，达到拟实时推理吞吐量。

---

## 问题背景

### 要解决的问题

现有的机器人[[生成世界模型|视频世界模型]]（如 GE-Sim 1.0、Ctrl-World）仅能生成视觉帧，但策略与世界模型的闭环交互还需要**本体感知状态**（关节角度、夹爪开合）和**任务成功奖励信号**，缺少这两者使得世界模型无法作为真正的闭环仿真器用于策略评估和数据生成。

### 现有方法的局限

- **仅视图仿真**：Ctrl-World、DreamDojo 等只输出视频，策略无法在其中闭环运行（没有状态反馈）。
- **缺乏机器可验证奖励**：人工评估成本高，通用 VLM（如 Qwen）在机器人成功分类上准确率只有 60%，严重限制大规模自动化评估。
- **推理速度慢**：多步扩散导致生成延迟高，无法支持大规模闭环 rollout。

### 本文的动机

打造"omni 仿真器"——在视觉保真度之上叠加**状态**和**奖励**两个维度，实现从"看-only"到"感-评-学"的闭环升级，最终通过合成数据+奖励过滤的[[行为克隆]]提升真实机器人成功率。

---

## 方法详解

### 模型架构

GE-Sim 2.0 由三个并行/串联模块构成：

- **输入**: 初始观测帧 $\mathbf{x}_0$、稀疏历史记忆 $\mathbf{m}_{0:t-1}$、任务指令 $\mathcal{T}(q)$、动作子轨迹 $\mathbf{A}_t$
- **Vision Expert Backbone**: [[Diffusion Transformer|Cosmos-Predict2-2B DiT]]，16 通道视频 latent
- **条件编码**: [[RoPE]] 3D 旋转位置编码 + 可学习视角 embedding + 动作射线图 + EE 位姿图
- **并行分支**: Proprioceptive State Expert（轻量 Transformer）从 Vision Expert 各层特征聚合预测关节状态
- **串联模块**: World Judge（冻结视觉编码器 + 可训练语言模型 + 成功预测头）
- **总参数**: Vision Expert ~2B；State Expert 轻量级（维度远小于 Vision Expert）

### 核心模块

#### 模块1：Vision Expert（视觉专家）

**设计动机**: 利用大规模预训练 [[Diffusion Transformer]] 的视频生成能力，配合动作条件信号，生成多视角机器人操作视频。

**具体实现**:
- 使用 [[RoPE]] 对每视角 token 注入 3D 旋转位置信息和可学习视角 embedding
- 动作 $\mathbf{a}_i \in \mathbb{R}^{14}$ 编码为两种互补表示：**射线图**（6 通道，编码摄像机几何和视角变化）和 **EE 位姿图**（3 通道，End-Effector 位置、朝向、夹爪开合的图像空间渲染）
- 通过**深度感知圆形渲染**绘制 EE：距离越远圆圈越小，增强深度提示
- 带噪声 latent、射线图、位姿图、条件掩码沿通道拼接为 $\mathbf{z}_{cond}$，送入 [[DiT]] 主干
- 稀疏记忆帧通过**记忆帧增强**（渐进式噪声混合 + 局部高斯模糊 + 多视角同步颜色抖动）解决闭环自生成的分布偏移问题

#### 模块2：Proprioceptive State Expert（本体感知状态专家）

**设计动机**: 策略需要关节状态反馈才能闭环运行；通过从 Vision Expert 的视频 latent 中解码状态，避免额外传感器依赖。

**具体实现**:
- 从 Vision Expert 所有 $L$ 个 Transformer 层输出 $\mathbf{h}_l^{video}$ 中，用可学习标量权重 $\alpha_l$ 加权聚合得到融合特征 $\mathbf{H}_{fuse}$
- 轻量 Transformer 分支通过**交叉注意力**消费 $\mathbf{H}_{fuse}$，预测 16 维关节状态 $\mathbf{s}_t = [\boldsymbol{\theta}_t^L, g_t^L, \boldsymbol{\theta}_t^R, g_t^R]$（双臂各 7 关节角 + 夹爪开合）
- **历史轨迹重采样**增强：通过上采样+下采样注入低通失真，模拟异步闭环误差累积
- 使用与 Vision Expert 相同的 [[Flow Matching|flow-matching]] 训练目标

#### 模块3：World Judge（世界裁判）

**设计动机**: 通用 VLM 对机器人操作成功的判断准确率仅 60%，需要领域专用奖励模型支持自动化大规模评估。

**具体实现**:
- 冻结视觉主干编码器，仅训练语言模型和 MLP 预测头
- 帧级二值成功标签：$y_i = \mathbb{1}[i \geq i_{succ}]$（成功帧之后标 1）
- 类平衡二值交叉熵损失，通过帧级权重 $w_i$ 补偿成功/失败帧数量不均衡
- 聚焦二值成功分类（而非进度预测），与 [[VLM Critic]] 思路一致但更专业化

---

## 关键公式

### 公式1：[[生成世界模型|自回归块式视频生成]]

$$
\mathbf{x}_{1:N}^t = \mathcal{W}(\mathbf{x}_0, \mathbf{m}_{0:t-1}, \mathcal{T}(q))
$$

**含义**: 世界模型基于初始帧、稀疏记忆帧、任务指令，自回归生成下一块多视角视频帧。

**符号说明**:
- $\mathbf{x}_{1:N}^t$: 第 $t$ 时刻预测的 $N$ 帧多视角图像
- $\mathbf{x}_0$: 初始观测帧
- $\mathbf{m}_{0:t-1}$: 历史稀疏记忆帧
- $\mathcal{T}(q)$: 任务指令的文本编码

### 公式2：[[RoPE|多视角 RoPE 位置编码]]

$$
\tilde{v}_i = \text{RoPE}_{(t,h,w)} + v_i + e^{view}_i
$$

**含义**: 为每个视角 token 叠加 3D 时空旋转位置编码和可学习视角嵌入，使模型感知不同摄像机的空间关系。

**符号说明**:
- $v_i$: 第 $i$ 个视角的视频 token
- $\text{RoPE}_{(t,h,w)}$: 时间、高度、宽度三维旋转位置编码
- $e^{view}_i$: 可学习视角嵌入（区分头部/左腕/右腕相机）

### 公式3：[[Flow Matching|视频流匹配训练目标]]

$$
\mathcal{L}_{video} = w(\tau) \| (v_\theta - (\epsilon - l)) \odot (1 - M) \|_2^2
$$

**含义**: 仅在未来帧（掩码 $M$ 为 0 处）监督去噪速度预测，避免对历史条件帧的损失干扰。

**符号说明**:
- $v_\theta$: 模型预测的去噪速度
- $\epsilon$: 噪声
- $l$: 视频 latent
- $M$: 条件掩码（历史帧处为 1）
- $w(\tau)$: 扩散时间步权重

### 公式4：动作表示

$$
\mathbf{a}_i = [x, y, z, r, p, y, o]^L \oplus [x, y, z, r, p, y, o]^R \in \mathbb{R}^{14}
$$

**含义**: 14 维向量编码双臂末端执行器的位置（xyz）、朝向（roll/pitch/yaw）和夹爪开合状态。

**符号说明**:
- $L, R$: 左臂、右臂
- $x, y, z$: 末端执行器位置
- $r, p, y$: roll、pitch、yaw 朝向
- $o$: 夹爪开合度

### 公式5：[[Action-Conditioned World Model|动作条件 Latent 融合]]

$$
v_i = [\tilde{z}_i \| \text{down}(P_i) \| \text{down}(R_i)]
$$

**含义**: 将带噪声视频 latent、下采样位姿图和射线图沿通道维度拼接，作为 DiT 的联合输入。

**符号说明**:
- $\tilde{z}_i$: 带噪声视频 latent（16 通道）
- $P_i$: EE 位姿图（3 通道）
- $R_i$: 射线图（6 通道）
- $\text{down}(\cdot)$: 下采样到 latent 空间分辨率

### 公式6：[[Diffusion Transformer|深度感知 EE 圆形渲染]]

$$
r = \text{clamp}\!\left(1 - \frac{\|\mathbf{x}_{EE} - \mathbf{x}_{cam}\| - d_{min}}{d_{max} - d_{min}},\, 0,\, 1\right) \cdot r_{max}
$$

**含义**: 将末端执行器渲染为半径随深度减小的圆，距摄像机越远圆越小，提供深度感知线索。

**符号说明**:
- $\mathbf{x}_{EE}$: 末端执行器 3D 位置
- $\mathbf{x}_{cam}$: 摄像机 3D 位置
- $d_{min}, d_{max}$: 深度范围归一化参数
- $r_{max}$: 最大圆半径

### 公式7：[[Proprioceptive State|本体感知状态表示]]

$$
\mathbf{s}_t = [\boldsymbol{\theta}_t^L,\, g_t^L,\, \boldsymbol{\theta}_t^R,\, g_t^R] \in \mathbb{R}^{16}
$$

**含义**: 16 维向量完整描述双臂机器人的关节空间状态。

**符号说明**:
- $\boldsymbol{\theta}_t^L, \boldsymbol{\theta}_t^R$: 左/右臂各 7 维关节角度
- $g_t^L, g_t^R$: 左/右夹爪开合度

### 公式8：[[Proprioceptive State|视觉特征加权聚合]]

$$
\mathbf{H}_{fuse} = \text{LayerNorm}\!\left(\sum_{l=1}^{L} \alpha_l \mathbf{h}_l^{video}\right), \quad \alpha_l \in \mathbb{R}
$$

**含义**: 通过可学习标量权重对 Vision Expert 所有 Transformer 层输出加权求和，聚合多尺度视觉特征供状态专家的交叉注意力消费。

**符号说明**:
- $\mathbf{h}_l^{video}$: Vision Expert 第 $l$ 层输出
- $\alpha_l$: 可学习层权重
- $L$: Vision Expert 总层数

### 公式9：状态专家 [[Flow Matching]] 损失

$$
\mathcal{L}_{proprio} = \mathbb{E}_{\mathbf{s}_0, \epsilon, \tau}\, \| v_\phi(\mathbf{s}_\tau, \tau, \mathbf{H}_{fuse}) - (\epsilon - \mathbf{s}_0) \|_2^2
$$

**含义**: 对关节状态应用 flow-matching 目标，引导状态专家从噪声中去噪还原真实关节状态。

**符号说明**:
- $\mathbf{s}_\tau$: 时间步 $\tau$ 的噪声状态
- $v_\phi$: 状态专家预测的去噪速度
- $\mathbf{H}_{fuse}$: 融合视觉特征（条件信号）

### 公式10：[[行为克隆|历史轨迹重采样增强]]

$$
\mathbf{s}_{hist} \leftarrow \text{Upsample}(\text{Downsample}(\mathbf{s}_{hist},\, n_{prev}-1),\, n_{prev})
$$

**含义**: 对历史状态序列先降采样再升采样注入低通失真，模拟闭环执行时的异步误差累积。

**符号说明**:
- $\mathbf{s}_{hist}$: 历史关节状态序列
- $n_{prev}$: 历史帧数

### 公式11：[[VLM Critic|成功帧标签]]

$$
y_i = \mathbb{1}[i \geq i_{succ}]
$$

**含义**: 任务成功帧及其之后的所有帧标记为 1，之前标记为 0，生成用于 World Judge 训练的帧级二值标签。

**符号说明**:
- $i$: 帧索引
- $i_{succ}$: 任务成功发生的帧索引

### 公式12：[[VLM Critic|类平衡交叉熵损失]]

$$
\mathcal{L}_{judge} = \frac{\sum_i m_i w_i \text{BCE}(\sigma(\hat{s}_i), y_i)}{\sum_i m_i w_i}
$$

**含义**: 通过帧级权重 $w_i$ 补偿成功/失败帧数量不均衡，聚焦模型学习关键状态转变时刻。

**符号说明**:
- $m_i$: 帧有效掩码
- $w_i$: 类别平衡权重（成功/失败帧各自归一化）
- $\hat{s}_i$: 模型预测的成功 logit
- $\sigma(\cdot)$: Sigmoid 函数

### 公式13：[[Distribution Matching Distillation|DMD2 分布匹配梯度]]

$$
\nabla_{DMD} = \frac{x_0^{fake} - x_0^{teacher}}{\|x_0^{student} - x_0^{teacher}\|}
$$

**含义**: 归一化梯度将学生模型（4步推理）向教师模型分布靠拢，实现多步扩散蒸馏为少步快速推理。

**符号说明**:
- $x_0^{fake}$: 学生模型生成样本
- $x_0^{teacher}$: 教师模型生成样本
- 归一化防止梯度尺度不稳定

---

## 关键图表

### Figure 1：GE-Sim 2.0 系统概览

![Figure 1](https://arxiv.org/html/2605.27491v1/x1.png)

**说明**: GE-Sim 2.0 闭环仿真器全景图。输入真实世界初始观测和策略输出动作，通过 [[Action-Conditioned World Model]] 生成视觉帧和关节状态，由 [[VLM Critic|World Judge]] 评分，支持策略闭环评估和奖励过滤行为克隆。训练数据涵盖遥操作、接触丰富交互和在线策略部署 rollout。

### Figure 2：Vision Expert 与 State Expert 架构

![Figure 2](https://arxiv.org/html/2605.27491v1/x2.png)

**说明**: Vision Expert 处理历史帧和动作条件，生成未来视觉状态；Proprioceptive State Expert 并行运行，通过 [[Cross-Attention|交叉注意力]] 从 Vision Expert 各层特征解码关节角度和夹爪状态，实现视觉与本体感知的联合预测。

### Figure 3：World Judge 架构

![Figure 3](https://arxiv.org/html/2605.27491v1/x3.png)

**说明**: World Judge 将视觉编码器冻结，逐帧处理视频帧和任务文本指令，通过轻量 MLP 预测头输出逐帧成功概率。专门训练使其在机器人操作域准确率达 79%（WM rollout），远超通用 VLM 的 60%。

### Figure 4：Pull out plug 任务定性对比

![Figure 4](https://arxiv.org/html/2605.27491v1/x4.png)

**说明**: GE-Sim 2.0 与 GT、Ctrl-World、DreamDojo 对比。GE-Sim 2.0 在长时程操作中保持更高视觉保真度，Ctrl-World 在末端出现明显图像退化，DreamDojo 出现物体穿透等物理不一致问题。

### Figure 5：Pour water 任务定性对比

![Figure 5](https://arxiv.org/html/2605.27491v1/x5.png)

**说明**: 倒水任务中，GE-Sim 2.0 正确渲染液体流动轨迹，而 Ctrl-World 在细粒度接触状态（液位变化）上存在明显错误，体现了精细物理细节建模的挑战。

### Figure 6：不同时间段的回放视觉质量

![Figure 6](https://arxiv.org/html/2605.27491v1/x6.png)

**说明**: 将每个回放视频按 10 秒分为 5 段，分别评估 PSNR。GE-Sim 2.0 随时间推移的 PSNR 下降幅度最小，展示了更强的长时程视觉稳定性；竞争方法在时间靠后的片段中 PSNR 急剧下降。

### Figure 7：WorldArena 排行榜

![Figure 7](https://arxiv.org/html/2605.27491v1/x7.png)

**说明**: GE-Sim 2.0（仅 2B 参数）在 WorldArena benchmark 上取得最优综合分数，超越 Ctrl-World、DreamDojo 等专用机器人世界模型和闭源通用视频生成器。

### Figure 8：世界模型与真实机器人成功率对齐

![Figure 8](https://arxiv.org/html/2605.27491v1/x8.png)

**说明**: 在 6 个长时程任务上对比闭环世界模型预测成功率与真实机器人成功率。GE-Sim 2.0 的预测值与真实值的相关性远高于 Ctrl-World，任务级偏差在 1 个百分点以内。

### Figure 9：WM 过滤行为克隆的策略提升

![Figure 9](https://arxiv.org/html/2605.27491v1/x9.png)

**说明**: 将合成轨迹（由 World Judge 过滤）加入 π0.5 策略训练后，三个任务平均成功率从 41.7% 提升至 56.7%（+15 pp），验证了 GE-Sim 2.0 作为数据增强源的实用价值。

### Figure 10：闭环协议混淆矩阵

![Figure 10](https://arxiv.org/html/2605.27491v1/x10.png)

**说明**: 二值成功/失败分类的混淆矩阵。GE-Sim 2.0 的情节级准确率为 0.81（vs Ctrl-World 0.63），召回率为 0.82（vs Ctrl-World 0.25），大幅减少真实成功被误判为失败的漏检。

### Figures 11-16：多视角定性结果

| Figure | 任务 | 视角 |
|--------|------|------|
| Fig 11 | Command grasp & release | Multi-View |
| Fig 12 | Clean mirror stains | Multi-View |
| Fig 13 | Borrow flame | Multi-View |
| Fig 14 | Fold towels | Multi-View |
| Fig 15 | Pull out plug | Multi-View |
| Fig 16 | Pour water | Head-View |

![Figure 11](https://arxiv.org/html/2605.27491v1/x11.png)
![Figure 12](https://arxiv.org/html/2605.27491v1/x12.png)
![Figure 13](https://arxiv.org/html/2605.27491v1/x13.png)
![Figure 14](https://arxiv.org/html/2605.27491v1/x14.png)
![Figure 15](https://arxiv.org/html/2605.27491v1/x15.png)
![Figure 16](https://arxiv.org/html/2605.27491v1/x16.png)

### Figures 17-21：更多定性结果

![Figure 17](https://arxiv.org/html/2605.27491v1/x17.png)
![Figure 18](https://arxiv.org/html/2605.27491v1/x18.png)
![Figure 19](https://arxiv.org/html/2605.27491v1/x19.png)
![Figure 20](https://arxiv.org/html/2605.27491v1/x20.png)
![Figure 21](https://arxiv.org/html/2605.27491v1/x21.png)

---

### Table 1：机器人视频世界仿真器对比

| 能力 | IRASim | 1XWM | GE-Sim 1.0 | Ctrl-World | DreamDojo | Interactive WM | ABot-PhysWorld | WorldScape | MotuBrain | **GE-Sim 2.0** |
|------|--------|------|-----------|-----------|-----------|----------------|----------------|-----------|-----------|-----------|
| 长时程 | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| 多视角 | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | **✓** |
| 本体感知状态 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 拟实时 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | **✓** |
| 奖励信号 | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 开源 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | **✓** |

**说明**: GE-Sim 2.0 是唯一同时满足长时程、多视角、本体感知状态预测、拟实时、奖励信号、开源六项能力的系统。

### Table 2：回放视觉质量对比

| 方法 | Head-PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | MV-PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ |
|------|-----------|-------|--------|------|------|-----------|-------|--------|------|------|
| Ctrl-World | 19.09 | 0.786 | 0.296 | 62.70 | 1083.7 | 16.65 | 0.735 | 0.407 | 56.85 | 1527.5 |
| DreamDojo | 17.38 | 0.728 | 0.353 | 79.82 | 1155.0 | — | — | — | — | — |
| **GE-Sim 2.0** | **23.05** | **0.846** | **0.145** | **32.28** | **481.3** | **20.80** | **0.796** | **0.217** | **24.92** | **613.5** |

**关键发现**: 多视角 FID 24.92 vs Ctrl-World 56.85（降低 56%），FVD 613.5 vs 1527.5（降低 60%），全面领先。

### Table 3：World Judge 奖励质量

| 任务 | Ours (WM) acc↑ | dist↓ | Ours (GT) acc↑ | dist↓ | Qwen (WM) acc↑ | dist↓ | Qwen (GT) acc↑ | dist↓ |
|------|---------------|-------|---------------|-------|---------------|-------|---------------|-------|
| Borrow flame | .96 | 37.0 | .95 | 24.4 | .71 | 96.5 | .70 | 115.3 |
| Clean mirror stains | .82 | 9.3 | .90 | 12.4 | .50 | — | .50 | — |
| Command grasp & release | .72 | 7.8 | .80 | 4.0 | .81 | 38.7 | .85 | 43.4 |
| Fold towels | .65 | 13.0 | .60 | 9.5 | .63 | 38.0 | .50 | 30.5 |
| Pour water | .90 | 61.4 | 1.00 | 26.2 | .55 | 22.0 | .35 | 13.0 |
| Pull out plug | .80 | 24.2 | .95 | 7.5 | .55 | 34.0 | .60 | 17.0 |
| **Average** | **.79** | **28.2** | **.87** | **15.7** | .60 | 57.8 | .58 | 64.7 |

**关键发现**: World Judge 在世界模型 rollout 上准确率 79%（vs Qwen 60%），事件距离 28.2 帧（vs Qwen 57.8 帧），专业化训练带来显著提升。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人操作数据 | 数千小时 | 遥操作 + 接触丰富交互 + 在线策略 rollout，含成功和失败轨迹 | Vision Expert 训练 |
| WorldArena benchmark | — | 多任务机器人操作视频质量评估 | 视觉质量评估 |
| 6 个操作任务 | — | Pour water / Fold towels / Pull out plug / Borrow flame / Command grasp & release / Clean mirror stains | 闭环评估 + 策略提升 |

### 实现细节

- **Vision Expert Backbone**: [[Diffusion Transformer|Cosmos-Predict2-2B-Video2World DiT]]
- **动作条件**: 6 通道射线图 + 3 通道 EE 位姿图，经 [[AdaLN]] 注入时间步信息
- **蒸馏策略**: [[Distribution Matching Distillation|DMD2]]，目标 4 步推理（训练时随机 1-4 步）
- **随机步幅训练**: 训练时随机采样帧间距，推理时支持最高 4× 跳帧
- **推理速度**: 单张 H100，25 帧生成 2.3 秒
- **记忆帧增强**: 激活概率 0.8，含渐进噪声（$\sigma=0.2$/0.5）、局部高斯模糊（覆盖约 20% 区域）、多视角同步颜色抖动
- **World Judge 判别器**: 学生更新每 5 步，判别器在其余步更新

### 消融实验

| 配置 | 情节级准确率↑ | 召回率↑ |
|------|------------|--------|
| w/o State Expert（不含状态专家） | 0.74 | 0.67 |
| w/ 通用 VLM 替代 World Judge | 0.63 | — |
| **完整 GE-Sim 2.0** | **0.81** | **0.82** |

**关键发现**:
- 去除 State Expert 后准确率降 7pp、召回率降 15pp，证明本体感知状态对闭环策略评估不可或缺
- 用通用 VLM 替换 World Judge 造成 ~19pp 精度损失，领域专化奖励模型必要性得证

---

## 批判性思考

### 优点

1. **系统完整性**: 同时提供视觉、状态、奖励三个维度，是目前功能最完备的机器人视频世界仿真器。
2. **强烈的实验对齐**: 闭环成功率预测与真实机器人结果的任务级偏差在 1pp 以内，实用性有力证明。
3. **可扩展性**: 拟实时推理吞吐量（2.3s/25帧）支持大规模合成数据生成，已验证 +15pp 真实策略提升。

### 局限性

1. **细粒度物理感知不足**: 液位变化、火焰转移等小尺度视觉信号仍难以精确建模，在 Pour water 和 Borrow flame 任务上 World Judge 事件距离较大（>30帧）。
2. **跨机体泛化未验证**: 当前仅在双臂平台上验证，不同机器人本体（四足、单臂等）的泛化能力未知。
3. **统一模型缺失**: 状态预测和奖励判断当前是分离模块，作者也指出未来应合并为统一模型。
4. **训练数据透明度低**: 未提供具体数据规模（小时数、轨迹数）和数据构成比例。

### 潜在改进方向

1. **统一状态-奖励-视觉联合生成**: 将三模块合并为单一 Transformer，减少模块间信息损失。
2. **在线 RL 集成**: 当前仅用于离线数据增强，结合在线强化学习可进一步提升数据利用效率。
3. **接触感知建模**: 引入物理约束（接触力、形变）改善精细操作场景的视觉一致性。

### 可复现性评估

- [x] 代码开源（计划开源）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（文中描述较为详细）
- [ ] 数据集可获取（私有真实机器人数据）

---

## 关联笔记

### 基于

- [[Action-Conditioned World Model]]: GE-Sim 2.0 在动作条件视频生成框架上扩展
- [[Diffusion Transformer]]: 采用 Cosmos-Predict2-2B DiT 作为 Vision Expert 主干
- [[Flow Matching]]: Vision Expert 和 State Expert 均使用 flow-matching 训练目标
- [[Distribution Matching Distillation]]: 用 DMD2 实现扩散步骤蒸馏加速
- [[RoPE]]: 多视角 token 的 3D 时空位置编码

### 对比

- [[Ctrl-World]]: 主要竞争对手，仅支持视觉输出无状态/奖励
- [[DreamDojo]]: 另一竞争视频世界模型，不支持多视角

### 方法相关

- [[生成世界模型]]: GE-Sim 2.0 所属的仿真范式
- [[VLM Critic]]: World Judge 的专业化实例
- [[Cross-Attention]]: State Expert 消费视觉特征的机制
- [[AdaLN]]: 时间步信号注入机制
- [[行为克隆]]: 奖励过滤行为克隆作为策略提升手段
- [[闭环控制]]: GE-Sim 2.0 的核心应用场景

### 硬件/数据相关

- [[PSNR]]: 视觉质量主要评估指标之一
- [[SSIM]]: 视觉质量评估指标
- [[LPIPS]]: 感知图像质量评估指标
- [[FVD]]: 视频质量评估指标

---

## 速查卡片

> [!summary] GE-Sim 2.0
> - **核心**: 首个同时提供视觉/本体状态/奖励三维度的闭环机器人视频世界仿真器
> - **方法**: Cosmos-Predict2-2B DiT 视觉专家 + 轻量状态专家 + 专域奖励判别器 + DMD2 加速
> - **结果**: WorldArena 最优综合分数；真实成功率对齐偏差 <1pp；奖励过滤 BC 平均 +15pp 策略提升；单 H100 25帧 2.3秒
> - **代码**: [ge-sim-v2.github.io](https://ge-sim-v2.github.io/)

---

*笔记创建时间: 2026-05-29*

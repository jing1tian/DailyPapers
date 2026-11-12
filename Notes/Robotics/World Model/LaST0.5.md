---
title: "LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning"
method_name: "LaST0.5"
authors: [Xuheng Zhang, Yiming Zhang, Jialin Gao, Hao Chen, Jiaming Liu, Ji Li, Runzhong Huang, Yinxi Wang, Chenyang Gu, Zihan Yang, Yaowei Liu, Zhonghao Yan, Ziyu Guo, Renrui Zhang, Peng Jia, Pheng-Ann Heng, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, robotic-manipulation, latent-reasoning, mixture-of-transformers, bimanual, value-estimation, visual-prompting]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.14048v1
created: 2026-11-12
---

# 论文笔记：LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多机构（含 Shanghang Zhang @ PKU, Pheng-Ann Heng @ CUHK, Renrui Zhang @ CUHK） |
| 日期 | November 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[Cosmos-Policy]], [[Fast-WAM]], [[π0.5\|π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.14048) / 代码未开源 |

---

## 一句话总结

> LaST0.5 在 [[世界动作模型（WAM）]] 中引入交互中心潜空间（Interaction-Centric Latent Expert）和进度感知自适应执行（AWAT），通过动作轨迹曲线 + 目标框的视觉监督信号连接世界动力学与局部操作语义。

---

## 核心贡献

1. **Interaction-Centric Latent Reasoning（ICLR）**: 在世界专家与动作专家之间插入轻量 [[Interaction-Centric Latent Reasoning|潜空间专家]]，以 2D 末端执行器轨迹曲线 + 目标物体 bounding box 作为离线视觉监督目标，使潜空间聚焦于局部机器人–物体交互，而非全局场景语义。
2. **Mixture-of-Transformers（MoT）三专家架构**: 由[[世界专家（World Expert）|世界专家]]（Cosmos-Predict2.5-2B DiT）、[[Interaction-Centric Latent Reasoning|潜空间专家]]（Janus 初始化）、[[Action Expert|动作专家]]（Flow Matching）通过 blockwise 因果注意力掩码串联，梯度从潜/动作专家不回传到世界专家以保护预训练先验。
3. **Adaptive World-to-Action Transfer（AWAT）**: 冻结基础模型后，接入轻量 MLP 进度值头，依据预测进度值 $\hat{v}_i$ 动态截断动作执行；进度下降时触发重采样并行筛选"最优"未来预测，从而缓解想象错误传播。

---

## 问题背景

### 要解决的问题

现有 [[世界动作模型（WAM）]] 将世界模型的**全局场景表征**直接送入动作生成，而机器人操作高度依赖**局部末端执行器–目标物体交互**，全局表征可能弱化这一关键信号。

### 现有方法的局限

- Cosmos Policy、Fast-WAM 等直接用 DiT 隐状态指导动作，缺乏显式的局部交互语义；
- [[Latent Chain-of-Thought]] 类方法虽引入中间推理步骤，但监督信号仍是全局图像特征，未专门对准操作接触区域；
- 现有 WAM 多为静态"预测-执行"，无法在执行过程中因世界模型失败而主动重规划。

### 本文的动机

用**离线可自动生成**的交互视觉线索（末端执行器轨迹投影 + Grounding DINO 目标框）作为潜空间的监督目标，使中间表征显式捕获操作关键区域；再利用 [[Progress Estimation|进度估计]] 实现运行时动态决策。

---

## 方法详解

### 模型架构

LaST0.5 采用 **[[Mixture-of-Transformers]]（MoT）** 架构，包含三个并行但因果关联的专家：

- **输入**: 语言指令 + 多视角 RGB 图像（历史 5 帧）
- **World Expert**: 由 [[Cosmos-Predict-2.5|Cosmos-Predict2.5-2B]] DiT 初始化，预测未来 12 帧（$256\times256$），保留原始生成损失 $\mathcal{L}_\text{world}$
- **Latent Expert**: 由 [[Janus]] 初始化，条件化于世界专家输出，生成 M=3 个紧凑潜向量 $\mathcal{Z}$，每段一个可学习 query token
- **Action Expert**: 语言模型主干独立权重，[[Flow-Matching|Flow Matching]] 解码 24 个动作 token（12 步 × 2 臂），Dual Noise Token + MLP 头支持异构本体
- **总预训练数据**: 约 2.48M episodes / 536M 帧 / 14,892 小时

注意力掩码为 **blockwise 因果掩码**（World → Latent → Action），世界专家的 Key-Value 被 detach，阻止梯度回传破坏物理先验。

### 核心模块

#### 模块 1: Interaction-Centric Latent Reasoning（ICLR）

**设计动机**: 利用[[可供性（Affordance）|操作接触区域]]线索，使中间表征聚焦机器人-环境交互而非背景干扰。

**潜空间目标构建（离线）**：
- 对每段最后一帧，叠加两类注释：
  1. **末端执行器轨迹曲线**：从 3D 末端执行器轨迹经相机参数投影到 2D，绘制连续曲线
  2. **目标物体 Bounding Box**：用 [[DINO|Grounding DINO]] 生成候选框，选取在抓取/释放事件时刻最接近工具中心点的框
- 冻结 [[SigLIP]] encoder，对注释帧提取 patch tokens，可训练 MLP 聚合为每段一个目标向量 $z_i^\text{GT}$

**潜空间对齐训练**：
- Latent Expert 生成的 $\hat{z}_i$ 与目标 $z_i^\text{GT}$ 做余弦对齐（见关键公式 → [[Interaction-Centric Latent Reasoning|潜空间对齐损失]]）

#### 模块 2: Adaptive World-to-Action Transfer（AWAT）

**设计动机**: 世界模型预测可能偏离真实轨迹，不加过滤直接执行会导致动作失效；[[Progress Estimation|进度估计]] 提供实时任务状态信号。

**具体实现**：
- 冻结基础三专家模型后，附加轻量 MLP 进度值头 $g_\phi$，接受世界专家 DiT 隐状态作为输入
- 进度目标 $v_i^\text{GT} = t_i / T$（片段时间步占总时长比例）
- 训练时用帧反转/重复帧数据增强，使估计器能处理停滞/回退情况
- **自适应执行规则**：
  - 仅在 $\hat{v}_i > \hat{v}_{i-1}$（进度递增）时执行对应动作段
  - 进度回退 → 停止执行，触发新观测重新推理
  - 初始进度 < 上一轮估计 → 触发并行多路重采样，选 $\hat{v}_i$ 最高的未来预测
- **颗粒度**：每 4 步一个进度值效果最优（见消融 → 表格 2 颗粒度消融）

---

## 关键公式

### 公式 1: [[世界动作模型（WAM）|联合因式分解]]

$$
p(\mathbf{a}, \mathcal{Z}, \hat{V} \mid c_t) = p(\hat{V} \mid c_t) \cdot p(\mathcal{Z} \mid \hat{V}, c_t) \cdot p(\mathbf{a} \mid \mathcal{Z}, \hat{V}, c_t)
$$

**含义**: LaST0.5 将未来视频 $\hat{V}$、潜序列 $\mathcal{Z}$ 和动作块 $\mathbf{a}$ 的联合分布分解为因果链式结构（World → Latent → Action），每部分对应一个专家的生成目标。

**符号说明**:
- $\mathbf{a}$：动作块（12步 × 2臂的相对 6-DoF 末端执行器姿态）
- $\mathcal{Z}$：交互中心潜序列（M=3 个紧凑向量）
- $\hat{V}$：预测未来帧（12帧，$256\times256$）
- $c_t$：当前观测与语言指令

### 公式 2: [[Interaction-Centric Latent Reasoning|潜空间对齐损失]]

$$
\mathcal{L}_\text{lat} = 1 - \frac{1}{M} \sum_{i=1}^{M} \cos(\hat{z}_i,\, z_i^{\text{GT}})
$$

**含义**: 用余弦相似度对齐潜专家输出 $\hat{z}_i$ 与 [[SigLIP]] 编码的交互注释目标 $z_i^\text{GT}$，驱动潜空间捕获操作接触区域语义。

**符号说明**:
- $M=3$：时间对齐片段数
- $\hat{z}_i$：Latent Expert 对第 $i$ 片段的预测向量
- $z_i^\text{GT}$：冻结 SigLIP 对叠加末端轨迹曲线 + 目标框图像的 patch 聚合向量
- $\cos(\cdot, \cdot)$：余弦相似度

### 公式 3: [[Progress Estimation|进度值估计]]

$$
\hat{v}_i = g_\phi\!\left(\mathrm{Pool}\!\left(\mathcal{H}_i\right)\right)
$$

**含义**: 轻量 MLP 进度头 $g_\phi$ 对世界专家 DiT 隐状态 $\mathcal{H}_i$（每片段 4 帧对应的 token）做均值池化后估计该片段的任务完成进度。

**符号说明**:
- $\hat{v}_i \in [0, 1]$：第 $i$ 片段的预测进度值
- $g_\phi$：可训练 MLP 头（基础模型冻结后单独训练）
- $\mathcal{H}_i$：世界专家的 DiT 隐状态 token 集
- $\mathrm{Pool}(\cdot)$：均值池化

### 公式 4: [[Progress Estimation|进度损失]]

$$
\mathcal{L}_\text{value} = \frac{1}{M} \sum_{i=1}^{M} \ell_\text{reg}\!\left(\hat{v}_i,\, v_i^{\text{GT}}\right)
$$

**含义**: 对预测进度 $\hat{v}_i$ 与真实进度 $v_i^{\text{GT}} = t_i / T$ 做回归损失（L1 或 Huber）。

**符号说明**:
- $v_i^\text{GT} = t_i / T$：片段中间时间步 $t_i$ 除以总时步 $T$
- $\ell_\text{reg}$：回归损失（L1/Huber）

### 公式 5: [[世界动作模型（WAM）|联合训练目标]]

$$
\mathcal{L} = \lambda_1 \mathcal{L}_\text{world} + \lambda_2 \mathcal{L}_\text{lat} + \lambda_3 \mathcal{L}_\text{act}
$$

**含义**: 三专家损失加权求和，系数按量级平衡各损失大小。

**符号说明**:
- $\mathcal{L}_\text{world}$：Cosmos 原始扩散生成损失
- $\mathcal{L}_\text{lat}$：潜空间余弦对齐损失
- $\mathcal{L}_\text{act}$：[[Flow-Matching|Flow Matching]] 动作损失
- $\lambda_1, \lambda_2, \lambda_3$：加权系数（按损失量级选取）

### 公式 6: 相对动作计算

$$
\delta R = R_t^\top R_{t+1}, \quad \delta \mathbf{t} = R_t^\top\!\left(\mathbf{t}_{t+1} - \mathbf{t}_t\right)
$$

**含义**: 预训练数据标准化为相对末端执行器姿态，旋转和平移均在当前帧坐标系下表示。

**符号说明**:
- $R_t, \mathbf{t}_t$：第 $t$ 帧末端执行器的旋转矩阵和平移向量
- $\delta R, \delta \mathbf{t}$：相对旋转和相对平移（动作空间目标）

---

## 关键图表

### Figure 1: Overview / 方法动机与框架概览

![Figure 1 — LaST0.5 总览](https://arxiv.org/html/2611.14048v1/teaser.png)

**说明**: 左侧对比现有 WAM（全局场景表征直接传递）与 LaST0.5（插入 [[Interaction-Centric Latent Reasoning|潜空间专家]] 聚焦局部交互）；右侧展示 AWAT 进度监控机制。

### Figure 2: LaST0.5 完整框架

![Figure 2 — LaST0.5 完整框架](https://arxiv.org/html/2611.14048v1/method.png)

**说明**: (a) [[Mixture-of-Transformers|MoT]] 三专家架构：World Expert（灰）→ Latent Expert（蓝）→ Action Expert（绿）通过 blockwise 因果注意力串联；(b) AWAT 进度值估计与自适应执行/重采样机制；(c) 注意力掩码矩阵，Key-Value detach 防止梯度回传。

### Figure 3: 消融实验结果

**说明**: (a) 单独移除各专家的影响（移除世界专家 −3.6%，移除潜专家 −1.8%，基线 LaST0.5 w/o AWAT 99.6%）；(b) 交互中心潜目标构建消融（无轨迹 −0.6%，无目标框 −0.6%，二者均无 −1.0%）；(c) AWAT 时间颗粒度消融（4步颗粒度最优 99.7%）。
> ⚠️ arXiv HTML 中未提取到图片 URL，请参考 [PDF](https://arxiv.org/pdf/2611.14048)

### Figure 4: 注意力可视化

![Figure 4 — 注意力热力图对比](https://arxiv.org/html/2611.14048v1/attention_map.png)

**说明**: LaST0.5 注意力高度集中在末端执行器和目标物体；消融去掉潜专家后注意力向背景分散；去掉交互中心目标后注意力进一步扩散，验证 ICLR 监督的有效性。

### Figure 5: 真实世界任务执行可视化

![Figure 5 — 真实世界任务执行序列](https://arxiv.org/html/2611.14048v1/real_world.png)

**说明**: 展示 LaST0.5 在 Tianji Marvin 双臂机器人上执行 Bag Pulling、Object Organization、Screw Fastening、Parcel-Box Packing 四类任务的时序帧（从左到右），体现精密协调与工具使用能力。

### Figure 6: 分布外泛化场景设置

![Figure 6 — OOD 场景设置](https://arxiv.org/html/2611.14048v1/ood_setting.png)

**说明**: 三类 OOD 变化：位置偏移（物体初始位置随机偏移）、物体替换（新物体类别）、背景替换（桌面背景更换），用于衡量策略对未见场景的泛化能力。

### Figure 7: 完整注意力掩码矩阵（附录）

![Figure 7 — 完整注意力掩码](https://arxiv.org/html/2611.14048v1/apdx_attention_v1.png)

**说明**: 完整展示 LaST0.5 blockwise 因果掩码，行为 Query，列为 Key，色块表示可见，白块表示屏蔽，直观说明三专家间的信息流向与梯度隔离策略。

### Figure 8: 真实机器人实验平台（附录 C）

![Figure 8 — 真实机器人实验设置](https://arxiv.org/html/2611.14048v1/apdx_setup_v1.png)

**说明**: Tianji Marvin 7-DoF 双臂平行夹爪，配备 ZED 2i 头部相机（全局视角）和两个 Insta360 GO 3S 腕部相机，以 PICO VR 手柄遥控采集 150 条演示数据/任务。

### Figure 9: 动作到视觉的注意力热图（附录 F）

![Figure 9 — 动作-视觉注意力可视化](https://arxiv.org/html/2611.14048v1/apdx_attentionmap_v1.png)

**说明**: 追加可视化，展示 LaST0.5 完整版与消融变体在多任务上的注意力对比，完整模型持续聚焦末端执行器和目标接触点。

### Figure 10: 进度值轨迹可视化（附录 F）

![Figure 10 — 进度值曲线](https://arxiv.org/html/2611.14048v1/apdx_value_v2.png)

**说明**: 成功执行时进度值单调递增接近 1.0；失败案例中进度值在关键错误帧急剧下降，验证了 [[Progress Estimation|进度值]] 作为实时任务状态指示器的可靠性。

### Figure 11: 额外真实任务执行可视化（附录 F）

![Figure 11 — 更多真实任务执行示例](https://arxiv.org/html/2611.14048v1/real_world_appendix.png)

**说明**: 追加多任务执行帧序列，包括精细工具使用（胶带封箱）和双臂协调（拉包）。

### Figure 12: 失败案例分析（附录 G）

![Figure 12 — 失败案例序列](https://arxiv.org/html/2611.14048v1/apdx_failure_case_v1.png)

**说明**: 红框标注失败发生位置。四类典型失败：(1) Bag Pulling — 拉链被机械阻力卡住；(2) Object Organization — 抓取黄瓜时碰移收纳盒；(3) Parcel Packing — 平行夹爪夹持胶带力不足打滑；(4) Screw Fastening — 毫米级对准偏差导致插入失败（仅视觉反馈的瓶颈）。

---

### Table 1: RLBench 10 任务平均成功率

| Method | Mean S.R. (%) |
|--------|--------------|
| OpenVLA-OFT | 45.0 ± 3.0 |
| π0.5 | 65.0 ± 4.0 |
| Fast-WAM | 63.0 ± 3.0 |
| HybridVLA | 74.0 ± 4.0 |
| Cosmos Policy | 77.0 ± 2.0 |
| LaST0 | 82.0 ± 3.0 |
| LaST0.5 w/o AWAT | 85.0 ± 2.0 |
| **LaST0.5** | **88.0 ± 3.0** |

**关键发现**: AWAT 贡献 +3.0% (85→88)；ICLR 贡献 +3.0% (82→85)；LaST0.5 在 Close Laptop、Umbrella Out、Frame Off Hanger 等精细操作任务达 90-100%。

### Table 2: LIBERO 四套件最优成功率（%）

| Method | Spatial | Object | Goal | Long | Mean |
|--------|---------|--------|------|------|------|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| InternVLA | 98.6 | 99.8 | 98.6 | 98.4 | 98.9 |
| LaST0 | 99.2 | 99.6 | 98.0 | 95.6 | 98.1 |
| LaST0.5 w/o AWAT | 99.6 | 100.0 | 98.1 | 99.0 | 99.2 |
| **LaST0.5** | **99.7** | **100.0** | **99.1** | **99.3** | **99.5** |

**关键发现**: LIBERO-Long（长时序任务）提升最显著，从 95.6%（LaST0）→ 99.3%（LaST0.5），AWAT 在长时序中进度监控优势更突出。

### Table 3: 真实世界各任务成功率（%，20 rollouts 每任务）

| Method | Bag Pulling | Object Org. | Screw S1 | Screw S2 | Parcel S1 | Parcel S2 | Parcel S3 | Avg |
|--------|------------|-------------|----------|----------|----------|----------|----------|-----|
| Cosmos Policy | 60.0 | 85.0 | 60.0 | 20.0 | 60.0 | 35.0 | 10.0 | 43.8 |
| LaST0 | 60.0 | 75.0 | 75.0 | 25.0 | 65.0 | 50.0 | 15.0 | 43.8 |
| π0.5 | 75.0 | 85.0 | 80.0 | 30.0 | 95.0 | 75.0 | 20.0 | 52.5 |
| **LaST0.5** | **80.0** | **95.0** | **90.0** | **50.0** | **100.0** | **85.0** | **30.0** | **63.8** |

**关键发现**: Parcel S3（胶带封箱）最难（全局最高 30%，物理抓取限制）；Screw S2（螺丝插入）次难；LaST0.5 均取得显著提升，尤其在 Parcel S1（100%）和 Screw S1（90%）。

### Table 4: 分布外泛化成功率（Bag Pulling + Object Organization 均值，%）

| Method | 位置偏移 | 物体替换 | 背景替换 |
|--------|---------|---------|---------|
| Cosmos Policy | 17.5 | 35.0 | 62.5 |
| LaST0 | 20.0 | 40.0 | 55.0 |
| π0.5 | 15.0 | **55.0** | 67.5 |
| **LaST0.5** | **40.0** | **55.0** | **72.5** |

**关键发现**: LaST0.5 在位置偏移（+20% vs π0.5）和背景替换（+5%）上最优，物体替换与 π0.5 持平；交互中心注意力帮助模型忽略背景干扰。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件 × 10 任务 | 仿真 Franka，单/多阶段 | 微调 + 评测 |
| [[RLBench]] | 10 任务 | CoppeliaSim Franka，单前视角 | 微调 + 评测 |
| AgiBot / InternData-A1 / DROID / EgoDex / RoboCOIN / RoboMIND / Genrobot | 2.48M episodes / 14,892h | 异构机器人，10 Hz 标准化 | 大规模预训练 |

### 实现细节

- **Backbone**: [[Cosmos-Predict-2.5|Cosmos-Predict2.5-2B]] DiT（世界专家）+ [[Janus]]（潜专家）
- **动作解码**: [[Flow-Matching|Flow Matching]]（10步去噪，推理时）
- **优化器**: AdamW
- **训练轮数**: LIBERO 100 epochs / RLBench 300 epochs
- **硬件**: 8× NVIDIA H20 GPU
- **片段数**: M=3（每段 4 帧）
- **动作 token**: 24 个（12步 × 2臂），Dual Noise Token 处理异构本体，零填充到最大维度
- **潜 token 数**: 3（消融最优；6/12 token 无提升）
- **目标构建**: 离线，Grounding DINO 生成候选框，选取抓取/释放时刻最近 TCP 框

### 可视化结果

- 注意力热图（Figure 4, 9）：LaST0.5 完整版注意力高度集中在末端执行器和目标物体；去掉潜空间监督后注意力弥散到无关背景区域。
- 进度曲线（Figure 10）：成功执行单调递增；失败帧处急剧下降，可作为早停预警信号。

---

## 批判性思考

### 优点

1. **交互监督信号廉价且自动**: 末端执行器轨迹投影 + Grounding DINO 目标框可全自动从标注数据中生成，无需额外人工标注。
2. **架构设计优雅**: blockwise 因果掩码 + 梯度 detach 解耦世界先验与下游任务微调，避免 catastrophic forgetting。
3. **进度机制提供内在反馈**: AWAT 利用模型自身预测判断执行质量，不需要额外 reward 模型或外部评估器，系统封闭。
4. **真实 bimanual 表现强劲**: 在难度极高的 Screw S2（螺丝插入）和 Parcel S3（胶带封箱）上相比基线提升显著。

### 局限性

1. **视觉反馈的毫米级精度瓶颈**: Screw Fastening S2 仍因 mm 级对准误差失败，纯视觉策略难以突破力觉反馈缺失的物理限制。
2. **平行夹爪的物理约束**: 胶带任务失败暴露夹爪设计缺陷，非模型问题但限制了任务范围。
3. **预训练数据集中度高**: Genrobot (UMI) 占比约 61%，数据多样性有待改善。
4. **潜空间监督依赖相机外参**: 末端执行器轨迹 2D 投影需要精确相机参数，限制无标定场景的部署。

### 潜在改进方向

1. 引入力觉传感器反馈配合视觉进度估计，解决精密插入任务的物理限制。
2. 探索在线/实时目标框生成（替代离线预处理），增强泛化能力。
3. 更均衡的预训练数据分布，减少对 UMI 数据的依赖。

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（消融超参、数据规模、硬件均记录）
- [x] 评测数据集可获取（LIBERO、RLBench 均开源）

---

## 关联笔记

### 基于

- [[Cosmos-Predict-2.5|Cosmos-Predict2.5]]: 世界专家基础模型（2B DiT）
- [[Janus]]: 潜专家初始化模型
- [[AWAT]]: 本文改进的前序 WAM 机制（LaST 系列前身）

### 对比

- [[Cosmos-Policy]]: 直接 DiT 表征到动作，无交互中心潜空间
- [[Fast-WAM]]: 仿真 benchmark 对比基线（63% vs 88% on RLBench）

### 方法相关

- [[Mixture-of-Transformers]]: 三专家并行 Transformer 架构范式
- [[Flow-Matching]]: 动作专家的去噪解码方式
- [[Interaction-Centric Latent Reasoning]]: 本文核心新概念
- [[Progress Estimation]]: AWAT 的进度值估计基础
- [[SigLIP]]: 冻结视觉 encoder，提供潜空间目标特征
- [[可供性（Affordance）]]: 目标框作为操作可供性的视觉代理
- [[Action Chunking]]: 动作块预测（12步）
- [[Future Latent Alignment]]: 相关潜空间对齐技术

### 硬件/数据相关

- [[LIBERO]]: 仿真评测套件（99.5%）
- [[RLBench]]: 仿真评测 benchmark（88.0% mean S.R.）

---

## 速查卡片

> [!summary] LaST0.5 (arXiv 2611.14048)
> - **核心**: WAM 中插入交互中心潜专家 + 进度感知自适应执行
> - **方法**: MoT（World/Latent/Action 三专家）+ ICLR（末端轨迹曲线 + 目标框监督）+ AWAT（进度值头触发重采样）
> - **结果**: LIBERO 99.5% / RLBench 88.0% mean S.R. / 真实双臂 63.8% avg
> - **代码**: 未开源

---

*笔记创建时间: 2026-11-12*

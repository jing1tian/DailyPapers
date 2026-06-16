---
title: "Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack"
method_name: "HyVLA-0.5"
authors: [He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Han, Zhengyou Zhang]
year: 2026
venue: arXiv
tags: [vla, cross-embodiment, flow-matching, preference-optimization, data-collection, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.14409
created: 2026-06-16
---

# 论文笔记：Hy-Embodied-0.5-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent Robotics X |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[π0]], [[π0.5]], [[Motus]], [[starVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.14409) / Code（即将开源）|

---

## 一句话总结

> HyVLA-0.5 是一套覆盖数据采集、模型设计、持续预训练、监督微调、RL 后训练到真实部署的端到端机器人学习全栈系统，通过 [[Mixture-of-Transformers|MoT]] 主干 + [[Flow Matching|流匹配]] 动作专家 + 无奖励模型偏好优化 (FlowPRO)，在 RoboTwin 2.0 仿真基准达到 90.9% 成功率，并成功跨本体迁移到两款形态各异的机器人。

---

## 核心贡献

1. **Hy-UMI-10K 数据集**: 自研指尖 UMI 硬件 + 外部光学动捕，采集超过 10,000 小时、11M+ Episode、70 个任务的精细演示数据，精度达亚毫米级。
2. **HyVLA-0.5 全栈系统**: 以 4B 参数 [[Mixture-of-Transformers|MoT]] 主干搭配紧凑记忆编码器与 [[Flow Matching|流匹配]] 动作专家，通过 [[Delta-Chunk 动作表示]] 解耦策略学习与机器人运动学，实现无缝跨本体迁移。
3. **FlowPRO 离线 RL 后训练**: 仅凭真实机器人上的失败-成功对比轨迹，无需任何奖励模型，通过 Proximalized Preference Optimization (PRO) 将四项精细操作任务成功率推至 94–99%。

---

## 问题背景

### 要解决的问题

现有 [[VLA（视觉-语言-动作模型）|VLA]] 系统通常只关注模型架构或某一训练阶段，缺乏从数据采集到真实世界部署的完整闭环解决方案，难以在精细操作任务（亚毫米精度）和跨形态机器人迁移上同时取得高成功率。

### 现有方法的局限

- 遥操采集精度有限，SLAM 定位存在累积误差；
- 策略与机器人运动学强耦合，跨本体迁移需要大量目标机器人数据；
- RL 后训练依赖奖励模型或价值函数，样本效率低，易 reward hacking；
- 动作块拼接存在 C⁰ 不连续，造成真实机器人抖动。

### 本文的动机

以指尖 UMI + 光学动捕替代 SLAM 获得亚毫米精度演示；用 [[Delta-Chunk 动作表示]] 将策略预测与运动学解耦；以对比偏好对直接替代奖励模型；用三次 Bézier 曲线拼接保证动作 C¹ 连续。

---

## 方法详解

### 模型架构

![[HyVLA05_fig1_overview.png]]

HyVLA-0.5 采用 **[[Mixture-of-Transformers|MoT]]** 架构：

- **输入**: 语言指令 $\ell$ + 多视角多帧 RGB 图像 $\mathcal{I}_t = \{I^v_{t-k}\}_{v=1:n}^{k=0:K-1}$ + 本体感受状态 $s_t$
- **Backbone**: [[Mixture-of-Transformers|Hy-Embodied-0.5-MoT]]（4B 参数），视觉与文本 token 共享自注意力，各自独立 QKV 与 FFN 参数
- **视觉编码器**: Hy-ViT 2.0，原生分辨率处理，含[[紧凑记忆编码器]]做时序-空间注意力
- **动作专家**: [[Flow Matching|流匹配]] 速度场 $v_\theta$，连续预测 [[Action Chunking|动作块]] $\mathbf{A}_t$
- **输出**: 每臂 10 维增量 EEF 动作（3D 平移 + 6D 旋转 + 1D 夹爪）
- **总参数**: 4B（主干）+ 轻量动作头

### 核心模块

#### 模块 1: 紧凑记忆编码器（Compact Memory Encoder）

![[HyVLA05_fig2_architecture.png]]

**设计动机**: 利用 [[因果注意力|Causal Attention]] 跨帧捕捉时序依赖，同时用双向注意力在单帧内建模多视角关系，压缩历史信息。

**具体实现**:
- 每 4 层插入一个**时序注意力块**，对同一空间位置的 $K$ 帧施加[[Block Causal Attention|因果掩码注意力]]（Temporal Pass）：
  $$\tilde{\mathbf{V}}_p = \text{CausalAttn}(\mathbf{Q}_p, \mathbf{K}_p, \mathbf{V}_p)$$
- 再在每帧内对 $n$ 个 patch 施加双向注意力（Spatial Pass）：
  $$\tilde{\mathbf{X}}_k = \mathbf{W}_O \cdot \text{Attn}(\mathbf{Q}_k, \mathbf{K}_k, \tilde{\mathbf{V}}_k)$$

#### 模块 2: Delta-Chunk 动作表示

**设计动机**: 将策略输出与机器人绝对运动学解耦，使同一 checkpoint 可跨不同运动学结构的机器人部署。

**具体实现**:
- 策略预测**相对于当前末端执行器坐标系**的增量块 $^{G_t}T_{G_{t+k}}$；
- 固定底座臂（JAKA K1）平台映射：
  $${}^W T_{G_{t+k}} = {}^W T_{G_t} \cdot {}^{G_t} T_{G_{t+k}}$$
- 浮动底座人形（Astribot S1）平台映射：
  $${}^C T_{G_{t+k}} = ({}^W T_C)^{-1} \cdot {}^W T_{G_t} \cdot {}^{G_t} T_{G_{t+k}}$$
- 之后送入 IK 求解器即可，无需针对每款机器人重新训练策略。

#### 模块 3: FlowPRO 离线 RL 后训练

![[HyVLA05_fig5_flowpro_pipeline.png]]

![[HyVLA05_fig6_rpro_opt.png]]

**设计动机**: 利用操作员干预产生的失败-成功对比轨迹作为对比信号，无需奖励模型，直接用 [[DPO|偏好优化]] 范式提升策略。

**三大原则**:
1. (P1) 将失败直接转化为负样本对比信号；
2. (P2) 完全避免奖励和价值函数；
3. (P3) 用近端正则化防止 reward hacking。

**数据收集**: 干预-回滚协议 — 操作员在策略失败时介入，系统回滚到先前状态，记录执行段为负轨迹、遥操纠正段为正轨迹。

#### 模块 4: 异步推理 + C¹ Bézier 平滑

![[HyVLA05_fig7_async_timeline.png]]

![[HyVLA05_fig8_traj_comparison.png]]

**设计动机**: 解决动作块边界处的 C⁰ 不连续抖动问题，同时让推理与执行并行提升实时性。

**具体实现**:
1. **截断过期前缀**: 保留块长度 $M$，截去 $K = \lceil N/\alpha \rceil$ 步过期动作
2. **选择连接点**: $c = \text{clip}(\lfloor \gamma M \rfloor, 1, M-2)$
3. **三次 Bézier 拼接**（公式见后），保证 C¹ 连续

---

## 关键公式

### 公式 1: 多模态观测与动作块定义

$$
\mathbf{o}_t = (\mathcal{I}_t, \ell, \mathbf{s}_t), \quad \mathcal{I}_t = \{I^v_{t-k}\}_{v=1:n}^{k=0:K-1}, \quad \mathbf{A}_t = (\mathbf{a}_t, \ldots, \mathbf{a}_{t+H-1})
$$

**含义**: 定义时刻 $t$ 的观测（多视角多帧图像 + 语言 + 本体状态）与预测的动作块。

**符号说明**:
- $\mathcal{I}_t$: 视觉流，$n$ 个视角、$K$ 帧历史
- $\ell$: 语言指令
- $\mathbf{s}_t$: 本体感受状态
- $H$: 动作块长度

---

### 公式 2: [[Flow Matching|流匹配训练目标]]

$$
\mathcal{L}_{fm}(\theta) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t),\, q(\mathbf{A}^\tau_t|\mathbf{A}_t)} \left[ \left\| v_\theta(\mathbf{A}^\tau_t, \mathbf{o}_t) - (\epsilon - \mathbf{A}_t) \right\|^2_2 \right]
$$

其中加噪动作：

$$
\mathbf{A}^\tau_t = \tau \mathbf{A}_t + (1-\tau)\epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \tau \sim \mathcal{U}(0,1)
$$

**含义**: 训练速度场 $v_\theta$ 预测从噪声到目标动作的去噪方向。

**符号说明**:
- $v_\theta$: 动作专家速度场网络
- $\tau$: 流时间步，均匀采样自 $[0,1]$
- $\epsilon$: 高斯噪声
- $\mathbf{A}^\tau_t$: 加噪动作

---

### 公式 3: [[Co-training|联合训练目标]]

$$
\mathcal{L}(\theta) = \mathcal{L}_{fm}(\theta) + \lambda_{ntp} \mathcal{L}_{ntp}(\theta)
$$

$$
\mathcal{L}_{ntp}(\theta) = \mathbb{E}_{(\mathbf{c},y)\sim\mathcal{D}_{ct}} \left[ -\sum_{j=1}^M \log p_\theta(y_j | \mathbf{c}, y_{<j}) \right]
$$

**含义**: 流匹配损失 + 辅助下一 token 预测损失，保留 VLM 的语义与空间理解能力。

**符号说明**:
- $\lambda_{ntp}$: 混合系数（仅动作微调时为 0）
- $\mathcal{L}_{ntp}$: 视觉问答 / 空间接地辅助损失

---

### 公式 4: [[DPO|隐式奖励代理]]（FlowPRO）

$$
r_\theta(s, a) = \frac{\beta}{2} \left( \ell_{ref}(s,a) - \ell_\theta(s,a) \right)
$$

其中流匹配代理损失：

$$
\ell_\theta(s,a) = \mathbb{E}_{\tau\sim\mathcal{U}[0,1],\, \epsilon\sim\mathcal{N}(0,I)} \left[ \left\| v_\theta(a_\tau, \tau \mid s) - u(a_\tau|a) \right\|^2 \right]
$$

**含义**: 用参考策略与当前策略的流匹配损失之差作为无模型隐式奖励，避免单独训练奖励网络。

**符号说明**:
- $r_\theta$: 隐式奖励
- $\beta$: 缩放因子
- $\ell_{ref}$: 冻结参考策略的流匹配损失
- $u(a_\tau|a)$: 条件速度场

---

### 公式 5: [[RPRO 损失|Proximalized Preference Optimization 损失]]

$$
\mathcal{L}_{PRO}(\theta) = -\mathbb{E}_{(s,a^w,a^l)\sim\mathcal{D}} \left[ \log\sigma\bigl(r_\theta(s,a^w) - r_\theta(s,a^l)\bigr) + \frac{1}{2}\sum_{a\in\{a^w,a^l\}} \bigl[\log\sigma(r_\theta(s,a)) + \log\sigma(-r_\theta(s,a))\bigr] \right]
$$

$$
\mathcal{L}_{RPRO}(\theta) = \lambda_{PRO}\mathcal{L}_{PRO}(\theta) + \lambda_{SFT}\mathcal{L}_{SFT}(\theta)
$$

**含义**: 第一项为对比项（拉近胜出动作、推远失败动作），第二项为近端正则项（防止奖励幅度爆炸），合并 SFT 损失保持稳定性。

**符号说明**:
- $\sigma$: Sigmoid 函数
- $a^w / a^l$: 胜出 / 失败动作
- $\lambda_{PRO}, \lambda_{SFT}$: 权重系数（Round 1 为 80%/20%，之后为 70%/15%/15%）

---

### 公式 6: [[三次 Bézier 曲线|C¹ Bézier 动作拼接]]

截断与连接点选取：

$$
K = \lceil N/\alpha \rceil,\quad K \leq N-3; \qquad c = \text{clip}(\lfloor \gamma M \rfloor,\, 1,\, M-2)
$$

历史与未来切线方向：

$$
\hat{d}_{hist} = \frac{\mathbf{h}_0 - \mathbf{h}_{-1}}{\|\mathbf{h}_0 - \mathbf{h}_{-1}\|}, \qquad \hat{d}_{fut} = \frac{\mathbf{f}_{c+1} - \mathbf{f}_{c-1}}{\|\mathbf{f}_{c+1} - \mathbf{f}_{c-1}\|}
$$

控制点：

$$
\mathbf{P}_0 = \mathbf{h}_0,\quad \mathbf{P}_1 = \mathbf{P}_0 + \lambda\hat{d}_{hist},\quad \mathbf{P}_2 = \mathbf{P}_3 - \lambda\hat{d}_{fut},\quad \mathbf{P}_3 = \mathbf{f}_c
$$

三次 Bézier 曲线：

$$
\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3
$$

**含义**: 在当前执行位置 $\mathbf{h}_0$ 与新块连接点 $\mathbf{f}_c$ 之间插入 Bézier 段，端点切线与历史运动及未来预测对齐，保证 C¹ 连续。

**符号说明**:
- $N$: 原始块长度；$M = N - K$: 保留块长度
- $\alpha$: 截断比（$>1$）；$\gamma$: 连接比
- $\lambda = \sigma\|\mathbf{P}_3 - \mathbf{P}_0\|$: 切线长度缩放

---

## 关键图表

### Figure 1: 系统概览

![[HyVLA05_fig1_overview.png]]

**说明**: HyVLA-0.5 端到端全栈概览。[[Mixture-of-Transformers|MoT]] 主干配合[[Flow Matching|流匹配]]动作专家，以 [[Delta-Chunk 动作表示]] 预训练于 10K 小时 UMI 语料，再分 Track-A（同本体微调）和 Track-B（跨本体迁移）两路后训练，最后通过 FlowPRO 离线 RL 优化精细操作。

---

### Figure 2: 模型架构

![[HyVLA05_fig2_architecture.png]]

**说明**: [[Mixture-of-Transformers|MoT]] 框架通过共享联合注意力机制实现跨模态交互。紧凑记忆编码器每 4 层插入时序注意力块，右侧注意力掩码图展示了块状因果注意力策略：多视角内双向、帧间因果。

---

### Figure 3: UMI 数据采集硬件

![[HyVLA05_fig3_umi_hardware.jpeg]]

**说明**: 自研指尖 UMI 数据采集站。外部光学动捕系统提供亚毫米 6-DoF 精度跟踪，自我中心 RGB-D 相机捕捉场景语义，每手一个 6 维力/力矩传感器夹爪采集触觉信息。

---

### Figure 4: UMI 数据集分布

![[HyVLA05_fig4_dataset_dist.png]]

**说明**: Hy-UMI-10K 数据集的多维度分布图。覆盖洗衣房 (28.5%)、厨房 (19.2%)、个护杂务 (13.8%)、精细/工具使用 (10.4%)、收纳 (10.0%)、清洁 (5.7%) 六大场景家族，共 70 个任务，11M+ Episode。

---

### Figure 5: FlowPRO 数据流水线

![[HyVLA05_fig5_flowpro_pipeline.png]]

**说明**: 干预-回滚数据采集流程。操作员触发干预时系统回滚到先前状态，执行段记录为负轨迹，遥操纠正段为正轨迹，Bézier 平滑合成缺失的对应动作，生成逐状态偏好元组 $(s, a^w, a^l)$。

---

### Figure 6: RPRO 优化示意

![[HyVLA05_fig6_rpro_opt.png]]

**说明**: 可学习策略 $\pi_\theta$ 与冻结参考策略 $\pi_{ref}$ 对同一状态预测动作，对比项将 $\pi_\theta$ 拉向胜出动作、推离失败动作，近端正则项（蓝色虚线）防止奖励幅度爆炸。

---

### Figure 7: 异步执行时间线

![[HyVLA05_fig7_async_timeline.png]]

**说明**: 策略推理、Bézier 平滑、缓冲区写入与伺服率动作执行并行重叠，历史动作 $\mathcal{H}$ 用于估计下一块拼接切线。

---

### Figure 8: Bézier 平滑效果

![[HyVLA05_fig8_traj_comparison.png]]

**说明**: 原始动作块（橙色）vs. 异步 Bézier 平滑轨迹（蓝色）。平滑后双臂 x/y/z 各维度在块边界处的不连续性显著减少。

---

### Figure 9: 真实机器人评估

![[HyVLA05_fig9_realrobot_eval.png]]

**说明**: 六项双臂操作任务的真实机器人评估。左侧为任务执行快照，右侧为每任务成功率（%）。

---

### Figure 10: 力觉引导物体区分

![[HyVLA05_fig10_force_unitree.png]]

**说明**: Unitree G1 上的力觉引导物体区分任务。机器人依次抓取两个质量不同的箱子，将较轻的放入前篮，验证 UMI 工作站能采集可操作的触觉信息。

---

### Figure 11: 额外精细任务

![[HyVLA05_fig11_finegrained.png]]

**说明**: FlowPRO 后训练中的两项额外精细任务：USB 插入（亚毫米精度）和笔帽组装（空中双臂协同）。

---

### Figure 12: FlowPRO 迭代成功率

![[HyVLA05_fig12_iteration_sr.png]]

**说明**: 四项真实机器人任务的逐迭代成功率。Iteration 0 为共享 SFT checkpoint，Iterations 1-3 为连续后训练轮次。RPRO 在整个迭代过程中始终优于 DAgger 和 $\pi_{0.6}^*$。

---

### Table 1: RoboTwin 2.0 基准结果

| 方法 | Clean (%) | Randomized (%) |
|------|-----------|----------------|
| π₀ | 65.9 | 58.4 |
| ABot-M0 | 81.2 | 80.4 |
| π₀.₅ | 82.7 | 76.8 |
| Qwen-VLA | 86.1 | 87.2 |
| LingBot-VLA | 86.5 | 85.3 |
| starVLA | 88.2 | 88.3 |
| Motus | 88.7 | 87.0 |
| JoyAI-RA | 90.5 | 89.3 |
| **HyVLA-0.5** | **90.9** | **90.1** |
| w/o 紧凑记忆编码器 | 88.8 | 88.6 |
| w/o 记忆编码器 + UMI 预训练 | 88.1 | 87.9 |

**说明**: 50 任务 RoboTwin 2.0 测试集，每任务 100 次 rollout 平均。HyVLA-0.5 在 Clean 和 Randomized 两种设置下均排名第一，消融实验表明紧凑记忆编码器和 UMI 预训练各自带来 ~2% 提升。

---

### Table 2: 真实机器人 RL 后训练结果

| 微调方法 | Bottle (SR / CT) | Cap (SR / CT) | USB (SR / CT) | Zip (SR / CT) |
|---------|------------------|---------------|---------------|---------------|
| DAgger | 93±2.1% / 27s | 88±1.8% / 29s | 86±2.4% / 25s | 83±2.0% / 55s |
| π₀.₆* | 95±1.5% / 24s | 95±1.2% / 27s | 95±1.4% / 23s | 89±1.6% / 45s |
| **RPRO** | **99±0.6% / 16s** | **99±0.7% / 21s** | **98±0.9% / 22s** | **94±1.1% / 37s** |

**说明**: Dobot X-Trainer 上三轮后训练 (K=3)，3 个训练种子各 100 次随机 rollout 的均值 ± 标准差。RPRO 在成功率（SR）和完成时间（CT）上均全面领先 DAgger 和 $\pi_{0.6}^*$。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Hy-UMI-10K | 10K 小时 / 11M+ Episode / 70 任务 | 亚毫米光学动捕，含深度+触觉 | 预训练 |
| Dobot X-Trainer 遥操数据 | 300 demo × 4 任务（~18 小时） | Track-A 同本体微调 | SFT |
| JAKA K1 UMI 数据 | 300 UMI demo（1.2 小时） | Track-B 跨本体（固定底座臂） | SFT |
| Astribot S1 UMI 数据 | 200 UMI demo（1.5 小时） | Track-B 跨本体（人形） | SFT |
| RoboTwin 2.0 | 50 任务，每任务 100 rollout | 模拟基准测试 | 测试 |

### 实现细节

- **Backbone**: Hy-Embodied-0.5-MoT（4B 参数），共享联合注意力 + 分离模态 QKV/FFN
- **预训练步数**: 200K steps（全量 UMI 数据）
- **SFT 轨数**: Track-A（遥操数据，同本体）；Track-B（UMI 数据，跨本体）
- **FlowPRO**: 3 轮迭代，Batch 混合比例 Round 1: 80%/20%，后续 70%/15%/15%
- **控制频率**: 伺服率执行，异步推理并行

### 可视化结果

- 真实机器人 6 项双臂操作（Figure 9）：含精细任务（瓶子插入、拉链、USB、笔帽）
- 力觉引导物体区分（Figure 10）：Unitree G1 人形机器人
- Bézier 平滑对比（Figure 8）：块边界不连续性显著消除

---

## 批判性思考

### 优点

1. **全栈闭环**: 从硬件数据采集到真实部署一体化，减少各阶段脱节损耗。
2. **跨本体零额外遥操**: Track-B 仅用 UMI 数据即可迁移到形态各异的机器人，大幅降低新机器人部署成本。
3. **FlowPRO 高效率**: 无奖励模型、无价值函数，仅凭少量失败-成功对比轨迹将成功率推至 94–99%，且完成时间更短。

### 局限性

1. **不声称零样本泛化**: 当前数据规模尚不支持，仍需针对目标任务的少量演示数据。
2. **相机域差距**: UMI 自我中心相机与机器人挂载相机视角不同，可能影响泛化性能。
3. **动捕依赖**: 光学动捕系统限制了大规模野外数据采集的灵活性。
4. **执行效率**: 实时部署效率仍有提升空间。

### 潜在改进方向

1. 用外骨骼替代动捕系统，保留精度同时提高采集灵活性。
2. 系统性视觉增强研究，缩小 UMI 与机器人挂载相机的域差距。
3. 探索 zero-shot 涌现条件（类似 π₀.₇ 的规律）。

### 可复现性评估

- [ ] 代码开源（计划中）
- [ ] 预训练模型（计划中）
- [x] 训练细节完整（论文中详细描述）
- [ ] 数据集可获取（计划开源 2K 小时子集）

---

## 关联笔记

### 基于

- [[Mixture-of-Transformers|MoT]]: HyVLA-0.5 的主干架构基础
- [[Flow Matching|流匹配]]: 动作专家使用流匹配做连续动作预测
- [[DPO]]: FlowPRO 的偏好优化范式来源
- [[FastUMI]]: UMI 数据采集硬件的参考

### 对比

- [[π0]]: VLA 流匹配动作专家先驱，HyVLA-0.5 在仿真和跨本体上超越
- [[π0.5]]: 跨本体 VLA 对比基线，RoboTwin 2.0 上 HyVLA-0.5 高 8.2% (Clean)
- [[Motus]]: 仿真基准对比，HyVLA-0.5 超越 2.2%

### 方法相关

- [[Mixture-of-Transformers|MoT 架构]]: 核心主干，模态自适应计算
- [[Flow Matching|流匹配]]: 动作专家连续控制基础
- [[Delta-Chunk 动作表示]]: 跨本体迁移的关键解耦机制
- [[三次 Bézier 曲线]]: C¹ 连续动作拼接

### 硬件/数据相关

- [[FastUMI]]: UMI 采集硬件参考
- [[RoboTwin]]: 仿真评估基准

---

## 速查卡片

> [!summary] Hy-Embodied-0.5-VLA (HyVLA-0.5)
> - **核心**: 从数据到部署的端到端机器人学习全栈系统
> - **方法**: MoT 主干 + 流匹配动作专家 + Delta-Chunk 跨本体表示 + FlowPRO 无奖励 RL
> - **结果**: RoboTwin 2.0 90.9% (SOTA)，真实精细任务 94–99% 成功率
> - **代码**: 即将开源（含 2K 小时 UMI 子集）

---

*笔记创建时间: 2026-06-16*

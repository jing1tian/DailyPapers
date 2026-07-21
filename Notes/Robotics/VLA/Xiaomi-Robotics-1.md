---
title: "Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories"
method_name: "Xiaomi-Robotics-1"
authors: [Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, Nan Sun, Qiao Sun, Runze Suo, Heyun Wang, Yunhong Wang, Rujie Wu, Caoyu Xia, Lina Zhang, Jack Zhao, Guoliang Chen, Wenlong Chen, Xinze He, Bin Li, Qing Li, Zhuorong Li, Heng Qu, Wenxuan Song, Diyun Xiang, Yifan Xie, Peiran Xu, Hangjun Ye, Wen Ye, Han Zhao, Quanyun Zhou]
year: 2026
venue: arXiv
tags: [vla, scaling, flow-matching, mobile-manipulation, diffusion-transformer, mixture-of-transformers, data-scaling]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.15330v1
created: 2026-07-21
---

# 论文笔记：Xiaomi-Robotics-1

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Xiaomi Robotics Team |
| 日期 | July 2026 |
| 项目主页 | https://robotics.xiaomi.com/xiaomi-robotics-1.html |
| 对比基线 | [[RLDX-1]], [[Pi0.5]], [[GR00T-N1.6]], [[Xiaomi-Robotics-0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.15330) / [Code](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) |

---

## 一句话总结

> Xiaomi-Robotics-1 通过在 10 万小时 [[UMI]] 真实轨迹上进行大规模预训练，结合 [[Mixture-of-Transformers]]（[[Qwen3-VL]] + [[Diffusion Transformer|DiT]]）架构和两阶段训练范式，在四个仿真 Benchmark 上全面夺冠，并以不足 10 小时微调数据完成高难度下游任务。

---

## 核心贡献

1. **超大规模真实数据预训练**: 在 10 万小时以上 [[UMI]] 设备采集的真实操作轨迹上预训练 VLA，规模远超同类工作，验证了数据量是性能瓶颈
2. **自动标注流水线**: 利用 Qwen3.5-27B 将无标注轨迹片段自动标注为状态转换语言描述，大幅降低人工成本
3. **两阶段训练范式**: 预训练获取通用动作能力 → 后训练通过跨具身数据对齐到真实机器人和命令式指令，实现零样本泛化与高效微调

---

## 问题背景

### 要解决的问题

现有 VLA 模型受限于训练数据规模，无法在未见环境中零样本执行多样化语言指令引导的移动操作任务，且对新任务的适应需要大量标注数据。

### 现有方法的局限

- 大多数 VLA 方法训练数据量仅有数百到数千小时，性能受数据规模制约
- [[UMI]] 采集的轨迹数据缺乏语言标注，难以直接用于语言条件策略训练
- 预训练数据中的状态转换描述与人类自然使用的命令式指令之间存在域差距（"机械臂移动到瓶子旁" vs. "帮我倒水"）
- 跨具身迁移困难：从 UMI 夹具到真实机器人臂的动作空间差异

### 本文的动机

数据规模是 VLA 的首要瓶颈——作者假设用足够多的真实世界数据预训练，再用少量跨具身数据后训练，就能获得开箱即用的泛化能力。[[UMI]] 设备允许低成本大规模采集，配合自动标注流水线可将无标注轨迹转化为语言条件训练对。

---

## 方法详解

### 模型架构

Xiaomi-Robotics-1 采用 **[[Mixture-of-Transformers]]** 架构，将预训练 [[Qwen3-VL|VLM]] 与 [[Diffusion Transformer|DiT]] 耦合：

- **输入**: 语言指令 $l$（命令式）+ 多摄像头观测 $o_t$ + 机器人本体感知状态 $s_t$
- **Backbone（VLM 分支）**: [[Qwen3-VL]]，编码语言和图像，生成 KV Cache 供 DiT 使用，同时通过 [[Choice Policy|Choice Policies]] 输出辅助动作预测
- **动作分支（DiT）**: [[Diffusion Transformer]] 与 VLM 等层数，但 hidden size 更小以加速推理；通过 [[Flow Matching]] 从噪声中去噪生成 [[Action Chunking|动作块]] $a_{t:t+H}$
- **条件注入**: DiT 以 [[AdaLN]] 注入 [[Flow Matching|Flow-Matching]] 时间步 $\tau$，以交叉注意力注入 VLM KV Cache
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$（预测未来 H 步动作）
- **模型规模**: 2.6B / 5.1B / 10.5B 三档

### 核心模块

#### 模块1: 自动标注流水线（Auto-Labeling Pipeline）

**设计动机**: [[UMI]] 采集的轨迹无语言标注，人工标注 10 万小时数据不现实

**具体实现**:
- 将轨迹切分为固定长度片段
- 利用 Qwen3.5-27B 逐片段自动生成**状态转换描述**（"机械臂从左侧移向抽屉，打开抽屉"）
- 采用生产者-消费者流水线并行标注，约 2 周完成全量语料

#### 模块2: [[Choice Policy|Choice Policies]]（多模态动作选择）

**设计动机**: 轨迹数据天然存在多模态——同一语言指令对应多种合理动作序列，单一回归目标会导致均值模糊

**具体实现**:
- 将机器人状态 $s_t$ 用 MLP 编码为 token，追加到 VLM 序列末尾
- 在序列末尾追加 $K$ 组动作查询 token + $K$ 个分数查询 token
- VLM 预测 $K$ 个候选 [[Action Chunking|动作块]] $\tilde{a}^k_{t:t+H}$ 及其 L1 误差分数 $s_k = ||\tilde{a}^k_{t:t+H} - a_{t:t+H}||_1$
- **Winner-takes-all** 范式：选择分数最低的候选者参与梯度回传

#### 模块3: [[Flow Matching|Flow-Matching]] 动作生成（DiT 分支）

**设计动机**: 利用 [[Flow Matching]] 建模连续动作的分布，相比扩散模型推理步数更少

**具体实现**:
- 从噪声 $\varepsilon \sim \mathcal{N}(0, I)$ 出发，通过 5 步 Euler 积分去噪得到动作块
- 训练时构造中间状态 $\tilde{a}^\tau_{t:t+H} = \tau \cdot a_{t:t+H} + (1-\tau) \cdot \varepsilon$
- 从 VLM 动作预测中选出最佳候选 $\tilde{a}^*_{t:t+H}$ 作为 DiT Flow-Matching 的目标（不用 GT，增加多样性）
- 为降低 VLM 前向计算开销，每个 UMI 样本采样 4 个不同时间步共享同一 VLM KV Cache

#### 模块4: 两阶段训练（Pre-training + Post-training）

**预训练**: 大规模 [[UMI]] 数据 + 状态转换提示 → 通用动作生成能力

**后训练**: 跨具身数据（机器人臂 + 指令标注 UMI + 开源数据集）→ 与真实具身对齐 + 命令式指令理解

---

## 关键公式

### 公式1: [[Flow Matching|Flow-Matching 训练目标]]

$$
\mathcal{L}_{\text{Flow}}(\theta) = \left\| v_\theta(o_t, l, s_t, \tilde{a}^\tau_{t:t+H}, \tau) - u(\tilde{a}^\tau_{t:t+H}, a_{t:t+H}, \tau) \right\|_2^2
$$

**含义**: DiT 分支学习预测流速场 $v_\theta$，使噪声轨迹沿最优传输路径流向目标动作

**符号说明**:
- $v_\theta$: DiT 预测的速度场（向量场）
- $u(\cdot)$: 条件流的目标速度（简单线性插值的方向）
- $\tilde{a}^\tau_{t:t+H} = \tau \cdot a_{t:t+H} + (1-\tau) \cdot \varepsilon$: 时间步 $\tau$ 的中间轨迹
- $\tau \in [0, 0.999]$: Flow-Matching 时间步
- $\varepsilon \sim \mathcal{N}(0, I)$: 初始噪声

### 公式2: 时间步 Beta 分布采样

$$
u \sim \text{Beta}(1.5,\, 1), \quad \tau = (1 - u) \times 0.999
$$

**含义**: 对时间步 $\tau$ 施加偏向小值的非均匀采样（Beta 分布偏向 $u$ 接近 0 即 $\tau$ 接近 0.999），让模型更多地在去噪末段训练，有利于高频细节恢复

**符号说明**:
- $u$: Beta(1.5, 1) 采样值，偏向 0
- $\tau$: Flow-Matching 时间步，越大噪声越多

### 公式3: [[Choice Policy|Choice Policies]] 回归损失

$$
\mathcal{L}_{\text{Regression}}(\theta) = \left\| \tilde{a}^*_{t:t+H} - a_{t:t+H} \right\|_1 + \sum_{k=1}^{K} \left\| \hat{s}_k - s_k \right\|_2^2
$$

**含义**: VLM 分支同时学习预测最优动作候选（L1 重建损失）和各候选的排序分数（MSE 分数回归损失）

**符号说明**:
- $\tilde{a}^*_{t:t+H}$: VLM 预测的最优候选动作块
- $a_{t:t+H}$: GT 动作块
- $\hat{s}_k$: VLM 预测的第 $k$ 个候选的分数
- $s_k = ||\tilde{a}^k_{t:t+H} - a_{t:t+H}||_1$: 第 $k$ 个候选到 GT 的 L1 误差（用于监督分数预测）

### 公式4: 总训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{Flow}} + \mathcal{L}_{\text{Regression}} + \lambda \cdot \mathcal{L}_{\text{NTP}}
$$

**含义**: 联合优化 DiT 流匹配目标、VLM 动作回归目标和原始语言建模目标（保持 VLM 预训练能力）

**符号说明**:
- $\mathcal{L}_{\text{NTP}}$: 视觉-语言数据的下一 token 预测损失（Next-Token Prediction）
- $\lambda = 0.1$: NTP 损失权重
- VL 数据与 UMI 数据采样比例为 1:9

### 公式5: [[Flow Matching|Euler 积分推理]]

$$
a^{\tau+\Delta\tau}_{t:t+H} = a^\tau_{t:t+H} + \Delta\tau \cdot v_\theta(o_t, l, s_t, a^\tau_{t:t+H}, \tau), \quad \Delta\tau = 0.2
$$

**含义**: 推理时从纯噪声（$\tau=0$）出发，以步长 $\Delta\tau=0.2$ 执行 5 步 Euler 积分，逐步去噪到目标动作

**符号说明**:
- $a^0_{t:t+H} = \varepsilon \sim \mathcal{N}(0, I)$: 初始纯噪声
- $\Delta\tau = 0.2$: 积分步长（5 步到达 $\tau=1$）

### 公式6: 相对末端执行器位姿动作表示

$$
a_{t+i} = (T^{\text{EE}}_{\text{Base},t})^{-1} \hat{T}^{\text{EE}}_{\text{Base},t+i}
$$

**含义**: 动作以当前末端执行器坐标系下的**相对位姿变化**表示，实现跨具身动作空间的归一化

**符号说明**:
- $T^{\text{EE}}_{\text{Base},t}$: $t$ 时刻末端执行器在机器人基座坐标系中的变换矩阵
- $\hat{T}^{\text{EE}}_{\text{Base},t+i}$: 预测的 $t+i$ 时刻末端执行器位姿
- $(T^{\text{EE}}_{\text{Base},t})^{-1}$: 取当前坐标系作为参考，输出相对增量

---

## 关键图表

### Figure 1: Overview — 系统整体概览

![Figure 1: Xiaomi-Robotics-1 Overview](https://arxiv.org/html/2607.15330v1/x1.png)

**说明**: Xiaomi-Robotics-1 整体框架。(左) 预训练阶段：10 万小时 [[UMI]] 真实轨迹 + 自动标注状态转换描述；(中) 后训练阶段：跨具身数据对齐真实机器人；(右) 缩放曲线：数据量和模型规模均显示单调提升。

### Figure 2: Model Architecture — 模型架构

![Figure 2: Model Architecture](https://arxiv.org/html/2607.15330v1/x2.png)

**说明**: [[Mixture-of-Transformers]] 双分支架构。[[Qwen3-VL]] 处理观测+语言并通过 [[Choice Policy|Choice Policies]] 输出辅助动作预测；[[Diffusion Transformer|DiT]] 通过交叉注意力消费 VLM KV Cache，以 [[AdaLN]] 注入 $\tau$，输出 [[Action Chunking|动作块]]。两分支逐层耦合但参数独立。

### Figure 3: Pre-training Dataset — 预训练数据集

![Figure 3: Pre-training Dataset](https://arxiv.org/html/2607.15330v1/x3.png)

**说明**: 预训练数据集覆盖家庭、商业、工业、户外等多样环境，超过 10 万小时 [[UMI]] 真实操作轨迹，展示了前所未有的数据规模和场景多样性。

### Figure 4: Post-training Dataset — 后训练数据集

![Figure 4: Post-training Dataset](https://arxiv.org/html/2607.15330v1/x4.png)

**说明**: 约 1 万小时跨具身后训练数据构成：7.2k 小时移动操作臂/双臂机器人数据 + 1k 小时命令式标注 [[UMI]] 数据 + 开源数据集（Bridge V2、RT-1、DROID）。

### Figure 5: Scaling of Pre-training — 预训练缩放曲线

![Figure 5: Scaling Curves](https://arxiv.org/html/2607.15330v1/x5.png)

**说明**: 在 2 万小时子集上验证数据规模（12.5%→100%）和模型规模（2B/5B/10B）对验证集 MSE 的影响，两者均单调下降且未见饱和，表明**数据量是当前主要瓶颈**。

### Figure 6: Qualitative Results of Pre-training — 预训练定性结果

![Figure 6: Pre-training Qualitative](https://arxiv.org/html/2607.15330v1/x6.png)

**说明**: 预训练后模型在验证 [[UMI]] 数据上预测动作轨迹的可视化，展示了对多种抓取、放置动作的准确预测。

### Figure 7: Post-training Evaluation — 后训练评估场景

![Figure 7: Post-training Evaluation](https://arxiv.org/html/2607.15330v1/x7.png)

**说明**: 后训练后在 4 个未见环境中零样本评估：鞋子收纳、包包整理、桌面整理、沙发整理。所有场景均为模型从未见过的环境和物体。

### Figure 8: Quantitative Results of Post-training — 后训练定量结果

![Figure 8: Post-training Results](https://arxiv.org/html/2607.15330v1/x8.png)

**说明**: 不同预训练数据规模和模型尺寸下后训练的零样本成功率。无预训练基线 26%；12.5% 数据预训练提升至 53%；满量数据 5B 模型达 75%；10B 模型进一步达 79%。验证了预训练质量对后训练的直接传导。

### Figure 9: Downstream Fine-tuning Evaluation — 下游微调评估任务

![Figure 9: Downstream Fine-tuning Tasks](https://arxiv.org/html/2607.15330v1/x9.png)

**说明**: 4 个具有挑战性的下游微调任务：手机装盒（含说明书和盒盖）、衣物装洗衣机、打印机加纸、物品装箱。每任务少于 10 小时微调数据。

### Figure 10: Quantitative Results of Downstream Fine-tuning — 下游微调定量结果

![Figure 10: Downstream Results](https://arxiv.org/html/2607.15330v1/x10.png)

**说明**: 下游微调结果与 [[Pi0.5]] 和 [[Xiaomi-Robotics-0]] 的对比。Xiaomi-Robotics-1 在成功率和进度分数上均显著超越两者，平均成功率 75%。

### Figure 11: Examples of UMI data in Pre-training Dataset

![Figure 11: UMI Pre-training Examples](https://arxiv.org/html/2607.15330v1/x11.png)

**说明**: 预训练数据集中 [[UMI]] 数据样例，展示了状态转换描述标注方式（"夹具移向物体 → 抓取物体 → 将物体移至目标位置"）。

### Figure 12: Examples of UMI data in Post-training Dataset

![Figure 12: UMI Post-training Examples](https://arxiv.org/html/2607.15330v1/x12.png)

**说明**: 后训练 [[UMI]] 数据样例，与预训练的差异在于标注从状态转换描述切换为命令式指令（"将苹果放入篮子"），对齐人类自然指令风格。

### Table 1: 模型配置

| 模型 | VLM 层数 | VLM 隐藏维度 | VLM 参数量 | DiT 隐藏维度 | DiT 参数量 | 总参数量 |
|------|----------|-------------|-----------|-------------|-----------|---------|
| Xiaomi-Robotics-1-2B | 28 | 2048 | 2.1B | 1024 | 470M | **2.6B** |
| Xiaomi-Robotics-1-5B | 36 | 2560 | 4.4B | 1024 | 604M | **5.1B** |
| Xiaomi-Robotics-1-10B | 36 | 4096 | 8.8B | 2048 | 1.5B | **10.5B** |

**说明**: DiT 与 VLM 层数相同，但 hidden size 更小，实现计算效率与容量的平衡。

### Table 2: RoboCasa Benchmark 对比（平均成功率 %）

| 方法 | 平均成功率 (%) |
|------|---------------|
| UVA | 50.0 |
| UWM | 60.8 |
| π0.5 | 62.1 |
| π0-FAST | 63.6 |
| GR00T N1.6 | 66.2 |
| Cosmos Policy | 67.1 |
| RLDX-1 | 70.6 |
| World2Act | 72.6 |
| **Xiaomi-Robotics-1 (Ours)** | **74.5** |

**关键发现**: 超越此前最优 World2Act 约 2 个百分点，位居 RoboCasa 榜首。

### Table 3: RoboCasa365 Benchmark 对比

| 方法 | 平均 | Atomic | Comp.-Seen | Comp.-Unseen |
|------|------|--------|-----------|--------------|
| Diffusion Policy | 6.1 | 15.7 | 0.2 | 1.3 |
| π0.5 | 16.9 | 39.6 | 7.1 | 1.2 |
| GigaWorld-Policy 0.1 | 20.7 | 44.4 | 11.8 | 2.9 |
| GR00T-N1.6 | 21.9 | 51.1 | 9.4 | 1.7 |
| WorldDreamer | 35.3 | 66.3 | 26.7 | 9.0 |
| Qwen-RobotManip | 35.9 | 68.6 | 20.1 | 14.9 |
| RLDX-1 | 36.0 | 67.6 | 27.9 | 8.5 |
| ABot-M0.5 | 40.4 | 75.9 | 38.3 | 2.7 |
| ABot-M0.6 | 46.6 | 79.4 | 48.3 | 7.9 |
| **Xiaomi-Robotics-1 (Ours)** | **57.4** | **80.2** | **57.1** | **32.1** |

**关键发现**: 平均成功率超越此前 SOTA（ABot-M0.6）达 10.8 个百分点；Comp.-Unseen（未见组合任务）提升尤为显著（7.9% → 32.1%），体现强泛化能力。

### Table 4: VLABench Benchmark 对比

| 指标 | In-dist. SR | Cross-Cat. SR | Commonsense SR | Instruction SR | Texture SR | Avg. SR |
|------|-------------|---------------|----------------|----------------|------------|---------|
| Xiaomi-Robotics-1 | 75.6 | 53.0 | 48.4 | 55.8 | 62.6 | **59.1** |

**关键发现**: 在 VLABench 所有评测维度上取得当前最优，平均成功率 59.1%，平均进度分数 70.3%。

### Table 5: RoboDojo Benchmark 对比（分数 / 成功率 %）

| 方法 | Generalization | Precision | Long-Horizon | Memory | Open | 平均 |
|------|---------------|-----------|--------------|--------|------|------|
| GalaxeaVLA (G0) | 4.53/2.83 | 8.10/3.83 | 12.60/5.58 | 3.17/1.89 | 0.70/0.67 | 5.82/2.96 |
| π0.5 | 13.37/8.17 | 12.40/5.50 | 23.54/14.67 | 5.78/4.56 | 1.98/1.67 | 11.41/6.91 |
| Hy-Embodied-0.5-VLA | 11.77/8.39 | 13.81/8.00 | 25.74/14.92 | 13.37/12.11 | 0.65/0.58 | 13.07/8.80 |
| **Xiaomi-Robotics-1** | **23.55/17.00** | **26.69/18.83** | **38.39/23.67** | **7.81/6.56** | **3.94/3.58** | **20.07/13.93** |

**关键发现**: 平均分数超前 SOTA 达 7 分（20.07 vs. 13.07）；但 Memory 维度（7.81 vs. 13.37）落后于 Hy-Embodied，说明长时记忆能力有提升空间。

### Table 6: 下游任务进度里程碑定义

| 任务 | 里程碑 | 进度权重分配 |
|------|--------|------------|
| 手机装盒 | 抓手机→放手机→抓说明书→放说明书→抓盒盖→盖盖子 | 10, 10, 30, 10, 10, 30 (%) |
| 打印机加纸 | 抓纸→完成传递→插纸→完全插入→回位 | 20, 20, 20, 30, 10 (%) |
| 衣物装洗衣机 | 开门→移篮子→转移衣物→移走篮子→关门 | 各 20 (%) |
| 物品装箱 | 按语言指令抓取/放置 5 个物品 | 各 20 (%) |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| UMI 预训练数据集 | >100k 小时 | 真实世界多环境轨迹，自动标注状态转换描述 | 预训练 |
| 机器人后训练数据（内部） | 7.2k 小时 | 移动操作臂 + 双臂机器人，命令式指令标注 | 后训练 |
| 命令式标注 UMI 数据 | 1k 小时 | 带命令式指令的 UMI 数据 | 后训练 |
| Bridge V2 / RT-1 / DROID | 开源 | 跨具身开源数据集 | 后训练补充 |
| RoboCasa | - | 仿真 Benchmark | 测试 |
| RoboCasa365 | - | 仿真 Benchmark，含组合泛化 | 测试 |
| VLABench | - | 仿真 Benchmark，多维度评估 | 测试 |
| RoboDojo | - | 仿真 Benchmark，5 类能力 | 测试 |

### 实现细节

- **VLM Backbone**: [[Qwen3-VL]]（2.1B / 4.4B / 8.8B 三档）
- **动作 Backbone**: [[Diffusion Transformer|DiT]]（470M / 604M / 1.5B）
- **Flow-Matching 步数（推理）**: 5 步 Euler 积分
- **时间步范围**: $\tau \in [0, 0.999]$，Beta(1.5, 1) 采样
- **后训练数据采样比**: VL : 开源机器人 : 指令 UMI : 内部机器人 = 0.5 : 0.5 : 0.5 : 8.5
- **预训练 VL:UMI 比**: 1:9
- **NTP 损失权重** $\lambda$: 0.1
- **VLM KV Cache 分时复用**: 每个 UMI 样本采样 4 个时间步共享一次 VLM 前向

### 可视化结果

- 零样本实验：在 4 个全新家庭环境（鞋架、沙发、桌面、包包场景）中，10.5B 模型平均成功率 79%
- 下游微调：每任务 <10 小时数据，平均成功率 75%，进度分数更高
- 室内自主演示：端到端完成超 10 分钟的行李箱自主整理任务

---

## 批判性思考

### 优点
1. **数据规模前所未有**: 10 万小时真实轨迹远超同类工作（如 π0 约 1 万小时），提供了可信的 Scaling Law 验证
2. **自动标注突破瓶颈**: 利用 LLM 自动标注状态转换，解决了大规模无标注轨迹的利用问题，具有实际工程价值
3. **全面 SOTA**: 在四个独立 Benchmark（RoboCasa、RoboCasa365、VLABench、RoboDojo）上全面领先，泛化性有充分验证

### 局限性
1. **数据采集设备依赖**: 依赖 [[UMI]] 设备大规模采集，推广需要小米特有的设备和渠道，复现成本极高
2. **Memory 任务表现相对较差**: RoboDojo Memory 维度落后于 Hy-Embodied（6.56% vs. 12.11%），长时记忆能力有明显短板
3. **跨具身迁移代价**: 状态转换预训练提示到命令式后训练的域差距，需要 10k 小时后训练数据，仍有一定成本
4. **模型闭源**（代码暂未完全开源）：数据集不公开，研究社区难以直接复现

### 潜在改进方向
1. 引入长时记忆机制（如[[记忆注意力]]、外部记忆库）解决 Memory 维度短板
2. 研究更高效的跨具身迁移方法，减少后训练数据需求
3. 探索在线强化学习方法进一步提升下游任务成功率

### 可复现性评估
- [x] 代码开源（GitHub: XiaomiRobotics/Xiaomi-Robotics-1）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（论文详细描述超参数和流程）
- [ ] 数据集可获取（10 万小时内部数据不公开）

---

## 关联笔记

### 基于
- [[Qwen3-VL]]: 视觉语言骨干网络
- [[Mixture-of-Transformers]]: 整体架构设计范式
- [[Flow Matching]]: DiT 分支的动作生成方法
- [[UMI]]: 数据采集设备

### 对比
- [[RLDX-1]]: RoboCasa 上的主要竞争者（70.6% vs. 74.5%）
- [[Pi0.5]]: 多个 Benchmark 的对比基线（π0.5 用相近架构但数据规模更小）
- [[GR00T-N1.6]]: NVIDIA 的 VLA 基线（66.2% on RoboCasa）
- [[Xiaomi-Robotics-0]]: 小米前代模型，下游微调对比基线

### 方法相关
- [[Choice Policy|Choice Policies]]: 多模态动作预测机制
- [[Diffusion Transformer]]: DiT 动作生成分支
- [[Action Chunking]]: 动作块输出格式
- [[AdaLN]]: 时间步条件注入方式

### 硬件/数据相关
- [[UMI]]: Universal Manipulation Interface，数据采集设备

---

## 速查卡片

> [!summary] Xiaomi-Robotics-1
> - **核心**: 10 万小时 UMI 数据预训练 + 两阶段训练范式的 VLA Scaling Study
> - **方法**: [[Mixture-of-Transformers]]（[[Qwen3-VL]] + [[Diffusion Transformer|DiT]] + [[Choice Policy|Choice Policies]]）
> - **结果**: RoboCasa 74.5% / RoboCasa365 57.4% / VLABench 59.1% / RoboDojo 20.07 — 全面 SOTA
> - **代码**: https://github.com/XiaomiRobotics/Xiaomi-Robotics-1

---

*笔记创建时间: 2026-07-21*

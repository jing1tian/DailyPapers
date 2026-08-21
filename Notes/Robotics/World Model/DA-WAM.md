---
title: "DA-WAM: Decision-Aligned Future Latents for Driving World Models"
method_name: "DA-WAM"
authors: [Ruiguo Zhong, Benshan Ma, Xiaolong Chen, Lang Zhang, Mingyue Feng, Yaonong Wang, Pei Liu, Jun Ma]
year: 2026
venue: arXiv
tags: [autonomous-driving, world-model, trajectory-planning, jepa, action-conditioned, contrastive-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.19085
created: 2026-08-21
---

# 论文笔记：DA-WAM: Decision-Aligned Future Latents for Driving World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The Hong Kong University of Science and Technology (Guangzhou); Leapmotor; HKUST |
| 日期 | August 2026 |
| 项目主页 | – |
| 对比基线 | [[DiffusionDrive]], [[TransFuser]], DriveSuprim, DrivoR |
| 链接 | [arXiv](https://arxiv.org/abs/2608.19085) / [Code](https://github.com/LeapWM/da-wam) |

---

## 一句话总结

> DA-WAM 通过为每条候选轨迹生成专属的未来 latent 表示，并将其显式注入轨迹评分器，实现了预测与规划的决策对齐，在 [[NAVSIM]] 两代 benchmark 上达到 SOTA。

---

## 核心贡献

1. **决策对齐的未来 latent 学习**: 为每条候选轨迹独立预测一个动作条件化的未来 latent，不再共享，保证预测结果与对应决策后果一一对应
2. **预测-规划协同训练框架**: 利用 [[V-JEPA]] 2.1 预训练编码器 + [[LoRA]] 适配 + [[EMA|指数移动平均 (EMA)]] 目标网络，使 latent 空间在规划任务中持续演化而非保持冻结
3. **安全关键反事实负样本监督**: 从离线数据库检索与专家轨迹几何接近但安全性显著更差的负样本，强迫打分器区分决策边界附近的轨迹

---

## 问题背景

### 要解决的问题

自动驾驶规划器需要从 N 条候选轨迹中选出最优的一条。现有 [[World Model|世界模型]] 方法虽然能预测未来场景，但预测结果并没有直接影响轨迹选择，导致"预测"与"规划"之间存在断层。

### 现有方法的局限

- **轨迹专用预测**（图 1a）：仅对选中轨迹做预测，与规划解耦
- **松散耦合 latent 融合**（图 1b）：预测输出与轨迹评分器拼接，但梯度路径不统一
- **共享全局未来 latent**（图 1c）：所有候选共用同一预测，无法体现不同动作的不同后果
- 上述方法要么表示学习与规划优化脱节，要么稀释了动作特定的预测信号

### 本文的动机

将预测学习、动作条件化建模、轨迹评分三者统一到同一目标下，使每条轨迹的专属未来 latent 直接驱动该轨迹的得分计算——预测即规划，规划反哺预测。

---

## 方法详解

### 模型架构

DA-WAM 采用三阶段管线：

- **输入**: 当前帧 $X_t$（前向摄像头，含两帧历史）+ N 条候选轨迹 $\mathcal{T}=\{\tau_i\}_{i=1}^N$
- **Backbone**: [[V-JEPA]] 2.1（[[LoRA]] 适配在线分支，目标分支通过 [[EMA|指数移动平均]] 更新）
- **核心模块**: [[Action-Conditioned World Model|动作条件化未来预测]] → [[Transformer|Cross-Attention 评分器]]
- **输出**: 每条轨迹的效用分 $\hat{s}_i$，选分最高者 $\tau^*$

### 核心模块

#### 模块 1: JEPA 驱动的预测表示适配

**设计动机**: 利用 [[V-JEPA]] 的自监督视觉表示，同时避免 fine-tuning 破坏预训练知识

**具体实现**:
- 在线编码器 $E_\theta$：将当前帧映射到场景 latent token $Z_t \in \mathbb{R}^{M \times D}$，[[LoRA]] 模块注入 [[Transformer]] 各层，基础网络参数冻结
- 目标编码器 $E_{\bar{\theta}}$：对观测到的未来帧 $X_{t+\Delta}$ 提取特征，配合 [[Stop-Gradient|stop-gradient]] 和 [[EMA|EMA]] 更新
- 预测时域：未来 0.5 秒

#### 模块 2: 动作条件化反事实世界建模

**设计动机**: 为离线数据约束下无法观察到所有候选结果的问题提供监督信号

**具体实现**:
- 轨迹编码器 $E_\tau$ 将每条候选轨迹 $\tau_i$（包含 8 个未来 ego pose）编码为动作表示 $a_i$
- 共享预测器 $P_\phi$ 以 $a_i$ 为 Query，当前场景 token $Z_t$ 为 Key/Value，生成每条轨迹专属的反事实 latent $\hat{Z}_i$
- 仅对专家匹配候选（$i^\text{exp}$）施加预测监督；其余候选通过下游规划损失获得间接优化

#### 模块 3: 未来 latent 条件化轨迹评分

**设计动机**: 显式将每条轨迹的预测后果注入其评分，实现"预测对齐决策"

**具体实现**:
- 评分 [[Transformer]] 对当前场景 token、动作表示、预测未来 latent 三者做 Cross-Attention，输出统一表示 $h_i$
- 可解释规划因子头（分解预测）：NC（无过失碰撞）、DAC（可行驶区域合规）、EP（自车进度）、TTC（碰撞时间）、Comfort（舒适度）
- 效用头聚合特征和因子向量，输出最终排序分 $\hat{s}_i$

#### 模块 4: 轨迹级反事实安全监督

**设计动机**: 离线数据中安全负样本稀疏，难以学习决策边界

**具体实现**:
- 从离线轨迹库中检索满足双重约束的硬负样本 $\tau_j^-$：几何上接近专家轨迹（$d_\text{traj}$ 小），但安全评分显著劣化（$\Delta_\text{safety}$ 大）
- 与专家轨迹组成对比对，纳入排序损失训练

---

## 关键公式

### 公式 1: [[V-JEPA|在线编码器]]

$$
Z_t = E_\theta(X_t)
$$

**含义**: 在线编码器将当前观测帧映射为 $M$ 个维度 $D$ 的场景 latent token

**符号说明**:
- $X_t$: 当前帧视觉输入（含两帧历史）
- $Z_t \in \mathbb{R}^{M \times D}$: 场景 latent token 序列

---

### 公式 2: [[Stop-Gradient|目标编码器（含 stop-gradient）]]

$$
Z_{t+\Delta} = \operatorname{sg}(E_{\bar{\theta}}(X_{t+\Delta}))
$$

**含义**: 目标编码器处理观测到的未来帧，stop-gradient 切断梯度回传，确保目标 latent 稳定

**符号说明**:
- $X_{t+\Delta}$: 观测到的未来帧（真实）
- $\operatorname{sg}(\cdot)$: stop-gradient 算子
- $E_{\bar{\theta}}$: 目标编码器（参数由 EMA 更新）
- $Z_{t+\Delta}$: 未来 latent 回归目标

---

### 公式 3: [[EMA|指数移动平均 (EMA) 参数更新]]

$$
\bar{\theta} \leftarrow \mu\bar{\theta} + (1-\mu)\theta
$$

**含义**: 目标编码器参数由在线编码器参数的指数移动平均维护，提供稳定的监督目标

**符号说明**:
- $\mu \in [0, 1)$: 动量系数（控制目标编码器更新速度）
- $\theta$: 在线编码器可训练参数
- $\bar{\theta}$: 目标编码器参数

---

### 公式 4: 动作表示编码

$$
a_i = E_\tau(\tau_i)
$$

**含义**: 轨迹编码器将候选轨迹编码为动作查询向量

**符号说明**:
- $\tau_i$: 第 $i$ 条候选轨迹（8 个未来 ego pose）
- $a_i$: 对应的动作表示

---

### 公式 5: [[Action-Conditioned World Model|动作条件化未来预测]]

$$
\hat{Z}_i = P_\phi(Q=a_i,\; K=Z_t,\; V=Z_t), \quad i=1,\ldots,N
$$

**含义**: 共享预测器以轨迹动作为 Query、场景 token 为 Key/Value，生成候选专属反事实 latent

**符号说明**:
- $P_\phi$: 共享预测器（Cross-Attention）
- $\hat{Z}_i$: 第 $i$ 条轨迹的预测未来 latent

---

### 公式 6: 专家匹配

$$
i^\text{exp} = \arg\min_i \operatorname{ADE}(\tau_i, \tau^\text{exp})
$$

**含义**: 选取与专家轨迹平均位移误差最小的候选，作为接受预测监督的目标

**符号说明**:
- $\tau^\text{exp}$: 专家（标注）轨迹
- $\operatorname{ADE}$: 平均位移误差
- $i^\text{exp}$: 匹配专家的候选索引

---

### 公式 7: 预测损失

$$
\mathcal{L}_\text{pred} = \frac{1}{M}\sum_{m=1}^{M} \ell\!\left(\hat{Z}_{i^\text{exp},m},\; Z_{t+\Delta,m}\right)
$$

**含义**: 仅对专家匹配候选施加 token 级别的特征回归监督

**符号说明**:
- $M$: latent token 数量
- $\ell(\cdot, \cdot)$: 特征级别回归损失（如平滑 L1）
- $\hat{Z}_{i^\text{exp},m}$: 专家匹配候选第 $m$ 个 token 的预测
- $Z_{t+\Delta,m}$: 观测未来帧第 $m$ 个 token 的目标

---

### 公式 8: 评分编码器

$$
h_i = S_\psi^\text{enc}(Z_t,\; \hat{Z}_i,\; a_i)
$$

**含义**: Cross-Attention 评分器融合当前场景、预测未来、动作表示，输出统一轨迹表示

**符号说明**:
- $h_i$: 第 $i$ 条轨迹的综合表示向量
- $S_\psi^\text{enc}$: 编码器（Cross-Attention Transformer）

---

### 公式 9: 分解规划因子头

$$
\hat{q}_i = S_\psi^\text{factor}(h_i) = [\hat{q}_i^\text{NC},\; \hat{q}_i^\text{DAC},\; \hat{q}_i^\text{EP},\; \hat{q}_i^\text{TTC},\; \hat{q}_i^\text{Comfort}]
$$

**含义**: 线性头将轨迹表示解码为可解释的规划因子向量

**符号说明**:
- NC: 无过失碰撞（No-at-fault Collision）
- DAC: 可行驶区域合规（Drivable Area Compliance）
- EP: 自车进度（Ego Progress）
- TTC: 碰撞时间（Time To Collision）
- Comfort: 舒适度

---

### 公式 10: 效用分计算

$$
\hat{s}_i = S_\psi^\text{score}(h_i,\; \hat{q}_i)
$$

**含义**: 综合轨迹表示和规划因子，输出最终排序得分

---

### 公式 11a-b: 硬负样本选择约束

$$
d_\text{traj}(\tau_j^-, \tau^\text{exp}) < \varepsilon_\text{geo}
$$

$$
\Delta_\text{safety}(\tau_j^-, \tau^\text{exp}) > \varepsilon_\text{safety}
$$

**含义**: 负样本必须同时满足"几何上接近专家"和"安全上显著劣化"两个条件

**符号说明**:
- $\tau_j^-$: 第 $j$ 条硬负样本轨迹
- $\varepsilon_\text{geo}$: 几何接近阈值
- $\varepsilon_\text{safety}$: 安全劣化阈值
- $\Delta_\text{safety}$: 安全评分差异

---

### 公式 12: 规划因子损失

$$
\mathcal{L}_\text{factor} = \sum_i \sum_{k \in \mathcal{K}} \lambda_k \ell_k(\hat{q}_i^k, q_i^k)
$$

**含义**: 对所有候选轨迹的所有规划因子施加加权监督（连续量 MSE，二元量 BCE）

**符号说明**:
- $\mathcal{K}$: 受监督的规划因子集合
- $\lambda_k$: 第 $k$ 个因子的损失权重
- $q_i^k$: 第 $i$ 条轨迹第 $k$ 个因子的真实值

---

### 公式 13: 效用分损失

$$
\mathcal{L}_\text{score} = \sum_i \ell_\text{score}(\hat{s}_i, s_i)
$$

**含义**: 对最终效用分施加直接监督

---

### 公式 14: 偏好对构造

$$
y_{ij} = \mathbb{1}[s_i > s_j]
$$

**含义**: 基于真实效用分构造二元偏好标签，用于 [[Pairwise Ranking Loss|成对排序损失]]

---

### 公式 15: [[Pairwise Ranking Loss|成对排序损失]]

$$
\mathcal{L}_\text{rank} = -\sum_{(i,j)} \Bigl[y_{ij}\log\sigma(\hat{s}_i - \hat{s}_j) + (1 - y_{ij})\log\sigma(\hat{s}_j - \hat{s}_i)\Bigr]
$$

**含义**: 通过 sigmoid 差值构造成对 [[Contrastive Loss|对比排序]] 损失，重点强调硬负样本对

**符号说明**:
- $\sigma(\cdot)$: sigmoid 函数
- $(i, j)$: 轨迹对（含专家-负样本对）

---

### 公式 16: 总训练目标

$$
\mathcal{L} = \lambda_\text{pred}\mathcal{L}_\text{pred} + \lambda_\text{factor}\mathcal{L}_\text{factor} + \lambda_\text{score}\mathcal{L}_\text{score} + \lambda_\text{rank}\mathcal{L}_\text{rank}
$$

**含义**: 四项损失加权组合，统一预测对齐目标与规划优化目标

**符号说明**:
- $\lambda_\text{pred}, \lambda_\text{factor}, \lambda_\text{score}, \lambda_\text{rank}$: 各项权重（论文中未公开具体值）

---

### 公式 17: 推理轨迹选择

$$
\tau^* = \arg\max_{\tau_i \in \mathcal{T}} \hat{s}_i
$$

**含义**: 部署时选择效用分最高的候选轨迹执行

---

## 关键图表

### Figure 1: Prediction-Action Alignment Comparison / 预测-动作对齐方案对比

![Figure 1](https://arxiv.org/html/2608.19085v2/Figures/pipeline_compare.png)

**说明**: 对比四种方案：(a) 仅轨迹预测、不参与评分；(b) 松散融合预测 latent；(c) 所有候选共享同一未来 latent；(d) **DA-WAM** 每条候选独享一个预测 latent，并用于条件化各自的打分，实现一一对应的决策对齐。

---

### Figure 2: DA-WAM Overview / 系统总览

![Figure 2](https://arxiv.org/html/2608.19085v2/Figures/overview2.png)

**说明**: 三阶段框架示意图。第一阶段：[[V-JEPA]] 在线编码器将当前帧编码为场景 token，[[Stop-Gradient|stop-gradient]] + [[EMA]] 目标编码器提取未来帧 latent。第二阶段：动作条件化预测器为每条候选轨迹生成专属反事实 latent。第三阶段：评分 [[Transformer]] 显式以预测 latent 为条件，输出规划因子和效用分。

---

### Figure 3: Safety-Critical Hard-Negative Supervision / 安全关键硬负样本监督

![Figure 3](https://arxiv.org/html/2608.19085v2/Figures/counterfactual_trajectory_supervision2.png)

**说明**: 展示如何从离线轨迹库中检索满足"几何接近 + 安全劣化"双约束的负样本轨迹，与专家轨迹构成对比对，迫使打分器在决策边界附近正确区分安全性差异。

---

### Figure 4: Qualitative Comparison / 定性对比

![Figure 4](https://arxiv.org/html/2608.19085v2/camera_bev_score_comparison_32.png)

**说明**: 三个典型场景中，DA-WAM 与 [[DiffusionDrive]]、DrivoR 的相机视图 + [[BEV]] 轨迹 + 指标分数对比。DA-WAM 在无障碍转弯中保持进度，在交通冲突场景中优先保障安全。

---

### Table 1: NAVSIM-v1 Benchmark（仅摄像头方法）

| Method | Venue | NC | DAC | TTC | Comfort | EP | PDMS |
|--------|-------|-----|-----|-----|---------|-----|------|
| PDM-Closed | CoRL'23 | 94.6 | 99.8 | 89.9 | 86.9 | 99.9 | 89.1 |
| UniVLA | ICLR'26 | 96.9 | 91.1 | 91.7 | 96.7 | 76.8 | 81.7 |
| DrivingGPT | ICCV'25 | 98.9 | 90.7 | 94.9 | 95.6 | 79.7 | 82.4 |
| UniAD | CVPR'23 | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| World4Drive | ICCV'25 | 97.4 | 94.3 | 92.8 | 100.0 | 79.9 | 85.1 |
| VAD-v2 | ICLR'26 | 98.1 | 94.8 | 94.3 | 100.0 | 80.6 | 86.2 |
| PRIX | RA-L'26 | 98.1 | 96.3 | 94.1 | 100.0 | 82.3 | 87.8 |
| DiffusionDrive | CVPR'25 | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| AutoVLA | NeurIPS'25 | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | ICLR'26 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| ReCogDrive | ICLR'26 | 97.9 | 97.3 | 94.9 | 100.0 | 87.3 | 90.8 |
| Hydra-MDP++ | arXiv'25 | 98.6 | 98.6 | 95.1 | 100.0 | 85.7 | 91.0 |
| DiffusionDriveV2 | arXiv'25 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| iPad | arXiv'25 | 98.6 | 98.3 | 94.9 | 100.0 | 88.0 | 91.7 |
| SparseDriveV2 | arXiv'26 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | 92.0 |
| Centaur | arXiv'25 | 99.5 | 98.9 | 98.0 | 100.0 | 85.9 | 92.6 |
| DrivoR | CVPR'26 | 98.9 | 98.3 | 96.2 | 100.0 | 89.1 | 93.1 |
| DriveSuprim | AAAI'26 | 98.6 | 98.6 | 95.5 | 100.0 | 91.3 | 93.5 |
| Human driver | NeurIPS'24 | 100.0 | 100.0 | 100.0 | 99.9 | 87.5 | 94.8 |
| **DA-WAM** | **–** | **99.1** | **98.9** | **96.8** | **99.8** | **90.0** | **93.7** |

**表格说明**: DA-WAM 达到 93.7 [[PDMS|PDMS]]，超越 DriveSuprim（93.5）和 DrivoR（93.1），NC 和 DAC 指标尤为突出。

---

### Table 2: NAVSIM-v2 Benchmark

| Method | Backbone | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | EPDMS |
|--------|----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|--------|
| Ego Status MLP | ResNet-34 | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| TransFuser | ResNet-34 | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| Hydra-MDP++ | ResNet-34 | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim | ResNet-34 | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS | ResNet-34 | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | 98.3 | 83.1 |
| DiffusionDriveV2 | ResNet-34 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| SparseDriveV2 | ResNet-34 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 86.7 |
| Hydra-MDP++ | ViT/L | 98.4 | 98.0 | 99.4 | 99.8 | 87.5 | 97.7 | 95.3 | 98.3 | 77.4 | 85.1 |
| DriveSuprim | ViT/L | 97.8 | 97.9 | 99.5 | 99.9 | 90.6 | 97.1 | 96.6 | 98.3 | 77.9 | 86.0 |
| **DA-WAM** | **ViT/L** | **98.4** | **98.4** | **99.1** | **99.9** | **88.6** | **97.9** | **97.6** | **97.8** | **79.6** | **87.7** |

**表格说明**: NAVSIM-v2 指标更细粒度（新增 DDC、TL、LK、HC、EC），DA-WAM 达到 87.7 EPDMS，TTC（97.9）和车道保持（LK: 97.6）尤为出色。

---

### Table 3: 未来预测配置消融（NAVSIM-v1）

| 配置 | 硬负样本 | PDMS | NC | DAC | EP | TTC | Comfort |
|------|----------|------|-----|-----|-----|-----|---------|
| 无未来预测 | – | 93.31 | 98.45 | 98.27 | 91.36 | 95.48 | 99.99 |
| 共享全局 latent | – | 92.81 | 99.02 | 98.46 | 88.68 | 96.54 | 99.99 |
| 当前 latent 条件化 | – | 93.25 | 98.44 | 98.19 | 91.38 | 95.49 | 99.94 |
| 动作条件化未来 | ✗ | 93.46 | 98.88 | 98.58 | 90.47 | 96.33 | 99.69 |
| 动作条件化未来 | ✓ | **93.68** | **99.11** | **98.88** | 89.97 | **96.81** | 99.77 |

**关键发现**: 共享 latent（92.81）反而不如不预测（93.31），证明共享会稀释动作特异性信号；硬负样本带来 +0.22 PDMS 提升，主要体现在 NC 和 TTC。

---

### Table 4: 预测表示适配消融（NAVSIM-v1）

| 适配方式 | 密集损失 | 目标编码器 | PDMS |
|----------|----------|-----------|------|
| 冻结 | ✗ | 冻结 | 91.26 |
| 冻结 | ✓ | 冻结 | 91.95 |
| [[LoRA]] | ✗ | 冻结 | 92.74 |
| [[LoRA]] | ✓ | 冻结 | 92.98 |
| 全参数微调 | ✓ | 冻结 | 92.62 |
| [[LoRA]] | ✓ | 独立分支 | 93.10 |
| [[LoRA]] | ✓ | 共享分支 | 93.34 |
| [[LoRA]] | ✓ | [[EMA|EMA]] | **93.68** |

**关键发现**: LoRA 适配比全参数微调更优（92.98 vs 92.62），说明保留预训练知识至关重要；EMA 目标编码器在所有目标策略中最优。

---

### Table 5: 候选轨迹数量消融（NAVSIM-v1）

| 候选数量 | 1 | 8 | 16 | 32 | 64 |
|---------|---|---|----|----|-----|
| PDMS | 87.11 | 90.76 | 91.89 | **93.68** | 93.68 |

**关键发现**: 性能随候选数量持续提升，至 32 条时饱和（64 条无额外收益），最终选定 32 条候选。

---

### Table 6: 符号说明

| 符号 | 含义 |
|------|------|
| $X_t, X_{t+\Delta}$ | 当前帧与观测未来帧视觉输入 |
| $\mathcal{T}=\{\tau_i\}_{i=1}^N$ | N 条候选自车轨迹集合 |
| $\tau^\text{exp}, \tau_j^-, \tau^*$ | 专家、硬负样本、选定轨迹 |
| $E_\theta, E_{\bar{\theta}}$ | 在线编码器与 EMA 目标编码器 |
| $Z_t, Z_{t+\Delta}$ | 当前场景 latent 与未来 latent 目标 |
| $a_i = E_\tau(\tau_i)$ | 候选 $\tau_i$ 的动作表示 |
| $\hat{Z}_i = P_\phi(Z_t, a_i)$ | 候选 $\tau_i$ 的预测未来 latent |
| $S_\psi$ | 未来 latent 条件化轨迹打分器 |
| $i^\text{exp}$ | 与专家轨迹匹配的候选索引 |
| $\mathbf{q}_i, \hat{\mathbf{q}}_i$ | 真实与预测规划因子向量 |
| $s_i, \hat{s}_i$ | 真实与预测轨迹效用分 |
| $M, D$ | latent token 数量与维度 |
| $\mu$ | EMA 目标编码器动量系数 |
| $\mathcal{K}$ | 受监督规划因子集合 |
| $\lambda_\cdot$ | 各损失项权重系数 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[NAVSIM]] v1 navtest | 12,146 场景 | 真实传感器数据，前向摄像头 | 主要测试（PDMS） |
| [[NAVSIM]] v2 | 扩展场景 | 新增 DDC/TL/LK/HC/EC 指标 | 二次测试（EPDMS） |

### 实现细节

- **视觉 Backbone**: [[V-JEPA]] 2.1（ViT/L），[[LoRA]] 适配在线编码器，目标编码器 [[EMA]] 更新
- **候选生成**: 提案模块生成 32 条候选轨迹，每条含 8 个未来 ego pose
- **预测时域**: 0.5 秒
- **输入帧**: 当前帧 + 1 帧历史（前向摄像头）
- **训练**: 20 epochs，8 GPU，batch size 每卡 8（共 64）
- **损失**: $\mathcal{L}_\text{pred} + \mathcal{L}_\text{factor} + \mathcal{L}_\text{score} + \mathcal{L}_\text{rank}$（各项权重未公开）

### 可视化结果

三个典型场景定性分析：无障碍转弯场景中 DA-WAM 维持较高自车进度；交叉路口冲突场景中优先规避碰撞风险；与 [[DiffusionDrive]]、DrivoR 对比，DA-WAM 在 TTC 和 NC 指标上更稳健。

---

## 批判性思考

### 优点

1. **设计直觉清晰**: 一对一的 latent-轨迹对应关系在概念上自洽，消融实验也验证了共享 latent 确实有害
2. **预训练保留策略合理**: [[LoRA]] + 冻结基础网络 + [[EMA]] 目标编码器组合比全参数微调更优，说明对预训练知识的保护有价值
3. **可解释性好**: 分解规划因子头输出 NC/DAC/EP/TTC/Comfort，便于诊断失败案例

### 局限性

1. **离线数据约束**: 仅专家匹配候选有预测监督，其余候选的反事实 latent 质量依赖间接优化，存在潜在的监督不足问题
2. **图像来源单一**: 仅使用前向单目摄像头，缺乏多摄像头或激光雷达融合
3. **超参数不透明**: 各损失项权重 $\lambda$ 未公开，复现存在障碍

### 潜在改进方向

1. 扩展到多摄像头 / 多模态传感器融合
2. 将反事实 latent 预测监督推广到非专家候选（如通过数据增强构造更多在线"观测"）
3. 探索 latent 空间聚类以减少冗余候选

### 可复现性评估

- [ ] 代码开源（GitHub 已公开：https://github.com/LeapWM/da-wam）
- [ ] 预训练模型（待确认）
- [ ] 训练细节（损失权重未公开）
- [x] 数据集可获取（NAVSIM 公开）

---

## 关联笔记

### 基于

- [[V-JEPA]]: 视觉预测自监督预训练基础，DA-WAM 通过 LoRA 适配其在线分支
- [[JEPA]]: Joint-Embedding Predictive Architecture 框架思想来源
- [[Action-Conditioned World Model]]: DA-WAM 的核心建模范式
- [[World Model]]: 更广泛的背景概念

### 对比

- [[DiffusionDrive]]: NAVSIM-v1 主要对比 baseline（88.1 PDMS vs 93.7）
- [[TransFuser]]: NAVSIM-v2 中 ResNet-34 baseline（76.7 EPDMS vs 87.7）

### 方法相关

- [[LoRA]]: 在线编码器适配方式
- [[EMA|指数移动平均 (EMA)]]: 目标编码器更新策略
- [[Stop-Gradient]]: 防止梯度流入目标编码器
- [[Pairwise Ranking Loss]]: 轨迹排序损失
- [[Contrastive Loss]]: 硬负样本对比学习思想

### 数据集相关

- [[NAVSIM]]: 主要评测 benchmark（v1 + v2）
- [[BEV]]: 可视化和轨迹规划的视角

---

## 速查卡片

> [!summary] DA-WAM
> - **核心**: 为每条候选轨迹生成专属反事实未来 latent，显式条件化打分，实现预测-规划决策对齐
> - **方法**: V-JEPA 2.1 + LoRA + EMA 在线-目标编码器 + 动作条件化预测器 + 未来 latent 条件化打分器 + 硬负样本排序监督
> - **结果**: NAVSIM-v1 93.7 PDMS（SOTA）；NAVSIM-v2 87.7 EPDMS
> - **代码**: https://github.com/LeapWM/da-wam

---

*笔记创建时间: 2026-08-21*

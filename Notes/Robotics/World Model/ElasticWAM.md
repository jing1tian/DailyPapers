---
title: "Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control"
method_name: "ElasticWAM"
authors: [Haitian Li, Tianfu Li, Xuxi Zhao, Zizheng Liu, Yangcheng Yu, Yiwen Hou, Zeyu Xiao, Mutian Xu, Ziqian Yang, Haonan Chen, Lin Shao]
year: 2026
venue: arXiv
tags: [world-action-model, adaptive-computation, diffusion-policy, robot-manipulation, sparse-attention]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.14257
created: 2026-11-12
---

# 论文笔记：Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | National University of Singapore, RoboScience |
| 日期 | November 2026 |
| 项目主页 | [lihaitian.com/Elastic-WAM](https://lihaitian.com/Elastic-WAM/) |
| 对比基线 | [[Fast-WAM]], [[世界动作模型（WAM）\|Joint-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.14257) |

---

## 一句话总结

> Elastic-WAM 通过统一时空"精化需求场"，自适应地决定何时提交、推进多远、以及刷新哪些区域，以 32.2% 更低的延迟实现了 state-of-the-art 机器人控制。

---

## 核心贡献

1. **统一时空精化需求场（Refinement Demand Field）**: 单一[[Refinement Demand Field|需求场]]同时控制三个维度：WHEN（何时提交动作）、HOW FAR（每步去噪推进多远）、WHERE（刷新哪些空间 token）。
2. **动作接地三阶段训练**: 冻结 [[Fast-WAM]] 骨干，依次学习时序分布（HOW FAR）→ 空间优先级（WHERE）→ 总体需求（WHEN），每阶段都以动作误差为监督信号。
3. **循环稀疏推理**: 推理时保持完整 KV 缓存，[[Straight-Through Estimator|直通梯度掩码]]选出 $K_0=96$ 个 token 做稀疏更新，实现计算弹性而不损失全上下文。

---

## 问题背景

### 要解决的问题

[[世界动作模型（WAM）]] 通过迭代视频去噪预测未来动态来辅助机器人控制，但视频去噪带来大量推理开销。现有"部分去噪"方案固定精化预算，无法适应不同任务和状态的需求差异。

### 现有方法的局限

- **[[Fast-WAM]]**（部分去噪）：在所有任务和状态使用相同的固定去噪深度和 token 刷新数量，无法匹配每个任务/状态最优精化深度的差异。
- 实验发现：在 10 个 LIBERO-Plus 任务上，最优去噪深度从 1 步到 4 步不等，且同一任务内不同状态也有所不同；动作-视频注意力在不同空间区域分布极不均匀。

### 本文的动机

需求是任务和状态相关的，因此应该让模型自主学习"在哪个状态需要对哪些区域做多深的去噪精化"，而不是强加固定预算。需求场将时间（何时停、步子多大）和空间（哪些 token）整合到一个统一框架，并以动作质量为标准监督学习。

---

## 方法详解

### 模型架构

Elastic-WAM 冻结 [[Fast-WAM]] 的 FastWAM-Joint 骨干，在其上附加一个轻量**精化需求场网络**，实现自适应时空精化：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$ + 本体感觉 $s_t$
- **Backbone**: 冻结的 FastWAM-Joint（T5 文本编码器 + VAE 视觉编码器 + 196 个未来视频 token）
- **核心模块**: [[Refinement Demand Field]] 用于控制稀疏精化
- **输出**: 动作块 $a_{t:t+k}$（[[Action Chunking]]）
- **总参数**: 附加精化需求场为轻量网络（骨干冻结）

### 核心模块

#### 模块1：统一时空精化需求场（§3.1–3.2）

**设计动机**: 利用 [[Beta Distribution]] 的灵活单峰形状，对每组视频 token 建模"还剩多少精化需求"。

**具体实现**:
- 196 个未来视频 token 按几何感知方式分为 $G=64$ 组（每组含 1、2 或 4 个 token），组内尊重时序和摄像头视角边界。
- 每组 $g$ 有一个局部需求曲线，以归一化去噪进度 $z \in [0,1]$ 为自变量：
  $$q_g(z \mid s) = a_g(s) \cdot \mathrm{Beta}(z;\, m_g(s),\, c_g(s))$$
  曲线的面积恰好等于该组的"总剩余需求" $a_g(s)$。
- 为减少参数，所有组共享同一**时序基函数**：
  $$q_g(z \mid s) = a_g(s)\,B_0(z \mid s), \quad B_0 = \mathrm{Beta}(z;\,m_0,c_0)$$
  这样全局需求场退化为：
  $$Q(z \mid s) = D(s)\,B_0(z \mid s), \quad D(s)=\sum_g \frac{n_g}{N_F}a_g(s)$$
- 组幅度通过 [[Stop-Gradient]] 解耦空间优先级与总需求：
  $$a_g(s) = \mathrm{sigmoid}\!\left(b_0(s) + \mathrm{stopgrad}[\Delta u_g(s)]\right)$$

#### 模块2：三维决策（WHEN / HOW FAR / WHERE，§3.2）

**WHEN — 何时提交**:
- 每步的"继续概率"等于总需求 $D(s_k)$，累积提交概率为：
  $$F_k = 1 - \prod_{j \leq k} C_j, \quad C_j = D(s_j)$$
  当 $F_k \geq \eta = 0.5$ 时触发提交。

**HOW FAR — 步长（去噪进度）**:
- 以 $B_0$ 的众数作为当前的进度点 $z_k = m_0(s_k)$，对应噪声减少量：
  $$\Delta\sigma_k = \delta_{\min} + z_k(\delta_{\max} - \delta_{\min})$$
  其中 $\delta_{\min}=0.05$，$\delta_{\max}=\min(0.20,\,\sigma_k-\sigma_{\mathrm{floor}})$。

**WHERE — 空间选择**:
- 在严格 token 预算 $K_0=96$ 下，最大化高优先级组的覆盖：
  $$M_k^* = \arg\max_{M\in\{0,1\}^G} \sum_g n_g M_g \Delta u_g(s_k) \quad \text{s.t.} \quad \sum_g n_g M_g = K_0$$

#### 模块3：动作接地三阶段学习（§3.3）

**阶段一（HOW FAR）** — 时序教师网络：
- 对同一循环状态，对比不同去噪距离后强制提交的动作误差，误差小的候选获得更高目标概率。
- 停止信号为连续和提交动作误差之差，经温度 $T(s)$ 归一化：
  $$A_{\mathrm{stop}}(s) = E_{\mathrm{continue}}(s) - E_{\mathrm{commit}}(s), \quad p_{\mathrm{stop}}^* = \mathrm{sigmoid}\!\left(\frac{A_{\mathrm{stop}}}{T(s)}\right)$$

**阶段二（WHERE）** — 空间优先级网络：
- 时序分布冻结，比较稠密和稀疏更新的动作质量，使用[[Straight-Through Estimator|直通梯度掩码]] $M^{\mathrm{ST}}$ 训练：
  $$M^{\mathrm{ST}} = M^{\mathrm{hard}} + M^{\mathrm{soft}} - \mathrm{stopgrad}(M^{\mathrm{soft}})$$
  损失为：
  $$\mathcal{L}_{\mathrm{where}} = \mathcal{L}_{\mathrm{action\text{-}GT}} + \mathcal{L}_{\mathrm{action\text{-}Dense}} + 0.1\,\mathcal{L}_{\mathrm{video\text{-}GT}} + 0.01\,\mathcal{L}_{\mathrm{budget}}$$

**阶段三（WHEN）** — 总体需求学习：
- 时序教师的生存函数给出目标继续质量 $C_k^* = \exp(-\Delta H_k^{\mathrm{teacher}})$，转化为目标偏移 $b_0^*$：
  $$C_k^* = \sum_g \frac{n_g}{N_F}\,\mathrm{sigmoid}(b_0^*(s_k) + \Delta u_g(s_k))$$
  最终损失：
  $$\mathcal{L}_{\mathrm{final}} = \mathcal{L}_{b_0} + \mathcal{L}_{\mathrm{conditional}} + \mathcal{L}_{\mathrm{cumulative}} + 0.5\,\mathcal{L}_{\mathrm{commit\text{-}time}} + 0.1\,\mathcal{L}_{\mathrm{terminal\text{-}CF}}$$

#### 模块4：稀疏推理循环（§3.4）

每个规划周期：
1. **稠密联合去噪**（首步）：对所有未来视频 token 完整去噪一步，刷新循环状态和 KV 缓存。
2. **需求场决策**：确定是否继续（WHEN）、步长（HOW FAR）和刷新哪些 $K_0=96$ token（WHERE）。
3. **稀疏更新**：选中的视频 query 在完整 KV 上下文中做注意力，未选中的保持缓存状态复用。
4. **提交**：触发提交后（或安全兜底），直接复用缓存的视频表示进行动作去噪，无需额外视频前缀计算。

---

## 关键公式

### 公式1：[[Refinement Demand Field|局部需求曲线]]

$$
q_g(z \mid s) = a_g(s)\,\mathrm{Beta}(z;\, m_g(s),\, c_g(s))
$$

**含义**: 组 $g$ 的精化需求关于归一化去噪进度 $z$ 的分布，曲线面积等于该组的总剩余需求 $a_g(s)$。

**符号说明**:
- $z \in [0,1]$: 归一化去噪进度（0=噪声最大，1=完全干净）
- $a_g(s)$: 组 $g$ 的总剩余需求幅度
- $m_g(s)$: [[Beta Distribution]] 均值参数
- $c_g(s)$: 集中度参数

---

### 公式2：[[Refinement Demand Field|全局需求场]]

$$
Q(z \mid s) = D(s)\,B_0(z \mid s), \quad D(s) = \sum_{g=1}^G \frac{n_g}{N_F}\,a_g(s)
$$

**含义**: 利用共享时序基函数 $B_0$，全局需求场被分解为总需求标量 $D(s)$ 与形状 $B_0$ 的乘积。

**符号说明**:
- $D(s)$: 总需求（标量），作为继续概率
- $B_0(z \mid s) = \mathrm{Beta}(z; m_0, c_0)$: 共享时序基函数
- $n_g$: 组 $g$ 的 token 数量
- $N_F = 196$: 总未来视频 token 数

---

### 公式3：[[Stop-Gradient|组幅度的解耦参数化]]

$$
a_g(s) = \mathrm{sigmoid}\!\left(b_0(s) + \mathrm{stopgrad}[\Delta u_g(s)]\right)
$$

$$
\Delta u_g(s) = u_g^{\mathrm{rel}}(s) - \sum_{j=1}^G \frac{n_j}{N_F}\,u_j^{\mathrm{rel}}(s), \quad \sum_g \frac{n_g}{N_F}\Delta u_g = 0
$$

**含义**: 通过 [[Stop-Gradient]] 将绝对总需求 $b_0$ 与相对空间优先级 $\Delta u_g$ 解耦，使两者可以独立学习。

**符号说明**:
- $b_0(s)$: 共享偏移，控制总需求大小
- $\Delta u_g(s)$: 组 $g$ 相对于均值的空间优先级，均值为零
- $\mathrm{stopgrad}[\cdot]$: 梯度截断操作，防止空间优先级梯度影响绝对需求

---

### 公式4：[[Refinement Demand Field|WHEN — 累积提交概率]]

$$
F_k = 1 - \exp\!\left(-\sum_{j \leq k} \Delta H_j\right) = 1 - \prod_{j \leq k} C_j, \quad \Delta H_k = -\log\!\left(\mathrm{clamp}(C_k, \varepsilon, 1)\right)
$$

**含义**: 将每步总需求 $C_k = D(s_k)$ 视为继续概率，通过生存函数计算累积提交概率，当 $F_k \geq 0.5$ 时触发提交。

**符号说明**:
- $C_k = D(s_k)$: 第 $k$ 步的继续概率
- $\Delta H_k$: 单步危险（hazard）增量
- $\eta = 0.5$: 提交阈值

---

### 公式5：[[Refinement Demand Field|HOW FAR — 自适应步长]]

$$
z_k = m_0(s_k), \quad \Delta\sigma_k = \delta_{\min} + z_k(\delta_{\max} - \delta_{\min})
$$

**含义**: 以 $B_0$ 的众数作为当前进度点，映射为去噪步长，状态越"需要精化"则步子越小（进度越低，步长越小）。

**符号说明**:
- $m_0(s_k)$: 时序基函数 $B_0$ 的均值（近似众数）
- $\delta_{\min}=0.05$，$\delta_{\max}=\min(0.20,\sigma_k - \sigma_{\mathrm{floor}})$: 步长范围

---

### 公式6：[[Refinement Demand Field|WHERE — 稀疏 token 选择]]

$$
M_k^* = \arg\max_{M\in\{0,1\}^G}\sum_g n_g M_g \Delta u_g(s_k) \quad \text{s.t.} \quad \sum_g n_g M_g = K_0
$$

**含义**: 在严格 token 预算 $K_0=96$ 下，贪心选择最高优先级的组进行稀疏更新。

**符号说明**:
- $M_g \in \{0,1\}$: 组 $g$ 是否被选中
- $K_0=96$: token 预算（占 196 总量的 49%）

---

### 公式7：[[Straight-Through Estimator|直通梯度掩码]]

$$
M^{\mathrm{ST}} = M^{\mathrm{hard}} + M^{\mathrm{soft}} - \mathrm{stopgrad}(M^{\mathrm{soft}})
$$

**含义**: [[Straight-Through Estimator]] 技巧，前向传播使用硬离散掩码 $M^{\mathrm{hard}}$，反向传播通过软掩码 $M^{\mathrm{soft}}$ 传递梯度，实现可微分的 top-K 选择训练。

**符号说明**:
- $M^{\mathrm{hard}}$: 硬 0/1 掩码（前向用）
- $M^{\mathrm{soft}}$: 软概率掩码（梯度通道）

---

### 公式8：动作误差（阶段一监督）

$$
\mathcal{E}(s, y) = 0.6\,E_{\mathrm{seq}}(s, y) + 0.4\,E_{\Delta}(s, y)
$$

**含义**: 结合动作序列误差 $E_{\mathrm{seq}}$ 和连续动作差分误差 $E_\Delta$ 评估动作质量，强调动作的平滑性。

**符号说明**:
- $E_{\mathrm{seq}}$: 动作序列与 GT 的 L2 误差
- $E_\Delta$: 相邻动作差分的 L2 误差
- 0.6 / 0.4: 权重系数

---

## 关键图表

### Figure 1：动机分析

![Figure 1 - Motivation Analysis](https://lihaitian.com/Elastic-WAM/assets/overview.webp)

**说明**: 展示 [[Fast-WAM]] 固定精化预算的局限性：(a1)–(a3) 在 10 个 LIBERO-Plus 任务上，最优去噪深度从 1 到 4 不等，同一任务内不同状态也有差异；(b)–(c) 动作-视频注意力在空间上分布极不均匀，说明不同区域对动作生成的重要性有显著差异。

---

### Figure 2：Elastic-WAM 整体流程

![Figure 2 - Elastic-WAM Pipeline](https://arxiv.org/html/2611.14257v1/sec/figures/method.png)

**说明**: 上方为推理流程：初始稠密联合去噪刷新 KV 缓存 → 自适应稀疏世界精化（WHEN/HOW FAR/WHERE）→ 缓存复用的动作精化。下方展示时序模块（$B_0$，HOW FAR）、空间模块（$\Delta u_g$，WHERE）和统一需求场（$b_0 + \Delta u_g$，WHEN）三个分支的共同作用。

---

### Figure 3：与 Joint-WAM 的定性对比

![Figure 3 - Qualitative Comparison with Joint-WAM](https://arxiv.org/html/2611.14257v1/sec/figures/comparewithjoint_a.png)

**说明**: 在嘈杂观测条件下，Joint-WAM 失败而 Elastic-WAM 成功。右侧注意力热图显示 Elastic-WAM 聚焦于动作相关区域；token 更新计数图展示自适应稀疏更新模式，不同状态触发的更新次数和区域各异。

---

### Figure 4：真实世界评测设置

![Figure 4 - Real-World Settings](https://arxiv.org/html/2611.14257v1/sec/figures/real.png)

**说明**: 三种评测条件：Nominal（标准）、Lighting OOD（强光照分布外）、Object OOD（白色杯换成银杯），在双臂 xArm 7 机器人上执行四阶段泡茶任务（20 次 trial/条件）。

---

### Figure 5：精度敏感操作对比

![Figure 5 - Precision Tasks](https://arxiv.org/html/2611.14257v1/CompareWithFast.png)

**说明**: 在 [[RoboCasa-GR1]] 精度敏感任务（瓶子入柜、罐子入抽屉）上，[[Fast-WAM]] 在最终放置阶段失败，Elastic-WAM 因自适应精化能聚焦关键时刻而成功。

---

### Figure 6：长程泡茶任务展示

![Figure 6 - Long-Horizon Tea Preparation](https://arxiv.org/html/2611.14257v1/LongHorizon.png)

**说明**: Elastic-WAM 完成四阶段泡茶任务的关键帧序列（带腕部摄像头视图）：(1) 拿取茶包 → (2) 放入水壶 → (3) 拿取水杯 → (4) 倒水。展示了长程双臂操作的鲁棒性。

---

### Table 1：LIBERO 成功率 (%)

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Joint-WAM | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| **Elastic-WAM** | **99.4** | **99.6** | **98.0** | **98.2** | **98.8** |

**说明**: Elastic-WAM 在 Long 任务上相比 Joint-WAM 提升 1.4pp，最终平均最高。

---

### Table 2：LIBERO-Pro 成功率 (%)

| Method | Obj | Pos | Sem | Task | **Avg** |
|--------|-----|-----|-----|------|---------|
| π0.5 | 96.0 | 20.8 | 95.8 | 0.8 | 53.4 |
| OpenVLA | 93.0 | 0.0 | 97.3 | 0.0 | 47.6 |
| X-VLA | 79.0 | 0.8 | 96.8 | 6.8 | 45.9 |
| Fast-WAM | 78.0 | 1.0 | 83.2 | 15.8 | 44.5 |
| Joint-WAM | 84.5 | 6.3 | 96.2 | 24.2 | 52.8 |
| **Elastic-WAM** | **85.6** | **8.4** | **97.7** | **24.7** | **54.1** |

**说明**: [[LIBERO-PRO]] 评测超出训练分布的鲁棒性，Elastic-WAM 超越全部基线，在 Positional 和 Semantic 泛化上最强。

---

### Table 3：LIBERO-Plus 成功率 (%) — 7 种扰动

| Method | Cam | Rob | Lang | Light | Backg | Noise | Layout | **Avg** |
|--------|-----|-----|------|-------|-------|-------|--------|---------|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| Fast-WAM | 18.8 | 45.7 | 70.1 | 83.2 | 45.7 | 29.8 | 62.7 | 49.1 |
| Joint-WAM | 37.5 | 64.5 | 93.0 | 95.0 | 55.9 | 47.3 | 79.6 | 66.3 |
| LingBot-VA | 40.9 | 83.0 | 86.4 | 82.3 | 53.1 | 64.4 | 76.2 | 69.5 |
| **Elastic-WAM** | **52.1** | **76.3** | **95.5** | **95.1** | **60.0** | **61.1** | **81.4** | **73.7** |

**说明**: [[LIBERO-Plus]] 覆盖摄像头、机器人初始化、语言、光照、背景、传感器噪声、布局 7 种扰动，Elastic-WAM 在 Cam（+14.6pp）和 Noise（+13.8pp）上相比 Joint-WAM 有最大提升，平均最高。

---

### Table 4：RoboCasa-GR1 成功率 (%) 与延迟 (ms)

| Method | Success (%) | Latency (ms) |
|--------|-------------|--------------|
| π0.5 | 37.0 | 80.5 |
| GR00T-N1.6 | 47.6 | 77.0 |
| StarVLA | 48.8 | 82.0 |
| Fast-WAM | 54.9 | 331.1 |
| Joint-WAM | 55.0 | 572.6 |
| **Elastic-WAM** | **57.4** | **388.2** |

**说明**: [[RoboCasa-GR1]] 双足机器人操作 benchmark，Elastic-WAM 以 388.2ms 延迟（相比 Joint-WAM 的 572.6ms 降低 **32.2%**）达到最高 57.4% 成功率。

---

### Table 5：空间选择消融（RoboCasa-GR1）

| 配置 | 成功率 (%) |
|------|-----------|
| **Learned WHERE**, K=96 | **57.42** |
| Random WHERE, K=96 | 53.83 |
| All-active, K=128 | 55.17 |
| FirstKV（复用初始缓存） | 54.50 |

**关键发现**: 学习到的空间优先级比随机选择高 3.6pp，甚至超过使用更多 token（K=128）的稠密基线，说明关键区域选择的重要性。

---

### Table 6：LIBERO-Plus 消融实验

**(a) 时序自适应消融**

| 配置 | Avg (%) |
|------|---------|
| WHERE only（无时序自适应）| 71.71 |
| Fixed step（固定步长）| 72.31 |
| **Adaptive（完整，自适应步长 + WHEN）** | **73.71** |

**(b) Token 预算消融**

| Token 预算 K | Avg (%) |
|-------------|---------|
| 49 | 72.84 |
| **96** | **73.71** |
| 147 | 71.71 |
| 196 (全量) | 70.92 |

**关键发现**: 自适应时序控制（+1.4pp vs WHERE only）和最优 token 预算（K=96，占比 49%）均有明确贡献；过多 token 甚至比 K=96 效果差（70.92% vs 73.71%），说明稀疏性本身有正则化效果。

---

### Table 7：真实世界泡茶任务（20 次 trial/条件）

| Method | Nominal SR/PS | Lighting OOD SR/PS | Object OOD SR/PS |
|--------|--------------|-------------------|-----------------|
| Fast-WAM | 0.0 / 12.5 | 0.0 / 5.0 | 0.0 / 10.0 |
| Joint-WAM | 30.0 / 37.5 | 30.0 / 35.0 | 30.0 / 35.0 |
| **Elastic-WAM** | **50.0 / 57.5** | **50.0 / 52.5** | **30.0 / 47.5** |

**说明**: SR = 全任务成功率，PS = 进度得分（部分完成分）。Elastic-WAM 在 Nominal 和 Lighting OOD 下显著优于 Joint-WAM（50% vs 30%），Object OOD 下成功率持平但进度得分更高（47.5% vs 35.0%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件，每任务 50 rollout | 4 类任务（Spatial/Object/Goal/Long） | 训练 + 测试 |
| [[LIBERO-Plus]] | 7 种扰动 | 摄像头/机器人/语言/光照/背景/噪声/布局 | 零样本测试 |
| [[LIBERO-PRO]] | 4 类别 | Obj/Pos/Sem/Task 分布外 | 零样本测试 |
| [[RoboCasa-GR1]] | 24 任务 | 双足人形机器人，遵循 StarVLA 协议 | 测试 |
| 真实世界 | 200 次示范 | 四阶段泡茶，双臂 xArm 7 | 训练 + 测试（20 trial/条件）|

### 实现细节

- **Backbone**: FastWAM-Joint（T5 文本编码器，VAE 视觉编码器，本体感觉投影）
- **未来视频 token**: 196 个，分为 64 个几何感知组
- **Token 预算**: $K_0 = 96$（≈49% 的 196 总量）
- **硬件**: NVIDIA H800 GPU（延迟测试）
- **骨干冻结**: 是，仅训练精化需求场网络

### 可视化结果

- 在嘈杂观测下，Elastic-WAM 注意力热图集中于任务相关物体区域（瓶子、罐子），而 Joint-WAM 注意力分散。
- 不同状态触发不同的 token 更新模式：操作精细接触时更新频率更高，粗糙运动时更少。
- 真实世界 Lighting OOD 场景下，Elastic-WAM 能聚焦于变化光照下的对象区域，表现出更好的鲁棒性。

---

## 批判性思考

### 优点
1. **统一框架的优雅性**: 将时间（WHEN/HOW FAR）和空间（WHERE）决策统一到单一 [[Refinement Demand Field]] 中，理论上相当自洽。
2. **计算弹性显著**: 32.2% 延迟降低的同时性能提升，这在 WAM 效率研究中较为罕见（通常是以性能换速度）。
3. **渐进式冻结训练**: 分三阶段冻结骨干学习，避免破坏预训练特征，工程上易于扩展到其他 WAM。
4. **稀疏性有正则化效果**: Table 6b 中 K=196（全量）反而不如 K=96，说明强制关注关键区域有类似注意力正则化的作用。

### 局限性
1. **Figure 1 的分析局限于 LIBERO-Plus**：最优去噪深度的分析仅在特定 benchmark 上展示，不清楚该规律在其他任务域（如导航、灵巧操作）是否同样显著。
2. **三阶段训练复杂度**: 需要顺序训练三个阶段，每个阶段的超参数（温度 $T(s)$、权重系数等）调整增加了工程开销。
3. **实时性仍有差距**: 388.2ms 延迟相比非 WAM 的 VLA（π0.5 的 80.5ms、GR00T-N1.6 的 77.0ms）仍有明显差距，视频去噪本身的开销尚未根本解决。
4. **真实世界样本量较小**: 20 次 trial/条件，统计可信度有限。

### 潜在改进方向
1. 将需求场学习与骨干联合微调，探索端到端训练的潜力。
2. 将 token 预算 $K_0$ 也做自适应化（目前固定为 96），进一步压缩简单状态下的计算。
3. 探索需求场的跨任务迁移能力，验证其通用性。

### 可复现性评估
- [ ] 代码开源（项目页面暂无代码链接）
- [ ] 预训练模型
- [x] 训练细节完整（三阶段训练过程详细描述）
- [x] 数据集可获取（LIBERO、RoboCasa 均为公开数据集）

---

## 关联笔记

### 基于
- [[Fast-WAM]]: 基础骨干 FastWAM-Joint，Elastic-WAM 冻结其参数后附加需求场网络
- [[世界动作模型（WAM）]]: 所属方法范畴，联合视频-动作生成框架

### 对比
- [[Fast-WAM]]: Fast-WAM（固定部分去噪预算）→ Elastic-WAM（自适应时空预算）
- [[Diffusion Policy]]: 标准扩散策略，无世界模型

### 方法相关
- [[Refinement Demand Field]]: 核心新概念，统一控制 WHEN/HOW FAR/WHERE
- [[Beta Distribution]]: 需求曲线形状，建模单峰去噪进度分布
- [[Stop-Gradient]]: 解耦空间优先级与总需求的关键技巧
- [[Straight-Through Estimator]]: WHERE 阶段可微分 top-K 选择的训练方法
- [[Action Chunking]]: 输出动作的序列形式

### 硬件/数据相关
- [[RoboCasa-GR1]]: 主要测试 benchmark（人形机器人）
- [[LIBERO-Plus]]: 7 种扰动的鲁棒性测试 benchmark
- [[LIBERO-PRO]]: 分布外泛化测试

---

## 速查卡片

> [!summary] Elastic-WAM (arXiv 2611.14257)
> - **核心**: 统一时空精化需求场，自适应决策 WHEN/HOW FAR/WHERE 三维度
> - **方法**: 冻结 Fast-WAM 骨干 + 三阶段动作接地训练 + 循环稀疏推理（K=96 token）
> - **结果**: LIBERO-Plus 73.7%（SOTA），RoboCasa-GR1 57.4% / 延迟降低 32.2%，真实泡茶 50% SR
> - **代码**: [项目主页](https://lihaitian.com/Elastic-WAM/)（暂无代码）

---

*笔记创建时间: 2026-11-12*

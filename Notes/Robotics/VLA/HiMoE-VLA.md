---
title: "HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies"
method_name: "HiMoE-VLA"
authors: [Zhiying Du, Bei Liu, Yaobo Liang, Yichao Shen, Haidong Cao, Xiangyu Zheng, Zhiyuan Feng, Zuxuan Wu, Jiaolong Yang, Yu-Gang Jiang]
year: 2025
venue: arXiv
tags: [vla, mixture-of-experts, flow-matching, heterogeneous-data, multi-embodiment, generalist-policy, robot-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2512.05693
created: 2026-07-10
---

# 论文笔记：HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 复旦大学、微软研究院亚洲、西安交通大学、清华大学 |
| 日期 | December 2025 |
| 项目主页 | — |
| 对比基线 | [[pi0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2512.05693) / [Code](https://github.com/ZhiyingDu/HiMoE-VLA) |

---

## 一句话总结

> 通过层级 [[Mixture-of-Experts]] 架构将异构机器人数据的多种来源的 heterogeneity 分层解耦，将多数据集 [[Co-training]] 的负迁移转化为正迁移。

---

## 核心贡献

1. **层级 MoE 架构（HiMoE）**: 设计 AS-MoE（边界层，动作空间专化）、HB-MoE（邻近层，残余异构平衡）与共享 Dense Transformer（中心层，跨域知识融合）三层结构，系统性地隔离动作空间与具身异构性
2. **动作空间正则化（AS-Reg）**: 对 AS-MoE 路由分布施加有监督对比损失 [[Contrastive Loss]]，使同一动作空间的 token 趋向相似的专家路由模式
3. **异构平衡正则化（HB-Reg）**: 对 HB-MoE 施加负载均衡损失 [[Load Balancing Loss]]，防止专家塌陷，确保稀疏容量的均匀利用

---

## 问题背景

### 要解决的问题

在多源异构机器人数据集（如 [[Open X-Embodiment]]）上联合训练通用 VLA 策略时，不同数据来源的动作空间（关节角度指令 vs. 末端执行器增量）、具身结构（单臂 vs. 双臂）和观测配置各不相同，简单的密集架构导致严重的**负迁移**（[[Negative Transfer]]）。

### 现有方法的局限

- 现有 VLA（如 [[pi0]]、OpenVLA）使用几乎完全共享的动作模块，将所有异构性压缩进同一密集计算中
- 接口对齐、数据集特定头等方法仅在输入/输出层面处理异构性，未在动作模块内部显式分离动作空间差异与其他具身差异
- 单层 [[Mixture-of-Experts]] 方案（non-hierarchical）在混合动作空间协训练时效果受限

### 本文的动机

动作空间差异（如 EEF vs. 关节角度）与具身/场景差异在本质上属于不同层级的异构性：前者通常导致更强、更系统的路由偏好，后者则需要更均衡的稀疏容量分配。将两类异构性分层解耦，分别用 AS-MoE 和 HB-MoE 处理，再让中心 Dense 层整合共享表示，能够更有效地将负迁移转化为正迁移。

---

## 方法详解

### 模型架构

HiMoE-VLA 采用 **VLM + 动作模块（HiMoE）** 双模块架构：

- **输入**: 语言指令 $l$ + 本体感知向量 $q_t$ + RGB 图像观测 $o_t$（第三视角 + 两路腕部视角）
- **VLM Backbone**: [[PaliGemma]]（冻结 + 层级 KV 状态暴露）
- **核心模块**: [[Mixture-of-Experts|HiMoE 动作模块]] 负责处理带噪动作块与机器人状态
- **输出**: [[Action Chunking|动作块]] $A_t$（Horizon $H$）
- **总参数**: ~4B（[[PaliGemma]] 2B + 动作模块）

### 核心模块

#### 模块1: VLM 层级条件化

**设计动机**: 传统方案仅将 VLM 最后一层的表示传给动作模块，损失了低层视觉语义信息。

**具体实现**:
- 以 [[PaliGemma]] 为视觉语言骨干，提取**中间层**的 KV 状态（layer-wise KV states）
- 动作模块的各 Transformer 层通过交叉注意力对应地 attend 到 VLM 相同深度的 KV 表示
- 令动作 token 同时访问高层语义指令与低层视觉特征

#### 模块2: 层级 Mixture-of-Experts（HiMoE）

**设计动机**: 不同类型的数据异构性在网络深度上呈现出不同的最优处理位置。

**结构**（从外到内层级）:

```
[AS-MoE 块]  ←  边界层：专化于动作空间差异（EEF vs. 关节角）
[HB-MoE 块]  ←  邻近层：均衡处理残余异构性（具身、场景）
[Dense Transformer 块]  ←  中心层：跨域共享表示融合
[HB-MoE 块]
[AS-MoE 块]
```

每个 MoE 层使用 **top-K=4 路由**，N=32 专家，并参照 DeepSeekMoE 为每层配置 1 个**共享专家**（对所有 token 无条件激活），确保异构无关的通用计算。

AS-MoE 的路由分布 $\hat{r}_u$ 受 AS-Reg 引导，使同一动作空间的 token 趋向相似专家分布。HB-MoE 的路由受 HB-Reg 负载均衡约束，防止塌陷。

#### 模块3: 统一状态-动作接口

**设计动机**: 不同数据集的状态向量和动作向量语义各异，需在模块入口归一化。

**具体实现**:
- 将所有来源的本体感知向量和动作块映射到统一向量接口（维度对齐）
- 同时保留分类动作空间/具身标识 $c$，用于 AS-Reg 的监督信号
- 支持单臂（7-DoF，xArm7）和双臂（14-DoF，ALOHA）两种设置

---

## 关键公式

### 公式1: [[Action Chunking|策略定义]]

$$
\pi_{\theta}(l, q_t, o_t) \mapsto A_t
$$

**含义**: 策略以语言指令 $l$、本体感知 $q_t$、观测 $o_t$ 为输入，输出动作块 $A_t$（Horizon $H$）

**符号说明**:
- $l$: 语言任务指令
- $q_t$: 当前时刻本体感知向量（含义随数据源而异）
- $o_t$: RGB 图像观测
- $A_t \in \mathbb{R}^{H \times d_a}$: 动作块，$d_a$ 为动作维度

### 公式2: [[CFM|Flow Matching 插值轨迹]]

$$
A_t^{\tau} = \tau A_t + (1 - \tau)\epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \tau \in [0, 1]
$$

**含义**: 在噪声 $\epsilon$ 与真实动作 $A_t$ 之间做线性插值，形成连续时间流匹配轨迹

**符号说明**:
- $\tau$: 流时间步，$\tau \sim \mathcal{U}(0, 1)$
- $\epsilon$: 标准高斯噪声
- $A_t^{\tau}$: 时刻 $\tau$ 的插值动作

### 公式3: [[CFM|Flow Matching 训练损失]]

$$
\mathcal{L}_{\text{flow}} = \mathbb{E}_{\tau,\, A_t,\, \epsilon} \left[ \left\| v_\theta(A_t^{\tau}, \tau, o_t, l, q_t) - (A_t - \epsilon) \right\|_2^2 \right]
$$

**含义**: 训练速度场网络 $v_\theta$ 预测插值点到目标动作的方向，使流匹配目标条件分布

**符号说明**:
- $v_\theta$: 速度场网络（即动作模块）
- $A_t - \epsilon$: 从 $\epsilon$ 到 $A_t$ 的方向向量（目标速度）

### 公式4: [[Contrastive Loss|动作空间正则化（AS-Reg）]]

$$
\mathcal{L}_{\text{AS}} = \frac{1}{U_+} \sum_{u=1}^{U} \mathbb{1}[|P(u)| > 0] \cdot \frac{-1}{|P(u)|} \sum_{p \in P(u)} \log \frac{\exp(\hat{r}_u^\top \hat{r}_p / \beta)}{\sum_{v \in A(u)} \exp(\hat{r}_u^\top \hat{r}_v / \beta)}
$$

**含义**: 有监督对比损失，促使同一动作空间的 token 的路由分布 $\hat{r}$ 相互靠近，不同动作空间的路由分布相互远离

**符号说明**:
- $\hat{r}_u$: token $u$ 的归一化路由分布向量
- $P(u) = \{v : c_v = c_u, v \neq u\}$: 与 token $u$ 同一动作空间的正样本集合
- $A(u) = \{1, \dots, U\} \setminus \{u\}$: 除 $u$ 之外的所有 token（含负样本）
- $\beta = 0.1$: 温度系数
- $U_+$: 至少有一个正样本的 token 数量

### 公式5: [[Load Balancing Loss|异构平衡正则化（HB-Reg）]]

$$
f_i = \frac{N}{K \cdot U} \sum_{u=1}^{U} r_{i,u}, \quad P_i = \frac{1}{U} \sum_{u=1}^{U} s_{i,u}, \quad \mathcal{L}_{\text{HB}} = \sum_{i=1}^{N} f_i P_i
$$

**含义**: 负载均衡损失，确保各专家被均匀激活，防止专家塌陷

**符号说明**:
- $f_i$: 专家 $i$ 的归一化激活频率
- $P_i$: 专家 $i$ 的平均路由概率
- $r_{i,u}$: token $u$ 对专家 $i$ 的二值路由决策（0/1）
- $s_{i,u}$: token $u$ 对专家 $i$ 的路由 logit（softmax 前）
- $N=32$: 专家总数；$K=4$: top-K 激活数

### 公式6: 总训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_{\text{AS}} \mathcal{L}_{\text{AS}} + \lambda_{\text{HB}} \mathcal{L}_{\text{HB}}
$$

**含义**: 流匹配损失为主要动作生成信号，AS-Reg 和 HB-Reg 分别约束 AS-MoE 和 HB-MoE 的路由行为

**符号说明**:
- $\lambda_{\text{AS}}, \lambda_{\text{HB}}$: 正则项权重系数

---

## 关键图表

### Figure 1: HiMoE-VLA 系统概览

![Figure 1](https://arxiv.org/html/2512.05693v2/x1.png)

**说明**: HiMoE-VLA 整体架构。左侧蓝色部分为以 [[PaliGemma]] 初始化的 VLM 骨干，右侧橙色部分为提出的动作模块（HiMoE），负责处理不同机器人状态与带噪动作块并生成最终动作输出。VLM 的中间层 KV 状态逐层暴露给动作模块对应层，实现层级条件化。

### Figure 2: HiMoE 详细结构

![Figure 2](https://arxiv.org/html/2512.05693v2/x2.png)

**说明**: [[Mixture-of-Experts|层级 MoE（HiMoE）]] 的内部结构。边界 AS-MoE 层专化于动作空间差异，邻近 HB-MoE 层均衡处理残余异构性，中心 Dense Transformer 层融合跨域共享知识。每个 MoE 层含 N=32 路由专家与 1 个共享专家（参照 DeepSeekMoE），使用 top-4 稀疏激活。

### Figure 3: 真实机器人执行示例

![Figure 3](https://arxiv.org/html/2512.05693v2/x3.png)

**说明**: 真实世界执行的定性示例。上行：单臂 xArm7（水果到盘子、杯中杯、积木叠放等任务）；下行：双臂 [[ALOHA]]（杯子递送、勺舀倒、折短裤等任务）。HiMoE-VLA 在两类具身平台上均展现出稳定的操作能力。

### Table 1: CALVIN 仿真 Benchmark（D→D）

| 方法 | 1步 | 2步 | 3步 | 4步 | 5步 | 总分 |
|------|-----|-----|-----|-----|-----|------|
| Octo | 0.771 | 0.535 | 0.318 | 0.206 | 0.136 | 1.97 |
| OpenVLA | 0.716 | 0.385 | 0.180 | 0.088 | 0.042 | 1.41 |
| RDT-1B | 0.757 | 0.495 | 0.359 | 0.243 | 0.184 | 2.04 |
| DeeR | 0.853 | 0.696 | 0.549 | 0.420 | 0.312 | 2.83 |
| MDT | 0.937 | 0.845 | 0.741 | 0.644 | 0.556 | 3.72 |
| π₀ | 0.914 | 0.830 | 0.739 | 0.676 | 0.599 | 3.76 |
| **HiMoE-VLA** | **0.938** | **0.866** | **0.794** | **0.723** | **0.659** | **3.98** |
| FLOWER | 0.974 | 0.924 | 0.869 | 0.813 | 0.749 | 4.35 |
| FLOWER + HiMoE | **0.979** | **0.943** | **0.904** | **0.859** | **0.801** | **4.49** |

**关键发现**: HiMoE-VLA（3.98）超越 [[pi0]]（3.76）0.22分；将 HiMoE 集成到 FLOWER 框架后进一步达到 4.49，验证了 HiMoE 的通用性。

### Table 2a: [[LIBERO]] 仿真 Benchmark

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| **HiMoE-VLA** | **98.2** | **99.4** | **98.6** | **95.8** | **98.0%** |

**关键发现**: LIBERO 平均成功率 98.0%，在所有子任务（Spatial/Object/Goal/Long）上均超越 π₀ 和 π₀.₅。

### Table 2b: 真实机器人 xArm7 单臂评测

| 方法 | 水果→盘子 抓 | 水果→盘子 放 | 杯中杯 抓 | 杯中杯 插 | 积木叠放 抓 | 积木叠放 堆 | 平均 |
|------|-----------|-----------|---------|---------|-----------|-----------|------|
| Octo-Base | 31.3 | 18.8 | 33.3 | 16.7 | 16.7 | 0.0 | 19.3 |
| OpenVLA | 37.5 | 25.0 | 27.8 | 16.7 | 22.2 | 0.0 | 21.2 |
| CogACT | 65.6 | 59.4 | 77.8 | 63.9 | 69.4 | 33.3 | 61.5 |
| π₀ | 68.8 | 62.5 | 77.8 | 61.1 | 72.2 | 33.3 | 62.5 |
| **HiMoE-VLA** | **81.3** | **75.0** | **88.9** | **72.2** | **83.3** | **50.0** | **75.0** |

**关键发现**: xArm7 平均成功率 75.0%，较 [[pi0]]（62.5%）提升 12.5 个百分点。

### Table 2c: 真实机器人 [[ALOHA]] 双臂评测

| 方法 | 递杯 抓 | 递杯 传 | 递杯 放 | 舀取 | 倒取 | 折短裤 一次 | 折短裤 两次 | 平均 |
|------|--------|--------|--------|------|------|-----------|-----------|------|
| ACT | 40.0 | 0.0 | 73.3 | 6.6 | 0.0 | 20.0 | 6.6 | 20.9 |
| RDT-1B | 66.6 | 13.3 | 93.3 | 40.0 | 20.0 | 53.3 | 46.6 | 47.5 |
| π₀ | 80.0 | 13.3 | 93.3 | 46.6 | 26.6 | 66.6 | 53.3 | 54.2 |
| **HiMoE-VLA** | **80.0** | **26.6** | **100.0** | **53.3** | **40.0** | **80.0** | **66.6** | **63.7** |

**关键发现**: ALOHA 双臂平均成功率 63.7%，较 [[pi0]]（54.2%）提升 9.5 个百分点。

### Table 3: 泛化性评测（未见场景）

| 方法 | 单臂+干扰物 | 单臂+新场景 | 单臂均值 | 双臂+干扰物 | 双臂+新场景 | 双臂均值 |
|------|-----------|-----------|---------|-----------|-----------|---------|
| OpenVLA | 19.4 | 15.6 | 17.6 | — | — | — |
| CogACT | 52.8 | 50.0 | 51.5 | — | — | — |
| RDT-1B | — | — | — | 28.9 | 26.7 | 27.8 |
| π₀ | 58.3 | 53.1 | 55.9 | 40.0 | 26.7 | 33.4 |
| **HiMoE-VLA** | **69.4** | **65.6** | **67.6** | **53.3** | **46.7** | **50.0** |

**关键发现**: 在未见干扰物和新场景下，HiMoE-VLA 泛化性均优于所有基线，双臂新场景成功率较 π₀ 提升 20 个百分点。

### Table 4: HiMoE vs. 其他异构处理方案（CALVIN 消融）

| 方案 | CALVIN 总分 |
|------|-----------|
| Sep. Heads（分离头） | 3.827 |
| GR00T-Like（数据集条件化） | 3.856 |
| **HiMoE（ours）** | **4.012** |

### Table 5: 动作空间协训练效果（负迁移 → 正迁移）

| 方法 | 单独 CALVIN (D) | + CALVIN ABC+D 协训 | 差值 |
|------|---------------|-------------------|------|
| π₀ | 3.806 | 3.547 | **-0.259**（负迁移） |
| Ours w/o MoE | 3.819 | 3.777 | -0.042 |
| **Full HiMoE** | 3.826 | 4.012 | **+0.186**（正迁移） |

**关键发现**: [[Negative Transfer|负迁移]] 的存在被实验验证，而 HiMoE 是目前唯一将其转化为正迁移的方案。

### Table 6: HiMoE 组件消融（CALVIN ABC+D 协训）

| 配置 | CALVIN 总分 |
|------|-----------|
| w/o MoE（纯密集） | 3.777 |
| Full-HB-MoE（无 AS-MoE） | 3.901 |
| w/o AS-MoE | 3.873 |
| w/o HB-MoE | 3.836 |
| w/o Reg（无正则化） | 3.835 |
| Single-MoE + Reg | 3.813 |
| **Full HiMoE** | **4.012** |

**关键发现**: AS-MoE 和 HB-MoE 对性能提升均有独立贡献；仅用单层 MoE 即使加正则化也不及层级设计；正则化方案（Reg）对路由质量有显著帮助。

### Table 7: 传感器/场景协训练（共享 EEF 动作空间）

| 方法 | CALVIN | + LIBERO 协训 | 差值 |
|------|--------|-------------|------|
| π₀ | 3.776 | 3.504 | -0.272 |
| w/o MoE | 3.788 | 3.665 | -0.123 |
| Standard MoE | 3.808 | 3.862 | +0.054 |
| **Full HiMoE** | 3.819 | 3.966 | **+0.147** |

**关键发现**: 即使动作空间相同（均为 EEF），HiMoE 的 HB-MoE 也能将传感器/场景异构性的负迁移转化为正迁移，说明 HB-MoE 专化于其他类型的具身差异。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Open X-Embodiment]] | 22.5M 帧（26个子集） | 多具身、多动作空间 | 预训练 |
| ALOHA datasets | 1.6M 帧 | 双臂操作 | 预训练 |
| [[CALVIN]] (D→D) | — | 桌面操作 5 步链式任务 | 仿真评测 |
| [[LIBERO]] | — | 4 个子集（Spatial/Object/Goal/Long） | 仿真评测 |
| xArm7 真实任务 | — | 3 个任务（6 子步骤） | 真实评测 |
| ALOHA 真实任务 | — | 3 个任务（7 子步骤） | 真实评测 |

预训练数据集 [[Open X-Embodiment]] 前三大来源：Fractal（23.9%）、Kuka（12.9%）、Bridge（11.9%）。

### 实现细节

- **VLM Backbone**: [[PaliGemma]]（2B 参数，基于 Gemma）
- **总参数量**: ~4B（含动作模块）
- **MoE 配置**: N=32 专家，top-K=4 激活，每层含 1 个共享专家
- **观测输入**: 第三视角 RGB + 2 路腕部 RGB（共 3 路相机）
- **硬件**: 16 × A100 GPU + DeepSpeed 分布式训练
- **优化**: 两阶段热身（先微调 MoE 参数，再全参数微调，详见 Appendix C.6）
- **正则化温度**: $\beta = 0.1$（AS-Reg）

---

## 批判性思考

### 优点

1. **系统性分层解耦**: 首次明确区分"动作空间异构性"与"其他具身/场景异构性"，并用不同 MoE 层分别处理，设计动机清晰
2. **实验验证充分**: 通过控制实验（Table 5、7）直接证明了负迁移的存在及其被消除，因果关系明确
3. **通用性强**: 与 FLOWER 框架结合后性能进一步提升（4.49），说明 HiMoE 可作为即插即用模块

### 局限性

1. **计算开销**: N=32 专家 + top-4 激活在推理时需要更多内存；实际推理速度未在论文中报告
2. **层级深度超参数**: AS-MoE 和 HB-MoE 的层数分配（几层各自占据）依赖消融实验确定，可能对不同任务有不同的最优配置
3. **真实环境任务多样性有限**: 仅评测了 3 类单臂任务和 3 类双臂任务，与仿真的任务多样性相比偏少

### 潜在改进方向

1. **动态层级分配**: 自适应地学习在哪一层放置 AS-MoE/HB-MoE，而非手动指定
2. **在线专家扩展**: 针对新具身类型动态添加专家，无需重新预训练
3. **与更大规模 VLM 的结合**: 当前使用 PaliGemma（2B），结合更大模型（如 7B+）能否进一步提升待验证

### 可复现性评估

- [x] 代码开源（GitHub: ZhiyingDu/HiMoE-VLA）
- [x] 预训练模型（论文提及公开）
- [x] 训练细节完整（Appendix B、C.6 有详细说明）
- [x] 数据集可获取（[[Open X-Embodiment]] 为公开数据集）

---

## 关联笔记

### 基于

- [[pi0]]: 直接基于 π₀ 的架构（PaliGemma + Flow Matching 动作模块），HiMoE-VLA 是其在多数据源协训练方向的扩展
- [[CFM]]: Flow Matching 训练目标来自条件流匹配框架
- [[Mixture-of-Experts]]: HiMoE 的核心稀疏架构组件

### 对比

- [[pi0]]: 主要基线，在 CALVIN、LIBERO 和真实机器人任务上均超越
- [[Open X-Embodiment]]: 预训练数据集，各方法共享

### 方法相关

- [[Mixture-of-Experts]]: 核心稀疏架构
- [[CFM]]: 流匹配动作生成目标
- [[Contrastive Loss]]: AS-Reg 的实现基础
- [[Load Balancing Loss]]: HB-Reg 参照 DeepSeekMoE
- [[Action Chunking]]: 动作输出格式
- [[Co-training]]: 多数据集联合训练范式
- [[Negative Transfer]]: 本文核心解决的问题

### 硬件/数据相关

- [[ALOHA]]: 双臂机器人平台，实验主要硬件之一
- [[Open X-Embodiment]]: 24.1M 帧预训练数据集
- [[CALVIN]]: 仿真 benchmark
- [[LIBERO]]: 仿真 benchmark
- [[PaliGemma]]: VLM 骨干

---

## 速查卡片

> [!summary] HiMoE-VLA
> - **核心**: 层级 MoE 将动作空间异构性与具身异构性分层解耦，将负迁移转化为正迁移
> - **方法**: AS-MoE（边界）+ HB-MoE（邻近）+ Dense（中心）+ AS-Reg + HB-Reg
> - **结果**: CALVIN 3.98、LIBERO 98.0%、xArm7 75.0%、ALOHA 63.7%
> - **代码**: https://github.com/ZhiyingDu/HiMoE-VLA

---

*笔记创建时间: 2026-07-10*

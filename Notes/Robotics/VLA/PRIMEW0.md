---
title: "PRIME W0: Physical Priors from World Models for Embodied Manipulation"
method_name: "PRIMEW0"
authors: [Huabin Peng, Fagang Jin, Yancong Wei, Tengfei Ma, Chuhang Liu, Chao Yu, Liqing Chen]
year: 2026
venue: arXiv
tags: [vla, world-model, flow-matching, mixture-of-experts, bimanual-manipulation, action-tokenizer, temporal-reasoning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.39528
created: 2026-10-31
---

# 论文笔记：PRIME W0: Physical Priors from World Models for Embodied Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Lumos SZ Lab Team, Lumos Robotics, THU, CUHK, NTU |
| 日期 | October 2026 |
| 项目主页 | N/A |
| 对比基线 | [[π0.5]], [[OpenVLA]], [[GR00T N1.6]], [[EO-1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.39528) |

---

## 一句话总结

> PRIME W0 通过逐层 KV 注入、异步世界模型子目标条件化和协方差结构化流匹配噪声，构建了感知-预测-控制三路协同的分层 VLA，在 LIBERO（99.1%）和真机任务（78.1%）上达到 SOTA。

---

## 核心贡献

1. **Per-layer KV 注入（Adaptive MoE 动作专家）**: 将预训练 [[VLA]] 的所有层 KV 投影通过交叉注意力注入 [[MoE|Adaptive MoE 动作专家]]，避免"语义压缩"问题，充分利用 VLM 的层次化表征
2. **动态感知条件化（Dynamics-aware Conditioning）**: 异步世界模型预测子目标 RGB 图作为视觉 token 注入；基于 [[DINOv2]] 的历史帧编码器将过去 τ 帧投影到 LM 嵌入空间，解决"时间近视"问题
3. **[[ADM|Action Dynamic Manifold]] 流匹配**: 基于演示统计量估计的协方差结构化噪声（Cholesky 分解）替代各向同性高斯噪声，捕捉机器人运动学耦合的内在流形

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在长时域任务和精细操作中存在两个根本缺陷：

1. **时间近视（Temporal Myopia）**: 仅依赖当前观测预测动作，缺乏对历史状态和未来子目标的感知
2. **语义压缩（Semantic Compression）**: 动作解码器只接入 VLM 的最后一层池化向量，丢失了中间层的丰富层次化表征

### 现有方法的局限

- [[OpenVLA]] 等端到端 VLA：无历史建模，动作序列不连贯
- [[π0]] 等扩散策略：流匹配使用各向同性噪声，未考虑机器人关节耦合
- [[MemoryVLA]] 等历史感知方法：历史编码与策略解耦，缺乏语义层面的历史整合

### 本文的动机

同时解决三个层面的问题：(1) 语义层面——利用 VLM 所有层的表征；(2) 时序层面——通过历史帧编码和世界模型子目标预测感知过去和未来；(3) 物理层面——用数据驱动的协方差噪声尊重机器人运动学约束。

---

## 方法详解

### 模型架构

PRIME W0 采用**分层条件流匹配**架构：

- **输入**: 多视角图像 + 历史窗口（τ=8 帧）+ 本体感知 + 语言指令
- **Backbone**: 预训练 [[VLM]]（LLaVA 风格），输出所有层的 KV 投影
- **核心模块**: [[KV-Cache Conditioning|Per-layer KV 注入]] → [[MoE|Adaptive MoE 动作专家]] → [[ADM|ADM 流匹配解码器]]
- **世界模型**: 时空 VAE + 流式去噪器，异步预测子目标图像
- **历史编码器**: 冻结 [[DINOv2]] 特征 + LayerNorm 投影层（τ=8, d=4096）
- **输出**: 20 维双臂笛卡尔动作块（位置 + 6D 旋转 + 夹爪开度，每臂各 10 维）
- **动作块**: H=16，执行窗口 $h_{exec}=8$

### 核心模块

#### 模块 1: DAT（Discrete Action Tokenizer）

**设计动机**: 传统连续动作表示难以与 VLM 的离散 token 范式对齐，现有 [[FAST]] 产生可变长度 token 不稳定。

**具体实现**:
- 注册瓶颈编码器（Register-bottleneck encoder）+ 非对称注意力
- [[FSQ|有限标量量化（FSQ）]] 替代 VQ-VAE，产生固定长度离散 token
- 交叉注意力解码器重建动作序列
- 与下游 [[Action Chunking|动作块]] 解码对齐，便于 NTP 预训练

#### 模块 2: Per-layer KV 注入

**设计动机**: 利用 [[KV-Cache Conditioning|KV 注入]] 将 VLM 所有层的语义信息传递给动作专家，而非仅使用最后一层输出。

**具体实现**:
- 从 VLM 每一层提取 Key/Value 投影
- 通过交叉注意力注入 Adaptive [[MoE]] 动作专家的各解码层
- 实现语言指令到动作的细粒度语义对齐

#### 模块 3: 动态感知条件化（Dynamics-aware Conditioning）

**子目标预测（World Model）**:
- 时空 VAE 编码当前观测，流式去噪器预测未来子目标 RGB 图
- **异步注入**: 世界模型以更慢的频率运行，预测的子目标图像异步注入为视觉 token
- 世界模型参数量 5M–300M（消融研究中缩放）

**历史帧编码器**:
- 取过去 τ=8 帧，通过冻结 [[DINOv2]] 提取语义特征
- LayerNorm 投影到 LM 嵌入空间（d=4096）
- 汇聚为历史向量 $h_{hist}$，输入 MoE 门控网络

#### 模块 4: Adaptive MoE 动作专家

**架构**:
- 6 个解码层，每层 8 个专家，Top-2 路由
- **解耦设计**: 专家选择（gate）和加权（weight）使用独立子网络
- **相位感知路由**: 门控输入 = 当前 token + 历史汇聚向量 $h_{hist}$
- 负载均衡损失权重 α=0.01

#### 模块 5: ADM（Action Dynamic Manifold）流匹配

**设计动机**: 标准 [[Flow Matching]] 使用各向同性高斯噪声 $\mathcal{N}(0, I)$，忽视了双臂机器人关节之间的运动学耦合。

**具体实现**:
- 从 100 万个演示动作块离线估计协方差矩阵 $\hat{\Sigma}$
- Cholesky 分解：$\hat{\Sigma} = LL^T$
- 推理时从 $\mathcal{N}(0, \hat{\Sigma})$ 采样起点噪声
- 6 层 Transformer，512 维隐状态，8 注意力头
- 8 步 Euler ODE 积分

---

## 关键公式

### 公式 1: [[Flow Matching|条件流匹配目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, a_0 \sim \mathcal{N}(0,\hat{\Sigma}),\, a_1 \sim p_{data}} \Big[\| v_\theta(a_\tau, \tau, c) - (a_1 - a_0) \|^2\Big]
$$

**含义**: 训练流场估计网络 $v_\theta$ 预测从噪声 $a_0$ 到真实动作 $a_1$ 的直线路径切向量，条件 $c$ 包含 VLM KV 注入、子目标图像和历史编码。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间步，均匀采样
- $a_\tau = (1-\tau)a_0 + \tau a_1$: 线性插值中间状态
- $a_0 \sim \mathcal{N}(0, \hat{\Sigma})$: 协方差结构化噪声起点
- $a_1 \sim p_{data}$: 真实演示动作块
- $v_\theta$: ADM 解码器（6 层 Transformer）
- $c$: 多源条件信号

### 公式 2: [[ADM|协方差结构化噪声采样]]

$$
\hat{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (a^{(i)} - \bar{a})(a^{(i)} - \bar{a})^T, \quad \hat{\Sigma} = LL^T
$$

**含义**: 从 N=100 万个演示动作块统计协方差，Cholesky 分解后用于结构化采样，使初始噪声反映真实机器人运动学约束。

**符号说明**:
- $a^{(i)}$: 第 $i$ 个动作块（20 维双臂笛卡尔命令）
- $\bar{a}$: 动作均值
- $L$: 下三角 Cholesky 因子
- 推理时 $a_0 = L\epsilon,\; \epsilon \sim \mathcal{N}(0, I)$

### 公式 3: [[DAT|DAT 重建损失]]

$$
\mathcal{L}_{DAT} = \underbrace{\text{SmoothL1}(\hat{p}, p)}_{\text{位置}} + \underbrace{d_{geo}(\hat{R}, R)}_{\text{旋转}} + \underbrace{\text{BCE}(\hat{g}, g)}_{\text{夹爪}} + \underbrace{\mathcal{L}_{freq}}_{\text{DCT 频域}} + \underbrace{\mathcal{L}_{dist}}_{\text{双臂间距}}
$$

**含义**: DAT 的多组件重建损失，分别监督位置、旋转（SO(3) 测地距离）、夹爪、频域特性和双臂协调性。

**符号说明**:
- $\hat{p}, p$: 预测/真实末端执行器位置
- $d_{geo}(\hat{R}, R) = \|\log(\hat{R}^T R)\|_F$: SO(3) 上的测地线距离
- $\hat{g}, g$: 预测/真实夹爪开度
- $\mathcal{L}_{freq}$: DCT 系数域频率损失，确保动作轨迹平滑性
- $\mathcal{L}_{dist}$: 左右双臂末端执行器间距约束损失

### 公式 4: [[MoE|MoE 负载均衡损失]]

$$
\mathcal{L}_{balance} = \alpha \sum_{e=1}^{E} f_e \cdot p_e, \quad \alpha = 0.01
$$

**含义**: 鼓励各专家均匀分配负载，防止少数专家主导路由。

**符号说明**:
- $E$: 专家总数（每层 8 个）
- $f_e$: 第 $e$ 个专家的实际激活比例
- $p_e$: 第 $e$ 个专家的平均路由概率
- $\alpha = 0.01$: 均衡损失权重

### 公式 5: 端到端训练总损失（Stage 4 SFT）

$$
\mathcal{L}_{total} = \mathcal{L}_{FM} + \alpha \mathcal{L}_{balance} + \lambda_{NTP} \mathcal{L}_{NTP} + \lambda_{WM} \mathcal{L}_{WM}
$$

**含义**: 四阶段训练最终阶段的复合损失，同时优化流匹配质量、专家负载均衡、VLM 语言能力保持和世界模型一致性。

**符号说明**:
- $\mathcal{L}_{FM}$: ADM 流匹配主损失
- $\mathcal{L}_{balance}$: MoE 负载均衡辅助损失
- $\mathcal{L}_{NTP}$: 下一 token 预测损失（维持 VLM 语言能力）
- $\mathcal{L}_{WM}$: 世界模型子目标一致性损失

---

## 关键图表

### Figure 1: Model Architecture / PRIME W0 整体架构

![Figure 1 - PRIME W0 Architecture](https://arxiv.org/html/2610.39528v1/figures/model_structure.png)

**说明**: PRIME W0 的整体架构图。输入多视角图像、历史帧、本体感知和语言指令，经 VLM 提取逐层 KV，注入 Adaptive MoE 动作专家；异步世界模型预测子目标图像作为额外视觉 token；ADM 流匹配解码器输出 20 维双臂动作块。

### Figure 2: Training Pipeline / 四阶段训练流程

![Figure 2 - Training Pipeline](https://arxiv.org/html/2610.39528v1/figures/fig_pretrain_pipeline.png)

**说明**: 展示 PRIME W0 的四阶段训练：(1) VLM 预训练（图文描述和 VQA）；(2) DAT + NTP 视觉运动预训练（冻结 VLM 骨干）；(3) ADM 流匹配动作专家预训练（VLM 冻结，加入世界模型注入和历史编码器）；(4) 端到端 SFT（复合损失）。图标标注了可训练模块和冻结模块。

### Figure 3: Data Mixture / 训练数据构成

![Figure 3 - Data Mixture](https://arxiv.org/html/2610.39528v1/figures/fig_data_mixture.png)

**说明**: 约 10,000 小时机器人数据的来源分布。遥操作数据占 60%（~6,000 h），UMI 手持自主采集占 30%（~3,000 h），仿真数据占 10%（~1,000 h）。训练中随机遮蔽第三人称视角，使策略兼容多视角和纯 ego 视角输入。

### Figure 4: Cross-Attention 时序分析

（内嵌折线图，无独立 URL）

**说明**: 展示 ADM 解码器对世界模型子目标图像 token 的交叉注意力强度随操作阶段的变化。在接触时刻（contact）和放置阶段（placement）注意力权重显著上升，说明策略在关键时刻主动利用前瞻性子目标信息。PRIME W0 的路由熵在接触时刻下降 47%，说明 MoE 专家在关键阶段形成了专业化分工。

### Figure 5: World Model Scaling / 世界模型容量消融

（内嵌折线图，无独立 URL）

**说明**: LIBERO-Long 成功率随世界模型参数量（5M → 50M → 150M → 300M）单调递增，5M→50M 增幅最大（+5.9%），表明世界模型容量对子目标预测质量和策略性能有直接影响，但边际收益递减。

### Table 1: LIBERO 基准对比

| Method | Spatial | Object | Goal | Long | **Average** |
|--------|---------|--------|------|------|-------------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| Discrete Diffusion VLA | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| EO-1 | 99.7 | 99.8 | 99.2 | 94.8 | 98.4 |
| **PRIME W0（Ours）** | **98.8** | **100.0** | **99.0** | **98.6** | **99.1** |

**关键发现**: PRIME W0 在 LIBERO 四套任务上平均达 99.1%，超越前最优 EO-1（98.4%）。LIBERO-Long（+3.8%）提升最大，验证了历史编码和世界模型对长时域任务的价值。

### Table 2: 真机任务对比

| 任务 | OpenVLA | GR00T-N1.6 | π0.5 | **PRIME W0** |
|------|---------|------------|------|--------------|
| Pick-and-place | 39.1 | 75.3 | 81.4 | **93.7** |
| Stacking | 25.2 | 61.8 | 67.9 | **70.1** |
| Cloth folding | 4.6 | 21.2 | 85.4 | **89.1** |
| Small obj. grasp | 8.3 | 26.7 | 59.2 | **61.8** |
| Assembly | 1.4 | 22.1 | 51.8 | **75.9** |
| **Average** | **15.7** | **41.4** | **69.1** | **78.1** |

**关键发现**: PRIME W0 在真机 5 项任务全部超越 π0.5，Assembly 任务上提升尤为显著（+24.1%），说明分层动作分解和子目标引导对精细装配任务有显著优势。

### Table 3: 时序与动态行为量化

| Method | D1: $r$ (子目标相关) | D2: JR% (抖动减少) | D3: PE% (相位路由熵降) |
|--------|---------------------|--------------------|----------------------|
| OpenVLA | 0.09 | 3.2 | 4.1 |
| π0 | 0.14 | 8.7 | 9.3 |
| Seer | 0.22 | 11.4 | 12.6 |
| **PRIME W0** | **0.79** | **34.0** | **47.0** |

**关键发现**:
- **D1**: 子目标对齐与任务成功率的 Pearson 相关系数达 0.79（OpenVLA 仅 0.09），说明世界模型预测对任务完成有实质性引导
- **D2**: 历史编码将 LIBERO-Long 的归一化抖动减少 34%，布料折叠减少 41%，动作轨迹显著平滑
- **D3**: 接触时刻 MoE 路由熵下降 47%，专家在关键阶段形成专业化分工

### Table 4: 消融实验（逐步累积）

| 配置 | LIBERO-Long (%) | 真机 (%) |
|------|-----------------|---------|
| Base VLA (OpenVLA) | 53.7 | 15.7 |
| + DAT & NTP (Stage 2) | 61.2 | 22.4 |
| + ADM（各向同性噪声） | 72.4 | 45.3 |
| + Adaptive MoE | 81.6 | 58.2 |
| + 历史帧编码器 | 88.3 | 64.7 |
| + WM 子目标注入（异步） | 93.5 | 71.4 |
| + 协方差结构化噪声 | 96.2 | 75.8 |
| **PRIME W0 完整版（Stage 4 SFT）** | **98.6** | **78.1** |

**关键发现**: 每个组件均有正向贡献。历史编码器（+6.7%）和 WM 注入（+5.2%）对 LIBERO-Long 贡献最大；协方差噪声额外提升 2.7%，说明运动学感知噪声初始化有效。从 53.7% 到 98.6%，全组件叠加总提升达 44.9 个百分点。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 遥操作 | ~6,000 h (60%) | 双臂真机操作，高质量演示 | 训练 |
| UMI 手持采集 | ~3,000 h (30%) | 自主 ego 视角，场景多样 | 训练 |
| 仿真数据 | ~1,000 h (10%) | 低成本大规模采集 | 训练 |
| [[LIBERO]] | 4 套任务 | Spatial/Object/Goal/Long | 评测 |
| 真机任务（自建） | 5 类任务 | 抓取/堆叠/折叠/小物/装配 | 评测 |

### 实现细节

- **动作空间**: 20 维双臂笛卡尔命令（位置 3D + 6D 旋转 + 夹爪，各臂 10 维）
- **历史窗口**: τ = 8 帧
- **历史编码维度**: d = 4096（LM 嵌入空间）
- **Adaptive MoE**: 6 层解码，8 专家/层，Top-2 路由，均衡损失 α=0.01
- **ADM 解码器**: 6 层 Transformer，512 维隐状态，8 头，8 步 Euler ODE
- **世界模型**: 时空 VAE + 流式去噪器，5M–300M 参数（消融缩放研究）
- **动作块**: H=16，执行窗口 $h_{exec}=8$
- **协方差估计**: 离线从 100 万动作块统计，Cholesky 分解

### 可视化结果

- 世界模型子目标预测图在各操作阶段（接触/放置/完成）语义一致性强，r=0.79
- 历史编码有效抑制轨迹高频抖动，布料折叠任务中动作轨迹视觉更平滑
- MoE 相位感知路由在接触时刻自动激活特定专家子集（路由熵下降 47%）

---

## 批判性思考

### 优点

1. **三路协同设计完整**: 语义（KV 注入）+ 时序（历史/子目标）+ 物理（协方差噪声）三个维度全面覆盖，各有独立消融验证
2. **异步世界模型注入**: 世界模型与策略解耦的异步推理设计，避免串行等待，实际可部署
3. **双臂任务全面覆盖**: 在 LIBERO 和 5 类真机任务均超越 SOTA，泛化性强

### 局限性

1. **子目标保真度**: 世界模型在分布外场景或快速变化场景下可能预测失准，进而误导策略
2. **固定历史窗口**: τ=8 帧的固定窗口难以处理超长时域记忆需求（>8 步）
3. **MoE 路由偏移**: 在任务分布不均衡的数据集上，专家路由仍可能出现偏斜
4. **无触觉/力矩感知**: 仅使用视觉-语言-本体感知，精细装配等高接触任务仍受限

### 潜在改进方向

1. 自适应历史窗口（可变 τ）：根据任务复杂度动态扩展记忆范围
2. 多模态世界模型：引入触觉预测提升接触相位准确性
3. 在线协方差更新：根据当前任务实时更新协方差估计，适应新场景

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（四阶段流程、超参数完整）
- [ ] 数据集可获取（自建数据集，未公开）

---

## 关联笔记

### 基于

- [[OpenVLA]]: 消融实验起点 baseline，PRIME W0 在其基础上逐步叠加组件
- [[π0]]: 流匹配 VLA 的先驱，ADM 在其 flow matching 基础上引入协方差噪声
- [[Flow Matching]]: ADM 解码器的核心算法框架
- [[DINOv2]]: 历史帧语义编码器的骨干网络

### 对比

- [[π0.5]]: 同期最强 VLA baseline，真机平均 69.1% vs PRIME W0 78.1%
- [[GR00T N1.6]]: NVIDIA 开源 VLA，真机平均 41.4% vs PRIME W0 78.1%
- [[OpenVLA-OFT]]: LIBERO 上 97.1% vs PRIME W0 99.1%
- [[MemoryVLA]]: 历史感知 VLA 对比，96.7% vs 99.1%（LIBERO 平均）

### 方法相关

- [[ADM]]: PRIME W0 的核心动作解码器（本文提出，另见概念笔记）
- [[MoE]]: Adaptive MoE 动作专家的混合专家路由机制
- [[FSQ]]: DAT 离散量化的关键技术
- [[Action Chunking]]: PRIME W0 输出的动作块预测范式
- [[World Model]]: 异步子目标图像预测的世界模型组件
- [[KV-Cache Conditioning]]: Per-layer KV 注入的技术基础

### 硬件/数据相关

- [[LIBERO]]: 主评测基准（Spatial/Object/Goal/Long 四套任务）

---

## 速查卡片

> [!summary] PRIME W0 (2026)
> - **核心**: 三路协同 VLA——逐层 KV 注入 + 异步世界模型子目标 + 协方差流匹配
> - **方法**: Adaptive MoE 动作专家 + DAT 离散动作 tokenizer + ADM 流匹配
> - **结果**: LIBERO 99.1% SOTA，真机 5 任务平均 78.1%（>π0.5 的 69.1%）
> - **代码**: 未公开（arXiv:2610.39528）

---

*笔记创建时间: 2026-10-31*

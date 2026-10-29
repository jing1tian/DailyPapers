---
title: "PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations"
method_name: "PDRVLA"
authors: [Hanqing Wang, Zhaotian Cai, Ruichuan An, Bohan Zeng, Tao Zhang, Yilin Chen, Xu Jiang, Kaiquan Chen, Chongyang Wu, Haoran Li, Fan Yang, Mingan Lin, Wentao Zhang, Zenan Zhou, Youzheng Wu]
year: 2026
venue: arXiv
tags: [vision-language-action, physical-dynamics, flow-matching, cross-embodiment, embodied-chain-of-thought, robot-manipulation, diffusion-transformer]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.37381v1
created: 2026-10-29
---

# 论文笔记：PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, JD.com |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[Qwen-RobotManip]] / [[InternVLA-A1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.37381) / Code: — |

---

## 一句话总结

> PDRVLA 通过轻量级 [[Physical Dynamics Module]] 从 VLM 特征中提取物理 token，无需在线生成视频即可将物理动态理解注入 [[VLA|视觉语言动作模型]]，在四个仿真 benchmark 上达到 SOTA。

---

## 核心贡献

1. **Physical Dynamics Module**: 使用 50 个可学习查询从 VLM 第 8 层特征中提取物理 token，通过预测未来 RGB、深度、[[光流]]特征，并由冻结的视频世界模型辅助监督，将物理动态感知注入动作生成。
2. **跨具身公开数据语料库**: 整合真实机器人、UMI、仿真轨迹共约 430 万条，通过 [[Human-to-Robot Synthesis|人类到机器人数据合成]]（EgoDex）、[[Subtask Conditioning|子任务标注]]和 [[ECoT|具身思维链]]标注大幅扩充训练数据。
3. **四 benchmark SOTA**: 在 LIBERO (99.3%)、LIBERO-Plus (89.9%)、RoboTwin 2.0 (93.6%)、SimplerEnv (81.5%) 上均超越现有方法。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型在操控任务中缺乏对物理动态的显式理解（如物体运动轨迹、深度变化、接触状态），导致在复杂扰动下泛化能力不足。

### 现有方法的局限

- 视频世界模型（如 LingBot-Video）可以捕获场景动态，但需要在线生成视频帧，引入控制循环延迟，难以部署。
- 直接预测未来特征已有尝试，但缺乏高质量监督信号导致学习效果有限。

### 本文的动机

通过让 Physical Dynamics Module 预测未来多模态特征（RGB、深度、[[光流]]），并借助冻结的[[视频世界模型]]提供蒸馏式监督信号，实现无延迟的物理动态感知；同时通过大规模公开数据扩充和丰富标注（子任务、ECoT）提升跨场景泛化。

---

## 方法详解

### 模型架构

PDRVLA 采用 **[[VLA]] + 轻量物理模块 + [[DiT（扩散变换器）|DiT]] 动作专家** 架构：

- **输入**: 语言指令 $l$ + 多视角图像观测 $o_t$ + 机器人状态 $s_t$
- **Backbone**: Qwen3.5-4B（[[大语言模型|LLM]] 骨干）
- **物理模块**: [[Physical Dynamics Module]]（约 57.3M 参数）从第 8 层 VLM 特征提取 50 个物理 token
- **动作专家**: [[DiT（扩散变换器）|DiT]] 动作专家（约 150M 参数）用[[Flow-Matching|流匹配]]预测 8 步动作块
- **输出**: 58 维跨具身动作向量（[[Action Chunking|动作块]] $\mathbf{A}_t \in \mathbb{R}^{8 \times 58}$）
- **总参数**: Qwen3.5-4B + 57.3M（物理模块）+ 150M（动作专家）

### 核心模块

#### 模块 1: Physical Dynamics Module

**设计动机**: 无在线视频生成地实现[[视频世界模型]]级别的物理感知，避免控制循环延迟。

**具体实现**:
- 使用 50 个[[可学习查询]]对 VLM 第 8 层特征做交叉注意力（每层）
- 通过三个预测头分别预测未来 RGB、深度、[[光流]]的池化特征（标准化后的 $\widetilde{y}^m$）
- 在物理预训练阶段，同时用冻结的 LingBot-Video 模型提供[[流匹配]]监督

**物理 token 形成的条件序列**:

$$
X_t = [H_t^{\text{last}};\, C_t]
$$

其中 $H_t^{\text{last}}$ 为最终层 VLM 特征，$C_t$ 为物理 token，二者拼接后输入动作专家。

#### 模块 2: DiT 动作专家

**设计动机**: 用[[Flow-Matching|流匹配]]代替 DDPM 去噪，加速推理并提升动作预测精度。

**具体实现**:
- 输入: 条件序列 $X_t$（VLM 特征 + 物理 token）
- 从高斯噪声 $\xi$ 出发，通过 [[Flow-Matching|流匹配]] 积分生成动作块 $\mathbf{A}_t$
- 支持 58 维跨具身动作向量，含 [[End-Effector|末端执行器]] 6D 位姿增量 + 夹爪 + 关节角度

#### 模块 3: 数据扩充流水线

**设计动机**: 现有公开数据分布不均，需要多维度扩充。

**具体实现**:
- **[[Human-to-Robot Synthesis|人类到机器人合成]]**: 将 EgoDex 第一人称数据映射到 12 款机器人臂，用 SAM3 分割手臂、ProPainter 重建背景、DA3 深度辅助渲染
- **[[Subtask Conditioning|子任务标注]]**: 使用 Qwen3.8-27B 对稀疏采样帧进行滑动窗口标注（窗口 128 帧，重叠 16 帧），生成 6 种查询变体的单轮 VL 样本
- **[[ECoT|具身思维链]]标注**: 关键帧描述任务相关物体、空间关系、夹爪状态和下一步运动方向

---

## 关键公式

### 公式 1: [[End-Effector|末端执行器动作增量]]

$$
\Delta a^{\text{eef}}_t = (\Delta\mathbf{p}_{r,t},\, \Delta R_{r,t}),\quad \Delta\mathbf{p}_{r,t} = \mathbf{p}^*_{r,t} - \mathbf{p}_{r,t},\quad \Delta R_{r,t} = R^*_{re,t} R_{re,t}^\mathsf{T}
$$

**含义**: 在参考系下，当前帧到目标帧的末端执行器平移量和旋转量（用旋转矩阵之积表达的相对旋转）。

**符号说明**:
- $\Delta\mathbf{p}_{r,t}$: 参考系下的位移增量
- $\Delta R_{r,t}$: 旋转增量（旋转矩阵形式）
- $R^*_{re,t}$: 目标旋转矩阵，$R_{re,t}$: 当前旋转矩阵

---

### 公式 2: [[Action Chunking|块相对 EEF 动作（6D）]]

$$
\mathbf{a}^{\text{eef}}_{t,k} = \left[\mathbf{p}^*_{t+k} - \mathbf{p}_t,\; \operatorname{Log}\!\left(R^*_{t+k} R_t^\mathsf{T}\right)\right] \in \mathbb{R}^6
$$

**含义**: 动作块内第 $k$ 步的目标位姿相对块起始位姿的 6 维表达，旋转用轴角向量表示。

**符号说明**:
- $\mathbf{p}^*_{t+k}$: 第 $t+k$ 步的目标位置，$\mathbf{p}_t$: 块起始位置
- $\operatorname{Log}(\cdot)$: SO(3) 对数映射，输出轴角向量

---

### 公式 3: [[Physical Dynamics Module|物理 token 条件序列]]

$$
X_t = [H_t^{\text{last}};\, C_t]
$$

**含义**: 最终层 VLM 特征与物理 token 拼接形成动作专家的完整条件输入。

**符号说明**:
- $H_t^{\text{last}}$: VLM 最终层特征序列
- $C_t$: Physical Dynamics Module 生成的物理 token（50 个）

---

### 公式 4: [[Physical Dynamics Module|目标标准化]]

$$
\widetilde{y}^m = \frac{y^m - \mu^m}{s^m}
$$

**含义**: 对每个模态（RGB/深度/[[光流]]）的池化预测目标进行逐通道、逐空间位置标准化，消除量纲差异。

**符号说明**:
- $y^m$: 模态 $m$ 的池化特征目标
- $\mu^m, s^m$: 该模态的统计均值和标准差

---

### 公式 5: [[Physical Dynamics Module|直接预测损失]]

$$
\mathcal{L}_{\text{direct}} = \frac{1}{3} \sum_{m \in \{r, d, f\}} \operatorname{MSE}\!\left(g_\eta^m(C_t),\, \widetilde{y}^m\right)
$$

**含义**: 对 RGB ($r$)、深度 ($d$)、[[光流]] ($f$) 三个模态的标准化未来特征预测取平均 MSE。

**符号说明**:
- $g_\eta^m(\cdot)$: 模态 $m$ 对应的预测头（参数 $\eta$）
- $C_t$: 物理 token
- $\widetilde{y}^m$: 标准化后的未来特征目标

---

### 公式 6: [[视频世界模型|加噪未来 RGB 潜变量]]

$$
z^r_{\sigma,\text{current}} = z^r_{\text{current}},\qquad z^r_{\sigma,\text{future}} = (1-\sigma)z^r_{\text{future}} + \sigma\epsilon
$$

**含义**: 当前帧潜变量保持干净，未来帧潜变量以比例 $\sigma$ 混入高斯噪声，供冻结世界模型执行[[Flow-Matching|流匹配]]去噪以提供监督。

**符号说明**:
- $z^r$: VAE 编码后的 RGB 潜变量
- $\sigma$: 噪声水平（0 到 1）
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声

---

### 公式 7: [[视频世界模型|世界模型速度损失]]

$$
\mathcal{L}_{\text{WM}} = \sum_{m \in \{r, d, f\}} \operatorname{MSE}\!\left(\widehat{v}^m_{\text{future}},\, \epsilon - z^m_{\text{future}}\right)
$$

**含义**: 对三个模态等权重计算[[Flow-Matching|流匹配]]速度预测误差，来自冻结的 LingBot-Video 模型。

**符号说明**:
- $\widehat{v}^m_{\text{future}}$: Physical Dynamics Module 预测的流速
- $\epsilon - z^m_{\text{future}}$: 流匹配目标速度（噪声减干净潜变量）

---

### 公式 8: [[Physical Dynamics Module|物理预训练目标]]

$$
\mathcal{L}_{\text{pre}} = \mathcal{L}_{\text{WM}} + \lambda(k)\,\mathcal{L}_{\text{direct}},\qquad \lambda(k) = 0.5\min(k/2000,\, 1)
$$

**含义**: 物理预训练的总目标，直接预测损失的权重在前 2,000 步从 0 线性增至 0.5，避免初期直接损失主导训练。

**符号说明**:
- $\lambda(k)$: 第 $k$ 步时直接预测损失的权重
- $\mathcal{L}_{\text{WM}}$: 世界模型速度损失
- $\mathcal{L}_{\text{direct}}$: 直接预测损失

---

### 公式 9: [[Flow-Matching|流匹配插值与目标速度]]

$$
x_\tau = (1-\tau)\xi + \tau\mathbf{A}_t,\qquad v^* = \mathbf{A}_t - \xi
$$

**含义**: 动作块从噪声 $\xi$ 到目标 $\mathbf{A}_t$ 的线性插值路径，常数目标速度为动作减噪声。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间步
- $\xi \sim \mathcal{N}(0, I)$: 初始高斯噪声
- $\mathbf{A}_t$: 目标动作块
- $v^*$: 常数目标速度

---

### 公式 10: [[Action Chunking|带掩码的动作损失]]

$$
\mathcal{L}_{\text{act}} = \frac{\sum_{i \in \mathcal{B}} \|M_i \odot (\widehat{v}_i - v_i^*)\|_F^2}{\sum_{i \in \mathcal{B}} \|M_i\|_1 + \delta}
$$

**含义**: 仅对有监督的动作维度（由掩码 $M_i$ 指示）计算速度预测误差，适配跨具身训练中不同机器人的动作空间。

**符号说明**:
- $M_i$: 样本 $i$ 的动作维度掩码（有监督=1，否则=0）
- $\widehat{v}_i$: 预测速度，$v_i^*$: 目标速度
- $\delta$: 防止除零的小常数

---

### 公式 11: [[VLA|联合策略目标]]

$$
\mathcal{L}_{\text{policy}} = \mathcal{L}_{\text{act}} + 0.1\,\mathcal{L}_{\text{VL}} + 0.1\,\mathcal{L}_{\text{direct}}
$$

**含义**: 联合策略训练的总损失，同时优化动作预测、视觉语言理解和物理特征直接预测三个目标。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 掩码动作速度损失
- $\mathcal{L}_{\text{VL}}$: 视觉语言生成损失（next-token prediction）
- $\mathcal{L}_{\text{direct}}$: 物理特征直接预测损失

---

## 关键图表

### Figure 1: PDRVLA 系统概览

![Figure 1](https://arxiv.org/html/2610.37381v1/intro_overview_v3.png)

**说明**: (a) PDRVLA 简化架构图：[[Physical Dynamics Module]] 从 VLM 特征中提取物理 token，与最终层特征拼接后输入 [[DiT（扩散变换器）|DiT]] 动作专家；(b) 四个仿真 benchmark 的成功率对比，PDRVLA 在所有 benchmark 上达到 SOTA。

---

### Figure 2: 六种子任务 VL 查询变体

**说明**: 展示 [[Subtask Conditioning|子任务条件化]]的六种 VL 样本形式，在查询时机（帧级 / 轨迹级）、答案内容（当前子任务 / 下一子任务 / 全程规划）和历史上下文方面各有不同，训练时均不提供查询帧之后的观测。

---

### Figure 3: PDRVLA 模型架构详图

![Figure 3](https://arxiv.org/html/2610.37381v1/model_architecture_v9.png)

**说明**: (a) 物理 token 补充最终层 VLM 特征后输入动作专家，动作专家将高斯噪声通过[[Flow-Matching|流匹配]]积分为动作块；(b) [[可学习查询]]对 VLM 共享特征的每层做交叉注意力，经投影得到物理 token。

---

### Figure 4: 两阶段训练流程

**说明**: (a) **物理预训练**：直接预测未来特征 + 冻结[[视频世界模型]]（LingBot-Video）提供[[Flow-Matching|流匹配]]监督；(b) **联合策略训练**：去除世界模型，加入动作损失和 VL 损失，保留直接预测损失。

---

### Figure 5: 直接预测 vs. 直接预测 + 世界模型监督的训练 MSE

![Figure 5](https://arxiv.org/html/2610.37381v1/figures/physical_wm_training_main.png)

**说明**: 在 RoboCOIN 验证集上，加入世界模型监督后 RGB 预测 MSE 下降约 4.1%，深度下降约 2.1%，[[光流]]下降约 0.6%（不显著），整体平均下降约 1.3%。

---

### Table 1: 训练语料库概览

| 类别 | 统计 |
|------|------|
| Real-Robot (VLA) | 1.26M 轨迹 / 14,893 小时 |
| UMI (VLA) | 1.22M 轨迹 / 10,259 小时 |
| Simulation (VLA) | 1.87M 轨迹 / 8,868 小时 |
| General VL | 8.05M 条 VL 样本 |
| Spatial & Embodied VL | 23.63M 条 VL 样本 |

**说明**: 全部使用公开数据，覆盖真实机器人、UMI 双臂和仿真三大来源。

---

### Table 2: 合成与标注数据统计

| 增强方式 | 统计 |
|----------|------|
| [[Human-to-Robot Synthesis\|Egocentric 转换]] | 0.90M 机器人轨迹 / 2,646 小时 |
| [[Subtask Conditioning\|子任务标注]] | 1.79B 标注帧 / 1.1M VL 样本 |
| [[ECoT\|具身思维链]] | 2.56M 标注关键帧 |

---

### Table 3: 子任务 VL 样本来源分布（部分）

| 来源 | 标注类型 | 样本数 |
|------|----------|--------|
| EgoDex | VLM | 125,000 |
| InternVLA-sim | VLM | 103,415 |
| ABC-130K | VLM | 93,621 |
| RoboCOIN | Native | 91,642 |
| OpenNeoData | VLM | 83,917 |
| ... | ... | ... |
| **总计** | | **1,094,945** |

---

### Table 4: ECoT 关键帧标注数据集

| 数据集 | 类型 | 标注关键帧数 |
|--------|------|------------|
| AgiBot World | 真实 | 530,301 |
| ABC-130K | 真实 | 530,289 |
| RoboCOIN | 真实 | 691,908 |
| RoboMIND | 真实 | 153,508 |
| HiFi-UMI-2K | UMI | 575,197 |
| FastUMI-100K | UMI | 83,648 |
| **总计** | | **2,564,851** |

---

### Table 5: 四个 Benchmark 总览（成功率 %）

| 方法 | LIBERO | LIBERO-Plus | RoboTwin 2.0 | SimplerEnv |
|------|--------|-------------|--------------|------------|
| π0 | 94.4 | 53.6 | 62.2 | 27.1 |
| π0.5 | 96.9 | 84.4 | 79.8 | 57.1 |
| OpenVLA-OFT | 97.1 | 69.6 | – | – |
| GR00T-N1.7 | 97.0 | – | – | 59.3 |
| Cosmos-Policy | 98.5 | 82.2 | – | – |
| LingBot-VLA 2.0 | – | – | 93.2 | – |
| Qwen-RobotManip | 99.1 | 89.0 | 93.0 | – |
| InternVLA-A1.5 | 98.9 | 84.8 | 93.2 | 80.8 |
| **PDRVLA** | **99.3** | **89.9** | **93.6** | **81.5** |

**说明**: PDRVLA 在全部四个 benchmark 上超越所有已报告基线，LIBERO-Plus 提升最显著（相对 π0.5 +5.5pp）。

---

### Table 6: LIBERO 各子任务成功率 (%)

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| π0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| Cosmos-Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| InternVLA-A1.5 | 98.6 | 99.8 | 98.6 | 98.4 | 98.9 |
| Qwen-RobotManip | – | – | – | – | 99.1 |
| **PDRVLA** | **100.0** | **99.8** | **99.8** | **97.6** | **99.3** |

---

### Table 7: LIBERO-Plus 按扰动类别成功率 (%)

| 方法 | 传感器噪声 | 摄像头视角 | 机器人初始态 | 光照 | 语言 | 背景纹理 | 物体布局 | 总体 |
|------|-----------|-----------|-------------|------|------|---------|---------|------|
| π0 | 79.0 | 13.8 | 6.0 | 85.0 | 58.8 | 81.4 | 68.9 | 53.6 |
| Qwen-RobotManip | 97.7 | 87.2 | 75.5 | 96.6 | 85.6 | 97.7 | 87.3 | 89.0 |
| InternVLA-A1.5 | 95.6 | 83.1 | 55.1 | 96.4 | 86.9 | 98.2 | 85.2 | 84.8 |
| **PDRVLA** | **95.6** | **88.2** | **75.2** | **99.1** | **90.7** | **99.4** | **86.5** | **89.9** |

---

### Table 8: RoboTwin 2.0 成功率 (%)

| 方法 | Clean | Rand. | Avg. |
|------|-------|-------|------|
| LingBot-VLA 2.0 | 93.5 | 92.8 | 93.2 |
| Qwen-RobotManip | 93.4 | 92.5 | 93.0 |
| InternVLA-A1.5 | 93.3 | 93.0 | 93.2 |
| **PDRVLA** | **93.9** | **93.3** | **93.6** |

---

### Table 9: SimplerEnv（WidowX / Bridge）成功率 (%)

| 方法 | Spoon→Towel | Carrot→Plate | Stack Blocks | Eggplant→Basket | Avg. |
|------|-------------|--------------|--------------|-----------------|------|
| π0.5 | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| Xiaomi-Robotics-0 | 95.8 | 62.5 | 75.0 | 83.3 | 79.2 |
| InternVLA-A1.5 | 92.4 | 85.4 | 69.5 | 75.7 | 80.8 |
| **PDRVLA** | **92.4** | **64.6** | **71.2** | **97.6** | **81.5** |

**说明**: Eggplant→Basket 任务 PDRVLA 大幅领先（97.6% vs. 75.7%），但 Carrot→Plate 略低于 InternVLA-A1.5。

---

### Table 10: [[视频世界模型|世界模型监督]]消融（归一化预测 MSE，RoboCOIN 验证集）

| 预测目标 | 仅直接预测 | 直接预测 + WM | 降低幅度 (%) | 95% CI (%) |
|---------|-----------|--------------|-------------|-----------|
| RGB | 0.2558 | 0.2454 | **4.06** | [3.04, 4.99] |
| 深度 | 0.3154 | 0.3087 | **2.12** | [0.93, 3.31] |
| [[光流]] | 1.3606 | 1.3522 | 0.62 | [−0.07, 1.28] |
| 均值 | 0.6439 | 0.6354 | **1.32** | [0.71, 1.89] |

**关键发现**: 世界模型监督对 RGB 和深度预测有显著提升，[[光流]]增益统计上不显著（CI 包含 0）。

---

### Table 11: [[Subtask Conditioning|子任务条件化]]消融（LIBERO-Plus 成功率 %）

| 变体 | 传感器噪声 | 摄像头视角 | 机器人初始态 | 光照 | 语言 | 背景纹理 | 物体布局 | 总体 |
|------|-----------|-----------|-------------|------|------|---------|---------|------|
| SFT 基线 | 95.57 | 88.18 | 75.16 | 99.12 | 90.70 | 99.44 | 86.49 | 89.93 |
| + 子任务条件化 | 96.50 | 89.18 | 76.32 | 98.51 | 89.92 | 99.07 | 86.75 | **90.23** |

---

### Table 12: RoboCasa365 不同子任务刷新频率成功率 (%)

| 任务 | 基线 | 每 1 步刷新 | 每 8 步刷新 | 每 32 步刷新 |
|------|------|------------|------------|------------|
| CuttingToolSelection | 46 | 60 | 50 | 33 |
| PrepareCoffee | 7 | 13 | 7 | 7 |
| LoadDishwasher | 48 | 55 | 58 | 33 |
| **平均** | **26.25** | **32.25** | **32.13** | **20.63** |

**关键发现**: 刷新频率 1 或 8 步优于基线，过低频率（32 步）反而有害。

---

### Table 13: [[ECoT|具身思维链]]训练消融（成功率 %）

| 设置 | LIBERO Long | LIBERO Avg. | LIBERO-Plus Long | LIBERO-Plus Overall |
|------|-------------|-------------|-----------------|---------------------|
| 无 ECoT 训练 | 92.8 | 97.1 | 65.18 | 79.59 |
| ECoT 训练，推理时不生成 | **96.0** | **98.2** | **69.87** | **79.25** |
| ECoT 训练，推理时生成 | 95.6 | 98.3 | 68.72 | 79.12 |

**关键发现**: ECoT 训练本身提升 LIBERO Long 成功率（+3.2pp），推理时显式生成 ECoT 无一致额外收益。

---

### Table 14: 物体位姿辅助监督消融（LIBERO-Plus 成功率 %）

| 变体 | 总体 |
|------|------|
| SFT 基线 | 89.93 |
| + 物体位姿监督 | 90.12 |

**说明**: 探索性实验，效果轻微提升但不显著。

---

### Table 15: Physical Dynamics Module 设计消融（冻结表征探针，预测技能均值↑）

| 设计选项 | 配置 | 预测技能均值 |
|---------|------|------------|
| VLM 输入层 | Block 8（选用） | **0.3617** |
| | Block 16 | 0.3413 |
| | Block 24 | 0.3314 |
| | 最终层 | 0.2432 |
| 交叉注意力位置 | 第一层 | 0.3383 |
| | 最终层 | 0.3590 |
| | 每层（选用） | **0.3617** |
| 直接预测权重调度 | 无直接监督 | 0.2683 |
| | 常数 0.1（选用） | 0.3617 |
| | 线性预热至 0.5（2k 步） | **0.3834** |
| 可学习查询数 | 50（选用） | 0.3844 |
| | 128 | 0.3815 |
| | 256 | 0.3834 |

**关键发现**: 浅层（Block 8）VLM 特征比深层包含更多物理信息；50 个查询与 256 个性能相当；权重线性预热效果更好。

---

### Table 16: LIBERO-Plus 各子任务套件成功率 (%)

| 子任务套件 | SFT 基线 | + 子任务条件 | + 物体位姿监督 |
|-----------|---------|------------|--------------|
| Spatial | 96.50 | 96.59 | 95.71 |
| Object | 92.85 | 93.21 | 93.05 |
| Goal | 83.94 | 84.06 | 85.18 |
| Long | 86.90 | 87.53 | 86.94 |
| 总体 | 89.93 | 90.23 | 90.12 |

---

### Table 17: LIBERO 中 EEF 高度统计（预训练上界 0.772 m）

| 套件 | 中位数 z (m) | z p01–p99 (m) | 高于上界 (%) |
|------|------------|--------------|------------|
| Spatial | 1.041 | 0.915–1.285 | 100.0 |
| Object | 0.225 | 0.010–0.336 | 0.0 |
| Goal | 1.076 | 0.911–1.290 | 100.0 |
| Long | 0.712 | 0.447–1.267 | 49.1 |

**说明**: Spatial/Goal 套件中 EEF 高度普遍超出预训练数据上界，揭示了预训练与评测数据分布的差异。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Real-Robot | 1.26M 轨迹 | 真实机器人多任务 | 训练 |
| UMI | 1.22M 轨迹 | 双臂 UMI 平台 | 训练 |
| Simulation | 1.87M 轨迹 | 仿真多具身 | 训练 |
| EgoDex（合成后） | 0.90M 轨迹 | 第一人称→机器人合成 | 训练 |
| LIBERO | 4 子套件 | 桌面操控，标准 benchmark | 测试 |
| LIBERO-Plus | 7 扰动类别 | LIBERO 的鲁棒性变体 | 测试 |
| RoboTwin 2.0 | Clean/Rand. | 双臂仿真 | 测试 |
| SimplerEnv | WidowX/Bridge | 跨具身迁移 | 测试 |
| RoboCOIN | — | 世界模型监督验证 | 消融 |
| RoboCasa365 | 8 任务 | 子任务刷新频率消融 | 消融 |

### 实现细节

- **Backbone**: Qwen3.5-4B（冻结部分层）
- **物理模块**: 50 个可学习查询 + 每层[[交叉注意力]] + 3 个预测头，约 57.3M 参数
- **动作专家**: [[DiT（扩散变换器）|DiT]] 约 150M 参数，[[Flow-Matching|流匹配]]，8 步动作块
- **动作维度**: 58 维（跨具身统一表达）
- **训练**: 两阶段（物理预训练 → 联合策略训练）
- **WM 监督**: 冻结 LingBot-Video 模型

---

## 批判性思考

### 优点

1. **无延迟物理感知**: 离线预测未来特征，避免视频世界模型在线生成的控制延迟，工程可行性强。
2. **数据工程扎实**: EgoDex 合成、子任务标注、ECoT 三条扩充路径均有明确的数量统计，数据集构建细节完整。
3. **消融分析全面**: 对 WM 监督、子任务刷新频率、ECoT、物体位姿、查询数、VLM 层深等均有对照实验。

### 局限性

1. **[[光流]]监督效果有限**: Table 10 显示[[光流]]预测提升不显著（95% CI 包含 0），其对最终动作性能的真实贡献存疑。
2. **物理感知收益量化不足**: 没有直接对比"有/无 Physical Dynamics Module"对动作成功率的影响，只有特征预测层面的消融。
3. **SimplerEnv 分项差异大**: Carrot→Plate（64.6%）远低于 InternVLA-A1.5（85.4%），跨任务一致性有待提升。

### 潜在改进方向

1. 引入接触状态、物体几何、材质属性等更丰富的物理预测目标（论文 Conclusion 已提出）。
2. 对[[光流]]监督进行自适应权重或多分辨率处理，提升其对物理表征的贡献。
3. 探索 Physical Dynamics Module 与强化学习或在线微调结合，进一步提升真实机器人部署性能。

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（详细的两阶段训练描述）
- [x] 数据集可获取（全部使用公开数据）

---

## 关联笔记

### 基于

- [[Qwen-RobotManip]]: 数据合成流水线来源，LIBERO-Plus 主要基线
- [[ECoT]]: 具身思维链标注框架
- [[Flow-Matching]]: 动作专家的训练范式

### 对比

- [[InternVLA-A1.5]]: SimplerEnv 和 RoboTwin 2.0 主要竞争方法
- [[LingBot-VLA-2.0]]: RoboTwin 2.0 基线，其视频模型用作 WM 监督来源

### 方法相关

- [[Physical Dynamics Module]]: 核心物理表征模块
- [[DiT（扩散变换器）]]: 动作专家架构
- [[Cross-Embodiment]]: 跨具身动作空间设计
- [[视频世界模型]]: 物理预训练的监督来源

### 硬件/数据相关

- [[Action Chunking]]: 8 步动作块预测
- [[Human-to-Robot Synthesis]]: EgoDex 数据合成
- [[Subtask Conditioning]]: 子任务条件化训练

---

## 速查卡片

> [!summary] PDRVLA
> - **核心**: 轻量 Physical Dynamics Module 将物理动态理解注入 VLA，无在线视频生成延迟
> - **方法**: 50 个可学习查询 + 冻结视频世界模型蒸馏 + 流匹配 DiT 动作专家
> - **结果**: LIBERO 99.3% / LIBERO-Plus 89.9% / RoboTwin 2.0 93.6% / SimplerEnv 81.5%
> - **代码**: 未公开

---

*笔记创建时间: 2026-10-29*

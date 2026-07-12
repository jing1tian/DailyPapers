---
title: "From Foundation to Application: Improving VLA Models in Practice"
method_name: "LingBot-VLA-2.0"
authors: [Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, Kecheng Zheng]
year: 2026
venue: arXiv
tags: [vla, cross-embodiment, mixture-of-experts, knowledge-distillation, mobile-manipulation, bimanual-manipulation, dexterous-hand, temporal-reasoning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.06403
created: 2026-07-12
---

# 论文笔记：From Foundation to Application: Improving VLA Models in Practice

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RobbyAnt |
| 日期 | July 2026 |
| 项目主页 | [technology.robbyant.com/lingbot-vla-v2](https://technology.robbyant.com/lingbot-vla-v2) |
| 对比基线 | [[π0.5]]、[[LingBot]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.06403) / [Code](https://github.com/robbyant/lingbot-vla-v2) / [Checkpoints](https://huggingface.co/collections/robbyant/lingbot-vla-v2) |

---

## 一句话总结

> LingBot-VLA 2.0 通过 6 万小时多具身数据、55 维统一动作表征、[[Mixture-of-Experts|MoE]] 架构及双教师时空感知蒸馏，显著缩小实验室 [[VLA]] 基础模型与真实部署之间的鸿沟。

---

## 核心贡献

1. **大规模多具身预训练数据集**: 收集 6 万小时预训练数据，覆盖 20 种机器人构型（50k 小时机器人轨迹 + 10k 小时第一视角人类视频），并建立完整数据处理流水线（轨迹平滑性评估、视频-状态一致性验证、[[SLAM]] 重建）。
2. **统一动作表征 + MoE 策略架构**: 设计 55 维规范动作向量，支持灵巧手/腰部/移动底盘等全身控制；引入基于 sigmoid 路由的 [[Mixture-of-Experts|Token 级稀疏 MoE]]，实现无辅助损失的负载均衡。
3. **双教师时空感知蒸馏**: 通过 LingBot-Depth 注入几何空间先验，通过 [[DINO-Video]] 注入因果时序理解能力，同时预测当前帧与未来帧的表示。

---

## 问题背景

### 要解决的问题

当前 [[VLA]] 基础模型在实验室条件下表现出色，但在真实部署时面临三大挑战：
1. **泛化能力不足**：机器人具身差异大（单臂/双臂/移动平台/灵巧手），跨具身迁移困难；
2. **动作空间不完整**：多数模型仅覆盖末端执行器或关节，无法控制头部、腰部等全身自由度；
3. **时序推理能力弱**：缺乏对场景动态的预见性建模，导致长时程任务失败率高。

### 现有方法的局限

- [[π0.5]] 等基础模型在长时程移动操作和多具身泛化上性能有限；
- 以往 [[Cross-Embodiment]] 方法数据覆盖面窄，难以处理人形机器人的高自由度控制；
- 现有视觉骨干缺乏机器人场景特有的因果时序感知能力。

### 本文的动机

从数据扩展（量级 + 多样性）、动作表示统一（全身控制）、模型架构升级（MoE）和感知能力增强（时空蒸馏）四个维度协同改进，实现"从基础模型到应用部署"的完整闭环。

---

## 方法详解

### 模型架构

![Figure 1: LingBot-VLA 2.0 系统概览](https://arxiv.org/html/2607.06403v1/x1.png)

**说明**: LingBot-VLA 2.0 整体架构。输入 RGB 观测、语言指令和状态向量，经 [[VLA]] 主干（含 [[Mixture-of-Experts|MoE]] 动作专家层）预测统一动作表征，同时通过双教师蒸馏注入深度空间和时序视频先验。

LingBot-VLA 2.0 采用 **基于大语言模型的 [[VLA]]** 架构：
- **输入**: RGB 观测 $o_t$、语言指令 $l$、状态向量 $s_t$
- **视觉骨干**: [[DINOv3]] 系列（扩展了双教师视觉蒸馏头）
- **动作专家**: Token 级稀疏 [[Mixture-of-Experts|MoE]] 层，含 1 个共享专家 + $N_r$ 个路由专家
- **输出**: 55 维统一动作向量（[[Action Chunking|动作块]]）

### 核心模块

#### 模块 1: 大规模多具身预训练数据

**设计动机**: 通过提升数据量级（6 万小时）和多样性（20 种具身）增强 [[Cross-Embodiment]] 泛化。

**具体实现**:
- **轨迹平滑性评估**: 用三阶有限差分（jerk）+ 速度/加速度 Z-score 过滤抖动轨迹
- **视频-状态一致性**: 机器人重投影验证 + 人工标注双重核查
- **第一视角人类数据重建**: [[SLAM]] 估计相机位姿 → 手部姿态估计提取动作标签 → 世界坐标变换（公式 1）
- **VLM 预过滤**: 用 [[Qwen]] 3.6-27B 过滤非第一视角操作视频，构建闭合词汇（18 类原子动作）自动分割子任务

#### 模块 2: 55 维统一动作表征

**设计动机**: 统一异构具身的控制空间，支持全身自由度。

![Figure 4: 统一动作表征](https://arxiv.org/html/2607.06403v1/x4.png)

**说明**: 55 维规范向量将异构具身控制映射到同一空间，灰色维度对特定具身置零。

**维度分配**:
- $14 \times 2 = 28$ 维: 双臂关节位置
- $14 \times 2 = 28$ 维: 末端执行器位姿（每臂 7 维）（注：与关节共享部分维度，最终合并为 55 维）
- 12 维: 灵巧手关节（每手 6 维）
- 4 维: 腰部定位
- 2 维: 头部控制
- 3 维: 移动底盘信号（$x, y, yaw$）
- 2 维: 夹爪控制

消融表明：**相对关节动作**远优于绝对关节动作（55.0% vs 33.3% 成功率），**MeanStd 归一化**提供最佳动态范围（std $\approx 0.95$ vs MinMax/分位数的 0.15–0.32）。

#### 模块 3: Token 级稀疏 MoE 动作专家

**设计动机**: 用共享专家捕获跨具身通用先验，用路由专家捕获具身专有特征，同时控制激活参数量。

![Figure 7: Dense 模型与 MoE 模型的参数对比](https://arxiv.org/html/2607.06403v1/x7.png)

**说明**: 在相同激活参数下，MoE 模型的总参数量远大于 Dense 模型，从而实现更大的容量-效率比。

**路由机制**（受 DeepSeek-V3 启发）:
- Sigmoid 独立路由，避免 softmax 竞争效应（公式 4–6）
- 路由偏置矫正实现无辅助损失负载均衡（公式 7–8）
- 专家实现采用 [[SwiGLU]] MLP（公式 3）

#### 模块 4: 双教师时空感知蒸馏

**设计动机**: 单纯扩大数据不能补充几何空间先验和因果时序理解，需要专用教师模型注入。

**LingBot-Depth**（几何教师）:
- 使用深度估计模型生成 $\mathbf{D}_t$（当前帧）和 $\mathbf{D}_{t+T}$（未来帧）深度图
- 通过 L1 损失强制动作查询向量与深度 token 对齐（公式 9）

**[[DINO-Video]]**（时序教师）:
- 基于 [[DINOv3]] + 块因果注意力 + 3D [[Rotary Position Encoding|旋转位置编码]]，在 500 万视频片段上训练
- 生成时序感知视觉特征 $\mathbf{Z}_t$、$\mathbf{Z}_{t+T}$
- 通过 Frobenius 范数损失蒸馏至 VLA 查询（公式 10）

---

## 关键公式

### 公式 1: [[SLAM|世界坐标到相机坐标变换]]

$$
\mathbf{p}_{\tau}^{C_t} = \mathbf{T}_{C_t \leftarrow W} \mathbf{p}_{\tau}^{W}
$$

**含义**: 将人类手部轨迹从世界坐标系转换到当前相机坐标系，解耦手部运动与相机运动，用于第一视角数据的动作标注。

**符号说明**:
- $\mathbf{p}_{\tau}^{W}$: 世界坐标系下手部轨迹（时刻 $\tau$）
- $\mathbf{T}_{C_t \leftarrow W}$: 世界坐标系到当前相机坐标系的变换矩阵（由 SLAM 估计）
- $\mathbf{p}_{\tau}^{C_t}$: 相机坐标系下手部轨迹

### 公式 2: [[Mixture-of-Experts|MoE 层输出]]

$$
m_\ell(\mathbf{u}_{\ell,t}) = E_\ell^{(s)}(\mathbf{u}_{\ell,t}) + \lambda \sum_{j \in \mathcal{R}(\mathbf{u}_{\ell,t})} g_{\ell,j}(\mathbf{u}_{\ell,t}) \, E_{\ell,j}^{(r)}(\mathbf{u}_{\ell,t})
$$

**含义**: 第 $\ell$ 层 MoE 输出 = 共享专家输出 + 加权路由专家输出之和，实现通用与专有特征的融合。

**符号说明**:
- $E_\ell^{(s)}$: 共享专家（所有 token 均通过）
- $E_{\ell,j}^{(r)}$: 第 $j$ 个路由专家
- $\mathcal{R}(\mathbf{u}_{\ell,t})$: 为 token $\mathbf{u}_{\ell,t}$ 选出的 Top-K 专家集合
- $g_{\ell,j}$: 归一化混合权重
- $\lambda$: 路由专家输出的缩放系数

### 公式 3: [[SwiGLU|专家 MLP（SwiGLU）]]

$$
E(\mathbf{u}) = W_{\text{down}} \bigl( \operatorname{SiLU}(W_{\text{gate}} \mathbf{u}) \odot W_{\text{up}} \mathbf{u} \bigr)
$$

**含义**: 共享专家和路由专家均采用 SwiGLU 门控 MLP，通过逐元素乘法实现特征选择性激活。

**符号说明**:
- $\operatorname{SiLU}(\cdot)$: Sigmoid 线性单元激活
- $\odot$: 逐元素乘法
- $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$: 门控、上投影、下投影权重矩阵

### 公式 4: [[Mixture-of-Experts|路由器 Logit]]

$$
z_{\ell,j}(\mathbf{u}_{\ell,t}) = \mathbf{u}_{\ell,t}^\top \mathbf{e}_{\ell,j}
$$

**含义**: 计算 token 与每个路由专家的初始亲和度分数，在 FP32 精度下执行。

**符号说明**:
- $\mathbf{e}_{\ell,j}$: 第 $\ell$ 层第 $j$ 个专家的可学习路由嵌入向量

### 公式 5: [[Mixture-of-Experts|Sigmoid Token-专家亲和度]]

$$
s_{\ell,j}(\mathbf{u}_{\ell,t}) = \operatorname{Sigmoid}\bigl(z_{\ell,j}(\mathbf{u}_{\ell,t})\bigr)
$$

**含义**: 对 logit 施加 Sigmoid（而非 Softmax），使各专家可独立激活，避免竞争性抑制。

### 公式 6: [[Mixture-of-Experts|归一化混合权重]]

$$
g_{\ell,j}(\mathbf{u}_{\ell,t}) = \frac{s_{\ell,j}(\mathbf{u}_{\ell,t})}{\sum_{k \in \mathcal{R}(\mathbf{u}_{\ell,t})} s_{\ell,k}(\mathbf{u}_{\ell,t})}, \quad j \in \mathcal{R}(\mathbf{u}_{\ell,t})
$$

**含义**: 仅在已选中专家集合内归一化亲和度，将混合权重计算与专家选择解耦，避免偏置影响最终权重。

### 公式 7: [[Load Balancing Loss|基于偏置矫正的专家选择]]

$$
\mathcal{R}(\mathbf{u}_{\ell,t}) = \operatorname{TopK}_j\bigl(s_{\ell,j}(\mathbf{u}_{\ell,t}) + b_{\ell,j},\ K\bigr)
$$

**含义**: 在选择 Top-K 专家时加入负载均衡偏置 $b_{\ell,j}$，使过载专家被抑制、欠载专家被提升，无需辅助损失。

**符号说明**:
- $b_{\ell,j}$: 第 $\ell$ 层第 $j$ 个专家的路由矫正偏置（可学习，动态更新）
- $K$: 每个 token 激活的专家数量

### 公式 8: [[Load Balancing Loss|偏置更新规则]]

$$
b_{\ell,j} \leftarrow b_{\ell,j} - \gamma \cdot \operatorname{sign}\!\left(n_{\ell,j} - \frac{1}{N_r} \sum_{k=1}^{N_r} n_{\ell,k}\right)
$$

**含义**: 根据专家负载偏差动态调整路由偏置：若某专家超载则降低其偏置，欠载则提高，实现无辅助损失的软性负载均衡。

**符号说明**:
- $n_{\ell,j}$: 第 $j$ 个专家在当前批次中的 token 处理量（负载统计）
- $N_r$: 路由专家总数
- $\gamma$: 偏置更新步长（超参数）

### 公式 9: [[Knowledge Distillation|深度蒸馏损失]]

$$
\mathcal{L}_{\text{depth}} = \mathbb{E}\Bigl[\| \operatorname{Proj}_{\text{depth}}(\mathbf{Q}_t) - \mathbf{D}_t \|_1 + \| \operatorname{Proj}_{\text{depth}}(\mathbf{Q}_{t+T}) - \mathbf{D}_{t+T} \|_1\Bigr]
$$

**含义**: 强制 VLA 动作查询向量（当前帧 + 未来帧）同时对齐深度表示，注入几何空间先验，使模型具备空间感知预测能力。

**符号说明**:
- $\mathbf{Q}_t, \mathbf{Q}_{t+T}$: 当前帧和未来帧的动作查询向量
- $\mathbf{D}_t, \mathbf{D}_{t+T}$: 深度估计教师模型输出的深度表示
- $\operatorname{Proj}_{\text{depth}}$: 线性投影头（用于维度对齐）
- $T$: 预测时间跨度

### 公式 10: [[DINO-Video|视频蒸馏损失]]

$$
\mathcal{L}_{\text{video}} = \mathbb{E}\Bigl[\| \operatorname{Proj}_{\text{video}}(\mathbf{Q}_t) - \mathbf{Z}_t \|_F^2 + \| \operatorname{Proj}_{\text{video}}(\mathbf{Q}_{t+T}) - \mathbf{Z}_{t+T} \|_F^2\Bigr]
$$

**含义**: 将 VLA 查询对齐到 DINO-Video 的因果时序视觉特征（当前帧 + 未来帧），注入时序动态理解先验。

**符号说明**:
- $\mathbf{Z}_t, \mathbf{Z}_{t+T}$: DINO-Video 教师在因果前向传播下输出的 patch 级视觉特征
- $\operatorname{Proj}_{\text{video}}$: 线性投影头（映射到 patch 特征空间）
- $\| \cdot \|_F^2$: Frobenius 范数平方（patch 级对齐）

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2607.06403v1/x1.png)

**说明**: LingBot-VLA 2.0 整体系统图，展示数据流水线、[[Mixture-of-Experts|MoE]] 策略网络和双教师蒸馏三大组件的关系。

### Figure 2: Pre-training Dataset / 预训练数据集可视化

![Figure 2](https://arxiv.org/html/2607.06403v1/x2.png)

**说明**: 20 种机器人具身的预训练数据集可视化，涵盖单臂、双臂、半人形及全人形平台。

### Figure 3: Data Processing Pipeline / 数据处理流水线

![Figure 3](https://arxiv.org/html/2607.06403v1/x3.png)

**说明**: 完整数据处理流水线，包括轨迹平滑性过滤（jerk 评估 + Z-score）、视频-状态一致性验证、[[SLAM]] 重建和 VLM 预过滤模块。

### Figure 4: Unified Action Representation / 统一动作表征

![Figure 4](https://arxiv.org/html/2607.06403v1/x4.png)

**说明**: 55 维规范向量将头部、腰部、双臂、末端执行器、灵巧手、底盘等异构控制信号映射到统一空间；不适用的维度填零。

### Figure 5: Subtask Action Statistics / 子任务动作统计

![Figure 5](https://arxiv.org/html/2607.06403v1/x5.png)

**说明**: 子任务注释中各原子动作的时长分布、频率、总时间和平均时长，`move` 和 `transit` 两类动作占绝对主导。

### Figure 6: Manipulated Object Word Cloud / 操作对象词云

![Figure 6](https://arxiv.org/html/2607.06403v1/x6.png)

**说明**: 子任务注释中被操作对象的词频词云，频率越高字体越大，展示数据集覆盖的物体多样性。

### Figure 7: Dense vs. MoE Parameter Comparison / 参数量对比

![Figure 7](https://arxiv.org/html/2607.06403v1/x7.png)

**说明**: 在等激活参数量下，[[Mixture-of-Experts|MoE]] 模型总参数量可达 Dense 模型的数倍，以稀疏激活实现更大容量。

### Figure 8: Per-Subtask Mobile Manipulation Performance / 子任务移动操作性能

![Figure 8](https://arxiv.org/html/2607.06403v1/x8.png)

**说明**: 长时程移动操作 benchmark 中，LingBot-VLA 2.0 在各子任务阶段的完成率，展示了模型在任务分解上的鲁棒性。

### Figure 9: Mobile Manipulation Experiment Platform / 移动操作实验平台

![Figure 9](https://arxiv.org/html/2607.06403v1/x9.png)

**说明**: 长时程移动操作实验所使用的机器人平台（Astribot S1 和 Cobot Magic-ARX X5）及任务场景布置。

### Figure 10: GM-100 Real-Robot Task Results / GM-100 真机任务结果

![Figure 10](https://arxiv.org/html/2607.06403v1/x10.png)

**说明**: GM-100 benchmark 中 4 个典型任务（Retrieve Keychain、Pick Out Toy Bone 等）的真机执行结果，展示模型的感知-操作精度。

### Figure 11: Per-Dimension Action Distributions / 逐维动作分布

![Figure 11](https://arxiv.org/html/2607.06403v1/x11.png)

**说明**: GM-100 任务中，关节和末端执行器动作空间各维度的 MeanStd 归一化动作分布，标准差（≈0.95）接近均匀分布。

### Figure 12: Action Targets and Normalization Statistics / 动作目标与归一化统计

![Figure 12](https://arxiv.org/html/2607.06403v1/x12.png)

**说明**: GM-100 四个任务的动作目标值及不同归一化策略（MeanStd、MinMax、分位数）的统计对比，说明 MeanStd 的优越性。

### Figure 13: Causal Perception Under Visual Distillation / 视觉蒸馏因果感知

![Figure 13](https://arxiv.org/html/2607.06403v1/x13.png)

**说明**: [[DINO-Video]] 视觉蒸馏后，LingBot-VLA 2.0 的查询向量对场景变化（如物体移入/移出）的因果响应可视化，验证时序感知能力提升。

---

### Table 1: 机器人具身数据统计

| 机器人类型 | 末端执行器 | 手/夹爪 DoF | 臂 DoF | 身体 DoF | 总 DoF | 控制频率 (Hz) |
|----------|-----------|------------|--------|---------|--------|------------|
| **单臂机器人** | | | | | | |
| Franka | 夹爪 | 1 | 7 | 0 | 8 | 30 |
| Flexiv Rizon 4 | 夹爪 | 1 | 7 | 0 | 8 | 30 |
| **双臂机器人** | | | | | | |
| AgileX | 夹爪 | 2 | 12 | 0 | 14 | 30 |
| ARX Lift2 | 夹爪 | 2 | 12 | 0 | 14 | 30 |
| UR7e | 夹爪 | 2 | 12 | 0 | 14 | 30 |
| **半人形** | | | | | | |
| AgiBot G1 | 夹爪 | 2 | 14 | 4 | 20 | 30 |
| Galbot G1 | 夹爪 | 2 | 14 | 6 | 22 | 30 |
| Moz1 | 夹爪 | 2 | 14 | 0 | 16 | 30 |
| Realman Rs-02 | 夹爪 | 2 | 14 | 1 | 17 | 30 |
| Galaxea R1Pro | 夹爪 | 2 | 14 | 7 | 23 | 15 |
| Galaxea R1Lite | 夹爪 | 2 | 12 | 3 | 17 | 15 |
| Astribot S1 | 夹爪 | 2 | 14 | 9 | 25 | 30 |
| Zerith H1 | 夹爪 | 2 | 14 | 7 | 23 | 30 |
| **人形机器人** | | | | | | |
| Leju KUAVO 4 Pro | 夹爪/手 | 2/12 | 14 | 5 | 21/31 | 30 |
| Tienkung | 夹爪 | 2 | 14 | 0 | 16 | 30 |
| QingLong | 夹爪 | 2 | 14 | 0 | 16 | 30 |
| MagicBot Gen1 | 夹爪 | 2 | 14 | 4 | 20 | 30 |
| Unitree G1 | 灵巧手 | 12 | 14 | 0 | 26 | 30 |
| Fourier GR-2 | 灵巧手 | 12 | 14 | 6 | 32 | 30 |
| AgiBot A2 | 灵巧手 | 12 | 14 | 2 | 28 | 30 |
| Humanoid Ego | — | — | — | 14 | — | 30–60 |
| **合计** | 20 具身 | | | | | **60,000h** |

### Table 2: 子任务注释闭合动词词汇

| 原子操作动作 | 描述 |
|------------|------|
| move | 将物体从 A 移至 B |
| pour | 倾斜容器使内容物流出 |
| push | 滑动未持握的物体 |
| pull | 将物体拉近 |
| rotate | 原地旋转物体 |
| open | 打开铰链或关节式物体 |
| close | 关闭铰链或关节式物体 |
| detach | 从配件上移除/断开物体 |
| fold | 收缩柔性材料 |
| unfold | 展开柔性材料 |
| wipe | 用工具在表面移动（擦拭） |
| stir | 在容器中圆周运动搅拌 |
| cut | 用刀具切割目标 |
| press | 按压按钮/开关/表面 |
| attach | 将物体插入或连接到配件 |
| **辅助标签** | |
| transit | 夹爪移动但不与物体交互 |
| idle | 手臂保持静止 |
| other | 词汇外的动作 |

### Table 3: DINO-Video 在 LARYBench 上的性能

| 模型 | 参数量 (M) | Composite Human ↑ | Composite Robot ↑ | RoboCOIN ↓ | AgiBotWorld-Beta ↓ |
|------|-----------|-------------------|-------------------|------------|-------------------|
| V-JEPA 2 | 303.89 | 80.35 | 70.43 | 0.32 | 0.33 |
| DINOv3 | 303.13 | 76.19 | 69.06 | 0.22 | 0.24 |
| **DINO-Video** | **303.13** | **80.21** | **71.97** | **0.20** | **0.19** |

**关键发现**: [[DINO-Video]] 在保持与 [[DINOv3]] 相同参数量的前提下，Human 指标与 V-JEPA 2 相当，Robot 指标全面超越两者，且在时序预测误差上（RoboCOIN、AgiBotWorld-Beta）达到最优。

### Table 4: GM-100 Benchmark 任务评分标准

| 任务 ID | 任务名称 | 评分子步骤 |
|---------|---------|-----------|
| BM-19 | Block Sorting | 按大小排序 4 个方块（8 步，共 100 分） |
| BM-20 | Retrieve Keychain | 拉开抽屉→抓钥匙链→移前→放下（4×25分） |
| BM-25 | Scoop Rice | 取勺→舀米→倒入杯→放勺（4 步，共 100 分） |
| BM-39 | Replace Paper Roll | 移除纸芯→放托盘→取新纸卷→安装（4 步） |
| BM-45 | Sort Snacks | 依次取放 3 种零食进容器（6 步） |
| BM-47 | Pack Eggs | 依次取放 4 个鸡蛋进托盘（8 步） |
| BM-69 | Pick Out Toy Bones | 依次取出 2 根玩具骨头（4×25分） |
| BM-75 | Push Ball into Box | 推球向箱（30分）→推入箱（70分） |
| BM-82 | Squeeze Ketchup | 取瓶→倾斜喷嘴→双手挤压→放下（4×25分） |
| BM-97 | Take Bowl from Microwave | 开门→抓碗→移至桌面→关门（4×25分） |
| BM-105 | Tool Packing | 依次装入胶带、卷尺、螺丝刀、切割刀→关盖（5×20分） |
| BM-107 | Barcode Scan | 提箱→取扫描仪→扫码→放下（5×20分） |

### Table 5: GM-100 双臂操作 Benchmark 性能

| 任务 | Agilex Cobot Magic 进度(%) | Agilex 成功率(%) | Galaxea R1 Pro 进度(%) | Galaxea 成功率(%) |
|------|--------------------------|----------------|----------------------|----------------|
| **Overall** | **66.2** | **34.4** | **34.6** | **15.6** |
| Block Sorting | 56.8 | 0.0 | 33.2 | 0.0 |
| Retrieve Keychain | 100.0 | 100.0 | 0.0 | 0.0 |
| Replace Paper Roll | 55.2 | 20.0 | 57.6 | 40.0 |
| Sort Snacks | 66.2 | 10.0 | 26.1 | 0.0 |
| Pack Eggs | 44.4 | 0.0 | 4.4 | 0.0 |
| Pick Out Toy Bone | 95.0 | 90.0 | 87.5 | 70.0 |
| Push Ball into Box | 41.0 | 20.0 | 18.0 | 0.0 |
| Take Bowl from Microwave | 77.5 | 70.0 | 65.0 | 30.0 |
| Tool Packing | 60.0 | 0.0 | 20.0 | 0.0 |

**关键发现**: LingBot-VLA 2.0 在 Agilex 平台上取得 66.2% 进度 / 34.4% 成功率（对比 [[LingBot]] v1.0 的 58.2% / 30.0%）；Retrieve Keychain 任务实现 100% 成功；Pack Eggs 和 Block Sorting 成功率为 0 显示精细抓取仍是难点。

### Table 6: 长时程移动操作结果

| 具身 | 任务 | 设定 | LingBot-VLA 2.0（进度/成功率） | [[π0.5]]（进度/成功率） |
|------|------|------|-------------------------------|----------------------|
| Astribot S1 | 物品分拣入冰箱 | In-domain | **77.1% / 60.0%** | 65.3% / 46.7% |
| Astribot S1 | 物品分拣入冰箱 | Out-of-distribution | **37.0% / 13.3%** | 30.3% / 6.7% |
| Cobot Magic-ARX X5 | 灶台清洁 | In-domain | **84.3% / 66.7%** | 79.9% / 60.0% |
| Cobot Magic-ARX X5 | 灶台清洁 | Out-of-distribution | **67.5% / 40.0%** | 62.5% / 33.3% |

**关键发现**: 在分布内和分布外场景下，LingBot-VLA 2.0 均优于 [[π0.5]]，灶台清洁任务 OOD 成功率提升 +6.7 个百分点，验证了多具身预训练的泛化优势。

### Table 7: 长时程移动操作任务评分标准

| 任务 | 评分子步骤 |
|------|-----------|
| 物品分拣入冰箱 | 移至岛台(6)→取饮料入篮(9)→取第一件水果(9)→取第二件水果(11)→拿起篮子(8)→移至冰箱(10)→开门(12)→放第一件水果(9)→放第二件水果(11)→放饮料(9)→关门(6) |
| 灶台清洁 | 移至灶台(8)→取锅架放桌(16)→取海绵(12)→擦拭泡沫(18)→移动海绵(16)→放回锅架(16)→取锅放灶(14) |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 预训练多具身数据 | 60,000 小时 | 20 种具身，涵盖单臂/双臂/移动平台 | 预训练 |
| 人类第一视角视频 | 10,000 小时 | 通过 [[SLAM]] 重建相机位姿并标注动作 | 预训练 |
| GM-100 Bimanual | 9 任务，10 trials/task | Agilex + Galaxea 双平台，标准化评分 | 测试 |
| Long-Horizon Mobile | 2 任务，15 trials | Astribot S1 + Cobot Magic-ARX X5，In/OOD | 测试 |
| LARYBench | 分类+回归 | 用于 DINO-Video 视觉表示评估 | 视觉基准 |

### 实现细节

- **VLM 标注器**: [[Qwen]] 3.6-27B，用于视频分割与子任务语言标注
- **动作归一化**: MeanStd 归一化（标准差 $\approx 0.95$）
- **动作类型**: 相对关节动作（优于绝对关节动作，+21.3% 成功率）
- **损失函数**: L2 损失（优于 L1，+8.6% 成功率）
- **MoE 路由**: Sigmoid 路由 + Top-K 选择 + 无辅助损失负载均衡偏置
- **DINO-Video 预训练**: 500 万视频片段，块因果注意力 + 3D [[Rotary Position Encoding|RoPE]]

### 消融实验关键发现

| 配置 | 成功率(%) | 说明 |
|------|---------|------|
| 绝对关节动作 | 33.7 | 受关节零位偏差影响 |
| **相对关节动作** | **55.0** | 对各具身参考系具有鲁棒性 |
| L1 损失 | 46.4 | — |
| **L2 损失** | **55.0** | 对平滑轨迹优化更有效 |
| MinMax 归一化 | ~37 | 动态范围受极值压缩影响 |
| **MeanStd 归一化** | **55.0** | std≈0.95，动态范围均匀 |

---

## 批判性思考

### 优点
1. **数据规模和多样性领先**: 6 万小时 / 20 具身，涵盖从 Franka 到灵巧手人形的完整谱系，[[Cross-Embodiment]] 泛化能力强；
2. **无辅助损失负载均衡**：路由偏置矫正方法去除了传统 MoE 辅助损失的调参负担，训练更稳定；
3. **双教师蒸馏设计清晰**：空间几何先验（LingBot-Depth）和时序动态先验（[[DINO-Video]]）职责分离，可独立扩展；
4. **全栈数据流水线公开**：从 [[SLAM]] 重建、VLM 标注到闭合词汇注释的完整工程细节均有描述。

### 局限性
1. **精细操作成功率偏低**: Pack Eggs、Block Sorting 等需要高精度感知的任务在两个平台上成功率均为 0，说明当前架构对精度敏感任务仍有显著差距；
2. **OOD 性能下降明显**: Astribot S1 OOD 成功率仅 13.3%（vs. In-domain 60.0%），泛化至新场景的稳健性有待提升；
3. **机构/参数规模未公开**: 论文未披露具体模型参数量和所属机构，难以与其他工作公平比较；
4. **Galaxea 平台整体偏弱**: R1 Pro 总体成功率仅 15.6%，提示不同具身的微调数据不足。

### 潜在改进方向
1. 针对精细操作任务引入触觉传感器或力反馈信号（[[Action Diffusion]] 风格）；
2. 引入在线 RL 微调（如 [[π0.5]] 的 RLHF 路线）提升 OOD 成功率；
3. 将 [[DINO-Video]] 蒸馏扩展到多摄像头视角，提升空间感知全面性。

### 可复现性评估
- [x] 代码开源（github.com/robbyant/lingbot-vla-v2）
- [x] 预训练模型（HuggingFace）
- [ ] 训练细节完整（部分超参数未披露）
- [ ] 数据集可获取（60,000h 数据未发布）

---

## 关联笔记

### 基于
- [[LingBot]]: LingBot-VLA 1.0，本文的直接前身
- [[DINOv3]]: DINO-Video 教师模型的骨干网络
- [[Mixture-of-Experts]]: 动作专家层的核心架构
- [[SwiGLU]]: 专家 MLP 的激活函数

### 对比
- [[π0.5]]: 长时程移动操作 benchmark 的主要对比方法
- [[LingBot]]: v1.0 在 GM-100 的基线数字

### 方法相关
- [[DINO-Video]]: 本文提出的机器人场景视频表示模型
- [[Cross-Embodiment]]: 核心解决的泛化问题
- [[Knowledge Distillation]]: 双教师蒸馏的技术基础
- [[SLAM]]: 用于第一视角数据重建
- [[Action Chunking]]: VLA 动作预测的标准范式
- [[Load Balancing Loss]]: 本文用偏置矫正替代辅助损失

### 硬件/数据相关
- [[Qwen]]: 用于子任务自动标注的 VLM

---

## 速查卡片

> [!summary] LingBot-VLA 2.0 (arXiv 2607.06403)
> - **核心**: 通过 6 万小时多具身数据 + MoE 架构 + 双教师时空蒸馏，缩小 VLA 实验室模型到真实部署的差距
> - **方法**: 55 维统一动作表征 + Sigmoid 路由 MoE + LingBot-Depth/DINO-Video 双蒸馏
> - **结果**: GM-100 Agilex 66.2%/34.4%，移动操作超越 π0.5（+13.7% In-domain 成功率，Astribot）
> - **代码**: [github.com/robbyant/lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2)

---

*笔记创建时间: 2026-07-12*

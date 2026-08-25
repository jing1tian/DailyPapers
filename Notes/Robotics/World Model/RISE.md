---
title: "RISE: Adaptive Imagination for World Action Models"
method_name: "RISE"
authors: [Hongbo Lu, Liang Yao, Chenghao He, Hao Han, Fan Liu, Wenlong Liao, Tao He, Pai Peng]
year: 2026
venue: arXiv
tags: [world-action-model, adaptive-computation, autonomous-driving, counterfactual-data, diffusion-policy, planning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.20430
project_page: https://cowarobot-ai.github.io/RISE/
created: 2026-08-25
---

# 论文笔记：RISE: Adaptive Imagination for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未列出（工业界/学术界联合） |
| 日期 | August 2026 |
| 项目主页 | [cowarobot-ai.github.io/RISE](https://cowarobot-ai.github.io/RISE/) |
| 对比基线 | [[DriveFuture]]、[[DAWN]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.20430) |

---

## 一句话总结

> RISE 为[[WAM|World Action Model]]引入自适应想象框架，通过预测「未来规划收益（Future Planning Gain）」动态决定 rollout 深度，在提升规划性能的同时降低不必要计算开销。

---

## 核心贡献

1. **自适应 Rollout Scheduler**：提出 Future Planning Gain 指标，使 [[WAM]] 能按场景需求动态决定 Roll/Stop，而非固定 rollout 深度
2. **反事实训练数据集 CounterDrive**：使用视频生成模型构建配对反事实驾驶片段，显著提升风险识别能力（AUC 从 ~0.50 到 0.93-0.96）
3. **即插即用设计**：Scheduler 作为插件适用于不同 WAM 架构，迁移至 DAWN 后 PDMS 提升 1.2 分

---

## 问题背景

### 要解决的问题

现有 [[WAM|World Action Model]] 对所有驾驶场景分配**固定的 rollout 深度**（想象步数），无论是空旷路段还是密集交叉口，计算代价相同。这导致简单场景浪费算力，复杂场景却不一定分配了足够的预测深度。

### 现有方法的局限

三类主流 WAM 策略均采用固定预算：

- *Imagine and Plan*（如 JEPA 系列）：联合使用观测和预测 latent 进行规划
- *Imagine then Plan*：先完成完整 rollout 再生成动作
- *No Imagination*（如部分轻量方案）：直接从观测规划，完全跳过预测

这些方案将测试时的 imagination 视为固定的架构选择，而非可调的计算决策。

### 本文的动机

不同场景对 rollout 的需求截然不同：实验表明，将 h\*=0 场景强制使用 h=4 rollout，性能从 89.9 下降到 88.4 EPDMS；反之亦然。因此需要场景自适应的 rollout 策略，且该策略应轻量、可迁移。

---

## 方法详解

### 模型架构

RISE 在现有 [[WAM]] 基础上叠加一个轻量 **Rollout Scheduler**（含 [[Latent Evaluator]] + [[Rollout Gate]]），核心骨干保持冻结：

- **输入**: 4 帧 RGB 图像（256×512 分辨率）+ ego-motion 动作序列
- **Backbone**: 冻结的 [[V-JEPA 2|V-JEPA 2 ViT-L]]（16×16 patch，非重叠 tubelet），输出每时刻 512 个空间 token
- **Predictor**: 12 层 [[Transformer]]（384 隐维，12 头），自回归生成未来 latent，以 ego-motion 为条件，无需路由命令或相机外参
- **Planner**: 12 层 [[Diffusion Transformer|扩散 Transformer]]，用 [[DPM-Solver++]]（20 步）生成 6 条候选轨迹，每条包含 0.5 秒间隔的 8 个未来位姿
- **Latent Evaluator**: 轻量模块，预测 Risk Profile $R_h$ 和 Future Planning Gain Profile $B_h$
- **Rollout Gate**: 基于评估器输出 + 计算偏好 $\lambda$，输出二值 Roll/Stop 决策

### 核心模块

#### 模块 1: Latent Evaluator（潜空间评估器）

**设计动机**: 在不运行完整 Planner 的情况下，从 latent 空间直接预测继续 rollout 的收益

**具体实现**:
- 输入当前深度 $h$ 的 latent 表示 $z_{t+k}$，输出两个配置文件
- **Risk Profile** $R_h$：总结当前 prefix 已揭示的规划相关风险（如碰撞可能性）
- **Future Planning Gain Profile** $B_h$：描述继续 rollout 到各深度后规划质量的预期提升量
- 使用 CounterDrive 配对数据训练风险区分能力，使用 ranking loss + localization loss 监督

#### 模块 2: Rollout Gate（Rollout 门控器）

**设计动机**: 将预期收益与计算代价综合考量，输出 Roll 或 Stop 的硬决策

**具体实现**:
- 构建决策向量：pooled latent 特征 $e_h$ + 风险增益配置文件 $\xi_h$（补零对齐到最大深度 $H_a$）+ 归一化深度 + 累积/边际计算代价 + 计算偏好 $\lambda$
- 输出标量 $x_h$：$x_h > 0$ 则 Roll，否则 Stop
- 支持多个 $\lambda$ 值联合训练，推理时通过调节 $\lambda$ 控制速度-性能权衡

#### 模块 3: CounterDrive 数据集

**设计动机**: 真实驾驶日志中严重事故罕见，导致风险识别能力弱（基线 AUC ≈ 0.50，近随机）

**构建流程**:
1. 从 [[NAVSIM]] 和 [[nuScenes]] 中选取源场景（包含危险情境）
2. 使用 Wan 2.7 视频生成模型生成 10 秒反事实视频（1080p，2 Hz 采样为 20 帧）
3. OpenVO 恢复帧级 ego 位姿，计算相邻帧动作序列
4. 人工标注员验证运动一致性、标记事故帧、过滤失真、分类（正常/非自车原因/自车原因）
5. 每条保留片段与其原始来源配对，形成因果-反事实对

**最终规模**:

| 基准 | 训练 | 测试 |
|------|------|------|
| nuScenes | 2,432 | 511 |
| NAVSIM | 5,013 | 1,000 |

---

## 关键公式

### 公式 1: [[WAM|标准 WAM 联合规划]]

$$
p(z_{1:H},\, \tau_{1:P} \mid c) = p(z_{1:H} \mid c)\, p(\tau_{1:P} \mid c,\, z_{1:H})
$$

**含义**: 将 [[WAM]] 分解为 latent 预测 $p(z_{1:H}|c)$ 和条件规划 $p(\tau_{1:P}|c, z_{1:H})$ 两个因子，预测深度 $H$ 固定

**符号说明**:
- $z_{1:H}$：未来 $H$ 步 latent 状态序列
- $\tau_{1:P}$：长度为 $P$ 的规划轨迹
- $c$：上下文观测（当前帧 latent）

### 公式 2: [[Future Planning Gain|自适应 Rollout 深度]]

$$
K(c, \lambda) = \underset{h \in \{0,\ldots,H_a\}}{\text{Stop}} \bigl(c,\, h,\, \lambda\bigr)
$$

**含义**: RISE 的核心目标是学习一个场景感知调度器 $K$，根据上下文 $c$ 和计算偏好 $\lambda$ 自适应确定 rollout 深度，而非固定使用 $H_a$

**符号说明**:
- $H_a$：允许的最大 rollout 深度
- $\lambda$：计算偏好参数（推理时可调）

### 公式 3: [[Rollout Gate|Gate 决策规则]]

$$
x_h = G_{\theta_G}(e_h,\, \xi_h,\, \lambda)
$$

$$
d_h = \begin{cases} \text{Roll} & \text{if } x_h > 0 \text{ and } h < H_a \\ \text{Stop} & \text{otherwise} \end{cases}
$$

**含义**: Gate 网络 $G_{\theta_G}$ 接收当前深度 $h$ 的 latent 特征和评估器输出，输出标量得分 $x_h$；正值表示继续 rollout，否则停止并触发规划

**符号说明**:
- $e_h$：深度 $h$ 的 pooled latent 特征
- $\xi_h$：风险和增益配置文件（补零到 $H_a$ 维）
- $\lambda$：计算代价权重

### 公式 4: [[Predictor|Predictor 训练损失]]

$$
\mathcal{L}_{\text{Pre}} = \sum_{k=1}^{H} \ell_z\!\bigl(\hat{z}_{t+k},\; \operatorname{sg}(z_{t+k})\bigr)
$$

**含义**: 对预测的 latent $\hat{z}_{t+k}$ 和停止梯度的教师 latent $\operatorname{sg}(z_{t+k})$ 计算回归损失，在真实和 CounterDrive 序列上联合训练

**符号说明**:
- $\hat{z}_{t+k}$：Predictor 在第 $k$ 步的预测 latent
- $\operatorname{sg}(\cdot)$：stop-gradient 算子
- $z_{t+k}$：编码器输出的目标 latent

### 公式 5: [[Risk Profile|风险配置文件损失]]

$$
\mathcal{L}_R = \mathcal{L}_{\text{real}} + \beta_{\text{cf}} \cdot \mathcal{L}_{\text{rank}} + \beta_{\text{loc}} \cdot \mathcal{L}_{\text{loc}}
$$

**含义**: 三项组合损失：真实轨迹监督 $\mathcal{L}_{\text{real}}$、反事实配对 ranking loss $\mathcal{L}_{\text{rank}}$（保证危险场景风险值高于安全场景）、事故定位 loss $\mathcal{L}_{\text{loc}}$

**符号说明**:
- $\beta_{\text{cf}}, \beta_{\text{loc}}$：权重超参数
- $\mathcal{L}_{\text{rank}}$：利用因果-反事实配对的比较损失
- $\mathcal{L}_{\text{loc}}$：事故起始时刻定位损失

### 公式 6: [[Future Planning Gain|未来规划增益目标]]

$$
B_{i,h}^{*} = \bigl[q_{i,h+1} - q_{i,h},\; \ldots,\; q_{i,H_a} - q_{i,h}\bigr]
$$

**含义**: 以当前深度 $h$ 的规划质量 $q_{i,h}$ 为基准，记录继续 rollout 到各深度所带来的质量提升量，作为 Evaluator 的监督目标

**符号说明**:
- $q_{i,h}$：场景 $i$ 在 prefix 深度 $h$ 时的规划质量得分（由 Planner 输出推导）
- $H_a$：最大允许深度

### 公式 7: [[Rollout Gate|代价调整增益与 Gate 训练目标]]

$$
\begin{aligned}
g_{i,h}^{*} &= \max_{j \in \{h+1,\ldots,H_a\}} \Bigl[\bigl(q_{i,j} - q_{i,h}\bigr) - \lambda(c_j - c_h)\Bigr] \\[6pt]
y_{i,h}^{*} &= \mathbb{I}\bigl[g_{i,h}^{*} > 0\bigr]
\end{aligned}
$$

$$
\mathcal{L}_G = \frac{1}{N_G} \sum_{(i,h,\lambda) \in \Omega_G} \ell_{\text{BCE}}\!\bigl(\sigma(x_{i,h}),\; y_{i,h}^{*}\bigr)
$$

**含义**: 代价调整增益 $g^*$ 衡量继续 rollout 能否抵消额外计算代价；当最大可获得增益为正时，Gate 目标为 Roll（1），否则为 Stop（0）；BCE loss 在多个 $\lambda$ 值的组合上训练以保证泛化性

**符号说明**:
- $c_j - c_h$：从深度 $h$ 继续到深度 $j$ 的边际计算代价
- $\lambda$：将计算代价换算为性能单位的权重
- $\sigma(\cdot)$：sigmoid 函数
- $\Omega_G$：训练样本集合（场景 $\times$ 深度 $\times$ $\lambda$ 三元组）

---

## 关键图表

### Figure 1: Imagination Strategy Comparison / 想象策略对比

![Figure 1](https://cowarobot-ai.github.io/static/images/rise_imagination_strategies.png)

**说明**: 四种策略对比。(a) *Imagine and Plan* 联合使用观测和预测 latent；(b) *Imagine then Plan* 完整 rollout 后再规划；(c) *No Imagination* 直接从观测规划；(d) **RISE** 通过轻量 Scheduler 自适应选择继续 rollout 还是直接规划。

### Figure 2: RISE Overview / RISE 系统概览

![Figure 2](https://cowarobot-ai.github.io/static/images/rise_overview.png)

**说明**: RISE 整体架构。[[V-JEPA 2]] 编码器提取 latent，Predictor 自回归生成未来 latent，Latent Evaluator 同时预测 Risk Profile $R_h$ 和 Future Planning Gain Profile $B_h$，Rollout Gate 根据收益-代价决策 Roll/Stop，最终 Planner（[[Diffusion Transformer]]）生成轨迹。

### Figure 3: Rollout Depth Performance / 不同 rollout 深度性能

![Figure 3](https://cowarobot-ai.github.io/static/images/rise_rollout_depths.png)

**说明**: 在 NAVSIM v2 上，不同固定 rollout 深度 $h=0,1,2,3,4$ 的 EPDMS 性能分布。最优深度因场景而异：空旷路段偏好 $h=0$，密集交叉口偏好 $h=3$ 或 $h=4$，验证了自适应 rollout 的必要性。

### Figure 4: Scene-Dependent Rollout Depth / 场景偏好深度分析

![Figure 4](https://cowarobot-ai.github.io/static/images/rise_scene_groups.png)

**说明**: 按最优 rollout 深度分组的代表性场景。深度 $h^*=0$（1,248 场景）：开放路段、低交互；深度 $h^*=3$（4,036 场景）：红绿灯、车辆交汇、施工区；深度 $h^*=4$（2,180 场景）：密集行人流、横穿车辆。交互越复杂，通常需要更多 rollout。

### Figure 5: Qualitative Trajectory Comparison / 轨迹质量可视化

![Figure 5](https://cowarobot-ai.github.io/static/images/rise_qualitative_results.png)

**说明**: 在 NAVSIM 上从深度 $h=4$ 到 $h=0$ 的逐深度轨迹对比。行：前视摄像头图像和 [[BEV|鸟瞰图]]（Bird's-Eye-View）。颜色：人类 GT（绿）/ Drive-JEPA（黄）/ RISE（红）。展示 RISE 在复杂场景下随 rollout 深度增加轨迹质量的提升。

---

### Table 1: NAVSIM v1 主要结果（PDMS Leaderboard）

| 方法 | EP | NC | DAC | TTC | Comfort | PDMS |
|------|----|----|-----|-----|---------|------|
| 各 baseline（参考） | — | — | — | — | — | ≤90.7 |
| DriveFuture（前 SOTA） | — | — | — | — | — | 90.7 |
| **RISE（ours）** | **98.3** | — | — | **98.6** | — | **91.5** |

**说明**: RISE 在 NAVSIM v1 上超越前 SOTA DriveFuture 0.8 分，Trajectory Error（EP）+2.9，Time-to-Collision（TTC）+1.9。

### Table 2: NAVSIM v2 主要结果（EPDMS）

| 方法 | EPDMS | 组件指标排名 |
|------|-------|-------------|
| 基线 WAM | 89.9 | — |
| **RISE（ours）** | **90.8** | 9 个子指标中 7 个第一/并列第一 |

**说明**: RISE 在 NAVSIM v2 上 EPDMS 超出基线 0.9，覆盖安全性、合规性和规划质量多维指标。

### Table 3: nuScenes 结果

| 方法 | L2 误差（平均，m）| 碰撞率（平均） |
|------|-----------------|--------------|
| 竞品方法（最优） | >0.31 | >0.10 |
| **RISE（ours）** | **0.31** | **0.10** |

**说明**: RISE 在 nuScenes 上达到 L2=0.31m 和碰撞率 0.10 的 SOTA，在 nuScenes 场景中的泛化能力得到验证。

### Table 4: 消融实验 — 组件贡献（NAVSIM）

| 配置 | EPDMS（v2）| PDMS（v1）| 说明 |
|------|-----------|-----------|------|
| 基线（无附加） | 88.9 | 89.7 | 固定 rollout |
| + CounterDrive only | 89.8 | 90.5 | +0.9 / +0.8 |
| + Scheduler only | 90.4 | 91.2 | +1.5 / +1.5 |
| **+ Both（RISE）** | **90.8** | **91.5** | **+1.9 / +1.8** |

**关键发现**: CounterDrive 丰富监督信号，Scheduler 高效分配计算，二者协同效果优于任一单独使用。

### Table 5: 自适应 Rollout 方法对比

| 方法 | 平均 rollout 数 | EPDMS | 延迟（ms） |
|------|----------------|-------|-----------|
| Random Stop | 2.03 | 89.5 | 264 |
| Latent Margin | 2.98 | 89.7 | 308 |
| **RISE Scheduler** | **2.40** | **90.8** | **287** |

**关键发现**: RISE Scheduler 在比 Latent Margin 少 19% rollout 的情况下，EPDMS 高出 1.1 分；比 Random Stop 多 18% rollout 但性能提升 1.3 分，实现了最优速度-性能均衡。

### Table 6: CounterDrive 对风险识别的效果

| 配置 | AUC（各深度）| 事故识别率 |
|------|------------|-----------|
| 无 CounterDrive | 0.49–0.52 | 0.51（近随机）|
| **有 CounterDrive** | **0.93–0.96** | **0.96** |

**关键发现**: CounterDrive 将风险区分能力从随机水平（AUC≈0.50）提升至 0.93-0.96，事故识别率从 0.51 提升至 0.96。

### Table 7: 迁移至 DAWN 架构

| 方法 | NC | EP | TTC | PDMS |
|------|----|----|-----|------|
| DAWN（原始） | 98.7 | 84.3 | 96.0 | 89.1 |
| **DAWN + RISE Scheduler** | **99.9** | **87.0** | **98.3** | **90.3** |
| 提升 | +1.2 | +2.7 | +2.3 | +1.2 |

**关键发现**: RISE Scheduler 作为即插即用模块直接应用于 DAWN（无需修改 predictor/planner），PDMS 提升 1.2 分，验证了跨架构可迁移性。

---

## 三阶段训练流程

**Stage I — Predictor + 初始 Planner $\Pi_0$**:
- Predictor 在真实序列和 CounterDrive 序列上用教师编码目标训练
- 初始 Planner $\Pi_0$ 在所有前缀深度 $\{0,\ldots,H_a\}$ 上用轨迹监督训练

**Stage II — Latent Evaluator + 引导 Planner $\Pi_1$**:
- 用 $\Pi_0$ 的几何输出计算基于几何的风险目标 $r_k^*$
- 用验证过的因果-反事实对（含事故起始屏蔽）训练 Evaluator
- 对 prefix latent 做风险引导的迭代梯度下降精化（含梯度 norm 裁剪）
- 用精化后的 prefix 训练最终 Planner $\Pi_1$，计算 Future Planning Gain 目标 $B^*$

**Stage III — Rollout Gate**:
- 在多个 $\lambda$ 值组合下，用 BCE loss 训练 Rollout Gate

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[NAVSIM]] | 大规模真实驾驶 | 提供反应式仿真评估（PDMS/EPDMS） | 主要 benchmark |
| [[nuScenes]] | 1000 场景，700/150/150 分割 | 多传感器、标注轨迹 | 验证 L2/碰撞率 |
| CounterDrive | NAVSIM 5013+1000 / nuScenes 2432+511 | 视频生成反事实对 | Evaluator 训练 |

### 实现细节

- **Backbone**: 冻结 [[V-JEPA 2|V-JEPA 2 ViT-L]]，输入 4 帧（256×512，16×16 patch）
- **Predictor**: 12 层 Transformer，384 隐维，12 头
- **Planner**: 12 层扩散 Transformer，[[DPM-Solver++]] 20 步，输出 6 条候选轨迹
- **分辨率**: 256×512，512 个空间 token/时刻
- **视频生成**: Wan 2.7（1080p，10 秒，2 Hz 采样）

---

## 批判性思考

### 优点

1. **场景感知分配计算**: 实验证明不同场景确实需要不同深度，RISE 的 Scheduler 有效地将更多计算导向高收益场景
2. **反事实数据的巧妙利用**: CounterDrive 构建了真实驾驶日志中难以覆盖的危险场景，解决了风险识别数据稀缺问题
3. **即插即用设计**: 迁移至 DAWN 的实验证明框架的通用性，降低了工业应用门槛

### 局限性

1. **领域局限**: 仅在自动驾驶场景验证，向机器人操控等其他 WAM 应用场景的迁移性未探索
2. **CounterDrive 覆盖不完整**: 视频生成和人工过滤代价高，未能覆盖完整数据集，可能存在分布偏差
3. **视频生成质量瓶颈**: CounterDrive 质量上限依赖 Wan 2.7 的生成保真度，对物理规律的违背需要人工筛选

### 潜在改进方向

1. 将 Scheduler 扩展到机器人操控等其他 WAM 任务，探索 Future Planning Gain 在连续控制任务中的定义
2. 自动化 CounterDrive 的质量筛选（减少人工标注依赖）
3. 探索 $\lambda$ 的在线自适应调整（而非固定推理时偏好）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文中描述详细）
- [x] 数据集构建流程公开（CounterDrive 流程描述完整）

---

## 关联笔记

### 基于

- [[WAM]]: RISE 在 WAM 框架上叠加自适应调度器
- [[V-JEPA 2]]: RISE 使用冻结的 V-JEPA 2 ViT-L 作为视觉编码器
- [[JEPA]]: RISE 延续 JEPA 的 latent 预测范式

### 对比

- [[DriveFuture]]: NAVSIM v1 前 SOTA，被 RISE 超越 0.8 PDMS
- [[DAWN]]: RISE Scheduler 迁移验证对象，PDMS 提升 1.2

### 方法相关

- [[Future Planning Gain]]: RISE 的核心评估指标概念
- [[Diffusion Transformer]]: Planner 采用的架构
- [[DPM-Solver++]]: Planner 推理采用的快速采样器
- [[CounterDrive]]: RISE 构建的反事实驾驶数据集

### 数据/评估相关

- [[NAVSIM]]: 主要评估 benchmark
- [[nuScenes]]: 验证数据集
- [[BEV|BEV（Bird's-Eye-View）]]: 可视化和轨迹规划空间

---

## 速查卡片

> [!summary] RISE: Adaptive Imagination for World Action Models
> - **核心**: [[WAM]] 自适应 rollout 调度，按场景 Future Planning Gain 动态 Roll/Stop
> - **方法**: Latent Evaluator（预测风险+增益）+ Rollout Gate（二值决策）+ CounterDrive（反事实训练数据）
> - **结果**: NAVSIM v1 91.5 PDMS（+0.8 SOTA），NAVSIM v2 90.8 EPDMS，nuScenes L2=0.31m / 碰撞率=0.10
> - **代码**: 未开源（项目主页: https://cowarobot-ai.github.io/RISE/）

---

*笔记创建时间: 2026-08-25*

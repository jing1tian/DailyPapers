---
title: "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning"
method_name: "H-JEPA"
authors: [Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, jepa, hierarchical-planning, temporal-abstraction, robot-manipulation, self-supervised-learning, visual-planning]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.06805v2
created: 2026-11-04
---

# 论文笔记：H-JEPA — End-to-End Learning of Hierarchical World Models for Visual Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NYU · Meta AMI · INRIA Paris · Brown University |
| 日期 | October 2026 |
| 项目主页 | [h-jepa.com](https://h-jepa.com/) |
| 对比基线 | [[LeWM]] · [[HWM]] · [[DINO-WM]] · [[PLDM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.06805) / [HTML](https://arxiv.org/html/2610.06805v2) / [Code](https://github.com/kevinghst/H-JEPA) |

---

## 一句话总结

> 将 [[JEPA|JEPA 世界模型]] 堆叠成多级层次结构，每层在各自学得的潜空间中做更长时域的预测，[[Hierarchical Planning|层次化规划]] 自顶向下传递子目标，在 Visual AntMaze 上将成功率从 18% 提升至 73%，同时降低规划算力开销。

---

## 核心贡献

1. **端到端层次世界模型**: 提出 H-JEPA——多层 [[JEPA]] 堆叠，每层均有独立编码器与预测器，统一端到端训练，无需解码器或奖励信号
2. **自发选择性抽象**: 实证表明当相邻层动作频率比 ≥ 2× 时，高层自动丢弃快速细节、保留慢速任务相关特征（FourRoom、AntMaze、Humanoid 中均观察到）
3. **层次规划性能增益**: 相比平铺 [[LeWM]] 基线，三层 H-JEPA 在 Visual AntMaze 成功率从 18% 升至 73%，且使用更少规划算力
4. **扩展至真实机器人视频**: 通过[[IDM|逆动力学监督]]（IDM）防止潜空间坍塌，在无动作标注的 [[DROID]] 数据集上可学习层次世界模型

---

## 问题背景

### 要解决的问题

现有基于 [[JEPA]] 的世界模型（如 [[LeWM]]、[[DINO-WM]]）只在单一潜空间内、单一时间尺度上做预测规划。对于跨房间导航这类需要长时域规划的任务，低层潜空间需滚动大量步骤，导致预测误差累积严重，且目标匹配代价曲面不单调（Spearman 相关度仅 0.48）。

### 现有方法的局限

- **预测误差累积**: 单层模型滚动预测步数越多，误差指数增长
- **抽象语义缺失**: 低层潜变量携带大量动态背景噪声，难以精确匹配抽象目标
- **规划算力高昂**: 长视界规划需大量 [[CEM|CEM]] 候选样本，计算代价大

### 本文的动机

层次世界模型可以让高层处理"去哪里"的慢变量，低层处理"怎么走"的快变量，自然实现[[Temporal Abstraction|时序抽象]]。层次化规划使高层以低时间分辨率做长视界规划，提供子目标供低层追踪，解耦了抽象目标与精细动作的不匹配。

---

## 方法详解

### 模型架构

H-JEPA 采用 **多层堆叠 [[JEPA]] 预测架构**：

- **输入**: 像素观测 $o_t$（通过 [[ViT]] 编码）+ 低维动作 $a_t$
- **层级结构**: $L$ 层，$\ell=1$ 为最细粒度，$\ell=L$ 为最粗粒度
- **层间关系**: 每层以上一层的潜变量序列为输入，以步幅 $s_\ell$、窗口 $w_\ell$ 聚合时序信息
- **Level 1 Backbone**: [[ViT]]-Tiny（FourRoom/Push-T/Cube 用 ViT-Tiny/14，AntMaze 用 ViT-Tiny/4）+ MLP 投影头
- **Level 1 Predictor**: 因果 [[Causal Transformer]]（4 层、8 头、宽度 192，FeedForward 1024）
- **上层编码器**: 两层 MLP（隐层宽度 384）
- **总参数**: 约 15M（FourRoom 单层）

### 核心模块

#### 模块一：分层潜状态与动作编码

**设计动机**: 利用[[Temporal Abstraction|时序抽象]]将低频慢变量逐层提炼，使高层"看不见"快速扰动

**具体实现**:
- 编码器 $E^{(\ell)}$ 接收一段 $w_\ell$ 步的低层潜变量窗口，输出单一高层潜变量
- 动作编码器 $A^{(\ell)}$ 对对应窗口内的低层动作求池化，形成粗粒度宏动作
- 步幅 $s_\ell$ 控制时间压缩比，窗口 $w_\ell$ 控制感受野

#### 模块二：端到端预测学习

**设计动机**: 通过 [[SIGReg]] 正则化防止潜空间坍塌，同时用预测损失保留动力学信息

**具体实现**:
- 预测器 $F^{(\ell)}$ 在各自的潜空间内做因果自回归预测（Teacher Forcing）
- [[SIGReg]] 通过随机方向投影上的 Epps-Pulley 正态性统计量来推动潜变量分布多样性
- 各层独立计算预测损失并加权求和，联合反向传播端到端更新所有编码器

#### 模块三：自顶向下层次规划

**设计动机**: 高层先解决"去哪"，低层再负责"怎么走"，解耦抽象目标与精细控制

**具体实现**:
- 顶层用 [[CEM]] 在 $H_L$ 步视界内最小化终点目标距离（Eq. 4）
- 通过**子目标提升**（Eq. 5）：将低层的预测轨迹重新编码到高层空间，得到中间子目标序列
- 低层再次用 [[CEM]] 追踪这些子目标（Eq. 6），权重 $\beta$ 平衡终点与中间子目标

---

## 关键公式

### 公式 1：[[JEPA|层次潜状态与动作编码]]（Eq. 1）

$$
z^{(\ell)}_t = E^{(\ell)}\!\left(z^{(\ell-1)}_{t\cdot s_\ell\,:\,t\cdot s_\ell + w_\ell}\right), \quad a^{(\ell)}_t = A^{(\ell)}\!\left(a^{(\ell-1)}_{t\cdot s_\ell + w_\ell - 1\,:\,(t+1)\cdot s_\ell + w_\ell - 1}\right)
$$

**含义**: 第 $\ell$ 层潜变量由第 $\ell\!-\!1$ 层连续 $w_\ell$ 步的潜变量序列经编码器 $E^{(\ell)}$ 聚合而得；宏动作类似地由对应步幅窗口内的低层动作聚合

**符号说明**:
- $z^{(\ell)}_t$: 第 $\ell$ 层、时刻 $t$ 的潜变量
- $s_\ell$: 第 $\ell$ 层步幅（时间压缩比）
- $w_\ell$: 聚合窗口大小
- $E^{(\ell)}$: 第 $\ell$ 层编码器；$A^{(\ell)}$: 第 $\ell$ 层动作编码器

### 公式 2：[[JEPA|Teacher-Forced 预测损失]]（Eq. 2）

$$
\mathcal{L}^{(\ell)}_{\mathrm{pred}} = \frac{1}{c_\ell}\sum_{\tau=1}^{c_\ell} \left\| F^{(\ell)}(z^{(\ell)}_{<\tau},\, a^{(\ell)}_{<\tau}) - z^{(\ell)}_{\tau+1} \right\|_2^2
$$

**含义**: 在上下文长度 $c_\ell$ 内，用预测器 $F^{(\ell)}$ 滚动预测下一步潜变量并与目标编码计算 L2 误差

**符号说明**:
- $c_\ell$: 第 $\ell$ 层上下文长度
- $F^{(\ell)}$: 第 $\ell$ 层预测器（Level 1 为因果 Transformer，上层为 MLP）
- $z^{(\ell)}_{<\tau}$: 前 $\tau$ 步潜变量历史

### 公式 3：[[SIGReg|每层训练目标]]（Eq. 3）

$$
\mathcal{L}^{(\ell)} = \mathcal{L}^{(\ell)}_{\mathrm{pred}} + \lambda_\ell \,\mathrm{SIGReg}\!\left(Z^{(\ell)}\right)
$$

**含义**: 每层的总损失由预测损失与 [[SIGReg]] 正则项加权组成，防止潜空间坍塌同时保留动力学可预测性

**符号说明**:
- $\lambda_\ell$: [[SIGReg]] 权重（Level 1 约 0.09，FourRoom LeWM 用 0.72）
- $\mathrm{SIGReg}(Z^{(\ell)})$: 通过随机投影方向上 Epps-Pulley 正态性统计量度量嵌入多样性

### 公式 4：[[Hierarchical Planning|顶层规划目标]]（Eq. 4）

$$
a^{(L),*}_{0:H_L-1} = \arg\min_{a^{(L)}_{0:H_L-1}} \left\| \hat{z}^{(L)}_{H_L} - g^{(L)} \right\|_2^2
$$

**含义**: 在最高层（$L$ 层）的潜空间内，用 [[CEM]] 寻找使 $H_L$ 步后预测潜变量逼近目标 $g^{(L)}$ 的动作序列

**符号说明**:
- $H_L$: 顶层规划视界
- $\hat{z}^{(L)}_{H_L}$: 顶层预测器自回归滚动 $H_L$ 步后的预测潜变量
- $g^{(L)}$: 目标帧经顶层编码器编码后的目标潜变量

### 公式 5：[[Hierarchical Planning|子目标提升]]（Eq. 5）

$$
\tilde{z}^{(\ell+1)}_i = E^{(\ell+1)}\!\left(\hat{z}^{(\ell)}_{i s_{\ell+1}\,:\,i s_{\ell+1} + w_{\ell+1}}\right), \quad i = 1, \ldots, K_{\ell+1}
$$

**含义**: 将 $\ell$ 层的预测轨迹片段重新编码到 $\ell\!+\!1$ 层空间，得到 $K_{\ell+1}$ 个中间子目标，连接高低层的规划

**符号说明**:
- $\tilde{z}^{(\ell+1)}_i$: 第 $i$ 个提升子目标（在 $\ell\!+\!1$ 层潜空间）
- $K_{\ell+1}$: 子目标数量

### 公式 6：[[Hierarchical Planning|低层子目标追踪规划]]（Eq. 6）

$$
a^{(\ell),*}_{0:H_\ell-1} = \arg\min_{a^{(\ell)}_{0:H_\ell-1}} \left[ \left\| \tilde{z}^{(\ell+1)}_{K_{\ell+1}} - \hat{z}^{(\ell+1),*}_{K_{\ell+1}} \right\|_2^2 + \beta \sum_{i=1}^{K_{\ell+1}-1} \left\| \tilde{z}^{(\ell+1)}_i - \hat{z}^{(\ell+1),*}_i \right\|_2^2 \right]
$$

**含义**: 低层（$\ell$ 层）规划同时最小化终端子目标误差与中间子目标误差，$\beta$ 平衡两者的权重

**符号说明**:
- $\beta$: 中间子目标权重（$\beta \geq 0$）
- $\hat{z}^{(\ell+1),*}_i$: 顶层规划确定动作后，对应的高层预测潜变量
- 终端项（$i=K_{\ell+1}$）使轨迹抵达目标，中间项使轨迹形状接近高层规划路径

### 公式 7：[[Fréchet Distance|Fréchet 路径保真度]]（Eq. 7）

$$
\text{fidelity} = \frac{d_F(\mathbf{0},\,\tilde{p}) - d_F(p,\,\tilde{p})}{d_F(\mathbf{0},\,\tilde{p})}
$$

**含义**: 衡量规划路径 $p$ 与专家路径 $\tilde{p}$ 的相似度，分母是以零点为基线的 [[Fréchet Distance|Fréchet 距离]]，取值越接近 1 表示规划路径越忠实于专家轨迹

**符号说明**:
- $d_F(\cdot,\cdot)$: [[Fréchet Distance|Fréchet 距离]]（衡量两条曲线在形状上的最优匹配距离）
- $p$: 规划 xyz 位置轨迹
- $\tilde{p}$: 专家 xyz 位置轨迹

### 公式 8：[[IDM|逆动力学监督损失]]（Eq. 8）

$$
\mathcal{L}^{(\ell)}_{\mathrm{idm}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \frac{1}{d_a} \left\| I^{(\ell)}\!\left(z^{(\ell)}_t,\, z^{(\ell)}_{t+1}\right) - \operatorname{sg}\!\left[a^{(\ell)}_t\right] \right\|_2^2
$$

**含义**: 给定相邻两帧潜变量，[[IDM|逆动力学头]] $I^{(\ell)}$ 预测执行的动作；通过 stop-gradient 阻止编码器从动作中"抄捷径"，迫使潜变量真正编码状态信息以防止坍塌

**符号说明**:
- $I^{(\ell)}$: 第 $\ell$ 层逆动力学头（MLP）
- $d_a$: 动作维度
- $\operatorname{sg}[\cdot]$: stop-gradient 操作（动作标签不参与编码器梯度回传）

---

## 关键图表

### Figure 1: 系统概览与 AntMaze Pareto 前沿

![Figure 1 - Overview](https://h-jepa.com/assets/figures/overview.png)

**说明**: 左图展示三层 H-JEPA 在 Visual AntMaze 中的层次规划过程——顶层（Level 3）生成粗粒度路径，中层（Level 2）细化，底层（Level 1）执行；右图为规划成功率 vs 算力预算的 Pareto 前沿，H-JEPA 三层以更低算力达到更高成功率，主导 [[LeWM]] 基线。

### Figure 2: 分层深度探测（Selective Abstraction）

![Figure 2 - Depth Probes](https://h-jepa.com/assets/figures/depth_probes_wide.png)

**说明**: 探针实验显示，在 FourRoom 和 AntMaze 环境中，随层级升高（Level 1 → Level 3），位置预测线性探针误差显著下降，而速度预测误差升高——说明高层自发学会[[Temporal Abstraction|时序抽象]]，抛弃快速动力学、保留慢变位置信息。

### Figure 3: 目标代价曲面（Cost Geometry）

![Figure 3 - Goal Costs](https://h-jepa.com/assets/figures/goal-costs.png)

**说明**: 不同层次潜空间中规划代价曲面的可视化热图。高层代价曲面更加平滑单调（Spearman 相关度 0.89–1.00），而低层曲面崎岖不平（相关度 0.48–0.75），证明在高层空间做目标追踪更可靠。

### Figure 4: 四环境规划性能 Pareto 对比

![Figure 4 - Planning Results](https://h-jepa.com/assets/figures/planning-results-pareto.png)

**说明**: 在 FourRoom、AntMaze、Cube 四个环境上，H-JEPA 层次规划器（蓝线）在多个算力预算点均优于或等于平铺 [[LeWM]] 基线（灰线）；Push-T 中三层结构因训练数据减少而性能下滑（见消融讨论）。

### Figure 5: 场景多样性——Cube 与 DROID 视频

**说明**: 视频展示 Cube 操作与 [[DROID]] 真实机器人场景，说明 H-JEPA 在无模拟器的开放场景视频中也能学习层次表示。（原始为视频，无静态图）

### Figure 6: DROID 消融实验

![Figure 6 - DROID Ablation](https://h-jepa.com/assets/figures/droid-ablation.png)

**说明**: 在 [[DROID]] 数据集上比较四种方案：LeWM（平铺）、LeWM + IDM、[[HWM]]、H-JEPA。结果表明 [[IDM|逆动力学损失]] 是防止 DROID 场景下潜空间坍塌的关键，H-JEPA 在低规划算力下显著优于所有基线。

### Figure 7: DROID 规划保真度 vs 算力

![Figure 7 - DROID Compute](https://h-jepa.com/assets/figures/droid-compute.png)

**说明**: [[Fréchet Distance|Fréchet 路径保真度]] 随规划候选样本数增加的曲线，H-JEPA 曲线在整个算力区间均高于平铺模型，以更少样本达到同等保真度。

### Table 1：AntMaze 成功率（固定模型深度、仅变换规划代价空间）

| 代价空间 / 模型深度 | 2 层 | 3 层 | 4 层 |
|---|---|---|---|
| Native L1（底层原生） | 23.3 ± 3.3 | 16.7 ± 1.8 | 10.0 ± 0.0 |
| L2 层投影代价 | 31.3 ± 2.7 | 20.7 ± 4.8 | 20.7 ± 1.8 |
| L3 层投影代价 | — | 22.0 ± 2.3 | 21.3 ± 4.7 |
| L4 层投影代价 | — | — | 26.7 ± 2.9 |
| **H-JEPA 层次规划器** | **39.3 ± 3.7** | **73.3 ± 3.5** | **63.3 ± 7.7** |
| LeWM 基线（1 层） | 18.0 ± 3.5 | — | — |

**关键发现**: 层次规划器大幅超越所有单层投影代价方案，三层达到最优；四层轻微下滑可能因训练数据被更长片段稀释。

### Table 2：FourRoom / Cube / Push-T 规划代价消融（成功率 %，附录 Table 9）

| 代价空间 / 环境 | FourRoom 2层 | FourRoom 3层 | FourRoom 4层 | Cube 3层 | Push-T 2层 |
|---|---|---|---|---|---|
| Native L1 | 9.3 ± 0.7 | 43.3 ± 8.7 | 68.7 ± 4.7 | 24.7 ± 1.3 | 36.7 ± 1.8 |
| 最高层投影 | 38.7 ± 6.6 | 96.0 ± 1.1 | 70.7 ± 2.7 | 60.0 ± 5.0 | 45.3 ± 1.3 |
| **层次规划器** | — | — | **80.0 ± 4.0** | — | — |
| LeWM 基线 | 40.7 ± 6.8 | — | — | 32.0 ± 6.1 | 40.0 ± 2.3 |

**关键发现**: FourRoom 中最高层投影代价单独即可大幅超越基线（3 层达 96%）；Push-T 中三层因片段数仅剩基线 14%，性能显著下滑。

### Table 3：训练细节（单场景环境，参考附录 Table 4）

| 参数 | FourRoom | AntMaze | Push-T | Cube |
|---|---|---|---|---|
| 帧尺寸 / Patch | 224 / 14 | 64 / 4 | 224 / 14 | 224 / 14 |
| 潜变量维度 | 192 | 384 | 192 | 192 |
| H-JEPA 训练轮数 | 7 | 5 | 9 | 8 |
| SIGReg λ（H-JEPA） | 0.18 | 0.09 | 0.09 | 0.09 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| FourRoom | 1.0M 转移 | 2D 迷宫导航，含一个干扰物 | 单场景训练与规划评估 |
| Visual AntMaze | 10.05M 转移 | OGBench explore+stitch 混合 | 视觉 Ant 机器人导航，主要 benchmark |
| Push-T | 专家演示集 | T 型积木推送，75步一对帧评估 | 操作规划 |
| Cube | 10,000 轨迹 (200步/轨迹) | 拾取-放置，5-DoF 控制 | 操作规划 |
| [[DROID]] | 74,530 轨迹，220,084 视图 | 真实机器人操作，3 相机 256×256 | 离线保真度评估 |

### 实现细节

- **Level 1 Backbone**: [[ViT]]-Tiny，MLP 投影头（FourRoom/Push-T/Cube: ViT-Tiny/14；AntMaze: ViT-Tiny/4）
- **Level 1 Predictor**: 因果 Transformer（4层、8头、宽度 192、FF 1024，FourRoom）
- **上层编码器/预测器**: 两层 MLP（隐层宽度 384）
- **优化器**: AdamW，权重衰减 1e-3，1% 线性预热 + 余弦衰减，梯度裁剪 1.0，bfloat16
- **学习率**: Level 1 峰值 5e-5，上层 1e-5
- **Batch Size**: 128 个片段（单场景）；256（DROID）
- **硬件**: H200 GPU（使用 `stable-worldmodel` 和 `stable-pretraining` 库）
- **随机种子**: 42, 43, 44（单场景）；1, 1000, 10000（DROID）

### DROID 评估协议

- **离线评估**：规划在冻结的 16 条专家拾放轨迹上打分（全程排除训练集外）
- **指标**：Fréchet 路径保真度（Eq. 7），衡量 xyz 累计路径形状相似度
- **模型输入**：仅单外部相机像素 + 7-DoF 动作，无本体感知

---

## 批判性思考

### 优点

1. **原理清晰**: 层次 [[JEPA]] 的动机直接明了，时序抽象的出现方式可验证（探针实验）
2. **无需解码器/奖励**: 完全自监督，[[SIGReg]] 防坍塌方案已在 [[LeWM]] 中验证
3. **显著性能提升**: 在 AntMaze 上从 18% 到 73%，是同框架内质的飞跃
4. **扩展至真实数据**: DROID 结果证明方法不依赖模拟器，具有实际部署潜力

### 局限性

1. **Push-T 不稳定**: 三层/四层因每轮可用训练片段减少（仅剩 14%），性能反而弱于基线；短轨迹任务层次扩展性存疑
2. **离线评估**: DROID 实验只做离线轨迹打分，未在真实机器人闭环执行
3. **需要动作标签**: [[IDM]] 防坍塌需要动作监督，扩展到无标注视频需要替代方案；单独 [[SIGReg]] 在 DROID 上不够
4. **单相机输入**: 忽略腕部相机，遗失精细位置信息（作者指出这是提升精度最直接路径）
5. **Fréchet 指标局限**: 高保真度路径不等于操作成功（如抓取失败的轨迹仍可得高分）

### 潜在改进方向

1. **可变时序步幅**: 自适应段长度（非固定 $s_\ell$），可能缓解短轨迹数据稀疏问题
2. **语言对齐上层**: 顶层潜空间与语言嵌入对齐，支持指令条件规划
3. **无动作视频学习**: 开发不依赖动作标签的防坍塌机制，拓展到大规模无标注视频预训练

### 可复现性评估

- [x] 代码开源（GitHub: kevinghst/H-JEPA）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（附录中有详细超参数）
- [x] 数据集可获取（OGBench、DROID 均开源）

---

## 关联笔记

### 基于

- [[LeWM]]: 直接基于 LeWM 的 [[SIGReg]] 防坍塌框架，H-JEPA 是其层次扩展
- [[JEPA]]: 联合嵌入预测架构的核心设计哲学
- [[V-JEPA]]: 视频 JEPA 的前置工作，提供视频潜空间预测思路

### 对比

- [[LeWM]]: 单层平铺基线，H-JEPA 的直接对比方法
- [[HWM]]: 层次世界模型基线，与 H-JEPA 的区别在于 HWM 在同一潜空间做层次规划（无独立学得的高层空间）
- [[DINO-WM]]: 基于 DINO 特征的世界模型基线
- [[PLDM]]: 另一个 JEPA 类世界模型基线

### 方法相关

- [[JEPA]]: 核心架构范式
- [[SIGReg]]: 防坍塌正则化
- [[IDM]]: 逆动力学监督（DROID 场景关键）
- [[CEM]]: 交叉熵方法，规划优化器
- [[Temporal Abstraction]]: 层次世界模型的核心设计动机
- [[Hierarchical Planning]]: 自顶向下的子目标分解规划框架
- [[Fréchet Distance]]: DROID 评估指标

### 硬件/数据相关

- [[DROID]]: 真实机器人操作视频数据集
- [[ViT]]: Level 1 视觉编码器主干

---

## 速查卡片

> [!summary] H-JEPA (arXiv 2610.06805)
> - **核心**: 多层 JEPA 堆叠 + 自顶向下层次规划，每层在自己的潜空间做时序预测
> - **方法**: SIGReg 防坍塌 + 子目标提升（Eq. 5）连接相邻层规划
> - **结果**: AntMaze 18% → 73%（三层），DROID 离线规划超 HWM 和 LeWM
> - **代码**: [github.com/kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA)

---

*笔记创建时间: 2026-11-04*

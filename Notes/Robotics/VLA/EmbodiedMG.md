---
title: "Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting"
method_name: "EmbodiedMG"
authors: [Huosen Ou, Dongni Song, Yuncong Wang, Tao Zhou, Yiding Ji]
year: 2026
venue: ACM Multimedia 2026 (MM '26)
tags: [mobile-manipulation, 3d-gaussian-splatting, open-vocabulary, vision-language-action, grounding, quadruped-robot]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.10756v1
created: 2026-08-13
---

# 论文笔记：Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST Guangzhou + Midea Group |
| 日期 | August 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[PointVLA]] / [[DexVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.10756) |

---

## 一句话总结

> 将主动多视角 [[Semantic-3DGS]] 与可达性感知底盘定位、后期块 VLA 条件化融为一体，让四足机器人以 60% 成功率完成长链路开放词汇移动操作，远超 PointVLA（40%）和 DexVLA（28%）。

---

## 核心贡献

1. **主动多视角 Semantic-3DGS 构建**: 机器人主动选择 4 个腕部相机视角，利用 [[VGGT]] 进行几何初始化，蒸馏 [[CLIP]] 和 [[DINOv2]] 特征，构建紧凑的局部语义 3D 高斯场，支持开放词汇目标定位。
2. **可达性感知底盘定位**: 基于目标三维位置计算最优站姿，使用 [[PPO]] 训练残差腿关节控制，支持高度自适应（站姿/蹲姿切换），在 75 cm 高度变化下仍保持 75–80% 成功率。
3. **后期块 VLA 条件化**: 仅将 3D 语义信息注入扩散动作专家的最后 5 个 Transformer 块，通过可训练 adapter 融合，保留预训练动作先验的同时将动作推理延迟降至 80 ms/chunk。

---

## 问题背景

### 要解决的问题

开放词汇移动操作（Open-Vocabulary Mobile Manipulation）要求机器人根据自然语言指令抓取任意目标物体，涵盖目标定位、底盘姿态调整和灵巧手臂操作三个耦合环节，且目标可能被杂乱物体遮挡或处于未见过的高度。

### 现有方法的局限

- [[PointVLA]] 等基于点云的方法依赖全局点云，缺乏局部语义精度，对杂乱场景鲁棒性不足（46% vs 本文 74%）。
- [[DexVLA]] 等纯 VLA 方法没有显式的 3D 场景理解，无法判断目标真实位置，容易被照片级干扰物欺骗（假抓率 76%）。
- 已有 [[Semantic-3DGS]] 方法（如 LangSplat）面向离线大规模场景，不适合机器人实时、局部、少视图的操作需求。
- 没有对 VLA 条件化注入位置的系统研究：过早注入（early-block）会破坏预训练动作先验，导致动作质量下降。

### 本文的动机

将**局部视角主动采集**、**3D 语义高斯场**和**可达性感知定位**统一为一个轻量的局部感知 pipeline，再通过**后期块注入**将 3D 语义与 VLA 扩散策略对接，在保留少样本操控能力的前提下，突破开放词汇定位的瓶颈。

---

## 方法详解

### 系统概览

![Figure 1: 系统总体流程](https://arxiv.org/html/2608.10756v1/figures/flow_chart.png)

**说明**: 机器人巡逻导航，接到语言指令 $l$ 后切换到任务模式：主动采集 4 张腕部相机图像 → 构建局部 [[Semantic-3DGS]] → CLIP 语义评分定位目标 → 可达性感知底盘定位 → [[扩散策略|后期块条件化 VLA 动作推理]]。

### 硬件平台

![Figure 2: 机器人平台](https://arxiv.org/html/2608.10756v1/figures/model.png)

**说明**: Unitree Go2 Edu 四足机器人 + 6-DoF 机械臂 + 腕部 RGB 相机。板载 Jetson Orin NX 处理低层控制，离线 RTX 4090 负责感知与规划。

---

### 模块 1: 主动多视角 Semantic-3DGS

![Figure 3: 主动多视角语义 3DGS 构建](https://arxiv.org/html/2608.10756v1/figures/active_multiview_sem3dgs.png)

**说明**: 主动视角选择驱动腕部相机从 4 个最优姿态采集图像，[[VGGT]] 提供几何初始化，高斯场蒸馏 [[CLIP]] 和 [[DINOv2]] 特征，最终支持 CLIP 文本查询。

#### 1a. 局部高斯场表示

每个高斯原子 $g_k$ 编码几何、外观和双路语义特征：

### 公式 1: [[Semantic-3DGS|局部语义高斯场定义]]

$$
\mathcal{G}=\{g_{k}\}_{k=1}^{K},\qquad g_{k}=(\mu_{k},\Sigma_{k},c_{k},\alpha_{k},f_{k}^{C},f_{k}^{D})
$$

**含义**: 整个局部场景由 $K$ 个高斯原子描述，每个原子携带完整的几何与语义属性。

**符号说明**:
- $\mu_k \in \mathbb{R}^3$: 高斯中心位置（世界坐标）
- $\Sigma_k$: 协方差矩阵（控制形状/方向）
- $c_k$: 颜色
- $\alpha_k$: 不透明度
- $f_k^C$: CLIP 对齐特征（用于语言 Grounding）
- $f_k^D$: DINOv2 特征（提供细粒度外观信息）

#### 1b. 主动视角选择

候选相机姿态按联合得分排序，贪婪选出信息量最大的视角：

### 公式 2: [[主动视角选择|主动视角评分函数]]

$$
J(v)=\lambda_{\mathrm{cov}}C_{\mathrm{sem}}(v)+\lambda_{\mathrm{par}}C_{\mathrm{par}}(v)-\lambda_{\mathrm{move}}C_{\mathrm{move}}(v)
$$

**含义**: 平衡语义覆盖、视角多样性和运动代价的综合得分，驱动机器人高效选择最有信息量的腕部相机姿态。

**符号说明**:
- $C_{\mathrm{sem}}(v)$: 语义覆盖期望（候选视角能观测到多少语义区域）
- $C_{\mathrm{par}}(v)$: 视角多样性奖励（与已选视角的角度差异）
- $C_{\mathrm{move}}(v)$: 运动代价（到达该姿态所需关节运动量）
- $\lambda_{\mathrm{cov}}, \lambda_{\mathrm{par}}, \lambda_{\mathrm{move}}$: 权重系数

### 公式 3: [[主动视角选择|贪婪视角选择]]

$$
v_{n+1}=\operatorname*{arg\,max}_{v\in\mathcal{V}_{\mathrm{cand}}}J(v)
$$

**含义**: 从候选姿态集合 $\mathcal{V}_{\mathrm{cand}}$ 中选出得分最高的下一个视角，循环执行直至采集 4 张图像。

#### 1c. 几何初始化

### 公式 4: [[VGGT|VGGT 多视角几何估计]]

$$
\{(\hat{g}^{(i)},\hat{D}^{(i)},\hat{P}^{(i)})\}_{i=1}^{N}=\Phi_{\mathrm{VGGT}}(\mathcal{I})
$$

**含义**: 将 $N=4$ 张图像输入 [[VGGT]] Transformer，一次性输出每张图的相机内参 $\hat{g}^{(i)}$、深度图 $\hat{D}^{(i)}$ 和点图 $\hat{P}^{(i)}$，无需任何 SfM 或标定。

**符号说明**:
- $\mathcal{I} = \{I^{(i)}\}_{i=1}^{N}$: 输入图像集合
- $\hat{g}^{(i)}$: 估计的相机姿态
- $\hat{D}^{(i)}$: 估计的深度图
- $\hat{P}^{(i)}$: 估计的点图（点云）

#### 1d. 语义蒸馏

通过渲染-特征对比损失将 CLIP 和 DINOv2 特征蒸馏进高斯场：

### 公式 5: [[Semantic-3DGS|语义蒸馏损失]]

$$
\mathcal{L}_{\mathrm{feat}}=\sum_{i,u}\left(1-\cos(\hat{F}_{C}^{(i)}(u),\tilde{F}_{C}^{(i)}(u))\right)+\lambda_{D}\sum_{i,u}\left(1-\cos(\hat{F}_{D}^{(i)}(u),F_{D}^{(i)}(u))\right)
$$

**含义**: 最小化高斯场渲染出的 CLIP/DINOv2 特征图与真实特征提取器输出之间的余弦距离，实现语义对齐。

**符号说明**:
- $\hat{F}_C^{(i)}(u)$: 第 $i$ 视角像素 $u$ 处渲染的 CLIP 特征
- $\tilde{F}_C^{(i)}(u)$: 真实 CLIP 编码器提取的特征
- $F_D^{(i)}(u)$: 真实 DINOv2 特征
- $\lambda_D$: DINOv2 损失权重

---

### 模块 2: 开放词汇语言 Grounding

#### 2a. 语言相关性评分

### 公式 6: [[CLIP|CLIP 语义相关性评分]]

$$
s_{k}=\mathrm{softmax}\!\left(\tau[\cos(f_{k}^{C},e^{+}),\cos(f_{k}^{C},e^{-}_{1}),\ldots,\cos(f_{k}^{C},e^{-}_{m})]\right)_{1}
$$

**含义**: 每个高斯原子对目标语言描述的相关性得分——正向提示词的 softmax 概率；通过引入负向提示词抑制背景噪声。

**符号说明**:
- $f_k^C$: 第 $k$ 个高斯的 CLIP 特征
- $e^+$: 目标描述的 CLIP 文本嵌入（如 "banana"）
- $e^-_1, \ldots, e^-_m$: 负向提示词嵌入（如 "background", "table"）
- $\tau$: 温度系数
- $(\cdot)_1$: 取 softmax 输出中对应正向提示的分量

#### 2b. 目标 3D 位置估计

### 公式 7: [[Semantic-3DGS|加权目标位置估计]]

$$
p^{W}_{\mathrm{obj}}=\frac{\sum_{k\in\mathcal{K}}w_{k}\mu_{k}}{\sum_{k\in\mathcal{K}}w_{k}},\qquad w_{k}=\alpha_{k}\,\mathrm{trace}(\Sigma_{k})^{-1}
$$

**含义**: 利用相关性得分过滤后的高斯原子集合 $\mathcal{K}$，以不透明度和几何紧凑性为权重估计目标在世界坐标系中的 3D 位置。

**符号说明**:
- $\mathcal{K}$: 相关性得分超过阈值的高斯原子集合
- $w_k = \alpha_k \cdot \mathrm{trace}(\Sigma_k)^{-1}$: 越不透明、越紧凑的高斯权重越大
- $p^W_{\mathrm{obj}}$: 目标在世界坐标系中的估计位置

### 公式 8: [[多视角几何|坐标系变换]]

$$
p^{B}_{\mathrm{obj}}=({}^{W}T_{B,t})^{-1}p^{W}_{\mathrm{obj}}=[x_{\mathrm{obj}},y_{\mathrm{obj}},z_{\mathrm{obj}}]^{\top}
$$

**含义**: 将目标世界坐标转换到当前时刻机器人底盘坐标系，供后续底盘定位使用。

**符号说明**:
- ${}^{W}T_{B,t}$: 时刻 $t$ 底盘到世界坐标系的变换矩阵

---

### 模块 3: 可达性感知底盘定位

#### 3a. 几何目标站姿

### 公式 9: [[可达性感知底盘定位|目标底盘站姿计算]]

$$
x^{\star}=x_{\mathrm{obj}}-d_{x},\quad y^{\star}=y_{\mathrm{obj}}-\mathrm{sign}(y_{\mathrm{obj}})d_{y},\quad\psi^{\star}=\mathrm{atan2}(y_{\mathrm{obj}},x_{\mathrm{obj}})
$$

**含义**: 根据目标在底盘坐标系中的位置，计算操作前的最优底盘位置 $(x^\star, y^\star)$ 和朝向 $\psi^\star$，使机械臂工作空间覆盖目标。

**符号说明**:
- $d_x, d_y$: 目标与底盘的最优纵横偏移（由臂长和可达性分析确定）
- $\psi^\star$: 偏航角（yaw），使机器人朝向目标

#### 3b. 高度自适应站姿选择

### 公式 10: [[可达性感知底盘定位|离散高度模式选择]]

$$
h^{\star}=\begin{cases}h^{\mathrm{crouch}},&z_{\mathrm{obj}}<0.30\text{ m},\\ h^{\mathrm{stand}},&\text{otherwise}.\end{cases}
$$

**含义**: 根据目标高度 $z_{\mathrm{obj}}$ 离散切换站姿（蹲姿/站姿），使机械臂基座高度适应目标位置。

**符号说明**:
- $h^{\mathrm{crouch}}$: 蹲姿机身高度（低目标）
- $h^{\mathrm{stand}}$: 站姿机身高度（高目标）
- 0.30 m: 站/蹲切换阈值

**高度测试配置**:

![Figure 6: 高度自适应测试](https://arxiv.org/html/2608.10756v1/figures/height_adjustment_4.png)

**说明**: 目标置于不同高度平台（0→30→60→75 cm 偏移）测试站姿选择和可达性操作。无 Base-RL 变体在所有高度下完全失败（0%），说明底盘定位对高度泛化至关重要。

基站定位还通过 [[PPO]] 策略输出腿关节残差，实现精确的位置与姿态调节，在仿真中训练并零样本迁移到真实机器人。

---

### 模块 4: Semantic-3DGS 条件化 VLA 策略

![Figure 4: 后期块 VLA 条件化架构](https://arxiv.org/html/2608.10756v1/figures/semantic_3dgs_vla_v2.png)

**说明**: 轻量语义注入器（Semantic Injector）通过可训练 adapter 将 3D 语义特征注入扩散动作专家的最后 5 个 Transformer 块，冻结其他块，保留预训练动作先验。

#### 4a. 视觉观测编码

### 公式 11: [[Vision-Language-Action|多模态视觉编码]]

$$
z_{t}^{\mathrm{img}}=E_{\mathrm{img}}(\mathrm{concat}(I_{t},H_{t},O_{t},P_{t}))\in\mathbb{R}^{128}
$$

**含义**: 将当前帧 $I_t$、语义热图 $H_t$（目标得分渲染）、障碍物占用图 $O_t$ 和点图 $P_t$ 拼接后编码，形成 128 维的多模态视觉特征。

**符号说明**:
- $I_t$: 当前 RGB 帧
- $H_t$: 语义相关性热图（目标位置的可视化）
- $O_t$: 障碍物占用图（防碰撞）
- $P_t$: 点图（3D 几何）
- $E_{\mathrm{img}}$: 视觉编码器

#### 4b. 后期块注入

### 公式 12: [[扩散策略|后期块索引集合]]

$$
\mathcal{B}=\{L-4,L-3,L-2,L-1,L\}
$$

**含义**: 仅在扩散动作专家的最后 5 个 Transformer 块注入语义特征，保留早期块中的预训练动作先验。

**符号说明**:
- $L$: 动作专家 Transformer 的最后一层索引
- $\mathcal{B}$: 语义注入的目标层集合

### 公式 13: [[扩散策略|语义特征后期块注入]]

$$
h_{\ell}\leftarrow h_{\ell}+A_{\ell}(\mathrm{Proj}(z_{t}^{\mathrm{sem}})),\qquad\ell\in\mathcal{B}
$$

**含义**: 将语义特征经投影层和可训练 adapter $A_\ell$ 后以残差形式加入各后期块的隐状态，轻量注入不破坏早期块的动作先验。

**符号说明**:
- $h_\ell$: 第 $\ell$ 层的隐状态
- $A_\ell$: 第 $\ell$ 层的可训练 adapter（参数量极小）
- $\mathrm{Proj}(\cdot)$: 线性投影层（对齐维度）
- $z_t^{\mathrm{sem}}$: 当前时刻的 Semantic-3DGS 渲染特征

---

## 关键图表

### Figure 5: 少样本多任务场景

![Figure 5: 少样本多任务操作场景](https://arxiv.org/html/2608.10756v1/figures/multitask.png)

**说明**: 三类任务：(1) 香蕉→书本，(2) 瓶子→篮子，(3) 有序摆放（填充玩具+香蕉+瓶子→碗）。测试物体为任务微调阶段未见的新实例，语义相关的物体类别在少样本演示中可能出现过。

### Figure 7: 长链路操作序列

![Figure 7: 长链路任务序列](https://arxiv.org/html/2608.10756v1/figures/long_horizon_task.png)

**说明**: 完整序列：打开抽屉 → 取出香蕉 → 关上抽屉 → 导航至目标区域 → 放置到黑色椅子上。

### Figure 8: 少样本多任务成功率

![Figure 8: 多任务成功率对比](https://arxiv.org/html/2608.10756v1/figures/success_rate.png)

**说明**: 本方法在全部三个任务上均最优，总体成功率 81.7% vs PointVLA 64.0% vs DexVLA 37.7%。

### Figure 9: 照片欺骗测试

![Figure 9: 照片欺骗实验设置](https://arxiv.org/html/2608.10756v1/figures/deceptive_questions.png)

**说明**: 机器人须操作真实目标物，同时拒绝显示屏上照片级干扰物。本文方法 88% 真实成功、0% 假抓；DexVLA 假抓率高达 76%。

### Figure 10: 杂乱场景基准

![Figure 10: 杂乱香蕉到碗基准](https://arxiv.org/html/2608.10756v1/figures/bannana2.png)

**说明**: 目标周围放置密集物理干扰物并强遮挡，测试 3D 语义定位的鲁棒性。本文方法 74% 成功、88% 无碰撞执行。

### Figure 11: 后期块注入敏感性分析

![Figure 11: 注入块位置敏感性分析](https://arxiv.org/html/2608.10756v1/figures/block_sensitivity.png)

**说明**: 后期块（last-5）注入在成功率-延迟上均优于全块注入（all-block），证明注入位置对保留预训练动作先验的重要性。

---

### Table 1: 长链路任务成功率（50 次试验）

| 方法 | 成功率 | 95% CI | 平均用时 (s) |
|------|--------|--------|------------|
| DexVLA | 14/50 (28%) | [17.5, 41.7] | 178.6±18.4 |
| PointVLA | 20/50 (40%) | [27.6, 53.8] | 161.8±17.6 |
| Ours w/o Base-RL | 11/50 (22%) | [12.8, 35.2] | 204.1±21.3 |
| **Ours (full)** | **30/50 (60%)** | **[46.2, 72.4]** | **140.7±15.9** |

**关键发现**: 去掉 Base-RL 后成功率从 60% 骤降至 22%（低于所有 baseline），说明可达性感知底盘定位是长链路任务的决定性模块。

### Table 2: 高度自适应（不同高度偏移下成功率）

| 方法 | 0→30 cm | 0→60 cm | 0→75 cm |
|------|---------|---------|---------|
| DexVLA | 48% | 33% | 23% |
| PointVLA | 58% | 46% | 35% |
| Ours w/o Base-RL | 0% | 0% | 0% |
| **Ours (full)** | **80%** | **78%** | **75%** |

**关键发现**: 无底盘高度优化时在所有高度偏移下完全失败；本文方法在 75 cm 偏移下仍保持 75% 成功率。

### Table 3: 照片欺骗鲁棒性

| 方法 | 真实成功率 ↑ | 照片假抓率 ↓ |
|------|------------|------------|
| DexVLA | 78% | 76% |
| PointVLA | 80% | 0% |
| Ours Single-View | 76% | 70% |
| **Ours (full)** | **88%** | **0%** |

**关键发现**: 多视角 Semantic-3DGS 通过 3D 定位完全消除了照片假抓（0%），同时单视角变体假抓率 70%，说明多视角聚合对区分 2D 图像和 3D 实体至关重要。

### Table 4: 杂乱场景基准（50 次试验）

| 方法 | 成功率 | 95% CI | 无碰撞 | 假抓率 | 平均用时 (s) |
|------|--------|--------|--------|--------|------------|
| DexVLA | 13/50 (26%) | [15.9, 39.6] | 22/50 (44%) | 16/50 (32%) | 27.9±3.1 |
| PointVLA | 23/50 (46%) | [33.0, 59.6] | 31/50 (62%) | 10/50 (20%) | 30.7±3.4 |
| Ours Single-View | 26/50 (52%) | [38.5, 65.2] | 35/50 (70%) | 9/50 (18%) | 29.5±3.2 |
| **Ours (full)** | **37/50 (74%)** | **[60.4, 84.1]** | **44/50 (88%)** | **3/50 (6%)** | 33.2±3.6 |

**关键发现**: 全系统（多视角）比单视角提升 22 个百分点（74% vs 52%），无碰撞执行率提高 18 个百分点（88% vs 70%），代价仅是额外 3.7 秒（感知阶段耗时）。

### Table 5: 消融实验——注入策略对比

| 变体 | 平均成功率 | 动作块推理延迟 |
|------|----------|--------------|
| All-block 注入 | 75% | 175 ms |
| **Late-block（最后 5 层）** | **82%** | **80 ms** |

**关键发现**: 后期块注入同时在成功率（+7%）和延迟（-95 ms/chunk）上优于全块注入，验证了保留早期块动作先验的必要性。

### Table 6: 系统各模块延迟分析

| 组件 | 平均延迟 |
|------|--------|
| 4 视角腕部运动+采集 | 16.00 s |
| VGGT 位姿/深度初始化 | 0.62 s |
| Semantic-3DGS 特征更新 | 1.21 s |
| 语义渲染+定位 | 0.34 s |
| VLA 动作块推理 | 0.08 s/chunk |
| ROS/WiFi 通信 | 0.05 s/chunk |
| **杂乱场景完整任务** | **33.2±3.6 s** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人演示 | 10 次/任务 | 少样本，任务特定 | VLA 微调 |
| 仿真环境（Base-RL） | — | 模拟多种障碍与高度 | PPO 训练，零样本迁移 |
| 长链路测试集 | 50 trials/方法 | 5 步序列，抽屉+导航+放置 | 系统测试 |
| 杂乱场景测试集 | 50 trials/方法 | 密集干扰物+强遮挡 | 鲁棒性测试 |
| 照片欺骗测试集 | — | 真实物体+照片级干扰 | 语义区分测试 |

### 实现细节

- **VLA Backbone**: DexVLA（冻结动作专家，仅更新最后 5 块的 adapter）
- **几何初始化**: [[VGGT]] Transformer（无需标定）
- **语义特征**: [[CLIP]] ViT-B/32 + [[DINOv2]] ViT-S/14
- **底盘定位 RL**: [[PPO]]，在仿真训练，零样本迁移
- **VLA 微调**: 每任务 10 次真实机器人演示，少样本
- **硬件**: 离线 RTX 4090（感知/规划）+ 板载 Jetson Orin NX（控制）

### 可视化结果

系统在照片欺骗测试中展示了强大的 3D 感知优势：单视角变体（无多视角聚合）假抓率 70%，说明单张图像无法区分照片和真实物体；多视角 3D 定位后假抓率降至 0%，真实成功率提升至 88%。

---

## 批判性思考

### 优点
1. **感知精度高**: 主动多视角 + 双路语义特征（CLIP+DINOv2）提供比点云更丰富的局部语义表示，对遮挡和干扰物鲁棒。
2. **底盘定位创新**: 明确建模高度自适应和可达性约束，填补了"定位目标→移动底盘"这一 VLA 论文常忽视的环节。
3. **注入位置分析**: 系统研究了 Transformer 注入层位置对性能-延迟的影响，结论清晰可复现。
4. **实验充分**: 多维度测试（长链路/高度/杂乱/欺骗），统计严谨（95% CI）。

### 局限性
1. **感知延迟较高**: 4 视角采集需 16 秒，不适合快速动态场景（论文明确声明面向 quasi-static household tasks）。
2. **视角数固定**: 固定采集 4 张图，无法根据场景复杂度自适应调整（杂乱场景可能需要更多视角）。
3. **任务范围有限**: 少样本设置每任务 10 次演示，泛化到全新任务类别（zero-shot skill acquisition）未验证。

### 潜在改进方向
1. 使用连续视频帧替代主动单步采集，实现在线 Semantic-3DGS 更新（无需 16 秒停顿）。
2. 自适应视角数量：场景简单时 1-2 张，遮挡严重时 6-8 张，优化效率-精度 tradeoff。
3. 将底盘定位 RL 扩展到移动中操作（不需要在固定位置停下来感知）。

### 可复现性评估
- [ ] 代码开源（未找到公开链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（Base-RL/VLA 微调细节均有描述）
- [ ] 数据集可获取（真实机器人演示数据未开放）

---

## 关联笔记

### 基于
- [[DexVLA]]: VLA backbone，冻结动作专家用于后期块条件化
- [[VGGT]]: 少视角 3D 几何初始化（相机姿态+深度估计）
- [[Semantic-3DGS]]: 语义高斯场表示，CLIP/DINOv2 特征蒸馏
- [[PPO]]: 底盘定位 RL 策略的训练算法

### 对比
- [[PointVLA]]: 全局点云 VLA，作为主要对比基线
- [[DexVLA]]: 原始 VLA backbone（无 3D 语义增强），作为 naive baseline

### 方法相关
- [[CLIP]]: 语言 Grounding 的核心视觉-语言对齐编码器
- [[DINOv2]]: 提供细粒度外观特征补充 CLIP
- [[扩散策略]]: 动作专家的基础框架（diffusion-based action prediction）
- [[主动视角选择]]: 本文提出的信息量最大化视角选择方法
- [[可达性感知底盘定位]]: 本文提出的可达性感知底盘定位模块
- [[动作分块]]: VLA 输出动作块（action chunking）的推理范式

### 硬件/数据相关
- [[Unitree Go2]]: 四足机器人平台（Edu 版本 + 6-DoF 机械臂）

---

## 速查卡片

> [!summary] Embodied Multimodal Grounding (EmbodiedMG)
> - **核心**: 主动多视角 Semantic-3DGS + 可达性底盘定位 + 后期块 VLA 条件化，用于开放词汇移动操作
> - **方法**: VGGT 几何初始化 → CLIP/DINOv2 语义蒸馏 → 语言 Grounding → PPO 底盘定位 → 后期块注入 VLA
> - **结果**: 长链路 60%（vs PointVLA 40%），杂乱 74%（vs PointVLA 46%），假抓率 0%（vs DexVLA 76%）
> - **代码**: 暂无开源

---

*笔记创建时间: 2026-08-13*

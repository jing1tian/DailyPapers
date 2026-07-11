---
title: "EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data"
method_name: "EgoWAM"
authors: [Baoyu Li, Xinchen Yin, Mengying Lin, Yixin Zhang, Danfei Xu]
year: 2026
venue: arXiv
tags: [world-action-model, cross-embodiment, egocentric-data, diffusion-policy, robot-manipulation]
zotero_collection: 3-Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2607.08436
created: 2026-07-11
---

# 论文笔记：EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Georgia Institute of Technology |
| 日期 | July 2026 |
| 项目主页 | [gatech-rl2.github.io/egowam.github.io](https://gatech-rl2.github.io/egowam.github.io) |
| 对比基线 | [[BC\|Behavior Cloning Co-training]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.08436) / Code: Coming Soon |

---

## 一句话总结

> EgoWAM 通过预测未来场景动态（而非仅模仿动作）来从野外第一视角人类视频中高效迁移知识，实验证明 DINO 表征提升 OOD 泛化达 4×，3D 运动流提升域内精度达 30%。

---

## 核心贡献

1. **WAM vs. BC co-training 对比**: 系统证明 [[World Action Model|World Action Model (WAM)]] 联合训练在人类与机器人动作分布不一致时仍能持续受益，而 [[BC|Behavior Cloning]] 在相同条件下出现负迁移
2. **三种世界表征系统比较**: 在严格控制骨干网络与动作头不变的条件下，对比 Pixel VAE、DINO 特征、3D 运动流三种世界预测目标的迁移效果
3. **三个设计准则 (D1/D2/D3)**: 提出用于评估世界表征适用性的三个准则——外观抽象性、跨体态一致性、自我运动解耦

---

## 问题背景

### 要解决的问题

机器人操作数据稀缺，野外第一视角人类视频（[[EgoVerse]]）数量庞大，但直接用行为克隆对人类演示进行联合训练往往因体态差异（手型、视角、运动模式不同）导致负迁移。

### 现有方法的局限

- **BC co-training**: 强迫共享 encoder 和 action head 对人类数据学习可执行动作，但人类抓杯、折叠等动作无法映射到机械臂，造成负迁移
- **像素级世界模型**: 重建像素同时编码体态外观信息（皮肤颜色、手型），违背跨体态一致性
- **图像坐标预测**: 人类头部运动产生的自我运动 (ego-motion) 会混淆真正的场景变化，机器人静态摄像头无此问题

### 本文的动机

世界模型预测的是物理效果（物体如何移动），而非体态外观，因此更适合作为跨体态迁移的监督信号。关键假设：即使人类动作无法直接执行，人类操作引发的场景动态与机器人操作高度相似。

---

## 方法详解

### 模型架构

EgoWAM 采用 **[[HPT|Heterogeneous Pretrained Transformer (HPT)]]** 骨干架构：

- **输入**: 自我视角图像 $o^{ego}$ + 腕部图像 $o^{wrist}$（仅机器人）+ 本体感知 $q$
- **Stems**: 模态独立编码器（ResNet-18 视觉茎 + MLP 本体感知茎）
- **Trunk**: 共享 Transformer（dim=256，16 层，8 头），输出共享潜变量 $z_t = f_\phi(o_t)$
- **Action Head**: [[CFM|条件流匹配]] CrossTransformer（6 blocks，dim 128，4 heads），输出 14-D 动作块 $a_{t:t+k}$，[[Action Chunking|动作块长度]] $k=100$
- **World-Model Head**（可插拔）：三种变体（Pixel VAE / DINO / 3D Flow），推理时移除
- **总参数**: HPT trunk 约 16 层，world head 按变体不同

```
输入观测 o_t
    ├── Ego Vision Stem (ResNet-18 / DINO)
    ├── Wrist Vision Stem (ResNet-18, 仅机器人)
    └── Proprioception Stem (MLP: 14→256)
           ↓ concat
    Shared HPT Trunk (16L × 256d)
           ↓ z_t
    ├── Action Head → a_{t:t+k}  (推理时使用)
    └── World-Model Head → s_{t+T}  (训练时使用，推理时移除)
```

### 核心模块

#### 模块 1: World-Model Head — Pixel VAE

**设计动机**: 最直观的未来状态表征，作为表达能力上限的基线

**具体实现**:
- 使用 Wan VAE 预训练编码器将 RGB 帧压缩到潜空间（latent: $16 \times 16 \times 16$，分辨率 128px）
- [[DiT]] 解码器（6 blocks，dim 384，6 heads）从零训练，以 $z_t$ 为条件去噪
- 预测目标：未来帧的 VAE 潜变量

**局限**: 同时编码外观（体态、肤色）与运动，违反 D1（外观抽象）和 D2（跨体态一致）

#### 模块 2: World-Model Head — DINO Features

**设计动机**: 利用 [[DINOv2]] 的语义先验，提取体态无关的场景语义特征

**具体实现**:
- 冻结 DINOv2-B 提取未来帧的 patch 特征（去掉 [CLS] 和 registers，加 layer norm）
- DDT-style 宽解码器（Wide DiT）以 $z_t$ 为条件预测 DINO 特征
- 特征维度与空间分辨率与 DINOv2-B patch grid 对齐

**优势**: 对 OOD 物体/场景泛化性最强（+4×），满足 D1、D2，但仍以图像坐标为索引（轻微违反 D3）

#### 模块 3: World-Model Head — 3D Motion Flow

**设计动机**: 直接预测三维空间中的物理位移，解耦自我运动，满足全部三个准则

**具体实现**:
- 用 SpatialTrackerV2 + Project Aria VIO 位姿计算相机稳定化 3D 流 $F_{[t,t+T]}$
- 注意力解码器（交替 self/cross-attention）以 $z_t$ 和查询点 $q$ 为条件预测流向量
- 查询点从自我帧均匀采样，输出 3D 位移向量

**优势**: 不受视角对齐约束，在错位人类数据下依然鲁棒（对齐与否性能不变）

---

## 关键公式

### 公式 1: [[BC|BC Co-training 损失]]

$$
\mathcal{L}_{\text{BC-cotrain}}(\phi, \theta) = \sum_{\mathcal{D} \in \{\mathcal{D}_H, \mathcal{D}_R\}} \mathbb{E}_{(o,a) \sim \mathcal{D}} \mathcal{L}_{\text{BC}}\!\left(\pi_\theta(a \mid f_\phi(o)),\, a\right)
$$

**含义**: 对人类数据集 $\mathcal{D}_H$ 和机器人数据集 $\mathcal{D}_R$ 共同优化行为克隆损失，共享编码器 $f_\phi$ 和动作解码器 $\pi_\theta$

**符号说明**:
- $\phi$: 编码器参数
- $\theta$: 动作解码器参数
- $\mathcal{D}_H, \mathcal{D}_R$: 人类/机器人数据集
- $f_\phi(o)$: 共享 trunk 编码

---

### 公式 2: [[World Action Model|WAM 联合分布]] (Eq. 2)

$$
p_{\theta,\psi}(a_{t:t+k},\, s_{t+T} \mid o_t) = p_\psi(s_{t+T} \mid z_t)\, p_\theta(a_{t:t+k} \mid z_t),\quad z_t = f_\phi(o_t)
$$

**含义**: 给定共享潜变量 $z_t$，动作预测与世界状态预测条件独立，允许两路监督信号共享 trunk

**符号说明**:
- $\psi$: 世界模型头参数
- $s_{t+T}$: 未来时刻 $t+T$ 的场景状态（像素/DINO 特征/3D 流）
- $a_{t:t+k}$: 长度为 $k$ 的动作块
- $z_t = f_\phi(o_t)$: 共享 trunk 编码

---

### 公式 3: [[CFM|EgoWAM 训练总损失]] (Eq. 3)

$$
\mathcal{L}_{\text{EgoWAM}} = \underbrace{\left(\mathcal{L}^{\text{robot}}_{\text{action}} + \mathcal{L}^{\text{human}}_{\text{action}}\right)}_{\text{动作监督}} + \lambda \underbrace{\left(\mathcal{L}^{\text{robot}}_{\text{world}} + \mathcal{L}^{\text{human}}_{\text{world}}\right)}_{\text{世界模型监督}}, \quad \lambda = 1.0
$$

**含义**: 两路动作损失（流匹配）+ 两路世界预测损失并行优化，$\lambda=1$ 等权重

**符号说明**:
- $\mathcal{L}^{\cdot}_{\text{action}}$: [[CFM|条件流匹配]] 动作预测损失
- $\mathcal{L}^{\cdot}_{\text{world}}$: 世界模型预测损失（按变体不同：噪声预测/速度回归）
- $\lambda = 1.0$: 世界预测权重

---

### 公式 4: [[CFM|流匹配插值]]

$$
s^\tau = (1-\tau)\,\varepsilon + \tau\, s, \quad \tau \sim \text{Beta}(1.5, 1.0),\quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 在干净目标 $s$ 与高斯噪声 $\varepsilon$ 之间进行插值，得到训练用的噪声中间态 $s^\tau$，适用于动作头和世界模型头

**符号说明**:
- $s$: 干净目标（动作序列或世界状态）
- $\varepsilon$: 标准高斯噪声
- $\tau$: Beta 分布采样的插值系数

---

### 公式 5: 3D 流目标——[[Cross-Embodiment|自我运动解耦]]

$$
s = F_{[t,\,t+T]} = \tilde{X}_{t+T} - X_t, \quad \tilde{X}_{t+T} = (T^{\text{cam}}_t)^{-1}\, T^{\text{cam}}_{t+T}\, X_{t+T}
$$

**含义**: 将未来时刻点云位置反投影到当前相机坐标系，消除摄像头自身运动，得到纯场景物理位移

**符号说明**:
- $X_t, X_{t+T}$: 时刻 $t$ 和 $t+T$ 的世界坐标点
- $T^{\text{cam}}_t$: 时刻 $t$ 相机位姿（来自 VIO）
- $\tilde{X}_{t+T}$: 反投影后的未来点位置
- $F_{[t,t+T]}$: 相机稳定化 3D 运动流

---

### 公式 6: 3D 流预测损失

$$
\mathcal{L}^{\text{Flow}}_{\text{world}} = \mathbb{E}\,\bigl\lVert u_\psi(s^\tau,\, \tau,\, f_\phi(o),\, q) - (s_q - \varepsilon_q) \bigr\rVert^2
$$

**含义**: 训练速度预测网络 $u_\psi$ 回归查询点处的流向量，以 trunk 编码和查询点坐标为条件

**符号说明**:
- $u_\psi$: 速度预测注意力解码器
- $q$: 从自我帧均匀采样的查询点集合
- $s_q, \varepsilon_q$: 查询点处的目标流与噪声
- $s^\tau$: 噪声中间态（公式 4）

---

### 公式 7: 人类动作再表达（自我运动因子分解）

$$
a^H_{t:t+k} = \left[(T^{\text{device}}_t)^{-1}\, T^{\text{device}}_{t+i}\, p^H_{t+i}\right]_{i=1}^k
$$

**含义**: 将人类手部位姿序列在瞬时设备坐标系下重新表达，分解头部自我运动，使人类动作与机器人末端执行器动作可比

**符号说明**:
- $T^{\text{device}}_t$: 时刻 $t$ 设备（Project Aria 眼镜）位姿
- $p^H_{t+i}$: 时刻 $t+i$ 人类手部世界坐标位姿
- $a^H_{t:t+k}$: 设备坐标系下的人类相对动作序列

---

## 关键图表

### Figure 1: EgoWAM Co-training 概览

![[EgoWAM_fig1.png]]

**说明**: 展示实验室机器人数据与野外 [[EgoVerse]] 第一视角人类数据的联合训练框架，对比 [[BC|BC co-training]]（动作监督）与 WAM co-training（动作+世界预测监督）的本质区别。体态差距（人手 vs. 机械臂）在 BC 中造成负迁移，在 WAM 中通过世界模型监督绕过。

---

### Figure 2: EgoWAM 模型架构

![[EgoWAM_fig2.png]]

**说明**: [[HPT|HPT 骨干]] 的完整结构。左侧：模态独立的 stems（自我视觉、腕部视觉、本体感知）输入共享 [[Cross-Embodiment|cross-embodiment]] trunk；右侧：可替换的世界模型头（Pixel VAE / DINO / 3D Flow），推理时移除只保留 action head（30 Hz 部署）。

---

### Figure 3: 真实机器人实验主要结果

![[EgoWAM_fig3.png]]

**说明**: 三个双臂操作任务（Cup-on-Saucer / Fold-Clothes / Bag-Grocery）在域内（ID）和分布外（OOD）条件下的归一化成功率对比。WAM co-training 在所有条件下持续优于 BC；Pixel WAM 迁移最弱，DINO WAM 在 OOD 上最强（最高 4×），3D Flow WAM 在域内精度提升最大（+20–30%）。

---

### Figure 4: Trunk 嵌入 UMAP 可视化（Cup-on-Saucer）

![[EgoWAM_fig4.png]]

**说明**: BC co-training 下人类嵌入与机器人嵌入在 UMAP 空间中分离；WAM co-training（DINO/3D Flow）下两者对齐，说明世界模型监督促使共享 trunk 学习体态无关的场景表征，而非体态相关的动作模式。

---

### Figure 5: 真实机器人定性对比

![[EgoWAM_fig5.png]]

**说明**: 四种方法在三个任务上的执行轨迹。BC 在 Fold-Clothes 的 OOD 场景下完全失败（0%），Pixel WAM 有所改善，DINO WAM 和 3D Flow WAM 成功处理未见物体和新场景。

---

### Figure 6: 3D Flow 的空间精度提升

![[EgoWAM_fig6.png]]

**说明**: 在 Cup-on-Saucer 任务中标记机器人成功放置杯子的工作空间位置分布。BC 仅在中心区域成功；3D Flow WAM 覆盖更广的工作空间范围，体现 [[Cross-Embodiment|几何理解]] 带来的空间泛化能力。

---

### Figure 7: 世界预测质量对比

![[EgoWAM_fig7.png]]

**说明**: 三种世界模型头在联合训练后的预测质量。Pixel VAE 细节模糊；DINO head 捕捉语义结构；3D Flow head 精确预测物体运动轨迹。说明人类 co-training 同时提升了世界模型的预测精度。

---

### Figure 8: 错位人类数据消融实验

![[EgoWAM_fig8.png]]

**说明**: 故意使用行为错位的人类演示（如人类横抱杯子，机器人无法复现此动作）。BC 崩溃到低于仅机器人数据的基线；3D Flow WAM 保持不变（约 85%），证明世界模型监督天然过滤了无法执行的动作信息，只保留物理场景变化信号。

---

### Figure 9: 对齐数据消融实验

![[EgoWAM_fig9.png]]

**说明**: 演示者刻意模仿机器人行为（视角和动作对齐）时各方法的变化。BC 从负迁移转为正迁移，Pixel 从 35%→65%，DINO 从 50%→70%，而 3D Flow 在对齐/不对齐下均保持约 85%，体现 D3（自我运动解耦）的价值。

---

### Figure 10: 机器人平台与数据采集

![[EgoWAM_fig10.png]]

**说明**: 双臂机器人平台配置，使用 Meta Quest 3 遥操作采集 300–360 个机器人演示/任务；人类数据用 Project Aria 眼镜采集（[[EgoVerse]] 数据集），约 10:1 人机数据比例。

---

### Figure 11: 人类数据模态消融（Cup-on-Saucer）

![[EgoWAM_fig11.png]]

**说明**: 对比三种人类数据使用方式：仅动作标签（BC）、仅 3D Flow、动作+3D Flow（完整 EgoWAM）。"世界模型监督单独使用优于动作监督单独使用"；两者结合（EgoWAM）最优，证明两路信号互补。

---

### Figure 12: Bag-Grocery 失败分析——视频预测

![[EgoWAM_fig12.png]]

**说明**: Pixel WAM 在 Bag-Grocery 任务中出现像素幻觉（预测到不存在的物体），联合人类 co-training 后幻觉减少，预测更准确。揭示了世界模型预测质量与策略执行成功率之间的正相关关系。

---

### Figure 13: 域内与 OOD 测试对象（附录）

![[EgoWAM_fig13.png]]

**说明**: 展示三个操作任务中用于域内评估和分布外（OOD）评估的具体物体集合。OOD 测试分两类：未见物体置于训练场景（OOD Object，10 次滚出）、训练物体置于新场景（OOD Scene，10 次滚出）。体现实验对 OOD 泛化评估的精心设计。

---

### Table 1: 3D 流世界模型预测损失

| 流数据流 | 仅 Flow 训练 | 动作 + Flow 联合 |
|----------|-------------|-----------------|
| Human stream | 0.23 | **0.22** |
| Robot stream | 0.20 | **0.19** |

**关键发现**: 加入动作标签作为 context 后，两路流的预测损失均下降，证明"动作标签作为世界模型预测的辅助信息"这一设计是有效的。

---

### Table 2: 跨变体固定超参数

| 组件 | 参数 | 值 |
|------|------|-----|
| Trunk | Embedding dim | 256 |
| Trunk | Layers / Heads | 16 / 8 |
| Trunk | Stochastic depth | 0.1 |
| Vision Stem | 架构 | ResNet-18 (dim 256) |
| Proprioception | 维度 | 14 → 256 MLP |
| Action Head | Blocks / dim / heads | 6 / 128 / 4 |
| Action Head | 动作维度 | 14-D（6-DoF × 2 臂 + gripper） |
| Action Head | Chunk 长度 | 100 frames |
| Action Head | 采样步数 | 50 (v-prediction) |
| Optimizer | AdamW | lr=1e-4, wd=1e-4 |
| Scheduler | Cosine | T_max=1400, η_min=1e-5 |
| Batch size | Robot + Human | 32 + 32 |
| 训练 | Steps | 100/epoch × 2000 epochs |
| 精度 | | bf16 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 机器人演示 | 300–360 次/任务 | Meta Quest 3 遥操作，随机化位置/朝向 | 训练（动作+世界） |
| In-Domain 人类 | 1:1 与机器人 | 相同场景/物体，视角和行为不对齐 | 人类 co-training |
| [[EgoVerse]] | ~10:1 与机器人 | 多样场景/物体/演示者，无场景对齐 | 大规模野外 co-training |

**评估**: 1800 次真实机器人滚出（每任务 × 域内 20 + OOD 20 × 多种方法）

### 任务设置

| 任务 | 描述 | 难点 |
|------|------|------|
| Cup-on-Saucer | 精确将杯子放到碟子上，随机化位姿 | 精确抓取+放置 |
| Fold-Clothes | 三折叠衬衫，随机初始状态 | 软体操作 |
| Bag-Grocery | 打开袋子并装入随机物品 | 多步骤、物体多样 |

### 实现细节

- **Backbone**: HPT（Heterogeneous Pretrained Transformer）
- **优化器**: AdamW（lr=1×10⁻⁴，weight decay=1×10⁻⁴）
- **调度器**: Cosine annealing（$T_{max}=1400$，$\eta_{min}=1 \times 10^{-5}$）
- **Batch Size**: 32（机器人）+ 32（人类）
- **训练**: 每 epoch 100 步，最多 2000 epochs
- **精度**: bf16
- **动作空间**: 14-D（双臂各 6-DoF SE(3) + 夹爪）
- **速度对齐**: 机器人窗口 $T_R=1.5$s，人类窗口 $T_H=1.0$s；工作空间用分位数映射归一化至 $[-1,1]$
- **部署频率**: 30 Hz（推理时移除 world head）
- **数据增强**: Color jitter $(0.1, 0.1, 0.1, 0.05)$

### 可视化结果

UMAP 分析（Figure 4）揭示 WAM co-training 驱动人类/机器人嵌入对齐，BC 则产生分离的聚类。世界预测质量（Figure 7）与策略性能正相关，为诊断提供工具。

---

## 批判性思考

### 优点

1. **控制变量设计严谨**: 固定骨干和动作头，仅替换世界模型头，排除其他因素干扰，实验结论可信
2. **3D Flow 的体态无关性**: 数学上消除自我运动，天然处理人机视角差异，无需手动对齐数据
3. **1800 次真实机器人滚出**: 大规模真实验证，说服力强；UMAP 可视化提供机制解释

### 局限性

1. **动作迁移仍是瓶颈**: 世界模型提升的是上下文理解，但从人类视频学习新运动原语（如抓握技巧）尚未实现
2. **每任务独立训练**: 未验证多任务扩展能力，与 VLA 范式的差距
3. **3D 追踪依赖**: 3D Flow 需要 SpatialTrackerV2 + VIO 位姿，对部署环境有要求
4. **哪种表征在真正开放世界最优**: 三种方法各有优劣，没有一个明确的赢家

### 潜在改进方向

1. 探索语义 + 几何的融合表征（DINO + 3D Flow 混合）
2. 扩展到多任务 VLA 框架，验证跨任务迁移效果
3. 研究世界模型监督量（$\lambda$ 值）的自适应调节

### 可复现性评估

- [ ] 代码开源（"Coming Soon"）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、数据比例均有说明）
- [x] 数据集可获取（EgoVerse 数据集在 partners.mecka.ai/egoverse）

---

## 关联笔记

### 基于

- [[BC|Behavior Cloning]]: EgoWAM 扩展 BC co-training，用世界模型监督替代纯动作监督
- [[World Action Model|WAM]]: EgoWAM 是 WAM 框架在野外人类数据上的系统化应用
- [[HPT|HPT]]: 使用 Heterogeneous Pretrained Transformer 作为统一骨干
- [[DINOv2]]: DINO 变体的世界表征目标来自冻结的 DINOv2-B

### 对比

- [[FAWAM]]: 同为 WAM 类框架，FAWAM 聚焦于 Foundation Model 集成
- [[AdaWAM]]: 自适应 WAM，EgoWAM 关注跨体态数据来源
- [[Flash-WAM]]: 快速 WAM 推理，EgoWAM 关注训练数据来源

### 方法相关

- [[CFM|条件流匹配]]: 动作头和世界模型头的训练目标
- [[Action Chunking]]: 动作块长度 $k=100$，30 Hz 部署
- [[Cross-Embodiment]]: 核心动机，解决人机体态差异
- [[DiT]]: Pixel VAE 和 DINO 变体的解码器架构

### 硬件/数据相关

- [[EgoVerse]]: 野外第一视角人类数据集，主要人类 co-training 来源
- [[HPT|HPT Backbone]]: Heterogeneous Pretrained Transformer 统一多模态处理

---

## 速查卡片

> [!summary] EgoWAM
> - **核心**: 用世界动态预测（非动作模仿）从野外人类第一视角视频迁移知识到机器人操作
> - **方法**: HPT backbone + 可插拔世界模型头（Pixel VAE / DINO / 3D Flow），联合训练 $\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{world}}$
> - **结果**: DINO OOD +4×，3D Flow 域内 +20–30%；3D Flow 对数据对齐鲁棒
> - **代码**: Coming Soon — gatech-rl2.github.io/egowam.github.io

---

*笔记创建时间: 2026-07-11*

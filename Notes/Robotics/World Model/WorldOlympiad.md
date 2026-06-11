---
title: "WorldOlympiad: Can Your World Model Survive a Triathlon?"
method_name: "WorldOlympiad"
authors: [Yuke Zhao, Wangbo Zhao, Weijie Wang, Zeyu Zhang, Dakai An, Akide Liu, Yinghao Yu, Jiasheng Tang, Fan Wang, Wei Wang, Bohan Zhuang]
year: 2026
venue: arXiv
tags: [world-model, benchmark, video-generation, physical-reasoning, 3d-consistency, robotics]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.11129
created: 2026-06-11
---

# 论文笔记：WorldOlympiad: Can Your World Model Survive a Triathlon?

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University, DAMO Academy / Alibaba Group, HKUST, Monash University, TRE / Alibaba Group |
| 日期 | June 2025 |
| 项目主页 | [alibaba-damo-academy.github.io/WorldOlympiad](https://alibaba-damo-academy.github.io/WorldOlympiad) |
| 对比基线 | [[VBench]], [[WorldArena]], [[EWMBench]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.11129) / [Code](https://github.com/alibaba-damo-academy/WorldOlympiad) |

---

## 一句话总结

> WorldOlympiad 是首个横跨物理一致性、几何一致性和交互保真度三维评估视频世界模型的综合基准，覆盖游戏、机器人和真实世界三个领域的 1,000 段视频。

---

## 核心贡献

1. **统一多维度评估框架**: 首次将[[世界模型 (World Model)|世界模型]]评估分解为物理正确性、3D 几何一致性和交互保真度三个互补维度，覆盖游戏、机器人和真实场景。
2. **可解释的自动化评估指标**: 利用多模态 LLM 评判器、[[Gaussian Splatting|高斯溅射]]重建和 [[CLIP]] 嵌入相似度构建多层级自动化评估流水线，与人类偏好 Spearman ρ=0.95 高度相关。
3. **大规模实证分析**: 系统评估了 8 个最先进的长视频生成流水线，揭示了当前世界模型在几何一致性和长时序交互方面的显著差距。

---

## 问题背景

### 要解决的问题

现有[[视频生成 (Video Generation)|视频生成]]模型评估集中于帧级视觉质量（如 FID、FVD），无法衡量模型是否真正理解物理规律、三维空间结构以及长时序动作控制能力——这些是将[[世界模型 (World Model)|世界模型]]用于机器人规划和游戏仿真的关键要求。

### 现有方法的局限

- **VBench / VBench++**: 仅评估视觉质量，不涵盖物理规律或几何一致性
- **EWMBench / WorldEval**: 仅关注机器人单一领域或单一维度的交互
- **WorldArena**: 虽覆盖物理和几何，但仅针对机器人场景，不支持长视频评估
- 现有基准均为单领域、单维度，无法全面刻画世界模型能力边界

### 本文的动机

真正的[[世界模型 (World Model)|世界模型]]需要通过"三项全能"测试：物理规律遵守（重力、相变、材料属性）、跨帧几何一致性（可重建三维空间）以及长时序可控交互（持续响应动作指令）。三个维度缺一不可，且应在多样化领域中联合评估。

---

## 方法详解

### 评估框架总体架构

WorldOlympiad 采用**三轨道评估 + 统一数据流水线**架构：

- **输入**: 参考视频 $V_{ref}$ + 动作-字幕注释序列 $\{(a_i, p_i)\}_{i=1}^T$ + 起始帧 $f_0$
- **生成**: 被评估的[[世界模型 (World Model)|世界模型]]根据起始帧和动作序列生成长视频 $V_{gen}$
- **三轨道评分**: 物理轨（$S_{phys}$）、几何轨（$S_{3D}$）、交互轨（$S_{interact}$）
- **输出**: 综合评分 $S_{all}$

### 数据收集流水线

数据涵盖三个领域，共 1,000 段高质量视频：

| 领域 | 数量 | 来源 | 选取规则 |
|------|------|------|---------|
| 机器人 | 400 | RoboCOIN | 下载并人工筛选双手操作视频 |
| 游戏 | 400 | GameGen-X | 从 OGameData_50K.csv 随机采样，长视频分段 |
| 真实世界 | 200 | LVD-2M | 时长 >60 秒 且运动得分 >50 |

**三阶段注释流水线**：

1. **Stage I — Chunking**: 将视频切分为最多 6 段连续时间片段
2. **Stage II — Captioning**: 使用 Gemini-3-Pro-Preview 为每段生成动作标签和字幕
3. **Stage III — Refinement**: 基于完整视频上下文验证字幕一致性

### 核心模块一：物理正确性评估（Physical Track）

**设计动机**: 利用[[多模态大语言模型 (MLLM)]]的物理常识判断生成视频是否违反基本物理规律。

**具体实现**:
- 使用 [[SAM|SAM3]] 对参考视频进行物理现象相关的对象分割预处理
- **双评判器架构**：
  - **Relevance Judge（相关性评判器）**: 判断物理规则是否适用于参考视频
  - **Compliance Judge（合规评判器）**: 对比生成视频与参考视频，评分物理行为符合程度

评估规则涵盖三类共 14 条：

**力学规则（4 条）**:
- 重力：无支撑物体应向下坠落或抛物运动
- 浮力：漂浮物保持在液面附近，密实物下沉
- 压缩：固体在荷载下合理形变，软材料平滑压缩，刚性体保持形状
- 碰撞：碰撞产生合理的动量传递、弹跳或静止

**热力学规则（6 条）**:
- 熔化、升华、汽化、凝结、凝华、凝固

**材料属性规则（4 条）**:
- 颜色混合、溶解性、硬度、可燃性

### 核心模块二：几何一致性评估（Geometry Track）

**设计动机**: 通过[[Gaussian Splatting|高斯溅射]]三维重建，验证生成视频是否对应内在一致的三维空间。

**具体实现**:
1. 用动态前景掩码 + 视频修复（video inpainting）隔离静态场景几何
2. 调用 [[Depth Anything|Depth Anything 3]] ($\mathcal{F}_{DA3}$) 从遮掩后的视频 $\bar{V}$ 重建高斯表示 $\mathcal{G}$ 及相机外参/内参 $\{E_i, K_i\}$
3. 渲染重建视频 $\hat{V}_{GS}$ 和诊断元视角图像 $\hat{I}_{meta}$（取最远相机位姿）
4. 融合三个互补信号

### 核心模块三：交互保真度评估（Interaction Track）

**设计动机**: 评估长视频生成中，世界模型是否持续响应动作提示并保持全局语义一致性。

**具体实现**（三层级评分）:
- **Chunk 级（语义贴合度）**: 使用 [[CLIP]] 嵌入相似度衡量生成内容与动作字幕对的匹配
- **Transition 级（段间平滑度）**: 结构化 MLLM 评分相邻视频段边界的平滑程度
- **Global 级（全局一致性）**: 综合长程状态保持和整体语义对齐

---

## 关键公式

### 公式1: [[Gaussian Splatting|3D 重建流程]]

$$
\begin{aligned}
\mathcal{F}_{DA3}(\bar{V}) &\to (\mathcal{G}, \{E_i, K_i\}) \\
\hat{V}_{GS} &= \mathcal{R}(\mathcal{G}, \{E_i, K_i\}) \\
\hat{I}_{meta} &= \mathcal{R}(\mathcal{G}, E_i^*, K_i^*)
\end{aligned}
$$

**含义**: 从掩掉动态区域的视频 $\bar{V}$ 重建高斯场，再渲染出重建视频和诊断元视角。

**符号说明**:
- $\mathcal{F}_{DA3}$: Depth Anything 3 重建函数
- $\mathcal{G}$: 高斯场表示
- $\{E_i, K_i\}$: 各帧相机外参与内参
- $\mathcal{R}$: 高斯渲染函数
- $E_i^*, K_i^*$: 距离最远的相机位姿（用于元视角）

### 公式2: [[SSIM|重建质量得分]]

$$
\begin{aligned}
S_{recon} &= \operatorname{clamp}(J_{vid}(\hat{V}_{GS}, p),\ 0, 1) \\
S_{meta} &= \operatorname{clamp}(J_{img}(\hat{I}_{meta}, p),\ 0, 1)
\end{aligned}
$$

**含义**: 用 MLLM 视频/图像评判器对重建质量和元视角质量评分并限幅。

**符号说明**:
- $J_{vid}$: MLLM 视频质量评判器
- $J_{img}$: MLLM 图像质量评判器
- $p$: 评估提示词

### 公式3: [[相机轨迹|相机轨迹归一化]]

$$
\tilde{T}_i = T_1^{-1} T_i, \quad \hat{\tilde{T}}_i = \hat{T}_1^{-1} \hat{T}_i
$$

**含义**: 将参考轨迹和预测轨迹都归一化到以第一帧为参考坐标系，消除全局位姿偏差。

**符号说明**:
- $T_i$: 参考视频第 $i$ 帧的相机位姿
- $\hat{T}_i$: 生成视频第 $i$ 帧的相机位姿
- $\tilde{T}_i$: 归一化参考轨迹
- $\hat{\tilde{T}}_i$: 归一化预测轨迹

### 公式4: [[相机轨迹|轨迹一致性得分]]

$$
S_{traj} = A_{motion}(S_t, S_r;\ \{\tilde{T}_i\})
$$

**含义**: 综合平移得分 $S_t$ 和旋转得分 $S_r$，由运动感知聚合函数加权计算最终轨迹一致性。

**符号说明**:
- $A_{motion}$: 运动感知聚合函数
- $S_t$: 相机平移一致性得分
- $S_r$: 相机旋转一致性得分

### 公式5: [[世界模型 (World Model)|3D 一致性综合得分]]

$$
S_{3D} = \frac{1}{3}(S_{recon} + S_{meta} + S_{traj})
$$

**含义**: 三个互补信号等权平均，得到最终几何一致性得分。

**符号说明**:
- $S_{recon}$: 重建质量得分
- $S_{meta}$: 元视角质量得分
- $S_{traj}$: 轨迹一致性得分

### 公式6: [[CLIP|Chunk 级 CLIP 语义得分]]

$$
s_i^{clip} = \frac{1}{m_i} \sum_j \operatorname{sim}(\operatorname{CLIP}_v(f_{i,j}),\ \operatorname{CLIP}_t(p_i))
$$

**含义**: 对第 $i$ 个时间片段内所有帧，计算视觉嵌入与文本嵌入的余弦相似度均值。

**符号说明**:
- $m_i$: 第 $i$ 个片段的帧数
- $f_{i,j}$: 第 $i$ 片段第 $j$ 帧
- $p_i$: 第 $i$ 片段的动作字幕提示
- $\operatorname{sim}(\cdot, \cdot)$: 余弦相似度

### 公式7: [[CLIP|视频级 CLIP 得分]]

$$
S_{clip} = \frac{\sum_i \sum_j \operatorname{sim}(\operatorname{CLIP}_v(f_{i,j}),\ \operatorname{CLIP}_t(p_i))}{\sum_i m_i}
$$

**含义**: 所有片段帧数加权平均的 CLIP 语义相似度，得到全视频语义贴合度。

### 公式8: [[CLIP|校准 CLIP 得分]]

$$
\tilde{S}_{clip} = \operatorname{clip}\!\left(\frac{S_{clip} - \tau_{min}}{\tau_{max} - \tau_{min}},\ 0, 1\right), \quad \tau_{min}=0.20,\ \tau_{max}=0.40
$$

**含义**: 将原始 CLIP 得分（范围 [-1,1]）用固定阈值重新校准到 [0,1]，确保跨模型比较的稳定性。

**符号说明**:
- $\tau_{min}=0.20$: 下限阈值（以下映射为0）
- $\tau_{max}=0.40$: 上限阈值（以上映射为1）

### 公式9: [[多模态大语言模型 (MLLM)|MLLM 各层级交互得分]]

$$
S_{chunk} = \frac{1}{5T}\sum_i a_i, \quad S_{trans} = \frac{1}{5(T-1)}\sum_i b_i, \quad S_{global} = \frac{g}{5}
$$

**含义**: 分别计算片段语义得分、段间转换平滑度得分和全局一致性得分，均归一化到 [0,1]。

**符号说明**:
- $T$: 视频片段总数
- $a_i$: 第 $i$ 片段的 MLLM 语义评分（0-5分）
- $b_i$: 第 $i$ 个片段转换的平滑度评分（0-5分）
- $g$: 全局一致性评分（0-5分）

### 公式10: [[多模态大语言模型 (MLLM)|MLLM 综合交互得分]]

$$
S_{mllm} = \frac{1}{3}(S_{chunk} + S_{trans} + S_{global})
$$

**含义**: 三个层级评分等权平均，得到 MLLM 维度的交互保真度。

### 公式11: [[世界模型 (World Model)|最终交互得分]]

$$
S_{interact} = (1 - \lambda) S_{mllm} + \lambda \tilde{S}_{clip}, \quad \lambda = 0.1
$$

**含义**: 以 MLLM 评分为主（权重 0.9），CLIP 校准分为辅（权重 0.1）融合得到最终交互得分。

**符号说明**:
- $\lambda = 0.1$: CLIP 分量权重
- $S_{mllm}$: MLLM 综合交互得分
- $\tilde{S}_{clip}$: 校准后的 CLIP 得分

### 公式12: [[世界模型 (World Model)|总体综合得分]]

$$
S_{all} = \frac{1}{3}(S_{phys} + S_{3D} + S_{interact})
$$

**含义**: 物理、几何、交互三轨道得分等权平均，得到世界模型的综合评分。

---

## 关键图表

### Figure 1: WorldOlympiad 流水线总览

![Figure 1](https://arxiv.org/html/2606.11129v1/x1.png)

**说明**: WorldOlympiad 的完整流水线，从数据收集（游戏/机器人/真实世界三个领域）、长视频生成（8 个被评估模型），到三轨道多维度评估框架的整体概览。

### Figure 2: 数据收集总览

![Figure 2](https://arxiv.org/html/2606.11129v1/x2.png)

**说明**: 三个领域的数据来源展示，包含机器人双手操作、游戏交互环境和真实世界开放场景的代表性帧。

### Figure 3: 数据标准化流水线

![Figure 3](https://arxiv.org/html/2606.11129v1/x3.png)

**说明**: 从原始视频到精化动作-字幕注释的三阶段处理流程：视频分段（Chunking）→ Gemini-3-Pro-Preview 字幕生成（Captioning）→ 全视频上下文验证（Refinement）。

### Figure 4: 流水线统计数据

![Figure 4](https://arxiv.org/html/2606.11129v1/x4.png)

**说明**: 数据处理各阶段的统计信息，包括注释覆盖率、评估就绪样本数量等分布情况。

### Figure 5: 评估结果统计

![Figure 5](https://arxiv.org/html/2606.11129v1/x5.png)

**说明**: 8 个被评估世界模型在三个评估维度（物理/几何/交互）上的得分分布，揭示所有模型在几何一致性上普遍偏低（最高仅 0.424）。

### Figure 6: 典型案例分析

![Figure 6](https://arxiv.org/html/2606.11129v1/x6.png)

**说明**: 基准检测出的代表性案例，包括高质量生成（保持物理行为）和典型失败案例（规则违反），如物体违背重力漂浮、瞬间状态突变等。

### Table 1: 基准比较（与现有基准对比）

| Benchmark | 长视频 | 物理 | 几何 | 交互 | 游戏 | 机器人 | 真实世界 |
|-----------|--------|------|------|------|------|--------|---------|
| VBench | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| VBench++ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| VBench 2.0 | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| MIND | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| EWMBench | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| WorldEval | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| WorldArena | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| **WorldOlympiad** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**: WorldOlympiad 是唯一同时覆盖所有七个评估维度的基准，是目前最全面的世界模型评估框架。

### Table 2: 数据构成

| 领域 | 数量 | 来源 | 选取规则 |
|------|------|------|---------|
| 机器人 | 400 | RoboCOIN | 下载视频，人工筛选 |
| 游戏 | 400 | GameGen-X | 随机采样，长视频分段 |
| 真实世界 | 200 | LVD-2M | 时长 >60s，运动得分 >50 |

**说明**: 三个领域合计 1,000 段视频，覆盖多样化的物理现象和交互场景。

### Table 3: 主评测结果

| 类别 | 模型 | 物理 | 3D一致 | 交互 | 综合 | 排名 |
|------|------|------|--------|------|------|------|
| 游戏 | Matrix-Game 2.0 | 0.325 | 0.255 | 0.113 | 0.231 | 8 |
| 游戏 | **LingBot-World** | **0.942** | 0.373 | **0.734** | **0.683** | **1** |
| 机器人 | **Cosmos-Predict-2.5** | 0.906 | 0.399 | 0.707 | 0.671 | 2 |
| 机器人 | WoW | 0.708 | 0.250 | 0.345 | 0.434 | 7 |
| 通用 | Rolling Forcing | 0.873 | 0.321 | 0.636 | 0.610 | 3 |
| 通用 | LongLive | 0.863 | 0.363 | 0.526 | 0.584 | 5 |
| 通用 | Yume-1.5 | 0.863 | 0.301 | 0.649 | 0.604 | 4 |
| 通用 | Hunyuan-WorldPlay | 0.692 | **0.424** | 0.316 | 0.477 | 6 |

**关键发现**:
- 物理规律已成为多模型共同能力（多个模型 >0.85），但几何一致性仍是最大短板（最高仅 0.424）
- 领域专一化训练可支持跨领域迁移（如 LingBot-World 和 Cosmos-Predict-2.5）
- 窄领域专家如 WoW 跨领域泛化能力有限

### Table 4: 人类偏好对齐验证

| 类别 | 模型 | 人类得分 | 自动得分 | 人类排名 | 自动排名 | 排名差 |
|------|------|---------|---------|---------|---------|--------|
| 游戏 | LingBot-World | 0.721 | 0.683 | 1 | 1 | 0 |
| 机器人 | Cosmos-Predict-2.5 | 0.648 | 0.671 | 2 | 2 | 0 |
| 通用 | Rolling Forcing | 0.579 | 0.610 | 3 | 3 | 0 |
| 通用 | LongLive | 0.532 | 0.584 | 4 | 5 | -1 |
| 通用 | Yume-1.5 | 0.491 | 0.604 | 5 | 4 | +1 |
| 通用 | Hunyuan-WorldPlay | 0.423 | 0.477 | 6 | 6 | 0 |
| 游戏 | Matrix-Game 2.0 | 0.309 | 0.231 | 7 | 8 | -1 |
| 机器人 | WoW | 0.271 | 0.434 | 8 | 7 | +1 |

**Spearman ρ = 0.95**，证明自动评估指标与人类判断高度一致。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboCOIN | 400 段 | 双手操作，含物体接触和状态变化 | 机器人轨道评估 |
| GameGen-X / OGameData_50K | 400 段 | 游戏环境，含相机运动和状态演化 | 游戏轨道评估 |
| LVD-2M | 200 段 | 开放域运动场景，长视频 | 真实世界轨道评估 |

### 实现细节

- **注释工具**: Gemini-3-Pro-Preview（字幕生成与验证）
- **物理分割**: SAM3（相关对象语义分割）
- **几何重建**: Depth Anything 3（深度估计 + 高斯溅射）
- **语义评估**: CLIP 视觉-文本嵌入相似度
- **CLIP 校准参数**: $\tau_{min}=0.20$，$\tau_{max}=0.40$
- **交互权重**: $\lambda=0.1$（CLIP 分量权重）

### 可视化结果

三类典型失败模式：
1. **物理违规**: 物体反重力漂浮、无接触形变、瞬间状态突变
2. **几何不一致**: 原始视角合理但三维重建/替代轨迹下失败
3. **交互漂移**: 局部字幕匹配但跨片段丢失物体身份或状态

---

## 批判性思考

### 优点

1. **覆盖全面**: 三维度七类型的正交评估组合，填补了现有基准的空白，尤其是 3D 几何与长视频交互的组合评估
2. **人类对齐高**: Spearman ρ=0.95 的人类偏好相关性，证明了自动指标的有效性
3. **可解释性强**: 物理规则分类明确（14 条），评判器给出置信度分数，失败模式可追溯

### 局限性

1. **几何评估依赖静态场景假设**: 高斯溅射重建需要掩掉动态前景，对高动态场景（如高速动作）鲁棒性存疑
2. **物理规则覆盖有限**: 仅覆盖力学、热力学和材料属性共 14 条规则，复杂交互物理（流体、弹性体）未涉及
3. **数据规模相对偏小**: 仅 1,000 段视频，可能无法充分表征评估方差

### 潜在改进方向

1. 扩展物理规则库，加入流体动力学、弹性体变形等复杂物理现象
2. 探索端到端可微分评估，允许将评估指标用于模型训练信号
3. 扩大数据规模并引入跨文化/跨物理环境的多样化场景

### 可复现性评估

- [x] 代码开源（GitHub）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（注释流水线详细描述）
- [x] 数据集可获取（来源于公开数据集）

---

## 关联笔记

### 基于

- [[VBench]]: 视频生成评估基准，WorldOlympiad 在其基础上扩展到物理和几何维度
- [[Gaussian Splatting]]: 几何轨道的核心重建技术
- [[CLIP]]: 交互轨道语义相似度计算基础

### 对比

- [[WorldArena]]: 覆盖物理+几何+交互但仅限机器人场景
- [[EWMBench]]: 机器人交互评估基准
- [[VBench]]: 通用视频质量评估

### 方法相关

- [[世界模型 (World Model)]]: 核心评估对象
- [[Depth Anything]]: 几何重建使用的深度估计模型
- [[SAM]]: 物理评估的对象分割预处理
- [[多模态大语言模型 (MLLM)]]: 物理合规性评判核心工具

### 硬件/数据相关

- [[RoboCOIN]]: 机器人操作数据来源
- [[GameGen-X]]: 游戏视频数据来源

---

## 速查卡片

> [!summary] WorldOlympiad: Can Your World Model Survive a Triathlon?
> - **核心**: 首个覆盖物理/几何/交互三维度、游戏/机器人/真实场景三领域的视频世界模型综合基准
> - **方法**: 物理规则 LLM 评判 + 高斯溅射三维重建 + 层级化 CLIP/MLLM 交互评分
> - **结果**: LingBot-World 综合最优(0.683)，几何一致性是所有模型最大短板(最高仅0.424)，人类偏好 ρ=0.95
> - **代码**: https://github.com/alibaba-damo-academy/WorldOlympiad

---

*笔记创建时间: 2026-06-11*

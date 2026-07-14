---
title: "CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding"
method_name: "CLAP"
authors: [Yuri Ishitoya, Jeremy Siburian, Masashi Hamaya, Kuniaki Saito, Cristian C. Beltran-Hernandez, Mai Nishimura]
year: 2026
venue: arXiv
tags: [vla, language-grounding, action-chunking, imitation-learning, robot-manipulation, vlm-adaptation, embodied-ai]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.08974v1
created: 2026-07-14
---

# 论文笔记：CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | OMRON SINIC X Corporation |
| 日期 | July 2026 |
| 项目主页 | [omron-sinicx.github.io/clap](https://omron-sinicx.github.io/clap/) |
| 对比基线 | [[VLA-0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.08974) / Code (待发布) |

---

## 一句话总结

> CLAP 通过在动作 token 前追加自然语言动作描述，消除 VLM 预训练分布与 VLA 输出分布之间的不匹配，无需任何架构改动即可将 2B VLM 在 LIBERO 上达到 90.8% 成功率。

---

## 核心贡献

1. **Language-Action Grounding 表示**: 在数值动作 token 序列前插入固定模板生成的自然语言描述（如 "move forward 3 cm, close gripper"），使 [[Autoregressive Policy|自回归策略]] 先生成语言计划再生成动作 token，输出分布与 VLM 预训练保持一致
2. **零架构修改的 VLM-to-VLA 适配**: 无需额外动作专家（action expert）、词表扩展或梯度阻断，直接微调 [[Qwen3-VL|Qwen3.5 VLM]]，保持与 VLM 生态系统（蒸馏、量化、新主干）的完全兼容
3. **可选 Action Masking 增强**: 训练时随机遮蔽数值动作 token，对大模型（4B）有显著 OOD 鲁棒性收益，但对小模型（0.8B/2B）略有负面影响

---

## 问题背景

### 要解决的问题

将预训练 [[VLA（视觉-语言-动作模型）|VLM]] 适配为 [[VLA（视觉-语言-动作模型）|VLA]] 时，输出需要从自然语言 token 切换为高精度数值动作 token（如 `408 980 166 ...`）。这些离散化的数值 token 与 VLM 预训练语料的语言分布截然不同，导致灾难性遗忘和迁移困难。

### 现有方法的局限

- **VLA-0（裸 token 方法）**: 直接输出数值动作 token，存在输出分布不匹配问题，LIBERO 平均仅 75-76%
- **VLM2VLA**: 完全用语言表示动作，放弃数值精度，仍需外部 action expert
- **LAP**: 需要预训练阶段的语言-动作对，依赖外部动作生成器
- **ECoT/TraceVLA**: 引入中间链式推理表示，需要额外的监督信号和标注开销

### 本文的动机

若在数值动作 token **之前**先生成一段语言动作描述，则模型的输出空间起点仍在语言分布内，数值 token 的预测由语言计划因果条件化（causally conditioned），从而缓解分布不匹配。语言描述由固定模板从 ground-truth 动作确定性生成，无需额外标注。

---

## 方法详解

### 模型架构

CLAP 采用 **自回归语言模型** 架构（基于 [[Qwen3-VL|Qwen3.5]] 系列），无任何结构修改：

- **输入**: 语言指令 $l$ + 视觉观测 $o_t$（RGB 图像）
- **Backbone**: [[Qwen3-VL|Qwen3.5-VL]] (0.8B / 2B / 4B 三个规模)
- **核心模块**: [[Language-Action Grounding]] 表示，将语言描述 $d$ 与数值动作 $\boldsymbol{a}$ 拼接为单一自回归序列
- **输出**: 语言动作描述 + [[Action Chunking|动作块]] $a_{t:t+h}$（$h=8$，7 DoF）
- **总参数**: 0.8B / 2B / 4B（全量微调）

### 核心模块

#### 模块 1: Language-Action Representation

**设计动机**: 利用 [[Language-Action Grounding]] 消解 [[Output-Distribution Mismatch|输出分布不匹配]]，同时保留数值动作的精度

**具体实现**:
- 固定模板 $T(\boldsymbol{a})$：对 7 DoF [[Action Chunking|动作块]] 逐轴聚合 $h$ 步 delta，转为自然语言描述
  - 平移：$\Delta x = \text{round}(100 \sum_{t=1}^h \delta x_t)$ [cm]
  - 旋转：$\Delta\varphi = \text{round}_{10}(\frac{180}{\pi} \sum_{t=1}^h \delta\varphi_t)$ [deg]
- 输出序列结构：`<think>[语言描述]</think>[数值 token×56]`（7 DoF × 8 步 × 1000 bins）
- 训练时：语言描述由 ground-truth 动作生成，无需人工标注
- 推理时：模型先生成语言计划，再自回归生成数值 token

#### 模块 2: Action Masking（可选增强）

**设计动机**: 通过随机遮蔽数值 token，强迫模型从语言描述中提取更多信息，增强 OOD 鲁棒性

**具体实现**:
- 以概率 0.4 将数值动作 token 替换为 `<mask>` token
- 被遮蔽的 token 不参与梯度计算
- 对 4B 模型提升 +3.2 pts（平均），对 0.8B/2B 分别 −3.9 / −1.7 pts

---

## 关键公式

### 公式 1: [[Autoregressive Policy|标准 VLA 自回归策略]]

$$
\pi_\theta(\boldsymbol{a} | o, l) = \prod_{i=1}^{7h} \pi_\theta(a^{(i)} | a^{(<i)}, o, l)
$$

**含义**: 基线方法（VLA-0）直接对 $7h$ 个数值动作 token 进行自回归预测，每个 token 条件化于历史动作、观测和指令。

**符号说明**:
- $\boldsymbol{a}$: 展平的动作 token 序列，维度 $7h$（7 DoF × $h$ 步）
- $a^{(<i)}$: 第 $i$ 个 token 之前的所有历史 token
- $o$: 视觉观测
- $l$: 语言指令
- $h$: 动作块长度（本文 $h=8$）

### 公式 2: [[Language-Action Grounding|CLAP 联合语言-动作策略]]

$$
\pi_\theta(d, \boldsymbol{a} | o, l) = \left[\prod_{j=1}^{|d|} \pi_\theta(d^{(j)} | d^{(<j)}, o, l)\right] \cdot \left[\prod_{i=1}^{7h} \pi_\theta(a^{(i)} | a^{(<i)}, d, o, l)\right]
$$

**含义**: CLAP 先自回归生成语言动作描述 $d$，再以 $d$ 为条件生成数值动作 token $\boldsymbol{a}$，两者在单一序列中完成。

**符号说明**:
- $d$: 自然语言动作描述（由固定模板 $T(\boldsymbol{a})$ 从 ground-truth 动作生成）
- $d^{(j)}$: 描述序列的第 $j$ 个 token，约 149 个 token
- $\boldsymbol{a}$: 数值动作 token（1000 bins/dim，7 DoF × 8 步 = 56 个 token）

### 公式 3: [[Language-Action Grounding|CLAP 训练目标]]

$$
\mathcal{L}_{\text{CLAP}} = -\mathbb{E}_{(o,l,\boldsymbol{a}) \sim \mathcal{D}} \left[\log \pi_\theta(\tilde{d} | o, l) + \log \pi_\theta(\tilde{\boldsymbol{a}} | o, l, \tilde{d})\right]
$$

**含义**: 对语言描述和数值动作序列共同施加交叉熵损失，$\tilde{d} = T(\boldsymbol{a})$ 由模板从 ground-truth 动作确定性生成。

**符号说明**:
- $\tilde{d} = T(\boldsymbol{a})$: 由固定模板从 ground-truth 动作块生成的语言描述
- $\tilde{\boldsymbol{a}}$: 离散化后的 ground-truth 动作 token
- $\mathcal{D}$: 示范数据集

### 公式 4: 平移聚合

$$
\Delta x = \text{round}\left(100 \sum_{t=1}^{h} \delta x_t\right) \text{ [cm]}
$$

**含义**: 将动作块内 $h$ 步的 x 轴位移 delta 求和，转换为厘米并四舍五入，用于生成语言描述中的距离描述。

**符号说明**:
- $\delta x_t$: 第 $t$ 步的 x 轴位移（米）
- $h$: 动作块长度（$h=8$）

### 公式 5: 旋转聚合

$$
\Delta\varphi = \text{round}_{10}\left(\frac{180}{\pi} \sum_{t=1}^{h} \delta\varphi_t\right) \text{ [deg]}
$$

**含义**: 将动作块内 $h$ 步的旋转 delta 求和，转换为度数并四舍五入到最近的 10°。

**符号说明**:
- $\delta\varphi_t$: 第 $t$ 步的旋转 delta（弧度）
- $\text{round}_{10}$: 四舍五入到最近 10°

---

## 关键图表

### Figure 1: CLAP 核心思路

![Figure 1](https://arxiv.org/html/2607.08974v1/x1.png)

**说明**: CLAP 将预训练 VLM 直接转换为可部署的 VLA，通过在数值动作 token 前追加自然语言动作描述，在不修改架构的前提下缓解 [[Output-Distribution Mismatch|输出分布不匹配]]。

### Figure 2: CLAP 方法概览

![Figure 2](https://arxiv.org/html/2607.08974v1/x2.png)

**说明**: CLAP 仅修改 VLA 微调的目标输出序列。训练时，语言动作描述由固定模板 $T(\boldsymbol{a})$ 从 ground-truth [[Action Chunking|动作块]] 生成。推理时，模型先生成语言计划（`<think>...</think>`），再生成离散数值动作 token。

### Figure 3: VLABench 能力画像（1-shot, 微调前）

![Figure 3](https://arxiv.org/html/2607.08974v1/x3.png)

**说明**: Qwen3.5 0.8B/2B/4B 在 [[VLABench]] 上的能力评估（微调前），对比 non-CoT 和 [[Chain-of-Thought Reasoning|Chain-of-Thought]] 提示。揭示 VLM 规模与能力的非单调关系：Complex 和 Physics Law 任务上 0.8B/2B 模型持平甚至超过 4B。

### Figure 4: 真实机器人实验设置

![Figure 4](https://arxiv.org/html/2607.08974v1/x4.png)

**说明**: UR5e 机械臂配 Robotiq 2F-85 夹爪，在杂乱桌面场景执行抓取放置任务，共 120 条示范数据。

### Figure 5: 抓取任务场景

![Figure 5](https://arxiv.org/html/2607.08974v1/x5.png)

**说明**: 三种训练场景（carrot/cube/tomato 目标物）和一种 OOD 场景（引入未见过的干扰物：银色浇水壶、橙色玩具车）。

### Figure 6: CLAP@Qwen3.5-2B 在 LIBERO 各套件上的训练曲线

![Figure 6](https://arxiv.org/html/2607.08974v1/x6.png)

**说明**: CLAP 2B 在四个 [[LIBERO]] 套件上的成功率随训练 epoch 的变化曲线，单 epoch 即达到 90.8% 平均成功率。

### Figure 7: VLABench 能力画像（0-shot, 微调前）

![Figure 7](https://arxiv.org/html/2607.08974v1/x7.png)

**说明**: 与 Figure 3 相同的评估，但无 in-context 示例。0-shot 设置下各模型能力退化明显，但规模间的非单调关系依然存在。

### Table 1: CLAP 输出序列示例

| 组件 | 内容 |
|------|------|
| 系统提示 | "Analyze input image and predict robot actions for next 8 timesteps" |
| 用户指令 | "grasp cube and drop it onto the plate" |
| 语言动作描述 | `<think> move back 0.5 cm, move up 3.5 cm, move left 0.4 cm, tilt left 1 degrees, tilt back 1 degrees, rotate counterclockwise 3 degrees, open gripper </think>` |
| 数值动作 token | `408 980 166 319 505 491 1000 ... 412 978 137 277 484 479 1000`（56 个 token） |

**说明**: 语言描述约 149 个 token，数值动作 56 个 token（7 DoF × 8 步 × 1000 bins 离散化）。

### Table 2: LIBERO Benchmark 结果（h=8, 单 epoch 微调）

| 方法 | Backbone | 规模 | 预训练 | Spatial↑ | Object↑ | Goal↑ | Long↑ | Avg↑ |
|------|----------|------|--------|----------|---------|-------|-------|------|
| [[VLA-0]] (repro.) | Qwen3.5 | 0.8B | — | 77.6 | 87.8 | 78.4 | 60.6 | 76.1 |
| **CLAP (Ours)** | Qwen3.5 | 0.8B | — | **92.6** | **97.8** | **88.4** | **79.6** | **89.6 (+13.5)** |
| [[VLA-0]] (repro.) | Qwen3.5 | 2B | — | 77.8 | 86.6 | 77.0 | 62.4 | 75.9 |
| **CLAP (Ours)** | Qwen3.5 | 2B | — | **93.0** | **97.4** | **90.8** | **82.0** | **90.8 (+14.9)** |
| [[VLA-0]] (repro.) | Qwen3.5 | 4B | — | 37.6 | 84.8 | 85.8 | 48.6 | 64.2 |
| **CLAP (Ours)** | Qwen3.5 | 4B | — | **88.0** | **97.4** | **86.6** | **67.6** | **84.9 (+20.7)** |
| SmolVLA | SmolVLM2 | 2.25B | — | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| VLA-0 | Qwen2.5-VL | 3B | — | 97.0 | 97.8 | 96.2 | 87.6 | 94.7 |
| π0.5 | PaliGemma | 3B | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| OpenVLA | Prismatic | 7B | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |

**说明**: CLAP 在无 VLA 预训练的条件下，三个规模均大幅超越同主干 VLA-0 基线（+13.5 到 +20.7 pts），2B 模型性价比最优。

### Table 3: Action Masking 消融实验（1 epoch, h=8）

| 方法 | 规模 | Masking | Spatial↑ | Object↑ | Goal↑ | Long↑ | Avg↑ |
|------|------|---------|----------|---------|-------|-------|------|
| CLAP | 0.8B | — | 92.6 | 97.8 | 88.4 | 79.6 | 89.6 |
| CLAP | 0.8B | ✓ | 87.6 | 95.4 | 90.0 | 69.8 | 85.7 (−3.9) |
| CLAP | 2B | — | 93.0 | 97.4 | 90.8 | 82.0 | 90.8 |
| CLAP | 2B | ✓ | 94.8 | 96.8 | 91.6 | 73.2 | 89.1 (−1.7) |
| CLAP | 4B | — | 88.0 | 97.4 | 86.6 | 67.6 | 84.9 |
| CLAP | 4B | ✓ | 91.6 | 92.2 | 89.8 | 78.8 | **88.1 (+3.2)** |

**关键发现**: [[Action Masking]] 对 4B 模型有正向收益，对小模型有负向影响，建议按模型规模选择性使用。

### Table 4: LIBERO-PRO OOD 泛化结果（平均成功率 %）

| 方法 | 规模 | Spatial | Object | Goal | Long | Avg |
|------|------|---------|--------|------|------|-----|
| [[VLA-0]] | 0.8B | — | — | — | — | 34.5 |
| [[VLA-0]] | 2B | — | — | — | — | 32.9 |
| [[VLA-0]] | 4B | — | — | — | — | 30.6 |
| CLAP w/o masking | 0.8B | — | — | — | — | 40.0 (+5.5) |
| CLAP w/o masking | 2B | — | — | — | — | **44.0 (+11.1)** |
| CLAP w/o masking | 4B | — | — | — | — | 41.5 (+10.9) |
| CLAP w/ masking | 4B | — | — | — | — | **44.8 (+14.2)** |

**说明**: [[LIBERO-PRO]] 涵盖 4 类 OOD 扰动（Obj/Pos/Sem/Task），CLAP 2B（无 masking）在 OOD 平均上 +11.1 pts，4B（有 masking）达最佳 +14.2 pts，其中 Spatial 套件新物体场景 +42.6-54.4 pts。

### Table 5: 动作表示模板映射

| 轴 | Sign > 0 | Sign < 0 |
|----|----------|----------|
| $\Delta x$ | "move forward \|Δx\| cm" | "move back \|Δx\| cm" |
| $\Delta y$ | "move left \|Δy\| cm" | "move right \|Δy\| cm" |
| $\Delta z$ | "move up \|Δz\| cm" | "move down \|Δz\| cm" |
| $\Delta\varphi$ | "tilt left \|Δφ\| degrees" | "tilt right \|Δφ\| degrees" |
| $\Delta\theta$ | "tilt back \|Δθ\| degrees" | "tilt forward \|Δθ\| degrees" |
| $\Delta\psi$ | "rotate counterclockwise \|Δψ\| degrees" | "rotate clockwise \|Δψ\| degrees" |

### Table 6: 真实机器人抓取放置成功率

| 方法 | 规模 | ID (%) | OOD (%) |
|------|------|--------|---------|
| CLAP | 0.8B | 35.0 | 10.0 |
| CLAP | 2B | **60.0** | **60.0** |

**说明**: 2B 模型在真实 UR5e 上 ID 和 OOD 均达 60%，显示对未见干扰物的鲁棒性。0.8B 模型 OOD 性能急剧下降。

### Table 7: 推理延迟对比（RTX 4090, h=8）

| 方法 | 规模 | 延迟 (s) | 有效控制频率 | GPU 显存 (GiB) |
|------|------|---------|------------|--------------|
| [[VLA-0]] | 0.8B | 3.205 | 2.50 Hz | 3.9 |
| CLAP | 0.8B | 4.233 | 1.89 Hz | 3.9 |
| CLAP | 2B | 4.307 | 1.86 Hz | 9.1 |
| CLAP | 4B | 6.013 | 1.33 Hz | 18.0 |

**说明**: 语言前缀带来约 +1s（+32%）的延迟开销，0.8B/2B 约 1.86-1.89 Hz 的控制频率，在大多数桌面操作任务中可接受。

### Table 8: 训练超参数

| 超参数 | 值 |
|--------|---|
| 优化器 | [[AdamW]] |
| 训练步数 | 17,000 步（≈4小时，约 1 epoch） |
| 学习率 | 5.0×10⁻⁶（常数调度） |
| 权重衰减 | 10⁻¹⁰ |
| 有效批大小 | 128（16/GPU × 8 GPU，DDP） |
| 混合精度 | bfloat16 |
| Attention | [[FlashAttention|Flash Attention 2]] |
| 图像增强 | 随机裁剪（scale 0.875）+ 颜色抖动 |
| Action masking 率 | 0.4 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件 ×50 任务，500 demos/task | 桌面操作 benchmark，标准化评估 | 主要训练/测试 |
| [[LIBERO-PRO]] | 4 类 OOD 扰动 | 测试新物体/位置/语义/任务的 OOD 鲁棒性 | OOD 泛化评估 |
| [[VLABench]] | 多维能力画像 | VLM 多能力评估（空间/物理/推理等） | VLM 能力分析 |
| UR5e 真实数据 | 120 条示范 | 3 类目标物，杂乱桌面场景 | 真实机器人评估 |

### 实现细节

- **Backbone**: [[Qwen3-VL|Qwen3.5-VL]] (0.8B / 2B / 4B)
- **优化器**: [[AdamW]]，学习率 5.0×10⁻⁶（常数调度），权重衰减 10⁻¹⁰
- **Batch Size**: 128（16/GPU × 8 GPU，DDP）
- **训练轮数**: 17,000 步，约 1 epoch，≈4 小时
- **硬件**: 8× GPU（具体型号未指定），推理评估在 RTX 4090
- **动作离散化**: 1000 bins/dim，7 DoF × 8 步 = 56 个 token

### 可视化结果

- CLAP 2B 在 LIBERO Long 套件（最难）从 62.4% 提升至 82.0%，说明语言计划对长程任务尤为有效
- 真实机器人上 2B 模型 ID/OOD 均 60%，面对视觉相似干扰物（橙色玩具车 vs 胡萝卜）保持良好鲁棒性

---

## 批判性思考

### 优点

1. **极简适配**: 无架构改动、无额外模型、无额外标注，只改输出格式，工程实现极其简单
2. **VLM 生态兼容**: 可直接享用 VLM 侧的蒸馏、量化、新主干进展，无需重新设计动作管道
3. **OOD 鲁棒性显著提升**: 语言计划作为中间表示有效提升了对新物体/场景的泛化，LIBERO-PRO 平均 +11.1 pts

### 局限性

1. **单 VLM 家族评估**: 仅在 Qwen3.5 系列上验证，对 PaliGemma、LLaVA 等其他 VLM 的迁移能力未知
2. **推理延迟增加**: 语言前缀带来 +32% 延迟（约 1s），对实时性要求高的任务（如高速操作）存在挑战
3. **自回归生成瓶颈**: 相比并行解码方法（如 Diffusion Policy），顺序生成效率受限
4. **真实数据量有限**: 仅 120 条真实示范，泛化到其他机器人平台/任务的能力未验证

### 潜在改进方向

1. 与投机采样（speculative decoding）结合，降低语言前缀的生成延迟
2. 在其他 VLM 家族（PaliGemma、Qwen2.5-VL）上验证方法通用性
3. 探索更丰富的语言描述模板（如包含接触力、夹爪状态的描述），提升复杂操作的表达能力

### 可复现性评估

- [ ] 代码开源（承诺发布，暂未发布）
- [ ] 预训练模型（承诺发布 0.8B/2B/4B 权重，暂未发布）
- [x] 训练细节完整（超参数、数据集均已公开）
- [x] 数据集可获取（LIBERO 公开数据集）

---

## 关联笔记

### 基于

- [[VLA-0]]: 直接改进的基线，同样使用 Qwen3.5 主干的裸 token VLA
- [[Qwen3-VL]]: 使用的 VLM 主干网络

### 对比

- [[VLA-0]]: 主要对比基线（+13.5 to +20.7 pts）
- [[TraceVLA]]: 同样引入中间表示，但需额外监督
- [[OpenVLA]]: 更大参数（7B）但无 VLM 适配策略

### 方法相关

- [[Language-Action Grounding]]: 核心创新，语言描述条件化动作预测
- [[Action Chunking]]: 使用 $h=8$ 的动作块预测
- [[Autoregressive Policy]]: 整体框架
- [[Action Masking]]: 可选的鲁棒性增强
- [[Output-Distribution Mismatch]]: 所解决的核心问题

### 硬件/数据相关

- [[LIBERO]]: 主要评估 benchmark
- [[LIBERO-PRO]]: OOD 鲁棒性评估
- [[VLABench]]: VLM 能力分析

---

## 速查卡片

> [!summary] CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding
> - **核心**: 在数值动作 token 前插入自然语言动作描述，消解 VLM-VLA 输出分布不匹配
> - **方法**: 固定模板生成语言计划 → 自回归预测数值 token，零架构修改
> - **结果**: 2B 模型 LIBERO 90.8%（+14.9 vs VLA-0），单 epoch 微调
> - **代码**: 待发布（https://omron-sinicx.github.io/clap/）

---

*笔记创建时间: 2026-07-14*

---
title: "Self-Distilling Future Video Representations for World Action Models"
method_name: "FRED"
authors: [Huiwon Jang, Suhyeok Jang, Hyeonmo Kang, Seungku Kim, Jinwoo Shin, Sihyun Yu]
year: 2026
venue: arXiv
tags: [world-action-model, self-distillation, video-diffusion, robot-manipulation, embodied-ai, representation-learning, human-to-robot-transfer]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.30320
created: 2026-10-24
---

# 论文笔记：Self-Distilling Future Video Representations for World Action Models (FRED)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RLWRLD & KAIST |
| 日期 | October 2026 |
| 项目主页 | [huiwon-jang.github.io/FRED](https://huiwon-jang.github.io/FRED) |
| 对比基线 | [[AGRA]]、[[Fast-WAM]]、[[DreamZero]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.30320) |

---

## 一句话总结

> FRED 通过自蒸馏训练让视频扩散模型的单步噪声表征更好地逼近真实未来观测的表征，并引入历史帧条件来增强时序上下文，从而大幅提升 World Action Model 的控制性能，尤其在长时域泛化和人形机器人跨域迁移任务上表现突出。

---

## 核心贡献

1. **未来表征自蒸馏（FRED 核心）**: 用 [[EMA]] 动量教师编码轻微加噪的真实未来帧作为蒸馏目标，指导学生网络从纯噪声生成的单步表征向真实未来对齐，本质上将"生成质量"转化为"表征质量"
2. **历史帧时序条件化**: 将最近 4 帧历史观测同时输入学生和教师网络，补充当前运动状态信息，使表征具备时序连贯性
3. **动作解码器独立训练**: 动作损失不反向传播到视频骨干（`stop_gradient`），防止动作监督信号干扰视频表征的学习质量

---

## 问题背景

### 要解决的问题

[[World Action Model]]（WAM）将预训练视频扩散模型（VDM）与动作解码器耦合，推理时只做一步去噪（从纯高斯噪声 $\varepsilon$ 出发），提取 [[Video Diffusion Model]] 隐状态直接解码机器人动作。问题在于：**单步噪声生成的表征质量与实际控制性能之间存在显著差距。**

### 现有方法的局限

论文首先做了诊断实验（Figure 1），发现：

- 视频 DiT 隐状态中确实编码了未来运动信息（probing 准确率随训练提升），但始终低于干净未来观测编码的水平
- 视频生成质量（[[FVD]]）与控制成功率的 Pearson 相关系数仅为 **r = 0.24**，几乎不相关
- 干净未来观测的表征质量（[[CKNNA]] 相似度）与控制性能相关 **r = 0.49**
- 但单步噪声生成的表征质量与控制性能相关高达 **r = 0.89**

这说明优化视频生成质量（像素级重建）并不等于优化对控制有用的表征质量。

### 本文的动机

WAM 推理时使用的是从纯噪声单步去噪的表征，而非从干净未来帧提取的表征。那么，如果能让"从噪声生成的表征"更接近"从真实未来帧提取的表征"，控制性能应该大幅提升。[[自蒸馏]] 是实现这一目标的天然工具：用真实未来的轻微加噪版本作为蒸馏目标，引导噪声端的学生网络对齐。

---

## 方法详解

### 模型架构

FRED 基于已有的 [[World Action Model]] 框架（以 DiT4DiT 为基础 WAM），在视频 DiT 的训练目标中添加自蒸馏分支：

- **视频骨干**: 视频 [[DiT]]（5B 或 14B 参数），接受当前观测 $o_t$、历史观测 $o_{t-3:t}$ 和语言指令 $c$
- **核心模块**: [[自蒸馏]] 分支——学生从纯噪声 $\varepsilon$ 出发，EMA 教师从低噪声未来帧 $o_{t+1}^{\tau}$ 出发
- **动作解码器**: 独立训练，从视频骨干冻结隐状态解码 [[Action Chunking|动作块]]
- **总参数**: 视频骨干 5B / 14B + 动作解码器

### 核心模块

#### 模块 1: 自蒸馏目标（Self-Distillation Branch）

**设计动机**: 推理时表征来自纯噪声单步去噪，而控制性能强相关于此表征的质量。通过让学生（噪声端）对齐教师（真实未来端），缩小训练-推理的表征差距。

**具体实现**:

- **学生**: 视频 DiT $f_\theta$ 以纯噪声 $\varepsilon$ 为输入，结合历史 $o_{t-3:t}$ 和语言 $c$，提取第 $\ell$ 层表征后经投影头 $g_\theta$ 输出分布
- **教师**: 动量网络 $f_{\theta^-}$（[[EMA]] 更新），以轻微加噪的未来帧 $o_{t+1}^{\tau}$（$\tau=0.25$）为输入，提取对应表征
- **目标分布对齐**: 用 KL 散度最小化学生分布与教师分布之间的距离，教师端使用 DINO 风格的中心化 $\mathbf{m}$ 避免坍缩

#### 模块 2: 历史帧条件化（Historical Conditioning）

**设计动机**: 单帧观测无法传递当前运动状态（速度、方向），加入历史帧能让模型理解物体正在如何运动。

**具体实现**:
- 学生和教师都额外输入最近 4 帧历史观测 $o_{t-3:t}$
- 历史帧以与当前帧相同的方式编码，通过条件注入进入视频 DiT

#### 模块 3: 动作梯度阻断（Action Gradient Stopping）

**设计动机**: 动作损失（基于机器人状态标注）和视频自蒸馏损失优化目标不一致，若混合反传会降低表征质量。

**具体实现**:
- 动作解码器 $L_{\text{action}} = L_{\text{FM}}$ 只更新动作解码器参数
- 梯度不传回视频骨干（stop_gradient）
- 消融验证：去掉 stop_gradient 后 RoboCasa 从 68.8 降至 65.6

---

## 关键公式

### 公式 1: [[Video Diffusion Model|未来表征生成]]

$$
\mathbf{h} = f_\theta(\varepsilon \mid o_{t-3:t}, c)
$$

**含义**: 学生网络从纯高斯噪声 $\varepsilon$ 出发，给定历史观测和语言条件，单步生成对未来的表征（推理时也是这个路径）。

**符号说明**:
- $f_\theta$: 视频 DiT 骨干（学生，可训练）
- $\varepsilon \sim \mathcal{N}(0, I)$: 纯高斯噪声
- $o_{t-3:t}$: 4 帧历史观测
- $c$: 语言任务指令
- $\mathbf{h}$: 提取的未来表征向量

### 公式 2: [[自蒸馏|教师目标构建]]

$$
o_{t+1}^{\tau} = (1 - \tau)\,o_{t+1} + \tau\,\varepsilon, \quad \tau = 0.25
$$

$$
\mathbf{h}^{\tau} = f_{\theta^-}(o_{t+1}^{\tau} \mid o_{t-3:t}, c)
$$

**含义**: 教师网络以轻微加噪的真实未来帧为输入，提取对未来的"干净"表征作为蒸馏目标。$\tau = 0.25$ 是最优噪声水平（消融验证）。

**符号说明**:
- $o_{t+1}$: 真实未来观测帧
- $\tau$: 教师侧噪声水平（0.25 最优）
- $f_{\theta^-}$: [[EMA]] 动量教师网络（不参与梯度更新）
- $\mathbf{h}^{\tau}$: 教师生成的目标表征

### 公式 3: [[自蒸馏|学生-教师分布对齐]]

$$
\mathbf{p}_i = \text{softmax}\!\left(\frac{g_\theta(\mathbf{h}_i)}{\gamma_1}\right)
$$

$$
\mathbf{p}_i^{\tau} = \text{softmax}\!\left(\frac{g_{\theta^-}(\mathbf{h}_i^{\tau}) - \mathbf{m}}{\gamma_2}\right)
$$

**含义**: 学生和教师分别通过投影头 $g$ 将表征映射为概率分布；教师端使用可学习中心化向量 $\mathbf{m}$（DINO 风格），防止学生坍缩到均匀分布。

**符号说明**:
- $g_\theta, g_{\theta^-}$: 可学习投影头（学生/教师版本）
- $\gamma_1, \gamma_2$: 温度超参数
- $\mathbf{m}$: 中心化向量（[[EMA]] 更新，DINO 风格）

### 公式 4: [[自蒸馏|自蒸馏损失]]

$$
\mathcal{L}_{\text{SD}} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{KL}\!\left(\mathbf{p}_i^{\tau} \,\|\, \mathbf{p}_i\right)
$$

**含义**: 在 $N$ 个表征位置上，最小化教师分布与学生分布之间的 KL 散度，驱使学生从噪声生成的表征向教师的真实未来表征对齐。

**符号说明**:
- $N$: 批次内表征数量
- $\mathbf{p}_i^{\tau}$: 第 $i$ 个位置的教师分布
- $\mathbf{p}_i$: 第 $i$ 个位置的学生分布

### 公式 5: [[Flow Matching|视频训练目标]]

$$
\mathcal{L}_{\text{video}} = \mathcal{L}_{\text{FM}} + \lambda_{\text{SD}} \cdot \mathcal{L}_{\text{SD}}, \quad \lambda_{\text{SD}} = 0.01
$$

**含义**: 视频骨干联合优化流匹配损失（视频生成质量）和自蒸馏损失（未来表征质量），权重 $\lambda_{\text{SD}} = 0.01$ 保证主要目标仍是视频生成。

**符号说明**:
- $\mathcal{L}_{\text{FM}}$: [[Flow Matching]] 损失（视频生成主目标）
- $\lambda_{\text{SD}} = 0.01$: 自蒸馏损失权重

### 公式 6: [[Flow Matching|流匹配噪声化]]

$$
x^{\tau} = (1 - \tau)\,x + \tau\,\varepsilon
$$

$$
\text{速度目标}: \quad v = \varepsilon - x
$$

**含义**: 流匹配定义了从干净数据 $x$ 到噪声 $\varepsilon$ 的线性插值路径，模型学习预测从噪声走向干净数据的速度场。FRED 的教师侧加噪公式 $o_{t+1}^{\tau}$ 直接沿用此定义。

**符号说明**:
- $x$: 干净数据（未来帧）
- $\tau \in [0, 1]$: 噪声水平
- $v$: 速度目标

### 公式 7: [[Action Chunking|动作训练目标]]

$$
\mathcal{L}_{\text{action}} = \mathcal{L}_{\text{FM}}(\hat{a}_{1:K},\, a_{1:K}^*)
$$

**含义**: 动作解码器独立优化流匹配损失，梯度不传回视频骨干（stop_gradient），防止动作监督信号污染视频表征。

**符号说明**:
- $\hat{a}_{1:K}$: 预测的 $K$ 步动作块
- $a_{1:K}^*$: 真实 $K$ 步动作标注
- $K$: 动作预测时域

---

## 关键图表

### Figure 1: 动机诊断 — 表征质量分析

**Figure 1a: Future-Motion Probing 准确率**

![Figure 1a Probing](https://huiwon-jang.github.io/FRED/static/images/fig_probe_vs_iter.png)

**说明**: 视频 DiT 隐状态中编码的未来运动信息（线性探测准确率）随训练迭代增加，但始终低于干净未来观测的上界，证明存在"噪声-干净"表征差距。FRED 通过自蒸馏缩小这一差距。

**Figure 1b: [[CKNNA]] 相似度**

![Figure 1b CKNNA](https://huiwon-jang.github.io/FRED/static/images/fig_cknna_vs_iter.png)

**说明**: 以干净未来观测表征为参考，用 CKNNA 度量单步噪声生成表征的相似度。FRED 持续提升 CKNNA，而基线 WAM 进展缓慢，证明自蒸馏显著提升表征与真实未来的对齐程度。

### Figure 2: FRED 方法概览

![Figure 2 FRED Overview](https://arxiv.org/html/2610.30320v1/method_overview.png)

**说明**: FRED 的整体架构。左为基线 WAM（视频 DiT 单步去噪提取表征→动作解码器）；右为 FRED 新增的两个模块：① EMA 教师接收加噪未来帧作为蒸馏目标，② 学生和教师均输入 4 帧历史。动作解码器梯度不传回视频骨干。

### Figure 3: 真实世界实验设置

![Figure 3 Setup](https://arxiv.org/html/2610.30320v1/fig_real-world-exp_visualization.png)

**说明**: 展示两个真实世界评估场景：① DROID 零样本泛化（Franka Research 3 + Robotiq 夹爪，4 类任务套件）；② OpenArm 人形机器人人-机协同训练（双臂 7-DoF + Inspire 灵巧手，3 类任务分组）。

### Figure 4: 控制性能与表征质量的相关性

![Figure 4 Correlation](https://huiwon-jang.github.io/FRED/static/images/fig3_correlation.png)

**说明**: 在多个 WAM 变体上实测控制成功率 vs 三种表征质量指标：

| 指标 | Pearson r | 含义 |
|------|-----------|------|
| 一步视频生成质量 ([[FVD]]) | **0.24** | 像素级生成质量与控制几乎不相关 |
| 干净未来观测表征质量 | **0.49** | 表征质量有一定相关性 |
| 单步噪声生成表征质量 | **0.89** | **与控制性能强相关**，是优化目标 |

**关键洞见**: 应当优化"从噪声生成的表征质量"而非视频生成质量，这正是 FRED 的出发点。

### Figure 5: DROID 评估硬件设置

![Figure 5 DROID Setup](https://arxiv.org/html/2610.30320v1/droid_setup.png)

**说明**: DROID 评估使用 Franka Research 3 机械臂 + Robotiq 2F-85 夹爪 + 腕部摄像头 + 外部摄像头。评估分为 4 个任务套件：In-Domain Easy、Long-Horizon、Out-of-Domain、Object Grounding。

### Figure 6: DROID 评估任务可视化

![Figure 6 DROID Tasks](https://arxiv.org/html/2610.30320v1/droid_all_tasks.png)

**说明**: 展示 DROID [[DROID|数据集]] 上 4 类评估套件的任务示例，覆盖从简单抓取到长时域复合操作、跨域物体泛化等不同难度层次。

### Figure 7: OpenArm 评估硬件设置

![Figure 7 OpenArm Setup](https://arxiv.org/html/2610.30320v1/openarm_setup.png)

**说明**: OpenArm 人形机器人平台：双臂 7-DoF + ZED 2i 立体摄像头 + Inspire 灵巧手。数据来源包含机器人示教和人类视频示范，用于评估人-机跨域技能迁移。

### Figure 10: OpenArm 评估任务可视化

![Figure 10 OpenArm Tasks](https://arxiv.org/html/2610.30320v1/openarm_all_tasks.png)

**说明**: 展示 OpenArm 上 3 类任务分组：① 机器人数据训练的原子任务（in-domain atomic）；② 复合任务（in-domain composite）；③ 仅在人类视频中出现的技能迁移任务（human skill transfer）。

### Table 1: DROID 零样本泛化结果（成功率 %）

| 方法 | 参数量 | In-Domain Easy | Long-Horizon | Out-of-Domain | Object Grounding |
|------|--------|---------------|--------------|---------------|-----------------|
| DreamZero | 14B | 63.9 | 61.7 | 10.4 | 39.6 |
| Fast-WAM | 5B | — | 31.7 | 10.4 | — |
| AGRA | 5B | — | 31.7 | 10.4 | — |
| **FRED** | **5B** | — | **46.7** | **37.5** | — |
| **FRED** | **14B** | **79.2** | **81.7** | **52.1** | **54.2** |

**关键发现**: FRED 在零样本 Out-of-Domain 任务上提升最为显著（5B: 10.4→37.5 = **+27.1%**；14B: 10.4→52.1 = **+41.7%**），证明自蒸馏带来的表征泛化能力远超简单扩大模型规模。

### Table 2: OpenArm 人形机器人结果（成功率 %）

| 方法 | In-Domain Atomic | In-Domain Composite | Human Skill Transfer | Instruction Following |
|------|-----------------|--------------------|--------------------|----------------------|
| DiT4DiT | — | — | 19.4 | — |
| AGRA | — | — | — | — |
| **FRED** | **91.7** | **72.2** | **50.0** | **97.2** |

**关键发现**: 人类技能迁移任务 DiT4DiT 最优 19.4% → FRED **50.0%**，提升 **+30.6%**，证明自蒸馏帮助模型更好利用人类视频中的动作模式。

### Table 3: RoboCasa 消融实验

| 配置 | RoboCasa 成功率 |
|------|----------------|
| 基线 DiT4DiT（w/o SD, w/o History） | 61.4 |
| + History only | 64.8 |
| + Self-Distillation only | 65.1 |
| **+ SD + History（FRED 完整版）** | **68.8** |

**关键发现**: 自蒸馏和历史条件化均有贡献，两者结合有超加性效果（61.4→68.8 = **+7.4%**）。

### Table 4: 消融 — 教师噪声水平 $\tau$

| $\tau$ | RoboCasa 成功率 |
|--------|----------------|
| 0.00（干净未来帧）| 65.3 |
| 0.25（**最优**）| **68.8** |
| 0.50 | 67.2 |
| 0.75 | 67.6 |
| 1.00（纯噪声）| 61.4（退化为无蒸馏）|

**关键发现**: $\tau = 0.25$ 最优——轻微加噪既保留未来信息又避免分布过于平滑，完全干净（$\tau=0$）反而因分布过度尖锐不如 0.25。

### Table 5: 消融 — 特征提取层索引

| 层索引 | RoboCasa 成功率 |
|--------|----------------|
| 10 | 57.5 |
| 14 | 65.1 |
| 18（**最优**）| **68.8** |
| 22 | 66.3 |
| 26 | 62.1 |

**关键发现**: 中层（第 18 层，约 2/3 深度）最优；太浅（10）表征语义不足，太深（26）运动动态信息被过滤。

### Table 6: 推理延迟对比

| 方法 | 推理延迟（14B 骨干） |
|------|-------------------|
| DreamZero | 3.255 s |
| **FRED** | **0.406 s** |

**关键发现**: FRED 比 DreamZero **快约 8×**，因为 FRED 仍是单步推理，未引入额外推理计算开销。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | ~76K 轨迹 | 多场景、多物体真实机械臂数据 | 训练 + 零样本泛化测试 |
| OpenArm 机器人数据 | 未公开 | 双臂人形机器人操作 | 人-机协同训练+测试 |
| OpenArm 人类视频 | 未公开 | 人类示范，无机器人标注 | 人类技能迁移训练 |
| [[RoboCasa]] | 24 任务 | GR-1 人形机器人仿真 | 消融实验 |
| GR-1 仿真 | — | ByteDance GR-1 机器人仿真 | 主实验对比 |
| [[LIBERO]] | — | 桌面操作任务套件 | 补充实验 |

### 实现细节

- **视频骨干**: 视频 DiT（5B 和 14B 两版本），推理时从纯噪声单步去噪
- **历史帧**: 4 帧历史观测
- **教师噪声**: $\tau = 0.25$
- **蒸馏特征层**: 第 18 层（28 层骨干）
- **蒸馏权重**: $\lambda_{\text{SD}} = 0.01$
- **EMA 更新率**: 未明确，DINO 风格动量更新
- **动作解码器**: [[Flow Matching]] 目标，独立训练

### 主要结果

- DROID 零样本 OOD: 10.4% (best baseline) → **52.1%** (+41.7%)
- DROID 长时域任务: 61.7% (DreamZero 14B) → **81.7%** (+20%)
- OpenArm 人类技能迁移: 19.4% (DiT4DiT) → **50.0%** (+30.6%)
- RoboCasa: 61.4% (DiT4DiT) → **68.8%** (+7.4%)
- LIBERO: 97.4% (DiT4DiT) → **98.2%**

### 可视化结果

Figure 1 中的 probing 和 CKNNA 分析清晰展示了自蒸馏如何在训练过程中持续缩小"单步噪声表征"与"干净未来观测表征"之间的差距，而基线 WAM 的差距几乎停滞。这是一个强有力的诊断-方案对应设计。

---

## 批判性思考

### 优点

1. **理论动机清晰**: 相关性分析（r=0.89）直接证明了优化目标的正确性，诊断→方法→验证闭环完整
2. **方法简洁可插拔**: FRED 是纯训练时的改变，推理架构不变（仍是单步去噪），无推理开销
3. **人-机迁移结果惊人**: 50% vs 19.4% 的技能迁移成功率差异，证明自蒸馏帮助挖掘人类视频中的动作先验
4. **消融实验充分**: 每个设计选择（蒸馏权重、教师噪声、特征层、历史帧、梯度阻断）均有消融支撑

### 局限性

1. **对视频骨干语言理解的强依赖**: 论文指出，若不加入人类协同训练数据，FRED 在 human skill transfer 任务的语言指令跟随能力为 0%，说明跨域迁移依赖人类数据的语言对齐
2. **无显式跨具身对应建模**: 人-机迁移是隐式的（共享表征空间），没有显式的运动学对应学习，未来人形机器人场景可能需要更强的对应机制
3. **未探索更大/更专业的视频骨干**: 论文承认使用 robot-pretrained 视频骨干可能进一步提升效果
4. **OpenArm 数据集未公开**: 人形机器人实验的核心数据集不可复现

### 潜在改进方向

1. 与更大的 robot-pretrained 视频模型（如 Cosmos-Robot）结合，叠加 FRED 自蒸馏
2. 加入显式跨具身对应学习，提升人-机形态差异较大时的迁移能力
3. 探索在线 fine-tuning 场景下的 FRED，结合 RL 奖励信号

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（消融实验充分，超参数公开）
- [x] 数据集可获取（DROID/RoboCasa/LIBERO 公开，OpenArm 数据集未公开）

---

## 关联笔记

### 基于

- [[World Action Model]]: FRED 建立在 WAM 框架之上，以 DiT4DiT 为基础
- [[Flow Matching]]: 视频生成和动作预测均使用流匹配目标
- [[自蒸馏]]: 核心训练方法——EMA 教师引导学生表征对齐

### 对比

- [[AGRA]]: 同为 WAM 表征改进方向，AGRA 用 DINOv2 语义对齐，FRED 用自蒸馏；两者出发点不同
- [[Fast-WAM]]: 关注 WAM 推理效率，FRED 关注表征质量
- [[DreamZero]]: 14B 大模型 WAM，FRED 5B 在零样本任务上超越其 14B 版本

### 方法相关

- [[自蒸馏]]: 核心学习目标
- [[EMA]]: 教师网络更新策略（动量更新）
- [[CKNNA]]: 衡量表征相似度的关键指标
- [[Flow Matching]]: 视频和动作的生成目标
- [[Action Chunking]]: 动作预测的时域分块策略
- [[REPA]]: 类似的扩散模型表征对齐工作（对比参考）

### 硬件/数据相关

- [[DROID]]: 零样本泛化主要评估数据集
- [[RoboCasa]]: 消融实验仿真平台
- [[LIBERO]]: 补充评估数据集

---

## 速查卡片

> [!summary] FRED — Self-Distilling Future Video Representations for World Action Models
> - **核心**: 单步噪声表征质量 (r=0.89) 是控制性能的真正代理指标，非视频生成质量 (r=0.24)
> - **方法**: EMA 教师接收 τ=0.25 加噪未来帧，KL 散度蒸馏学生单步噪声表征；额外加 4 帧历史帧
> - **结果**: DROID OOD 10.4→52.1%；OpenArm 人类技能迁移 19.4→50.0%；快于 DreamZero 8×
> - **代码**: 未开源（项目页：huiwon-jang.github.io/FRED）

---

*笔记创建时间: 2026-10-24*

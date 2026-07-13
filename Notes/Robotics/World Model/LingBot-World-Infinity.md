---
title: "Infinite Worlds with Versatile Interactions"
method_name: "LingBot-World-Infinity"
authors: [Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, Hao Ouyang]
year: 2026
venue: arXiv
tags: [world-model, interactive-world, video-generation, causal-video, real-time-generation, agentic-ai, video-diffusion]
zotero_collection: 3-Robotics/World-Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.07534
created: 2026-07-13
---

# 论文笔记：Infinite Worlds with Versatile Interactions

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RobbyAnt |
| 日期 | July 2026 |
| 项目主页 | 未公开（见 arXiv 提交页） |
| 对比基线 | M-G 3.0、D-W、[[LingBot-Video]]、HappyOyster、Genie 3、MAGI-1、Worldplay |
| 链接 | [arXiv](https://arxiv.org/abs/2607.07534) |

---

## 一句话总结

> LingBot-World-Infinity 通过[[MoBA双向自回归注意力|MoBA 混合注意力]]、双阶段蒸馏和 [[Director-Pilot]] 智能体框架，实现了 720p/60fps 无限时长、无漂移的交互式世界模拟，首次将交互世界模型从"数分钟"扩展到"数小时"。

---

## 核心贡献

1. **无界交互视野（Unbounded Interaction Horizon）**: 通过[[MoBA双向自回归注意力|MoBA 混合注意力]]训练，消除[[Causal Video Generation|因果视频生成]]中的帧级误差累积，实现小时级无漂移生成
2. **实时生成（Real-Time at 720p/60fps）**: 通过[[一致性蒸馏]]和[[Distribution Matching Distillation|DMD]]两阶段蒸馏，将大模型压缩为实时推理的轻量版本
3. **丰富交互词汇**: 支持战斗、弓箭、施法、射击、环境操控等多样动作，并通过文本提示控制全局/局部事件
4. **Director-Pilot 智能体框架**: [[Director-Pilot]] 将高层语义规划（Director/VLM）与低层物理渲染（Pilot/视频生成器）解耦，支持自主世界演化和多人互动

---

## 问题背景

### 要解决的问题

[[Causal Video Generation|因果交互式世界模型]]（interactive world models）按帧生成视频环境，响应用户/智能体动作。当前存在两个核心瓶颈：

1. **长时域稳定性（Long-horizon stability）**: 帧级自回归生成中误差逐帧累积，导致数分钟内出现纹理退化、几何扭曲和场景漂移
2. **实时交互保真度**: 在高分辨率下以实时帧率响应用户输入，计算代价极高

### 现有方法的局限

- **HappyOyster / Genie 3**: 闭源，交互词汇有限，生成时长仅数分钟
- **MAGI-1 / Worldplay**: 开源因果模型，但数秒至数分钟内即出现明显质量退化
- **[[LingBot-Video]]**: 同团队前作，侧重 MoE 视频预训练，未专注于长时域无漂移交互

### 本文的动机

通过**反漂移（anti-drift）专项预训练**显式抑制误差累积，再通过**分布级别蒸馏**在自回归轨迹上优化，使轻量模型在学生生成的状态分布上也保持稳定——而不仅仅是在 teacher 的分布上。

---

## 方法详解

### 模型架构

LingBot-World-Infinity 采用**[[DiT|扩散 Transformer]]（DiT）**骨干的[[Causal Video Generation|因果视频生成]]架构：

- **输入**: 初始图像 $x_0$ + 背景描述 + 历史帧上下文 $x_{<t}$ + 用户输入（相机姿态 $a_t$ + 文本 Prompt $p_t$）
- **Backbone**: [[Video DiT|Video DiT]]，通过[[MoBA双向自回归注意力|MoBA 注意力掩码]]实现混合注意力
- **动作注入**: [[Plücker Embeddings|Plücker 嵌入]]编码相机位姿，经[[AdaLN]]注入 DiT 块；分块提示（chunk-wise prompts）提供时间定位语义控制
- **输出**: 下一帧 / 下一帧块的视觉状态 $x_t$
- **主模型规模**: 14B 参数（+ 蒸馏轻量版 1.3B）

### 核心模块

#### 模块 1：因果世界模型预训练

**设计动机**: 利用[[Causal Video Generation|因果生成]]约束（效果不能先于原因）保证时序一致性，同时用[[MoBA双向自回归注意力|MoBA 混合注意力]]对抗[[teacher forcing|Teacher Forcing]]退化。

**[[Plücker Embeddings|Plücker 嵌入]]相机控制**:
- 将每条视线（viewing ray）编码为 6 维坐标 $(\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6$，其中 $\mathbf{d}$ 为方向向量，$\mathbf{m} = \mathbf{o} \times \mathbf{d}$ 为矩向量（$\mathbf{o}$ 为光心）
- 通过[[AdaLN]]注入 [[DiT]] 块，实现精确相机轨迹控制

**[[MoBA双向自回归注意力|MoBA 混合注意力掩码]]**:
- **自注意力（Self-Attention）**: 带噪当前帧同时关注干净历史上下文（Teacher Forcing 组件）+ 双向全局块（正则化组件）
- **交叉注意力（Cross-Attention）**: Teacher Forcing 分量使用下三角掩码（防止未来信息泄露）关注背景 Prompt 和 Chunk Prompt；双向分量关注全局 Prompt

#### 模块 2：后训练蒸馏（双阶段）

**设计动机**: 将 14B 教师压缩为实时 1.3B 学生，同时在长自回归轨迹上抑制漂移积累。

**第一阶段——[[一致性蒸馏]]（Consistency Distillation）**:
- 同一教师概率流 ODE 轨迹上的相邻点应映射到相同输出，强制轨迹不变性

**第二阶段——[[Distribution Matching Distillation|DMD 分布匹配蒸馏]]**:
- 在**学生自回归长轨迹**上优化 KL 散度梯度，而非仅在教师分布上训练
- 有效抑制长时域的漂移累积

#### 模块 3：Director-Pilot 智能体框架

**设计动机**: 将高层语义因果推理与低层物理渲染解耦。

**[[Director-Pilot]] 双角色**:
- **Director**（VLM）: 分析当前帧，进行因果推理，生成"事件提案"（Event Proposals）
- **Pilot**（视频生成器）: 接收语义指令，渲染时空一致的物理视觉序列

**交互模式 A——直接语义交互**: VLM 分析全帧，无需对象掩码，生成整体环境动态事件卡

**交互模式 B——追踪辅助对象交互**: 集成 [[SAM2|SAM（Segment Anything Model）]]，精确追踪交互对象，VLM 为每个对象分配动作提案，跨 Chunk 保持空间一致性

#### 模块 4：世界干预（Agentic World Intervention）

用户可通过任意文本事件改变世界状态：

- **全局状态切换**: 昼夜变换、天气操控（暴风雪）、重大事件触发
- **局部实体注入**: 召唤特定生物/物体，VLM 决定合理的时空进入点

---

## 关键公式

### 公式 1：[[Causal Video Generation|因果世界模型分解]]

$$
p_\theta(x_{1:T} | a_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t | x_{<t},\, a_{\leq t})
$$

**含义**: 将交互式世界模拟建模为因果生成过程，每帧状态仅依赖历史上下文和当前动作，确保效果不先于原因

**符号说明**:
- $x_t$: 时刻 $t$ 的视觉状态（帧）
- $a_t$: 用户输入（相机姿态 + 动作文本 Prompt）
- $x_{<t}$: 历史干净帧上下文
- $\theta$: 模型参数

### 公式 2：[[FlowMatching|条件流匹配]]训练目标

$$
\mathcal{L}_{fm} = \mathbb{E}_{x,i,t,\varepsilon}\, \left\| v_\theta\!\left(x_i^t,\, t \mid x_{<i},\, p_{\leq i},\, a_{\leq i}\right) - \left(\varepsilon - x_i\right) \right\|^2
$$

**含义**: 训练网络预测流速度（flow velocity），将带噪隐变量 $x_i^t$ 引导回干净帧 $x_i$，条件为历史上下文、Prompt 和动作

**符号说明**:
- $x_i^t = (1-t)x_i + t\varepsilon$: 带噪隐变量（$t \in [0,1]$）
- $\varepsilon$: 标准高斯噪声
- $v_\theta(\cdot)$: 网络预测的流速度
- $p_{\leq i}$: 第 $i$ 帧前的所有 Chunk Prompt
- $a_{\leq i}$: 第 $i$ 帧前的所有相机/动作输入

### 公式 3：[[一致性蒸馏]]损失

$$
\mathcal{L}_{CD} = \mathbb{E}\left[d\!\left(G_\theta(x_i^t,\, t \mid c),\; G_{\theta^-}(\tilde{x}_i^{t-\Delta t},\, t - \Delta t \mid c)\right)\right]
$$

**含义**: 强制同一 ODE 轨迹上相邻两点经一致性模型 $G_\theta$ 映射后得到相同输出，实现轨迹不变性

**符号说明**:
- $G_\theta$: 学生一致性模型
- $\theta^-$: $\theta$ 的指数滑动平均（EMA）参数
- $\tilde{x}_i^{t-\Delta t}$: 教师 ODE 单步推进后的相邻点
- $d(\cdot,\cdot)$: 距离函数（如 LPIPS 或 $\ell_2$）
- $c$: 条件（历史上下文 + 动作 + Prompt）

### 公式 4：[[Distribution Matching Distillation|分布匹配蒸馏（DMD）]]梯度

$$
\nabla_\theta\, \mathbb{E}\!\left[D_{KL}(p_{\theta,t} \,\|\, p_{data,t})\right]
= -\mathbb{E}\!\left[\bigl(s_{real}(\hat{x}_i^t, t \mid c) - s_{fake}(\hat{x}_i^t, t \mid c)\bigr)\,\frac{\partial \hat{x}_i}{\partial \theta}\right]
$$

**含义**: 通过最小化学生生成分布与真实数据分布的 KL 散度，在长自回归轨迹上优化漂移抑制——学生采样 $\hat{x}_i$ 后加噪，计算真实/假分数差值作为梯度信号

**符号说明**:
- $p_{\theta,t}$: 学生在噪声水平 $t$ 的边缘分布
- $p_{data,t}$: 真实数据在噪声水平 $t$ 的边缘分布
- $s_{real}(\cdot)$: 真实分布的分数函数（score function）
- $s_{fake}(\cdot)$: 学生分布的分数函数
- $\hat{x}_i$: 学生在自回归轨迹上生成的第 $i$ 帧

---

## 关键图表

### Figure 1：系统 Teaser

![Figure 1](https://arxiv.org/html/2607.07534v1/x1.png)

**说明**: LingBot-World-Infinity 实时生成无限世界，展示了多样化交互场景（战斗、探索、施法等）。

### Figure 2：数据引擎总览

![Figure 2](https://arxiv.org/html/2607.07534v1/x2.png)

**说明**: 异质原始视频（自我中心视频 + 合成数据 + 网络视频）经时序切割、多维质量过滤，路由到类别专属标注流水线，生成优化后的 Chunk-wise 字幕。

### Figure 3：LingBot-World-Infinity 流水线总览

![Figure 3](https://arxiv.org/html/2607.07534v1/x3.png)

**说明**: 系统以初始图像和背景描述初始化，[[Causal Video Generation|因果视频模型]]根据历史上下文和用户输入（相机姿态 + Prompt）自回归生成未来世界状态。

### Figure 4：DiT Block 与 MoBA 注意力掩码

![Figure 4](https://arxiv.org/html/2607.07534v1/x4.png)

**说明**: 左侧展示将动作（相机姿态 + Chunk Prompt）注入 [[DiT]] 块的方式；右侧对比自注意力（Teacher Forcing + 双向块）和交叉注意力（下三角 + 全局 Prompt）的[[MoBA双向自回归注意力|MoBA 混合掩码]]设计。

### Figure 5：智能体交互框架总览

![Figure 5](https://arxiv.org/html/2607.07534v1/x5.png)

**说明**: [[Director-Pilot]] 框架中，用户可通过语义动作或对象中心动作与现有世界交互，或通过高层文本事件进行世界干预。VLM（Director）进行因果推理，视频生成器（Pilot）将语义决策渲染为物理一致的视觉序列。

### Figure 6：追踪模式界面

![Figure 6](https://arxiv.org/html/2607.07534v1/x6.png)

**说明**: VLM 理解场景内交互对象，[[SAM2|追踪模型]]实时追踪目标并显示动态交互浮窗（事件卡）。[[Director-Pilot]] 框架基于用户动作（如推门、旋转足球）进行因果推理，渲染物理逻辑一致的时空动态。

### Figure 7：可控世界探索（1/2）

![Figure 7](https://arxiv.org/html/2607.07534v1/x7.png)

**说明**: 支持跨不同世界时域灵活切换 Prompt，场景语义平滑演化，同时支持多种主角和物体的可控导航。

### Figure 8：可控世界探索（2/2）

![Figure 8](https://arxiv.org/html/2607.07534v1/x8.png)

**说明**: 续图 7，进一步展示多场景下的多样交互，包括各类动作词汇（战斗、弓箭、施法等）在生成世界中的效果。

### Figure 9：交互界面设计

![Figure 9](https://arxiv.org/html/2607.07534v1/figures/UI.png)

**说明**: 界面三区域——中央生成视口（实时响应用户输入）、底部面板（WASD 移动 + IJKL 视角控制）、右侧智能体控制面板（VLM 提案事件 + 热键绑定）。

### Figure 10：一小时连续世界生成

![Figure 10](https://arxiv.org/html/2607.07534v1/x9.png)

**说明**: 从单个 60 分钟生成会话中采样帧，覆盖 20 个不同场景。时间轴展示模型在长时无中断 rollout 中保持连贯场景结构和视觉质量，是长时域稳定性的定性压力测试。

### Figure 11：定性对比（蒸馏模型）

![Figure 11](https://arxiv.org/html/2607.07534v1/x10.png)

**说明**: 与闭源基线（HappyOyster、Genie 3）及开源基线对比，LingBot-World-Infinity 在长时域生成中保持稳定的视觉和物理一致性。

### Figure 12：定性对比（因果预训练模型）

![Figure 12](https://arxiv.org/html/2607.07534v1/x11.png)

**说明**: 与 MAGI-1、Worldplay 等因果预训练基线对比，LingBot-World-Infinity 骨干网络展示出显著更优的长时域稳定性，归因于反漂移（anti-drift）训练策略。

### Table 1：与近期交互世界模型对比

| Metric | M-G 3.0 | D-W | LingBot-Video | HappyOyster | Genie 3 | **Ours** |
|--------|---------|-----|---------------|-------------|---------|---------|
| 生成时长 | 数分钟 | 数分钟 | 数分钟 | 数分钟 | 数分钟 | **无限（小时级）** |
| 语义交互丰富度 | 无 | 无 | 无 | 少量 | 少量 | **无限** |
| 领域 | 游戏 | 通用 | 通用 | 通用 | 通用 | **通用** |
| 动态程度 | 中 | 中 | 高 | 中 | 中 | **高** |
| 实时生成 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 开源 | ✓ | ✓ | ✓ | ✗ | ✗ | **✓** |

**说明**: 本方法是唯一同时满足无限生成时长、丰富语义交互和开源的系统。

---

## 实验

### 数据集

| 数据集来源 | 特点 | 用途 |
|------------|------|------|
| 自我中心视频（Egocentric） | 第一人称真实交互、自然手-物行为 | 训练 |
| 合成数据（Synthetic） | 精确场景几何、时序对齐交互信号（跳跃/攻击/驾驶/飞行） | 训练 |
| 网络视频（Web Videos） | 大规模开放域覆盖、长尾视觉分布 | 训练 |

### 实现细节

- **骨干**: [[Video DiT|Video DiT]]，14B 主模型 + 1.3B 蒸馏轻量版
- **相机控制**: [[Plücker Embeddings|Plücker 嵌入]]（6D）经[[AdaLN]]注入
- **注意力机制**: [[MoBA双向自回归注意力|MoBA 混合注意力掩码]]
- **训练目标**: [[FlowMatching|条件流匹配]]损失 $\mathcal{L}_{fm}$
- **蒸馏**: [[一致性蒸馏]] + [[Distribution Matching Distillation|DMD]]
- **推理优化**: 编译器优化 + 高效注意力核 + 混合并行推理 + 异步流水线 + 动态 KV Cache 调度
- **输出**: 720p / 60 fps 实时生成

### 数据标注流水线

1. **技术验证**: 解码校验、镜头边界检测（TransNet V2）、时长/分辨率约束
2. **质量打分**: 视觉质量、亮度范围、OCR 文字占比、运动统计、编码稳定性
3. **VLM 画像**: 视点、活动类别、交互模式等语义评估
4. **多轨道事件级标注**: 跨主体可见性、运动状态、交互状态、环境动态等独立轨道解耦标注，降低时序歧义
5. **Chunk-wise 优化**: 将轨道状态组合为 Chunk 字幕，移除未来泄露/推测性表达

### 可视化结果

- 1 小时无中断生成会话（20 个场景），视觉质量和场景连贯性始终无可感知退化
- 多种交互模式均有定性展示（战斗、弓箭、环境变换等）

---

## 批判性思考

### 优点

1. **长时域稳定性突破**: 首次实现小时级无漂移交互世界生成，从根本上解决了因果视频模型的误差累积问题
2. **分布级蒸馏**: 在学生自回归轨迹上优化而非仅在教师分布上训练，确保蒸馏后模型同样稳定
3. **系统完整性强**: 从数据流水线到模型架构到智能体框架到用户界面，完整闭环
4. **开源优势**: 在匹配闭源基线质量的同时保持完全开源

### 局限性

1. **长期记忆缺失**: 模型维持视觉稳定性但无真正的世界记忆——离开上下文窗口的区域会被重新生成而非召回，无"持久身份"
2. **身份与风格漂移**: 长时 rollout 中存在人物外观细微偏移和艺术风格渐变
3. **物理理解局限**: 纯像素学习无显式几何/碰撞建模，偶尔产生物理不合理的穿插现象
4. **计算资源需求大**: 即便蒸馏后，实时高保真世界建模仍需大量算力，消费级硬件部署仍有差距

### 潜在改进方向

1. 引入显式 3D 场景表征（如 [[3D Gaussian Splatting]]）实现持久世界记忆
2. 融合物理引擎先验知识以改善碰撞和动力学
3. 探索更激进的蒸馏策略以进一步降低消费级硬件门槛

### 可复现性评估

- [x] 代码开源（arXiv 提交页提及）
- [ ] 预训练模型（未明确发布）
- [x] 训练细节（流水线、损失、蒸馏策略有详述）
- [ ] 数据集（混合私有/公开数据）

---

## 关联笔记

### 基于

- [[LingBot-Video]]: 同团队前作，大规模 MoE 视频预训练骨干
- [[FlowMatching]]: 条件流匹配训练目标
- [[一致性蒸馏]]: 第一阶段蒸馏方法（LCM 路线）
- [[Distribution Matching Distillation]]: 第二阶段分布匹配蒸馏

### 对比

- HappyOyster: 对比交互世界生成时长和交互丰富度
- Genie 3: 闭源对比基线
- MAGI-1: 因果预训练基线对比
- Worldplay: 因果预训练基线对比

### 方法相关

- [[MoBA双向自回归注意力]]: 核心注意力机制（本文提出）
- [[Director-Pilot]]: 智能体框架（本文提出）
- [[Plücker Embeddings]]: 相机姿态编码
- [[AdaLN]]: 动作注入方式
- [[DiT]]: 骨干架构
- [[Video DiT]]: 视频扩散 Transformer
- [[teacher forcing]]: 需要用 MoBA 克服的退化问题
- [[SAM2]]: 对象追踪组件

### 前序概念

- [[Causal Video Generation]]: 本文所在的方法范畴
- [[Video Diffusion Model]]: 视频生成基础技术

---

## 速查卡片

> [!summary] LingBot-World-Infinity
> - **核心**: 首个实现小时级无漂移、720p/60fps 实时交互世界模拟的开源系统
> - **方法**: MoBA 双向-自回归混合注意力 + 双阶段蒸馏（CD+DMD）+ Director-Pilot 智能体
> - **结果**: 超越 HappyOyster/Genie 3（闭源）和 MAGI-1/Worldplay（因果开源）
> - **代码**: 见 arXiv 提交页链接

---

*笔记创建时间: 2026-07-13*

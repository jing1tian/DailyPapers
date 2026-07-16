---
title: "FlowWAM: Optical Flow as a Unified Action Representation for World Action Models"
method_name: "FlowWAM"
authors: [Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, Jiabing Yang, Kai Wang, Jianhua Yang, Dong An, He Guan, Gaoteng Liu, Jianlou Si, Jun Huang, Jing Liu, Nianfeng Liu, Yan Huang, Liang Wang]
year: 2026
venue: arXiv
tags: [world-action-model, optical-flow, robot-manipulation, diffusion-transformer, embodied-ai, video-generation, bimanual]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.13017
created: 2026-07-16
---

# 论文笔记：FlowWAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NLPR, Institute of Automation, Chinese Academy of Sciences; UCAS; FiveAges; MBZUAI; Alibaba Group |
| 日期 | July 2026 |
| 项目主页 | [flow-wam.github.io](https://flow-wam.github.io) |
| 对比基线 | [[π₀.₅]], [[Motus]], [[Fast-WAM]], [[GigaWorld]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.13017) / 代码暂未开源 |

---

## 一句话总结

> FlowWAM 以 [[光流 (Optical Flow)|光流]] 作为统一动作表示，将可执行的机器人动作与视频生成先验桥接在同一个[[双流扩散变换器]]框架内，既可做策略推断，又可做动作条件视频生成。

---

## 核心贡献

1. **光流作为统一动作表示**: 将 [[光流 (Optical Flow)|光流]] 视频化编码（[[HSV光流编码]]）后与 RGB 保持相同格式，消除模态差距，使预训练视频生成器可直接复用。
2. **双流扩散推断框架**: [[双流扩散变换器]]并行处理 RGB 与光流流，通过 [[Joint Self-Attention|联合自注意力]] 实现深度时空交互；策略模式生成动作，世界模型模式条件生成视频。
3. **无动作标注预训练**: 光流可从无标注 [[EgoDex]] 人类操作视频中直接提取，无需机器人动作标签即可进行大规模预训练。

---

## 问题背景

### 要解决的问题

[[世界动作模型]]（WAM）需要将机器人动作表示为某种形式，但现有方法中：
- 数值关节角动作与视频像素空间存在模态鸿沟
- 静态空间提示（目标图像、点轨迹）无法充分表达时序运动信息
- 直接复用预训练视频生成器受限于动作格式不兼容

### 现有方法的局限

| 方法类型 | 代表 | 局限 |
|----------|------|------|
| VLA 数值动作 | [[π₀.₅]], X-VLA | 与视频先验存在模态鸿沟 |
| 图像/帧条件 WAM | [[GigaWorld]] | 静态条件，缺乏密集时序运动信息 |
| 视频 WAM | [[Motus]], [[Fast-WAM]] | 要求机器人动作标注，难以利用海量无标注视频 |

### 本文的动机

[[光流 (Optical Flow)|光流]] 既是"像素级位移视频"（格式与 RGB 相同，可直接输入视频生成器），又编码了密集的每像素运动信息，还可以从无标注视频中无监督提取。这三个性质使光流成为理想的"视频原生动作表示"。

---

## 方法详解

### 模型架构

FlowWAM 采用**双流[[Video DiT|视频扩散变换器]]**架构：

- **基础模型**: [[Wan2.2-TI2V-5B]]（50 亿参数图生视频 DiT）
- **文本编码器**: UMT5-XXL
- **VAE**: [[Causal VAE|Wan2.2 因果 VAE]]（两流共享）
- **流**: RGB 流 + 光流流，通过 [[Joint Self-Attention|联合自注意力]] 深度交互
- **Action Expert**: 约 7.8 亿参数，30 层，隐层 1024，16 头注意力
- **输出**: 14 维关节角动作块（32 步）
- **输入分辨率**: 320×384（多视角拼贴）

### 核心模块

#### 模块 1: [[HSV光流编码]]

**设计动机**: 将光流转换为 RGB 图像格式，使其与预训练视频生成器完全兼容，无需额外适配。

**具体实现**:
- 色调（H）编码运动方向：$H = (\text{atan2}(v, u) + \pi) / (2\pi)$
- 饱和度（S）编码位移幅度：$S = \|f_t\| / m$，$m$ 为归一化常数
- 明度（V）恒为 1
- 该映射可逆，支持从 RGB 图像精确恢复光流向量

#### 模块 2: 双流扩散变换器

**设计动机**: 让 RGB 与光流在同一模型内深度交互，既保留视频生成能力，又学习运动表示。

**具体实现**:
- 冻结 [[Wan2.2-TI2V-5B]] 主干权重（VAE 编码器 + Transformer 块）
- 为光流流添加独立的 Patch Embedding 和输出头
- 两流共享 [[Joint Self-Attention|联合自注意力]]（joint self-attention），在 token 层面跨流交互
- 各流独立计算 [[RoPE|旋转位置编码]]（Rotary Position Embedding）

#### 模块 3: [[Action Expert|动作专家]]

**设计动机**: 从视频生成器的中间特征中提取可执行的机器人动作，充分利用视频先验。

**具体实现**:
- 独立的 30 层 Transformer，与视频 DiT 等深
- 以视频 DiT 中间层的光流特征 + 本体感受状态（qpos）为输入
- 通过[[流匹配]]目标预测动作序列

#### 模块 4: 双推断模式

**策略模式（Policy Mode）**:
- 联合去噪生成 RGB 和光流潜变量
- 动作专家从光流特征解码关节角动作
- 无需外部光流条件，端到端推断

**世界模型模式（World-Model Mode）**:
- 目标光流序列作为固定条件输入
- 仅对 RGB 潜变量去噪
- 忠实生成遵循指定运动的未来帧

---

## 关键公式

### 公式 1: [[HSV光流编码|光流 HSV 编码]]

$$
F_t = \varphi(f_t): \quad H = \frac{\text{atan2}(v,u)+\pi}{2\pi},\quad S = \frac{\|f_t\|}{m},\quad V = 1
$$

**含义**: 将每像素位移场 $f_t \in \mathbb{R}^{H \times W \times 2}$ 可逆地映射为 RGB 图像 $F_t$，使光流可直接由标准 VAE 处理。

**符号说明**:
- $f_t$: 时刻 $t$ 的光流场，形状 $H \times W \times 2$
- $(u, v)$: 每像素横向和纵向位移分量
- $m$: 幅度归一化常数
- $F_t$: HSV 编码后的 RGB 光流图像

### 公式 2: [[联合视频损失|视频生成损失]]

$$
\mathcal{L}_{\text{video}} = (1 - \lambda_f)\,\mathcal{L}_{\text{RGB}} + \lambda_f\,\mathcal{L}_{\text{flow}}
$$

**含义**: RGB 流与光流流的[[流匹配]]均方误差加权组合，$\lambda_f = 0.1$。

**符号说明**:
- $\mathcal{L}_{\text{RGB}},\,\mathcal{L}_{\text{flow}}$: 各流的流匹配 MSE
- $\lambda_f = 0.1$: 光流损失权重

### 公式 3: [[运动感知重加权|运动感知空间重加权]]

$$
w_{\text{motion}} = 1 + \alpha \cdot \frac{\langle |z^f - z^f_{\text{ref}}| \rangle_c}{\max\langle |z^f - z^f_{\text{ref}}| \rangle_c}
$$

**含义**: 对运动丰富区域赋予更高损失权重，抑制静态背景的主导效应；$\alpha = 2.0$。

**符号说明**:
- $z^f$: 光流潜变量
- $z^f_{\text{ref}}$: 参考帧光流潜变量
- $\langle \cdot \rangle_c$: 跨潜通道均值
- $\alpha = 2.0$: 重加权强度

### 公式 4: [[随机潜变量注入|随机噪声潜变量混合]]

$$
\tilde{z}^f = (1-\sigma)z^f + \sigma\varepsilon^f,\quad \tilde{z} = (1-\sigma)z + \sigma\varepsilon^r,\quad \sigma \sim \mathcal{U}[0,1]
$$

**含义**: 在训练动作专家时，以概率 $p=0.5$ 对潜变量注入随机噪声，缩小训练与推断时的分布差距（推断时潜变量含去噪噪声）。

**符号说明**:
- $z^f,\,z$: 干净光流和 RGB 潜变量
- $\varepsilon^f,\,\varepsilon^r \sim \mathcal{N}(0,I)$: 独立高斯噪声
- $\sigma$: 噪声级别，均匀采样
- $p = 0.5$: 噪声注入比例

### 公式 5: [[联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{video}} + \lambda_a\,\mathcal{L}_{\text{action}}
$$

**含义**: 阶段 2 联合训练时的总损失，$\lambda_a = 1.0$。

**符号说明**:
- $\mathcal{L}_{\text{video}}$: 视频生成损失（公式 3 加运动感知重加权后）
- $\mathcal{L}_{\text{action}}$: 动作专家的[[流匹配]]预测损失
- $\lambda_a = 1.0$: 动作损失权重

---

## 关键图表

### Figure 1: 框架总览

![Figure 1](https://arxiv.org/html/2607.13017v1/x1.png)

**说明**: FlowWAM 以 [[光流 (Optical Flow)|光流]] 视频作为动作表示，桥接可执行机器人动作与像素空间视频先验。同一框架支持无动作标注预训练、策略推断和动作条件世界建模三种模式。

### Figure 2: 双流架构详图

![Figure 2](https://arxiv.org/html/2607.13017v1/x2.png)

**说明**: RGB 帧和光流图像经共享 [[Causal VAE|因果 VAE]] 编码后，在[[双流扩散变换器]]中通过 [[Joint Self-Attention|联合自注意力]] 深度交互。策略模式下[[Action Expert|动作专家]]从光流中间特征解码动作；世界模型模式下目标光流作为固定条件，仅去噪 RGB 潜变量。

### Figure 3: 真实环境实验结果

![Figure 3](https://arxiv.org/html/2607.13017v1/x3.png)

**说明**: 单臂 Franka（4 任务）和双臂 ARX（3 任务）上的成功率对比。FlowWAM 平均 75.7%，显著优于 [[π₀.₅]]（61.4%）和 [[Motus]]（57.1%），双臂任务优势更大。

### Figure 4: 消融实验与相关性分析

![Figure 4](https://arxiv.org/html/2607.13017v1/x4.png)

**说明**: (a) RoboTwin 策略消融：HSV 编码优于原始数值动作（89.8% vs 69.8%）；(b) WorldArena 世界模型条件变体对比；(c) 光流预测误差与任务成功率呈强负相关（$r = -0.81$），证明策略增益源自光流质量而非解码器捷径。

### Figure 5: 定性可视化

![Figure 5](https://arxiv.org/html/2607.13017v1/x5.png)

**说明**: FlowWAM 在策略模式和世界模型模式下的输出可视化，展示光流预测与 RGB 视频生成的对应关系。

### Table 1: RoboTwin 2.0 策略成功率（21 任务展示，完整 50 任务见附录）

| 方法 | Clean 均值 | Random 均值 |
|------|-----------|------------|
| [[π₀.₅]] | 42.98% | 43.84% |
| X-VLA | 72.88% | 72.84% |
| [[Motus]] | 88.66% | 87.02% |
| GigaWorld-Policy | 86.36% | 85.04% |
| X-WAM | 89.76% | 90.68% |
| [[Fast-WAM]] | 91.88% | 91.78% |
| **FlowWAM (w/o PT)** | 82.40% | 80.80% |
| **FlowWAM (w/ PT)** | **92.94%** | **92.14%** |

**关键发现**: FlowWAM 在 Clean 和 Random 两种设置下均取得最优，EgoDex 预训练（PT）在 Random 设置（视觉多样性更高）上带来更大收益。

### Table 2: WorldArena 动作条件世界建模结果

| 方法 | 条件类型 | Traj. Acc. | Depth Acc. | EWMScore |
|------|----------|-----------|-----------|---------|
| CogVideoX | 文本 | 34.79 | 91.09 | 58.79 |
| Wan 2.6 | 文本 | 12.18 | 75.43 | 59.80 |
| ABot-PhysWorld | 文本 | 31.50 | 71.99 | 62.63 |
| Cosmos-Predict 2.5 | 动作 | 27.49 | 90.35 | 54.29 |
| IRASim | 动作 | 35.92 | 93.50 | 56.15 |
| Ctrl-World | 动作 | 48.20 | 93.25 | 59.98 |
| [[GigaWorld]]-1 | 图像+动作 | 54.27 | 98.44 | 62.34 |
| **FlowWAM** | **光流** | **64.26** | **98.97** | **63.71** |

**关键发现**: FlowWAM 在轨迹精度（Trajectory Accuracy）上相对第二名（GigaWorld-1）提升 18.4%，EWMScore 63.71 排名第一。光流条件比数值动作条件在运动控制上具有显著优势。

### Table 3: 消融实验（RoboTwin Clean 设置）

| 配置 | 成功率 |
|------|--------|
| 原始数值动作 | 69.8% |
| 无运动感知重加权 | 83.9% |
| 无随机潜变量注入 | 82.1% |
| **FlowWAM 完整模型** | **89.8%** |

**关键发现**: HSV 光流编码（vs 数值动作）贡献最大（+20.0pp）；[[运动感知重加权]] 和[[随机潜变量注入]]各贡献约 7pp。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[EgoDex]] | 大规模 | 无动作标注的第一人称操作视频 | 阶段 1 预训练 |
| [[RoboTwin 2.0]] | 50 任务，Clean: 50 demo/任务；Random: 500 demo/任务 | 双臂操作仿真 | 阶段 2 训练 + 评估 |
| [[WorldArena]] | 标准 benchmark | 121 帧@24fps 评测序列 | 世界模型评估 |

### 实现细节

- **基础模型**: [[Wan2.2-TI2V-5B]]（50 亿参数，冻结主干权重）
- **动作专家**: ~7.8 亿参数，30 层 Transformer
- **阶段 1 学习率**: $5 \times 10^{-5}$（双流 DiT，仅视频损失）
- **阶段 2 学习率**: $1 \times 10^{-4}$（全 DiT + 动作专家，联合损失）
- **时序配置**: 9 帧像素帧/序列，32 步动作块，帧间隔 4
- **输入分辨率**: 320×384（多视角拼贴输入）
- **超参数**: $\lambda_f = 0.1$，$\lambda_a = 1.0$，$\alpha = 2.0$，$p = 0.5$

### 可视化结果

- 策略模式：模型生成光流后由动作专家解码，轨迹精确对应目标物体
- 世界模型模式：给定光流条件后，生成视频真实反映指定运动路径
- 双臂任务：光流编码双臂协调运动，相比数值关节角直觉上更自然

---

## 批判性思考

### 优点

1. **格式统一**: 光流与 RGB 共享相同格式，消除模态鸿沟，无缝复用视频 DiT 权重
2. **可扩展性**: 不依赖动作标注即可预训练，为利用海量互联网视频开辟路径
3. **双用途**: 同一模型同时支持策略推断和动作条件世界建模，架构简洁
4. **强实验验证**: 光流质量与任务成功率的强相关（$r = -0.81$）有力佐证设计合理性

### 局限性

1. **光流提取依赖**: 推断时需要额外光流提取步骤（如 RAFT），增加计算开销
2. **分辨率限制**: 320×384 较低，可能影响精细操作任务（如插孔、拧螺丝）
3. **域迁移未充分探索**: EgoDex 预训练效果在 Random 设置显著但 Clean 设置有限，人机域差异影响有待深入分析
4. **代码未开源**: 重复实验依赖项目主页发布进展

### 潜在改进方向

1. **端到端光流**: 将光流提取集成到模型内部，避免推断时外部依赖
2. **分辨率扩展**: 结合 FramePack 等高效视频生成技术提升分辨率
3. **更丰富视频预训练**: 扩大 EgoDex 规模或引入更多人类操作数据集

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（暂无）
- [x] 训练细节完整（论文中较详细）
- [x] 数据集可获取（RoboTwin 2.0、EgoDex、WorldArena 均公开）

---

## 关联笔记

### 基于

- [[Wan2.2-TI2V-5B]]: 基础视频 DiT 骨干网络
- [[光流 (Optical Flow)]]: 核心动作表示形式
- [[世界动作模型]]: 所属研究范式

### 对比

- [[π₀.₅]]: 代表性 VLA，数值动作表示
- [[Motus]]: WAM 基线，视频条件动作预测
- [[Fast-WAM]]: WAM 基线，高效视频生成策略
- [[GigaWorld]]: WAM 基线，图像条件世界建模

### 方法相关

- [[光流 (Optical Flow)]]: 核心表示
- [[HSV光流编码]]: 视频格式化编码
- [[Action Expert|动作专家]]: 动作解码模块
- [[Joint Self-Attention|联合自注意力]]: 双流交互机制
- [[流匹配]]: 扩散训练目标
- [[运动感知重加权]]: 训练增强技术

### 数据集/Benchmark

- [[EgoDex]]: 无标注预训练数据
- [[RoboTwin 2.0]]: 操作任务 benchmark
- [[WorldArena]]: 世界建模 benchmark

---

## 速查卡片

> [!summary] FlowWAM
> - **核心**: 光流 = 视频格式化统一动作表示，桥接视频先验与机器人控制
> - **方法**: 双流扩散 DiT（RGB + 光流），策略模式 + 世界模型模式
> - **结果**: RoboTwin 92.94%（SOTA），WorldArena EWMScore 63.71（SOTA），真实场景 75.7%
> - **主页**: [flow-wam.github.io](https://flow-wam.github.io)

---

*笔记创建时间: 2026-07-16*

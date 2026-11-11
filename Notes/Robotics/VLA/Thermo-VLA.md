---
title: "Thermo-VLA: A Thermal-Aware Vision-Language-Action Model with Gated Residual Modulation"
method_name: "ThermoVLA"
authors: [Jingxuan Chen, Junshi Luo, Chencheng Dong, Xiaojian Nie, Tongge Huang, Qiang Huang]
year: 2026
venue: arXiv
tags: [thermal-perception, vla, multimodal-fusion, gated-residual, robot-manipulation, bimanual, flow-matching]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.11981v1
created: 2026-11-11
---

# 论文笔记：Thermo-VLA: A Thermal-Aware Vision-Language-Action Model with Gated Residual Modulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确披露（通讯作者 Qiang Huang） |
| 日期 | November 2026 |
| 项目主页 | [Thermo-VLA](https://j-x-chen.github.io/Thermo-VLA-web/) |
| 对比基线 | [[π0.5]]、[[ThermoAct]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.11981) / Code: 未开源 |

---

## 一句话总结

> 通过 [[ThermoRes Encoder|双分支热属性分解]] + [[Gated Residual Modulation|语言门控残差注入]]，将热感知能力以"加法残差"的方式注入到现有 VLA 模型的 RGB token 中，在温度依赖型机器人操控任务上比 π0.5(RGBT) 基线高出 18 个百分点。

---

## 核心贡献

1. **[[ThermoRes Encoder|ThermoRes 编码器]]**: 将热图像分解为冷/热双分支属性，通过[[Gated Residual Modulation|语言条件门控残差]]注入 RGB token，不破坏原始 RGB 表示（恒等路径）
2. **六任务真实世界 Benchmark**: 在 [[Unitree G1]] 双臂平台上构建了 1,307 条专家演示的六类任务——从空间定位到温度排序再到温度异常排查
3. **[[RGBRes Encoder|RGBRes 泛化]]***: 将同一残差注入机制推广到颜色属性（RGB 三通道分解），在仿真和真实场景中均有性能提升

---

## 问题背景

### 要解决的问题

大多数[[VLA（视觉-语言-动作模型）|视觉-语言-动作（VLA）模型]]仅依赖 RGB 图像，无法执行需要感知物体温度的指令（如"取走热饮"、"避开高温异常物"）。热图像（infrared/thermal）提供了 RGB 不可见的温度信息。

### 现有方法的局限

- **简单 RGBT 拼接**：直接将热图像作为第四通道或并行视觉输入，热信息未经语义对齐就被送入模型，导致任务泛化差
- **温度相对关系推理不足**：现有方法几乎不测试"哪个更热/冷"之类的相对温度排序任务
- **平台局限**：极少在双臂平台上进行验证

### 本文的动机

热属性是一种"残差属性"——它在 RGB 空间中不可见，但可以作为对 RGB 语义表示的**条件修正量**叠加。若用语言指令来门控这个修正量，就能实现"语言条件下的热属性对齐"，且不破坏原 VLA 的 RGB 能力。

---

## 方法详解

### 模型架构

Thermo-VLA 采用 **双流 + 残差注入** 架构，以 [[π0.5]] 为基础 [[VLA（视觉-语言-动作模型）|VLA]] 骨干：

- **输入**: 语言指令 $l$ + RGB 观测 $x_r$（三路摄像头：头部、左腕、右腕）+ 热图像 $x_t$（单路）+ 机器人状态 $s_t$
- **RGB 流**: [[SigLIP]] 视觉编码器 $E_r$ + 投影层 $P_r$ 生成 RGB tokens
- **热流**: [[ThermoRes Encoder]] — 先分解为 $x_\text{cold}$/$x_\text{hot}$，再经共享热编码器 $E_t$（SigLIP 结构）+ 同一投影层 $P_r$ 生成热属性 tokens
- **融合**: [[Gated Residual Modulation|语言门控残差]] 将热 tokens 加到 RGB head-camera tokens 上
- **动作生成**: [[Flow-Matching|流匹配动作专家]]（继承自 π0.5），生成 [[Action Chunking|动作块]] $A_t \in \mathbb{R}^{H \times 2d_a}$，$d_a=8$（七轴关节 + 夹爪开合）
- **总参数**: 未在论文中明确给出（π0.5 量级）

### 核心模块

#### 模块1: [[ThermoRes Encoder|ThermoRes 编码器]]

**设计动机**: 将热图分离为"冷于背景"与"热于背景"两个属性视图（双分支分解），各自编码后由[[Gated Residual Modulation|语言条件门控权重]]加权注入

**具体实现**:

1. **背景估计**: 从热图上边界的 patch 均值估计背景参考亮度 $b$，$\mu$ 为固定灰度强度
2. **属性分解**（见公式2）: 将热图减去背景后截断，分别生成 $x_\text{cold}$（低于背景的部分）和 $x_\text{hot}$（高于背景的部分）
3. **共享编码**: $Z_\text{cold} = P_r(E_t(x_\text{cold}))$，$Z_\text{hot} = P_r(E_t(x_\text{hot}))$
4. **门控权重计算**（见公式3）: 聚合文本 token 均值 $\bar{Z}_\text{text}$ 与冷/热 token 均值的差值拼接，经 LN + MLP + softmax 得到权重 $[\alpha, \beta]$
5. **残差注入**（见公式4）: 将加权热 tokens 加到 RGB head-camera tokens 上（恒等路径保留原 RGB 表示）

#### 模块2: [[RGBRes Encoder|RGBRes 编码器]]（泛化扩展）

**设计动机**: 将热属性残差机制推广到颜色属性，验证方法的通用性

**具体实现**:
- 将 RGB 图像分解为红/绿/蓝三个颜色属性视图 $x_j^\text{col}$（$j \in \{\text{red, green, blue}\}$）
- 各颜色 token: $Z_j = P_\text{col}(E_\text{col}(x_j^\text{col}))$
- 三路门控权重 $[\alpha', \beta', \gamma]$ 由语言条件计算
- 残差注入（见公式6）

---

## 关键公式

### 公式1: [[VLA（视觉-语言-动作模型）|策略函数]]

$$
A_t \sim \pi_\theta(\cdot \mid x_r, x_t, l, s_t)
$$

**含义**: Thermo-VLA 策略以 RGB 观测、热图像、语言指令、机器人状态为条件，输出长度为 $H$ 的动作块

**符号说明**:
- $A_t = [a_t, \ldots, a_{t+H-1}] \in \mathbb{R}^{H \times 2d_a}$: 动作块，$d_a = 8$（七关节 + 夹爪）
- $x_r$: RGB 观测（三路摄像头）
- $x_t$: 热图像（单路）
- $l$: 语言指令
- $s_t$: 机器人关节状态

### 公式2: [[ThermoRes Encoder|热图分解]]

$$
\begin{aligned}
k &= \text{clip}(x_t - b + \mu,\ 0,\ I_\text{max}) \\
x_\text{cold} &= I_\text{max} \cdot \frac{\min(k, \mu)}{\mu} \\
x_\text{hot} &= I_\text{max} \cdot \frac{I_\text{max} - \max(k, \mu)}{I_\text{max} - \mu}
\end{aligned}
$$

**含义**: 以背景亮度 $b$ 为基准，将热图分解为低温（冷）视图和高温（热）视图，各自归一化到 $[0, I_\text{max}]$ 范围

**符号说明**:
- $x_t$: 原始热图灰度图
- $b$: 背景参考亮度（热图上边界 patch 均值的众数）
- $\mu$: 固定灰度强度（背景对应灰度值）
- $I_\text{max}$: 灰度范围上界
- $x_\text{cold}$: 冷属性视图（低于背景的部分放大到全范围）
- $x_\text{hot}$: 热属性视图（高于背景的部分放大到全范围）

### 公式3: [[Gated Residual Modulation|语言门控权重计算]]

$$
[\alpha, \beta] = \text{softmax}\!\left(\text{MLP}\!\left(\text{LN}(u_t)\right)\right)
$$

$$
u_t = [\bar{Z}_\text{text};\ \bar{O}_c;\ \bar{O}_h;\ \bar{O}_c - \bar{Z}_\text{text};\ \bar{O}_h - \bar{Z}_\text{text}]
$$

**含义**: 以语言 token 均值为查询，冷/热 token 均值及其与语言的差值为特征，经 MLP + softmax 计算动态门控权重，决定冷/热信息对 RGB 的贡献比例

**符号说明**:
- $\alpha, \beta$: 冷/热分支的动态权重（$\alpha + \beta = 1$）
- $u_t$: 拼接特征向量（维度为 $5d$）
- $\bar{Z}_\text{text}$: 语言 token 的 token-wise 池化均值
- $\bar{O}_c, \bar{O}_h$: 冷/热 token 的 token-wise 池化均值
- LN: Layer Normalization

### 公式4: [[Gated Residual Modulation|热残差注入]]

$$
Z_\text{rgb}^\text{cold} = Z_\text{rgb}^\text{head} + \alpha \cdot Z_\text{cold}, \quad
Z_\text{rgb}^\text{hot} = Z_\text{rgb}^\text{head} + \beta \cdot Z_\text{hot}
$$

**含义**: 将加权后的冷/热属性 token 以残差形式叠加到 RGB head-camera token 上，原始 RGB 表示作为恒等路径保留

**符号说明**:
- $Z_\text{rgb}^\text{head}$: RGB head-camera token（恒等路径）
- $Z_\text{cold}, Z_\text{hot}$: 冷/热属性 token（已过投影层）
- $\alpha, \beta$: 由语言条件决定的门控权重

### 公式5: [[Gated Residual Modulation|属性残差通用形式]]

$$
y = x + f(a, l)
$$

**含义**: 通用残差注入公式——原始 RGB 表示 $x$ 加上由属性观测 $a$ 和语言条件 $l$ 共同决定的残差修正量 $f(\cdot)$，该形式与 [[Residual Adapter|残差适配器]] 结构同构

**符号说明**:
- $x$: 原始 RGB 视觉表示
- $a$: 属性观测（冷/热分支，或颜色分支）
- $l$: 语言条件
- $f(\cdot)$: 属性条件残差函数

### 公式6: [[RGBRes Encoder|RGBRes 颜色属性残差注入]]

$$
\begin{aligned}
Z_\text{rgb}^\text{red} &= Z_\text{rgb}^\text{head} + \alpha' \cdot Z_\text{red} \\
Z_\text{rgb}^\text{green} &= Z_\text{rgb}^\text{head} + \beta' \cdot Z_\text{green} \\
Z_\text{rgb}^\text{blue} &= Z_\text{rgb}^\text{head} + \gamma \cdot Z_\text{blue}
\end{aligned}
$$

**含义**: 对 RGB 三通道颜色属性各自独立地进行语言门控残差注入，机制与 ThermoRes 完全一致

**符号说明**:
- $\alpha', \beta', \gamma$: 红/绿/蓝三路门控权重（由语言条件计算）
- $Z_\text{red}, Z_\text{green}, Z_\text{blue}$: 颜色属性 token（经颜色编码器 + 投影层）

### 附：RGB token 投影（无编号）

$$
Z_\text{rgb}^i = P_r(E_r(x_r^i)), \quad i \in \{\text{head, left, right}\}
$$

$$
Z_\text{cold} = P_r(E_t(x_\text{cold})), \quad Z_\text{hot} = P_r(E_t(x_\text{hot}))
$$

### 附：[[Flow-Matching|流匹配]]插值（无编号）

$$
x_\tau = (1 - \tau) \cdot A_t + \tau \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: [[Flow-Matching|流匹配]]训练时的插值采样，$\tau$ 为流匹配时间步

---

## 关键图表

### Figure 1: Thermo-VLA 整体框架与性能概览

![Figure 1](https://arxiv.org/html/2611.11981v1/head.png)

**说明**: 左侧展示真实世界 RGB 和热图像输入（[[Unitree G1]] 双臂机器人）；中间为 [[ThermoRes Encoder]] + [[Flow-Matching|流匹配]] 动作生成的整体框架；右侧为六任务成功率对比图，展示 Thermo-VLA 在热感知和热推理任务上的显著优势，Task VI 为异常动物笼子移除的示例演示。

### Figure 2: ThermoRes Encoder 详细架构

![Figure 2](https://arxiv.org/html/2611.11981v1/pipeline1.png)

**说明**: 展示 [[ThermoRes Encoder]] 的完整数据流。热图像经背景估计后分解为冷/热双分支属性视图，共享热编码器 $E_t$ 编码后经投影层与 RGB token 对齐，再由[[Gated Residual Modulation|语言门控残差注入模块]]（文本条件交叉注意力 + MLP + softmax）将加权热 token 以残差形式叠加到 RGB head-camera tokens 上，最终与语言 token 和机器人状态 token 拼接，送入 [[π0.5]] Gemma 骨干 + 流匹配动作专家。

### Figure 3: 典型成功与失败执行对比

![Figure 3](https://arxiv.org/html/2611.11981v1/result1.png)

**说明**: 左侧展示 Thermo-VLA 成功执行的典型样例（温度感知抓取、温度排序服务等）；右侧展示竞品 RGB-热方法的典型失败案例；底行展示 Thermo-VLA 的 [[Cross-Attention|注意力图]]，可见模型在温度依赖任务中注意力集中在热目标区域，而在外观任务中关注颜色/形状特征。

### Figure 4: 泛化测试场景与执行示例

![Figure 4](https://arxiv.org/html/2611.11981v1/result2.png)

**说明**: 展示五类泛化设置——未见瓶子、未见对象、温度重分配（已见对象的冷热对调）、光照变化、未见桌布——以及 Thermo-VLA 在每类设置下的成功执行样例。温度重分配场景下模型性能下降最明显（依赖颜色-温度映射的记忆倾向）。

### Figure 5: RGBRes Encoder 架构与颜色任务场景

![Figure 5](https://arxiv.org/html/2611.11981v1/pipeline_small.png)

**说明**: 左侧为 [[RGBRes Encoder]] 的流程图（与 ThermoRes 结构对称，将冷/热分支替换为红/绿/蓝三路颜色属性）；右侧展示 Task VII（LIBERO 仿真颜色条件任务）和 Task VIII（真实世界颜色条件操作）的场景图。

---

### Table 1: 数据集规格

| 任务编号 | 任务名称 | 难度类型 | 演示数量 |
|----------|----------|----------|----------|
| I | 位置引导瓶子选择 | Spatial Grounding | 156 |
| II | 温度引导瓶子选择 | Thermal Perception | 303 |
| III | 外观引导物体选择 | Appearance Grounding | 236 |
| IV | 温度引导物体选择 | Thermal Grounding | 340 |
| V | 温度感知咖啡服务 | Thermal Ranking | 173 |
| VI | 异常动物笼子移除 | Thermal Exclusion | 99 |
| **合计** | — | — | **1,307** |

**说明**: 六类任务由简到难覆盖了空间定位、外观识别、热感知（单物体冷/热识别）、热推理（相对温度排序与异常排查），仅 Task II 为纯热任务，其余均含视觉-热联合信息。

---

### Table 2: 六任务真实世界操控成功率

> 列头：RGB ｜ RGBT ｜ ACT ｜ π0.5 ｜ π0.5(RGBT) ｜ ThermoAct ｜ **Ours**
> `—` = 该任务类型不适用；n/a = 方法不支持

**Task I: 位置引导瓶子选择（Spatial Grounding）**

| 指令 | RGB | ACT | π0.5 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----|-----|------|-----------|-----------|---------|
| 左瓶 → 黑色托盘 | 0.75 | 0.95 | 0.95 | 0.80 | — | **0.95** |
| 右瓶 → 黑色托盘 | 0.75 | 0.90 | 0.75 | 0.10 | — | **0.80** |
| **平均** | 0.75 | 0.93 | 0.85 | 0.45 | — | **0.88** |

**Task II: 温度引导瓶子选择（Thermal Perception）**

| 指令 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----------|-----------|---------|
| 冷瓶 → 黑色托盘 | 0.80 | 0.45 | **1.00** |

**Task III: 外观引导物体选择（Appearance Grounding）**

| 指令 | RGB | ACT | π0.5 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----|-----|------|-----------|-----------|---------|
| 牛奶 → 黄色托盘 | 0.65 | 0.85 | 0.75 | 0.55 | — | **0.80** |
| 牛奶 → 白色托盘 | 0.75 | 0.85 | 0.70 | 0.45 | — | **0.85** |
| 可乐 → 黄色托盘 | 0.65 | 0.80 | 0.70 | 0.45 | — | 0.70 |
| 可乐 → 白色托盘 | 0.70 | 0.85 | 0.70 | 0.50 | — | 0.70 |
| **平均** | 0.69 | 0.84 | 0.71 | 0.49 | — | **0.76** |

**Task IV: 温度引导物体选择（Thermal Grounding）**

| 指令 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----------|-----------|---------|
| 热物体 → 黄色托盘 | 0.60 | 0.85 | **0.90** |
| 热物体 → 白色托盘 | 0.55 | 0.90 | **1.00** |
| 冷物体 → 黄色托盘 | 0.50 | 0.75 | **0.70** |
| 冷物体 → 白色托盘 | 0.55 | 0.90 | **0.85** |
| **平均** | 0.55 | 0.85 | **0.86** |

**Task V: 温度感知咖啡服务（Thermal Ranking）**

| 指令 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----------|-----------|---------|
| 糖 → 少冰 | 0.00 | 0.50 | **0.60** |
| 糖 → 多冰 | 0.00 | 0.50 | **0.55** |
| **平均** | 0.00 | 0.50 | **0.58** |

**Task VI: 异常动物笼子移除（Thermal Exclusion）**

| 指令 | π0.5(RGBT) | ThermoAct | **Ours** |
|------|-----------|-----------|---------|
| 死鼠笼子 → 危险箱 | 0.50 | 0.55 | **1.00** |

**关键发现**: 四项热依赖任务（II/IV/V/VI）加权平均成功率 Ours = **0.86**，比 π0.5(RGBT) 高 **+18pp**。Task VI 中 Thermo-VLA 比次优方法高出 **45pp**（1.00 vs 0.55）。

---

### Table 3: 泛化性能（成功率）

> 三组数值对应：ThermoAct ｜ π0.5(RGBT) ｜ **Ours**

| 任务 | 已见对象 | 未见对象 | 光照变化 | 未见桌布 |
|------|----------|----------|----------|----------|
| Task II | — / — / — | 0.55 / 0.85 / **1.00** | 0.50 / 0.70 / **0.90** | 0.45 / 0.80 / **0.90** |
| Task IV | 0.50 / 0.10 / **0.60** | 0.45 / 0.70 / **0.73** | 0.55 / 0.85 / **0.83** | 0.58 / 0.85 / **0.88** |
| Task V | — / 0.00 / 0.50 / 0.58 | 0.00 / 0.50 / **0.55** | 0.00 / 0.50 / **0.58** | — |
| Task VI | — / 0.50 / 0.50 / 0.85 | — | — | — |
| **平均** | 0.50 / 0.10 / **0.60** | 0.38 / 0.64 / **0.79** | 0.35 / 0.68 / **0.76** | 0.34 / 0.72 / **0.79** |

**关键发现**: 在未见对象场景，Thermo-VLA 平均 0.79，比 ThermoAct 高 **+41pp**，比 π0.5(RGBT) 高 **+15pp**。温度重分配（已见对象）场景下，模型相比其他扰动类型性能下降最多，表明存在颜色-温度映射的记忆偏差。

---

### Table 4: 消融实验

| 配置 | Task II | Task IV (avg) | Task II 光照 | Task IV 桌布 | Task VI 未见对象 |
|------|---------|--------------|-------------|-------------|----------------|
| Thermo-VLA（完整） | **1.00** | **0.86** | **0.90** | **0.88** | **0.85** |
| w/o 门控残差注入 | 1.00 | 0.80 | 0.90 | 0.85 | 0.60 |
| w/o 双分支分解 | 1.00 | 0.85 | 0.85 | 0.80 | 0.85 |
| w/o 两者（单路热输入） | 0.80 | 0.85 | 0.70 | 0.85 | 0.50 |
| 热残差 = 0（无热输入） | 0.50 | 0.45 | 0.50 | 0.45 | 0.50 |

**关键发现**:
- **门控残差注入**对小目标精细任务（Task VI 0.85 vs 0.60）和泛化场景（光照、桌布）影响最大
- **双分支分解**对 Task II（单一冷瓶识别）影响较小，但对光照和桌布泛化有贡献
- **热残差归零**使所有热任务成功率骤降至 ~0.5，证明热信息是必要的

---

### Table 5: 颜色条件任务成功率（RGBRes 扩展）

| 方法 | Task VII (仿真) | Task VII (仿真-桌布扰动) | Task VIII (真实) | Task VIII (真实-桌布扰动) |
|------|----------------|--------------------------|-----------------|--------------------------|
| π0.5 | 0.707 | 0.307 | 0.600 | 0.675 |
| π0.5 + [[RGBRes Encoder]] | **0.873** | **0.453** | **0.900** | **0.875** |

**关键发现**: [[RGBRes Encoder]] 在四项颜色条件任务上将平均成功率从 0.572 提升到 **0.775**（+20.3pp），验证残差注入机制在颜色属性上的可迁移性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自建真实世界 Thermo-VLA | 1,307 条专家演示 | 六类任务，双臂，含热/RGB 配对 | 训练 + 测试（20次/任务） |
| LIBERO（仿真） | 标准 LIBERO 数据集 | 颜色条件操控，模拟器环境 | Task VII RGBRes 验证 |

### 实现细节

- **基础模型**: [[π0.5]]（Gemma-based VLA + [[Flow-Matching|流匹配]]动作专家）
- **视觉编码器**: [[SigLIP]]（RGB 和热流共享结构，热编码器独立初始化）
- **动作专家**: 继承 π0.5 流匹配权重，不重新训练
- **硬件平台**: [[Unitree G1]] 双臂人形机器人，配备 RGB 相机（头部+双腕）和热成像相机
- **每任务测试**: 20 次独立试验
- **训练细节**: 未在摘要中披露（无开源代码）

### 可视化结果

- 注意力图（Figure 3 底行）显示模型在温度任务中注意力正确聚焦在热目标上，在外观任务中关注颜色/形状，表明[[Gated Residual Modulation|门控机制]]确实实现了任务条件下的属性选择性
- Task VI（异常排查）的案例展示了 Thermo-VLA 能够通过"排除非目标（非异常热源）"完成复杂温度推理

---

## 批判性思考

### 优点

1. **设计优雅**：残差注入的"恒等路径"设计使热/颜色属性以"可选加成"而非"强制替换"的方式注入，理论上不损伤原始 RGB 能力，且对 [[Residual Adapter|残差适配器]] 范式有良好对应
2. **泛化性强**：同一残差机制可直接迁移到颜色属性（RGBRes），证明方法不依赖热模态的特殊性
3. **真实世界验证充分**：1,307 条演示、六类任务、五类泛化扰动，测试粒度细致

### 局限性

1. **热编码器预训练不足**：作者自承热编码器在通用热图数据集上的预训练规模有限，限制了对细粒度温差的区分能力（如"哪个更热一点"）
2. **温度大小信息缺失**：当前分解只有冷/热二值化分支，无法感知实际温差幅度，对需要精确温度量化的任务力不从心
3. **平台单一**：仅在 Unitree G1 上验证，跨平台迁移能力未知

### 潜在改进方向

1. **热编码器大规模预训练**：在开放热图数据集上预训练 $E_t$，提升热语义理解
2. **温度幅度分支**：引入连续温度回归分支，支持"哪个更热X度"的精确推理
3. **扩展到深度/触觉等其他模态**：残差注入框架理论上可推广到任意"属性补充模态"

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [ ] 训练细节完整（部分缺失：学习率、batch size、epochs 未披露）
- [x] 数据集可获取（真实世界数据集为自建，不可获取；LIBERO 为公开数据集）

---

## 关联笔记

### 基于

- [[π0.5]]: 基础 VLA 骨干（Gemma + 流匹配动作专家）
- [[SigLIP]]: RGB 和热流共享的视觉编码器结构
- [[Flow-Matching]]: 动作块生成的核心生成方法

### 对比

- [[ThermoAct]]: 本文主要竞品，以不同方式将热图输入 VLA
- [[π0.5]]: 仅用 RGB 的基线；π0.5(RGBT) 为简单 RGBT 拼接基线

### 方法相关

- [[ThermoRes Encoder]]: 本文核心热感知模块
- [[Gated Residual Modulation]]: 语言条件门控残差注入机制
- [[RGBRes Encoder]]: 热残差机制在颜色属性上的泛化扩展
- [[Residual Adapter]]: 结构同构的通用残差注入范式
- [[Gated Cross-Attention]]: 门控权重计算中使用的注意力变体
- [[Action Chunking]]: 动作块输出格式
- [[VLA（视觉-语言-动作模型）]]: 方法所属的模型类别

### 硬件/数据相关

- [[Unitree G1]]: 实验平台（双臂人形机器人）
- [[Thermal Camera]]: 热成像传感器，提供 $x_t$ 输入

---

## 速查卡片

> [!summary] Thermo-VLA (arXiv 2611.11981)
> - **核心**: 通过 ThermoRes 双分支热分解 + 语言门控残差注入，将热感知能力以"加法残差"注入 π0.5 VLA
> - **方法**: 冷/热属性分解 → 共享 SigLIP 编码 → softmax 门控权重 → 残差叠加到 RGB head token
> - **结果**: 四热任务平均 0.86（+18pp vs π0.5(RGBT)）；泛化未见对象 0.79；RGBRes 颜色任务 0.775（+20pp）
> - **代码**: 未开源（项目主页：https://j-x-chen.github.io/Thermo-VLA-web/）

---

*笔记创建时间: 2026-11-11*

---
title: "Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models"
method_name: "Qwen-RobotManip"
authors: [Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixuan Lei, Tong Zhang, Jiazhao Zhang, Jie Zhang, Jingyang Fan, Gengze Zhou, Qihang Peng, Chenxu Lv, Xiaoyue Chen, An Yang, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou, Chenfei Wu, Xiong-Hui Chen]
year: 2026
venue: arXiv
tags: [vla, cross-embodiment, flow-matching, robotic-manipulation, alignment, foundation-model, diffusion-transformer]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/abs/2606.17846
created: 2026-07-01
---

# 论文笔记：Qwen-RobotManip Technical Report

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team（Alibaba / 阿里巴巴） |
| 日期 | June 2026 |
| 项目主页 | [qwen.ai/blog?id=qwen-robotmanip](https://qwen.ai/blog?id=qwen-robotmanip) |
| 对比基线 | [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.17846) / [Code](https://github.com/QwenLM/Qwen-RobotManip) |

---

## 一句话总结

> Qwen-RobotManip 通过表示、运动、行为三维对齐统一异构机器人数据，在 ~38,100 小时开源数据上训练，使 VLA 首次展现出清晰的对数线性数据扩展规律，并在多个 OOD 基准和 RoboChallenge 上超越 π0.5 等 SOTA。

---

## 核心贡献

1. **三维对齐框架（Alignment Before Scale）**: 提出跨具身的表示对齐（80D 规范状态-动作空间）、相机坐标系运动对齐（Camera-Frame Delta Pose + [[Camera Positional Encoding|CaPE]]）、以及行为对齐（结构化具身提示 + 上下文适配），使异构数据不再产生梯度冲突
2. **Human-to-Robot 合成管线**: 将 1,933 小时的自我中心人类视频跨 15 种机器人平台转化为 24,808 小时机器人演示数据，无需任何专有数据采集
3. **首个清晰数据扩展律 VLA**: 实验证明只有在统一对齐下，[[Flow Matching]] 训练才能实现 log-linear 数据扩展，OOD 泛化性能随数据量单调提升

---

## 问题背景

### 要解决的问题

机器人操控数据天然是**异构的**：不同机器人平台有不同关节空间、不同末端执行器坐标系、不同数据采集协议，导致大规模联合训练时数据间产生冲突而非协同。

### 现有方法的局限

以 [[π0.5]] 为代表的大规模 VLA 方法虽尝试多源数据训练，但由于缺乏统一的跨具身对齐，增加数据量有时反而降低性能——数据多样性与模型对齐能力之间存在根本矛盾。

### 本文的动机

**"对齐解锁扩展"**——在对齐到位之前，扩展数据规模是有害的；一旦实现统一对齐，模型才能真正从海量开源数据中受益，实现泛化能力的跃升。

---

## 方法详解

### 模型架构

Qwen-RobotManip 采用 **VLM + 动作专家双流** 架构：

- **输入**: 语言指令 + 多视角图像观测 + 结构化具身提示（机器人平台、执行速度、FPS）+ 历史观测-动作块（上下文）
- **Backbone**: [[Qwen3-VL|Qwen3.5-4B]] 视觉-语言主干（感知与推理）
- **动作专家**: [[Diffusion Transformer|Flow-Matching Diffusion Transformer (DiT)]]，通过交叉注意力与 VLM 隐状态交互
- **输出**: 80D 规范动作空间下的末端执行器增量预测（[[Action Chunking|动作块]]）
- **推理优化**: [[Real-Time Chunking]]（RTC），$K_{\text{repeat}}=8$ 步扩散采样，实现无缝闭环控制

### 核心模块

#### 模块1：规范状态-动作表示（Canonical State-Action Representation）

**设计动机**: 不同机器人平台的自由度（DOF）差异导致直接联合训练时梯度冲突。

**具体实现**:
- 将所有机器人状态/动作统一映射到 **80 维向量**，涵盖：
  - 每条手臂 7 维关节角位置
  - 每条手臂 9 维末端执行器位姿（3D 位置 + 6D 旋转）
  - 每条手臂 1 维夹爪状态
  - 每条手臂 12 维灵巧手关节
  - 保留维度用于扩展
- **逐维二值掩码**（per-dimension binary mask）：缺失 DOF 对应位置置零并从损失中排除，确保梯度只流经有效维度
- 支持单臂、双臂、灵巧手、移动底盘等所有构型

#### 模块2：相机坐标系运动对齐（Camera-Frame Motion Alignment）

**设计动机**: 同一视觉场景下的相似操作动作，因不同机器人的基坐标系不同，在机器人坐标系中数值差异巨大，难以统一学习。

**具体实现**:
- 将末端执行器动作表示为**相机坐标系下的增量位姿**（delta EEF pose in camera frame），而非机器人基坐标系
- 引入 **[[Camera Positional Encoding|CaPE（Camera Positional Encoding）]]**，将相机外参和内参几何信息注入动作专家的交叉注意力层
- 不同机器人的视觉相似动作在数值上也变得相近，实现真正的跨具身迁移

#### 模块3：行为对齐（Behavior Alignment）

**设计动机**: 不同数据源的操控行为风格差异（执行速度、控制频率）以及策略训练时的平凡上下文复制问题。

**具体实现**:
- **结构化具身提示**（Structured Embodiment Prompt）：为每条数据指定机器人平台、执行速度、FPS 等元信息，作为条件输入
- **上下文策略适配**（In-Context Policy Adaptation）：将近期的观测-动作块作为上下文条件化动作生成，并在训练时使用随机上下文采样（Stochastic Context Sampling），防止策略简单复制上下文中的历史动作

#### 模块4：双流联合训练（Dual-Stream Co-Training）

- VLA 操控数据 与 精心筛选的 VLM 视觉语言数据按 **9:1 比例** 联合训练
- 防止 VLM 主干在动作预测压力下丧失感知和推理能力

---

## 关键公式

### 公式1：[[Human-to-Robot Retargeting|人手到夹爪的重定向]]

虚拟手指位置（加权融合食指与中指）：

$$
k_{\text{vf}} = 0.7\, k_{\text{index}} + 0.3\, k_{\text{middle}}
$$

末端执行器位置与夹爪宽度：

$$
p = \frac{1}{2}(k_{\text{thumb}} + k_{\text{vf}}), \quad w = \|k_{\text{thumb}} - k_{\text{vf}}\|_2
$$

**含义**: 将人手关键点（拇指 + 食指/中指加权）映射到夹爪的抓握位置和开合宽度

**符号说明**:
- $k_{\text{index}}, k_{\text{middle}}, k_{\text{thumb}}$: 食指、中指、拇指指尖关键点位置
- $k_{\text{vf}}$: 虚拟手指位置
- $p$: 末端执行器中心位置
- $w$: 夹爪张开宽度

### 公式2：[[Camera Positional Encoding|相机坐标系动作表示]]

$$
a_p = \begin{bmatrix} {}^c_e R\, {}^e_{e'} R\, {}^e_c R & {}^c_e R(t_{ee'} - t_{ec}) \\ \mathbf{0}^\top & 1 \end{bmatrix}
$$

**含义**: 末端执行器在相机坐标系下的位姿增量，使跨机器人的相似视觉运动在数值上对齐

**符号说明**:
- ${}^c_e R$: 末端执行器到相机的旋转矩阵
- ${}^e_{e'} R$: 末端执行器当前到目标的旋转增量
- $t_{ee'}$: 末端执行器位移
- $t_{ec}$: 相机到末端执行器的平移

### 公式3：[[Flow Matching|Flow Matching 损失]]

$$
\mathcal{L}_{\text{FM}} = \frac{1}{B} \sum_{i=1}^{B} \frac{\sum_{t,j} m_{i,t,j} \left(f_\theta(x_{i,t}, t_i, s_i, o_i)_j - v_{i,t,j}\right)^2}{\sum_{t,j} m_{i,t,j}}
$$

其中插值带噪动作为：

$$
x_t = (1 - t)\varepsilon + t\, a
$$

**含义**: 带逐维掩码的 Flow Matching 目标，让网络预测从噪声 $\varepsilon$ 到动作 $a$ 的速度场，掩码 $m$ 排除缺失 DOF 对应的维度

**符号说明**:
- $B$: 批大小
- $f_\theta$: DiT 动作专家
- $x_{i,t}$: 时间步 $t$ 处的带噪动作
- $v_{i,t,j}$: 目标速度场（$= a - \varepsilon$）
- $m_{i,t,j}$: 逐维掩码（1 表示有效，0 表示该 DOF 缺失）
- $s_i$: 规范状态向量
- $o_i$: 视觉观测

### 公式4：[[Dual-Stream Co-Training|双流联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{VLM}}
$$

**含义**: 同时优化动作预测损失和 VLM 语言建模损失，防止操控训练侵蚀视觉语言理解能力

**符号说明**:
- $\mathcal{L}_{\text{FM}}$: Flow Matching 动作损失
- $\mathcal{L}_{\text{VLM}}$: 视觉语言建模损失
- $\lambda$: 权重系数（数据比例为 VLA:VLM = 9:1）

---

## 关键图表

### Figure 1: 整体架构概览（Method Overview）

![Figure 1: Method Overview](https://yqintl.alicdn.com/006d336a178877315d255d0ff785fe6efb648971.png)

**说明**: Qwen-RobotManip 的整体框架。[[Qwen3-VL|Qwen3.5-4B VLM 主干]] 处理语言指令和多视角图像，输出视觉语言隐状态；[[Diffusion Transformer|Flow-Matching DiT]] 动作专家通过交叉注意力接收这些特征，并以规范状态 $s$、具身提示和历史上下文为条件，预测 80D 规范动作空间下的末端执行器增量。

### Figure 2: Human-to-Robot 合成管线

![Figure 2: H2R Pipeline](https://yqintl.alicdn.com/13da180b7351860193c685cff54fd556a8000d76.png)

**说明**: 将自我中心人类视频转化为机器人演示的四阶段管线：(1) **动作重定向**——将人手轨迹映射到夹爪动作；(2) **手部去除与修复**——用 SAM3 分割手部区域并以 ProPainter 修复背景；(3) **仿真渲染**——在 MuJoCo 中渲染 15 种双臂机器人平台；(4) **深度引导合成**——将渲染机器人按深度融合回真实场景。总计将 1,933 小时人类视频扩展为 24,808 小时机器人演示。

### Figure 3: 数据预处理管线

![Figure 3: Data Preprocessing](https://yqintl.alicdn.com/0c468b0513bd911c868eebde366ad5d6f2c63b64.png)

**说明**: 五阶段质量过滤（突变检测、状态-动作趋势对齐、极值过滤、正向运动学一致性、基坐标系与末端姿态对齐）+ 三阶段跨模态一致性校验（指令-视频一致性、视频-状态对应性、视频质量评估），确保 38,100 小时数据的质量。

### Figure 4: 评估设置总览

![Figure 4: Evaluation Settings](https://yqintl.alicdn.com/97d7a58790244f3c3236ec29945d23aaed1946bf.png)

**说明**: 论文采用 OOD（分布外）评估作为基础模型能力的核心衡量标准，涵盖 LIBERO-Plus、RoboTwin-Clean2Rand、EBench、RoboCasa365、RoboTwin-IF、RoboTwin-XE 等六大 OOD 基准，以及 RoboChallenge 竞赛（30 个任务跨 4 个机器人平台）和真实 ALOHA 平台测试。

### Figure 5: OOD 泛化性能汇总

![Figure 5: OOD Summary](https://yqintl.alicdn.com/a3d9c609ee7c022ba8392320f63d36a9b5cae66e.png)

**说明**: Qwen-RobotManip 在所有 OOD 基准上全面超越 [[π0.5]]，其中 RoboCasa365 高达 3× 改进，RoboTwin-XE 跨具身迁移达 3.2× 改进。

### Figure 6: 数据扩展规律

![Figure 6: Data Scaling](https://yqintl.alicdn.com/d0d5c3230f0aaabcff0a783dd0ad6aedb681459a.png)

**说明**: 展示对齐是实现数据扩展律的前提。在三维对齐框架下，OOD 性能随预训练数据量呈清晰的 **log-linear** 增长曲线；而没有统一对齐时，增加数据量会因梯度冲突导致性能下降。

### Figure 7: RoboChallenge 真实机器人测试

![Figure 7: RoboChallenge Results](https://yqintl.alicdn.com/c288de9ed0ce8299ddc996d062db61d869e1043b.png)

**说明**: Qwen-RobotManip 在 RoboChallenge Table30 v1 上排名第 1，在 AgileX ALOHA、Franka、UR、ARX 四个平台上以 45% 成功率和 59.83 过程分完成 30 个任务，相对第三名提升 20%。

### Figure 8: 双臂操控与跨具身测试

![Figure 8: Bimanual and Pick-and-Place](https://yqintl.alicdn.com/dc1cde503a39cf4377b97690978c1eb740f1899d.png)

**说明**: 在 AgileX ALOHA 双臂协作任务上（域内 88.6%、域外 87.5%），以及 ARX ALOHA 跨具身迁移上（55.0%，4× 消融基线），体现了 [[In-Context Policy Adaptation|上下文策略适配]] 带来的泛化增益。

### Table 1: 域内基准性能

| Model | LIBERO SR (%) | RoboTwin Easy SR (%) | RoboTwin Hard SR (%) |
|-------|--------------|---------------------|---------------------|
| 基线方法 | ~85 | ~80 | ~78 |
| Qwen-RobotManip | **99.1** | **93.4** | **92.5** |
| Qwen-RobotManip-Context | **99.2** | **93.7** | **94.0** |

**说明**: 域内测试中 Qwen-RobotManip 达到近乎饱和的性能（99.1% LIBERO），Context 版本（加入上下文适配）进一步提升 Hard 场景性能。

### Table 2: OOD 泛化性能对比

| Benchmark | Qwen-RobotManip | Qwen-RobotManip-Context | π0.5 |
|-----------|----------------|------------------------|------|
| LIBERO-Plus Overall SR (%) | 89.0 | **91.4** | 84.4 |
| RoboTwin-C2R Hard SR (%) | 62.6 | **69.4** | 47.9 |
| EBench Overall SR (%) | **45.6** | 43.6 | 27.1 |
| RoboCasa365 Total SR (%) | **35.9** | 33.8 | 16.9 |
| RoboTwin-IF Avg SR (%) | **72.2** | 72.0 | 49.6 |
| RoboTwin-XE Avg SR (%) | **23.9** | — | 7.5 |

**关键发现**: 在所有 6 个 OOD 基准上超越 π0.5，RoboCasa365（未见场景）提升约 2×，RoboTwin-XE（零样本跨具身）达 3.2× 改进

### Table 3: 真实机器人测试结果

| 平台 | 任务类型 | Qwen-RobotManip | π0.5 |
|------|---------|----------------|------|
| AgileX ALOHA | 域内双臂操控 | **88.6%** | 42.9% |
| AgileX ALOHA | 域外双臂操控 | **87.5%** | 37.5% |
| ARX ALOHA | 跨具身技能迁移 | **55.0%** | — |
| RoboChallenge Table30 v1 | 30任务综合（4平台） | **45%（排名第1）** | — |

---

## 数据集与实验

### 数据集

| 数据来源 | 规模 | 特点 | 用途 |
|---------|------|------|------|
| 开源机器人数据集（OXE 等） | ~11,420 小时 | 9 个来源，单臂/双臂/移动/人形 | 预训练 |
| 自我中心人类视频 | 1,933 小时 | EPIC-KITCHENS 等 | H2R 合成原料 |
| H2R 合成数据 | 24,808 小时 | 15 个双臂平台，视觉+动作对齐 | 预训练 |
| 视觉-语言数据混合 | 28M 条 | 通用 VLM 数据 | 联合训练 |
| **总计** | **~38,100 小时** | 全部开源，无专有采集 | — |

开源机器人数据来源包括：OXE、AgiBotWorld-Beta、RoboMIND、DROID、RH20T。

### 实现细节

- **VLM Backbone**: Qwen3.5-4B（感知+推理主干）
- **动作专家**: Flow-Matching DiT（带 VLM 交叉注意力）
- **动作空间**: 80D 规范末端执行器位姿，相机坐标系增量
- **联合训练比**: VLA:VLM = 9:1
- **推理**: Real-Time Chunking，$K_{\text{repeat}}=8$ 扩散步
- **真实平台**: AgileX ALOHA、Franka Research 3、UR、ARX

### 数据质量管线

**五阶段过滤**:
1. 突变检测（sudden change detection）
2. 状态-动作趋势对齐校验
3. 极值过滤
4. 正向运动学一致性验证
5. 基坐标系与末端姿态方向对齐

**三阶段跨模态一致性校验**:
1. 指令-视频一致性
2. 视频-状态对应性
3. 视频质量评估

---

## 批判性思考

### 优点

1. **理论清晰，实践有力**: "对齐解锁扩展"这一核心论点有严格的消融实验支持（log-linear 扩展律），不仅是工程技巧，更是对 VLA 训练范式的深刻洞察
2. **数据效率极高**: 完全使用开源数据，H2R 合成管线将 1,933 小时人类视频放大至 24,808 小时机器人数据（12.8×），大幅降低数据采集成本
3. **真实部署验证充分**: RoboChallenge 在 4 个真实平台 30 个任务上排名第 1，不仅是仿真基准的 SOTA，更是真实世界验证

### 局限性

1. **模型权重不开放**: 论文明确表示暂无计划开放 Qwen-RobotManip 或 Qwen-RobotNav 的模型权重，社区复现困难
2. **依赖精确相机标定**: Camera-Frame Motion Alignment 需要准确的相机外参，实际部署时标定误差可能影响性能
3. **合成数据质量上限**: H2R 管线虽然规模大，但合成数据与真实机器人数据之间仍存在外观/动力学差距（sim-to-real gap）
4. **灵巧手支持有限**: 虽然 80D 表示理论上支持灵巧手（12 维/臂），但实验主要集中在夹爪式机器人

### 潜在改进方向

1. 将 CaPE 机制扩展到力/触觉传感器，实现力感知的跨具身对齐
2. 结合在线 RL 微调，利用扩展数据预训练的基础进一步提升任务成功率
3. 探索更高效的 H2R 合成管线，进一步提升视觉保真度

### 可复现性评估

- [ ] 代码开源（基础代码已开源但不完整）
- [ ] 预训练模型（明确表示暂不开放）
- [ ] 训练细节完整（技术报告中有较完整描述）
- [x] 评估基准可获取（LIBERO、RoboTwin 等均为公开基准）

---

## 关联笔记

### 基于

- [[π0.5]]: 主要对比基线，Qwen-RobotManip 在所有 OOD 指标上超越 π0.5
- [[QwenVL|Qwen-VL]]: VLM 主干的前身
- [[Qwen3-VL]]: 实际使用的 Qwen3.5-4B VLM 骨干

### 对比

- [[π0.5]]: Physical Intelligence 的 VLA，两者都做大规模多数据源训练，但对齐策略不同
- [[Cross-Embodiment]]: 跨具身方法对比

### 方法相关

- [[Flow Matching]]: 动作专家的核心训练目标
- [[Diffusion Transformer]]: DiT 动作专家架构
- [[Real-Time Chunking]]: 推理时使用的实时分块技术
- [[Action Chunking|动作分块]]: 基础动作预测框架
- [[Camera Positional Encoding|CaPE]]: 相机几何信息注入机制

### 硬件/数据相关

- [[Franka Research 3]]: 实验使用的机械臂平台之一

---

## 速查卡片

> [!summary] Qwen-RobotManip (2026)
> - **核心**: 三维对齐（表示/运动/行为）解锁 VLA 数据扩展律
> - **方法**: Qwen3.5-4B VLM + Flow-Matching DiT + 80D 规范状态 + CaPE 相机帧动作 + H2R 合成管线
> - **结果**: RoboChallenge 第1（45%），OOD 全面超越 π0.5（最高 3.2×），38,100小时开源数据
> - **代码**: [github.com/QwenLM/Qwen-RobotManip](https://github.com/QwenLM/Qwen-RobotManip)（权重暂不开放）

---

*笔记创建时间: 2026-07-01*

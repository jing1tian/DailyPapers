---
title: "ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation"
method_name: "Omega0"
authors: [Zhe Li, Zhenzhe Zhang, Yangyang Wei, Wenjie Zhang, Xichen Yuan, Peiyuan Zhi, Gen Li, Xinying Guo, Fengjie Gao, Jianfei Yang, Shanghang Zhang]
year: 2025
venue: arXiv
tags: [world-action-model, humanoid, loco-manipulation, latent-prediction, diffusion-policy]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.06375
created: 2026-08-08
---

# 论文笔记：ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MARS Lab (NTU)、北京大学 (PKU)、BAAI、香港科技大学（广州）HKUST(GZ) |
| 日期 | August 2025 |
| 项目主页 | [gentlefress.github.io/OMEGA-0_page](https://gentlefress.github.io/OMEGA-0_page/) |
| 对比基线 | [[Fast-WAM]]、[[EgoVLA]]、[[ψ-0\|psi-0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.06375) / Code: N/A |

---

## 一句话总结

> ω-0 通过轻量级潜在视觉预测与扩散式全身动作生成的协同训练，实现了单一模型在11个家庭任务上的并发人形机器人运动-操作，成功率达81.8%，大幅超越所有基线。

---

## 核心贡献

1. **[[Latent Predictive World Model|潜在预测世界-动作模型]]**: 不重建完整未来视频帧，而是预测紧凑的未来观测嵌入，与[[Action Diffusion|扩散式全身动作]]生成耦合，降低计算开销同时保留任务进度感知能力
2. **ω-HOME 数据集**: 40+ 小时、4,827 集、24 个家庭任务的多模态人形机器人演示数据集，覆盖自我中心 RGB、外中心 RGB-D、[[SMPL]] 运动参考和全身动作潜变量
3. **三阶段训练流水线**: 全身 VLM 预训练 → 人类到人形机器人动作潜变量预训练 → 实时组块（[[Real-Time Chunking|RTC]]）真实环境微调，统一解决运动和操作的协调问题

---

## 问题背景

### 要解决的问题

并发人形机器人"运动-操作"（[[Loco-Manipulation]]）要求腿部运动与手臂操作同步协调，现有方法难以生成与低级控制器接口兼容的**全身**动作指令，同时保持长时域任务一致性。

### 现有方法的局限

- **经典 IL（[[BC]]、[[Action Diffusion]]）**: 动作接口不支持统一全身控制，缺乏对长时域操作的泛化
- **VLA 模型（[[EgoVLA]]、[[InternVLA]]）**: 动作接口未为全身控制设计，视频世界模型方法（[[Fast-WAM]]、DiT4DiT）存在视频到动作反演瓶颈，在有噪声/遮挡环境中效果下降
- **[[ψ-0]]**（前驱工作）: 分解式运动-操作架构限制了并发协调能力

### 本文的动机

直接学习**紧凑潜在视觉预测**作为辅助目标，避免完整视频重建的高计算成本，同时借助 [[SONIC]] 控制器的仿真回放将人类/公开数据集中的视觉-运动先验迁移到可执行机器人动作潜变量。

---

## 方法详解

### 模型架构

![Figure 2: ω-0 系统概览](https://arxiv.org/html/2608.06375v1/x2.png)

ω-0 采用**三阶段训练 + 扩散 Transformer** 架构：

- **输入**: 语言指令 $\ell$ + 当前多视角观测 $o_t^v$（自我中心 RGB / 外中心 RGB / 外中心深度）+ 机器人本体感受状态 $s_t$（47维）
- **当前帧编码**: [[V-JEPA 2.1|V-JEPA 2.1]]（单帧语义特征，冻结）
- **未来帧编码**: [[Wan|Wan 编码器]]（视频训练，冻结）
- **VLM 骨干**: [[Qwen3-VL]] Qwen3-VL-2B-Instruct（细调加入视角 token）
- **动作解码**: [[ActionDiT|Action DiT]]（扩散 Transformer）+ 双查询注意力（运动查询 $m$ + 视频查询 $v$）
- **输出**: 全身动作潜变量 $z_0$（64维 [[SONIC]] 潜变量 + 2维灵巧手指令）+ 预测未来视觉潜变量 $h^v$
- **位置编码**: 视觉 token 用 2D [[3D RoPE|RoPE]]，视频查询用 3D [[3D RoPE|RoPE]]，动作 token 用 1D 时序 [[3D RoPE|RoPE]]

### Stage 1: 全身动作 VLM 预训练

**Step A — [[FAST]] Tokenizer 训练**

将连续动作轨迹转换为离散 token 序列：

$$
c_{1:N} = \mathcal{E}_{act}(a_{t:t+H})
$$

**Step B — VLM 自回归微调**

冻结 [[FAST]] 分词器，在 Qwen3-VL 上对视角条件化动作 token 自回归预测：

$$
\mathcal{L}_{vlm} = -\sum_{i=1}^{N} \log p_\theta\!\left(c_i \mid c_{<i},\, \ell,\, o_t^v,\, e^v\right)
$$

其中 $e^v$ 为视角 token，区分自我中心与外中心摄像头。

### Stage 2: 人类-人形机器人动作潜变量预训练

**前缀条件 Prefix $p$**:

$$
p = \left[f_{vlm},\, f_{\ell},\, r^v,\, f_t^v\right]
$$

- $f_{vlm}$: VLM 输出特征（来自 Stage 1）
- $f_{\ell}$: T5 文本编码
- $r^v$: 机器人状态编码
- $f_t^v$: 当前视觉帧特征（[[V-JEPA 2.1]]）

**双目标联合训练**:

视频潜变量预测损失（[[Wan|Wan 编码器]] 目标）：

$$
\mathcal{L}_{video} = \left\| h^v - y_{t+1:t+K}^v \right\|_2^2
$$

动作扩散去噪损失：

$$
\mathcal{L}_{action} = \left\| \hat{z}_0 - z_0 \right\|_2^2
$$

联合损失：

$$
\mathcal{L}_{stage2} = \mathcal{L}_{action} + \lambda_{video}\, \mathcal{L}_{video}
$$

其中 $\lambda_{video}$ 为视频预测权重系数，$y_{t+1:t+K}^v$ 为未来 $K$ 帧的 Wan 编码器目标潜变量，$\hat{z}_0$ 为预测动作潜变量。

### Stage 3: 真实环境微调 — 训练时[[Real-Time Chunking|实时组块（RTC）]]

标准 action chunking 在组块边界处存在跳变。RTC 在训练时将前 $M$ 步设为干净前缀，后续步骤加噪推理：

$$
\tilde{z}_\tau^{1:M} = z_0^{1:M} \quad (\text{干净前缀})
$$

$$
\tilde{z}_\tau^{M+1:H} = \sqrt{\bar{\alpha}_\tau}\, z_0^{M+1:H} + \sqrt{1 - \bar{\alpha}_\tau}\, \epsilon \quad (\text{噪声后缀})
$$

损失只在噪声后缀部分计算：

$$
\mathcal{L}_{RTC} = \left\| \hat{z}_0^{M+1:H} - z_0^{M+1:H} \right\|_2^2
$$

---

## 关键图表

### Figure 1: Overview — ω-0 与 ω-HOME 数据集概览

![Figure 1](https://arxiv.org/html/2608.06375v1/x1.png)

**说明**: ω-0 从语言、多视角观测和机器人状态生成全身动作潜变量；ω-HOME 提供 40 小时多模态家庭人形机器人演示用于训练和评估。

### Figure 2: ω-0 模型架构

![Figure 2](https://arxiv.org/html/2608.06375v1/x2.png)

**说明**: 首先训练全身 VLM 用于动作感知视觉-语言表示学习（Stage 1），然后使用联合视频-动作潜变量预测器耦合未来视觉潜变量与全身动作生成（Stage 2）。[[SONIC]] 控制器与 [[ActionDiT|Action DiT]] 通过潜变量接口连接。

### Figure 3: ω-HOME 数据集统计与演示

![Figure 3](https://arxiv.org/html/2608.06375v1/x3.png)

**说明**: 40+ 小时数据集统计，包含 8 大任务类别的代表性多模态演示——物体取回、表面清洁、设备交互、容器转移、布料处理、收纳整理、移动操作、工具清洁。

### Figure 4: 遥操作数据采集系统

![Figure 4](https://arxiv.org/html/2608.06375v1/x4.png)

**说明**: 采用 Pico 4 Ultra VR 头显（头部运动）、手持控制器（手部指令）、Pico 足部追踪器（下肢提示）、ZED Mini 摄像头（自我中心视角）和 ZED 深度摄像头（外中心视角）采集演示数据。Inspire DexHands 提供灵巧操作能力。

### Figure 5: 真实机器人任务场景（Part A）

![Figure 5](https://arxiv.org/html/2608.06375v1/x5.png)

**说明**: 11 个家庭长时域运动-操作任务评估场景 Part A，涵盖桌面操作、货架取物、床上衣物取放等需要全身动作协调的场景。

### Figure 6: 真实机器人任务场景（Part B）

![Figure 6](https://arxiv.org/html/2608.06375v1/x6.png)

**说明**: 11 个家庭任务评估场景 Part B，包含擦桌、拖地、多高度垃圾分类、抽屉互动（膝盖关闭）、转身扫地等需要[[Loco-Manipulation]]的复杂任务。

### Figure 7: 泛化实验与人类数据迁移

![Figure 7](https://arxiv.org/html/2608.06375v1/x7.png)

**说明**: 跨物体（新外观实例）、跨场景（新房间布局和家居配置）以及人类数据迁移实验，验证 ω-0 的 OOD 泛化能力。

### Figure 8: 真实长时域任务展示

![Figure 8](https://arxiv.org/html/2608.06375v1/x8.png)

**说明**: 完整长时域任务的连续帧展示，覆盖任务中运动与操作的并发执行过程。

### Table 1: 主实验结果（11 个任务平均）

| Method | Success Rate (%) | Score (Max 41) | Task Progress (%) |
|--------|:-----------------:|:---------------:|:-----------------:|
| ACT | 8.2 | 10.6 | 32.4 |
| Diffusion Policy | 15.5 | 14.8 | 40.6 |
| π-0.5 | 27.3 | 20.9 | 52.8 |
| InternVLA-M1 | 31.8 | 21.8 | 55.6 |
| EgoVLA | 25.5 | 18.6 | 49.1 |
| GR00T-N1.7 | 22.7 | 19.7 | 49.8 |
| ψ-0 | 44.5 | 23.6 | 59.6 |
| Fast-WAM | 37.1 | 22.3 | 57.8 |
| DiT4DiT | 43.6 | 23.1 | 61.0 |
| **ω-0_Ego** | **79.1** | **35.8** | **88.7** |
| **ω-0_Omni** | **81.8** | **36.7** | **90.3** |

**关键发现**: ω-0_Omni（多视角输入）在所有指标上大幅超越所有基线，尤其在需要下肢运动的任务中外中心监督带来显著收益（+37.3% SR vs ψ-0）。

### Table 2: 消融实验

| 消融配置 | Success Rate (%) | Score | Task Progress (%) |
|---------|:-----------------:|:-----:|:-----------------:|
| w/o Robot state | 60.9 | 29.8 | 75.6 |
| w/o VLM prefix | 66.4 | 31.7 | 79.8 |
| w/o Video queries | 64.5 | 30.6 | 77.9 |
| w/o RTC training | 71.8 | 33.4 | 84.1 |
| w/ V-JEPA (不用 Wan 编码器) | 63.6 | 30.9 | 77.3 |
| **Full ω-0_Ego** | **79.1** | **35.8** | **88.7** |
| **Full ω-0_Omni** | **81.8** | **36.7** | **90.3** |

**关键发现**:
- **机器人状态**最关键（移除后 -18.2% SR），提供当前物理状态锚点
- **视频查询**（未来视觉潜变量预测）贡献 +14.6%，验证轻量级世界模型的有效性
- **[[Real-Time Chunking|RTC 训练]]**减少组块边界跳变，贡献 +7.3%
- **[[Wan|Wan 编码器]]**（视频预训练）优于 [[V-JEPA 2.1|V-JEPA]] 作为未来预测目标（+15.5%）

### Table 3: 真实任务逐项结果

| # | 任务 | 下肢运动 | ω-0_Ego SR | ω-0_Omni SR |
|---|------|---------|-----------|------------|
| 1 | 拾取苹果放入篮子 | ✗ | 较高 | 较高 |
| 2 | 货架整理苹果 | ✓ | — | — |
| 3 | 从床上拾衣物扔入篮子 | ✓ | — | — |
| 4 | 移动毛巾至洗衣机 | ✓ | — | — |
| 5 | 擦桌子 | ✓ | — | — |
| 6 | 拖地 | ✓ | — | — |
| 7 | 多高度垃圾分类 | ✓ | — | — |
| 8 | 拾苹果→扔抽屉→膝盖关闭 | ✓ | — | — |
| 9 | 扫垃圾→转身→扔入垃圾桶 | ✓ | — | — |
| 10 | 从洗衣机取衣物 | ✓ | — | — |
| 11 | 从冰箱取饮品 | ✓ | — | — |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| ω-HOME（自建） | 4,827 集 / 40+ 小时 | 多模态家庭任务，含 SMPL 运动参考 | 训练 + 评估 |
| ARCTIC | — | 自我中心 + 外中心视频 | Stage 2 预训练 |
| Xperience-10M（过滤） | 大规模 | 自我中心数据 | Stage 2 预训练 |
| Motion-X（过滤） | 多样 | 多样人体运动（物理可行性过滤） | Stage 2 预训练 |

### 实现细节

- **VLM 骨干**: [[Qwen3-VL]] Qwen3-VL-2B-Instruct（加视角 token 细调）
- **当前帧编码器**: [[V-JEPA 2.1]]（冻结）
- **未来帧目标编码器**: [[Wan|Wan 编码器]]（冻结）
- **文本编码器**: 预训练 T5
- **低级控制器**: [[SONIC]] 全身控制器
- **动作表示**: 66 维（64 维 SONIC 动作潜变量 + 2 维手部指令）
- **机器人状态**: 47 维（关节位置 + 手关节位置 + IMU 6D 旋转）
- **推理频率**: >7 Hz（~0.14 s/forward pass）
- **动作组块长度** $H$: 25 步
- **执行窗口** $K$: 8 步（滚动时域控制）
- **推理**: [[DDIM]] 反向过程（少步去噪） + 相邻组块重叠混合
- **硬件**: 8 × NVIDIA H100 GPU

### 可视化结果

泛化实验表明 ω-0 能适应新物体外观、新房间布局，并在人类数据直接迁移后成功部署于真实人形机器人，验证了跨来源视觉-运动先验的通用性。

---

## 批判性思考

### 优点

1. **轻量级世界模型设计**: 仅预测紧凑潜变量而非完整视频帧，推理效率高（>7 Hz）且无视频反演瓶颈
2. **全身统一动作接口**: 通过 [[SONIC]] 控制器的 64 维潜变量直接控制全身（上肢 + 下肢），避免分解式架构的协调缺陷
3. **人类数据利用**: 利用 ARCTIC、Xperience-10M、Motion-X 等公开数据集的视觉-运动先验，配合仿真回放提取可执行动作潜变量，数据效率高
4. **ω-HOME 数据集贡献**: 家庭场景多模态数据集本身是重要开源资源

### 局限性

1. **任务成功率逐项数据未公开**: 表格中仅给出 11 任务平均值，各任务细粒度结果不够透明
2. **SONIC 控制器依赖**: 全身动作接口强依赖专有 SONIC 控制器，移植到其他硬件平台需重新设计
3. **遮挡与光照鲁棒性**: 文中提到视频世界模型在噪声/遮挡下效果下降，ω-0 的鲁棒性边界未深入量化

### 潜在改进方向

1. 扩展到更高自由度灵巧手（当前 2 维手部指令较粗糙）
2. 融合触觉传感器作为额外观测模态
3. 探索在线 RL 微调进一步提升长时域精度

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（三阶段损失、超参数）
- [ ] ω-HOME 数据集可获取（待发布）

---

## 关联笔记

### 基于

- [[World Action Model]]: ω-0 属于 WAM 范式，通过潜在预测替代视频重建
- [[FAST]]: Stage 1 使用 FAST tokenizer 将连续动作离散化
- [[Action Diffusion]]: Stage 2 使用扩散 Transformer 生成全身动作
- [[Real-Time Chunking]]: Stage 3 核心训练技术，解决组块边界跳变

### 对比

- [[Fast-WAM]]: 同为 WAM 范式，但 ω-0 避免视频反演瓶颈（SR 79.1% vs 37.1%）
- [[EgoVLA]]: 自我中心 VLA，但动作接口不支持全身控制（SR 79.1% vs 25.5%）
- [[ψ-0]]: 分解式运动-操作，ω-0 的统一并发架构实现更优协调（SR 81.8% vs 44.5%）

### 方法相关

- [[V-JEPA 2.1]]: 当前帧语义特征提取（冻结）
- [[Wan]]: 未来视觉潜变量预测目标编码器（冻结）
- [[SONIC]]: 低级全身控制器，提供 64 维动作潜变量接口
- [[Loco-Manipulation]]: 运动-操作并发协调核心问题
- [[SMPL]]: 多来源数据的统一人体运动表示
- [[3D RoPE]]: 视频查询的 3D 时序位置编码

### 硬件/数据相关

- [[Qwen3-VL]]: VLM 骨干 Qwen3-VL-2B-Instruct
- [[DDIM]]: 推理阶段少步去噪
- [[Action Chunking]]: 滚动时域控制策略

---

## 速查卡片

> [!summary] ω-0 (2025, arXiv 2608.06375)
> - **核心**: 潜在预测世界-动作模型，预测未来视觉潜变量而非完整视频，耦合扩散式全身动作生成
> - **方法**: Qwen3-VL + V-JEPA (当前帧) + Wan 编码器 (未来帧目标) + Action DiT + SONIC 控制器；三阶段：VLM 预训练 → 联合视频-动作预训练 → RTC 微调
> - **结果**: 11 个家庭任务平均 SR 81.8%（ω-0_Omni），显著领先 ψ-0 (44.5%)、Fast-WAM (37.1%)
> - **代码**: 暂未开源（项目主页: https://gentlefress.github.io/OMEGA-0_page/）

---

*笔记创建时间: 2026-08-08*

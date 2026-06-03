---
type: concept
aliases: [Action-Conditioned FiLM, 动作条件 FiLM 插值, Action-Conditioned Interpolation]
---

# AC-FILM（Action-Conditioned FiLM Interpolation）

## 定义

SKIP 提出的动作条件视频插值模块，基于光流插值网络，通过逐层 FiLM 调制将机器人动作序列注入插值过程，并引入门控机制在静态阶段抑制插值残差。

## 数学形式

**FiLM 调制**:
$$
F'_\ell = \gamma_\ell \odot F_\ell + \beta_\ell
$$

**静态门控**:
$$
s = \sigma(w_s \bar{m} + b_s), \quad F'_\ell \leftarrow s \cdot F'_\ell
$$

其中 $\bar{m}$ 为当前插值窗口内动作幅度均值，$s \in (0,1)$ 为门控值。

## 核心要点

1. **动作条件对齐**: 将动作序列通过 MLP 预测 $\gamma_\ell, \beta_\ell$，使插值帧的运动与真实动作轨迹对齐，避免视觉自由插值与实际控制不一致
2. **静态阶段门控**: 当动作幅度小（机械臂静止）时，$\bar{m}$ 趋近 0，门控值 $s$ 趋近 $\sigma(b_s)$，如果 $b_s < 0$ 则抑制插值残差，减少静态帧的人工伪影
3. **Zero-init 稳定训练**: $\gamma$ 初始化为 1，$\beta$ 初始化为 0，门控参数初始化使 $s \approx 1$，训练初期为恒等变换
4. **多层注入**: 在光流插值网络的多个特征金字塔层级分别注入，实现粗到细的动作对齐

## 训练细节

- 基础插值网络：光流插值（Flow-based）
- 损失：$\ell_1$ + VGG 感知损失 + Gram 风格损失
- 训练策略：先冻结 1 epoch 预热 FiLM 参数，再联合训练 19 epochs

## 代表工作

- [[SKIP]]: AC-FILM 是 SKIP-Reconstructor 的核心组件，相比无动作条件基线显著提升插值质量（p<0.01 显著性）

## 相关概念

- [[FiLM]]: AC-FILM 使用的特征调制机制
- [[视频插值]]: AC-FILM 所扩展的基础任务
- [[Kernel Temporal Segmentation]]: SKIP 流水线中与 AC-FILM 协同工作的上游模块

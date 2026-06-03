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

1. **动作条件对齐**: 将动作序列通过 MLP 预测 $\gamma_\ell, \beta_\ell$，使插值帧的运动与真实动作轨迹对齐
2. **静态阶段门控**: 当动作幅度小（机械臂静止）时，门控值 $s$ 趋小，抑制插值残差，减少静态帧伪影
3. **Zero-init 稳定训练**: 训练初期为恒等变换，避免初始噪声破坏预训练插值网络
4. **多层注入**: 在光流插值网络的多个特征金字塔层级分别注入，实现粗到细的动作对齐

## 训练细节

- 基础插值网络：光流插值（Flow-based）
- 损失：$\ell_1$ + VGG 感知损失 + Gram 风格损失
- 训练：1 轮冻结预热 + 19 轮联合训练，Batch 4

## 代表工作

- [[SKIP]]: AC-FILM 是 SKIP-Reconstructor 的核心组件，相比无动作条件基线显著提升插值质量（p<0.01）

## 相关概念

- [[FiLM]]: AC-FILM 使用的特征调制机制
- [[Kernel Temporal Segmentation]]: SKIP 流水线中与 AC-FILM 协同工作的上游模块

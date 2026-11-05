---
type: concept
aliases: [未来表征蒸馏, Future Repr. Distillation, Predictive Representation Distillation]
---

# Future Representation Distillation（未来表征蒸馏）

## 定义

一种训练范式，通过将**冻结视频扩散模型**（Teacher）的"未来感知"中间表征蒸馏到仅接收**当前观测**的视觉编码器（Student）中，使编码器在单次前向传播中即可预测未来视觉信息，从而无需生成未来视频即可获得预测性特征用于动作预测。

## 数学形式

设学生编码器为 $\boldsymbol{u}_\theta$，冻结教师 VDM 为 $\boldsymbol{g}$，可训练投影器为 $\boldsymbol{p}_\psi$，蒸馏损失为：

$$
\mathcal{L}_{\text{distill}}(\theta, \psi) := \mathbb{E}_{\boldsymbol{x}^*, \boldsymbol{\epsilon}_d} \left[ -\text{sim}\left( \boldsymbol{p}_\psi(\boldsymbol{h}_i),\, \boldsymbol{g}_j(\boldsymbol{x}_{t_d}, t_d, \ell) \right) \right]
$$

其中：
- $\boldsymbol{h}_i$：学生编码器第 $i$ 层的**当前帧**隐状态
- $\boldsymbol{g}_j(\boldsymbol{x}_{t_d}, t_d, \ell)$：教师在噪声水平 $t_d$ 下处理**完整视频**（含未来帧）得到的第 $j$ 层表征
- $\boldsymbol{p}_\psi$：轻量级 Transformer 解码器，将当前帧表征投影到教师的时空 token 布局（仅训练时使用）
- $\text{sim}$：在逐 token 非仿射 LayerNorm 后计算的平均余弦相似度
- 仅对**未来 token 位置**（非当前帧）做对齐监督

通常与视频生成正则化损失联合使用：

$$
\mathcal{L}(\theta, \psi) = \mathcal{L}_{\text{gen}}(\theta) + \mathcal{L}_{\text{distill}}(\theta, \psi)
$$

## 核心要点

1. **目的**: 将视频扩散模型的预测性知识（时空动力学）注入到只接收当前帧的视觉编码器，使其在特征空间中预测未来，而非生成未来像素。
2. **无动作标注**: Stage 1 训练不需要动作标注，可使用任意视频演示数据（包括来自不同机体的视频）扩展训练集。
3. **对比 REPA**: 受 [[REPA]] 启发但关键区别在于对齐目标是**未来帧**的表征（教师看完整视频），而非当前帧自身，因此蒸馏的是"预测性"而非"重建性"特征。
4. **推理时丢弃**: 教师 VDM 和投影器 $\boldsymbol{p}_\psi$ 仅在训练时使用，部署时完全丢弃，不带来任何推理开销。
5. **可扩展性**: 更大的视频骨干（Teacher/Student 同架构）提供更好的预测性先验，显著提升动作性能（5B→14B: +13.2 pp）。

## 代表工作

- [[DreamFree]]: 首次将未来表征蒸馏应用于 WAM，以单次编码器前向传播替代迭代视频去噪，实现 11Hz 实时机器人控制，真实世界操作成功率 64%（arXiv 2611.04527, 2026）

## 相关概念

- [[REPA]]：表征对齐技术，Future Repr. Distillation 的设计灵感来源
- [[世界动作模型（WAM）]]：Future Repr. Distillation 的应用场景
- [[Knowledge Distillation]]：广义知识蒸馏，本技术的特化形式（面向时序预测）
- [[Video Diffusion Transformer]]：作为 Teacher 和 Student 的视频扩散骨干
- [[Flow Matching]]：与蒸馏损失配合使用的视频生成正则化损失

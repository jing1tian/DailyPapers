---
type: concept
aliases: [适配器对齐, ControlNet Realignment, Adapter Re-alignment]
---

# Adapter Realignment

## 定义

Adapter Realignment 是一种针对条件控制网络（如 ControlNet）的蒸馏后微调方法：在视频/图像骨干网络被加速蒸馏（如单步 DMD）之后，将原始多步适配器复制并在新的快速骨干上重新对齐，弥合多步与单步推理环境之间的分布差距。

## 数学形式

$$
\mathcal{L}_{adapt} = \mathcal{L}_{recon}(z_{adapt}, z_{slow}) + \lambda_{adv} \cdot \mathcal{L}_{adv}(D_\psi(z_{adapt}))
$$

其中 $z_{adapt}$ 为快速适配器与单步生成器的联合输出，$z_{slow}$ 为多步慢速适配器的参考输出，$D_\psi$ 为扩散判别器。

## 核心要点

1. **解耦于骨干蒸馏**: 先完成骨干（视频模型）的蒸馏，再独立对齐适配器，避免两阶段梯度干扰
2. **双重损失**: 潜码重建损失保证生成内容一致性，对抗损失保证生成质量（分布匹配）
3. **判别器设计**: 扩散判别器读取 Transformer 中间层特征（如 block 14、22、30），提供丰富的感知级监督
4. **轻量微调**: 仅微调复制的快速适配器权重，骨干和快速生成器保持冻结

## 代表工作

- [[TrajWAM]]: 将 Trajectory ControlNet 从多步环境迁移至单步快速视频生成器，训练 1,000 步，带来约 1 pp 的成功率提升

## 相关概念

- [[Trajectory ControlNet]]: Adapter Realignment 在 TrajWAM 中的具体应用对象
- [[Distribution Matching Distillation]]: 先于 Adapter Realignment 进行的骨干蒸馏步骤
- [[Action Rollout Distillation]]: 与 Adapter Realignment 并列的第三蒸馏阶段

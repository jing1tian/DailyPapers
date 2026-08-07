---
type: concept
aliases: [Spatial Forcing, 空间强制对齐]
---

# Spatial Forcing

## 定义

将 VLA 中间层视觉嵌入与 3D 基础模型对齐的辅助训练方法，通过空间感知监督信号强化模型的空间推理能力，同时实现 3.8× 训练加速。

## 核心要点

1. 在 VLA 的中间视觉特征层添加空间对齐损失，与 3D 基础模型的输出对齐
2. 额外引入 5.0T FLOPs (+28%) 和 10.9G GPU 内存的训练开销
3. 在 LIBERO 上达到 SOTA 水平，尤其在长序列任务（LIBERO-Long）上提升显著

## 代表工作

- [[CapVector]]: 将 Spatial Forcing 的 capability vector 提取并迁移，以 <0.002% 额外开销复现其性能
- [[Mind-VLA]]: 将场景级 VGGT 对齐（Spatial Forcing 范式）扩展为指令感知的目标物体三视图对齐，解决指令无关问题

## 相关概念

- [[VLA]]
- [[辅助目标微调]]
- [[参数空间向量运算]]

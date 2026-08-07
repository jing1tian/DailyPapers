---
type: concept
aliases: [场景查询, Scene Queries, 场景查询 token]
---

# Scene Query

## 定义

Transformer 基 VLA 模型（如 Mind-VLA）中一组可学习查询 token，专门负责解码场景级别的辅助信息（逐 patch RGB 重建、稠密 2D 运动预测等），与目标物体查询和动作查询分离。

## 核心要点

1. **功能**: 驱动场景级辅助监督（图像重建 $\mathcal{L}_{\text{obs}}$ + 稠密 2D 运动 $\mathcal{L}_{\text{traj}}$）
2. **注意力隔离**: 通过注意力掩码与 Object Query、Action Query 隔离，避免梯度干扰
3. **仅训练时有效**: 辅助解码器在推理时移除，Scene Query 输出不参与最终动作预测

## 代表工作

- [[Mind-VLA]]: 三类查询（Scene / Object / Action）分离设计

## 相关概念

- [[Object Query]]
- [[Action Query]]
- [[辅助监督]]
- [[Vision-Language-Action]]

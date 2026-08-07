---
type: concept
aliases: [目标物体查询, Object Queries, 物体查询 token]
---

# Object Query

## 定义

Transformer 基 VLA 模型（如 Mind-VLA）中一组可学习查询 token，负责预测语言指令指定的目标物体的三视图 VAE 潜变量，实现指令感知的目标物体几何表示学习。

## 核心要点

1. **功能**: 经 MLP 解码头将查询特征映射为目标物体三视图 VAE 潜变量 $\hat{\mathbf{Z}}_t$
2. **监督信号**: 与离线编码的三视图潜变量 $\mathbf{Z}_{m(\ell)}$ 做 MSE 监督（$\mathcal{L}_{\text{tri}}$）
3. **指令感知**: 指令 $\ell$ 变化时，监督目标 $m(\ell)$ 随之变化，使查询 token 学习"语言→物体几何"映射
4. **仅训练时有效**: 推理时辅助解码器移除，零推理开销

## 代表工作

- [[Mind-VLA]]: 三类查询设计中的目标物体查询

## 相关概念

- [[Scene Query]]
- [[Action Query]]
- [[指令感知 3D 对齐]]
- [[VAE]]
- [[Vision-Language-Action]]

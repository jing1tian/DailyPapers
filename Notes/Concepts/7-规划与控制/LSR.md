---
type: concept
aliases: [Latent Space Roadmap]
---

# LSR

## 定义
在视觉世界模型的潜在空间中构建稠密 roadmap（类似 PRM），通过图搜索实现目标条件的长时域规划。

## 核心要点
1. 用随机采样在潜在空间构建 roadmap 图结构
2. 以 L2 距离或学到的可达性度量连接近邻节点
3. 推理时找最短路径，再用 IDM 将路径点转换为动作
4. 瓶颈：潜在空间的局部 metric 不一定对应语义可达性

## 代表工作
- Pertsch et al. (2020): LSR, CoRL
- [[STRIPS-WM]]: 与 LSR 在 BlocksWorld 上对比

## 相关概念
- [[世界模型 (World Model)]]
- [[逆动力学模型]]
- [[符号规划]]

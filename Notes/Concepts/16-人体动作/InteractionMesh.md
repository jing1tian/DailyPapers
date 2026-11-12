---
type: concept
aliases: [Interaction Mesh, GraspTTA]
---

# InteractionMesh

## 定义
一种利用人手-物体交互的 mesh 形变场进行运动重定向的方法，将人手的接触/抓取姿态映射到不同构型（如机器人手）上，同时保持接触关系的合理性。

## 核心要点
1. 以人手和物体的 mesh 表面接触关系为约束
2. 通过 cage-based deformation 或类似变形场进行跨形态重定向
3. 保留抓取功能性（哪个点接触物体）而非精确复制关节角度

## 代表工作
- [[DexSeed]]: 利用 InteractionMesh 把人手交互 demo 重定向到机器人手

## 相关概念
- [[DexSeed]]
- [[UniDex]]
- [[Cross-Embodiment]]

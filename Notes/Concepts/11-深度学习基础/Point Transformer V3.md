---
type: concept
aliases: [PTv3, Point Transformer V3]
---

# Point Transformer V3

## 定义

第三版点云 Transformer 架构，专为 3D 点云数据设计，通过局部自注意力机制高效建模点云中的几何关系，在大规模 3D 感知任务上取得优异性能。

## 核心要点

1. 基于局部窗口自注意力，适配点云的不规则采样结构
2. 高效的点云编码，支持大规模场景理解
3. 在 OrbiSim 中用于对关节物体的各部件进行网格采样后的特征编码

## 代表工作

- [[OrbiSim]]: 用 PTv3 对关节物体各部件进行编码

## 相关概念

- [[Transformer]]
- [[自注意力机制]]
- [[对象中心表示]]

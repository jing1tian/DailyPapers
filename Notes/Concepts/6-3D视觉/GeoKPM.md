---
type: concept
aliases: [Geometric Keypoint Matching, 几何关键点匹配]
---

# GeoKPM

## 定义
Multics 中提出的跨玩家视角空间对齐模块，通过检测并匹配多视角观测中的几何关键点，建立玩家间的空间对应关系。

## 数学形式
$$\mathcal{M}_{ij} = \text{match}(\mathbf{K}_i, \mathbf{K}_j), \quad \mathbf{K}_i = f_\text{kp}(o_i)$$

其中 $o_i$ 是第 $i$ 个玩家的观测，$\mathbf{K}_i$ 是提取的关键点集合，$\mathcal{M}_{ij}$ 是玩家 $i,j$ 之间的匹配矩阵。

## 核心要点
1. 解决多玩家 world model 中不同视角的空间对齐问题
2. 配合层级注意力，将全连接 O(P²) 通信降至近似线性
3. 几何对齐先于特征融合，避免错误匹配导致的特征污染

## 代表工作
- [[Multics]]：GeoKPM 的提出论文

## 相关概念
- [[FVD]]
- [[IDM]]

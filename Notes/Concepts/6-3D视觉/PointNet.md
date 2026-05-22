---
type: concept
aliases: [PointNet, 点云网络]
---

# PointNet

## 定义
首个直接处理无序 3D 点云的深度学习模型，通过对称函数（max pooling）消除点的排列不变性，实现点云分类与分割。

## 数学形式
全局特征提取：
$$f(\{x_1,\ldots,x_n\}) = \gamma\left(\max_{i=1\ldots n} h(x_i)\right)$$
其中 $h$ 是逐点 MLP，$\max$ 是逐维 max pooling，$\gamma$ 是聚合 MLP。

## 核心要点
1. 置换不变性：max pooling 使输出与点云顺序无关
2. 轻量高效：无需体素化，直接处理原始点坐标
3. PointNet++ 引入局部邻域聚合，捕获多尺度结构
4. 在机器人抓取、3D 物体识别中广泛使用

## 代表工作
- [[DeformMaster]]：用 PointNet 编码可形变物体的粒子点云

## 相关概念
- [[SAM]]
- [[PointACT]]

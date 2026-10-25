---
type: concept
aliases: [Dynamic Graph CNN, 动态图卷积]
---

# DGCNN

## 定义
Dynamic Graph CNN：一种点云深度学习方法，通过在特征空间中动态构建 k-NN 图并用 EdgeConv 聚合邻域特征，实现对无序点云的有效编码。

## 数学形式
$$h_i = \max_{j \in \mathcal{N}(i)} \text{MLP}(x_i \| (x_j - x_i))$$

其中 $\mathcal{N}(i)$ 为点 $i$ 在特征空间的 k 近邻，每层重新计算图结构。

## 核心要点
1. EdgeConv：边特征由点自身特征与邻域差值拼接后经 MLP 映射，再 max-pool
2. 动态图：每层在最新特征空间重新构建图，而非固定原始坐标空间的图
3. 输出对点顺序置换不变，对刚体变换具有一定鲁棒性
4. 常用于点云分类、分割、位姿估计等任务的几何编码模块

## 代表工作
- [[F2G-Pose]]: 用 DGCNN 对 RGB-D 点云做局部几何 tokenization，配合 MoE Transformer 做类别级位姿估计

## 相关概念
- [[PointNet]]: 早期点云编码方法，无图结构
- [[FoundationPose]]: 位姿估计中的另一条技术路线

---
type: concept
aliases: [Adaptive Point Transformer, 自适应点云补全]
---

# AdaPoinTr

## 定义
AdaPoinTr：一种自适应 Transformer 架构用于点云补全（shape completion），通过 geometry-aware proxy points 在局部与全局之间传递信息，补全残缺点云。

## 数学形式
$$\hat{\mathcal{P}} = \text{Decoder}(\text{ProxyAttention}(\text{Encoder}(\mathcal{P}_{\text{partial}})))$$

编码器提取稀疏代理点（proxy points），解码器逐步上采样还原完整形状。

## 核心要点
1. Proxy points 桥接稀疏全局特征与密集局部几何，减少 attention 计算量
2. 可处理任意残缺率的点云，包括严重遮挡
3. 在 F2G-Pose 中被用作 baseline，F2G-Pose 通过联合位姿预测与形状补全超越了其独立形状补全性能

## 代表工作
- [[F2G-Pose]]: 在 HouseCat6D 上对比，F2G-Pose 形状补全更精确

## 相关概念
- [[DGCNN]]: 点云编码基础组件
- [[F2G-Pose]]: 应用场景

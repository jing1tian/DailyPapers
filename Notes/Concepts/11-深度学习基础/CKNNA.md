---
type: concept
aliases: [Centered Kernel Nearest Neighbor Alignment, CKNNA, 表征对齐度量]
---

# CKNNA（Centered Kernel Nearest Neighbor Alignment）

## 定义

CKNNA 是一种用于衡量两组神经网络表征之间相似度的指标，通过比较各数据点在两个表征空间中的最近邻结构是否一致来评估对齐程度。与 CKA 相比，CKNNA 更关注局部邻域结构的保持，对于评估"一个表征是否编码了另一个表征的信息拓扑"更为敏感。

## 数学形式

给定两组表征 $\mathbf{H}_1, \mathbf{H}_2 \in \mathbb{R}^{n \times d}$：

$$
\text{CKNNA}(\mathbf{H}_1, \mathbf{H}_2) = \frac{1}{n} \sum_{i=1}^{n} \left| \text{NN}_k(\mathbf{h}_1^{(i)}, \mathbf{H}_1) \cap \text{NN}_k(\mathbf{h}_2^{(i)}, \mathbf{H}_2) \right| \Big/ k
$$

其中先对表征进行中心化（减去均值），再计算 $k$ 最近邻的重叠比例。

## 核心要点

1. **局部结构对齐**: 与全局的 CKA 不同，CKNNA 关注每个样本的邻域是否在两个表征空间中保持一致
2. **中心化预处理**: 先减去均值再计算邻域，消除全局偏移的影响
3. **对控制性能的预测力**: [[FRED]] 发现，单步噪声生成表征与干净未来观测表征之间的 CKNNA 与机器人控制成功率高度相关（Pearson r=0.89）

## 代表工作

- [[FRED]]: 使用 CKNNA 诊断 WAM 表征质量，证明"从噪声生成的表征质量"是控制性能的真正代理指标

## 相关概念

- [[自蒸馏]]: FRED 用自蒸馏提升 CKNNA
- [[World Action Model]]: CKNNA 在 WAM 中的应用场景
- [[REPA]]: 另一类扩散表征对齐方法

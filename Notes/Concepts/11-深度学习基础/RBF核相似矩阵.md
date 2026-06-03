---
type: concept
aliases: [RBF Kernel Similarity Matrix, 径向基函数核矩阵, Gaussian Kernel Matrix]
---

# RBF 核相似矩阵（RBF Kernel Similarity Matrix）

## 定义

用径向基函数（高斯核）计算数据点两两之间相似度构成的矩阵，常用于时序数据的段内一致性建模和图卷积网络的邻接权重。

## 数学形式

$$
G_{ij} = \exp\!\left(-\frac{\|f_i - f_j\|^2}{2\sigma^2}\right)
$$

其中 $f_i, f_j$ 为数据点特征，$\sigma$ 为带宽超参数，$G_{ij} \in [0,1]$。

## 核心要点

1. **归一化处理**: 实际使用时通常需要 center（减均值）和 trace-normalize（除以迹），消除不同模态尺度差异
2. **带宽选择**: $\sigma$ 通常通过中位数启发式（median heuristic）自动估计：$\sigma = \text{median}(\|f_i - f_j\|)$
3. **多模态融合**: 不同模态的核矩阵可直接等权平均融合：$\hat{G} = \frac{1}{R}\sum_r G^{(r)}$
4. **对称正半定**: RBF 核矩阵天然对称且正半定，适合作为 KTS 等算法的输入

## 代表工作

- [[SKIP]]: 分别为视觉流、语义流、本体感知流构建 RBF 核矩阵，归一化后平均输入 KTS 进行时序分割
- 广泛用于核方法（SVM、GPR）、图神经网络等

## 相关概念

- [[Kernel Temporal Segmentation]]: 以 RBF 核矩阵为输入的时序分割算法
- [[DINO]]: SKIP 语义流中用于构建 RBF 核的特征提取器
- [[光流 (Optical Flow)]]: SKIP 视觉流中用于构建 RBF 核的特征来源

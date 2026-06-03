---
type: concept
aliases: [KTS, 核时序分割, Kernel-based Temporal Segmentation]
---

# Kernel Temporal Segmentation（KTS）

## 定义

一种基于核矩阵和动态规划的无监督时序分割算法，通过最小化段内特征方差将时间序列划分为语义一致的子段，常用于视频分割和关键帧提取。

## 数学形式

$$
\min_{\{c_k\}} \sum_{k=1}^{L} \text{Var}(\mathcal{S}_k) + \lambda L
$$

其中 $\mathcal{S}_k$ 为第 $k$ 段内的帧集合，$\lambda$ 为分段数惩罚系数，通过动态规划求全局最优分割。

## 核心要点

1. **基于核相似矩阵**: 输入为帧间核相似矩阵（如 RBF 核），无需显式特征距离计算
2. **动态规划全局最优**: 相比启发式方法（谱聚类、凝聚聚类）保证段内方差最小化的最优解
3. **超参数敏感**: $\lambda$ 控制段数，需根据任务平均时长校准；SKIP 中通过操作平均数预估分段数 $L$
4. **多模态融合扩展**: SKIP 将多个模态的 RBF 核均值后输入 KTS，实现多源信息联合分割

## 代表工作

- [[SKIP]]: 将 KTS 应用于机器人操作轨迹的事件保留关键帧提取，优于 RDP、谱聚类等替代方案（+0.02~0.09 OAS）
- 原始 KTS 工作：Potapov et al., "Category-Specific Video Summarization", ECCV 2014

## 相关概念

- [[RBF 核]]: KTS 的输入，编码帧间特征相似度
- [[光流 (Optical Flow)]]: SKIP-Selector 中为 KTS 提供视觉特征输入
- [[DINO]]: SKIP 语义流中用于构建 RBF 核的特征提取器

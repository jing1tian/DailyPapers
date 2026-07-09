---
type: concept
aliases: [神经扩展律, Neural Scaling Laws, 幂律]
---

# Scaling Law

## 定义

**神经扩展律**描述深度学习模型的损失与模型参数量、训练数据量、计算量之间的幂律关系，可用于预测更大规模模型/数据的性能。

## 数学形式

$$
\mathcal{L}(N, D) \approx \mathcal{L}_{\infty} + AN^{-\alpha} + BD^{-\beta}
$$

$$
C \approx \kappa ND
$$

**符号说明**:
- $N$: 模型参数量
- $D$: 训练 token 数量
- $\mathcal{L}_\infty$: 不可减少损失（训练数据分布的信息熵下界）
- $A, B$: 拟合系数
- $\alpha, \beta$: 幂律指数（通常 $\alpha \approx \beta \approx 0.05$–$0.1$）
- $\kappa$: 计算效率系数
- $C$: 总计算量（FLOPs）

## 核心要点

1. 损失随 $N$ 和 $D$ 均呈**幂律下降**，具有高度可预测性
2. **Chinchilla 最优**：给定计算预算 $C$，$N$ 和 $D$ 应同比扩展
3. 扩展律在 LLM、视频生成、世界模型中均有所验证
4. 世界模型中"数据天花板原则"正是扩展律的推论：数据多样性决定泛化上限

## 代表工作

- [[WorldModelRoadmap]]: 扩展律在世界模型中的适用性分析
- Chinchilla (Hoffmann et al., 2022): 计算最优扩展比例

## 相关概念

- [[World Model]]
- [[Transformer]]
- [[Next-Token Prediction]]

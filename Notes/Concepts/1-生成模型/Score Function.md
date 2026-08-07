---
type: concept
aliases: [得分函数, score function, 分数函数]
---

# Score Function

## 定义
数据分布对数概率密度关于数据点的梯度：$\nabla_x \log p(x)$，用于指示概率密度增加最快的方向，是扩散模型和流匹配去噪采样的核心概念。

## 数学形式

$$
s_\theta(x, t) = \nabla_x \log p_t(x)
$$

**与流匹配速度场的等价关系**（高斯噪声下）:

$$
\nabla_{A^\tau} \log p^\tau(A^\tau \mid \cdot) = -\frac{A^\tau + \tau v_\theta(A^\tau, \tau \mid \cdot)}{1 - \tau}
$$

## 核心要点
1. **扩散模型中**: 得分函数是去噪网络 $\epsilon_\theta$ 的变换形式
2. **流匹配中**: 速度场 $v_\theta$ 与得分函数存在线性等价，可以相互转换
3. **用于采样**: Langevin dynamics 等采样方法直接使用得分函数
4. **条件生成**: 条件得分函数 $\nabla \log p(x|c)$ 通过贝叶斯分解与无分类器引导联系

## 代表工作
- [[CofactVLA]]: 利用流匹配速度场与得分函数的线性等价关系，在速度场域实现因果干预（OPG）

## 相关概念
- [[Flow Matching]]
- [[Classifier-Free Guidance]]
- [[Orthogonal Projection Guidance]]

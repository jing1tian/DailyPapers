---
type: concept
aliases: [Discrete Cosine Transform, 离散余弦变换]
---

# DCT (Discrete Cosine Transform)

## 定义
将时域/空间域信号分解为不同频率余弦函数加权和的变换，低频系数捕获信号主体结构，高频系数捕获细节和噪声。

## 数学形式
$$X[k] = \sum_{n=0}^{N-1} x[n]\cos\left(\frac{\pi(2n+1)k}{2N}\right), \quad k=0,1,\ldots,N-1$$

## 核心要点
1. 能量压缩特性：大多数信号能量集中在低频系数
2. 在 [[HAF]] 中用于分解机器人动作序列：低频 DCT 系数表示身体整体姿态趋势，高频表示接触细节
3. 谱域 RL：直接在 DCT 域学习策略，比时域更稳定

## 相关概念
- [[HAF]]
- [[Flow-Matching]]

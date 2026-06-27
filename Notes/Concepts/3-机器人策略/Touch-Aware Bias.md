---
type: concept
aliases: [触觉感知偏置, touch-aware bias]
---

# Touch-Aware Bias

## 定义
[[Tactile Asymmetric Attention Mechanism (TAAM)|TAAM]] 的第二个组件，由预测的[[Touch-State Proxy|触觉变化代理]]映射得到的标量加性偏置，仅叠加到"动作 query → 触觉 key"的注意力 logit 上，让动作生成在预测到接触状态发生变化的触觉锚点上投入更多注意力。

## 数学形式

$$
b_{i}^{\tau}=g(\delta_{i}^{\tau}), \qquad
B_{q,k}=\bar{B}_{q,k}+\mathbb{I}[G(q)=A,\;G(k)=\tau]\,b^{\tau}_{a(k)}
$$

其中 $g(\cdot)$ 是阈值饱和函数：

$$
s_{i}=\mathrm{clip}_{[0,1]}\left[\tanh\left(\frac{\mathrm{ReLU}(d_{i}-\theta)}{T_{c}}\right)\right], \qquad
b^{\tau}_{i}=\mathrm{clip}\left(\alpha s_{i},\,0,\,b_{\max}\right)
$$

## 核心要点
1. 训练时由**目标**触觉变化代理（[[teacher forcing|teacher forcing]]）构造；推理时第一步无偏置，之后用上一步**预测**的触觉变化代理构造
2. 偏置只依赖"变化幅度"（$d_i = \|\Delta c_i^\tau\|_2$），而非接触量级本身——刻意偏向捕捉滑动开始、卡阻等阶跃式事件
3. 当 $b_i^\tau = 2$ 时，未归一化 softmax 权重等价于乘以 $\exp(2)$，提供直观的强度解释
4. 不缩放触觉特征本身，不影响视频 query，不增加其他 token 组对触觉的注意力——精确局部调制

## 代表工作
- [[Tactile-WAM]]: 提出该偏置机制，是消融实验（Table II）中带来最大性能跃升的组件（4.9% → 44.7%）

## 相关概念
- [[Tactile Asymmetric Attention Mechanism (TAAM)]]
- [[Touch-State Proxy]]
- [[VideoClean Attention Mask]]
- [[teacher forcing]]

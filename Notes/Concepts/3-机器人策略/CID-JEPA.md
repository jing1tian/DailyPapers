---
type: concept
aliases: [Contrastive Inverse Dynamics JEPA, CID]
---

# CID-JEPA

## 定义
在 JEPA（Joint Embedding Predictive Architecture）世界模型中以对比逆动力学（Contrastive Inverse Dynamics）替代高斯后验分布，用 NCE/CPC 类目标函数从状态对中恢复动作，去除 KL 正则约束。

## 数学形式
正向预测（JEPA）：
$$\hat{z}_{t+1} = \text{fwd}_\phi(z_t, a_t)$$

逆动力学对比损失（NCE）：
$$\mathcal{L}_{\text{NCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+1}) / \tau)}{\sum_{j} \exp(\text{sim}(z_t, z_j) / \tau)}$$

## 核心要点
1. 不需要显式高斯假设，避免 RSSM 类方法的后验坍缩问题
2. 逆动力学头通过对比学习恢复动作，和正向预测联合训练
3. 在 PushT、TwoRoom 等简单环境中验证

## 代表工作
- [[LeWM]]: CID-JEPA 的基础框架
- No Gaussian Required (2608.17542): 提出 CID 替换 Gaussian

## 相关概念
- [[LeWM]]
- [[DreamerV3]]
- [[World Model]]

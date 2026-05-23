---
type: concept
aliases: [Model-Agnostic Meta-Learning, 模型无关元学习]
---

# MAML

## 定义
模型无关元学习算法，通过在元训练集上优化"初始化参数"，使模型经过少量梯度步后能快速适应新任务，实现 few-shot 泛化。

## 数学形式
$$\theta^* = \arg\min_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

## 核心要点
1. 内循环（inner loop）：对每个任务做 $K$ 步梯度更新
2. 外循环（outer loop）：对"内循环后的参数"计算损失，更新初始化
3. 需要二阶梯度（Hessian），计算昂贵
4. FOMAML（一阶近似）忽略二阶项，显著降低计算成本

## 代表工作
- [[MAML]]：Finn et al. 2017，ICML，元学习经典
- [[ForeSplat]]：用 MAML/FOMAML 让前馈 3DGS 预见优化梯度方向

## 相关概念
- [[FOMAML]]
- [[模仿学习]]

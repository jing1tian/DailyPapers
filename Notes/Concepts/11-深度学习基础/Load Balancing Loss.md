---
type: concept
aliases: [负载均衡损失, 专家负载均衡, Expert Load Balancing, Auxiliary Load Balancing Loss]
---

# Load Balancing Loss

## 定义

混合专家（MoE）模型中用于防止路由塌陷（大量 token 集中路由到少数专家）的辅助正则化损失，通过惩罚专家激活不均匀来促使所有专家被充分利用。

## 数学形式

**DeepSeekMoE / Switch Transformer 版本**:

$$
f_i = \frac{N}{K \cdot U} \sum_{u=1}^{U} r_{i,u}, \quad P_i = \frac{1}{U} \sum_{u=1}^{U} s_{i,u}
$$

$$
\mathcal{L}_{\text{balance}} = \alpha \sum_{i=1}^{N} f_i \cdot P_i
$$

## 核心要点

1. **专家崩溃（Expert Collapse）**: 若不施加约束，路由网络倾向于将 token 集中送给少数"强"专家，弱化 MoE 的多样性优势
2. $f_i$: 专家 $i$ 的实际激活比例（离散、不可微）；$P_i$: 专家 $i$ 的平均路由概率（连续、可微）；两者之积构成可微的代理损失
3. **系数 $\alpha$**: 通常取小值（0.001–0.01），平衡主任务损失与均衡性约束

## 代表工作

- Switch Transformer: 最早引入辅助负载均衡损失
- DeepSeekMoE: 精化了计算形式，结合共享专家使用
- [[HiMoE-VLA]]: 将 Load Balancing Loss 作为 HB-Reg 施加于 HB-MoE 层，防止异构容量塌陷

## 相关概念

- [[Mixture-of-Experts]]
- [[Contrastive Loss]]

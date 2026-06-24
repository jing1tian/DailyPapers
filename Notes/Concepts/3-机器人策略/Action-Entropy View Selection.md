---
type: concept
aliases: [动作熵视角选择, Action-Entropy-based View Selection]
---

# Action-Entropy View Selection

## 定义
一种训练免费（training-free）的动态视角选择机制：在推理阶段对每个候选（生成或物理）视角，计算策略输出动作分布的 Shannon 熵，选择熵最低（模型最有信心）的视角作为当前决策依据，无需额外训练视角选择器或人工标注视角标签。

## 数学形式
候选视角 $v$ 下的动作熵：

$$
H_v = \frac{1}{N_a} \sum_{n=1}^{N_a} \mathcal{H}\left[p_\theta\left(a_{t,n} \mid a_{t,<n},\ \hat{o}_{t+1}^{\prime v},\ o_{t-1:t},\ L\right)\right]
$$

其中 $\mathcal{H}[p] = -\sum_a p(a) \log p(a)$ 为 Shannon 熵。选择规则：

$$
v^* = \arg\min_{v \in \mathcal{V}_{\text{sel}}} H_v
$$

## 核心要点
1. **以熵代理"视角质量"**: 假设当某个视角能提供更充分的动作关键信息（如解除遮挡）时，策略对该视角下的动作预测会更确定，熵更低
2. **完全免训练**: 不需要专门的视角选择网络或监督标签，复用策略自身已学到的动作分布即可
3. **动态、周期性刷新**: 典型实现每隔固定步数（如 30 个策略步）重新评估一次候选视角，而非每步都切换，平衡稳定性与响应速度
4. **可与生成式视角结合**: 候选视角不必是物理相机，也可以是世界模型生成的虚拟辅助视角

## 代表工作
- [[UniviewVLA]]: 提出该机制，在遮挡专项任务上相比固定（人工选定）视角进一步提升 6.3 个百分点（67.0% → 73.3%）

## 相关概念
- [[Motion-Informative Token Compression]]
- [[World Model]]
- [[VLA]]

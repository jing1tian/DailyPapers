---
type: concept
aliases: [rCM, score-regularized Continuous-time consistency Model, Score-Regularized CM]
---

# rCM（Score-Regularized Continuous-time Consistency Model）

## 定义

一种扩散模型蒸馏框架，将[[一致性蒸馏|一致性模型]]（CM，前向散度、保持模式覆盖/多样性）与[[Distribution Matching Distillation|DMD]]（反向散度、追求分布匹配质量）结合训练，利用两者散度方向的互补性同时获得高质量与高多样性的少步采样器。

## 数学形式

联合训练目标（示意）：

$$
\mathcal{L}_{\text{rCM}} = \mathcal{L}_{\text{sCM}}(\theta;\ \text{离线轨迹数据}) + \lambda \, \mathcal{L}_{\text{DMD}}(\theta;\ \text{在线学生样本})
$$

其中 sCM 项在教师 PF-ODE 轨迹上做连续时间一致性回归（前向散度，KL$(p_{\text{teacher}} \| p_\theta)$ 方向，保多样性），DMD 项在学生自身生成样本上做分布匹配（反向散度，KL$(p_\theta \| p_{\text{teacher}})$ 方向，保质量）。

## 核心要点

1. **前向-反向散度互补原则**：CM 类目标对应前向 KL（mode-covering，防止多样性坍缩），DMD 类目标对应反向 KL（mode-seeking，追求样本质量），联合使用可同时缓解纯 DMD 的 mode collapse 和纯 CM 的样本质量瓶颈
2. **JVP 驱动的连续时间一致性**：用 [[JVP]]（Jacobian-Vector Product）高效计算教师 PF-ODE 轨迹的切线方向，实现 [[sCM]] 而非离散时间 dCM，显著提升收敛速度
3. **大规模基础设施兼容**：使 JVP 计算与 [[FlashAttention]]、[[FSDP2]]、[[Context Parallel]] 等大规模并行训练设施协同工作，是其能扩展到大型图像/视频扩散模型的关键
4. **双向（非自回归）场景**：原始 rCM 面向标准（非因果）扩散模型蒸馏，在图像和视频生成上验证

## 代表工作

- rCM (Zheng et al., 2025): 原始提出工作，验证 CM+DMD 联合训练在双向扩散蒸馏中的有效性
- [[Causal-rCM]]: 将 rCM 的"前向-反向散度互补"原则迁移到自回归（因果）视频扩散场景，把 CM 替换为 Teacher-Forcing 一致性模型（TF-CM），DMD 替换为 Self-Forcing DMD（SF-DMD）

## 相关概念

- [[一致性蒸馏]]
- [[Distribution Matching Distillation|DMD]]
- [[sCM]]
- [[JVP]]
- [[扩散模型]]

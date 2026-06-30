---
type: concept
aliases: [Group Sampling Policy Optimization]
---

# GSPO

## 定义
GRPO 系列变体之一，着重改进 group sampling 策略，通过更精细的组内奖励归一化提升训练稳定性。

## 数学形式
$$
\hat{A}_i^{\text{GSPO}} = \frac{r_i - \text{median}(\mathbf{r})}{\text{IQR}(\mathbf{r})}
$$

用 median 和 IQR 替代 mean/std，对奖励异常值更鲁棒。

## 核心要点
1. 鲁棒奖励归一化：使用中位数和四分位距降低异常奖励的影响
2. 与 DAPO 类似，针对 reasoning 任务优化
3. 在 VLM 推理 benchmark 上与 GRPO/DAPO 竞争

## 代表工作
- [[PRISM]]: 作为 RL 基线之一对比
- [[QwenAgentWorld]]: 在三阶段训练的 RL 阶段使用 GSPO，结合五维评分+规则混合奖励（9:1）

## 相关概念
- [[GRPO]]
- [[DAPO]]

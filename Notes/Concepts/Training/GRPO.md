---
type: concept
aliases: [Group Relative Policy Optimization, 组相对策略优化]
---

# GRPO

## 定义
Group Relative Policy Optimization：一种无需 critic 网络的策略优化算法，通过对同一问题采样多个输出并计算组内相对奖励来替代 value function 估计。

## 数学形式
$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{G} \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]
$$

其中 $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$，$\mathbf{r}$ 为同组 $G$ 个输出的奖励向量。

## 核心要点
1. 对每个 query 采样 $G$ 个输出，组内做奖励归一化得到 advantage，省去 value model
2. 使用 clip + KL 惩罚双重约束防止策略偏移过大
3. 适合 verifiable reward（可验证答案正确性）场景，如数学推理
4. 在 LLM/VLM post-training 中广泛应用（DeepSeek-R1、Qwen3 等）

## 代表工作
- [[PRISM]]: 在 GRPO 前插入 on-policy distillation 预对齐步骤
- [[DAPO]]: GRPO 的改进版，动态调整采样
- [[PAPO-VLA]]: 将规划动作因果重要度融入 GRPO 优势估计，提升 VLA 策略可靠性
- [[Z-1]]: 为 flow-based VLA 模型设计的 GRPO 后训练框架，引入共享前缀 rollout 构建、树状分支、成功感知奖励衰减

## 相关概念
- [[DAPO]]
- [[GSPO]]
- [[EMA]]
- [[Flow Matching]]

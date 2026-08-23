---
type: concept
aliases: [Robotics Reward Benchmark]
---

# RoboRewardBench

## 定义
评估机器人奖励模型（reward model）对轨迹偏好判断能力的 benchmark，用于衡量 verifier / reward model 在机器人任务中的判断准确性。

## 数学形式
$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{score}(\tau^+_i) > \text{score}(\tau^-_i)]$$

其中 $\tau^+$ 为正样本轨迹，$\tau^-$ 为负样本轨迹，评估 reward model 是否能正确区分好坏轨迹。

## 核心要点
1. 输入为机器人执行轨迹对（正/负样本），输出为 reward model 的偏好判断准确率
2. 涵盖多种机器人操作任务和场景
3. 用于评估 trained reward model 和 LLM-as-verifier 两类方法
4. 87.4% 是截至 2026-08 的 SOTA（LLM-as-a-Verifier）

## 代表工作
- [[LLM-Verifier]]：在此 benchmark 上达到 87.4% SOTA

## 相关概念
- [[SAC]]
- [[GRPO]]

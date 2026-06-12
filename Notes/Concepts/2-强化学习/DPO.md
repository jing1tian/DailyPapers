---
type: concept
aliases: [Direct Preference Optimization, 直接偏好优化]
---

# DPO

## 定义
Direct Preference Optimization：一种无需显式 reward model 的偏好学习方法，直接用偏好对数据（chosen vs rejected）优化语言模型或策略，等价于优化一个隐式 reward。

## 数学形式
$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

## 核心要点
1. 把 RLHF 的两步（reward 训练 + RL 优化）合并为一步监督学习
2. 需要参考策略 $\pi_{\text{ref}}$（通常是 SFT 模型）作为 KL 约束基准
3. 对比 [[PPO]]：DPO 无需 value function，训练更稳定但离线；PPO 可在线探索
4. 机器人策略应用：用成功/失败轨迹对代替 NLP 偏好对，但 offline RL 分布偏移是主要风险

## 代表工作
- [[FlowPRO]]: 基于 DPO 思路做 flow-matching VLA 的偏好微调
- Rafailov et al. (2023): 原始 DPO 论文

## 相关概念
- [[PPO]]
- [[流匹配]]
- [[VLA]]

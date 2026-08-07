---
type: concept
aliases: [Random Network Distillation, 随机网络蒸馏]
---

# RND (Random Network Distillation)

## 定义
RND 是一种用于生成内在奖励（intrinsic reward）的探索方法，通过测量 agent 当前状态对随机初始化目标网络的预测误差来衡量"新奇度"——误差越大说明该状态越陌生，给予更高内在奖励。

## 数学形式
$$r^{int}_t = \| f(o_t) - \hat{f}(o_t) \|^2$$

其中 $f$ 是固定随机目标网络，$\hat{f}$ 是可训练的预测网络。

## 核心要点
1. 目标网络权重固定不变，预测网络学习匹配目标输出
2. 训练过程中预测误差在熟悉区域降低，在新区域保持高误差
3. 计算开销低，可与任意 RL 算法结合
4. 用于 offline RL 中估计 OOD 不确定性

## 代表工作
- [[Curiosity-Diffuser]]: 用 RND 内在奖励引导 diffusion policy 避免 OOD 动作

## 相关概念
- [[IQL]]
- [[CQL]]
- [[Diffusion Policy]]

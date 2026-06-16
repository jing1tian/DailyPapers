---
type: concept
aliases: [MuZero, 无模型蒙特卡洛树搜索]
---

# MuZero

## 定义
DeepMind 提出的 model-based RL 算法，在不知道环境规则的情况下学习隐空间世界模型并结合 MCTS 规划，在 Atari 和棋类游戏上均达到 SOTA。

## 数学形式
三个网络：
$$h = h_\theta(o_{1:t}) \quad \text{(表示网络)}$$
$$r_k, s_k = g_\theta(s_{k-1}, a_k) \quad \text{(动态网络)}$$
$$p_k, v_k = f_\theta(s_k) \quad \text{(预测网络，策略+价值)}$$

MCTS 在 $s$ 空间搜索，反向传播更新价值估计。

## 核心要点
1. 无需环境模型知识，直接从经验学习隐空间规划
2. MCTS 在隐空间搜索，远比像素空间高效
3. [[COMET]] 在 object-centric slot 空间上使用 MCTS，是 MuZero 思路的 OC 扩展
4. [[UniZero]] 在 MuZero 基础上统一多任务学习

## 代表工作
- [[COMET]]：slot 空间 MCTS，以 MuZero/UniZero 为 baseline

## 相关概念
- [[MCTS]]
- [[UniZero]]
- [[PlaNet]]
- [[Dreamer]]

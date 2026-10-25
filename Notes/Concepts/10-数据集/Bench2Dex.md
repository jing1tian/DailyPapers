---
type: concept
aliases: [Bench2Dex benchmark, 双臂灵巧手基准]
---

# Bench2Dex

## 定义

双臂灵巧手操作 benchmark，用于评估机器人在复杂双手协调任务中的性能。

## 核心要点

1. 包含 78 个 episodes，覆盖多种双臂灵巧手操作任务
2. 步数预算动态计算：$B_t = \lceil 1.5 \cdot E_t / 3 \rceil$，范围 593-1542 控制步
3. 配套项目主页：[bench2dex.github.io](https://bench2dex.github.io)

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: GPT-6-Astra 在 Bench2Dex 上 1× 25.6%，10× 61.5%（π0.5 基线 25.5%）

## 相关概念

- [[DexJoCo]]
- [[Long-horizon Manipulation]]

---
type: concept
aliases: [CUPID, Cross-Embodiment Policy]
---

# CUPID

## 定义
CUPID 是一个跨 embodiment 的数据选择和策略迁移框架，通过相似度度量在大型人类演示数据集中为特定机器人平台筛选最相关的数据子集。

## 核心要点
1. 跨 embodiment 迁移的核心挑战：不同手部形态的动作空间不对齐
2. 用 morphology-agnostic 的相似度指标（不依赖具体手部结构）比较运动轨迹
3. 数据选择比数据量更重要——适配的少量数据优于大量不适配数据
4. 适用于从 ego 视频数据库（如 [[EgoDex]]）迁移到特定 robot

## 代表工作
- [[SiMDex]]: 扩展 CUPID 思路，做 similarity-based data mining 用于 cross-embodiment 灵巧手

## 相关概念
- [[EgoDex]]
- [[Cross-Embodiment]]
- [[VLA]]

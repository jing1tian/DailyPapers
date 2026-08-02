---
type: concept
aliases: [PRISM]
---

# PRISM

## 定义
一种基于潜在空间规划的目标条件控制方法，通过在学习到的潜在世界模型中进行搜索/规划来生成动作，常被用作 search-based 世界模型控制的 baseline。

## 核心要点
1. 在潜在世界模型中执行在线规划（测试时搜索）
2. 目标：找到从当前状态到目标状态的动作序列
3. 属于 model-based RL 中需要测试时计算的一类方法
4. 被 [[INTACT]] 作为"需要搜索的 baseline"进行对比

## 代表工作
- [[INTACT]]: 将 PRISM 列为对比方法，说明无搜索方法的相对优势

## 相关概念
- [[JEPA]]
- [[GCSL]]
- [[CEM]]
- [[MPPI]]

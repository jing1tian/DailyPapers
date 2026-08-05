---
type: concept
aliases: [Positional Blind Spot, 位置盲区]
---

# PBS (Positional Blind Spot)

## 定义
VLA 模型在工作空间内存在的空间相干失败区域：即使任务指令和物体摆放完全相同，仅将无关干扰物移入盲区位置，VLA 的失败概率就会急剧上升。

## 核心要点
1. 盲区并非随机噪声，而是空间上**相干的**失败热区（spatially coherent），可通过热图可视化
2. 传统基于成功率的评估隐含"工作空间内空间均匀能力"假设，PBS 揭示该假设不成立
3. 诊断方法：在固定物体位置的前提下，系统地移动无关干扰物，统计各位置的失败率
4. 缓解方案：针对盲区位置进行数据增强（空间增强训练）

## 数学形式
$$\text{PBS}_{\theta}(x, y) = \Pr[\text{failure} \mid \text{distractor at } (x,y)]$$

## 代表工作
- [[PBS]]（论文）: 首次提出 PBS 度量，在 LIBERO 上验证

## 相关概念
- [[OpenVLA]]
- [[VLA]]

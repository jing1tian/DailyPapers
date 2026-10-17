---
type: concept
aliases: [VLM Verifier, RoboMonkey Verifier]
---

# RoboMonkey

## 定义
一种在部署时对 VLA 策略进行测试时验证的方法：对同一状态采样多个动作候选，利用 VLM（视觉语言模型）打分选出最优执行，属于 Best-of-N 范式下的 VLM 验证器。

## 核心要点
1. **VLM 打分**: 使用预训练 VLM 对（观测, 动作候选执行的预测结果）进行成功/失败分类，选择评分最高的候选动作执行
2. **无需额外训练**: 利用 VLM 的通用场景理解能力，直接迁移到目标任务
3. **局限性**: 缺乏对物理动态的显式建模，在精细操作（接触富集任务）中难以区分微妙状态差异

## 代表工作
- [[VeriVLA]]: 将 RoboMonkey 重新训练为 delta progress 预测器作为对比基线，VeriVLA 在所有任务上均优于该方法

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[Best-of-N]]
- [[视觉语言模型]]
- [[GPC-RANK]]

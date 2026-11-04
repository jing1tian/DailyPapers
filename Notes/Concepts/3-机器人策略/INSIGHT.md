---
type: concept
aliases: [INSIGHT]
---

# INSIGHT

## 定义
利用 action token 不确定性检测 VLA 策略失败的方法，通过分析 action chunk 中各 token 的预测分布来判断执行置信度。

## 核心要点
1. 分析 action chunk 整体的 token 不确定性分布
2. 是轻量级的 VLA 失败预测方案，不需要额外训练数据
3. 是 CalVLA 中对比的失败检测基线

## 代表工作
- [[CalVLA]]: 与 INSIGHT 对比，提出 outcome-supervised 的置信度 shaping

## 相关概念
- [[CalVLA]]
- [[ActProbe]]
- [[SAFE]]

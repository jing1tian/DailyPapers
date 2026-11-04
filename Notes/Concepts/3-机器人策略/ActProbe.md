---
type: concept
aliases: [ActProbe]
---

# ActProbe

## 定义
一种利用 VLA action chunk 中的不确定性信息来预测任务失败的方法，通过探测动作分布的置信度区分成功和失败状态。

## 核心要点
1. 不修改 policy 本身，只在推理时探测 action token 的不确定性
2. 是 VLA failure detection 领域的一种轻量方案
3. 与 CalVLA 的对比：ActProbe 是外部探测器，CalVLA 修改 policy 内部置信度

## 代表工作
- [[CalVLA]]: 与 ActProbe 对比，提出 native confidence shaping

## 相关概念
- [[CalVLA]]
- [[INSIGHT]]
- [[SAFE]]

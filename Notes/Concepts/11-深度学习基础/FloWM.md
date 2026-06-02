---
type: concept
aliases: [FloWM]
---

# Flow Equivariant World Model

## 定义
将光流等变性引入世界模型，把 ego-motion 光流和外部物体运动分离，解决部分可观测动态环境下的记忆问题。

## 数学形式
$$h_{t+1} = f_\text{equiv}(h_t, \text{flow}_t^{\text{ego}}, \text{flow}_t^{\text{obj}})$$

## 核心要点
1. 1. Ego-motion 光流 + 外部物体光流分离
2. 2. 等变表示作为归纳偏置
3. 3. 基于 SSM（状态空间模型）架构

## 代表工作
- (待补充)

## 相关概念
- [[DreamerV3]]
- [[World Model]]

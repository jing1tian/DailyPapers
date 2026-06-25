---
type: concept
aliases: [Intelligent Driver Model]
---

# IDM

## 定义
经典的微观交通跟车模型，根据与前车的相对速度和间距，用一个连续可微的加速度公式描述车辆纵向跟驰行为。

## 数学形式
$$\dot{v} = a\left[1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*(v,\Delta v)}{s}\right)^2\right]$$

## 核心要点
1. 规则驱动、参数可解释，常用作驾驶仿真/baseline 的纵向控制模型
2. 与 [[MOBIL]]（横向换道模型）搭配，构成经典的规则式驾驶行为基线
3. 常被神经网络/学习式驾驶策略拿来做"可解释但表达力有限"的对比对象

## 相关概念
- [[MOBIL]]

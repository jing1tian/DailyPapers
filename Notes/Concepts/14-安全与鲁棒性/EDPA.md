---
type: concept
aliases: [Embedding-Driven Patch Attack]
---

# EDPA

## 定义
Embedding-Driven Patch Attack：一种针对 VLA 的物理对抗补丁攻击，通过操控 VLA 的视觉 embedding 来误导动作输出。

## 核心要点
1. 在输入图像上添加物理可打印的对抗补丁
2. 攻击目标是 VLA 的视觉 encoder 输出 embedding
3. 比像素级梯度攻击对物理实现更友好

## 代表工作
- [[SARF]]: 防御 EDPA 等物理补丁攻击
- [[DRIFT]]: 与 EDPA 对比，针对 flow-matching VLA 的新型攻击

## 相关概念
- [[对抗补丁攻击]]
- [[UADA]]
- [[UPA]]

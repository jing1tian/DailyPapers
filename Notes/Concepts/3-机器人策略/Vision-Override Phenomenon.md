---
type: concept
aliases: [视觉覆盖现象, vision-override]
---

# Vision-Override Phenomenon

## 定义
VLA 模型在训练过程中，由于视觉流信号密度远大于语言指令信号，策略倾向于忽略语言指令、转而依赖视觉捷径（如显著物体、熟悉背景布局）完成任务的现象。

## 数学形式

$$
I(A; C \mid O) > 0
$$

其中视觉混淆变量 $C$ 在给定观测 $O$ 条件下对动作 $A$ 仍有正的条件互信息，说明模型被视觉偏差所影响。

## 核心要点
1. **成因**: 视觉特征维度高、训练数据中语言-视觉相关性强，形成虚假因果路径
2. **表现**: 环境背景/物体布局改变时性能急剧下降（OOD 场景失效）
3. **危害**: 策略实际上对语言指令不鲁棒，难以泛化到新场景
4. **解决思路**: 因果干预（截断虚假视觉路径）or 数据增强

## 代表工作
- [[CofactVLA]]: 首次通过 [[Dual-path Deconfounding Graph|DDG]] 形式化建模该现象，并提出 OPG+CCR 双层因果干预解决方案

## 相关概念
- [[Dual-path Deconfounding Graph]]
- [[Out-of-Distribution Generalization]]
- [[VLA]]
- [[因果推断]]

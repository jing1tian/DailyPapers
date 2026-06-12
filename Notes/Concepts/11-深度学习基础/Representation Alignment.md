---
type: concept
aliases: [表征对齐, Feature Alignment, 特征对齐, Cross-modal Alignment]
---

# Representation Alignment

## 定义

将一个模型的内部表征对齐到另一个（通常更优质的）目标表征空间，通过损失函数约束使源特征获得目标特征的期望组织属性。

## 数学形式

常见形式——余弦对齐损失：

$$
\mathcal{L}_{\text{align}} = -\frac{1}{N} \sum_{i=1}^{N} \cos(P(h_i),\ y_i^*)
$$

其中 $h_i$ 为源模型隐状态，$y_i^*$ 为目标表征（通常来自冻结的教师模型），$P$ 为可学习投影头。

## 核心要点

1. **教师-学生范式**: 冻结的高质量教师模型提供对齐目标，学生模型通过梯度更新对齐
2. **投影头解耦**: 可学习投影头 $P$ 桥接维度差异，避免直接修改源模型结构
3. **层选择**: 对齐位置影响性能——过浅层缺乏语义，过深层干扰任务特定特征
4. **与蒸馏的区别**: 表征对齐不要求输出分布匹配，仅约束中间层特征的空间结构

## 代表工作

- [[AGRA]]: 将视频 DiT 隐状态对齐至冻结 DINOv2 特征，解决 action-grounding gap；Layer 8（1/3 深度）为最优对齐位置

## 相关概念

- [[Cosine Similarity]]
- [[DINOv2]]
- [[Knowledge Distillation]]
- [[World Action Model]]

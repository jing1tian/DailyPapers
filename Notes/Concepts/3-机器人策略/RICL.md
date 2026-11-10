---
type: concept
aliases: [RICL, Retrieval-based In-Context Learning, 检索增强上下文学习]
---

# RICL（Retrieval-based In-Context Learning for VLAs）

## 定义

一种为预训练 [[VLA（视觉-语言-动作模型）|VLA]] 模型添加上下文学习能力的框架：通过检索演示库中的示范块并将其注入策略上下文，使 VLA 无需重新微调即可适应新任务。

## 核心要点

1. 将演示数据切分为固定长度的"示范块"（demonstration chunks），以 [[DINOv2]] 视觉嵌入建立索引
2. 推理时，计算当前观测与所有示范块的视觉相似度，检索最近邻块
3. VLA 经过后训练（post-training），学习以检索到的块为条件生成动作块

## 局限

纯视觉相似度检索可能选到**错误执行阶段**的示范块（如外观相似但处于任务初始阶段的帧），在域外任务上尤为明显。

## 代表工作

- [[VICTR]]: 以 IC-VFE 进度估计替代视觉相似度，解决 RICL 的阶段错配问题（OOD 进度率 42.1% → 55.5%）

## 相关概念

- [[In-Context Imitation Learning]]
- [[In-Context Learning]]
- [[DINOv2]]
- [[Pi0.5]]
- [[IC-VFE]]

---
type: concept
aliases: [Grounding DINO, 开放词汇目标检测]
---

# GroundingDINO

## 定义
开放词汇目标检测模型，将 DINO 检测器与文本 grounding 能力结合，支持用自然语言描述（如"red cup on the left"）直接检测目标物体，无需预定义类别。

## 数学形式
$$\text{boxes} = f_\theta(I, \text{text\_query}), \quad \text{score} = \cos(v_i, t_j)$$

## 核心要点
1. DINO backbone 提取视觉特征，BERT 类编码器提取文本特征
2. Cross-attention 融合视觉-文本特征做 grounding
3. 无需目标类别 ID，用自然语言描述定位物体
4. GesVLA 中用于将手势指向映射到目标物体的开放词汇检测

## 代表工作
- [[GroundingDINO]]：Liu et al. 2023，开放集目标检测
- [[GesVLA]]：用 GroundingDINO 做手势-物体关联

## 相关概念
- [[DINO]]
- [[CLIP]]
- [[SAM]]

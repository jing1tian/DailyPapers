---
type: concept
aliases: [OpenGaussian, Open-Vocabulary 3D Gaussian Splatting]
---

# OpenGaussian

## 定义

OpenGaussian 是一种开放词汇表的 3D Gaussian Splatting 场景理解方法，为场景中的每个 Gaussian 附加开放词汇语义特征，支持任意语言 query 的 3D 定位和分割，无需预定义类别。

## 核心要点

1. **开放词汇**: 支持任意自然语言 query，不限于训练时见过的类别
2. **Gaussian 级别语义**: 直接在 3DGS 的 Gaussian 粒子上存储语义特征
3. **与 LangSplat 对比**: 两者都是 3DGS + 语言，OpenGaussian 偏向开放词汇，LangSplat 偏向 CLIP 语义场

## 相关概念

- [[3D Gaussian Splatting]]
- [[LangSplat]]
- [[LERF]]
- [[ReferSplat]]

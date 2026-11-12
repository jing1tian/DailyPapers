---
type: concept
aliases: [NeRF with Reflections]
---

# NeRFReN

## 定义
一种专门处理场景反射的 NeRF 方法，将场景分解为 transmitted（透射/漫反射）和 reflected（镜面反射）两个分量，分别建模，从而去除反射伪影。

## 核心要点
1. 双分支分解：transmitted NeRF + reflected NeRF
2. 利用几何先验（如深度）区分两种分量
3. 比标准 NeRF 在含反光玻璃、金属表面的场景中重建质量显著更好

## 代表工作
- Guo et al., NeRFReN: Neural Radiance Fields with Reflections, CVPR 2022
- [[RF3DGS]]: 在 3DGS 框架下完成类似的无反射重建

## 相关概念
- [[3DGS]]
- [[RF3DGS]]
- [[NeRF]]

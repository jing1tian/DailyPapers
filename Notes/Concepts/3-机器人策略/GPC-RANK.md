---
type: concept
aliases: [GPC RANK, Pixel-Level World Model Verifier]
---

# GPC-RANK

## 定义
一种用于 VLA 测试时验证的像素级世界模型排序方法：通过生成（generate）候选动作执行后的像素级未来帧，再用一个评估器（estimator）对候选动作进行排序（rank），属于像素重建验证范式。

## 核心要点
1. **生成-评估两阶段**: 先用像素级生成模型预测各候选动作执行后的图像，再用评估器排序
2. **计算成本高**: 像素生成本身是更难的问题，计算开销大
3. **分布外退化**: 生成模型在测试时轨迹帧（分布外）质量下降，排序精度受损
4. **物理建模能力中等**: 像素级预测包含一定物理信息，但噪声较大

## 代表工作
- [[VeriVLA]]: 将 GPC-RANK 作为像素级世界模型验证器基线，在所有任务上 VeriVLA（潜在世界模型）均优于 GPC-RANK

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[Best-of-N]]
- [[RoboMonkey]]
- [[Latent World Dynamics]]

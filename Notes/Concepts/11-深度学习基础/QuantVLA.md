---
type: concept
aliases: [QuantVLA, VLA Quantization baseline]
---

# QuantVLA

## 定义

针对 Vision-Language-Action 模型的量化方法，通过跳过 DiT 动作头的注意力层（避开量化最困难的部分）来实现 W4A8 量化，但以牺牲完整量化覆盖率换取精度保留。

## 核心要点

1. **跳过 DiT 注意力层**: 不对 DiT 动作头的注意力层做激活量化，只做语言主干的量化
2. **W4A8 精度**: 权重 4 位，激活 8 位（不含 DiT 注意力层）
3. **局限**: 真正的全量化（W4A4）下性能崩溃（LIBERO 从 87% 跌至 69.8%），说明其方法无法处理 DiT 的激活动态范围漂移

## 对比

| 方法 | 精度 | DiT 覆盖 | GR00T-N1.5 | Pi-0.5 |
|------|------|----------|------------|--------|
| QuantVLA | W4A8 | 不含注意力 | 88.0% | 97.6% |
| QuantVLA | W4A4 | 全量化 | 69.8% | 82.0% |
| [[Omega-QVLA]] | W4A4 | 全量化 | 87.8% | 98.0% |

## 代表工作

- [[Omega-QVLA]]: 超越 QuantVLA 的方法，通过复合旋转真正解决 DiT 量化难题

## 相关概念

- [[后训练量化（PTQ）]]
- [[Diffusion Transformer (DiT)]]
- [[Vision-Language-Action Model]]

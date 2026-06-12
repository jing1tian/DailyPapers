---
type: concept
aliases: [Chain-of-Thought VLA, CoT VLA]
---

# CoT-VLA（链式推理 VLA）

## 定义

将 chain-of-thought（CoT）推理显式集成到 VLA 训练与推理流程中的方法，通过在动作预测前生成自然语言推理步骤来提升空间理解和任务规划能力。

## 核心要点

1. **显式文本推理**：推理时生成 CoT 文本，再据此预测动作，增加推理延迟。
2. **LIBERO 性能**：平均 83.9%（Spatial 87.5%、Long 69.0%），在 3DThinkVLA 的对比中是明显弱于近期方法的 baseline。
3. **推理开销**：需要生成完整推理文本，推理速度较慢。
4. **3DThinkVLA 的改进**：3DThinkVLA 通过潜在蒸馏实现类似效果，无需显式文本生成，推理效率更高。

## 代表工作

- [[3DThinkVLA]]: CoT-VLA 作为早期 reasoning VLA 基线，3DThinkVLA 在无需显式 CoT 文本的前提下超越其性能（98.7% vs 83.9%）

## 相关概念

- [[VLA]]
- [[Chain-of-Thought Reasoning]]
- [[知识蒸馏]]

---
type: concept
aliases: [RMT, Recurrent Memory Transformer]
---

# RMT

## 定义

RMT（Recurrent Memory Transformer）是一种将循环记忆机制嵌入 Transformer 架构的方法，通过可学习的 memory token 在 segment 之间传递信息，使标准 Transformer 具备长程记忆能力，无需修改注意力计算。

## 核心要点

1. **Memory Token**: 在每个输入 segment 首尾插入固定数量的 memory token，处理后写回供下一 segment 读取
2. **循环信息传递**: Memory token 跨 segment 传递，形成类 RNN 的循环结构
3. **计算效率**: 每个 segment 长度固定，计算复杂度不随总序列长度增长
4. **VLA 应用**: 在 RoboMME benchmark 中作为 VLA 记忆机制之一评测

## 数学形式

$$h_{t+1} = \text{Transformer}([m_t; x_t; m_t^{\text{write}}])$$

其中 $m_t$ 是输入记忆 token，$m_t^{\text{write}}$ 是输出写回 token，传递到下一 segment。

## 代表工作

- Bulatov et al. 2022: RMT 原始论文，NeurIPS 2022

## 相关概念

- [[FrameSamp]]
- [[TokenDrop]]
- [[MemER]]
- [[Memory Module]]

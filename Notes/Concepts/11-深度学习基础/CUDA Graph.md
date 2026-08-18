---
type: concept
aliases: [CUDA图, 静态计算图, CUDA静态图]
---

# CUDA Graph

## 定义

NVIDIA CUDA 的一项优化机制：将一系列 GPU kernel 调用预先捕获为静态计算图，后续推理时仅更新输入缓冲区并直接回放（replay）该图，消除重复的 kernel 启动开销和 CPU-GPU 同步延迟。

## 数学形式

$$
\hat{\mathbf{a}}_{t:t+H-1} = \mathcal{G}\!\left(\mathbf{x}_{\text{text}},\, \mathbf{x}_{\text{vision}},\, \mathbf{x}_{\text{state}}\right)
$$

其中 $\mathcal{G}$ 为预编译的 CUDA 静态图，每次推理只需更新输入张量内容，无需重新调度 kernel。

## 核心要点

1. **一次捕获，多次回放**：首次运行时录制 kernel 调用序列，后续直接回放，省去每次的调度开销
2. **延迟显著下降**：适合 LLM 等推理路径固定的模型，可将单次推理延迟降低 20–50%
3. **输入形状固定**：要求每次推理的张量形状（batch size、序列长度）一致，对动态形状场景需额外处理
4. **与批量编码配合**：搭配批量视觉编码（多帧并行），可进一步提升 GPU 利用率

## 代表工作

- [[ReflexVLA]]：与批量视觉编码结合，将推理延迟从 125ms 降至 65ms（-48%）

## 相关概念

- [[KV Cache]]
- [[FlashAttention]]
- [[Roofline Model]]

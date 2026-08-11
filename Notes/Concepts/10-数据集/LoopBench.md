---
type: concept
aliases: [LoopBench]
---

# LoopBench

## 定义
专门评估 interactive video world model 在 long-horizon rollout 下视觉持久性的 benchmark，由 NVIDIA + Princeton 提出。

## 核心要点
1. 测试模型在 rollout 超出训练 horizon 后能否可靠检索早期帧内容
2. 设计闭环场景（loop），要求模型"记住"进入某区域时看到的内容，在离开后再回来时能一致还原
3. 用 TempSSIM（temporal structural similarity）量化视觉持久性
4. 核心发现：autoregressive 视频模型依赖 KV cache 位置 bias，超 horizon 后几乎无法寻址早期内容

## 代表工作
- [[WorldTrace]]: 提出 addressable memory 解决 LoopBench 暴露的持久性问题

## 相关概念
- [[WorldTrace]]
- [[Addressable Memory]]

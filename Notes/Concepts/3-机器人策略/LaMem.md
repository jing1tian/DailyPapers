---
type: concept
aliases: [Dual Latent Memory, 双潜在记忆VLA]
---

# LaMem

## 定义
VLA 模型中的双潜在记忆机制，将 episodic 记忆和 semantic 记忆同时嵌入 VLA 原生的 latent 空间，通过 cross-attention 与当前多模态 token 交织，支持长程任务依赖建模。

## 核心要点
1. **Episodic memory**：跨步骤历史 token 的压缩表示，记录"发生了什么"
2. **Semantic memory**：指令级别语义依赖，记录"在做什么任务"
3. 两路 memory 通过 cross-attention 与当前 VLM token 深度交织（非简单拼接）
4. 区别于外挂 RAG 式 memory：history 直接存在 latent 空间内，与 VLM 推理原生结合

## 代表工作
- [[Dual Latent Memory]]: Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation (2607.07608)

## 相关概念
- [[HAMLET]]
- [[NativeMEM]]
- [[MemoryVLA]]
- [[OpenVLA]]

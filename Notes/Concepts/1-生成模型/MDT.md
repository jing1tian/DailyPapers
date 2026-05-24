---
type: concept
aliases: [Masked Diffusion Transformer]
---

# MDT

## 定义
在扩散 Transformer 中引入 masked token prediction 任务，在训练时随机 mask 部分 token 并要求模型重建，通过掩码自监督增强 context token 的语义表示，提升条件生成质量。

## 核心要点
1. 训练时对 latent token 随机 mask，要求扩散模型兼做重建任务
2. 强迫 Transformer 在去噪过程中利用更丰富的上下文信息
3. 在图像/视频生成中提升语义一致性
4. EvoScene-VLA 参考 MDT 的掩码机制做场景 token 的序列建模

## 代表工作
- [[MDT]]：Gao et al. 2023，Masked Diffusion Transformer for Video Generation

## 相关概念
- [[Diffusion Transformer]]
- [[Diffusion Model]]
- [[DiT]]

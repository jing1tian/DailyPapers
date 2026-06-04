---
type: concept
aliases: [Wan2.2-5B, Wan 2.2]
---

# Wan 2.2-5B

## 定义

Wan 2.2-5B 是一个 50 亿参数的视频生成基础模型，基于 Video Diffusion Transformer 架构，包含预训练 VAE 编码器、T5 文本编码器和 Video DiT 主干，广泛被用作机器人策略和世界模型的骨干网络。

## 核心要点

1. 5B 参数规模的开源视频生成模型，训练于大规模视频数据
2. 预训练 VAE 提供高质量的视频潜变量编码/解码能力
3. T5 文本编码器支持文本条件生成
4. 在机器人领域被 Fast-WAM、GeoSem-WAM 等 WAM 方法用作骨干，通过 fine-tuning 适配操作任务

## 代表工作

- [[Fast-WAM]]: 以 Wan 2.2-5B 为骨干的 World Action Model
- [[GeoSem-WAM]]: 在 Fast-WAM 架构（Wan 2.2-5B 骨干）基础上增加几何语义辅助监督

## 相关概念

- [[Video Diffusion Transformer]]
- [[T5]]
- [[VAE]]
- [[World Action Model]]

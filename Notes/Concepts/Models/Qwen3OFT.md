---
type: concept
aliases: [Qwen3-OFT, Qwen3 OFT, Qwen3OFT VLA]
---

# Qwen3OFT

## 定义

基于 Qwen3 VLM + [[OFT|Orthogonal Fine-Tuning (OFT)]] 微调的 [[Vision-Language-Action Model|VLA]] 基线模型，参数量约 2.2B，在 LIBERO 基准上具有竞争力的推理吞吐（10.49 Hz）。

## 核心要点

1. **Backbone**: Qwen3 系列 [[Vision-Language Model|VLM]]（约 2.2B 参数），配合 [[OFT]] 微调方法保留预训练知识。
2. **LIBERO 性能**: 平均成功率 94.9%（Spatial 95.0%, Object 97.0%, Goal 97.1%, Long 90.5%）。
3. **作为 LoopVLA 的基础**: [[LoopVLA]] 的 LoopOFT 变体以 Qwen3OFT 相同 Backbone，但引入循环精炼后以 1.2B 参数超越其 2.2B 性能（96.0% vs 94.9%）。
4. **推理效率**: 10.49 Hz 推理速度（LoopOFT₀ 早退版本达 18.41 Hz，约 1.76× 加速）。

## 代表工作

- [[LoopVLA]]: 在 Qwen3OFT 基础上引入 Loop Block，实现参数减少 45% 同时性能超越。

## 相关概念

- [[OFT]]
- [[Vision-Language Model]]
- [[Vision-Language-Action Model]]
- [[LoopVLA]]
- [[LIBERO]]

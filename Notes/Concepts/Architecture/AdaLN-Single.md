---
type: concept
aliases: [AdaLN Single, Shared AdaLN, 单次自适应层归一化]
---

# AdaLN-Single（单次自适应层归一化）

## 定义
对标准 [[AdaLN]]（Adaptive Layer Normalization）的轻量化改进：在整个 Transformer 栈前计算一次共享时步调制（而非每个块单独计算），每层仅添加可学习的小型偏移表（per-layer modulation table），大幅减少调制开销。

## 核心要点
1. **共享投影**：时步条件 $t$ 只经过一次 MLP 投影生成共享调制基向量，而非每个 Transformer 块各自运行时步 MLP
2. **层专属偏移表**：每层将可学习的偏移量加到共享信号上，再解码为 shift/scale/gate 三元组，提供层级差异化的调制能力
3. **初始化策略**：共享投影零初始化（DiT 风格），层偏移表以小随机值初始化，支持稳定的残差学习
4. **效果**：显著降低每块的时步处理计算量，适合参数量极大的 MoE 扩散模型

## 代表工作
- PixArt-α (Chen et al., ICLR 2024): 引入 AdaLN-single 设计
- [[LingBot-Video]]: 在单流 MoE DiT 中使用 AdaLN-Single 降低调制开销
- Wan 系列视频模型

## 相关概念
- [[AdaLN]]
- [[DiT]]
- [[Diffusion Transformer (DiT)]]

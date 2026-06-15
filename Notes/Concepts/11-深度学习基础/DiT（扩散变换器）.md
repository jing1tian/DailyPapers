---
type: concept
aliases: [DiT, Diffusion Transformer, 扩散变换器]
---

# DiT（扩散变换器）

## 定义

将 Transformer 替代 U-Net 作为扩散模型骨干的架构范式（Peebles & Xie, 2023），通过序列建模处理图像/视频的潜在表示，利用自注意力机制捕获全局依赖关系。

## 数学形式

$$
x_0 = F_\theta(x_\sigma, \sigma, c)
$$

DiT 将噪声潜变量 $x_\sigma$ 通过多层 Transformer 块去噪，条件信息 $c$（文本/类别标签）通过 AdaLN-Zero 调制层注入。

## 核心要点

1. **序列化**: 将图像 patch 或视频 token 展开为序列，用标准 Transformer 处理
2. **可扩展性**: 模型规模（深度/宽度）可线性扩展，FLOPs 与性能正相关
3. **条件注入**: AdaLN-Zero（自适应层归一化）将时间步和条件信息调制到每层
4. **视频扩展**: Video DiT 在时空维度共同建模，处理多帧潜变量

## 代表工作

- [[NavWAM]]: 使用 DiT 主干（Cosmos Predict2）构建导航世界动作模型
- [[Cosmos-Policy]]: 基于 DiT 的机器人操作策略
- [[视频扩散模型]]: 大量现代视频生成模型采用 Video DiT

## 相关概念

- [[扩散模型]]
- [[视频扩散模型]]
- [[Cosmos Predict2]]
- [[因果VAE]]

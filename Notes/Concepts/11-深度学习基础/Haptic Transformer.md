---
type: concept
aliases: [HaT, Haptic Transformer, 触觉 Transformer]
---

# Haptic Transformer (HaT)

## 定义

LingBot-Touch 中的共享 Transformer 编码器骨干，通过 5 维旋转位置编码（时间 + 手指标识 + xyz 坐标）处理来自不同触觉传感器的异构 token，学习跨传感器统一的接触物理表征。

## 数学形式

5 维 RoPE 位置地址：

$$
a_i = (t_i, f_i, p_i) \in \mathbb{R}^5
$$

5 个轴各自作用于注意力头维度的不同切片，无需固定 token 数目或传感器特定位置表。

## 核心要点

1. **模态专用边界层**：传感器特定参数仅限于浅层输入 / 输出投影（两层 MLP tokenizer），HaT 本体由所有传感器共享
2. **5D RoPE**：时间、手指标识、指尖规范坐标（xyz）各自独立的旋转位置编码，支持任意 token 数量和不同布局的传感器
3. **两种尺寸**：Tiny（5.46M，dim=192）和 Small（21.76M，dim=384），均为 12 层，含 tokenizer 和骨干，不含预训练 predictor
4. **多层特征拼接**：推理时从 4 个深度提取特征，LayerNorm 后拼接，提供多尺度表征给下游任务头
5. **计算效率**：Tiny 仅占下游策略参数量的 ≤ 0.16%，推理开销极低

## 代表工作

- [[LingBot-Touch]]: 提出 HaT 架构，作为 Tactile-JEPA 的编码器骨干

## 相关概念

- [[Tactile-JEPA]]: HaT 通过此目标预训练
- [[Canonical Fingertip Space]]: HaT 通过 RoPE 注入规范坐标位置信息
- [[Rotary Position Encoding]]: HaT 使用的位置编码机制
- [[Transformer]]: 基础架构
- [[Self-Supervised Learning]]: 预训练范式

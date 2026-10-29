---
type: concept
aliases: [PAS-VF, 进度感知语义值函数]
---

# Progress-Aware Semantic Value Function (PAS-VF)

## 定义

一种基于视觉-语言余弦相似度的密集进度估计器，通过冻结的 V-JEPA 和 CLIP 编码器将轨迹帧与语言指令映射到共享空间，以成功轨迹为参照为任意质量的轨迹帧赋予 [0,1] 的进度值。

## 数学形式

**进度对齐损失（成功轨迹）**:

$$
\mathcal{L}_{align} = \mathbb{E}_{\tau \in \mathcal{T}_{succ}} \left[ \frac{1}{T_\tau} \sum_t \left( \cos(z_t^v, z^l) - \frac{t}{T_\tau} \right)^2 \right]
$$

**失败轨迹自参照进度值**:

$$
V(\tau_{fail}) = \max_{z \in \mathcal{C}_{ref}} \cos(z_{\tau_{fail}}^v, z)
$$

## 核心要点

1. 使用冻结的 [[VJEPA-2]] 视频编码器 + 轻量 adapter 生成视觉轨迹嵌入 $z_t^v$。
2. 使用冻结 CLIP 文本编码器 + 轻量 adapter 生成语言嵌入 $z^l$。
3. 在成功轨迹上训练：cos 相似度应等于 $t/T$（归一化进度）。
4. 对失败轨迹：用其与成功参考聚类的最大 cos 相似度衡量"达到了多远"。

## 代表工作

- [[RePO-VLA]]: 提出 PAS-VF，为 VCR 阶段提供帧级值标签

## 相关概念

- [[Value-Conditioned Refinement]]
- [[VJEPA-2]]
- [[Cosine Similarity]]
- [[Value Function]]

---
type: concept
aliases: [基于RGB的世界动作模型, RGB World Action Model, Appearance-based WAM]
---

# RGB-based WAM

## 定义
基于 RGB 的世界动作模型（RGB-based WAM）是在 **RGB 外观空间**进行世界状态展开的一类方法，通过预测未来 RGB 帧来表示世界演化，再从中生成动作。

## 数学形式

$$
\pi_\theta(a_{1:H} \mid r_{\leq 0}, c) = \int p_\theta(a_{1:H} \mid r_{1:F}, r_{\leq 0}, c) \cdot p_\theta(r_{1:F} \mid r_{\leq 0}, c) \, dr_{1:F}
$$

即状态轨迹 $x_{1:F}$ 用 RGB 帧序列 $r_{1:F}$ 表示。

## 核心要点

1. **外观纠缠问题**: RGB 表示将语义（颜色、纹理）与物理配置（位姿、占据）混合，难以分离
2. **随机化场景脆弱**: 在颜色/纹理随机化场景下成功率急剧下降（FastWAM 仅 1.90%）
3. **预训练数据充足**: 大量无标注 RGB 视频可用于预训练，是 RGB-based WAM 的主要优势
4. **代表方法**: FastWAM、OpenWAM-α 等

## 代表工作
- [[FastWAM]]: 典型 RGB-based WAM，随机化场景仅 1.90%
- [[OpenWAM]]: 带动作预训练的 RGB-based WAM

## 相关概念
- [[World Action Model]]
- [[Decoupled Representation]]
- [[Depth Prediction]]
- [[PrismWAM]]

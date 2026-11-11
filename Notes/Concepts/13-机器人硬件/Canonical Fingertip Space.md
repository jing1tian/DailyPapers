---
type: concept
aliases: [规范指尖空间, 规范指尖坐标, Canonical Fingertip Space]
---

# Canonical Fingertip Space（规范指尖空间）

## 定义

LingBot-Touch 中用于统一异构触觉传感器空间布局的参考坐标系：将每种传感器的 taxel 坐标从传感器特定的索引/几何空间仿射变换至以单位半径球为参考的共享手指局部坐标系，从而使不同传感器的空间位置具有可比性。

## 数学形式

$$
p_i = \alpha \tilde{p}_i + b
$$

- $\tilde{p}_i$：经朝向对齐后的原始 taxel 坐标
- $\alpha > 0$：缩放系数（PaXini/XHand1：$\alpha = 1/r$，$r$ 为最小外接球半径）
- $b \in \mathbb{R}^3$：平移向量（PaXini/XHand1：$b = -o/r$，$o$ 为球心）
- 坐标原点 $o = (u_x, c_y, u_z)$，$u$ 为接触中心锚点，$y$ 为手指方向

## 核心要点

1. **传感器无关**：taxel 位置以连续 3D 坐标 + 手指标识（5 维）描述，而非传感器内部排序索引，同一物理位置在不同传感器上坐标相近
2. **参数离线计算**：每个传感器–手指模板的 $\alpha$ 和 $b$ 由几何资产计算一次后固定，不参与梯度优化
3. **多传感器对齐方法**：
   - PaXini / XHand1：最小外接球归一化
   - Sharpa Wave：等高度对齐 + 深度平移匹配参考球
   - Juqiao / Wujie：使用固定的 PaXini 参考半径和接触区域锚点
   - Tachin：参考球内最大各向同性拟合
4. **与 Layout 增广配合**：规范坐标 + 位置扰动（$p'_i = p_i + \delta_i, \|\delta_i\|_2 \le 0.5 d_i$）防止模型记忆特定 taxel 坐标模板，提升对未见传感器的泛化
5. **零样本迁移**：未见于预训练的 XHand1 磁性传感器直接使用磁性 tokenizer 和相同规范坐标，无任何参数更新即可获得良好接触物理表征

## 代表工作

- [[LingBot-Touch]]: 提出并验证规范指尖球面空间用于异构触觉预训练

## 相关概念

- [[触觉传感器]]: 本概念用于统一多类触觉传感器布局
- [[Haptic Transformer]]: 通过 5D RoPE 注入规范坐标位置信息
- [[Tactile-JEPA]]: 训练时使用规范坐标作为 token 地址

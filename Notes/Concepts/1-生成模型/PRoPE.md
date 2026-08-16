---
type: concept
aliases: [Projective Rope Embeddings, 投影旋转位置编码, Projective RoPE]
---

# PRoPE（Projective Rope Embeddings）

## 定义

将相机内参和外参通过投影矩阵注入 [[RoPE|旋转位置编码]] 的自注意力机制，使视频扩散模型能够感知并跟随指定相机轨迹。

## 数学形式

**提升投影矩阵**：

$$
\tilde{P}_i = \begin{bmatrix} K_i & \mathbf{0} \\ T_i^{cw} \\ \mathbf{e}_4^T \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

**块对角变换**：

$$
D_t^{\text{PRoPE}} = \begin{bmatrix} I_{d/8} \otimes \tilde{P}_{i(t)} & \mathbf{0} \\ \mathbf{0} & \begin{bmatrix} \text{RoPE}_{d/4}(x_t) & \mathbf{0} \\ \mathbf{0} & \text{RoPE}_{d/4}(y_t) \end{bmatrix} \end{bmatrix}
$$

**相机感知注意力**：

$$
\text{Attn}_{\text{PRoPE}}(Q,K,V) = D^{\text{PRoPE}} \odot \text{Attn}\bigl((D^{\text{PRoPE}})^T \odot Q,\ (D^{\text{PRoPE}})^{-1} \odot K,\ V\bigr)
$$

## 核心要点

1. 同时编码相机内参（$K_i$）和外参（$T_i^{cw}$），token 对之间隐式包含相对投影变换
2. 以块对角形式与空间 RoPE 并行注入，不改变原始注意力结构
3. 通过 GTA（Grouped Token Attention）形式实现，保持与原始自注意力的兼容性

## 代表工作

- [[minWM]]: 在 HY1.5 和 Wan2.1 上验证 PRoPE 用于相机控制微调
- [[G3VLA]]: 将 PRoPE 思想用于机器人多相机 VLA 的跨视角注意力 bias，从相机内外参推导投影变换分别作用于 Query/Key/Value，使跨视角几何关系由相机模型而非纯学习到的外观相似度决定
- [[DreamX-Phi]]: 将 PRoPE 推广至 SE(3) 机器人末端执行器动作条件注入，以臂分组形式将双臂轨迹几何结构注入注意力头，实现双臂身份保持和动作-视觉精确对应

## 相关概念

- [[RoPE]]: 旋转位置编码，PRoPE 的设计基础
- [[3D RoPE]]: 三维扩展版本
- [[视频生成世界模型]]: 应用场景

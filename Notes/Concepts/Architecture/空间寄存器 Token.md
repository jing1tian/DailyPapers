---
type: concept
aliases: [Spatial Register Token, 寄存器Token, Register Token]
---

# 空间寄存器 Token

## 定义

一种可学习的轻量级 token，在训练时作为几何读出器将预训练深度估计模型的空间先验蒸馏进视频 Transformer，推理时移除以保持高效性。

## 数学形式

$$
\mathbf{G}_t = \mathcal{P}_g\!\left(\{\mathbf{R}^{\ell+1}_t\}_{\ell \in \mathcal{L}_r}\right),\quad \hat{\mathbf{D}}^{fut}_t = \mathcal{G}_\phi(\mathbf{G}_t)
$$

## 核心要点

1. **训练时存在，推理时消失**：作为训练-部署之间的紧凑桥梁，蒸馏完成后移除整个几何分支
2. **空间网格排布**：按 12×10 网格重复至每个未来深度时间步，与多视角拼图坐标对齐
3. **交叉注意力更新**：通过深度提取块（自注意力 + 交叉注意力）从历史视频特征获取信息
4. **几何先验迁移**：损失信号经预训练几何头反向传播，将 3D 空间感知注入视频表示

## 代表工作

- [[WAM4D]]: 提出 Spatial Register Distillation，将 Depth Anything 3 先验蒸馏进 MoT WAM 骨干

## 相关概念

- [[知识蒸馏]]
- [[Depth Anything 3]]
- [[因果混合注意力]]
- [[World Action Model]]

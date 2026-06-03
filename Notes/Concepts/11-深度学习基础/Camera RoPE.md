---
type: concept
aliases: [Camera RoPE, 相机旋转位置编码, 视角旋转位置编码]
---

# Camera RoPE

## 定义

Camera RoPE 是对标准旋转位置编码（RoPE）的扩展，通过在频率库中加入视角轴（view axis），为多相机系统中的每个摄像头提供无需外参标定的身份位置表示。

## 数学形式

在原始 3D RoPE 的 $(f, h, w)$ 三轴基础上，扩展为 $(f, h, w, \text{view})$ 四轴频率分区：

$$
\text{RoPE}_{4D}(\mathbf{x}, f, h, w, v) = \text{RoPE}(\mathbf{x}; \theta_{f,h,w,v})
$$

每个视角的嵌入为可学习的、跨层共享的参数，不依赖相机内外参。

## 核心要点

1. 将 RoPE 的频率库在视角维度（view axis）上进行分区，每个视角占据独立的频率区间
2. 可学习的每视角嵌入在所有 Transformer 层间共享，减少参数量
3. 无需相机标定矩阵即可表示不同相机的身份差异，支持多机身（multi-embodiment）部署
4. 与 [[交叉注意力|跨视图自注意力]] 配合，实现几何感知的多视图特征融合

## 代表工作

- [[WALL-WM]]: 提出 Camera RoPE，用于多视图视频 DiT 的无标定多相机身份编码

## 相关概念

- [[交叉注意力]]
- [[多视图几何建模]]
- [[多视图适配]]

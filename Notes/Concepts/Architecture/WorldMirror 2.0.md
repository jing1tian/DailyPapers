---
type: concept
aliases: [WorldMirror2, 世界镜像重建, Unified 3D Reconstruction]
---

# WorldMirror 2.0

## 定义

HY-World 2.0 的统一 3D 重建模块，单次前向传播同时预测点云、深度图、表面法线、相机参数和 3DGS 属性，支持任意分辨率输入。

## 核心要点

1. **归一化 RoPE**：将 patch 坐标归一化到 $[-1, 1]$，将分辨率外推转化为插值，实现跨分辨率推理
2. **Depth-to-Normal 耦合监督**：从预测深度导出法线，通过角度损失增强几何一致性
3. **深度掩码预测头**：专用头预测无效像素（传感器噪声/遮挡），提升鲁棒性
4. 共享 Transformer 骨干 + 任务特定 DPT 解码头
5. 三阶段课程训练：联合几何学习 → 冻结几何 + 3DGS 训练
6. Token-budget 动态分辨率采样（50K–500K 像素）

## 代表工作

- [[HYWorld2]]: HY-World 2.0 四阶段流水线的第四阶段（3D 重建）

## 相关概念

- [[3D Gaussian Splatting]]
- [[Rotary Position Encoding]]
- [[Depth-to-Normal Loss]]
- [[Depth-to-Normal Conversion]]
- [[WorldStereo 2.0]]

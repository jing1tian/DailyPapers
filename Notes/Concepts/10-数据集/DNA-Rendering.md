---
type: concept
aliases: [DNA Rendering, Dynamic Neural Avatar Rendering]
---

# DNA-Rendering

## 定义
大规模动态人体渲染数据集，提供高质量多视角视频用于评估动态神经人体重建方法。

## 核心要点
1. 包含密集多摄像机阵列拍摄的人体表演序列，视角数量远多于普通数据集
2. 提供精确的人体分割掩码、相机参数、SMPL 拟合结果
3. 测试集评估指标：NV-PSNR（新视角渲染质量）、Gen. Video Consistency（生成视频一致性）
4. 动态人体重建领域的主要 benchmark 之一

## 代表工作
- [[4DAnyone]]：在此数据集上验证多视角一致性和 4DGS 重建质量

## 相关概念
- [[DyMVHumans]]
- [[4DGS]]
- [[HMR]]

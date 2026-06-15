---
type: concept
aliases: [DPFlow光流, Dense Prediction Flow]
---

# DPFlow

## 定义
一种用于机器人操作场景的光流估计方法，从多视角 RGB 图像序列中预测稠密运动场，为世界动作模型提供运动先验监督信号。

## 数学形式

$$
\Phi_t = \text{DPFlow}(o_t, o_{t+1}) \in \mathbb{R}^{H \times W \times 2}
$$

其中 $\Phi_t$ 为帧间稠密光流场，$H \times W$ 为图像分辨率。

## 核心要点
1. 多视角独立估计，配合视角嵌入（view embedding）进行融合
2. 跨摄像头分辨率归一化处理，保证多视角一致性
3. 训练时作为监督信号，推理时无需实时估计（仅离线预处理）
4. 捕获任务完成所需的运动信息，而非视觉外观细节

## 代表工作
- [[HiMem-WAM]]: Stage I 低层 tokenizer 使用 DPFlow 作为运动先验

## 相关概念
- [[Optical Flow]]
- [[World Action Model]]
- [[Variational Autoencoder]]

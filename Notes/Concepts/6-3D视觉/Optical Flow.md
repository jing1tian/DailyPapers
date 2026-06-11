---
type: concept
aliases: [光流, 光流估计, Optic Flow]
---

# Optical Flow

## 定义
描述图像序列中像素运动的稠密向量场，表示图像中每个像素从一帧到下一帧的位移量，是捕获视觉运动信息的基础技术。

## 数学形式

$$
\Phi(x, y, t) = (u, v) \quad \text{s.t.} \quad I(x+u, y+v, t+1) \approx I(x, y, t)
$$

其中 $(u, v)$ 为像素 $(x,y)$ 的运动向量，$I$ 为图像强度。

## 核心要点
1. 光流满足亮度恒常性假设（brightness constancy）
2. 稀疏光流（如 Lucas-Kanade）追踪特征点；稠密光流（如 RAFT）计算全图运动场
3. 在机器人操作中用于捕获运动先验，比原始 RGB 更专注于动作语义
4. 可归一化处理以适应不同相机分辨率

## 代表工作
- [[HiMem-WAM]]: 使用 [[DPFlow]] 提取多视角光流，作为低层 tokenizer 的监督信号

## 相关概念
- [[DPFlow]]
- [[Variational Autoencoder]]
- [[World Action Model]]

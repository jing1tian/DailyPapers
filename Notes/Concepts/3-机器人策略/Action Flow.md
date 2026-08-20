---
type: concept
aliases: [动作流, Action Flow, 像素动作表示]
---

# Action Flow

## 定义
将机器人关节指令映射为图像平面内可见机器人表面点的运动轨迹，作为跨本体通用的视觉动作接口。

## 数学形式

$$
\mathbf{x}_{n,t} = \pi\!\left(\mathbf{K}\,\begin{bmatrix}\mathbf{I}_{3}&\mathbf{0}\end{bmatrix}\mathbf{T}_{CW}\,\mathbf{T}_{\ell(n)}(\mathbf{q}_{t})\,\bar{\mathbf{X}}_{n}\right)
$$

Action Flow $\mathcal{F} = \{(\mathbf{x}_{n,0}, \ldots, \mathbf{x}_{n,H})\}_{n=1}^N$，即 $N$ 个表面点在 $H+1$ 时刻的像素轨迹集合。

## 核心要点
1. **本体无关**: 像素运动轨迹对本体形态无假设，单臂/双臂/人手均可统一表示
2. **双路径构造**: 在线部署用机器人运动学 + 相机标定精确计算；离线训练用稠密光流 + grounded 分割恢复
3. **可执行性**: 与关节命令双射，部署时可直接从命令空间生成预测，不依赖无监督轨迹
4. **四种采样模式**: Embodiment (p=0.40) / Object (p=0.40) / All (p=0.15) / None/dropout (p=0.05)

## 代表工作
- [[Hydra-0]]: 提出 Action Flow 概念，用于跨四种本体的世界模型统一训练

## 相关概念
- [[Optical Flow]]: 视频追踪时恢复 Action Flow 的底层工具
- [[Motion Conditioning]]: 基于 Action Flow 构建视频生成的运动条件
- [[IsaacLab]]: 在线部署时执行命令获取机器人变换

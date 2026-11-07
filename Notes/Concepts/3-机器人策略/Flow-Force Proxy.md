---
type: concept
aliases: [流力代理, 光流力代理, Flow-Force Proxy]
---

# Flow-Force Proxy（流力代理）

## 定义

从触觉传感器图像（如 GelSight）的光学流场中导出的轻量 3D 力代理信号，用切向位移均值（$f_x, f_y$）和法向散度（$f_z$）近似表示接触力，无需力传感器或学习式力估计器。

## 数学形式

$$
f_{x,k} = \mathbb{E}_{\mathbf{p}}[u_{x,k}(\mathbf{p})], \quad
f_{y,k} = \mathbb{E}_{\mathbf{p}}[u_{y,k}(\mathbf{p})], \quad
f_{z,k} = \mathbb{E}_{\mathbf{p}}[\nabla \cdot u_k(\mathbf{p})]
$$

$$
F_k^v = [f_{x,k},\; f_{y,k},\; f_{z,k}]^\top \in \mathbb{R}^3
$$

其中 $u_k(\mathbf{p})$ 为当前触觉帧与无接触参考帧之间的光学流向量。

## 核心要点

1. **无需力传感器**: 利用触觉图像的光学流代替昂贵/难以安装的力矩传感器（尤其适合并联夹爪）
2. **物理可解释性**: $f_x, f_y$ 对应切向摩擦力，$f_z$（散度）对应法向压力
3. **监督信号**: 在 WAM 训练中作为联合去噪目标之一，引导模型主动关注接触动态
4. **注意力调制效果**: 加入该监督后，Action Expert 在接触阶段对触觉视图的交叉注意力权重显著提高（0.22 → 0.26）

## 代表工作

- [[VTAM]]: 提出 Flow-Force Proxy 并验证其在接触丰富操作中的有效性

## 相关概念

- [[Optical Flow]]
- [[GelSight]]
- [[世界动作模型（WAM）]]
- [[Action Expert]]
- [[Visuo-Tactile World Models]]

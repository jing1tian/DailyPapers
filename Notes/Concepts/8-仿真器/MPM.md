---
type: concept
aliases: [Material Point Method, 物质点法]
---

# MPM (Material Point Method)

## 定义
混合 Lagrangian-Eulerian 的物理仿真方法，将连续介质（流体、固体、颗粒）离散为物质点，在背景网格上求解运动方程，特别适合模拟大变形和拓扑变化。

## 数学形式
物质点到网格（P2G）插值：
$$m_i v_i = \sum_p m_p v_p w_{ip}$$
动量更新（网格上）：
$$v_i^{n+1} = v_i^n + \frac{\Delta t}{m_i} f_i$$
网格到物质点（G2P）更新：
$$v_p^{n+1} = \sum_i v_i^{n+1} w_{ip}$$

## 核心要点
1. 无网格撕裂：拓扑无关，自动处理断裂、融合
2. 多材质统一框架：弹性体（Neo-Hookean）、流体、雪、沙土
3. GPU 加速（Taichi lang 等）使实时仿真成为可能
4. 在可形变物体操作仿真中广泛使用

## 代表工作
- [[DeformMaster]]：用 MPM 作为物理引擎，从视频中估计材质参数并做交互式预测

## 相关概念
- [[FEM]]
- [[Spring-Gaus]]

---
type: concept
aliases: [Recurrent All-Pairs Field Transforms, 光流估计]
---

# RAFT

## 定义
RAFT（Recurrent All-Pairs Field Transforms）是 ECCV 2020 提出的光流估计方法，通过构建全对相关体积（all-pairs correlation volume）并用 GRU 迭代更新光流场，在精度和速度上超越了此前 SOTA。

## 数学形式
$$
\mathbf{f}_{t+1} = \mathbf{f}_t + \Delta\mathbf{f}_t, \quad \Delta\mathbf{f}_t = \text{GRU}(\mathbf{h}_t, C(\mathbf{f}_t))
$$

其中 $C(\mathbf{f}_t)$ 为在当前光流 $\mathbf{f}_t$ 位置查询的 4D 相关体积特征。

## 核心要点
1. **全对相关体积**：对所有像素对计算点积相关，形成 $H \times W \times H \times W$ 体积，支持任意位移的查询
2. **迭代精化**：GRU 在固定相关体积上迭代更新光流，而非单次前向传播
3. **多尺度查询**：平均池化得到多分辨率相关体积，支持大位移
4. **分辨率不变**：在原始分辨率 1/8 的特征图上运行，保留细节

## 代表工作
- [[RAFT]]: Teed & Deng, ECCV 2020，奠定现代光流估计范式
- [[FlowWAM]]: 将光流作为 WAM 中的统一动作表示，基于 RAFT 思路

## 相关概念
- [[FlowMatching]]
- [[3D Motion Flow]]

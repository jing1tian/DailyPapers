---
type: concept
aliases: [CaPE, Camera Positional Encoding, 相机位置编码]
---

# Camera Positional Encoding (CaPE)

## 定义

Camera Positional Encoding（CaPE）是一种将相机几何信息（外参矩阵和内参矩阵）注入神经网络的位置编码技术，在机器人操控的跨具身训练中用于将末端执行器动作与视觉观测的相机坐标系对齐，使不同机器人平台的视觉相似动作在动作空间中也数值相近。

## 核心要点

1. **动机**: 不同机器人的基坐标系差异导致同一视觉场景下的相似操作动作在机器人基坐标系中数值差异巨大，难以跨具身统一学习
2. **实现方式**: 将相机外参 $[R|t]$ 和内参 $K$ 编码后注入动作专家（DiT）的交叉注意力层，作为几何条件
3. **与动作表示结合**: 搭配相机坐标系下的末端执行器增量位姿（Camera-Frame Delta EEF Pose），实现视觉-动作几何一致性
4. **跨具身效果**: 安装在相似位置的相机，其坐标系下的抓取动作数值相近，从而让跨平台联合训练受益

## 数学形式

相机坐标系下的末端执行器增量位姿（Camera-Frame Delta Pose）：

$$
a_p = \begin{bmatrix} {}^c_e R\, {}^e_{e'} R\, {}^e_c R & {}^c_e R(t_{ee'} - t_{ec}) \\ \mathbf{0}^\top & 1 \end{bmatrix}
$$

- ${}^c_e R$: 末端执行器到相机坐标系的旋转
- ${}^e_{e'} R$: 末端执行器的旋转增量
- $t_{ee'}, t_{ec}$: 末端执行器位移和相机到末端距离

## 代表工作

- [[Qwen-RobotManip]] (2026): 首次系统性提出 CaPE 并用于跨 15 种机器人平台的大规模联合训练，是三维对齐框架中的运动对齐核心

## 相关概念

- [[Cross-Embodiment]]: CaPE 是实现跨具身动作对齐的核心工具
- [[Flow Matching]]: Qwen-RobotManip 中与 CaPE 配合使用的动作生成方法
- [[Diffusion Transformer]]: 注入 CaPE 信息的动作专家架构

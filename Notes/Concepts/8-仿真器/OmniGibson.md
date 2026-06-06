---
type: concept
aliases: [OmniGibson, Gibson Environment]
---

# OmniGibson

## 定义
Stanford 开发的高保真机器人仿真平台，基于 NVIDIA Omniverse/Isaac Sim 构建，支持真实物理（布料、液体、关节体）和光线追踪渲染，是 iGibson 的升级版。

## 数学形式
$$\text{—（平台工具，无核心公式）}$$

## 核心要点
1. 基于 [[IsaacLab]]/NVIDIA Omniverse 物理引擎，支持刚体、软体、流体仿真
2. 提供大量 photorealistic 室内场景（300+ 房间，100+ 物体类别）
3. 适合 navigation + manipulation 联合任务，支持全身机器人（包括 humanoid）
4. 与 [[MuJoCo]] 对比：物理精度更高、场景更丰富，但速度更慢
5. [[StereoPolicy]] 使用 OmniGibson 进行立体视觉 manipulation 仿真评估

## 代表工作
- Li et al. (2023): OmniGibson 原始论文
- [[StereoPolicy]]: 使用 OmniGibson 的 RoboCasa 场景

## 相关概念
- [[MuJoCo]]
- [[IsaacLab]]

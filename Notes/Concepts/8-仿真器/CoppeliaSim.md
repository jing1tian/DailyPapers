---
type: concept
aliases: [V-REP, CoppeliaSim Simulator]
---

# CoppeliaSim

## 定义
Coppelia Robotics 开发的通用机器人仿真平台（前称 V-REP），支持多种物理引擎（Bullet、ODE、Vortex、Newton），提供 Python/Lua/C++ API，常用于操控任务的仿真验证。

## 核心要点
1. 支持多物理引擎可切换，模拟不同刚体/软体场景
2. 内置逆运动学求解和路径规划模块
3. 相比 MuJoCo 更灵活但精度略低；相比 IsaacLab 不支持 GPU 并行
4. 提供远程 API，可与 ROS/Python 策略代码联动
5. LACY、多篇 VLA 论文用 CoppeliaSim 作为仿真测试平台

## 代表工作
- [[LACY]]: 在 CoppeliaSim 中验证语言-动作循环的自我进化

## 相关概念
- [[MuJoCo]]
- [[Isaac Lab]]

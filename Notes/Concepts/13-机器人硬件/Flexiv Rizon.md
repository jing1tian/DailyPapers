---
type: concept
aliases: [Flexiv Rizon 4S, Flexiv 机器人臂, Flexiv]
---

# Flexiv Rizon

## 定义

Flexiv 公司开发的力控工业机械臂系列，具备高精度力-位混合控制能力，常用于精密装配和双臂操作研究。

## 核心要点

1. **型号**: Rizon 4S（7自由度），具备末端执行器力矩传感器
2. **力控能力**: 原生支持阻抗控制和力-位混合控制，适合接触丰富任务
3. **双臂配置**: 常以两臂对称布置，末端搭配 Robotiq 夹爪
4. **视觉集成**: 配合 Intel RealSense D435i（正面）和 D405（腕部）相机
5. **动作空间**: 14 维双臂格式（每臂 7D：末端执行器位姿 6D + 夹爪状态 1D）

## 代表工作

- [[A2World]]: 使用双臂 Flexiv Rizon 4S 平台进行 5 个操作任务的真机实验（举盒、翻盒、插 RAM、拨开关、链条入盒）

## 相关概念

- [[Action Conditioning]]
- [[Action-Conditioned World Model]]

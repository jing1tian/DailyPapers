---
type: concept
aliases: [LinkTouch, LinkTouch F/T sensor, 六轴力矩传感器]
---

# LinkTouch 六轴 F/T 传感器

## 定义

LinkTouch 是一款安装于机械臂腕部的六轴力/力矩（F/T）传感器，可同时测量三轴力（$F_x, F_y, F_z$）和三轴力矩（$\tau_x, \tau_y, \tau_z$）。

## 核心要点

1. **采样率**: 150 Hz，CW-WAM 中经 10 Hz 截止频率二阶 Butterworth 低通因果滤波器处理。
2. **补偿**: 使用前需进行偏置补偿（静态偏置去除）和工具重力补偿（消除末端执行器自重引起的虚假力矩）。
3. **坐标系**: 力矩以预测 TCP 坐标系表达，力矩参考点为 TCP 原点。

## 代表工作

- [[CW-WAM]]: 使用 LinkTouch 采集腕部力矩数据用于 WAM 训练和实时控制

## 相关概念

- [[Contact-Wrench]]
- [[导纳控制（Admittance Control）]]

---
type: concept
aliases: [Diablo, Direct Drive Tech Diablo]
---

# Diablo 机器人

## 定义

Direct Drive Tech 公司开发的轮式/腿式机器人平台，以直驱电机（Direct Drive）为特色，常用于导航和移动操作研究。

## 核心要点

1. **驱动方式**: 直驱电机，低延迟力矩控制
2. **传感器配置（NavWAM 版本）**: Intel RealSense D455 RGB-D 相机
3. **计算单元**: NVIDIA Jetson AGX Orin（边缘推理）
4. **接口**: ROS 2 cmd_vel（局部坐标系速度指令）
5. **测试环境**: 办公室、储藏室、会议室、走廊

## 代表工作

- [[NavWAM]]: 在 Diablo 上验证目标条件视觉导航，79.2% 成功率

## 相关概念

- [[导航世界模型]]
- [[目标条件策略]]

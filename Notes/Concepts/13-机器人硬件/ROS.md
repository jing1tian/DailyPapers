---
type: concept
aliases: [ROS, Robot Operating System, 机器人操作系统]
---

# ROS

## 定义

ROS（Robot Operating System）是机器人软件开发的事实标准中间件框架，提供节点间通信（topic/service）、硬件抽象、可视化调试等工具集，广泛用于机器人系统的感知-决策-控制集成。

## 核心要点

1. **中心化通信架构**: 经典 ROS1 依赖 roscore 主节点协调话题（topic）通信；ROS2 改进为基于 DDS 的去中心化发现机制
2. **生态成熟**: 拥有丰富的驱动包、可视化工具（RViz）、仿真集成（Gazebo），是学术与工业机器人系统的常用基础设施
3. **与轻量级方案的对比**: 在大规模并行数据采集场景中，部分工作选择更轻量的去中心化方案（如 [[ZeroMQ]] PUB/SUB）替代完整 ROS 栈以降低部署与维护成本

## 代表工作

- [[ABC]]: 在论述其机器人控制框架设计时，将自建的 ZeroMQ 去中心化 PUB/SUB 方案与传统 ROS 架构对比，强调轻量化部署的优势

## 相关概念

- [[ZeroMQ]]
- [[YAM Robot]]

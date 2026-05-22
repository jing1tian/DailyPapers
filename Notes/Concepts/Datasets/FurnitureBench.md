---
type: concept
aliases: [Furniture Bench]
---

# FurnitureBench

## 定义

FurnitureBench 是一个真实机器人操作 benchmark，专注于家具组装任务（如椅子、桌子组件的精确插接），使用 [[AprilTag]] 实现高可复现性，但硬件成本较高。

## 核心要点

1. 任务集中于精密组装（tight-tolerance assembly），需要厘米级精度
2. 基于 [[AprilTag]] 标定确保场景可复现
3. 硬件成本高，限制了跨实验室推广
4. 提供专家演示数据集

## 代表工作

- [[VLA-REPLICA]]: 将 FurnitureBench 作为对比，指出其高成本是复现障碍

## 相关概念

- [[AprilTag]]
- [[模仿学习]]

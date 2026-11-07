---
type: concept
aliases: [RoboTwin UMI, RoboTwin-UMI Benchmark]
---

# RoboTwin-UMI

## 定义
基于 [[RoboTwin]] 仿真框架为 UMI 夹爪策略专门构建的 20 任务仿真基准，测试环境变化（光照随机化）和 embodiment 迁移（Aloha-AgileX → ARX X5）下 UMI 策略的泛化能力。

## 核心要点
1. 基于 [[RoboTwin]] 构建，同时支持 Aloha-AgileX 和 ARX X5 两种 embodiment
2. 两种机器人均配备 UMI 夹爪和腕部鱼眼摄像头，使用 KB4 鱼眼模型合成摄像头图像
3. 评测维度：in-domain（Aloha, clean）、光照随机化（Aloha, rand. light）、embodiment+光照迁移（ARX X5, rand. light）
4. 每个 embodiment 下 20 个任务，步数限制 400–1400 步，训练时每任务提供 50 次干净演示

## 评测结果（主要方法对比）

| 方法 | Aloha Clean | Aloha Rand. Light | ARX X5 Rand. Light |
|------|------------|-------------------|--------------------|
| [[π₀.₅]] | 74.45% | 66.15% | 61.85% |
| [[Spatial-UMI-Dream]] | 79.10% | 73.35% | 72.15% |

## 代表工作
- [[Spatial-UMI-Dream]]: 提出并首次使用 RoboTwin-UMI 进行系统化 UMI 策略泛化评测

## 相关概念
- [[RoboTwin]]: 基础仿真框架
- [[UMI]]: 目标硬件平台

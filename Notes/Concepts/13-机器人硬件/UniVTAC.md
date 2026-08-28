---
type: concept
aliases: [UniVTAC tactile sensor, 触觉传感器]
---

# UniVTAC

## 定义
用于机器人操作的视觉-触觉传感器，提供接触点位置、法向力、切向力等实时触觉信号。

## 核心要点
1. 视觉-触觉融合传感器，基于相机采集接触变形图像推断触觉信息
2. 提供实时触觉反馈，适合接触密集型操作任务
3. 与 VLA 策略结合时，触觉信号通过 EATA 等适配模块注入

## 代表工作
- [[TacForcing]]: 使用 UniVTAC 做 execution-time 触觉反馈

## 相关概念
- [[TacForcing]]
- [[EATA]]

---
type: concept
aliases: [PySceneDetect, Scene Detect]
---

# PySceneDetect

## 定义
一个开源 Python 库，用于自动检测视频中的镜头切换（scene cut）和渐变（fade/dissolve），将长视频分割为独立镜头片段，常用于视频生成模型的训练数据预处理管线。

## 核心要点
1. 支持基于阈值的内容检测（content-aware）和基于帧差的检测
2. 可输出每个 scene 的时间戳和截图，方便下游标注和过滤
3. LongCat-Video 使用 PySceneDetect 对训练视频进行镜头分割

## 代表工作
- [[LongCat]]: 训练数据管线中使用 PySceneDetect 切割视频

## 相关概念
- [[LongCat]]
- [[Open-Sora]]

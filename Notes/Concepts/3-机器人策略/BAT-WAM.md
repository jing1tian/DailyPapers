---
type: concept
aliases: [Bridging Action and Touch WAM]
---

# BAT-WAM

## 定义
将触觉信号集成到 World-Action Model 框架的方法，专为灵巧手操作设计，通过 HandTouch 模块实现触觉-视觉-动作联合建模。

## 核心要点
1. 基于 [[OpenWAM]] 框架，新增 HandTouch 触觉编码模块
2. 触觉传感器压力分布经 encoder 后作为额外 token 注入 WAM
3. 世界状态预测同时覆盖视觉和触觉未来状态
4. 在 Rimbot 灵巧手平台上做了真实实验

## 代表工作
- [[BAT-WAM]]: 原始论文（Fudan/Tsinghua/Southeast Univ，2611.10967）

## 相关概念
- [[WAM]]
- [[OpenWAM]]
- [[VLA]]

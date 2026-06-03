---
type: concept
aliases: [OmniPaint]
---

# OmniPaint

## 定义
通用图像 inpainting 模型，在 RoboDream 中用于场景背景重绘，将操作场景的背景替换为新环境。

## 核心要点
1. 给定 mask 区域和上下文，生成视觉一致的填充内容
2. 与 SAM 配合：SAM 分割物体 mask，OmniPaint 重绘背景
3. 用于机器人数据合成中的域随机化

## 代表工作
- [[RoboDream]]: 背景重绘子模块

## 相关概念
- [[AnchorDream]]
- [[SAM]]

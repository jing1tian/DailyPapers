---
type: concept
aliases: [AnchorDream]
---

# AnchorDream

## 定义
RoboDream 中的组合式物体放置模块，将目标物体锚定到场景中的指定位置，再与背景合成，实现物体位置可控的合成演示生成。

## 核心要点
1. 解耦物体放置与背景生成，分别控制
2. 结合 SAM 分割 + OmniPaint 背景重绘
3. 组合生成避免 holistic WM 的整体一致性问题

## 代表工作
- [[RoboDream]]: AnchorDream 作为物体放置子模块

## 相关概念
- [[OmniPaint]]
- [[DreamGen]]
- [[SAM]]

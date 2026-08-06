---
type: concept
aliases: [Video Twin VAE]
---

# VidTwin

## 定义
VidTwin：将视频分解为 content（静态场景）和 motion（动态变化）两个解耦表征的视频 VAE 架构。

## 核心要点
1. 双分支 VAE：content encoder + motion encoder
2. content 捕捉静态背景，motion 捕捉帧间变化
3. 解耦表征提高视频生成的可控性

## 代表工作
- [[EmbodiedVAE]]: 继承 VidTwin 解耦思路，专为 manipulation 场景优化

## 相关概念
- [[EmbodiedVAE]]
- [[LVDM]]

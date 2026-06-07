---
type: concept
aliases: [GENMO, Generative Motion]
---

# GENMO

## 定义

GENMO 是一种人体运动估计模型，能够从单目视频中逐帧估计 SMPL-X 人体参数（包括身体姿态、手型、面部），为 4D HOI 重建提供初始人体运动轨迹。

## 核心要点

1. 输入：单目 RGB 视频序列
2. 输出：逐帧 [[SMPL-X]] 参数 $\bm{\Theta}^{\mathcal{H}}_t$（全身姿态、手型、面部）
3. 在 GRAIL 中作为初始估计，后续通过联合优化进一步精化
4. 手部细节由 [[WiLoR]] 独立精化，提高手指姿态质量

## 代表工作

- [[GRAIL]]: 使用 GENMO 作为 Stage 2 人体运动初始估计的核心工具

## 相关概念

- [[SMPL-X]]
- [[WiLoR]]
- [[运动捕捉]]
- [[FoundationPose]]

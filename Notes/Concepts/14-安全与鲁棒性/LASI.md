---
type: concept
aliases: [Low-Frequency Action Source Imprinting, 低频动作源印记]
---

# LASI

## 定义
Low-Frequency Action Source Imprinting：生成式控制中的失效模式，采样动作源的低频分量系统性地偏转生成的相机运动，即使条件信号固定也会产生错误的粗粒度动作。

## 核心要点
1. 在 flow-matching 动作生成中，采样噪声的低频成分会泄露到输出动作序列
2. 导致相机/末端运动在宏观方向上出现系统性偏差，即使输入条件相同
3. 揭示了生成式控制中 source sensitivity 这一潜在失效模式
4. 与具体任务内容无关，是架构层面的归纳偏置问题

## 代表工作
- [[GameWAM]]: 首次在视频游戏 WAM 中发现并命名 LASI

## 相关概念
- [[Flow Matching]]
- [[WAM]]

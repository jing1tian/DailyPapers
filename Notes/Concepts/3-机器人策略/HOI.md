---
type: concept
aliases: [Human-Object Interaction, 人物交互]
---

# HOI (Human-Object Interaction)

## 定义
研究人类与物体之间接触、抓取、操作等交互行为的建模与重建任务。

## 核心要点
1. 核心挑战：深度歧义、人体形态与机器人形态不匹配、遮挡
2. 数据来源：动捕系统（GRAB、BEHAVE）或 RGB 视频重建
3. 4D HOI 重建需同时估计人体姿态 + 物体 6DoF 轨迹 + 时序一致性
4. 应用于人形机器人 loco-manipulation：将 HOI 轨迹 retarget 到机器人关节

## 数学形式
$$\text{HOI} = \{(\mathbf{q}_t^h, \mathbf{T}_t^o, \mathbf{c}_t) \mid t = 1, \ldots, T\}$$
其中 $\mathbf{q}_t^h$ 为人体关节角，$\mathbf{T}_t^o$ 为物体变换矩阵，$\mathbf{c}_t$ 为接触标签

## 代表工作
- [[GRAIL]]: 用视频生成 + 3D 资产合成 HOI 数据训练人形机器人

## 相关概念
- [[SMPL]]
- [[MPJPE]]
- [[CHOIS]]
- [[ResMimic]]

---
type: concept
aliases: [人类到机器人合成, 第一人称数据合成, Egocentric-to-Robot, Human-to-Robot Data Synthesis]
---

# Human-to-Robot Synthesis

## 定义

将人类第一视角（egocentric）视频数据转化为机器人训练轨迹的流水线：将人手关键点映射到机器人末端执行器位姿，通过图像合成技术将原始视频中的人手替换为机器人手臂，得到可直接用于 VLA 训练的机器人操控演示数据。

## 核心要点

1. **运动重定向**: 3D 手部关键点 → 末端执行器位姿 + 夹爪命令 → 逆运动学（IK）计算关节角度。
2. **视觉合成**: 使用 SAM3 分割手臂、ProPainter 重建被遮挡背景、DA3 深度辅助渲染机器人并合成到原始场景。
3. **规模优势**: 人类第一视角数据（如 EgoDex）远多于机器人演示数据，可低成本扩充训练集。
4. **局限性**: 合成图像存在域差异（渲染质量、光照等），可能导致视觉分布偏移。

## 代表工作

- [[PDRVLA]]: 将 EgoDex 74,813 条演示合成为 896,832 条双臂机器人轨迹（约 2,646 小时）
- [[Qwen-RobotManip]]: 原始合成流水线来源

## 相关概念

- [[Human-to-Robot Retargeting]]
- [[Cross-Embodiment]]
- [[ECoT]]

---
type: concept
aliases: [EgoVerse, EgoVerse-A, 野外第一视角数据集]
---

# EgoVerse

## 定义
EgoVerse 是一个大规模野外第一视角人类视频数据集，由配备 Project Aria 眼镜的人类演示者在多样化真实场景中采集，包含 RGB 视频和精确 VIO 6-DoF 相机位姿，主要用于跨体态迁移学习研究。

## 核心要点

1. **采集设备**: Meta Project Aria 眼镜，同步采集 RGB 视频 + VIO 位姿（相机 6-DoF）
2. **数据规模**: EgoVerse-A flagship split 每操作任务约 10:1 的人机数据比例
3. **多样性**: 涵盖多样场景、多样物体、多样演示者，无刻意场景/物体对齐
4. **位姿信息**: VIO 提供的精确相机位姿支持 3D 运动流计算（自我运动解耦）
5. **访问**: https://partners.mecka.ai/egoverse

## 用途

- **WAM co-training**: 提供无动作标注（仅世界变化）的大规模预训练信号
- **跨体态研究**: 人类手部操作场景与机器人操作的物理效果高度相似

## 与机器人数据对比

| 属性 | EgoVerse | 机器人演示 |
|------|----------|-----------|
| 采集难度 | 低（日常佩戴） | 高（遥操作） |
| 规模 | 大（多样） | 小（300-360/任务） |
| 动作标签 | 无（或不可执行） | 有（14-D EE） |
| 视角 | 移动头部相机 | 固定机器人相机 |

## 代表工作

- [[EgoWAM]]: 首次系统评估 EgoVerse 在 WAM co-training 中的效果

## 相关概念

- [[World Action Model|WAM]]: EgoVerse 是 WAM co-training 的主要人类数据来源
- [[Cross-Embodiment]]: EgoVerse 支持跨体态迁移研究

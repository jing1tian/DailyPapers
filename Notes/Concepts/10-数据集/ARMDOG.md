---
type: concept
aliases: [ARMDOG Dataset, ARM-DOG]
---

# ARMDOG

## 定义

ARMDOG（Arm-Robot Mobile Manipulation Dataset on Quadruped）是首个同步多模态四足移动操作数据集，在真实四足机器人平台上采集，同步记录 RGB-D、本体感知、IMU、全身状态指令和语言标注。

## 核心要点

1. **规模**: 质量过滤后 1,487 段 / 343,550 帧 / 321.3 分钟，帧率 15 Hz
2. **任务分布**: Bottle Pick&Place (56%)、Place Block (39%)、Object Pick&Place (4%)、Climb Slope (1%)
3. **模型接口**: 输入为文本 + 当前 RGB + 机器人状态，预测目标为未来 RGB 和 14-D 全身动作块
4. **全身状态**: 包含底座速度 $(v_x, v_y, \omega_z)$、关节角度、夹爪状态等完整本体信息
5. **发布计划**: 含原始/清洗 HDF5 文件、世界-动作数据转换代码、不可变分割清单和数据说明书

## 代表工作

- [[DECOWAM]]: 基于 ARMDOG 训练和评估的提出方，Stage-1 和 Stage-2 均在此数据集上微调

## 相关概念

- [[FastWAM]]
- [[Action Chunking]]

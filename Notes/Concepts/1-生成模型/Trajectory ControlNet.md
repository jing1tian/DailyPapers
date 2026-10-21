---
type: concept
aliases: [轨迹 ControlNet, Traj-ControlNet]
---

# Trajectory ControlNet

## 定义

Trajectory ControlNet 是对标准 [[ControlNet]] 的扩展，以物体轨迹（如遮罩序列、点迹、边界框序列）为条件信号注入预训练视频生成模型，在不修改骨干权重的前提下实现轨迹可控的视频生成。

## 数学形式

$$
h_i' = h_i + \Delta h_i\bigl(\operatorname{enc}(M^+)\bigr)
$$

其中 $h_i$ 为冻结视频骨干第 $i$ 层输出，$\Delta h_i$ 为 ControlNet 产生的残差，$\operatorname{enc}(M^+)$ 为轨迹遮罩的潜码表示。

## 核心要点

1. **轻量化旁路**: 保持原视频骨干深度，将隐藏维度从骨干的 3,072 压缩至 512，大幅减少参数量
2. **零初始化投影**: 输出层零初始化，确保训练初始时不破坏预训练视频模型
3. **残差叠加**: 逐 Transformer block 产生残差并叠加，使条件信号渐进地影响视频生成
4. **冻结骨干**: 视频骨干参数完全冻结，仅训练 ControlNet 和投影层

## 代表工作

- [[TrajWAM]]: 将物体遮罩序列轨迹通过轻量 ControlNet 注入 5B 视频模型，用于动态机械臂操作的动作预测

## 相关概念

- [[ControlNet]]: 原始设计，用于图像扩散模型的条件控制
- [[Flow Matching]]: TrajWAM 中 Trajectory ControlNet 训练所用的目标函数
- [[World Action Model]]: 应用 Trajectory ControlNet 的系统框架

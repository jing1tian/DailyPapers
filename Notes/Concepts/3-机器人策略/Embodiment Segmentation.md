---
type: concept
aliases: [本体分割, 机器人本体分割, Embodiment Mask]
---

# Embodiment Segmentation（本体分割）

## 定义

在具身机器人视频中，**本体分割**是指对机器人手臂、手爪或人类手部等执行本体（embodiment）进行二值掩码分割，将本体像素与背景/物体像素区分开来。

## 核心要点

1. **任务定义**: 以视频帧为输入，输出与之对齐的二值分割掩码，标注当前帧中机器人或人手所在的像素区域。
2. **与语义分割的区别**: 普通语义分割关注场景中所有类别，而本体分割专注于"谁在执行操作"——执行主体本体的像素范围。
3. **在世界模型中的角色**: 作为密集感知输出之一，本体分割为下游控制提供本体定位信息，并与深度、光流联合用于 3D 场景理解和操作规划。
4. **表示方式**: 通常以 RGB 形式编码（三通道全 0 或全 255），解码后以固定阈值（如 127）二值化为最终掩码。
5. **跨本体泛化**: 优秀的本体分割模型应对不同形态的本体（Franka、UR5、人手等）均能有效分割，而无需为每种机器人单独训练。

## 数学形式

对于帧 $I \in \mathbb{R}^{H \times W \times 3}$，本体分割预测二值掩码：

$$S(p) = \begin{cases} 1 & \text{if pixel } p \text{ belongs to embodiment} \\ 0 & \text{otherwise} \end{cases}$$

编码为 RGB 温度计表示时：

$$\text{RGB}_{enc}(S) = \begin{cases} (255, 255, 255) & S(p) = 1 \\ (0, 0, 0) & S(p) = 0 \end{cases}$$

## 代表工作

- [[GenSight]]: 将本体分割作为四模态统一具身世界模型（GenSight）的密集感知输出之一，与 RGB、深度、光流共享同一 Video DiT 骨干。
- [[SAM3]]: 通用视频分割基线，GenSight 在本体分割 IoU 上（88.7 vs. 73.5）大幅超越 SAM3。

## 相关概念

- [[Semantic Segmentation]]
- [[Dense Prediction]]
- [[Video DiT]]
- [[World Model]]
- [[Optical Flow]]
- [[深度估计]]

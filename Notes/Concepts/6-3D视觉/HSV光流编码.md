---
type: concept
aliases: [HSV Flow Encoding, 光流 HSV 编码, HSV flow visualization]
---

# HSV 光流编码

## 定义

将二维光流场 $f_t \in \mathbb{R}^{H \times W \times 2}$ 可逆地映射为 RGB 图像的编码方案：用色调（H）表示运动方向，饱和度（S）表示位移幅度，明度（V）固定为 1。

## 数学形式

$$
H = \frac{\text{atan2}(v, u) + \pi}{2\pi}, \quad S = \frac{\|(u,v)\|}{m}, \quad V = 1
$$

其中 $(u, v)$ 为每像素横向和纵向位移分量，$m$ 为归一化常数。

## 核心要点

1. **格式统一性**: 编码后光流与 RGB 帧格式完全相同，可直接输入标准 VAE 和视频 DiT
2. **可逆性**: 色调和饱和度一一对应方向和幅度，可精确从 RGB 恢复光流向量
3. **视觉直观**: 彩色轮盘编码使运动方向可视化，便于调试和分析

## 代表工作

- [[FlowWAM]]: 将 HSV 光流编码作为机器人动作的视频格式化表示，在 RoboTwin 上达到 92.94% 成功率

## 相关概念

- [[光流 (Optical Flow)]]
- [[Video DiT]]
- [[Causal VAE]]

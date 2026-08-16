---
type: concept
aliases: [nuScenes Dataset, nuScenes数据集]
---

# nuScenes

## 定义

由 Motional（前身 nuTonomy）发布的大规模自动驾驶感知与预测数据集，包含城市真实驾驶场景的多传感器数据，是自动驾驶轨迹规划研究的标准 benchmark。

## 数学形式

$$
\text{评估指标：Avg. L2} = \frac{1}{T}\sum_{t=1}^{T}\|\hat{p}_t - p_t^*\|_2, \quad T \in \{1,2,3\}\text{ s}
$$

## 核心要点

1. **传感器配置**: 6 个摄像头（360° 环视）+ 1 个 LiDAR + 5 个 radar + IMU/GPS
2. **规模**: 1000 个场景（700 训练 / 150 验证 / 150 测试），每场景 20 秒，2Hz 关键帧标注
3. **标注内容**: 3D 目标检测框、分类、跟踪 ID；车道线、地图语义分割；自车轨迹真值
4. **规划任务**: 预测未来 1–3 s（或 1–6 步）的自车 BEV 航点，用 L2 误差和碰撞率评估

## 代表工作

- [[FIRE-VLA]]: 在 nuScenes 上做 RL 后训练和轨迹规划评估（6019 验证样本，150 场景）

## 相关概念

- [[VLA]]
- [[GRPO]]

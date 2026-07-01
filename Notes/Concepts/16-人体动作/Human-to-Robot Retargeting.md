---
type: concept
aliases: [Human-to-Robot Retargeting, H2R, 人手到机器人重定向, Human-to-Robot Pipeline]
---

# Human-to-Robot Retargeting (H2R)

## 定义

Human-to-Robot Retargeting（H2R）是一种将人类自我中心视频（egocentric video）中的手部动作自动转化为机器人操控演示数据的合成管线，通过动作重定向、视觉修复和仿真渲染，将廉价易得的人类视频扩展为昂贵的机器人演示数据。

## 核心要点

1. **动作重定向**: 提取人手关键点（拇指、食指、中指），计算虚拟末端执行器位置 $p = \frac{1}{2}(k_{\text{thumb}} + k_{\text{vf}})$ 和夹爪宽度 $w = \|k_{\text{thumb}} - k_{\text{vf}}\|_2$，映射到并联夹爪动作
2. **手部去除与背景修复**: 用 SAM3 分割手部区域，用 ProPainter 进行视频修复，还原无手部的场景背景
3. **仿真渲染**: 在 MuJoCo 等仿真器中渲染 15 种双臂机器人平台执行重定向后的动作
4. **深度引导合成（Depth-Guided Compositing）**: 利用深度信息将渲染的机器人按正确的遮挡关系融合回真实场景背景，提升视觉真实性

## 数学形式

人手到夹爪的关键点映射：

$$
k_{\text{vf}} = 0.7\, k_{\text{index}} + 0.3\, k_{\text{middle}}
$$

$$
p = \frac{1}{2}(k_{\text{thumb}} + k_{\text{vf}}), \quad w = \|k_{\text{thumb}} - k_{\text{vf}}\|_2
$$

## 数据放大效果

- **输入**: 1,933 小时自我中心人类视频
- **输出**: 24,808 小时跨 15 种机器人平台的演示数据（约 12.8× 放大）
- 不需要任何专有的机器人数据采集

## 代表工作

- [[Qwen-RobotManip]] (2026): 将 H2R 管线与视觉对齐（手部修复）和运动对齐（相机坐标系）结合，实现大规模跨具身数据合成

## 相关概念

- [[具身重定向]]: 更广义的动作重定向概念
- [[模仿学习]]: H2R 生成的数据用于行为克隆训练
- [[Cross-Embodiment]]: H2R 生成跨具身训练数据的核心工具

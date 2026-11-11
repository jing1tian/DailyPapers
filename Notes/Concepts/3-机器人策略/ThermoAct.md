---
type: concept
aliases: [Thermo Act, 热感知动作模型]
---

# ThermoAct

## 定义

Thermo-VLA 论文中用作对比基线的热感知 VLA 方法，以不同于 [[ThermoRes Encoder]] 的方式将热图像输入 VLA 模型；具体实现细节未在论文正文中详细披露。

## 核心要点

1. **基线角色**：在 Thermo-VLA 的六任务真实世界 Benchmark 中作为主要竞品
2. **性能对比**：在热任务（Task IV avg）上达到 0.85，与 Thermo-VLA（0.86）相近；但在 Task VI（热异常排查）上仅 0.55，远低于 Thermo-VLA（1.00）
3. **泛化差距**：在泛化测试中表现明显弱于 Thermo-VLA（平均 0.38 vs 0.79，未见对象场景）

## 代表工作

- [[Thermo-VLA]]: 以 ThermoAct 作为热感知方法的主要比较基线

## 相关概念

- [[ThermoRes Encoder]]: Thermo-VLA 提出的替代方案，在泛化性上显著优于 ThermoAct
- [[VLA（视觉-语言-动作模型）]]: 两者均属于 VLA 范畴
- [[Thermal Camera]]: 两者均使用热成像传感器输入

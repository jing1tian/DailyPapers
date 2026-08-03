---
type: concept
aliases: [CogVLA Model]
---

# CogVLA

## 定义
一种 Vision-Language-Action (VLA) 模型，作为 TurboVLA 等高效 VLA 工作中的对比基线之一出现在实验评测中。

## 数学形式
$$\pi_\theta(a | o, l) = \text{VLA}(v_\text{enc}(o), l_\text{enc}(l))$$

## 核心要点
1. VLA 模型家族成员之一
2. 在 TurboVLA 论文中作为对比方法出现
3. 具体架构细节需参考原始论文

## 代表工作
- [[TurboVLA]]: 与 CogVLA 进行推理效率对比

## 相关概念
- [[TurboVLA]]
- [[SmolVLA]]
- [[OpenVLA]]
- [[StarVLA]]

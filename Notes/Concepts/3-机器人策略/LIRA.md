---
type: concept
aliases: [Local Cross-Layer Information Routing]
---

# LIRA

## 定义
一种 VLA 动作解码接口设计，通过滑动窗口式的跨层路由机制（LIRA Query tokens）聚合 VLM 不同深度的特征，为动作解码器提供更丰富的任务语义。

## 核心要点
1. **Local Cross-Layer Routing**：LIRA Query tokens 在以对应 VLM 层为中心的局部窗口内聚合跨层特征
2. **Depth-Aligned Interface**：每个动作解码器块对应一个 VLM 层附近的跨层聚合，而非死板地 1-to-1 对应
3. 在 CALVIN ABC→D 和 LIBERO-Long 上有提升，Franka 真实机械臂验证
4. 系统开销增量小（见 resource profile 表格）

## 核心区别
相比 [[OpenVLA]] 等只用 VLM 最后一层特征的方式，LIRA 利用了中间层的互补任务证据；相比固定 1-to-1 对应的方式，LIRA 的局部窗口更灵活。

## 代表工作
- LIRA: Local Cross-Layer Information Routing for Vision-Language-Action Decoding (2026)

## 相关概念
- [[OpenVLA]]
- [[UniVLA]]
- [[CALVIN]]

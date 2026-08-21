---
type: concept
aliases: [ORCA Hand, ORCA 灵巧手, ORCA 17-DoF]
---

# ORCA Hand

## 定义
高自由度**电驱灵巧手**平台，具有 17 个受控自由度，覆盖 5 指完整运动范围，广泛用于灵巧操作研究与世界模型评估。

## 核心要点
1. **17 个关节自由度**: 覆盖全手指（拇指 + 四指）的屈伸、外展/内收运动
2. **遥操作支持**: 通常与 Rokoko 数据手套配合使用，实现人手映射遥操作
3. **与 Franka 集成**: 常安装在 [[Franka Emika Panda]] 机械臂末端，构成 24-DoF 操作平台
4. **高维动作空间**: 23 维动作（6 维笛卡尔 EE + 17 维手关节）对世界模型构成挑战

## 代表工作
- [[Mask2Real-WM]]: 以 ORCA Hand 为平台验证 23 维动作条件世界模型，动作可控性 ID 0.95 / OOD 0.87

## 相关概念
- [[Franka Emika Panda]]
- [[Dexterous Manipulation]]

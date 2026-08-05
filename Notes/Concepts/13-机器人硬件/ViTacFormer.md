---
type: concept
aliases: [Visuo-Tactile Transformer]
---

# ViTacFormer

## 定义
ReTouch 中用于在线精修触觉预测的 Transformer 模块：接收离线预测的触觉状态和执行时真实触觉反馈，输出精修后的触觉表征，用于指导 VLA 动作生成。

## 核心要点
1. 两阶段解耦：离线大规模训练触觉预测模块（RGB → 触觉状态）；在线用少量真实触觉反馈精修
2. ViTacFormer 只在执行时运行，不需要大量真实触觉-视觉配对数据
3. 输入：预测触觉状态 + 真实传感器读数；输出：精修后的视觉-触觉联合表征
4. 支持的触觉传感器：TacForeSight、OmniVTA（论文验证）

## 代表工作
- [[ReTouch]]: 提出 ViTacFormer，真实灵巧手实验

## 相关概念
- [[ACT]]
- [[VLA]]

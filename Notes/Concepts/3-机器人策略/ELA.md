---
type: concept
aliases: [End-to-End Latent Action, End-to-End Latent Action Policy]
---

# ELA

## 定义
End-to-End Latent Action Policy Learning：从视频示例（无 action label）端到端学习 latent action representation 和 policy，latent action 直接对 downstream control 任务优化，解决两阶段方法目标不对齐问题。

## 核心要点
1. 现有方法（[[LAPA]], [[LAM]]）两阶段：先学 latent action，再训 policy，两个 objective 不对齐
2. ELA 合并为一个端到端 objective，latent action 对 control 任务直接优化
3. 通过 [[IDM]] 结构生成 latent action，ElaDP 和 ElaUWM 是两个下游变体
4. 对比 [[LAPA]], [[UWM]], [[UniVLA]]；在 [[LIBERO-10]] 验证

## 相关概念
- [[LAPA]]
- [[LAM]]
- [[IDM]]
- [[Latent Action]]

---
type: concept
aliases: [VT-WM, Visuo-tactile world models]
---

# Visuo-Tactile World Models

## 定义
Higuera 等（2026，arXiv 2602.06001，项目主页 https://carolinahiguera.github.io/vtml/）提出的视触觉世界模型，将触觉模态纳入未来状态预测，证明触觉是接触密集型操作任务中有价值的预测信号。

## 核心要点
1. 与 [[Dream-Tac]]、[[OmniVTA]]、[[VTAM]]、[[DreamTacVLA]]、[[TacForeSight]] 同属"现有触觉 WAM/视触觉世界模型"一类工作，共同证明了触觉对未来状态预测的价值
2. 在 [[Tactile-WAM]] 的 Related Work 定位中，被归为"已证明触觉有用，但未解决注意力层面路由问题"的代表
3. [[Tactile-WAM]] 指出此类工作未区分"触觉应自由参与所有注意力路径"还是"仅在接触相关时路由到动作生成路径"这一架构问题

## 代表工作
- 原始论文: Higuera, Arnaud, Boots, Mukadam, Hogan, Meier (2026) "Visuo-tactile world models" (arXiv 2602.06001)
- [[Tactile-WAM]]: 在 Related Work 中与之对比，强调"触觉作为生成式物理状态、经非对称动作路径路由"的架构差异

## 相关概念
- [[World Action Model]]
- [[Tactile-WAM]]
- [[Dream-Tac]]

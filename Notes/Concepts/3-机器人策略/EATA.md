---
type: concept
aliases: [Execution-Aware Tactile Adaptation]
---

# EATA (Execution-Aware Tactile Adaptation)

## 定义
在 VLA 动作执行期间将实时触觉信号注入 action decoder，实现执行时自适应修正的轻量模块。

## 核心要点
1. 在 action generation 的 streaming 阶段注入触觉条件
2. 不修改 base policy 结构，以适配器形式插入
3. 只在接触发生时激活，减少非接触阶段的干扰

## 代表工作
- [[TacForcing]]: EATA 是其核心技术组件

## 相关概念
- [[UniVTAC]]
- [[TacForcing]]
- [[RDP]]

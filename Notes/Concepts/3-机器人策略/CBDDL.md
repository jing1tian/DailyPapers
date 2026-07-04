---
type: concept
aliases: [Constrained BDDL, Constrained Behavior Domain Definition Language, 约束行为域定义语言]
---

# CBDDL

## 定义

约束行为域定义语言（Constrained Behavior Domain Definition Language），由 VLA-Arena 提出，在标准 [[BDDL]] 基础上扩展了**动态实体**、**扰动支持**和**形式化安全约束**，专门用于量化机器人操作任务中的安全违规行为。

## 核心扩展

相比 BDDL，CBDDL 新增三个关键能力：

1. **动态实体支持**: 支持随时间变化的物体（如移动障碍物），而非仅静态场景配置
2. **扰动内置**: 原生支持语言指令扰动（W0-W4）和视觉观测扰动（V0-V4）的声明式规范
3. **形式化安全谓词**: 定义 10 类安全约束谓词，分为瞬态（instantaneous）和终态（terminal）两类

## 安全谓词分类

| 类别 | 示例谓词 | 检测时机 |
|------|----------|----------|
| 瞬态（instantaneous） | 碰撞（collision）、力超限（force limit） | 每步检测 |
| 终态（terminal） | 物体掉落（object drop）、状态不保持 | 轨迹终止时检测 |

## 相关度量

CBDDL 直接支持 [[Cumulative Cost]]（CC）指标的计算，通过安全谓词的累积违规量化安全性能。

## 代表工作

- [[VLA-Arena]] (2512.22539): CBDDL 的提出论文，用于构建 170 个安全约束任务

## 相关概念

- [[BDDL]]: 前身语言
- [[Cumulative Cost]]: 基于 CBDDL 谓词的安全度量
- [[MuJoCo]]: 执行 CBDDL 任务的物理引擎

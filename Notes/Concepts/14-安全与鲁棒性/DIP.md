---
type: concept
aliases: [Defensive Input Processing]
---

# DIP

## 定义
防御性输入处理（Defensive Input Processing）方法，通过在输入预处理阶段过滤对抗样本来提升模型鲁棒性。

## 核心要点
1. 对输入图像进行去噪/平滑处理以破坏对抗扰动
2. 不修改模型本身，只处理输入
3. 可用于 VLA 的对抗防御

## 代表工作
- TRAP 论文中作为防御 baseline

## 相关概念
- [[PGD]]
- [[TRAP]]
- [[POPA-VLA]]

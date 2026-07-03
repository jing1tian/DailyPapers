---
type: concept
aliases: [DreamBooth]
---

# DreamBooth

## 定义

DreamBooth 是 Google 提出的个性化文本-图像扩散模型微调方法，通过少量（3-5 张）参考图片将特定概念/人物/风格绑定到唯一标识符，实现个性化图像生成。

## 核心要点

1. 用唯一标识符（如 "sks dog"）绑定特定概念
2. 引入 Prior Preservation Loss 防止过拟合和语言漂移
3. 全量微调（非 LoRA），计算成本较高
4. 是扩散模型个性化领域的奠基工作

## 代表工作

- [[DriftScope]]: 用 SAE 分析 DreamBooth 微调产生的 concept drift 副作用

## 相关概念

- [[SD]]
- [[ESD]]
- [[SPM]]

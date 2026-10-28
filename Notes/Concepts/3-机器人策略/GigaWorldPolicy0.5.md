---
type: concept
aliases: [GigaWorld Policy 0.5]
---

# GigaWorldPolicy0.5

## 定义
GigaWorldPolicy 0.5：像素级 World Action Model 的代表工作，在大规模数据上预训练 video prediction + action generation，用 future frame 作为中间监督信号。

## 核心要点
1. 联合预测未来帧（pixel space）和 robot action
2. 大规模数据（Giga 级别）训练 WAM 的早期代表
3. 与 [[LaWAM]] 形成对比——LaWAM 将预测移到 latent space 以降低计算代价

## 代表工作
- [[LaWAM]]: 通过 latent prediction 避免 pixel-level 生成的高计算成本
- [[WAM-Survey]]: 综述 WAM 发展，包含 GigaWorldPolicy 系列

## 相关概念
- [[WAM-Survey]]
- [[LaWAM]]

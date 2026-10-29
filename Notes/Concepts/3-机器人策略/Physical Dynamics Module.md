---
type: concept
aliases: [物理动态模块, PDM, Physical Dynamics Representations]
---

# Physical Dynamics Module

## 定义

一种轻量级模块，通过对 VLM 中间层特征施加可学习查询，提取"物理 token"，并以预测未来 RGB / 深度 / 光流特征为目标进行训练，从而将物理动态感知注入 VLA 的动作生成过程，无需在线生成视频帧。

## 数学形式

$$
X_t = [H_t^{\text{last}};\, C_t]
$$

其中 $C_t$ 为物理 token（由 50 个可学习查询对 VLM 第 8 层特征做每层交叉注意力得到），$H_t^{\text{last}}$ 为最终层 VLM 特征，拼接后作为动作专家的条件输入。

## 核心要点

1. 从 VLM 浅层（第 8 层）提取特征，包含比深层更丰富的物理信息。
2. 用 50 个可学习查询（类似 Perceiver）做多层交叉注意力，输出紧凑的物理 token（约 57.3M 参数）。
3. 预训练时用冻结视频世界模型（LingBot-Video）的流匹配速度预测提供蒸馏式监督。
4. 推理时无需生成视频帧，消除在线世界模型的控制循环延迟。

## 代表工作

- [[PDRVLA]]: 首次提出该模块

## 相关概念

- [[Video World Model|视频世界模型]]
- [[Flow-Matching]]
- [[ECoT]]
- [[光流 (Optical Flow)|光流]]

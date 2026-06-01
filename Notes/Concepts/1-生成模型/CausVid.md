---
type: concept
aliases: [Causal Video, CausalVid]
---

# CausVid

## 定义
Meta 提出的因果视频生成方法，将双向扩散模型蒸馏为单向（因果）自回归生成模型，实现低延迟的流式视频生成，无需等待完整序列即可开始输出。

## 数学形式

因果蒸馏目标：将 teacher（双向）的分布转移到 student（因果）：
$$\mathcal{L}_{\text{distill}} = \mathbb{E}\left[D_{\text{KL}}\left(p_{\text{teacher}}(x_t | x_{<t}) \| p_{\text{student}}(x_t | x_{<t})\right)\right]$$

## 核心要点
1. 核心贡献是把双向扩散（需要看全序列）蒸馏成因果模型（只看历史帧）
2. 实现流式推理：生成第 t 帧时不需要等待后续帧
3. LongLive-2.0 和 Self-Forcing 均引用 CausVid 作为基础方法或对比
4. 延迟大幅降低，但生成质量略低于双向模型

## 代表工作
- [[LongLive]]: 在 CausVid 基础上扩展 NVFP4 量化和长视频支持

## 相关概念
- [[DMD]]
- [[Self-Forcing]]
- [[DiT]]

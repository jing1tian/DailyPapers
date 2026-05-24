---
type: concept
aliases: [Chain of Continuous Thought, 连续思维链]
---

# COCONUT

## 定义
Chain of Continuous Thought：将 Chain-of-Thought 推理从离散 token 序列转移到连续潜空间中执行，允许模型在潜空间内完成推理而无需生成中间文本 token。

## 数学形式
$$h_{t+1} = f_\theta(h_t, x)$$
推理步在隐状态 $h_t$ 上迭代进行，不经过 token embedding/de-embedding 往返，直接在连续空间传递信息。

## 核心要点
1. 标准 CoT 每步推理需要解码到词表再重编码，COCONUT 在连续隐状态空间直接迭代
2. 显著降低推理 token 数量，缩短 latency，适合实时性要求高的场景
3. 潜空间推理的"草稿"不可读（无语言解释性），但计算效率高

## 代表工作
- [[OneVL]]: 将 COCONUT 思路用于自动驾驶 VLA 的单步推理加速

## 相关概念
- [[VLA]]
- [[Vision Transformer]]

---
type: concept
aliases: [InternVL3, InternVL]
---

# InternVL3

## 定义
上海 AI Lab 开源的多模态大语言模型系列，第三代版本。采用 InternViT 视觉编码器 + LLM，在视觉理解和推理任务上达到商业模型水平，并完全开源。

## 数学形式
$$
p(a | v, q) = \prod_t p_\theta(a_t | v, q, a_{<t})
$$
视觉特征通过 dynamic resolution 处理后与文本拼接，自回归生成回答。

## 核心要点
1. InternViT-6B：专为高分辨率图像理解设计的视觉编码器
2. Dynamic Resolution：自适应处理不同分辨率输入
3. 开源最强 VLM 之一，在 MMBench、MMMU 等 benchmark 上表现突出
4. 常用于机器人导航（OpenFrontier）、视觉问答等需要强语义理解的场景

## 代表工作
- Chen et al. (2024): "InternVL: Scaling up Vision Foundation Models"
- [[OpenFrontier]]: 用 InternVL3 做 frontier 语义评估

## 相关概念
- [[InternVL]]
- [[VLM]]
- [[CLIP]]

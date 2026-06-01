---
type: concept
aliases: [Janus Pro, Janus-Pro]
---

# Janus

## 定义
DeepSeek 提出的统一多模态理解+生成模型，通过分离理解和生成的视觉编码路径（Decoupled Visual Encoding）来解决两者之间的表示冲突，在统一框架下同时支持图像理解和文本引导图像生成。

## 数学形式

解耦编码：理解用 CLIP-like encoder $f_u$，生成用 tokenizer $f_g$：
$$p(y|x) = p_{\text{understand}}(y|f_u(x)) \cdot p_{\text{generate}}(y|f_g(x))$$

## 核心要点
1. 核心洞察：理解需要高级语义特征，生成需要低级纹理特征，应解耦而非共享编码器
2. 基于自回归 LLM 架构，图像生成通过 visual token 预测实现
3. Janus Pro 提升了生成质量和理解能力的平衡
4. SenseNova-U1、BAGEL 等后续工作均以 Janus 作为对比

## 代表工作
- [[SenseNova-U1]]: 与 Janus 在统一生成任务上对比

## 相关概念
- [[BAGEL]]
- [[OmniGen2]]
- [[CLIP]]

---
type: concept
aliases: [跨模态对齐, Cross-Modality Alignment, 语言-视频-动作对齐]
---

# Cross-Modal Alignment（跨模态对齐）

## 定义

在多模态模型（尤其是 [[World Action Model]] 和 [[VLA]]）中，确保语言、视觉和动作三种模态之间的语义表示保持一致、相互对应的训练目标和方法论体系。

## 核心要点

1. **语言-视频对齐**: 保留视频预训练阶段建立的语言语义与视觉内容的对应关系，防止动作学习破坏该关联
2. **语言-视频-动作三级对齐**: 在 WAM 中进一步将动作纳入对齐链，使语言指令能通过视频特征传导到动作预测
3. **标注质量的决定作用**: 语言标注的详细程度（Brief/Medium/Detailed）和时间粒度（Episode/Subtask）显著影响跨场景泛化
4. **信息流向控制**: 在双流架构中，语言仅单向（Language → Video → Action）传递，避免动作梯度干扰视频语义

## 数学形式

在 SWAM 框架下，对齐通过联合分布的分解实现：

$$
p_\theta(V, A \mid V_{0:n}, P, S_n) = p_\theta(V \mid V_{0:n}, P)\, p_\theta(A \mid V_{0:n+K}, P, S_n)
$$

其中视频分支先与语言对齐，动作分支再以视频特征为桥梁间接获取语言信息。

## 代表工作

- [[SWAM]]: 明确提出三级跨模态对齐（语言-视频-动作），系统研究标注质量对对齐效果的影响
- [[OpenWAM]]: 双流 WAM 基线，但未专门优化语言-视频对齐质量
- [[LingBot-Video]]: 以语言作为视频生成的强监督信号

## 相关概念

- [[World Action Model]]
- [[Mixture-of-Transformers]]
- [[Flow Matching]]
- [[Teacher Forcing]]

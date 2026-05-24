---
type: concept
aliases: [CLIP, Contrastive Language-Image Pretraining]
---

# CLIP (Contrastive Language-Image Pretraining)

## 定义
OpenAI 提出的对比学习预训练框架，将图像和文本映射到统一的语义嵌入空间，通过大规模图文对的对比损失训练。

## 数学形式
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, t_j)/\tau)}
$$
其中 $v_i$、$t_i$ 为图像/文本嵌入，$\tau$ 为温度系数。

## 核心要点
1. 零样本迁移：训练后可直接用于零样本图像分类
2. 双编码器架构：图像编码器（ViT/ResNet）+ 文本编码器（Transformer）
3. 语义对齐：图文共享嵌入空间，余弦相似度度量语义距离
4. 广泛用于语义监督、特征提取和开放词汇识别

## 代表工作
- Radford et al. (2021): "Learning Transferable Visual Models From Natural Language Supervision"

## 相关概念
- [[SigLIP2]]
- [[DINOv2]]
- [[Vision Transformer]]

---
type: concept
aliases: [Tokenized Gaussian Splatting]
---

# TokenGS

## 定义
TokenGS：将 3D Gaussian Splatting 参数离散化为 token 的方法，使 Transformer 可以直接自回归生成 3DGS 表示。

## 核心要点
1. 量化 Gaussian 属性（位置、颜色、透明度、协方差）为离散 token
2. Transformer 自回归预测下一个 Gaussian token
3. 无需 test-time 优化，单次前向推理生成场景

## 代表工作
- [[InfiniSplat]]: 与 TokenGS 对比，提出 implicit Gaussian decoder 替代 tokenization

## 相关概念
- [[InfiniSplat]]
- [[3DGS]]

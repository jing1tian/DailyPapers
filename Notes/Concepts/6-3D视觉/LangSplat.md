---
type: concept
aliases: [Language-embedded 3D Gaussians]
---

# LangSplat

## 定义
将 CLIP 语言特征嵌入到 3D Gaussian Splatting 的每个 Gaussian 中，实现可查询的语义 3D 场景表示，支持开放词汇的 3D 语义分割和自然语言定位。

## 数学形式
$$G_i = \{x_i, \sigma_i, c_i, f_i^\text{lang}\}, \quad f_i^\text{lang} = \text{CLIP-embed}(\text{patch}_i)$$

每个 Gaussian 携带语言特征 $f^\text{lang}$，查询时做 dot-product 相似度检索。

## 核心要点
1. 在标准 3DGS 每个 Gaussian 上附加 CLIP 语言特征
2. 支持"找到红色杯子"等开放词汇空间查询
3. 多尺度 SAM mask 提取训练信号，增强语义精度
4. MIF 中用 LangSplat 构建可语义查询的 3D 场景表示

## 代表工作
- [[LangSplat]]：Qin et al. 2023，CVPR 2024
- [[MIF]]：人形机器人导航中使用 LangSplat 做语义场景图

## 相关概念
- [[3D Gaussian Splatting]]
- [[CLIP]]
- [[ConceptGraphs]]

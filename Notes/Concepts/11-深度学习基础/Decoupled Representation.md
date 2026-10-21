---
type: concept
aliases: [解耦表示, 表示解耦, Disentangled Representation]
---

# Decoupled Representation

## 定义
解耦表示是指将数据中**不同语义因素**（如外观与几何、内容与风格）分离到独立的潜在子空间中，使各子空间只捕获特定属性，减少相互干扰。

## 数学形式

以 PrismWAM 中的外观-物理解耦为例：

$$
z = (z_{\text{semantic}}, z_{\text{physical}})
$$

其中 $z_{\text{semantic}}$ 捕获语义外观（RGB），$z_{\text{physical}}$ 捕获物理配置（深度/占据）。

## 核心要点

1. **单因素变化**: 理想的解耦表示中，改变一个因素不影响其他因素
2. **泛化收益**: 解耦物理与外观后，物理预测不受外观变化（光照、纹理、颜色）干扰
3. **监督方式**: 可通过稀疏/密集混合监督、对比学习或 VAE 约束实现
4. **与 [[Mixture-of-Transformers]] 的关系**: 多流架构天然支持不同模态的独立建模

## 代表工作
- [[PrismWAM]]: 将 WAM 状态解耦为 RGB 语义流和深度物理流，显著提升泛化能力

## 相关概念
- [[Chain Attention]]
- [[World Action Model]]
- [[Depth Prediction]]
- [[Factorized Policy]]

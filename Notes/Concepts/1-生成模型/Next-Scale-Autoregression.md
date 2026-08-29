---
type: concept
aliases: [Next-Scale Autoregression, Scale-wise Autoregression, 下一尺度自回归]
---

# Next-Scale Autoregression

## 定义
从粗到细的多尺度自回归生成范式：先预测低分辨率关键帧确定宏观动态，再逐步生成高分辨率细节，替代 token-by-token 的逐像素/逐 patch 预测。

## 数学形式
$$p(x) = \prod_{s=1}^{S} p(x^{(s)} | x^{(1:s-1)}, c)$$

其中 $x^{(s)}$ 是第 $s$ 个尺度的表示，分辨率从低到高递增。

## 核心要点
1. 先生成低分辨率全局结构，再生成高分辨率局部细节
2. 支持灵活的预测 horizon，适合长程世界模型
3. 比 token-by-token AR 计算效率更高（粗尺度步数少）
4. 在世界模型中：粗尺度捕捉语义动态，细尺度保留纹理细节

## 代表工作
- [[WALL-SS]]: 将 Next-Scale AR 用于长程机器人世界模型

## 相关概念
- [[Action-Conditioned World Model]]
- [[VideoGPT]]
- [[DreamerV3]]

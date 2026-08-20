---
type: concept
aliases: [运动条件化, Motion Feature Propagation, 运动特征传播]
---

# Motion Conditioning

## 定义
将动作流轨迹上的首帧 latent 特征沿时间轴传播，通过 Gaussian 加权聚合构造运动感知视觉条件 $C_{\mathrm{motion}}$，注入视频扩散模型（DiT）。

## 数学形式

特征聚合：
$$
M_{k}(\widetilde{\mathbf{p}}) = \sum_{n\in\mathcal{N}_{K}(\widetilde{\mathbf{p}},k)}\widetilde{w}_{n,k}(\widetilde{\mathbf{p}})\,\mathbf{h}_{n}
$$

Gaussian 加权权重：
$$
\widetilde{w}_{n,k}(\widetilde{\mathbf{p}})=\widetilde{m}_{n,0}\widetilde{m}_{n,k}\exp\!\left(-\beta\lVert\widetilde{\mathbf{p}}-\widetilde{\mathbf{x}}_{n,k}\rVert_{2}^{2}\right)
$$

Presence gate：
$$
g_{k}(\widetilde{\mathbf{p}})=\operatorname{clip}_{[0,1]}\!\left(\sum_{n\in\mathcal{N}_{K}(\widetilde{\mathbf{p}},k)}\widetilde{w}_{n,k}(\widetilde{\mathbf{p}})\right)
$$

## 核心要点
1. 只需首帧 latent 特征，不引入额外编码器参数
2. Gaussian 核保证空间局部性；可见性 mask 过滤遮挡点
3. Presence gate 明确标注哪些 latent 格点有轨迹覆盖
4. $C_{\mathrm{motion}}$ 与噪声 latent $Z_t$ 拼接后送入 DiT patch embedding

## 代表工作
- [[Hydra-0]]: 提出 Motion Conditioning 模块，实现 Action Flow 到视频生成条件的转换

## 相关概念
- [[Action Flow]]: 提供运动轨迹输入
- [[DiT]]: 接收 $C_{\mathrm{motion}}$ 的视频生成主干
- [[Flow Matching]]: 整体训练框架

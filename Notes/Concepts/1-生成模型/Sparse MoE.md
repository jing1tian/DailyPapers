---
type: concept
aliases: [Sparse Mixture-of-Experts, 稀疏混合专家, Token-Choice MoE]
---

# Sparse MoE（稀疏混合专家）

## 定义
每个输入 token 只激活专家池中少数几个专家参数（top-K routing）的 Mixture-of-Experts 变体，实现"参数容量-计算量解耦"：模型总参数量随专家数线性扩展，但每 token 的 FLOPs 固定不变。

## 核心要点
1. **参数容量-计算量解耦**：总参数 ∝ 专家数 × 单专家大小，每 token 激活参数 = $K_r$ 个专家参数；可将模型能力扩展 10× 而 FLOPs 几乎不变
2. **减少子任务干扰**：视频生成中的空间纹理 vs. 时序运动、不同噪声区间、多种任务格式等异构特征可被路由到不同专家，避免单一 FFN 的梯度冲突
3. **对长序列特别有利**：视频扩散重复处理全序列（去噪步骤），稀疏 FFN 在每一步均节省计算，使 1M+ token 的长视频生成实际可行
4. **常见变体**：Switch Transformer（top-1）、GShard（top-2）、DeepSeekMoE（细粒度 + 共享专家 + Sigmoid 路由）

## 数学形式

$$
m(u_t) = \sum_{i=1}^{N_s} E_i^{(s)}(u_t) + \sum_{j \in \mathcal{R}(u_t)} g_{t,j} E_j^{(r)}(u_t)
$$

其中 $N_s$ 为共享专家数，$\mathcal{R}(u_t)$ 为 token $t$ 的路由专家集合（$|\mathcal{R}| = K_r \ll N_r$）。

## 代表工作
- Switch Transformers (Fedus et al., JMLR 2022)
- GShard (Lepikhin et al.)
- [[DeepSeekMoE]] (Dai et al., ACL 2024)
- [[LingBot-Video]]: 首个大规模开源 Sparse MoE 视频扩散模型（13B-120B）

## 相关概念
- [[MoE]]
- [[Mixture-of-Experts]]
- [[DeepSeekMoE]]
- [[DiT]]
- [[Load Balancing]]

---
type: concept
aliases: [Mixture of Experts, MoE, MoT, Mixture of Transformers]
---

# MoE（Mixture of Experts）

## 定义

一种条件计算架构，将模型拆分为多个专用专家子网络，每个输入通过门控路由机制动态激活部分专家，在不线性增加推理计算量的前提下扩大模型容量。

## 数学形式

给定输入 $x$，门控网络 $G(x)$ 选择 Top-K 专家：

$$
\text{MoE}(x) = \sum_{i \in \text{Top-K}} G_i(x) \cdot E_i(x)
$$

其中 $E_i$ 为第 $i$ 个专家网络，$G_i(x)$ 为对应的门控权重（Softmax 归一化）。

在 **MoT（Mixture of Transformers）** 变体中，专家对应不同模态（如视频专家、动作专家），通过跨注意力耦合：

$$
(\mathbf{h}^v_{\ell+1}, \mathbf{h}^a_{\ell+1}) = \mathcal{F}^{\text{mix}}_\ell(\mathbf{h}^v_\ell, \mathbf{h}^a_\ell)
$$

## 核心要点

1. **稀疏激活**: 每次前向传播只激活 K 个专家（通常 K=1 或 2），推理 FLOPs 远小于模型总参数对应的 FLOPs
2. **容量扩展**: 总参数量随专家数线性增长，但单次推理计算量保持不变
3. **负载均衡**: 需要辅助损失（auxiliary loss）防止所有 token 路由到同一专家，造成专家崩溃
4. **模态专用 MoT**: 在机器人策略中，视频专家和动作专家分别处理视觉预测和动作生成，通过跨注意力保持耦合

## 代表工作

- [[WMRobotSurvey]]: 综述 MoE/MoT 式策略范式（Motus、LingBot-VA、DiT4DiT、LDA-1B）
- [[WAMSurvey]]: Joint WAM 的 Motus（三专家 Tri-modal Joint Attention）
- F1: MoT 框架（Generation Expert + Action Expert）
- [[DyGRO-VLA]]: 混合残差专家（MoRR），每个专家学习相对基础策略的残差修正，动态路由解决多任务 VLA 梯度冲突

## 相关概念

- [[Diffusion Model]]: MoT 中视频专家通常基于视频扩散模型
- [[Video Policy]]: MoT 式策略的视频骨干
- [[Cross-Attention]]: 跨专家信息交换的主要机制

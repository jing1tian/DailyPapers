---
type: concept
aliases: [DeepSeek MoE, DeepSeek-MoE, 细粒度专家专业化]
---

# DeepSeekMoE

## 定义
DeepSeek 提出的 Mixture-of-Experts 设计范式，通过"细粒度专家分割 + 共享专家隔离 + 辅助损失无关负载均衡"三项创新，将专家专业化程度最大化，成为后续大模型和视频扩散 MoE 设计的重要参考。

## 核心要点
1. **细粒度专家分割**：将 FFN 参数分解为更多、更小的专家（相比传统 MoE），大幅扩展路由组合空间 $\binom{N_r}{K_r}$，使每个 token 可形成更定制化的执行路径
2. **共享专家隔离**：设置若干"常驻"共享专家（所有 token 必经），捕获通用知识，与路由专家分工——路由专家专注差异化特征
3. **Sigmoid 路由**：使用 Sigmoid 而非 Softmax，专家间互不竞争，独立评分，避免路由决策相互干扰
4. **组限制路由（Group-Limited Routing）**：将路由专家分组，先选 top-$K_g$ 组（按组内 top-2 得分之和），再在选中组内选 top-$K_r$ 专家，控制分布式通信开销
5. **无辅助损失负载均衡**：使用在线偏置校正（Online Bias Correction）动态调整专家负载，无需辅助损失函数

## 数学形式（路由）

$$
\alpha_{t,j} = \text{Sigmoid}(u_t^\top r_j)
$$

$$
\tilde{\alpha}_{t,j} = \alpha_{t,j} + b_j, \quad \text{选 top-}K_r \text{ from top-}K_g\text{ groups}
$$

$$
g_{t,j} = \gamma \cdot \frac{g'_{t,j}}{\sum_k g'_{t,k}}
$$

## 代表工作
- DeepSeekMoE (Dai et al., ACL 2024)
- DeepSeek-V3 (arXiv 2412.19437): 序列级辅助平衡损失
- [[LingBot-Video]]: 视频扩散 MoE 中借鉴 DeepSeekMoE 设计原则

## 相关概念
- [[MoE]]
- [[Mixture-of-Experts]]
- [[Sparse MoE]]
- [[Load Balancing]]

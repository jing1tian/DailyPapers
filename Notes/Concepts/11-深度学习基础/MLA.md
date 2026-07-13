---
type: concept
aliases: [Multi-head Latent Attention, 多头潜变量注意力]
---

# MLA (Multi-head Latent Attention)

## 定义
DeepSeek 提出的注意力变体：将 KV cache 压缩到低维潜变量，推理时从潜变量上投影还原，显著降低 KV cache 内存占用。

## 数学形式
$$c_t^{KV} = W^{DKV} h_t \quad (\text{压缩})$$
$$k_t, v_t = W^{UK} c_t^{KV}, \; W^{UV} c_t^{KV} \quad (\text{还原})$$

其中 $c_t^{KV}$ 是低维 KV 潜变量，维度远小于标准 KV。

## 核心要点
1. KV cache 从 $O(n \cdot d_{model})$ 压缩到 $O(n \cdot d_{latent})$，$d_{latent} \ll d_{model}$
2. 推理时按需解压，不影响注意力计算语义
3. 在 DeepSeek-V2/V3 中实现了 5.75× KV cache 压缩

## 代表工作
- [[DeepSeek-V2]]: 首次提出 MLA
- [[GLM-5]]: 采用 MLA 降低推理成本

## 相关概念
- [[Multi-head Attention]]
- [[KV Cache]]
- [[GQA]]

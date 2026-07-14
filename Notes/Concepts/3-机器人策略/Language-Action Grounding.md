---
type: concept
aliases: [语言动作对齐, 语言动作接地, Language-Action Representation]
---

# Language-Action Grounding

## 定义

在 VLA（视觉-语言-动作模型）框架中，将自然语言动作描述与数值动作 token 联合生成的表示方式，使数值动作预测因果地条件化于语言计划，从而缓解 [[Output-Distribution Mismatch|输出分布不匹配]]。

## 数学形式

$$
\pi_\theta(d, \boldsymbol{a} | o, l) = \left[\prod_{j=1}^{|d|} \pi_\theta(d^{(j)} | d^{(<j)}, o, l)\right] \cdot \left[\prod_{i=1}^{7h} \pi_\theta(a^{(i)} | a^{(<i)}, d, o, l)\right]
$$

其中 $d$ 为语言动作描述（由固定模板从 ground-truth 动作生成），$\boldsymbol{a}$ 为数值动作 token 序列。

## 核心要点

1. 模型先生成语言计划（如 "move forward 3 cm, close gripper"），再生成离散数值 token
2. 语言描述由固定模板 $T(\boldsymbol{a})$ 确定性生成，无需人工标注
3. 无需架构修改，直接在单一自回归序列中完成，与 VLM 预训练分布兼容
4. 相比 ECoT/TraceVLA 等方法，无需额外监督信号

## 代表工作

- [[CLAP]]: 提出 Language-Action Grounding 的核心论文，将 VLM 直接适配为 VLA

## 相关概念

- [[Output-Distribution Mismatch]]
- [[Action Chunking]]
- [[Autoregressive Policy]]
- [[VLA（视觉-语言-动作模型）]]

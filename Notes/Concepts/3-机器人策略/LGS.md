---
type: concept
aliases: [Linguistic Grounding Score, 语言基础分数]
---

# LGS

## 定义
衡量 VLA 模型在推理时注意力分配中语言 token 相对于视觉 token 权重占比的诊断指标，用于检测 OOD 指令下的语言基础（linguistic grounding）失效程度。

## 数学形式
$$
\text{LGS} = \frac{\sum_{i \in \text{lang}} \bar{A}_{i}}{\sum_{i \in \text{lang}} \bar{A}_{i} + \sum_{j \in \text{vis}} \bar{A}_{j}}
$$

其中 $\bar{A}_i$ 为 cross-attention 层中第 $i$ 个 token 的平均注意力权重。

## 核心要点
1. LGS 越低，说明模型在该 step 越依赖视觉 token 而忽视指令语言 token
2. In-distribution 指令下 LGS 保持稳定；OOD 指令下 LGS 显著下降，是语言基础失效的早期信号
3. 可用于无标注地诊断 VLA 的 OOD 鲁棒性，不需要真实机器人执行
4. [[IGAR]] 通过推理时 attention recalibration 提升 LGS，从而改善 OOD 指令跟随性能

## 代表工作
- [[IGAR]]: 提出 LGS 指标并用于诊断 OpenVLA 和 RDT 在 OOD 场景下的语言基础失效

## 相关概念
- [[IGAR]]
- [[OpenVLA]]
- [[VLA]]

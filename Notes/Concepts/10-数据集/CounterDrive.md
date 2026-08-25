---
type: concept
aliases: [CounterDrive Dataset, 反事实驾驶数据集]
---

# CounterDrive

## 定义

CounterDrive 是由 RISE 团队构建的**反事实自动驾驶视频数据集**，通过视频生成模型对真实驾驶场景进行反事实编辑，生成包含不同风险等级结果的配对片段，用于训练 World Action Model 的风险识别能力。

## 数学形式

无固定公式，核心是构建因果-反事实配对：

$$
\mathcal{D}_{\text{CF}} = \{(v_i^{\text{real}}, v_i^{\text{cf}}, r_i, t_i^{\text{inc}})\}
$$

其中 $v^{\text{real}}$ 为原始驾驶视频，$v^{\text{cf}}$ 为反事实生成视频，$r_i$ 为风险类别，$t_i^{\text{inc}}$ 为事故起始帧。

## 核心要点

1. **规模**: NAVSIM 5,013 训练 + 1,000 测试；nuScenes 2,432 训练 + 511 测试
2. **生成流程**: Wan 2.7 → 1080p 10s 视频 → OpenVO 位姿恢复 → 人工标注
3. **三类别**: 正常 / 非自车原因事故 / 自车原因事故
4. **效果**: 风险识别 AUC 从 ~0.50（随机）提升至 0.93-0.96

## 代表工作

- [[RISE]]: 数据集的提出者，用于训练 Latent Evaluator 的风险区分能力

## 相关概念

- [[NAVSIM]]
- [[nuScenes]]
- [[WAM]]

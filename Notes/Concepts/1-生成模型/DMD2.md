---
type: concept
aliases: [Distribution Matching Distillation 2, 分布匹配蒸馏2]
---

# DMD2

## 定义

DMD 的改进版本，结合分布匹配蒸馏、得分一致性损失和对抗监督，实现扩散/流匹配模型的 few-step 蒸馏，相比原始 DMD 加入 GAN 判别器头提升生成质量。

## 数学形式

综合蒸馏目标：

$$
\mathcal{L}_{DMD2} = \lambda_{dm}\mathcal{L}_{distill} + \lambda_{score}\mathcal{L}_{score} + \lambda_{GAN}\mathcal{L}_{GAN}
$$

ODE 蒸馏（可选预热）：

$$
\mathcal{L}_{ODE} = \mathbb{E}_{x_t,t}\left[\|\hat{x}^0_{teacher}(x_t, t) - \hat{x}^0_{student}(x_t, t)\|_2^2\right]
$$

## 核心要点

1. **三组件损失**：分布匹配 + 得分一致性 + 对抗损失，提供比 DMD 更稳定的训练
2. **GAN 判别器头**：多层特征提取（如第 5、15、25、35、39 层）提升感知质量
3. **TTUR 训练**：生成器与判别器异步更新（TTUR=5 表示判别器每步更新5次）
4. **EMA 稳定**：通过指数移动平均保持训练稳定
5. **ODE 预热可选**：先做 ODE 蒸馏热启动再做 DMD2，可进一步提升最终效果

## 代表工作

- [[GigaWorld1]]: Stage 4 蒸馏阶段使用 DMD2，将多步世界模型压缩为 few-step 推理，训练 2250 步（32× H20）

## 相关概念

- [[DMD]]: 前身方法，DMD2 在其基础上加入对抗监督
- [[Distribution Matching Distillation]]: 同义词
- [[流匹配]]: DMD2 可应用于流匹配模型的蒸馏
- [[对抗训练]]: GAN 判别器头的核心

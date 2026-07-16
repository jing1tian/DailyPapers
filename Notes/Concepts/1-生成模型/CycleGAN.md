---
type: concept
aliases: [Cycle-Consistent GAN, 循环一致 GAN, 无监督图像翻译]
---

# CycleGAN

## 定义
CycleGAN 是 Berkeley 提出的无监督图像域翻译方法，通过循环一致性约束（cycle-consistency loss）在无配对数据的情况下学习两个图像域之间的双向映射。

## 数学形式
$$
\mathcal{L}_\text{cyc} = \mathbb{E}[\|F(G(x)) - x\|_1] + \mathbb{E}[\|G(F(y)) - y\|_1]
$$

$G: X \to Y$，$F: Y \to X$，循环一致要求 $F(G(x)) \approx x$。

## 核心要点
1. **无需配对数据**：只需两个域的无标签图像集合即可训练
2. **循环一致性**：正向+逆向翻译后还原原图，约束翻译的信息保存
3. **两对判别器**：$D_X$ 和 $D_Y$ 分别保证各域的真实感
4. **局限**：不能改变图像结构（形状、布局），仅能迁移纹理风格

## 代表工作
- [[CycleGAN]]: Zhu et al., ICCV 2017
- [[DayNightRobot]]: 用 CycleGAN + CLIP 做农业机器人白天到夜间的感知域迁移

## 相关概念
- [[Classifier-Free Guidance]]
- [[ControlNet]]

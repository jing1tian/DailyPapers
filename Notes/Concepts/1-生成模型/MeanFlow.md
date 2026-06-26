---
type: concept
aliases: [MeanFlow, 平均流]
---

# MeanFlow

## 定义

用"平均速度场"参数化一致性轨迹映射（CTM）的少步生成模型训练目标：网络直接学习区间 $[s, t]$ 上的平均速度，单次前向即可从噪声跳跃到任意中间时刻，再结合切线修正项保证与瞬时速度场的一致性。

## 数学形式

CTM 一般形式（区间 $t \to s$ 的一致性映射）：

$$
\mathcal{L}_{\text{sCTM}}(\theta)=\mathbb{E}_{\bm{x}_{0},\bm{\epsilon},t,s}\left[\frac{\|\Delta_{\theta}\|_{2}^{2}}{\|\Delta_{\theta^{-}}\|_{2}^{2}+c}\right]
$$

MeanFlow 中切线差的具体形式：

$$
\Delta_{\theta}=\bm{v}_{\theta}(\bm{x}_{t},t,s)-\bm{v}+(t-s)\,\texttt{JVP}(\bm{v}_{\theta^{-}},(\bm{x}_{t},t,s),(\bm{v},1,0))
$$

- $\bm{v}$: 瞬时速度目标（如 RF 下的 $\bm{\epsilon}-\bm{x}_0$）
- $\texttt{JVP}(\cdot,\cdot,\cdot)$: Jacobian-向量积算子，第三个参数为切向方向
- $s$: 目标时间步，可为任意中间时刻（而非仅 0）

## 核心要点

1. **平均速度 vs 瞬时速度**：标准流匹配/扩散模型学习瞬时速度场 $\bm{v}_\theta(\bm{x}_t, t)$；MeanFlow 学习区间 $[s,t]$ 上的平均速度 $\bm{v}_\theta(\bm{x}_t,t,s)$，使单次网络评估即可完成从 $t$ 到 $s$ 的跳跃式生成
2. **切线修正项保证一致性**：用 [[JVP]] 计算的切线修正项 $(t-s)\cdot\texttt{JVP}(\cdot)$ 把瞬时速度场的局部信息注入平均速度场，确保两者理论一致
3. **属于 CTM（一致性轨迹模型）路线**：与 [[sCM]] 同属连续时间一致性建模家族，sCM 关注终点一致性（$s=0$），MeanFlow/CTM 关注任意区间一致性
4. **少步/单步采样**：训练好后可单步或少步采样，是扩散蒸馏中重要的少步加速范式之一

## 代表工作

- MeanFlow (Geng et al., 2025): 原始提出工作
- [[Causal-rCM]]: 首次将 MeanFlow 类连续时间一致性目标引入 Teacher-Forcing 自回归视频扩散训练，与 [[sCM]] 并列为 TF-CM 阶段可选目标函数

## 相关概念

- [[sCM]]
- [[JVP]]
- [[一致性蒸馏]]
- [[Rectified Flow]]

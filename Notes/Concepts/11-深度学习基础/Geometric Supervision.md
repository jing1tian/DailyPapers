---
type: concept
aliases: [几何监督, Geometry Supervision, Geometric Grounding]
---

# Geometric Supervision

## 定义

通过冻结的 3D 感知基础模型（如 VGGT）对视觉编码特征施加余弦相似度约束，迫使视觉表示保留空间几何结构信息的训练技术。

## 数学形式

$$
\mathcal{L}_{\mathrm{geo}} = \frac{1}{N_v^m} \sum_{j=1}^{N_v^m} \left[1 - \cos\left(\hat{\bm{Z}}_{t,j}^{G}, \bm{Z}_{t,j}^{G}\right)\right]
$$

其中 $\hat{\bm{Z}}^G$ 为策略视觉特征的投影，$\bm{Z}^G$ 为冻结几何模型的目标特征。

## 核心要点

1. 几何教师（如 VGGT）在训练时提供空间感知目标，**推理时完全去除**，无额外开销
2. 通过余弦相似度损失对齐视觉 patch 特征与 3D 特征，注入空间归纳偏置
3. 改善操作策略对背景变换、光照变化、新颖物体等分布偏移的鲁棒性
4. 可与其他辅助监督（如世界建模）互补组合

## 代表工作

- [[SG-WAM]]: 结合 VGGT 几何监督 + 自引导世界预测器，仿真均值提升 +1.3pp，OOD 鲁棒性显著增强

## 相关概念

- [[VGGT]]
- [[Stop-Gradient]]
- [[EMA]]
- [[Self-Guided World Predictor]]

---
type: concept
aliases: [Human Mesh Recovery, 人体网格重建]
---

# HMR

## 定义
从单张图像或视频中回归人体三维网格参数（通常是 SMPL/SMPL-X 参数）的任务与方法族。

## 数学形式
$$\hat{\theta}, \hat{\beta} = f_\text{HMR}(I)$$

其中 $I$ 为输入图像，$\hat{\theta}$ 为 SMPL 姿态参数，$\hat{\beta}$ 为体型参数，$f_\text{HMR}$ 为回归网络。

## 核心要点
1. 输入为 RGB 图像（单帧或序列），输出为 SMPL/SMPL-X 参数，不直接输出顶点坐标
2. 典型 pipeline：图像特征提取 → 迭代回归（IEF）或 Transformer 解码 → SMPL 参数
3. 挑战：深度歧义、遮挡、视角变化、多人场景中的个体区分
4. 常用于 4D 人体重建的前置步骤（骨架估计 → 引导多视角生成 → 4DGS 重建）

## 代表工作
- [[4DAnyone]]：以 HMR 作为骨架条件，驱动多视角一致视频生成再升维为 4DGS

## 相关概念
- [[SMPL]]
- [[4DGS]]
- [[16-人体动作/ViTPose]]

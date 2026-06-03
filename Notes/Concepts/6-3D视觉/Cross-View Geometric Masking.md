---
type: concept
aliases: [Cross-View Geometric Masking, 跨视图几何遮罩, 视锥遮罩, Sight-Cone Masking, Tube Patch Masking]
---

# Cross-View Geometric Masking

## 定义

跨视图几何遮罩是 WALL-WM 中用于多视图视频 DiT 训练的两种互补注意力约束机制，通过几何约束确保多相机特征的跨视图交互符合物理可见性约束，同时通过时序管状遮罩强化自我一致性。

## 数学形式

### 视锥注意力遮罩（Sight-Cone Masking）

像素 $u$ 的视锥射线：

$$
\mathbf{p}(u, t) = \mathbf{p}_0(u) + t\hat{\mathbf{v}}(u)
$$

两视角射线最近点时刻：

$$
(t_1, t_2) = \arg\min_{t_1, t_2} \|\mathbf{p}(u, t_1) - \mathbf{p}(u', t_2)\|_2
$$

截断到深度范围后判断视锥相交：

$$
\mathcal{C}(u) \cap \mathcal{C}(u') \Longleftrightarrow \|\mathbf{p}(u, \hat{t}_1) - \mathbf{p}(u', \hat{t}_2)\|_2 \leq \hat{t}_1 \gamma(u) + \hat{t}_2 \gamma(u')
$$

二值遮罩矩阵：

$$
\mathcal{M}_{sc}[u, u'] = 1 \Longleftrightarrow \mathcal{C}(u) \cap \mathcal{C}(u') \neq \emptyset
$$

## 核心要点

1. **视锥注意力遮罩（Sight-Cone Masking）**: 仅允许几何视野相交的补丁对跨视角互相注意，将 3D 物理共可见性编码为注意力偏置 $-\infty$（屏蔽不可见对），视角内部对始终允许
2. **管状补丁遮罩（Tube Patch Masking）**: 以概率 $p_\text{tube}$ 对某一视角的 $k \times k$ 空间窗口在所有潜帧上施加遮罩，强迫跨视图注意力从其他视角恢复被遮罩内容，对管内 vv-prediction 损失上采样权重
3. 两种遮罩均为**仅训练时使用**（training-only），推理时路径保持无标定、无额外计算开销
4. 解耦视锥约束与相机标定：无需精确外参即可提供几何感知的注意力引导

## 代表工作

- [[WALL-WM]]: 提出并使用 Cross-View Geometric Masking，在多视图视频 DiT 预训练中增强跨视角几何一致性

## 相关概念

- [[多视图几何建模]]
- [[Camera RoPE]]
- [[流匹配]]

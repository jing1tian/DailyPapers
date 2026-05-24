---
type: concept
aliases: [Neural Radiance Fields, 神经辐射场]
---

# NeRF

## 定义
Neural Radiance Fields：用 MLP 将 3D 坐标和观察方向映射为体密度和颜色，通过体渲染合成任意视角图像的隐式场景表示方法。

## 数学形式
$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$$

体渲染积分：
$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\,dt$$

其中 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,ds\right)$ 为累积透射率。

## 核心要点
1. 隐式表示：场景编码在 MLP 权重中，存储紧凑但渲染慢
2. 位置编码：用傅里叶特征提升 MLP 对高频细节的拟合能力
3. 分层采样：粗网络 + 细网络双阶段采样提高效率
4. 局限性：单场景训练慢（数小时），推理慢，不支持动态场景

## 代表工作
- [[NeRF]]（Mildenhall et al., 2020）: 原始论文
- [[3D Gaussian Splatting]]: 用显式 Gaussian 基元替代隐式 MLP，训练和渲染速度大幅提升

## 相关概念
- [[3D Gaussian Splatting]]
- [[TideGS]]
- [[GaussianDream]]

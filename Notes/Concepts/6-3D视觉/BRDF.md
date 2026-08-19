---
type: concept
aliases: [Bidirectional Reflectance Distribution Function, 双向反射分布函数]
---

# BRDF (Bidirectional Reflectance Distribution Function)

## 定义
描述光从入射方向 $\omega_i$ 照射到表面后，沿出射方向 $\omega_o$ 反射的辐亮度比值函数，是物理渲染的核心材质表示。

## 数学形式
$$f_r(\omega_i, \omega_o) = \frac{dL_o(\omega_o)}{L_i(\omega_i)\cos\theta_i \, d\omega_i}$$

## 核心要点
1. **漫反射分量**：Lambert 模型（各向同性）
2. **镜面反射分量**：Cook-Torrance 微表面模型（依赖法向量分布 D、菲涅尔项 F、遮蔽项 G）
3. **在 3DGS 中的应用**：[[SpotlessGS]] 用 BRDF 分解把外观拆为材质属性 + 光照，实现 relightable 3DGS

## 相关概念
- [[NeRF]]
- [[3D Gaussian Splatting]]
- [[SpotlessGS]]

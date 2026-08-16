---
type: concept
aliases: [Digital Beamforming, 数字波束成形, DBF]
---

# Digital Beamforming

## 定义

通过对天线阵列各单元接收信号进行相位加权求和，在数字域合成指定方向的波束，将稀疏天线信号转化为角度域的空间功率分布图（热力图）。

## 数学形式

$$
P(\theta, \phi) = 20 \log_{10} \left\| \sum_{k=1}^{K} a_k e^{j\phi_k} e^{-j\Delta_k(\theta, \phi)} \right\|
$$

$$
\Delta_k(\theta, \phi) = \frac{2\pi}{\lambda}\!\left( d_k^x \cos\phi \sin\theta + d_k^y \sin\phi \right)
$$

其中 $K$ 为天线单元数，$a_k$、$\phi_k$ 为幅度和初相，$\Delta_k$ 为相位延迟，$\lambda$ 为信号波长。

## 核心要点

1. **空间滤波**: 通过调整权重矩阵，增强特定方向信号、抑制其他方向干扰
2. **2D 热力图输出**: 扫描所有方向 $(\theta, \phi)$ 生成角度-强度热力图，可与 RGB 图像对齐
3. **应用场景**: 毫米波雷达、麦克风阵列均可用波束成形生成空间分布图
4. **与声学的联系**: 声学阵列的波束成形原理相同，用于生成声强方向图

## 代表工作

- [[MuseVLA]]: 使用数字波束成形将 Calterah 4T4R 60GHz 毫米波雷达和麦克风阵列信号转化为 2D 热力图

## 相关概念

- [[mmWave Radar]]
- [[Grounded Sensor Image]]
- [[Homography]]

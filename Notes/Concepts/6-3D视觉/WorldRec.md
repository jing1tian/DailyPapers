---
type: concept
aliases: [WorldRec 1.0, WorldRec 2.0, 世界重建, World Reconstruction]
---

# WorldRec（世界重建模块）

## 定义

小米汽车世界模型系列中的 3D 场景重建组件，以**单次前向传播**同时重建静态背景和动态目标的 [[3DGS|3D 高斯]]，渲染为深度/语义/光流先验后作为 World Control 注入生成模型，支持长程时序一致性。

## 核心架构（WorldRec 2.0）

三阶段寄存器交互机制：

$$
\mathbf{r}' = \text{Broadcast}\!\left(\text{Communicate}\!\left(\text{Compress}(\mathbf{f}_{\text{patch}},\ \mathbf{r})\right)\right)
$$

- **Compress**: 局部 patch 特征聚合进全局寄存器 token
- **Communicate**: 寄存器间全局通信（跨视角/跨时序信息融合）
- **Broadcast**: 全局寄存器信息广播回 patch 特征

**World Control 注入**:

$$
\mathbf{w}_t = \text{Cat}\!\left[\text{Render}_{\text{depth}}(\mathcal{G}_t),\ \text{Render}_{\text{sem}}(\mathcal{G}_t),\ \text{Render}_{\text{flow}}(\mathcal{G}_t)\right]
$$

## 核心要点

1. **动态实例解耦**（2.0 新增）：用动态实例 Query 初始化分离静态背景高斯与动态目标高斯，动态 D-RMSE 较 1.0 降低 34%（6.07 → 3.99，Waymo）
2. **单次前向**：静态 + 动态高斯联合重建，推理效率高于两阶段方案
3. **渲染先验**：深度/语义/光流三通道渲染为显式几何约束，解决纯隐式模型长程漂移问题
4. **跨数据集泛化**：Waymo 训练后，nuScenes 零样本 PSNR 28.37，Argoverse 2 零样本 PSNR 26.49

## 代表工作

- [[XiaomiAutoWM2]]: WorldRec 2.0 作为 3D 记忆组件，支持 7 路 45 秒长程驾驶视频生成及闭环仿真

## 相关概念

- [[3DGS]]: 3D 高斯场景表示基础
- [[4DGS]]: 动态 3D 高斯（WorldRec 动态实例部分的相关方法）
- [[View-Temporal Attention]]: WorldRec 渲染的 World Control 注入的生成架构
- [[3D记忆库]]: 场景记忆相关概念

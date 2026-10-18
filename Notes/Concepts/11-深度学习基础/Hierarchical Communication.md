---
type: concept
aliases: [分层通信, 层次通信, Hierarchical Player-Group-Global Communication]
---

# Hierarchical Communication（分层通信）

## 定义

在多智能体或多实体 Transformer 中，将通信分为局部组内通信（player→group）和全局跨组通信（group→global）的两级分层结构，以避免全连接的二次方代价和中心化方法的跨事件干扰。

## 数学形式

分层通信序列（Multics 的注意力掩码）：

$$
\mathbf{x}_{\mathrm{hier}} = [\mathbf{x}_1, \ldots, \mathbf{x}_N, \mathbf{h}^L, \mathbf{h}^G]_{\mathrm{seq}}
$$

通信路径：

$$
\mathbf{x}_i \leftrightarrow \mathbf{h}^L_{g(i)} \leftrightarrow \mathbf{h}^G
$$

## 核心要点

1. **三级层次**: 玩家层（个体信息）→ 组层（局部群体信息）→ 全局层（全场景信息）
2. **复杂度控制**: 局部组内通信降低组间不必要的信息交换，总代价随玩家数线性增长（而非全连接的 $O(N^2)$）
3. **与 MARL 的关联**: 从多智能体强化学习（CommNet、ATOC、TarMAC）的通信拓扑设计中借鉴思想，应用于高维视觉 Transformer 表示

## 代表工作

- [[Multics]]: 将分层通信引入多玩家视频世界模型，在4玩家设置下 GeoKPM 从 Solaris 的 48.51% 提升至 54.17%

## 相关概念

- [[Group Token]]
- [[Dynamic Grouping]]
- [[自注意力机制]]
- [[世界模型 (World Model)]]

---
type: concept
aliases: [Physics-Guided Residual Dynamics, PGRD]
---

# PGRD

## 定义
Physics-Guided Residual Dynamics：将物理仿真器的基础预测与学习到的残差修正相结合的可形变物体仿真框架。

## 数学形式
$$\hat{s}_{t+1} = f_{\text{physics}}(s_t, a_t) + f_{\text{residual}}(s_t, a_t; \theta)$$

其中 $f_{\text{physics}}$ 为 Spring-Mass 或 MPM 物理模型，$f_{\text{residual}}$ 为可学习残差网络。

## 核心要点
1. 物理模型提供基础轨迹（低数据需求），残差网络修正系统误差
2. 用 RealSense 相机采集真实变形序列作为训练监督
3. 下游用 MPPI 进行操作规划
4. 对比纯学习方法（GBND）和纯物理方法均有优势

## 代表工作
- PGRD（2607.13451, UIUC + Columbia）: 针对布料/软体物体的系统实现

## 相关概念
- [[MPM]]
- [[Spring-Mass Model]]
- [[MPPI]]

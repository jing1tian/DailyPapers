---
type: concept
aliases: [JVP, Jacobian-Vector Product, 雅可比向量积, 前向模式自动微分, forward-mode autodiff]
---

# JVP（Jacobian-Vector Product）

## 定义

前向模式自动微分算子，给定函数 $f$ 在点 $x$ 处的输入及一个切向量 $v$，单次前向传播即可计算雅可比矩阵与该切向量的乘积 $J_f(x) v$，无需显式构造完整雅可比矩阵或执行反向传播。

## 数学形式

$$
\texttt{JVP}(f, x, v) = J_f(x)\, v = \lim_{\epsilon \to 0} \frac{f(x+\epsilon v) - f(x)}{\epsilon}
$$

其中 $J_f(x) = \partial f/\partial x$ 为雅可比矩阵。在一致性模型训练中常用于计算教师网络沿 PF-ODE 轨迹方向的切线：

$$
\bm{g}=w(t)\frac{\mathrm{d}\bm{f}_{\theta^{-}}(\bm{x}_{t},t)}{\mathrm{d}t} = \texttt{JVP}\big(\bm{f}_{\theta^-}, (\bm{x}_t, t), (\bm{v}_{\text{teacher}}, 1)\big)
$$

## 核心要点

1. **前向模式 vs 反向模式**：反向模式自动微分（标准反向传播）一次计算 VJP（向量-雅可比积，适合标量损失对多参数求梯度）；前向模式 JVP 适合"少量输入方向、需要完整输出"的场景，如沿时间轴求解的切线方向
2. **单次前向计算切线**：在连续时间一致性模型（[[sCM]]、[[MeanFlow]]）中，JVP 让"教师轨迹切线方向"无需显式多步 ODE 求解或反向传播，只需一次扩展前向即完成
3. **与并行/注意力基础设施的兼容性挑战**：JVP 需要对网络每层暴露成对的 primal-tangent 接口（$(\bm{x}, t\bm{x}) \to (\bm{y}, t\bm{y})$），与 [[FlashAttention]]、[[FSDP2]]、上下文并行等大规模训练设施的协同需要专门工程设计（如自研 custom-mask FlashAttention-2 JVP 内核）
4. **层级实现优于全局实现**：在 FSDP2 包装的大模型上，按层实现 JVP（FSDP2(JVP) 设计）比对整个模型套用全局 `torch.func.jvp`（JVP(FSDP2)）更可行，因为切线传播可以在每层前向局部完成

## 代表工作

- [[rCM]]: 首次让 JVP 计算兼容 FlashAttention、FSDP、上下文并行等大规模训练设施，用于双向扩散模型的连续时间一致性蒸馏
- [[Causal-rCM]]: 设计 custom-mask FlashAttention-2 JVP 内核，并提出 FSDP2(JVP) 分层实现方案与 JVP×Ulysses CP 兼容方案，首次在因果自回归 Teacher-Forcing 设定下落地 JVP 驱动的连续时间一致性模型（TF-sCM/TF-MeanFlow）

## 相关概念

- [[sCM]]
- [[MeanFlow]]
- [[FlashAttention]]
- [[FSDP2]]

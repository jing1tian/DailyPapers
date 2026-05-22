---
type: concept
aliases: [Control Barrier Function, 控制屏障函数]
---

# CBF (Control Barrier Functions)

## 定义
控制屏障函数，一种保证系统安全约束（如碰撞避免）的控制理论工具，通过在控制器中嵌入不等式约束，确保系统状态始终在安全集内。

## 数学形式
定义安全集 $\mathcal{S} = \{x : h(x) \geq 0\}$，CBF 要求存在控制 $u$ 使得：
$$\dot{h}(x, u) \geq -\alpha(h(x))$$
其中 $\alpha$ 是 class-$\mathcal{K}$ 函数。通常通过 QP 求解：
$$u^* = \arg\min_u \|u - u_{ref}\|^2 \quad \text{s.t.} \quad L_fh + L_gh \cdot u \geq -\alpha h$$

## 核心要点
1. 前向不变性：一旦系统在安全集内，CBF 保证它永远不会离开
2. 与 CLF（Control Lyapunov Function）结合，可同时保证安全性和稳定性
3. 通过 QP 在线求解，计算高效
4. 在机器人导航、自动驾驶中广泛应用

## 代表工作
- [[SAFER]]：在 3DGS 场景中用 CBF 平衡安全导航与主动感知

## 相关概念
- [[EIG]]
- [[QP]]

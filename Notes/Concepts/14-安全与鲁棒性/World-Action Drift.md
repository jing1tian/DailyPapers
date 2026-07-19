---
type: concept
aliases: [世界动作漂移, WAM-specific vulnerability, World-Action Desynchronization]
---

# World-Action Drift（世界动作漂移）

## 定义
World-Action Drift 是 [[WAM|World-Action Model]] 特有的安全漏洞：在保持预测未来视觉合理性（imagination plausibility）的同时，动作输出可被单独劫持并偏离正确轨迹，导致任务失败。

## 数学形式

$$
\max_{\|\delta\|_\infty \leq \varepsilon} D_{\text{act}}(a^\delta, a) \quad \text{s.t.} \quad D_{\text{img}}(z^\delta, z) \leq \tau_{\text{img}}
$$

即：在想象漂移 $D_{\text{img}}$ 受约束的条件下，最大化动作空间偏移 $D_{\text{act}}$。

## 核心要点
1. **与传统攻击的区别**：传统对抗攻击只针对动作/分类输出；World-Action Drift 专注于 WAM 中动作与世界预测的解耦性
2. **实证基础**：失败 episode 的动作位移显著大于成功 episode，而预测未来位移在两者之间几乎无差别（成功/失败轨迹的想象漂移分布重叠）
3. **安全监控漏洞**：基于预测未来的安全监控器（`safe = M(z_{t+1:t+K}, g)`）无法检测此类攻击，因为想象看起来合理
4. **跨架构普遍性**：在 Joint WAM、IDM WAM、Action-only WAM 三种架构中均存在

## 代表工作
- [[BadWAM]]: 首次提出并系统研究 World-Action Drift 漏洞，设计两种攻击实例化（纯动作攻击 + [[Imagination-Preserving Attack|保形攻击]]）

## 相关概念
- [[WAM]]: 存在此漏洞的模型类别
- [[Imagination-Preserving Attack]]: World-Action Drift 的一种攻击实例化
- [[对抗补丁攻击]]: 传统视觉对抗攻击（不针对 WAM 解耦性）
- [[BadVLA]]: 针对 VLA 的后门攻击（训练时攻击，与 BadWAM 的推理时攻击不同）

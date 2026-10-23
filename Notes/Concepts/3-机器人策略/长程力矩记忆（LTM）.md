---
type: concept
aliases: [LTM, Long-Term Wrench Memory, 长程力矩记忆, wrench memory]
---

# 长程力矩记忆（LTM）

## 定义

长程力矩记忆（Long-Term Wrench Memory，LTM）是对近期运动-力矩配对历史的补充：仅记录更长时间窗口内的力矩序列（无运动命令），为需要感知持续载荷张力或累积接触历史的任务提供足够的上下文。

## 数学形式

$$
\mathcal{H}_t^W = [W_{-H_\ell}^-, \ldots, W_{-1}^-]
$$

共 $H_\ell = 150$ 个间隔，覆盖约 10 s 历史（$f_h = 15$ Hz）。与近期配对历史 $\mathcal{H}_t^{AW}$（$H_s = 24$ 步，1.6 s）并联作为模型输入。

## 核心要点

1. **解耦长短期记忆**: 近期配对历史关注运动-力矩的即时因果关系，LTM 专注于捕获较慢的载荷变化或持续接触状态。
2. **对插销插入影响显著**: LTM 将插销插入成功率从 7/20 提升至 14/20，表明 1.6 s 内的信息不足以感知插入进度积累的阻力。
3. **对力矩预测也有改善**: 开环消融中 LTM 降低 $e_F$（2.84→2.79 N）和 $e_I$（2.17→2.01 N·s）。

## 代表工作

- [[CW-WAM]]: 提出 LTM，通过 10 s 力矩记忆实现载荷自适应操作

## 相关概念

- [[Contact-Wrench]]
- [[世界动作模型（WAM）]]
- [[Pair-Causal Attention]]
- [[Force Memory Token]]

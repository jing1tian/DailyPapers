---
type: concept
aliases: [语言-目标绑定, language grounding, 指称定位]
---

# Language-Object Binding（语言-目标绑定）

## 定义

语言-目标绑定指机器人策略将自然语言指令中的指称词（referential expression）准确对应到当前视觉观测中的具体物体实例的过程；绑定成功意味着策略能稳定追踪任务相关物体并生成可靠动作。

## 数学形式

无标准统一公式；在 [[Bind-WAM]] 中通过 IoU（指称定位）、可见性标志和 [[DINOv3]] 特征相似度三项指标联合衡量绑定质量：

$$
y_t^r = \begin{cases} 1, & \mathrm{IoU}_r < 0.3 \;\lor\; v_t^r = 0 \;\lor\; s_t^r < 0.9 \\ 0, & \text{otherwise} \end{cases}
$$

## 核心要点

1. **指称定位（Reference Grounding）**: 将语言描述（如 "the red mug on the left"）定位到视觉中唯一物体实例；
2. **绑定失效来源**: 物体被遮挡（不可见）、外观漂移（光照/角度变化导致特征不一致）、指称歧义（多个相似物体）；
3. **与动作质量强耦合**: 绑定不稳定时，策略动作生成抖动或失败，需要未来想象辅助。

## 代表工作

- [[Bind-WAM]]: 以语言-目标绑定不确定性为信号，选择性触发 WAM 的未来想象

## 相关概念

- [[Binding Uncertainty]]
- [[SAM3]]
- [[DINOv3]]
- [[Qwen3-VL]]
- [[WAM|世界动作模型（WAM）]]

---
type: concept
aliases: [Set-of-Mark Prompting, SoM, 标记集合提示]
---

# Set-of-Mark Prompting

## 定义

Set-of-Mark（SoM）Prompting 是一种视觉标注方法，先用分割模型（如 SAM）对图像中的所有区域进行实例分割，然后为每个区域覆盖数字/字母标记，再将标注后图像输入 VLM，使模型能够通过引用标记 ID 来指代特定区域，从而实现精确的空间理解和目标定位。

## 数学形式

无固定公式；本质是图像增强流程：

$$\text{SoM}(I) = \text{overlay\_marks}(\text{SAM}(I), I) \rightarrow \text{VLM 可引用的视觉输入}$$

## 核心要点

1. 利用 SAM 等分割模型分割所有可见区域，无需人工标注。
2. 为每个区域添加数字 ID 覆盖层，VLM 可在文本中以"区域 3"等形式引用。
3. 显著提升 VLM 的细粒度空间定位和指代能力，尤其适合操作任务中的目标选择。
4. 在 MR0 中用于生成 2D Grounding 和目标可供性训练数据，配合 SAM2 自动化标注。

## 代表工作

- [[MR0]]: 用 Set-of-Mark + SAM 生成空间 Grounding 和可供性数据，CALVIN 消融验证其有效性

## 相关概念

- [[SAM2]]
- [[Vision-Language-Action]]

---
type: concept
aliases: [模板化游程编码, Template RLE, 基于模板的RLE]
---

# Template-based RLE

## 定义

一种将 3D 体素几何编码为紧凑文本序列的表示方法：将体素沿 z 轴切片为 2D 二值掩码，对每层应用游程编码（RLE），相邻层共享结构模板、只记录残差变化，大幅减少 token 数量。

## 数学形式

$$
\text{encoded}_k = \begin{cases} \text{RLE}(\text{slice}_k) & \text{if } k \text{ is template layer} \\ \Delta(\text{slice}_k, \text{template}) & \text{otherwise} \end{cases}
$$

## 核心要点

1. **Z轴切片**: 将零件级体素分解为按 z 轴排列的 2D 二值掩码序列
2. **2D RLE 编码**: 对每个切片的占据区域应用经典游程编码，转为文本 token
3. **模板层共享**: 多个相邻切片共享同一结构模板，只存储相对变化（残差），避免重复编码
4. **VLM 友好**: 无需引入额外特殊 token，输出为纯文本 token，可直接用于自回归 VLM
5. **结构保持**: 相比坐标索引表示，保留显式几何结构，自回归预测更鲁棒

## 代表工作

- [[PhysX-Omni]]: 首次提出 Template-based RLE 用于仿真就绪 3D 资产生成

## 相关概念

- [[游程编码]]
- [[视觉语言模型]]
- [[自回归语言模型]]
- [[TRELLIS]]

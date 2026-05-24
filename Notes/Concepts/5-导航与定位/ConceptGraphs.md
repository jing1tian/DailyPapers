---
type: concept
aliases: [Open-Vocabulary 3D Scene Graphs]
---

# ConceptGraphs

## 定义
开放词汇 3D 场景图，将 LLM/VLM 的语义理解与 3D 空间表示结合，自动从 RGB-D 视频中构建层次化场景图（物体节点 + 空间关系边），支持自然语言查询和任务规划。

## 数学形式
$$\mathcal{G} = (\mathcal{V}, \mathcal{E}), \quad v_i = \{\text{3D bbox}_i, \text{CLIP feat}_i, \text{LLM desc}_i\}$$

## 核心要点
1. 从 RGB-D 流自动分割、跟踪、合并物体，赋予 CLIP 语义特征
2. 用 LLM 自动生成物体描述和关系，构建可查询场景图
3. 支持"找到厨房里的杯子"等高层任务查询
4. 不需要预定义物体类别（开放词汇）

## 代表工作
- [[ConceptGraphs]]：Gu et al. 2023，ICRA 2024
- [[MIF]]：人形机器人动态导航中集成 ConceptGraphs 做场景理解

## 相关概念
- [[LangSplat]]
- [[SLAM]]
- [[World Model]]

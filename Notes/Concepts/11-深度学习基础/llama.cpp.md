---
type: concept
aliases: [llama cpp, llama.cpp 推理引擎]
---

# llama.cpp

## 定义

由 Georgi Gerganov 开发的纯 C/C++ LLM 推理引擎，基于 [[ggml]] 张量库，支持量化推理，可在消费级硬件和边缘设备上运行大型语言模型。

## 核心要点

1. **跨平台**: 无需 CUDA，支持 CPU / Apple Metal / CUDA / Vulkan 等多后端
2. **量化支持**: 原生支持 Q4/Q8 等多种量化格式，通过 GGUF 格式打包模型
3. **低内存占用**: 专为内存受限设备设计，可在 8GB 内存设备上运行 7B 模型
4. **扩展性**: 社区基于其构建了多种变体（如 whisper.cpp、vla.cpp）

## 代表工作

- [[vla.cpp]]: 在 llama.cpp 基础上扩展，支持 VLA 模型推理（cross-attention KV cache、flow-matching 动作头）

## 相关概念

- [[ggml]]: 底层张量计算库
- [[GGUF]]: llama.cpp 使用的模型格式
- [[vla.cpp]]: VLA 专用推理运行时

---
type: concept
aliases: [GGUF format, gguf, GGUF 格式]
---

# GGUF

## 定义

GGUF（GGML Universal Format）是 [[llama.cpp]] 定义的模型存储格式，将模型权重、tokenizer、元数据打包为单一二进制文件，替代旧版 GGML 格式。

## 核心要点

1. **单文件打包**: 权重、tokenizer 词表、归一化统计量等全部存于一个 `.gguf` 文件
2. **自描述**: 文件头包含模型架构、量化类型、超参数等完整元数据
3. **版本兼容**: 支持向前兼容，新版运行时可读旧版 GGUF 文件
4. **vla.cpp 扩展**: vla.cpp 在标准 GGUF 中额外存储动作归一化统计量

## 代表工作

- [[vla.cpp]]: 将七种 VLA 架构统一打包为 GGUF bundle，含权重 + tokenizer + 归一化统计

## 相关概念

- [[ggml]]: GGUF 的底层库
- [[llama.cpp]]: GGUF 的主要消费方

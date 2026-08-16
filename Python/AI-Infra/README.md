# AI-Infra

AI 基础设施相关的算子实现与推理优化笔记。按「可运行算子」和「概念+核心代码笔记」两类归置。

| 文件 | 类型 | 内容 | 运行 |
|---|---|---|---|
| `gemm.py` | 可运行 (numpy) | GEMM 的分块 / 循环展开 / 预取 / 块主序布局 | `python gemm.py` 自测对齐 `np.matmul` |
| `flash_attention.py` | 可运行 (torch/CPU) | Vanilla / Flash(单循环) / Optimized(双循环 FA2) | `python flash_attention.py` 自测对齐 Vanilla（含 causal） |
| `triton_flash_attention.py` | 可运行 (torch+triton, **GPU-only**) | FA2 前向的 Triton kernel | `python triton_flash_attention.py`，无 GPU/triton 时自动跳过 |
| `llm_inference_notes.py` | 笔记 + 核心代码 | 开篇注意力 9 步；Part A–K 覆盖 Transformer/MLA/GDN/HiCache/EAGLE/通信/compile·graph/DeepGEMM·DeepEP/EP 模型 | 以阅读为主；不保证单文件一键跑通 |

## `llm_inference_notes.py` 目录

| Part | 主题 |
|---|---|
| 开篇 | 注意力 9 步总览 + 优化轴与冲突 |
| A | RoPE / GQA / KV Cache / FLOPs |
| B | MLA 矩阵吸收 |
| C | 线性注意力 / Gated DeltaNet |
| D | Paged + RadixTree + HiCache + 异构 KV |
| E | EAGLE 投机解码 |
| F | 集合通信 Mesh / Ring / 分层 AR / EP all2all |
| G | torch.compile × CUDA Graph（语义 mock） |
| H | vLLM/SGLang 风格执行栈骨架 |
| I | DeepGEMM masked grouped GEMM |
| J | DeepEP low-latency dispatch/combine |
| K | EP 三耦合：显存→batch、步时间、EPLB |

## 设计约定

- **公共骨架抽取**：`flash_attention.py` 里 QKV 投影 / 多头 reshape / 输出投影收敛到
  `AttentionBase`，各变体只实现 `_attention(q, k, v)`。
- **自测即文档**：每个可运行文件的 `__main__` 都用 `allclose` 对齐基线（Vanilla / `np.matmul`），
  改代码后直接 `python <file>.py` 即可回归。
- **笔记**：`llm_inference_notes.py` 保留原理与核心实现，已去掉原附录的 `__main__` 自测打印。
- **环境**：numpy、torch(CPU 即可)。Triton kernel 需 GPU + triton，未安装时自动跳过。

## 已知限制

- `triton_flash_attention.py` 需 GPU + triton；无此环境时未做数值验证，仅作算法结构参考。
- `gemm.py` 的 `gemm_unrolled` 是纯 Python 三重循环，仅演示循环展开，性能远低于 BLAS。
- `llm_inference_notes.py` 中 Part H 等片段依赖 GPU API，仅供对照阅读。

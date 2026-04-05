# LLM Inference Benchmarking & Optimization Suite

> GPT-2 · PyTorch · FP32 / FP16 / INT8 · CUDA & AMD ROCm compatible

A clean, well-documented pipeline for benchmarking and optimizing LLM inference performance across precisions and batch sizes — designed to run on any GPU backend (NVIDIA CUDA or AMD ROCm/HIP).

---

## What it does

| Step | Description |
|------|-------------|
| **Backend Detection** | Auto-detects CUDA, AMD ROCm (HIP), or CPU — same code runs on both |
| **Baseline Benchmarking** | Measures latency (ms), throughput (tok/s), and peak VRAM across batch sizes |
| **FP16 Optimization** | Half-precision inference with Tensor Core utilization analysis |
| **INT8 Quantization** | Dynamic INT8 via `torch.quantization` with model size comparison |
| **Operator Profiling** | `torch.profiler` breakdown of GEMM, Attention, and LayerNorm costs |
| **Visualization** | Clean plots comparing all configurations |

---

## Results (Tesla T4 · GPT-2 124M)

| Precision | Batch Size | Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) |
|-----------|-----------|-------------|-------------------|----------------|
| FP32      | 1         | ~310        | ~420              | ~530           |
| FP32      | 8         | ~850        | ~844              | ~890           |
| FP16      | 1         | ~308        | ~425              | ~270           |
| FP16      | 8         | ~810        | ~884              | ~460           |
| INT8      | 1 (CPU)   | ~1400       | ~45               | N/A            |

**Key finding:** FP16 ≈ FP32 throughput on T4 at small batch sizes — GPT-2 inference is **memory-bandwidth-bound**, not compute-bound. FP16 still delivers ~2x VRAM reduction, allowing larger batch sizes or more model instances per GPU. Meaningful compute-bound speedup appears at larger model scales (7B+) or higher batch sizes where GEMM dimensions fully utilize Tensor Cores.

> On AMD MI300X (5.3 TB/s HBM3 vs T4's 320 GB/s GDDR6), the crossover from memory-bound to compute-bound happens at much smaller batch sizes, making FP16/BF16 gains more pronounced even for smaller models.

---

## Roofline Intuition

```
Arithmetic Intensity = FLOPs / Bytes accessed

GPT-2 @ bs=1:  ~0.3 FLOP/byte  ← well below T4's ridge point (~9 FLOP/byte)
GPT-2 @ bs=64: ~8.0 FLOP/byte  ← approaching compute-bound regime
LLaMA-7B @ bs=1: ~1.2 FLOP/byte ← still memory-bound but FP16 helps more
```

This is why AMD's MI300X HBM3 bandwidth is a genuine competitive advantage for LLM inference at production batch sizes.

---

## AMD ROCm Compatibility

This notebook is designed to run unmodified on AMD ROCm. Key mappings:

| Operation | NVIDIA CUDA | AMD ROCm |
|-----------|------------|----------|
| GEMM (`aten::mm`) | cuBLAS | rocBLAS / Composable Kernel |
| Attention | FlashAttention-2 | CK-Flash / ROCm FA2 fork |
| Synchronization | `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| Profiling | CUPTI | ROCm Tracer / rocprof |
| INT8 quantization | TensorRT / torch.ao | AMD Quark |

To run on ROCm:
```bash
# Install ROCm PyTorch (replace cu121 with rocm5.7)
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
# Everything else is identical
```

For BF16 on MI300X (preferred over FP16 on AMD CDNA3):
```python
model = model.bfloat16()  # replace model.half()
```

---

## Setup & Usage

### Run in Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sai18022001/llm-inference-benchmark/blob/main/LLM_Inference_Benchmarking.ipynb)

> Runtime → Change runtime type → **T4 GPU** → Run All (~5–8 min)

### Run locally

```bash
git clone https://github.com/sai18022001/llm-inference-benchmark.git
cd llm-inference-benchmark

pip install torch transformers accelerate matplotlib seaborn pandas
jupyter notebook LLM_Inference_Benchmarking.ipynb
```

---

## Project Structure

```
llm-inference-benchmark/
├── LLM_Inference_Benchmarking.ipynb   # Main notebook (11 cells)
├── README.md
└── benchmark_results.png              # Output plots (generated on run)
```

---

## Tech Stack

- **PyTorch** — model loading, quantization, profiling
- **HuggingFace Transformers** — GPT-2 weights and tokenizer
- **torch.profiler** — operator-level kernel timing
- **torch.quantization** — dynamic INT8 quantization
- **matplotlib / seaborn** — result visualization

---

## Extending This

- Swap `gpt2` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for a more production-realistic benchmark
- Add `torch.compile()` (inductor backend) for an additional optimization pass
- Enable Flash Attention 2: `model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")`
- Multi-GPU: wrap with `torch.nn.DataParallel` or DeepSpeed ZeRO

---

## References

- [PyTorch ROCm documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD Composable Kernel](https://github.com/ROCm/composable_kernel)
- [Roofline Model — Williams et al.](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [AMD Quark Quantization](https://quark.docs.amd.com/)

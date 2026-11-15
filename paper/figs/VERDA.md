# What ran on Verda

Machine: Verda instance with 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, compute capability 12.0), 30 vCPU, 88 GB RAM, driver 580.178.04 / CUDA 13.0.
Software: PyTorch 2.14.0+cu130, vLLM 0.33.0 (runs on Blackwell; `VLLM_USE_FLASHINFER_SAMPLER=0` because the FlashInfer sampler wants to JIT-compile without `ninja`/`nvcc`), transformers 5.20.2, peft 0.21.3.
Date: 2026-11-15, 02:17 to about 04:20 UTC (about 2.1 hours of machine time; this is the GPU-hour figure to bill).

| Job | What | GPU time |
|---|---|---|
| Setup | install, vLLM smoke test, simulator timing | ~0.2 h |
| `grpo/factorial.sh` | GRPO, Gemma 3 1B + LoRA, think+anon and think+named, 50 steps each, run concurrently on the one GPU | ~0.5 h wall (both together) |
| `grpo/phase2.sh` | no-think+anon and no-think+named (50 steps), think+anon continued to 100 steps, test-seed evaluation, zero-shot Gemma 3 1B and 4B | ~0.5 h wall |
| `grpo/phase3.sh` | GRPO, Gemma 3 4B + LoRA, think+anon, up to 50 steps (stopped at a fixed deadline), test-seed evaluation | ~0.6 h wall |

Per-run step time (summed, the runs shared the GPU) is in the macros `grpo<Run>GpuHours`; the whole session used the machine for about 2 hours. Cost: about 2.1 GPU hours. At an assumed on-demand rate of `grpoGpuRate` (1.70) EUR per hour (not checked against the Verda invoice; replace it with the real rate) that is about 3.6 EUR, far inside the 200 EUR budget. No paid external API was called.

## Throughput
- vLLM with the live LoRA adapter (re-loaded from disk each step): 64 reasoning answers of up to 1024 tokens in a few seconds; the 64 simulator rewards per step (20-year episodes, `fast=True`, about 1.7 s each on one core) dominate, run in a process pool of 7 to 10 workers per run.
- One GRPO step (8 prompts x 8 answers, dynamic sampling, LoRA update of a 1B model over up to 1500-token sequences): about 20 to 35 s with three or four runs sharing the GPU and CPUs.
- Peak GPU memory: about 43 GB per 1B think run (HF backward over a 262k vocabulary for 1024-token answers, plus a 11 GB vLLM engine), so two think runs plus two no-think runs did not fit at once; the no-think runs went second.

## Why not a laptop
Run A (Gemma 3 270M, 64 tokens) took about 75 s per step on a 32-core CPU VM. A 1B model writing up to 1024 reasoning tokens for 64 answers per step, then back-propagating through all of them, is several hundred times more compute per step; on CPU one step would take hours. A laptop GPU (8 to 16 GB) cannot hold a 1B-model backward pass over 64 sequences of 1500 tokens with a 262k-token vocabulary together with a vLLM engine, let alone the four concurrent runs here, and the 4B zero-shot evaluation. The 96 GB card ran four GRPO runs and the evaluation side by side in under two hours.

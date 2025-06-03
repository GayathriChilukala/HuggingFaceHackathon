# HuggingFaceHackathon

# 📊 Vision-Language Model Benchmark

This repository benchmarks the performance of state-of-the-art vision-language models that generate descriptive captions from input images.

---

## 🚀 Performance Summary

| Model ID        | Type        | Provider   | Success Rate | Avg Latency | Median Latency | Min / Max Latency | Avg Caption Length | Requests |
|------------------|-------------|------------|--------------|-------------|----------------|--------------------|---------------------|----------|
| `Qwen2.5-VL`     | HuggingFace | Hyperbolic | 85.0%        | 2.35s       | 2.19s          | 1.47s / 3.93s      | 296 chars           | 17 / 20  |
| `LLaMA-4-Scout`  | HuggingFace | Novita     | 85.0%        | 2.60s       | 2.39s          | 1.17s / 5.78s      | 243 chars           | 17 / 20  |
| `Gemma-3-27B`    | HuggingFace | Nebius     | 85.0%        | **1.82s**   | 1.78s          | 1.32s / 2.40s      | **359 chars**       | 17 / 20  |
| `GPT-4.1`        | GitHub      | GitHub     | **95.0%**    | 2.16s       | 2.14s          | 1.20s / 3.82s      | 239 chars           | 19 / 20  |

---

## 🏢 Provider Comparison

| Provider    | Models Tested | Success Rate | Avg Latency |
|-------------|----------------|--------------|-------------|
| Hyperbolic  | 1              | 85.0%        | 2.35s       |
| Novita      | 1              | 85.0%        | 2.60s       |
| Nebius      | 1              | 85.0%        | **1.82s**   |
| GitHub      | 1              | **95.0%**    | 2.16s       |

---

## 🏅 Highlights

- ⚡ **Fastest Model:** `Gemma-3-27B` (1.82s avg latency) – *Provider: Nebius*
- 🔒 **Most Reliable:** `GPT-4.1` (95.0% success rate) – *Provider: GitHub*
- ✨ **Most Descriptive Captions:** `Gemma-3-27B` (359 characters avg)

---

> 📌 All benchmarks were conducted with consistent input prompts and image URLs. Metrics reflect real-world performance under standard API use for caption generation tasks.

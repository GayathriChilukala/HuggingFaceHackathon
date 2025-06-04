# HuggingFaceHackathon

## 🔍 Model Benchmark Overview

### 🖼️ Vision-Language Models

| Model ID          | Type        | Provider   | Avg Latency | Median Latency | Min / Max Latency | Avg Caption Length |
|-------------------|-------------|------------|-------------|----------------|--------------------|---------------------|
| `qwen_2_5_vl`      | huggingface | hyperbolic | 2.35s       | 2.19s          | 1.47s / 3.93s      | 296 chars           |
| `llama_4_scout`    | huggingface | novita     | 2.60s       | 2.39s          | 1.17s / 5.78s      | 243 chars           |
| `gemma_3_27b`      | huggingface | nebius     | 1.82s       | 1.78s          | 1.32s / 2.40s      | 359 chars           |
| `github_gpt41`     | github      | github     | 2.16s       | 2.14s          | 1.20s / 3.82s      | 239 chars           |

---

### 📄 Text Models

| Model ID                        | Type        | Provider   | Avg Latency | Median Latency | Avg Captions/Request | Total Captions | Avg Response Length |
|---------------------------------|-------------|------------|-------------|----------------|-----------------------|----------------|----------------------|
| `mixtral_8x7b_together`         | huggingface | together   | 4.88s       | 3.73s          | 3.0                   | 27             | 374 chars            |
| `llama_3_2_1b_novita`           | huggingface | novita     | 2.65s       | 2.70s          | 3.0                   | 27             | 599 chars            |
| `google/gemma-2-9b-nebius`      | huggingface | nebius     | 1.63s       | 1.45s          | 3.0                   | 30             | 531 chars            |
| `deepseek_r1_qwen3_8b_novita`   | huggingface | novita     | 7.87s       | 7.80s          | 3.0                   | 27             | 2414 chars           |

---

> These benchmarks were conducted to evaluate the performance of various vision-language and text models across key metrics such as latency and output quality. The results help identify optimal models for different multimodal and generative use cases.

## 🔍 Model Benchmark Overview

This benchmark evaluates the performance of various **Vision-Language** and **Text Generation** models based on latency and output characteristics.

---

### 🖼️ Vision-Language Models

| Model ID        | Type        | Provider   | Avg Latency | Median Latency | Min / Max Latency | Avg Caption Length |
|-----------------|-------------|------------|-------------|----------------|--------------------|---------------------|
| `qwen_2_5_vl`    | huggingface | hyperbolic | 2.35s       | 2.19s          | 1.47s / 3.93s      | 296 chars           |
| `llama_4_scout`  | huggingface | novita     | 2.60s       | 2.39s          | 1.17s / 5.78s      | 243 chars           |
| `gemma_3_27b` 🏆 | huggingface | nebius     | **1.82s**   | **1.78s**      | 1.32s / 2.40s      | 359 chars           |
| `github_gpt41`   | github      | github     | 2.16s       | 2.14s          | 1.20s / 3.82s      | 239 chars           |

**✅ Chosen Vision Model:** `gemma_3_27b`  
- Fastest average and median latency  
- Generates rich captions (359 chars avg)  
- Ideal for high-performance visual captioning tasks

---

### 📄 Text Models

| Model ID                      | Type        | Provider   | Avg Latency | Median Latency | Avg Captions/Request | Total Captions | Avg Response Length |
|-------------------------------|-------------|------------|-------------|----------------|-----------------------|----------------|----------------------|
| `mixtral_8x7b_together`       | huggingface | together   | 4.88s       | 3.73s          | 3.0                   | 27             | 374 chars            |
| `llama_3_2_1b_novita`         | huggingface | novita     | 2.65s       | 2.70s          | 3.0                   | 27             | 599 chars            |
| `google/gemma-2-9b-nebius`    | huggingface | nebius     | **1.63s**   | **1.45s**      | 3.0                   | 30             | 531 chars            |
| `deepseek_r1_qwen3_8b_novita` 🏆 | huggingface | novita     | 7.87s       | 7.80s          | 3.0                   | 27             | **2414 chars**        |

**✅ Chosen Text Model:** `google/gemma-2-9b-nebius`  
- Fastest overall latency among all text models  
- Balanced output length with strong quality  
- Ideal for generating **multiple caption variants** efficiently

---

### 💡 Strategy & Key Points

- For **Vision Captioning**, I selected a **general inference model** (`gemma_3_27b`) due to its superior speed and descriptive output quality.
- For **Text Captioning**, I chose a **lower-cost model** (`gemma-2-9b-nebius`) to **generate versions of captions** from the primary visual output.
- The selected text model:
  - Generates **multiple high-quality variants** per caption.
  - Is optimized for **latency and cost**, making it suitable for real-time pipelines.
  - Enables efficient caption expansion from the vision model’s output.

> This two-stage approach ensures optimized latency and cost-efficiency without compromising caption quality.

# Rexx HR Q&A System - Evaluation Results

Generated: 2025-11-24

## Table 1: Baseline vs Fine-Tuned Model Comparison

| Model | F1 | Cosine | Precision | Recall | Time (s) |
|-------|---:|-------:|----------:|-------:|---------:|
| TinyLlama-1.1B (Baseline) | 0.1873 | 0.1568 | 0.1440 | 0.3613 | 25.4 |
| TinyLlama-1.1B (Fine-tuned) | 0.2034 | 0.1686 | 0.1654 | 0.3535 | 25.1 |
| Qwen2.5-1.5B (Baseline) | 0.1222 | 0.1803 | 0.0708 | 0.4853 | 44.5 |
| Qwen2.5-1.5B (Fine-tuned) | 0.1290 | 0.1738 | 0.0769 | 0.4890 | 47.3 |

## Table 2: RAG Generator Model Comparison

| Generator Model | F1 | Cosine | Precision | Recall | Time (s) |
|-----------------|---:|-------:|----------:|-------:|---------:|
| gemma2:2b | 0.2024 | 0.2482 | 0.1313 | 0.4933 | 7.6 |
| tinyllama | 0.1621 | 0.2040 | 0.1027 | 0.4031 | 3.5 |
| llama2 | 0.1196 | 0.1252 | 0.0781 | 0.2733 | 11.5 |

**Best:** gemma2:2b (F1: 0.2024)

## Table 3: Embedding Model Comparison (Retrieval)

| Embedding Model | Dim | Relevance | Top Sim | Time (s) |
|-----------------|----:|----------:|--------:|---------:|
| all-mpnet-base-v2 | 768 | 0.4008 | 0.6634 | 19.8 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.3540 | 0.6947 | 2.0 |
| all-MiniLM-L6-v2 | 384 | 0.3431 | 0.5770 | 2.7 |

**Best:** all-mpnet-base-v2 (Relevance: 0.4008)

## Table 4: Overall Comparison (All Methods)

| Method | F1 Score | Cosine Sim |
|--------|----------:|-----------:|
| Fine-tuned (TinyLlama + LoRA) | 0.2034 | 0.1686 |
| RAG (gemma2:2b) | 0.2024 | 0.2482 |
| Baseline (TinyLlama-1.1B) | 0.1873 | 0.1568 |
| Fine-tuned (Qwen2.5 + LoRA) | 0.1290 | 0.1738 |
| Baseline (Qwen2.5-1.5B) | 0.1222 | 0.1803 |

**Best Method:** Fine-tuned (TinyLlama + LoRA) (F1: 0.2034)

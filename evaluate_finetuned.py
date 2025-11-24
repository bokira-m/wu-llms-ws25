"""
Evaluation Script for Fine-tuned Qwen2.5-1.5B-Instruct Model
Compares fine-tuned model performance against baseline
"""

import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_baseline import BaselineEvaluator

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def load_finetuned_model(model_path: str):
    """Load the fine-tuned model with LoRA weights."""
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading base model: {base_model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from: {model_path}", flush=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    """Generate answer using the fine-tuned model."""
    prompt = f"""### Instruction:
Beantworte die folgende Frage über rexx HR Software präzise und auf Deutsch.

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer part
    if "### Answer:" in full_response:
        answer = full_response.split("### Answer:")[-1].strip()
        # Stop at next ### if model starts repeating
        if "###" in answer:
            answer = answer.split("###")[0].strip()
    else:
        answer = full_response

    return answer


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    model_path = os.path.join(script_dir, "qwen25_finetuned_model")

    # Load fine-tuned model
    print("Loading fine-tuned model...", flush=True)
    model, tokenizer = load_finetuned_model(model_path)
    print("Model loaded!", flush=True)

    # Load test dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_questions = data['test']
    print(f"\nEvaluating {len(test_questions)} test questions with fine-tuned model...", flush=True)
    print("=" * 60, flush=True)

    # Initialize evaluator for metrics
    evaluator = BaselineEvaluator(model_name="qwen25-finetuned")

    results = []
    start_time = time.time()

    for i, item in enumerate(test_questions):
        question = item['question']
        expected = item['expected_answer']

        print(f"\n[{i+1}/{len(test_questions)}] {question[:50]}...", flush=True)

        # Generate answer
        q_start = time.time()
        generated = generate_answer(model, tokenizer, question)
        q_time = time.time() - q_start

        # Calculate metrics
        metrics = evaluator.calculate_metrics(generated, expected)

        result = {
            "id": item.get('id', f'test_{i}'),
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "category": item.get('category', 'unknown'),
            "difficulty": item.get('difficulty', 'unknown'),
            "metrics": metrics,
            "response_time": round(q_time, 2)
        }
        results.append(result)

        print(f"   Cosine Sim: {metrics['cosine_similarity']:.2f} | F1: {metrics['f1_score']:.2f} | Time: {q_time:.1f}s", flush=True)

    total_time = time.time() - start_time

    # Calculate aggregate metrics
    aggregate = {}
    metrics_keys = ['cosine_similarity', 'precision', 'recall', 'f1_score', 'key_terms_ratio']

    for key in metrics_keys:
        values = [r['metrics'][key] for r in results]
        aggregate[f'avg_{key}'] = round(sum(values) / len(values), 4)

    # By category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['metrics']['f1_score'])

    aggregate['by_category'] = {
        cat: round(sum(scores)/len(scores), 4)
        for cat, scores in categories.items()
    }

    aggregate['total_time_seconds'] = round(total_time, 2)
    aggregate['avg_response_time'] = round(total_time / len(test_questions), 2)

    # Save results
    output = {
        "metadata": {
            "model": "Qwen2.5-1.5B-Instruct-finetuned",
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "finetuned": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_questions": len(test_questions)
        },
        "aggregate_metrics": aggregate,
        "results": results
    }

    output_path = os.path.join(script_dir, "qwen25_finetuned_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("FINE-TUNED MODEL EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Model: Qwen2.5-1.5B-Instruct (fine-tuned on rexx data)", flush=True)
    print(f"Questions Evaluated: {len(test_questions)}", flush=True)
    print(f"Total Time: {aggregate['total_time_seconds']}s", flush=True)
    print(f"Avg Response Time: {aggregate['avg_response_time']}s", flush=True)

    print("\n--- Overall Metrics ---", flush=True)
    print(f"Avg Cosine Similarity: {aggregate['avg_cosine_similarity']:.4f}", flush=True)
    print(f"Avg Precision: {aggregate['avg_precision']:.4f}", flush=True)
    print(f"Avg Recall: {aggregate['avg_recall']:.4f}", flush=True)
    print(f"Avg F1 Score: {aggregate['avg_f1_score']:.4f}", flush=True)

    print("\n--- F1 Score by Category ---", flush=True)
    for cat, score in aggregate['by_category'].items():
        print(f"  {cat}: {score:.4f}", flush=True)

    # Load baseline for comparison
    baseline_path = os.path.join(script_dir, "qwen25_baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        print("\n" + "=" * 60, flush=True)
        print("COMPARISON: BASELINE (Qwen2.5) vs FINE-TUNED (Qwen2.5)", flush=True)
        print("=" * 60, flush=True)

        baseline_agg = baseline['aggregate_metrics']

        metrics_compare = ['avg_cosine_similarity', 'avg_precision', 'avg_recall', 'avg_f1_score']

        print(f"\n{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Improvement':>15}", flush=True)
        print("-" * 66, flush=True)

        for metric in metrics_compare:
            base_val = baseline_agg.get(metric, 0)
            fine_val = aggregate.get(metric, 0)
            improvement = fine_val - base_val
            pct_improvement = (improvement / base_val * 100) if base_val > 0 else 0

            sign = "+" if improvement > 0 else ""
            print(f"{metric:<25} {base_val:>12.4f} {fine_val:>12.4f} {sign}{improvement:>10.4f} ({sign}{pct_improvement:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

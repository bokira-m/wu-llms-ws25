"""
Evaluation Script for RAG System
Evaluates RAG performance on test dataset and compares with baseline
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_system import RAGSystem
from evaluate_baseline import BaselineEvaluator

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    chroma_path = os.path.join(script_dir, "chroma_db")

    # Check if index exists
    if not os.path.exists(chroma_path):
        print("ERROR: RAG index not found. Run 'python rag_system.py' first to build the index.", flush=True)
        return

    # Initialize RAG system with Ollama llama2
    print("Initializing RAG system...", flush=True)
    rag = RAGSystem(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generator_model="llama2",
        persist_directory=chroma_path
    )

    # Load models
    rag.load_embedding_model()
    rag.init_vector_store()
    rag.load_generator_model()

    print(f"Index contains {rag.collection.count()} document chunks", flush=True)

    # Load test dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_questions = data['test']
    print(f"\nEvaluating {len(test_questions)} test questions with RAG...", flush=True)
    print("=" * 60, flush=True)

    # Initialize evaluator for metrics
    evaluator = BaselineEvaluator(model_name="tinyllama-rag")

    results = []
    start_time = time.time()

    for i, item in enumerate(test_questions):
        question = item['question']
        expected = item['expected_answer']

        print(f"\n[{i+1}/{len(test_questions)}] {question[:50]}...", flush=True)

        # Query RAG
        q_start = time.time()
        generated, sources = rag.query(question, top_k=3)
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
            "response_time": round(q_time, 2),
            "sources": [s['source'] for s in sources]
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
            "model": "llama2 + RAG",
            "embedding_model": "all-MiniLM-L6-v2",
            "generator_model": "llama2 (Ollama)",
            "retrieval_top_k": 3,
            "rag_enabled": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_questions": len(test_questions)
        },
        "aggregate_metrics": aggregate,
        "results": results
    }

    output_path = os.path.join(script_dir, "rag_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("RAG EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Model: TinyLlama-1.1B + RAG (MiniLM-L6-v2)", flush=True)
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

    # Compare with baseline
    baseline_path = os.path.join(script_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        print("\n" + "=" * 60, flush=True)
        print("COMPARISON: BASELINE vs RAG (llama2 + MiniLM)", flush=True)
        print("=" * 60, flush=True)

        baseline_agg = baseline['aggregate_metrics']

        metrics_compare = ['avg_cosine_similarity', 'avg_precision', 'avg_recall', 'avg_f1_score']

        print(f"\n{'Metric':<25} {'Baseline':>12} {'RAG':>12} {'Improvement':>15}", flush=True)
        print("-" * 66, flush=True)

        for metric in metrics_compare:
            base_val = baseline_agg.get(metric, 0)
            rag_val = aggregate.get(metric, 0)
            improvement = rag_val - base_val
            pct_improvement = (improvement / base_val * 100) if base_val > 0 else 0

            sign = "+" if improvement > 0 else ""
            print(f"{metric:<25} {base_val:>12.4f} {rag_val:>12.4f} {sign}{improvement:>10.4f} ({sign}{pct_improvement:.1f}%)", flush=True)


if __name__ == "__main__":
    main()

"""
RAG Model Comparison Script
Compares multiple Ollama models for RAG generation quality
"""

import json
import os
import sys
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_system import RAGSystem
from evaluate_baseline import BaselineEvaluator

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Models to compare (3 smallest/fastest)
MODELS_TO_TEST = [
    "tinyllama",        # 1.1B - sehr schnell
    "llama2",           # 7B - Baseline
    "gemma2:2b",        # 2B - gut für Deutsch
]


def get_available_models():
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'].replace(':latest', '') for m in data.get('models', [])]
    except:
        pass
    return []


def evaluate_model(rag, model_name, test_questions, evaluator, max_questions=10):
    """Evaluate a single model on test questions."""
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Switch model
    rag.generator_model_name = model_name

    results = []
    start_time = time.time()

    # Limit questions for quick comparison
    questions_to_test = test_questions[:max_questions]

    for i, item in enumerate(questions_to_test):
        question = item['question']
        expected = item['expected_answer']

        print(f"[{i+1}/{len(questions_to_test)}] {question[:40]}...", flush=True)

        q_start = time.time()
        answer, sources = rag.query(question, top_k=3)
        q_time = time.time() - q_start

        metrics = evaluator.calculate_metrics(answer, expected)

        results.append({
            "question": question,
            "expected": expected,
            "generated": answer,
            "metrics": metrics,
            "time": round(q_time, 2)
        })

        print(f"   F1: {metrics['f1_score']:.2f} | Cosine: {metrics['cosine_similarity']:.2f} | {q_time:.1f}s", flush=True)

    total_time = time.time() - start_time

    # Calculate averages
    avg_metrics = {}
    for key in ['cosine_similarity', 'precision', 'recall', 'f1_score', 'key_terms_ratio']:
        values = [r['metrics'][key] for r in results]
        avg_metrics[key] = round(sum(values) / len(values), 4)

    avg_metrics['avg_response_time'] = round(total_time / len(questions_to_test), 2)
    avg_metrics['total_time'] = round(total_time, 2)

    return {
        "model": model_name,
        "num_questions": len(questions_to_test),
        "metrics": avg_metrics,
        "results": results
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    chroma_path = os.path.join(script_dir, "chroma_db")

    # Check prerequisites
    if not os.path.exists(chroma_path):
        print("ERROR: RAG index not found. Run 'python rag_system.py' first.", flush=True)
        return

    # Get available models
    available = get_available_models()
    print(f"Available Ollama models: {available}", flush=True)

    # Filter to available models
    models_to_test = [m for m in MODELS_TO_TEST if m in available or m.replace(':latest', '') in available]
    print(f"Models to test: {models_to_test}", flush=True)

    if not models_to_test:
        print("ERROR: No models available to test!", flush=True)
        return

    # Initialize RAG system
    print("\nInitializing RAG system...", flush=True)
    rag = RAGSystem(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generator_model=models_to_test[0],
        persist_directory=chroma_path
    )
    rag.load_embedding_model()
    rag.init_vector_store()
    print(f"RAG ready! Index: {rag.collection.count()} chunks", flush=True)

    # Load test questions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_questions = data['test']

    # Initialize evaluator
    evaluator = BaselineEvaluator(model_name="rag-comparison")

    # Test each model
    all_results = {}
    for model in models_to_test:
        result = evaluate_model(rag, model, test_questions, evaluator, max_questions=10)
        all_results[model] = result

    # Save detailed results
    output_path = os.path.join(script_dir, "rag_model_comparison.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}", flush=True)

    # Print comparison table
    print("\n" + "=" * 90, flush=True)
    print("RAG MODEL COMPARISON SUMMARY", flush=True)
    print("=" * 90, flush=True)

    # Header
    print(f"\n{'Model':<20} {'F1 Score':>10} {'Cosine':>10} {'Precision':>10} {'Recall':>10} {'Avg Time':>10}", flush=True)
    print("-" * 90, flush=True)

    # Sort by F1 score
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['metrics']['f1_score'], reverse=True)

    for model, data in sorted_models:
        m = data['metrics']
        print(f"{model:<20} {m['f1_score']:>10.4f} {m['cosine_similarity']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['avg_response_time']:>9.1f}s", flush=True)

    print("-" * 90, flush=True)

    # Best model
    best_model = sorted_models[0][0]
    best_f1 = sorted_models[0][1]['metrics']['f1_score']
    print(f"\nBest performing model: {best_model} (F1: {best_f1:.4f})", flush=True)

    # Save summary CSV for easy import
    csv_path = os.path.join(script_dir, "rag_model_comparison.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Model,F1_Score,Cosine_Similarity,Precision,Recall,Key_Terms_Ratio,Avg_Response_Time\n")
        for model, data in sorted_models:
            m = data['metrics']
            f.write(f"{model},{m['f1_score']},{m['cosine_similarity']},{m['precision']},{m['recall']},{m['key_terms_ratio']},{m['avg_response_time']}\n")
    print(f"CSV summary saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

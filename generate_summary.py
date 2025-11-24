"""
Summary Report Generator
Generates comparison tables from all evaluation JSON files
"""

import json
import os
import sys
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def load_json_safe(path):
    """Load JSON file if exists, return None otherwise."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load all result files
    results = {
        # Qwen2.5 results
        'qwen25_baseline': load_json_safe(os.path.join(script_dir, 'qwen25_baseline_results.json')),
        'qwen25_finetuned': load_json_safe(os.path.join(script_dir, 'qwen25_finetuned_results.json')),
        # TinyLlama results (older)
        'tinyllama_baseline': load_json_safe(os.path.join(script_dir, 'baseline_results.json')),
        'tinyllama_finetuned': load_json_safe(os.path.join(script_dir, 'finetuned_results.json')),
        # RAG results
        'rag_results': load_json_safe(os.path.join(script_dir, 'rag_results.json')),
        'rag_model_comparison': load_json_safe(os.path.join(script_dir, 'rag_model_comparison.json')),
        'embedding_comparison': load_json_safe(os.path.join(script_dir, 'embedding_comparison.json')),
    }

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("REXX HR Q&A SYSTEM - EVALUATION SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # ==========================================
    # TABLE 1: BASELINE vs FINE-TUNED
    # ==========================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("TABLE 1: BASELINE vs FINE-TUNED MODEL COMPARISON")
    report_lines.append("=" * 80)

    baseline_finetuned_data = []

    # TinyLlama results
    if results['tinyllama_baseline']:
        agg = results['tinyllama_baseline']['aggregate_metrics']
        baseline_finetuned_data.append({
            'model': 'TinyLlama-1.1B (Baseline)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
            'precision': agg.get('avg_precision', 0),
            'recall': agg.get('avg_recall', 0),
            'key_terms': agg.get('avg_key_terms_ratio', 0),
            'time': agg.get('avg_response_time', 0),
        })

    if results['tinyllama_finetuned']:
        agg = results['tinyllama_finetuned']['aggregate_metrics']
        baseline_finetuned_data.append({
            'model': 'TinyLlama-1.1B (Fine-tuned)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
            'precision': agg.get('avg_precision', 0),
            'recall': agg.get('avg_recall', 0),
            'key_terms': agg.get('avg_key_terms_ratio', 0),
            'time': agg.get('avg_response_time', 0),
        })

    # Qwen2.5 results
    if results['qwen25_baseline']:
        agg = results['qwen25_baseline']['aggregate_metrics']
        baseline_finetuned_data.append({
            'model': 'Qwen2.5-1.5B (Baseline)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
            'precision': agg.get('avg_precision', 0),
            'recall': agg.get('avg_recall', 0),
            'key_terms': agg.get('avg_key_terms_ratio', 0),
            'time': agg.get('avg_response_time', 0),
        })

    if results['qwen25_finetuned']:
        agg = results['qwen25_finetuned']['aggregate_metrics']
        baseline_finetuned_data.append({
            'model': 'Qwen2.5-1.5B (Fine-tuned)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
            'precision': agg.get('avg_precision', 0),
            'recall': agg.get('avg_recall', 0),
            'key_terms': agg.get('avg_key_terms_ratio', 0),
            'time': agg.get('avg_response_time', 0),
        })

    if baseline_finetuned_data:
        report_lines.append(f"\n{'Model':<30} {'F1':>10} {'Cosine':>10} {'Precision':>10} {'Recall':>10} {'Time (s)':>10}")
        report_lines.append("-" * 80)
        for row in baseline_finetuned_data:
            report_lines.append(f"{row['model']:<30} {row['f1']:>10.4f} {row['cosine']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['time']:>10.1f}")

        # Calculate improvement if both exist
        if len(baseline_finetuned_data) == 2:
            base = baseline_finetuned_data[0]
            fine = baseline_finetuned_data[1]
            report_lines.append("-" * 80)
            f1_imp = ((fine['f1'] - base['f1']) / base['f1'] * 100) if base['f1'] > 0 else 0
            cos_imp = ((fine['cosine'] - base['cosine']) / base['cosine'] * 100) if base['cosine'] > 0 else 0
            report_lines.append(f"{'Improvement':<30} {f1_imp:>+9.1f}% {cos_imp:>+9.1f}%")
    else:
        report_lines.append("\nNo baseline/fine-tuned results found.")

    # ==========================================
    # TABLE 2: RAG GENERATOR COMPARISON
    # ==========================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("TABLE 2: RAG GENERATOR MODEL COMPARISON")
    report_lines.append("=" * 80)

    rag_gen_data = []

    if results['rag_model_comparison']:
        for model_name, data in results['rag_model_comparison'].items():
            if 'metrics' in data:
                m = data['metrics']
                rag_gen_data.append({
                    'model': model_name,
                    'f1': m.get('f1_score', 0),
                    'cosine': m.get('cosine_similarity', 0),
                    'precision': m.get('precision', 0),
                    'recall': m.get('recall', 0),
                    'time': m.get('avg_response_time', 0),
                })

    # Also add single RAG result if available
    if results['rag_results'] and not rag_gen_data:
        agg = results['rag_results']['aggregate_metrics']
        meta = results['rag_results'].get('metadata', {})
        rag_gen_data.append({
            'model': meta.get('generator_model', 'llama2') + ' + RAG',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
            'precision': agg.get('avg_precision', 0),
            'recall': agg.get('avg_recall', 0),
            'time': agg.get('avg_response_time', 0),
        })

    if rag_gen_data:
        # Sort by F1 score descending
        rag_gen_data.sort(key=lambda x: x['f1'], reverse=True)

        report_lines.append(f"\n{'Generator Model':<25} {'F1':>10} {'Cosine':>10} {'Precision':>10} {'Recall':>10} {'Time (s)':>10}")
        report_lines.append("-" * 80)
        for row in rag_gen_data:
            report_lines.append(f"{row['model']:<25} {row['f1']:>10.4f} {row['cosine']:>10.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['time']:>10.1f}")

        if len(rag_gen_data) > 1:
            report_lines.append("-" * 80)
            report_lines.append(f"Best: {rag_gen_data[0]['model']} (F1: {rag_gen_data[0]['f1']:.4f})")
    else:
        report_lines.append("\nNo RAG generator comparison results found.")

    # ==========================================
    # TABLE 3: EMBEDDING MODEL COMPARISON
    # ==========================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("TABLE 3: EMBEDDING MODEL COMPARISON (Retrieval)")
    report_lines.append("=" * 80)

    emb_data = []

    if results['embedding_comparison']:
        for model_name, data in results['embedding_comparison'].items():
            if 'error' not in data:
                emb_data.append({
                    'model': model_name.replace('sentence-transformers/', ''),
                    'relevance': data.get('avg_relevance', 0),
                    'top_sim': data.get('avg_top_similarity', 0),
                    'avg_sim': data.get('avg_similarity', 0),
                    'dim': data.get('embedding_dim', 0),
                    'time': data.get('eval_time', 0),
                })

    if emb_data:
        # Sort by relevance descending
        emb_data.sort(key=lambda x: x['relevance'], reverse=True)

        report_lines.append(f"\n{'Embedding Model':<45} {'Dim':>6} {'Relevance':>10} {'Top Sim':>10} {'Time (s)':>10}")
        report_lines.append("-" * 85)
        for row in emb_data:
            report_lines.append(f"{row['model']:<45} {row['dim']:>6} {row['relevance']:>10.4f} {row['top_sim']:>10.4f} {row['time']:>10.1f}")

        if len(emb_data) > 1:
            report_lines.append("-" * 85)
            report_lines.append(f"Best: {emb_data[0]['model']} (Relevance: {emb_data[0]['relevance']:.4f})")
    else:
        report_lines.append("\nNo embedding comparison results found.")

    # ==========================================
    # TABLE 4: OVERALL COMPARISON (All Methods)
    # ==========================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("TABLE 4: OVERALL COMPARISON (All Methods)")
    report_lines.append("=" * 80)

    overall_data = []

    # Add TinyLlama baseline
    if results['tinyllama_baseline']:
        agg = results['tinyllama_baseline']['aggregate_metrics']
        overall_data.append({
            'method': 'Baseline (TinyLlama-1.1B)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
        })

    # Add TinyLlama fine-tuned
    if results['tinyllama_finetuned']:
        agg = results['tinyllama_finetuned']['aggregate_metrics']
        overall_data.append({
            'method': 'Fine-tuned (TinyLlama + LoRA)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
        })

    # Add Qwen2.5 baseline
    if results['qwen25_baseline']:
        agg = results['qwen25_baseline']['aggregate_metrics']
        overall_data.append({
            'method': 'Baseline (Qwen2.5-1.5B)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
        })

    # Add Qwen2.5 fine-tuned
    if results['qwen25_finetuned']:
        agg = results['qwen25_finetuned']['aggregate_metrics']
        overall_data.append({
            'method': 'Fine-tuned (Qwen2.5 + LoRA)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
        })

    # Add best RAG
    if rag_gen_data:
        best_rag = rag_gen_data[0]
        overall_data.append({
            'method': f"RAG ({best_rag['model']})",
            'f1': best_rag['f1'],
            'cosine': best_rag['cosine'],
        })
    elif results['rag_results']:
        agg = results['rag_results']['aggregate_metrics']
        overall_data.append({
            'method': 'RAG (llama2 + MiniLM)',
            'f1': agg.get('avg_f1_score', 0),
            'cosine': agg.get('avg_cosine_similarity', 0),
        })

    if overall_data:
        # Sort by F1 score descending
        overall_data.sort(key=lambda x: x['f1'], reverse=True)

        report_lines.append(f"\n{'Method':<35} {'F1 Score':>12} {'Cosine Sim':>12}")
        report_lines.append("-" * 60)
        for row in overall_data:
            report_lines.append(f"{row['method']:<35} {row['f1']:>12.4f} {row['cosine']:>12.4f}")

        report_lines.append("-" * 60)
        report_lines.append(f"BEST METHOD: {overall_data[0]['method']} (F1: {overall_data[0]['f1']:.4f})")
    else:
        report_lines.append("\nNo results found for overall comparison.")

    # ==========================================
    # PRINT AND SAVE REPORT
    # ==========================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report to file
    report_path = os.path.join(script_dir, "evaluation_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    # Also save as CSV for Excel
    csv_path = os.path.join(script_dir, "evaluation_summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Overall comparison CSV
        f.write("Method,F1_Score,Cosine_Similarity\n")
        for row in overall_data:
            f.write(f"{row['method']},{row['f1']},{row['cosine']}\n")

        f.write("\n")

        # RAG generators CSV
        if rag_gen_data:
            f.write("RAG_Generator,F1_Score,Cosine_Similarity,Precision,Recall,Avg_Time\n")
            for row in rag_gen_data:
                f.write(f"{row['model']},{row['f1']},{row['cosine']},{row['precision']},{row['recall']},{row['time']}\n")

        f.write("\n")

        # Embeddings CSV
        if emb_data:
            f.write("Embedding_Model,Dimension,Relevance,Top_Similarity,Eval_Time\n")
            for row in emb_data:
                f.write(f"{row['model']},{row['dim']},{row['relevance']},{row['top_sim']},{row['time']}\n")

    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

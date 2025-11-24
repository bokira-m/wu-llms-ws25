"""
Baseline Evaluation Script for Rexx HR Q&A System
Evaluates Qwen2.5-1.5B-Instruct model (without fine-tuning) on test dataset
"""

import json
import sys
import time
from typing import Dict
from datetime import datetime
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


class BaselineEvaluator:
    """Evaluate LLM baseline performance on Q&A dataset."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.results = []
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load Qwen2.5 model from HuggingFace."""
        print(f"Loading model: {self.model_name}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("Model loaded!", flush=True)

    def query_model(self, question: str, max_new_tokens: int = 300) -> str:
        """Generate answer using Qwen2.5."""
        prompt = f"""### Instruction:
Beantworte die folgende Frage über rexx HR Software präzise und auf Deutsch.

### Question:
{question}

### Answer:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer part
        if "### Answer:" in full_response:
            answer = full_response.split("### Answer:")[-1].strip()
            # Stop at next ### if model starts repeating
            if "###" in answer:
                answer = answer.split("###")[0].strip()
        else:
            answer = full_response

        return answer

    def calculate_metrics(self, generated: str, expected: str) -> Dict:
        """Calculate evaluation metrics between generated and expected answer."""

        # Clean texts
        gen_clean = generated.lower().strip()
        exp_clean = expected.lower().strip()

        # 1. Exact Match
        exact_match = 1.0 if gen_clean == exp_clean else 0.0

        # 2. TF-IDF Cosine Similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gen_clean, exp_clean])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0

        # 3. Word Overlap (Jaccard-like)
        gen_words = set(re.findall(r'\w+', gen_clean))
        exp_words = set(re.findall(r'\w+', exp_clean))

        if len(exp_words) > 0:
            precision = len(gen_words & exp_words) / len(gen_words) if len(gen_words) > 0 else 0
            recall = len(gen_words & exp_words) / len(exp_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision, recall, f1 = 0, 0, 0

        # 4. Contains key terms check
        key_terms_found = sum(1 for word in exp_words if word in gen_words and len(word) > 4)
        key_terms_ratio = key_terms_found / max(len([w for w in exp_words if len(w) > 4]), 1)

        # 5. Length ratio
        length_ratio = min(len(gen_clean), len(exp_clean)) / max(len(gen_clean), len(exp_clean), 1)

        return {
            "exact_match": exact_match,
            "cosine_similarity": round(cosine_sim, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "key_terms_ratio": round(key_terms_ratio, 4),
            "length_ratio": round(length_ratio, 4)
        }

    def evaluate_dataset(self, dataset_path: str) -> Dict:
        """Run evaluation on test dataset."""

        # Load model if not loaded
        if self.model is None:
            self.load_model()

        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_questions = data['test']
        print(f"\nEvaluating {len(test_questions)} test questions...", flush=True)
        print(f"Model: {self.model_name}", flush=True)
        print("=" * 60, flush=True)

        self.results = []
        start_time = time.time()

        for i, item in enumerate(test_questions):
            question = item['question']
            expected = item['expected_answer']

            print(f"\n[{i+1}/{len(test_questions)}] {question[:50]}...", flush=True)

            # Query model
            q_start = time.time()
            generated = self.query_model(question)
            q_time = time.time() - q_start

            # Calculate metrics
            metrics = self.calculate_metrics(generated, expected)

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
            self.results.append(result)

            print(f"   Cosine Sim: {metrics['cosine_similarity']:.2f} | F1: {metrics['f1_score']:.2f} | Time: {q_time:.1f}s", flush=True)

        total_time = time.time() - start_time

        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics()
        aggregate['total_time_seconds'] = round(total_time, 2)
        aggregate['avg_response_time'] = round(total_time / len(test_questions), 2)

        return {
            "metadata": {
                "model": self.model_name,
                "finetuned": False,
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(test_questions)
            },
            "aggregate_metrics": aggregate,
            "results": self.results
        }

    def calculate_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all results."""
        if not self.results:
            return {}

        metrics_keys = ['cosine_similarity', 'precision', 'recall', 'f1_score', 'key_terms_ratio']

        aggregate = {}
        for key in metrics_keys:
            values = [r['metrics'][key] for r in self.results]
            aggregate[f'avg_{key}'] = round(sum(values) / len(values), 4)

        # By category
        categories = {}
        for r in self.results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['metrics']['f1_score'])

        aggregate['by_category'] = {
            cat: round(sum(scores)/len(scores), 4)
            for cat, scores in categories.items()
        }

        # By difficulty
        difficulties = {}
        for r in self.results:
            diff = r['difficulty']
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(r['metrics']['f1_score'])

        aggregate['by_difficulty'] = {
            diff: round(sum(scores)/len(scores), 4)
            for diff, scores in difficulties.items()
        }

        return aggregate

    def save_results(self, output_path: str, evaluation_results: Dict):
        """Save evaluation results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}", flush=True)

    def print_summary(self, evaluation_results: Dict):
        """Print evaluation summary."""
        agg = evaluation_results['aggregate_metrics']

        print("\n" + "=" * 60, flush=True)
        print("BASELINE EVALUATION SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"Model: {evaluation_results['metadata']['model']}", flush=True)
        print(f"Fine-tuned: {evaluation_results['metadata']['finetuned']}", flush=True)
        print(f"Questions Evaluated: {evaluation_results['metadata']['num_questions']}", flush=True)
        print(f"Total Time: {agg.get('total_time_seconds', 0)}s", flush=True)
        print(f"Avg Response Time: {agg.get('avg_response_time', 0)}s", flush=True)

        print("\n--- Overall Metrics ---", flush=True)
        print(f"Avg Cosine Similarity: {agg['avg_cosine_similarity']:.4f}", flush=True)
        print(f"Avg Precision: {agg['avg_precision']:.4f}", flush=True)
        print(f"Avg Recall: {agg['avg_recall']:.4f}", flush=True)
        print(f"Avg F1 Score: {agg['avg_f1_score']:.4f}", flush=True)
        print(f"Avg Key Terms Ratio: {agg['avg_key_terms_ratio']:.4f}", flush=True)

        print("\n--- F1 Score by Category ---", flush=True)
        for cat, score in agg.get('by_category', {}).items():
            print(f"  {cat}: {score:.4f}", flush=True)

        print("\n--- F1 Score by Difficulty ---", flush=True)
        for diff, score in agg.get('by_difficulty', {}).items():
            print(f"  {diff}: {score:.4f}", flush=True)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")

    # Initialize evaluator with Qwen2.5
    evaluator = BaselineEvaluator(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    print("Starting Qwen2.5-1.5B baseline evaluation...", flush=True)

    # Run baseline evaluation
    print("\n" + "=" * 60, flush=True)
    print("BASELINE EVALUATION (Qwen2.5-1.5B-Instruct without fine-tuning)", flush=True)
    print("=" * 60, flush=True)

    results = evaluator.evaluate_dataset(dataset_path)

    # Save results
    output_path = os.path.join(script_dir, "qwen25_baseline_results.json")
    evaluator.save_results(output_path, results)

    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
    
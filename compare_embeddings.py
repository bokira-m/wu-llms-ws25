"""
Embedding Model Comparison Script for RAG Retrieval
Compares different sentence-transformer models for document retrieval quality
"""

import json
import os
import sys
import time
import numpy as np
from typing import List, Dict

# Embeddings
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Embedding models to compare
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",                        # 384 dim, English (Baseline)
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # 384 dim, Multilingual (für Deutsch!)
    "sentence-transformers/all-mpnet-base-v2",                       # 768 dim, English (higher quality)
]


def load_chunks_from_chroma(chroma_path: str) -> List[Dict]:
    """Load existing chunks from ChromaDB."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("rexx_docs")

    # Get all documents
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for i, doc in enumerate(results['documents']):
        chunks.append({
            'content': doc,
            'metadata': results['metadatas'][i]
        })

    return chunks


def evaluate_retrieval(model: SentenceTransformer, chunks: List[Dict],
                       test_questions: List[Dict], top_k: int = 3) -> Dict:
    """Evaluate retrieval quality for a given embedding model."""

    # Embed all chunks
    print("  Embedding chunks...", flush=True)
    chunk_texts = [c['content'] for c in chunks]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)

    results = []

    for item in test_questions:
        question = item['question']
        expected_answer = item['expected_answer']

        # Embed question
        q_embedding = model.encode([question])[0]

        # Calculate similarities
        similarities = np.dot(chunk_embeddings, q_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        # Check if retrieved chunks contain relevant information
        retrieved_text = " ".join([chunks[i]['content'] for i in top_indices])

        # Simple relevance check: do retrieved docs contain key terms from expected answer?
        expected_words = set(expected_answer.lower().split())
        retrieved_words = set(retrieved_text.lower().split())

        # Key terms overlap (words > 4 chars for significance)
        key_expected = {w for w in expected_words if len(w) > 4}
        key_found = len(key_expected & retrieved_words)
        key_total = len(key_expected) if key_expected else 1

        relevance_score = key_found / key_total

        results.append({
            'question': question,
            'top_similarity': float(top_scores[0]),
            'avg_similarity': float(np.mean(top_scores)),
            'relevance_score': relevance_score,
            'key_terms_found': key_found,
            'key_terms_total': len(key_expected)
        })

    # Aggregate metrics
    return {
        'avg_top_similarity': round(np.mean([r['top_similarity'] for r in results]), 4),
        'avg_similarity': round(np.mean([r['avg_similarity'] for r in results]), 4),
        'avg_relevance': round(np.mean([r['relevance_score'] for r in results]), 4),
        'results': results
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")
    chroma_path = os.path.join(script_dir, "chroma_db")

    # Check prerequisites
    if not os.path.exists(chroma_path):
        print("ERROR: ChromaDB not found. Run 'python rag_system.py' first.", flush=True)
        return

    # Load chunks
    print("Loading chunks from ChromaDB...", flush=True)
    chunks = load_chunks_from_chroma(chroma_path)
    print(f"Loaded {len(chunks)} chunks", flush=True)

    # Load test questions
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_questions = data['test'][:15]  # Use 15 questions for quick comparison
    print(f"Testing with {len(test_questions)} questions", flush=True)

    # Test each embedding model
    all_results = {}

    for model_name in EMBEDDING_MODELS:
        print(f"\n{'='*60}", flush=True)
        print(f"Testing: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            # Load model
            start_load = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_load
            print(f"  Model loaded in {load_time:.1f}s (dim: {model.get_sentence_embedding_dimension()})", flush=True)

            # Evaluate
            start_eval = time.time()
            metrics = evaluate_retrieval(model, chunks, test_questions, top_k=3)
            eval_time = time.time() - start_eval

            metrics['load_time'] = round(load_time, 2)
            metrics['eval_time'] = round(eval_time, 2)
            metrics['embedding_dim'] = model.get_sentence_embedding_dimension()

            all_results[model_name] = metrics

            print(f"  Avg Top Similarity: {metrics['avg_top_similarity']:.4f}", flush=True)
            print(f"  Avg Relevance: {metrics['avg_relevance']:.4f}", flush=True)
            print(f"  Eval time: {eval_time:.1f}s", flush=True)

            # Free memory
            del model

        except Exception as e:
            print(f"  ERROR: {str(e)}", flush=True)
            all_results[model_name] = {"error": str(e)}

    # Save detailed results
    output_path = os.path.join(script_dir, "embedding_comparison.json")

    # Remove detailed results for JSON (too large)
    save_results = {}
    for name, data in all_results.items():
        if "error" in data:
            save_results[name] = data
        else:
            save_results[name] = {k: v for k, v in data.items() if k != 'results'}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)

    # Print comparison table
    print("\n" + "=" * 100, flush=True)
    print("EMBEDDING MODEL COMPARISON SUMMARY", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Model':<55} {'Dim':>5} {'Top Sim':>10} {'Relevance':>10} {'Time':>8}", flush=True)
    print("-" * 100, flush=True)

    # Sort by relevance score
    sorted_models = sorted(
        [(k, v) for k, v in all_results.items() if "error" not in v],
        key=lambda x: x[1]['avg_relevance'],
        reverse=True
    )

    for model_name, data in sorted_models:
        short_name = model_name.replace("sentence-transformers/", "")
        print(f"{short_name:<55} {data['embedding_dim']:>5} {data['avg_top_similarity']:>10.4f} {data['avg_relevance']:>10.4f} {data['eval_time']:>7.1f}s", flush=True)

    print("-" * 100, flush=True)

    # Recommendation
    if sorted_models:
        best = sorted_models[0]
        print(f"\nBest for retrieval: {best[0].replace('sentence-transformers/', '')}", flush=True)
        print(f"  Relevance: {best[1]['avg_relevance']:.4f}", flush=True)

        # Check for multilingual recommendation
        multilingual = [m for m in sorted_models if 'multilingual' in m[0].lower()]
        if multilingual:
            print(f"\nBest multilingual (recommended for German): {multilingual[0][0].replace('sentence-transformers/', '')}", flush=True)
            print(f"  Relevance: {multilingual[0][1]['avg_relevance']:.4f}", flush=True)

    # Save CSV
    csv_path = os.path.join(script_dir, "embedding_comparison.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Model,Dimension,Avg_Top_Similarity,Avg_Relevance,Load_Time,Eval_Time\n")
        for model_name, data in sorted_models:
            short_name = model_name.replace("sentence-transformers/", "")
            f.write(f"{short_name},{data['embedding_dim']},{data['avg_top_similarity']},{data['avg_relevance']},{data['load_time']},{data['eval_time']}\n")
    print(f"\nCSV saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

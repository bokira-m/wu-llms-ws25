"""
Flask Web App for Rexx HR RAG Q&A System
"""

import json
import os
import sys
import time
import requests
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_system import RAGSystem
from evaluate_baseline import BaselineEvaluator

app = Flask(__name__)

# Global RAG system instance
rag = None
evaluator = None

# Available models
AVAILABLE_GENERATORS = []
AVAILABLE_EMBEDDINGS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def get_ollama_models():
    """Fetch available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
    except:
        pass
    return ["llama2"]  # Default fallback


def init_rag(generator_model="llama2", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize RAG system."""
    global rag, evaluator, AVAILABLE_GENERATORS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(script_dir, "chroma_db")

    if not os.path.exists(chroma_path):
        print("ERROR: RAG index not found. Run 'python rag_system.py' first.")
        return False

    # Get available Ollama models
    AVAILABLE_GENERATORS = get_ollama_models()
    print(f"Available Ollama models: {AVAILABLE_GENERATORS}", flush=True)

    print(f"Initializing RAG system with {generator_model}...", flush=True)
    rag = RAGSystem(
        embedding_model=embedding_model,
        generator_model=generator_model,
        persist_directory=chroma_path
    )

    rag.load_embedding_model()
    rag.init_vector_store()
    rag.load_generator_model()

    # Initialize evaluator for metrics
    evaluator = BaselineEvaluator(model_name=f"rag-{generator_model}")

    print(f"RAG system ready! Index contains {rag.collection.count()} chunks.", flush=True)
    return True


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question and return answer with metrics."""
    global rag, evaluator

    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    data = request.json
    question = data.get('question', '').strip()
    expected_answer = data.get('expected_answer', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Query RAG
    start_time = time.time()
    answer, sources = rag.query(question, top_k=3)
    response_time = time.time() - start_time

    # Calculate metrics if expected answer provided
    metrics = None
    if expected_answer:
        metrics = evaluator.calculate_metrics(answer, expected_answer)

    # Format sources
    source_list = [
        {
            'filename': s['source'],
            'distance': round(s['distance'], 3),
            'relevance': round((1 - s['distance']) * 100, 1),
            'snippet': s['content'][:200] + '...' if len(s['content']) > 200 else s['content']
        }
        for s in sources
    ]

    return jsonify({
        'question': question,
        'answer': answer,
        'sources': source_list,
        'response_time': round(response_time, 2),
        'metrics': metrics,
        'expected_answer': expected_answer if expected_answer else None,
        'model_used': rag.generator_model_name
    })


@app.route('/test_questions')
def get_test_questions():
    """Return test questions from dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "rexx_qa_dataset_curated.json")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = [
        {
            'id': q['id'],
            'question': q['question'],
            'expected_answer': q['expected_answer'],
            'category': q.get('category', 'unknown'),
            'difficulty': q.get('difficulty', 'unknown')
        }
        for q in data['test']
    ]

    return jsonify(questions)


@app.route('/stats')
def get_stats():
    """Return system statistics."""
    global rag

    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    return jsonify({
        'embedding_model': rag.embedding_model_name,
        'generator_model': rag.generator_model_name,
        'index_size': rag.collection.count(),
        'status': 'ready'
    })


@app.route('/models')
def get_models():
    """Return available models."""
    global AVAILABLE_GENERATORS, AVAILABLE_EMBEDDINGS, rag

    # Refresh Ollama models list
    AVAILABLE_GENERATORS = get_ollama_models()

    current_generator = rag.generator_model_name if rag else "llama2"
    current_embedding = rag.embedding_model_name if rag else AVAILABLE_EMBEDDINGS[0]

    return jsonify({
        'generators': AVAILABLE_GENERATORS,
        'embeddings': AVAILABLE_EMBEDDINGS,
        'current_generator': current_generator,
        'current_embedding': current_embedding
    })


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch the generator model."""
    global rag, evaluator

    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    data = request.json
    new_generator = data.get('generator', '').strip()

    if not new_generator:
        return jsonify({'error': 'No model specified'}), 400

    # Check if model is available in Ollama
    available = get_ollama_models()
    if new_generator not in available:
        return jsonify({'error': f'Model {new_generator} not found in Ollama'}), 400

    # Switch the model
    old_model = rag.generator_model_name
    rag.generator_model_name = new_generator
    rag.load_generator_model()

    # Update evaluator
    evaluator = BaselineEvaluator(model_name=f"rag-{new_generator}")

    print(f"Switched generator from {old_model} to {new_generator}", flush=True)

    return jsonify({
        'success': True,
        'old_model': old_model,
        'new_model': new_generator,
        'message': f'Switched to {new_generator}'
    })


@app.route('/switch_embedding', methods=['POST'])
def switch_embedding():
    """Switch the embedding model (requires reloading)."""
    global rag

    if rag is None:
        return jsonify({'error': 'RAG system not initialized'}), 500

    data = request.json
    new_embedding = data.get('embedding', '').strip()

    if not new_embedding:
        return jsonify({'error': 'No embedding model specified'}), 400

    # Switch the embedding model
    old_model = rag.embedding_model_name
    rag.embedding_model_name = new_embedding
    rag.load_embedding_model()

    print(f"Switched embedding from {old_model} to {new_embedding}", flush=True)

    return jsonify({
        'success': True,
        'old_model': old_model,
        'new_model': new_embedding,
        'message': f'Switched embedding to {new_embedding}'
    })


if __name__ == '__main__':
    # Initialize RAG on startup
    if init_rag():
        print("\n" + "=" * 50)
        print("Starting Flask server...")
        print("Open http://localhost:5001 in your browser")
        print("=" * 50 + "\n")
        app.run(debug=False, host='0.0.0.0', port=5001)
    else:
        print("Failed to initialize RAG system. Exiting.")

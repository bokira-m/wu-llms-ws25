"""
RAG System for Rexx HR Q&A
Uses MiniLM-L6-v2 for retrieval and Ollama llama2 for generation
"""

import json
import os
import sys
import time
import requests
from typing import List, Dict, Tuple
import numpy as np

# PDF processing
import pdfplumber

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

OLLAMA_URL = "http://localhost:11434/api/generate"


class RAGSystem:
    """RAG system with MiniLM retriever and Ollama llama2 generator."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "llama2",
        collection_name: str = "rexx_docs",
        persist_directory: str = "./chroma_db"
    ):
        self.embedding_model_name = embedding_model
        self.generator_model_name = generator_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.embedding_model = None
        self.chroma_client = None
        self.collection = None

    def load_embedding_model(self):
        """Load the embedding model for retrieval."""
        print(f"Loading embedding model: {self.embedding_model_name}", flush=True)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("Embedding model loaded!", flush=True)

    def load_generator_model(self):
        """Check Ollama is available."""
        print(f"Using Ollama generator: {self.generator_model_name}", flush=True)
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama connection OK!", flush=True)
            else:
                print("WARNING: Ollama not responding properly", flush=True)
        except:
            print("WARNING: Could not connect to Ollama. Make sure it's running.", flush=True)

    def init_vector_store(self, reset: bool = False):
        """Initialize ChromaDB vector store."""
        print(f"Initializing vector store at: {self.persist_directory}", flush=True)

        if reset and os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("Removed existing vector store.", flush=True)

        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents", flush=True)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new collection", flush=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}", flush=True)
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def index_pdfs(self, pdf_directory: str, chunk_size: int = 500, overlap: int = 100):
        """Index all PDFs in directory."""
        if self.embedding_model is None:
            self.load_embedding_model()

        if self.collection is None:
            self.init_vector_store(reset=True)

        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        print(f"\nIndexing {len(pdf_files)} PDF files...", flush=True)

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"  [{i+1}/{len(pdf_files)}] Processing: {pdf_file[:40]}...", flush=True)

            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                continue

            # Chunk text
            chunks = self.chunk_text(text, chunk_size, overlap)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{pdf_file}_{j}")
                all_metadatas.append({
                    "source": pdf_file,
                    "chunk_index": j
                })

        print(f"\nTotal chunks: {len(all_chunks)}", flush=True)

        # Generate embeddings in batches
        print("Generating embeddings...", flush=True)
        batch_size = 32

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]

            embeddings = self.embedding_model.encode(batch_chunks).tolist()

            self.collection.add(
                documents=batch_chunks,
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_metadatas
            )

            if (i + batch_size) % 100 == 0:
                print(f"  Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks", flush=True)

        print(f"\nIndexing complete! Total documents: {self.collection.count()}", flush=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if self.embedding_model is None:
            self.load_embedding_model()

        if self.collection is None:
            self.init_vector_store()

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                "content": results['documents'][0][i],
                "source": results['metadatas'][0][i]['source'],
                "distance": results['distances'][0][i]
            })

        return retrieved

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 300) -> str:
        """Generate answer using Ollama llama2 with retrieved context."""
        # Truncate context to avoid overwhelming the model
        context_words = context.split()[:400]
        context_truncated = " ".join(context_words)

        prompt = f"""Du bist ein Experte für rexx HR Software. Beantworte die Frage basierend auf dem Kontext.

Kontext:
{context_truncated}

Frage: {question}

Antwort (auf Deutsch, präzise):"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.generator_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_new_tokens
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: Ollama returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Full RAG pipeline: retrieve and generate."""
        # Retrieve
        retrieved_docs = self.retrieve(question, top_k=top_k)

        # Build context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])

        # Generate
        answer = self.generate_answer(question, context)

        return answer, retrieved_docs


def main():
    """Build the RAG index."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(script_dir, "rexx_pdfs")

    # Initialize RAG system with Ollama llama2
    rag = RAGSystem(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generator_model="llama2",
        persist_directory=os.path.join(script_dir, "chroma_db")
    )

    # Index PDFs
    print("=" * 60)
    print("BUILDING RAG INDEX")
    print("=" * 60)

    rag.index_pdfs(pdf_dir, chunk_size=400, overlap=80)

    print("\n" + "=" * 60)
    print("RAG INDEX READY")
    print("=" * 60)

    # Test query
    print("\nTesting RAG with a sample question...")
    rag.load_generator_model()

    test_question = "Was ist der Jobportal-Konfigurator in rexx?"
    answer, sources = rag.query(test_question, top_k=3)

    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer: {answer}")
    print(f"\nSources used: {len(sources)}")
    for s in sources:
        print(f"  - {s['source'][:40]}... (distance: {s['distance']:.3f})")


if __name__ == "__main__":
    main()

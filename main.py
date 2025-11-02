#!/usr/bin/env python3
"""
Main script for RAG Document Provenance Recovery.

This script orchestrates the entire pipeline:
1. Data collection (or load existing)
2. Forward RAG system
3. Query generation with ground truth
4. Inverse provenance methods
5. Evaluation and result generation
"""

import os
import json
import argparse
from src.data_collection import DocumentCollector
from src.forward_rag import ForwardRAGSystem
from src.inverse_methods import InverseProvenanceMethods
from src.evaluation import EvaluationFramework
from config import Config


def main():
    """Main execution function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='RAG Document Provenance Recovery')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection, use existing data')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='Number of test queries to evaluate')
    args = parser.parse_args()

    # Validate configuration
    Config.validate()
    Config.create_directories()

    print("="*60)
    print("RAG DOCUMENT PROVENANCE RECOVERY")
    print("="*60)

    # ========================================================================
    # PHASE 1: DATA COLLECTION
    # ========================================================================
    print("\n[PHASE 1] DATA COLLECTION")
    print("-"*60)

    if args.skip_collection and os.path.exists('data/metadata.json'):
        print("Skipping data collection, loading existing documents...")

        # Load from metadata
        with open('data/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        documents = []
        for meta in metadata:
            with open(meta['file_path'], 'r', encoding='utf-8') as f:
                text = f.read()

            documents.append({
                'doc_id': meta['doc_id'],
                'title': meta['title'],
                'url': meta['url'],
                'source': meta['source'],
                'text': text
            })

        print(f"Loaded {len(documents)} existing documents")

    else:
        print("Collecting new documents...")

        collector = DocumentCollector()

        # Define queries and topics
        arxiv_queries = [
            "quantum machine learning",
            "quantum entanglement",
            "quantum computing algorithms"
        ]

        wikipedia_topics = [
            "Quantum mechanics",
            "Machine learning",
            "Quantum computing"
        ]

        documents = collector.collect_all_documents(arxiv_queries, wikipedia_topics)
        collector.save_documents(documents)

    print(f"\n[OK] {len(documents)} documents ready")

    # ========================================================================
    # PHASE 2: FORWARD RAG SYSTEM
    # ========================================================================
    print("\n[PHASE 2] FORWARD RAG SYSTEM")
    print("-"*60)

    rag = ForwardRAGSystem()

    # Check if vector store already exists
    if os.path.exists(Config.CHROMA_DB_DIR) and args.skip_collection:
        print("Attempting to load existing vector store...")
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            rag.vectorstore = client.get_collection("rag_documents")
            rag.chroma_client = client

            # Load chunks metadata (without embeddings)
            if os.path.exists("data/processed/chunks.json"):
                rag.load_chunks()
                print(f"[OK] Loaded existing vector store ({rag.vectorstore.count()} vectors)")
                print(f"[OK] Loaded {len(rag.chunks)} chunks metadata")
            else:
                raise FileNotFoundError("Chunks metadata not found, rebuilding...")

        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            print("Rebuilding from scratch...")
            args.skip_collection = False

    if not args.skip_collection or rag.vectorstore is None:
        print("Building new RAG system...")

        # Chunk documents
        print("  [1/3] Chunking documents...")
        chunks = rag.chunk_documents(documents)
        print(f"    Created {len(chunks)} chunks")

        # Generate embeddings
        print("  [2/3] Generating embeddings...")
        chunks = rag.generate_embeddings(chunks)
        print(f"    Generated embeddings ({len(chunks[0]['embedding'])} dims)")

        # Build vector store
        print("  [3/3] Building vector store...")
        vectorstore = rag.build_vectorstore(chunks)
        print(f"    Built vector store ({vectorstore.count()} vectors)")

        # Save chunks metadata
        rag.save_chunks()

    print(f"\n[OK] Forward RAG system ready")

    # ========================================================================
    # PHASE 3: QUERY GENERATION WITH GROUND TRUTH
    # ========================================================================
    print("\n[PHASE 3] QUERY GENERATION")
    print("-"*60)

    # Define test queries
    test_queries_raw = [
        "What is quantum entanglement?",
        "Explain quantum computing",
        "How does quantum machine learning work?",
        "What is superposition in quantum mechanics?",
        "Describe quantum algorithms",
        "What are qubits?",
        "Explain quantum teleportation",
        "How do quantum computers differ from classical computers?",
        "What is quantum decoherence?",
        "Describe quantum gates",
    ]

    # Limit to requested number
    test_queries_raw = test_queries_raw[:args.num_queries]

    print(f"Generating ground truth for {len(test_queries_raw)} queries...")

    queries_with_truth = []
    for i, query in enumerate(test_queries_raw):
        print(f"  [{i+1}/{len(test_queries_raw)}] {query[:50]}...")

        # Run forward RAG to establish ground truth
        result = rag.retrieve_and_generate(query)

        queries_with_truth.append({
            'query_id': f'q{i+1:03d}',
            'query': query,
            'true_chunk_ids': result['chunk_ids'],  # GROUND TRUTH
            'ground_truth_answer': result['answer']
        })

    print(f"\n[OK] {len(queries_with_truth)} queries with ground truth")

    # ========================================================================
    # PHASE 4: INVERSE PROVENANCE METHODS
    # ========================================================================
    print("\n[PHASE 4] INVERSE PROVENANCE METHODS")
    print("-"*60)

    inverse = InverseProvenanceMethods()

    print(f"[OK] Inverse methods initialized")

    # ========================================================================
    # PHASE 5: EVALUATION
    # ========================================================================
    print("\n[PHASE 5] EVALUATION")
    print("-"*60)

    evaluator = EvaluationFramework()

    print(f"Running evaluation on {len(queries_with_truth)} queries...")

    results = evaluator.run_full_evaluation(
        queries_with_truth,
        rag,
        inverse,
        rag.chunks
    )

    # ========================================================================
    # PHASE 6: SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    print(f"\nResults Summary:")
    print(f"  Documents collected: {len(documents)}")
    print(f"  Chunks created: {len(rag.chunks)}")
    print(f"  Test queries: {len(queries_with_truth)}")
    print(f"  Results directory: {evaluator.output_dir}")

    print(f"\nMethod Performance:")
    for method, metrics in results['method_results'].items():
        print(f"  {method:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    print(f"\nOutput Files:")
    print(f"  - {evaluator.output_dir}/evaluation_results.json")
    print(f"  - {evaluator.output_dir}/figures/method_comparison.png")
    print(f"  - logs/")

    print("\n" + "="*60)
    print("[SUCCESS] All phases completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

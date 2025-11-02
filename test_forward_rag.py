"""
Quick test of Forward RAG System.
"""

import json
from src.forward_rag import ForwardRAGSystem

def main():
    print("Testing Forward RAG System...")
    print("="*60)

    # Load documents from metadata
    with open('data/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load document texts
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

    print(f"Loaded {len(documents)} documents")

    # Initialize Forward RAG
    rag = ForwardRAGSystem()

    # Step 1: Chunk documents
    print("\n[1/4] Chunking documents...")
    chunks = rag.chunk_documents(documents)
    print(f"[OK] Created {len(chunks)} chunks")

    # Step 2: Generate embeddings
    print("\n[2/4] Generating embeddings...")
    chunks = rag.generate_embeddings(chunks)
    print(f"[OK] Generated embeddings ({len(chunks[0]['embedding'])} dims)")

    # Step 3: Build vector store
    print("\n[3/4] Building vector store...")
    vectorstore = rag.build_vectorstore(chunks)
    print(f"[OK] Vector store built ({vectorstore.count()} vectors)")

    # Step 4: Test retrieval
    print("\n[4/4] Testing retrieval and generation...")
    test_queries = [
        "What is quantum entanglement?",
        "Explain quantum computing",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        result = rag.retrieve_and_generate(query)
        print(f"  Answer: {result['answer'][:150]}...")
        print(f"  Sources: {len(result['chunk_ids'])} chunks")
        print(f"  Ground truth: {result['chunk_ids']}")

    print("\n[OK] Forward RAG System test successful!")

if __name__ == "__main__":
    main()

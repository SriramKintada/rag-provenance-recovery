"""
Quick test of data collection module.
"""

from src.data_collection import DocumentCollector

def main():
    print("Testing Document Collection...")
    print("="*60)

    collector = DocumentCollector()

    # Test with just 1 query each for speed
    arxiv_queries = ["quantum computing"]
    wikipedia_topics = ["Quantum mechanics"]

    # Collect documents
    documents = collector.collect_all_documents(
        arxiv_queries=arxiv_queries,
        wikipedia_topics=wikipedia_topics
    )

    # Save documents
    if len(documents) > 0:
        collector.save_documents(documents)
        print("\n[OK] Test successful!")
        print(f"Collected {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc['doc_id']}: {doc['title'][:60]}... ({len(doc['text'])} chars)")
    else:
        print("\n[WARNING] No documents collected")

if __name__ == "__main__":
    main()

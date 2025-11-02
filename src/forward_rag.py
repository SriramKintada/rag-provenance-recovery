"""
Forward RAG System Module for RAG Document Provenance Recovery.

This module implements the complete RAG pipeline:
1. Chunking documents
2. Generating embeddings
3. Building vector store
4. Retrieval and generation with ground truth tracking
"""

import os
import json
import logging
from typing import List, Dict, Optional

# Langchain imports
try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB for vector storage
import chromadb
from chromadb.config import Settings

# OpenAI for embeddings and LLM
from openai import OpenAI

# Token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

from config import Config
from src.utils import setup_logging


class ForwardRAGSystem:
    """
    Manages forward RAG pipeline: chunking → embedding → retrieval → generation.

    Attributes:
        chunks (list): All document chunks with embeddings
        vectorstore (chromadb.Collection): Persistent vector database
        llm_model (str): OpenAI model for generation
        embedding_model (str): OpenAI embedding model
        client (OpenAI): OpenAI client instance
        logger: Logging instance
    """

    def __init__(
        self,
        embedding_model: str = None,
        llm_model: str = None,
        persist_directory: str = None,
        log_file: str = "logs/forward_rag.log"
    ):
        """
        Initialize ForwardRAGSystem.

        Args:
            embedding_model: OpenAI embedding model name
            llm_model: OpenAI LLM model name
            persist_directory: Directory to persist ChromaDB
            log_file: Path to log file
        """
        # Use config defaults if not provided
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.llm_model = llm_model or Config.LLM_MODEL
        self.persist_directory = persist_directory or Config.CHROMA_DB_DIR
        self.log_file = log_file

        self.chunks = []
        self.vectorstore = None
        self.chroma_client = None

        # Setup logging
        self.logger = setup_logging("ForwardRAGSystem", log_file)

        # Create directories
        os.makedirs(self.persist_directory, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

        self.logger.info("ForwardRAGSystem initialized")
        self.logger.info(f"  Embedding model: {self.embedding_model}")
        self.logger.info(f"  LLM model: {self.llm_model}")
        self.logger.info(f"  Persist directory: {self.persist_directory}")

    def chunk_documents(
        self,
        documents: List[Dict],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.

        Args:
            documents: List of document dicts with 'text' field
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk dicts with chunk_id, text, document_id, etc.
        """
        chunk_size = chunk_size or Config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        self.logger.info("="*60)
        self.logger.info(f"CHUNKING {len(documents)} DOCUMENTS")
        self.logger.info(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        self.logger.info("="*60)

        # Initialize splitter with proven parameters (from TDD)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=Config.TEXT_SPLITTER_SEPARATORS,
            length_function=len
        )

        all_chunks = []

        for doc in documents:
            # Split document
            text_chunks = splitter.split_text(doc['text'])

            # Create chunk objects with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'chunk_id': f"{doc['doc_id']}_chunk_{i}",
                    'text': chunk_text,
                    'document_id': doc['doc_id'],
                    'document_title': doc['title'],
                    'chunk_index': i,
                    'source': doc['source']
                }

                all_chunks.append(chunk)

            self.logger.info(f"  {doc['doc_id']}: {len(text_chunks)} chunks")

        self.chunks = all_chunks

        # Validation: Check chunk sizes
        chunk_sizes = [len(c['text']) for c in all_chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)

        self.logger.info("\n" + "="*60)
        self.logger.info(f"CHUNKING COMPLETE: {len(all_chunks)} chunks")
        self.logger.info(f"  Size: min={min_size}, max={max_size}, avg={avg_size:.0f}")
        self.logger.info("="*60)

        # Assert reasonable chunk sizes
        if not (800 < avg_size < 1200):
            self.logger.warning(f"Average chunk size {avg_size} outside expected range 800-1200")

        return all_chunks

    def generate_embeddings(
        self,
        chunks: List[Dict] = None,
        batch_size: int = None
    ) -> List[Dict]:
        """
        Generate embeddings for chunks using OpenAI API.

        Args:
            chunks: List of chunk dicts (uses self.chunks if None)
            batch_size: Number of chunks to embed at once

        Returns:
            List of chunks with added 'embedding' field
        """
        if chunks is None:
            chunks = self.chunks

        batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE

        self.logger.info("="*60)
        self.logger.info(f"GENERATING EMBEDDINGS FOR {len(chunks)} CHUNKS")
        self.logger.info(f"  Model: {self.embedding_model}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info("="*60)

        total_tokens = 0

        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [c['text'] for c in batch]

            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            self.logger.info(f"Processing batch {batch_num}/{total_batches}")

            try:
                # Actual OpenAI API call
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                # Add to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk['embedding'] = embedding

                # Track token usage
                total_tokens += response.usage.total_tokens

                self.logger.info(
                    f"  Batch completed: {len(embeddings)} embeddings, "
                    f"{response.usage.total_tokens} tokens"
                )

            except Exception as e:
                self.logger.error(f"  Embedding batch failed: {e}")
                raise

        # Validation: Check embedding dimensions
        first_embedding = chunks[0]['embedding']
        embedding_dim = len(first_embedding)

        self.logger.info("\n" + "="*60)
        self.logger.info(f"EMBEDDINGS COMPLETE")
        self.logger.info(f"  Dimensions: {embedding_dim}")
        self.logger.info(f"  Total tokens: {total_tokens}")
        self.logger.info(f"  Est. cost: ${total_tokens * 0.00002:.4f}")
        self.logger.info("="*60)

        # Assert correct dimensions
        expected_dim = Config.EMBEDDING_DIMENSIONS
        assert embedding_dim == expected_dim, \
            f"Wrong dimensions: {embedding_dim} != {expected_dim}"

        # Assert not all zeros
        assert sum(first_embedding) != 0, "Embedding is all zeros!"

        self.logger.info("[OK] Embedding validation passed")

        return chunks

    def build_vectorstore(
        self,
        chunks: List[Dict] = None,
        collection_name: str = "rag_documents"
    ) -> chromadb.Collection:
        """
        Build ChromaDB vector store.

        Args:
            chunks: List of chunk dicts with embeddings
            collection_name: Name of collection

        Returns:
            ChromaDB collection object
        """
        if chunks is None:
            chunks = self.chunks

        self.logger.info("="*60)
        self.logger.info(f"BUILDING VECTOR STORE")
        self.logger.info(f"  Collection: {collection_name}")
        self.logger.info(f"  Chunks: {len(chunks)}")
        self.logger.info("="*60)

        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        try:
            # Try to delete existing collection
            try:
                self.chroma_client.delete_collection(collection_name)
                self.logger.info(f"  Deleted existing collection: {collection_name}")
            except:
                pass

            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"  Created new collection: {collection_name}")

        except Exception as e:
            self.logger.error(f"  Failed to create collection: {e}")
            raise

        # Add chunks to collection
        ids = [c['chunk_id'] for c in chunks]
        embeddings = [c['embedding'] for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [
            {
                'document_id': c['document_id'],
                'document_title': c['document_title'],
                'chunk_index': c['chunk_index'],
                'source': c['source']
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        self.vectorstore = collection

        # Verification
        count = collection.count()
        self.logger.info("\n" + "="*60)
        self.logger.info(f"VECTOR STORE BUILT")
        self.logger.info(f"  Collection size: {count}")
        self.logger.info(f"  Persisted to: {self.persist_directory}")
        self.logger.info("="*60)

        # Assert collection not empty
        assert count > 0, "Collection is empty!"
        assert count == len(chunks), f"Collection size mismatch: {count} != {len(chunks)}"

        self.logger.info("[OK] Vector store validation passed")

        return collection

    def retrieve_and_generate(
        self,
        query: str,
        k: int = None,
        temperature: float = 0
    ) -> Dict:
        """
        Retrieve relevant chunks and generate answer.

        CRITICAL: Tracks ground truth sources for evaluation.

        Args:
            query: User query string
            k: Number of chunks to retrieve
            temperature: LLM temperature

        Returns:
            Dict with query, answer, retrieved_chunks, chunk_ids (ground truth), scores
        """
        k = k or Config.RETRIEVAL_K

        self.logger.info("\n" + "="*60)
        self.logger.info(f"QUERY: {query}")
        self.logger.info("="*60)

        # 1. Embed query
        self.logger.info("[1/3] Embedding query...")
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = response.data[0].embedding

        # 2. Retrieve similar chunks
        self.logger.info(f"[2/3] Retrieving top-{k} chunks...")
        results = self.vectorstore.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Extract results
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            retrieved_chunks.append(chunk)

            self.logger.info(
                f"  [{i+1}] {chunk['chunk_id']} (dist: {chunk['distance']:.3f})"
            )

        # 3. Generate answer
        self.logger.info("[3/3] Generating answer...")

        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Count tokens
        if tiktoken:
            encoding = tiktoken.encoding_for_model(self.llm_model)
            context_tokens = len(encoding.encode(context))
            query_tokens = len(encoding.encode(query))
            total_input_tokens = context_tokens + query_tokens + 100

            self.logger.info(
                f"  Token count: {total_input_tokens} "
                f"(context: {context_tokens}, query: {query_tokens})"
            )

            # Check token limit
            max_tokens = Config.get_token_limit(self.llm_model)
            if total_input_tokens > max_tokens - 500:
                self.logger.warning(
                    f"  HIGH TOKEN USAGE: {total_input_tokens}/{max_tokens}"
                )

        # Create prompt
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            usage = response.usage

            self.logger.info(f"  Generated: {len(answer)} chars")
            self.logger.info(
                f"  Tokens used: {usage.total_tokens} "
                f"(prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})"
            )

        except Exception as e:
            self.logger.error(f"  Generation failed: {e}")
            raise

        # Prepare result with ground truth
        result = {
            'query': query,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'chunk_ids': [c['chunk_id'] for c in retrieved_chunks],  # GROUND TRUTH
            'scores': [1 - c['distance'] for c in retrieved_chunks],  # Convert to similarity
            'token_usage': {
                'total': usage.total_tokens,
                'prompt': usage.prompt_tokens,
                'completion': usage.completion_tokens
            }
        }

        self.logger.info("\n" + "="*60)
        self.logger.info("RETRIEVAL + GENERATION COMPLETE")
        self.logger.info(f"  Answer preview: {answer[:100]}...")
        self.logger.info(f"  Ground truth sources: {len(result['chunk_ids'])}")
        self.logger.info("="*60)

        return result

    def save_chunks(self, filepath: str = "data/processed/chunks.json"):
        """Save chunks to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Remove embeddings for serialization (too large)
        chunks_without_embeddings = [
            {k: v for k, v in c.items() if k != 'embedding'}
            for c in self.chunks
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_without_embeddings, f, indent=2)

        self.logger.info(f"Chunks saved to {filepath}")

    def load_chunks(self, filepath: str = "data/processed/chunks.json"):
        """Load chunks from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        self.logger.info(f"Loaded {len(self.chunks)} chunks from {filepath}")
        return self.chunks

"""
Inverse Provenance Methods Module for RAG Document Provenance Recovery.

Implements 4 methods to recover source documents from generated answers:
1. Embedding Similarity - Find chunks nearest to answer embedding
2. TF-IDF Matching - Keyword overlap analysis
3. N-gram Overlap - Exact phrase matching
4. LLM Attribution - Ask LLM to identify sources
"""

import os
import json
import logging
from typing import List, Dict

# OpenAI for embeddings and LLM
from openai import OpenAI

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# N-grams
try:
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    print("Warning: NLTK not available for n-gram method")
    ngrams = None

from config import Config
from src.utils import setup_logging


class InverseProvenanceMethods:
    """
    Implements 4 inverse provenance recovery methods.

    Methods:
        1. embedding_similarity - Semantic similarity via embeddings
        2. tfidf_matching - Lexical similarity via TF-IDF
        3. ngram_overlap - Exact phrase matching via n-grams
        4. llm_attribution - LLM-based source identification
    """

    def __init__(self, log_file: str = "logs/inverse.log"):
        """
        Initialize InverseProvenanceMethods.

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file

        # Setup logging
        self.logger = setup_logging("InverseProvenance", log_file)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

        self.logger.info("InverseProvenanceMethods initialized")

    def embedding_similarity(
        self,
        answer: str,
        vectorstore,
        embedding_model: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Method 1: Find chunks most similar to answer using embeddings.

        Args:
            answer: Generated answer text
            vectorstore: ChromaDB collection
            embedding_model: OpenAI embedding model name
            k: Number of chunks to recover

        Returns:
            List of dicts with chunk_id, text, similarity_score, method
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("METHOD 1: EMBEDDING SIMILARITY")
        self.logger.info(f"  Answer length: {len(answer)} chars")
        self.logger.info(f"  k: {k}")
        self.logger.info("="*60)

        # Embed answer
        response = self.client.embeddings.create(
            model=embedding_model,
            input=[answer]
        )
        answer_embedding = response.data[0].embedding

        self.logger.info("  Answer embedded")

        # Query vector store
        results = vectorstore.query(
            query_embeddings=[answer_embedding],
            n_results=k
        )

        # Format results
        recovered = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'method': 'embedding'
            }
            recovered.append(chunk)

            self.logger.info(f"  [{i+1}] {chunk['chunk_id']}: {chunk['similarity_score']:.3f}")

        # Validation
        assert len(recovered) == k, f"Expected {k} results, got {len(recovered)}"
        assert all(0 <= r['similarity_score'] <= 1 for r in recovered), "Invalid similarity scores"
        assert recovered[0]['similarity_score'] >= recovered[-1]['similarity_score'], "Not sorted"

        self.logger.info(f"[OK] Recovered {len(recovered)} chunks")

        return recovered

    def tfidf_matching(
        self,
        answer: str,
        all_chunks: List[Dict],
        k: int = 5
    ) -> List[Dict]:
        """
        Method 2: TF-IDF similarity matching.

        Args:
            answer: Generated answer text
            all_chunks: List of all chunk dicts
            k: Number of chunks to recover

        Returns:
            List of dicts with chunk_id, text, tfidf_score, method
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("METHOD 2: TF-IDF MATCHING")
        self.logger.info(f"  Corpus size: {len(all_chunks)} chunks")
        self.logger.info("="*60)

        # Prepare corpus
        corpus = [c['text'] for c in all_chunks] + [answer]

        # Vectorize
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Answer is last vector
        answer_vec = tfidf_matrix[-1]
        chunk_vecs = tfidf_matrix[:-1]

        # Compute similarities
        similarities = cosine_similarity(answer_vec, chunk_vecs)[0]

        # Get top-k
        top_k_indices = similarities.argsort()[-k:][::-1]

        # Format results
        recovered = []
        for idx in top_k_indices:
            chunk = all_chunks[idx]
            recovered.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'tfidf_score': float(similarities[idx]),
                'method': 'tfidf'
            })

            self.logger.info(f"  [{len(recovered)}] {chunk['chunk_id']}: {similarities[idx]:.3f}")

        # Validation
        assert len(recovered) == k
        assert all(r['tfidf_score'] >= 0 for r in recovered), "Negative TF-IDF scores"

        self.logger.info(f"[OK] Recovered {len(recovered)} chunks")

        return recovered

    def ngram_overlap(
        self,
        answer: str,
        all_chunks: List[Dict],
        n: int = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Method 3: N-gram Jaccard similarity.

        Args:
            answer: Generated answer text
            all_chunks: List of all chunk dicts
            n: N-gram size (default from config)
            k: Number of chunks to recover

        Returns:
            List of dicts with chunk_id, text, jaccard_score, method
        """
        if ngrams is None:
            self.logger.error("NLTK not available for n-gram method")
            return []

        n = n or Config.NGRAM_SIZE

        self.logger.info("\n" + "="*60)
        self.logger.info("METHOD 3: N-GRAM OVERLAP")
        self.logger.info(f"  N-gram size: {n}")
        self.logger.info("="*60)

        # Generate answer n-grams
        answer_tokens = word_tokenize(answer.lower())
        answer_ngrams = set(ngrams(answer_tokens, n))

        self.logger.info(f"  Answer n-grams: {len(answer_ngrams)}")

        # Compute Jaccard for each chunk
        scores = []
        for chunk in all_chunks:
            chunk_tokens = word_tokenize(chunk['text'].lower())
            chunk_ngrams = set(ngrams(chunk_tokens, n))

            # Jaccard similarity
            intersection = len(answer_ngrams & chunk_ngrams)
            union = len(answer_ngrams | chunk_ngrams)
            jaccard = intersection / union if union > 0 else 0

            scores.append((chunk, jaccard))

        # Sort and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:k]

        # Format results
        recovered = []
        for chunk, score in top_k:
            recovered.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'jaccard_score': float(score),
                'method': 'ngram'
            })

            self.logger.info(f"  [{len(recovered)}] {chunk['chunk_id']}: {score:.3f}")

        # Validation
        assert len(recovered) == k
        assert all(0 <= r['jaccard_score'] <= 1 for r in recovered), "Invalid Jaccard scores"

        self.logger.info(f"[OK] Recovered {len(recovered)} chunks")

        return recovered

    def llm_attribution(
        self,
        answer: str,
        candidate_chunks: List[Dict],
        llm_model: str = None
    ) -> List[Dict]:
        """
        Method 4: LLM-based attribution.

        Args:
            answer: Generated answer text
            candidate_chunks: List of candidate chunk dicts (pre-filtered)
            llm_model: OpenAI model name (default from config)

        Returns:
            List of dicts with chunk_id, text, confidence, method
        """
        llm_model = llm_model or Config.LLM_MODEL

        self.logger.info("\n" + "="*60)
        self.logger.info("METHOD 4: LLM ATTRIBUTION")
        self.logger.info(f"  Candidates: {len(candidate_chunks)}")
        self.logger.info("="*60)

        # Prepare prompt
        chunks_text = ""
        for i, chunk in enumerate(candidate_chunks):
            # Truncate chunk text for prompt
            chunk_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            chunks_text += f"[{i}] {chunk['chunk_id']}: {chunk_preview}\n\n"

        prompt = f"""Given this answer:
"{answer}"

And these candidate source chunks:
{chunks_text}

Which chunks (by index) provide evidence for statements in the answer?
Respond ONLY with a JSON object in this exact format:
{{"supporting_chunks": [list of indices], "confidence": [list of confidence scores 0-1]}}

Do not include any other text."""

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            llm_response = response.choices[0].message.content
            self.logger.info(f"  LLM response: {llm_response[:100]}...")

            # Parse JSON (handle potential markdown wrapping)
            llm_response = llm_response.strip()
            if llm_response.startswith('```'):
                llm_response = llm_response.split('```')[1]
                if llm_response.startswith('json'):
                    llm_response = llm_response[4:]
                llm_response = llm_response.strip()

            attribution = json.loads(llm_response)

            # Format results
            recovered = []
            for idx, conf in zip(attribution['supporting_chunks'], attribution['confidence']):
                if 0 <= idx < len(candidate_chunks):
                    chunk = candidate_chunks[idx]
                    recovered.append({
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'confidence': float(conf),
                        'method': 'llm'
                    })

                    self.logger.info(f"  [{len(recovered)}] {chunk['chunk_id']}: {conf:.2f}")

            # Sort by confidence
            recovered.sort(key=lambda x: x['confidence'], reverse=True)

            self.logger.info(f"[OK] Recovered {len(recovered)} chunks")

            return recovered

        except json.JSONDecodeError as e:
            self.logger.error(f"  Failed to parse LLM response: {e}")
            self.logger.error(f"  Response was: {llm_response}")
            return []
        except Exception as e:
            self.logger.error(f"  LLM call failed: {e}")
            return []

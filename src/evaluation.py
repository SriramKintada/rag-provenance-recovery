"""
Evaluation Framework Module for RAG Document Provenance Recovery.

Provides comprehensive evaluation of inverse provenance methods:
- Precision, Recall, F1 computation
- Method comparison plots
- Results persistence
"""

import os
import json
import logging
from typing import List, Dict, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from src.utils import setup_logging


class EvaluationFramework:
    """
    Comprehensive evaluation of inverse provenance methods.

    Features:
        - Precision, recall, F1 computation
        - Comparison plots (actual matplotlib figures)
        - Results persistence to JSON
    """

    def __init__(
        self,
        output_dir: str = None,
        log_file: str = "logs/evaluation.log"
    ):
        """
        Initialize EvaluationFramework.

        Args:
            output_dir: Directory to save results
            log_file: Path to log file
        """
        self.output_dir = output_dir or Config.RESULTS_DIR
        self.log_file = log_file

        # Create directories
        os.makedirs(f"{self.output_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.output_dir}/tables", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Setup logging
        self.logger = setup_logging("Evaluation", log_file)

        self.logger.info("EvaluationFramework initialized")
        self.logger.info(f"  Output directory: {self.output_dir}")

    def compute_metrics(
        self,
        true_chunk_ids: List[str],
        predicted_chunk_ids: List[str]
    ) -> Dict:
        """
        Compute precision, recall, F1.

        Args:
            true_chunk_ids: Ground truth chunk IDs
            predicted_chunk_ids: Predicted chunk IDs

        Returns:
            Dict with precision, recall, f1, true_positives, false_positives, false_negatives
        """
        true_set = set(true_chunk_ids)
        pred_set = set(predicted_chunk_ids)

        # True positives, false positives, false negatives
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

        self.logger.debug(
            f"  Metrics: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} "
            f"(TP={tp}, FP={fp}, FN={fn})"
        )

        return metrics

    def generate_comparison_plot(
        self,
        results: Dict,
        output_path: str = None
    ) -> str:
        """
        Generate bar chart comparing methods.

        Args:
            results: Dict of method -> metrics
            output_path: Path to save plot

        Returns:
            Path to saved plot file
        """
        if output_path is None:
            output_path = f"{self.output_dir}/figures/method_comparison.png"

        self.logger.info(f"Generating comparison plot: {output_path}")

        # Extract data
        methods = list(results.keys())
        precision = [results[m]['precision'] for m in methods]
        recall = [results[m]['recall'] for m in methods]
        f1 = [results[m]['f1'] for m in methods]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(methods))
        width = 0.25

        # Plot bars
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        bars3 = ax.bar(x + width, f1, width, label='F1', color='#2ecc71')

        # Customize
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Inverse Provenance Method Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()

        # Save to file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Verification
        assert os.path.exists(output_path), f"Plot file not created: {output_path}"
        file_size = os.path.getsize(output_path)
        assert file_size > 1000, f"Plot file suspiciously small: {file_size} bytes"

        self.logger.info(f"[OK] Plot saved: {output_path} ({file_size / 1024:.1f} KB)")

        return output_path

    def run_full_evaluation(
        self,
        queries: List[Dict],
        forward_rag_system,
        inverse_methods,
        all_chunks: List[Dict]
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            queries: List of query dicts with query, true_chunk_ids
            forward_rag_system: ForwardRAGSystem instance
            inverse_methods: InverseProvenanceMethods instance
            all_chunks: List of all chunk dicts

        Returns:
            Dict with method_results and per_query_results
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING FULL EVALUATION")
        self.logger.info(f"  Queries: {len(queries)}")
        self.logger.info("="*60)

        per_query_results = []
        method_aggregates = {
            'embedding': {'precision': [], 'recall': [], 'f1': []},
            'tfidf': {'precision': [], 'recall': [], 'f1': []},
            'ngram': {'precision': [], 'recall': [], 'f1': []},
            'llm': {'precision': [], 'recall': [], 'f1': []}
        }

        # Process each query
        for i, query_data in enumerate(queries):
            query = query_data['query']
            true_chunk_ids = query_data['true_chunk_ids']

            self.logger.info(f"\n[{i+1}/{len(queries)}] Query: {query[:60]}...")
            self.logger.info(f"  Ground truth: {len(true_chunk_ids)} chunks")

            try:
                # Get forward RAG result (to get answer)
                forward_result = forward_rag_system.retrieve_and_generate(query)
                answer = forward_result['answer']

                # Run all 4 inverse methods
                methods_results = {}

                # Method 1: Embedding
                recovered_emb = inverse_methods.embedding_similarity(
                    answer,
                    forward_rag_system.vectorstore,
                    forward_rag_system.embedding_model,
                    k=len(true_chunk_ids)
                )
                pred_emb = [r['chunk_id'] for r in recovered_emb]
                metrics_emb = self.compute_metrics(true_chunk_ids, pred_emb)
                methods_results['embedding'] = metrics_emb
                method_aggregates['embedding']['precision'].append(metrics_emb['precision'])
                method_aggregates['embedding']['recall'].append(metrics_emb['recall'])
                method_aggregates['embedding']['f1'].append(metrics_emb['f1'])

                # Method 2: TF-IDF
                recovered_tfidf = inverse_methods.tfidf_matching(
                    answer,
                    all_chunks,
                    k=len(true_chunk_ids)
                )
                pred_tfidf = [r['chunk_id'] for r in recovered_tfidf]
                metrics_tfidf = self.compute_metrics(true_chunk_ids, pred_tfidf)
                methods_results['tfidf'] = metrics_tfidf
                method_aggregates['tfidf']['precision'].append(metrics_tfidf['precision'])
                method_aggregates['tfidf']['recall'].append(metrics_tfidf['recall'])
                method_aggregates['tfidf']['f1'].append(metrics_tfidf['f1'])

                # Method 3: N-gram
                recovered_ngram = inverse_methods.ngram_overlap(
                    answer,
                    all_chunks,
                    n=3,
                    k=len(true_chunk_ids)
                )
                pred_ngram = [r['chunk_id'] for r in recovered_ngram]
                metrics_ngram = self.compute_metrics(true_chunk_ids, pred_ngram)
                methods_results['ngram'] = metrics_ngram
                method_aggregates['ngram']['precision'].append(metrics_ngram['precision'])
                method_aggregates['ngram']['recall'].append(metrics_ngram['recall'])
                method_aggregates['ngram']['f1'].append(metrics_ngram['f1'])

                # Method 4: LLM (use top-10 from embedding as candidates)
                candidates = recovered_emb[:min(10, len(recovered_emb))]
                recovered_llm = inverse_methods.llm_attribution(answer, candidates)
                pred_llm = [r['chunk_id'] for r in recovered_llm]
                metrics_llm = self.compute_metrics(true_chunk_ids, pred_llm)
                methods_results['llm'] = metrics_llm
                method_aggregates['llm']['precision'].append(metrics_llm['precision'])
                method_aggregates['llm']['recall'].append(metrics_llm['recall'])
                method_aggregates['llm']['f1'].append(metrics_llm['f1'])

                # Store per-query result
                per_query_results.append({
                    'query': query,
                    'true_chunk_ids': true_chunk_ids,
                    'answer': answer,
                    'methods': methods_results
                })

                # Log query results
                self.logger.info(
                    f"  Embedding: P={metrics_emb['precision']:.3f}, "
                    f"R={metrics_emb['recall']:.3f}, F1={metrics_emb['f1']:.3f}"
                )
                self.logger.info(
                    f"  TF-IDF:    P={metrics_tfidf['precision']:.3f}, "
                    f"R={metrics_tfidf['recall']:.3f}, F1={metrics_tfidf['f1']:.3f}"
                )
                self.logger.info(
                    f"  N-gram:    P={metrics_ngram['precision']:.3f}, "
                    f"R={metrics_ngram['recall']:.3f}, F1={metrics_ngram['f1']:.3f}"
                )
                self.logger.info(
                    f"  LLM:       P={metrics_llm['precision']:.3f}, "
                    f"R={metrics_llm['recall']:.3f}, F1={metrics_llm['f1']:.3f}"
                )

            except Exception as e:
                self.logger.error(f"  Query failed: {e}")

        # Compute averages
        method_results = {}
        for method, aggregates in method_aggregates.items():
            method_results[method] = {
                'precision': sum(aggregates['precision']) / len(aggregates['precision']) if aggregates['precision'] else 0,
                'recall': sum(aggregates['recall']) / len(aggregates['recall']) if aggregates['recall'] else 0,
                'f1': sum(aggregates['f1']) / len(aggregates['f1']) if aggregates['f1'] else 0,
                'n_queries': len(aggregates['f1'])
            }

        # Final results
        results = {
            'method_results': method_results,
            'per_query_results': per_query_results
        }

        # Save to JSON
        results_file = f"{self.output_dir}/evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATION COMPLETE")
        self.logger.info("="*60)
        self.logger.info("\nAVERAGE RESULTS:")
        for method, metrics in method_results.items():
            self.logger.info(
                f"  {method:12s}: P={metrics['precision']:.3f}, "
                f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            )

        # Generate plots
        self.generate_comparison_plot(
            {m: {'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1']}
             for m, v in method_results.items()}
        )

        self.logger.info(f"\n[OK] Results saved: {results_file}")
        self.logger.info("="*60)

        return results

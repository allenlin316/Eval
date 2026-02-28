"""
Retrieval evaluation metrics.

This module provides traditional IR metrics for evaluating retrieval systems:
- Recall@k: Proportion of relevant documents retrieved in top-k
- nDCG@k: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank

These metrics do NOT require an LLM - they use ground truth relevance judgments.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Result of retrieval evaluation for a single query."""
    query_id: str
    recall_at_k: Dict[int, float]  # k -> recall score
    ndcg_at_k: Dict[int, float]    # k -> nDCG score
    mrr: float                      # Mean Reciprocal Rank
    num_relevant: int               # Number of relevant documents in qrels
    num_retrieved: int              # Number of retrieved documents
    hits_at_k: Dict[int, int]       # k -> number of relevant docs in top-k


@dataclass 
class AggregatedRetrievalMetrics:
    """Aggregated retrieval metrics across all queries."""
    mean_recall_at_k: Dict[int, float]
    mean_ndcg_at_k: Dict[int, float]
    mean_mrr: float
    num_queries: int
    per_query_results: List[RetrievalResult]


class RetrievalMetrics:
    """
    Calculator for retrieval evaluation metrics.
    
    Usage:
        qrels = load_qrels("qrels.tsv")  # {query_id: {doc_id: relevance}}
        results = load_results("results.jsonl")  # {query_id: [doc_id1, doc_id2, ...]}
        
        calculator = RetrievalMetrics(k_values=[1, 3, 5, 10])
        metrics = calculator.evaluate(qrels, results)
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize the retrieval metrics calculator.
        
        Args:
            k_values: List of k values to compute metrics at (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[str]],
    ) -> AggregatedRetrievalMetrics:
        """
        Evaluate retrieval results against relevance judgments.
        
        Args:
            qrels: Relevance judgments {query_id: {doc_id: relevance_score}}
            results: Retrieved documents {query_id: [doc_id1, doc_id2, ...]} (ranked order)
            
        Returns:
            AggregatedRetrievalMetrics with mean metrics across all queries
        """
        per_query_results = []
        
        for query_id in qrels.keys():
            retrieved_docs = results.get(query_id, [])
            relevant_docs = qrels.get(query_id, {})
            
            result = self._evaluate_single_query(
                query_id=query_id,
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
            )
            per_query_results.append(result)
        
        # Aggregate metrics
        return self._aggregate_results(per_query_results)
    
    def _evaluate_single_query(
        self,
        query_id: str,
        retrieved_docs: List[str],
        relevant_docs: Dict[str, int],
    ) -> RetrievalResult:
        """Evaluate metrics for a single query."""
        
        recall_at_k = {}
        ndcg_at_k = {}
        hits_at_k = {}
        
        num_relevant = len(relevant_docs)
        num_retrieved = len(retrieved_docs)
        
        # Calculate Recall@k for each k
        for k in self.k_values:
            top_k = retrieved_docs[:k]
            hits = sum(1 for doc in top_k if doc in relevant_docs)
            hits_at_k[k] = hits
            recall_at_k[k] = hits / num_relevant if num_relevant > 0 else 0.0
        
        # Calculate nDCG@k for each k
        for k in self.k_values:
            ndcg_at_k[k] = self._calculate_ndcg(
                retrieved_docs[:k],
                relevant_docs,
                k
            )
        
        # Calculate MRR
        mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
        
        return RetrievalResult(
            query_id=query_id,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            num_relevant=num_relevant,
            num_retrieved=num_retrieved,
            hits_at_k=hits_at_k,
        )
    
    def _calculate_ndcg(
        self,
        retrieved_docs: List[str],
        relevant_docs: Dict[str, int],
        k: int,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        nDCG@k = DCG@k / IDCG@k
        DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
        """
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            rel = relevant_docs.get(doc, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate IDCG (ideal DCG with best possible ranking)
        ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_mrr(
        self,
        retrieved_docs: List[str],
        relevant_docs: Dict[str, int],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank_of_first_relevant_doc
        """
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def _aggregate_results(
        self,
        results: List[RetrievalResult],
    ) -> AggregatedRetrievalMetrics:
        """Aggregate per-query results into mean metrics."""
        
        if not results:
            return AggregatedRetrievalMetrics(
                mean_recall_at_k={k: 0.0 for k in self.k_values},
                mean_ndcg_at_k={k: 0.0 for k in self.k_values},
                mean_mrr=0.0,
                num_queries=0,
                per_query_results=[],
            )
        
        num_queries = len(results)
        
        # Aggregate Recall@k
        mean_recall_at_k = {}
        for k in self.k_values:
            mean_recall_at_k[k] = sum(r.recall_at_k[k] for r in results) / num_queries
        
        # Aggregate nDCG@k
        mean_ndcg_at_k = {}
        for k in self.k_values:
            mean_ndcg_at_k[k] = sum(r.ndcg_at_k[k] for r in results) / num_queries
        
        # Aggregate MRR
        mean_mrr = sum(r.mrr for r in results) / num_queries
        
        return AggregatedRetrievalMetrics(
            mean_recall_at_k=mean_recall_at_k,
            mean_ndcg_at_k=mean_ndcg_at_k,
            mean_mrr=mean_mrr,
            num_queries=num_queries,
            per_query_results=results,
        )


def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from a TSV file.
    
    Expected format (tab-separated):
        query-id    corpus-id    score
        q1          doc1         1
        q1          doc2         1
        q2          doc3         1
    
    Args:
        file_path: Path to the qrels TSV file
        
    Returns:
        Dictionary {query_id: {doc_id: relevance_score}}
    """
    qrels = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header if present
        first_line = f.readline().strip()
        if not first_line.startswith('query-id'):
            # Not a header, process this line
            parts = first_line.split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        
        # Process remaining lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
    
    return qrels


def extract_retrieved_docs_from_rag_data(
    rag_data: List[Dict],
    top_k: int = None,
) -> Dict[str, List[str]]:
    """
    Extract retrieved document IDs from RAG JSONL data.
    
    Args:
        rag_data: List of RAG task dictionaries with 'task_id' and 'contexts'
        top_k: Optional limit on number of documents per query
        
    Returns:
        Dictionary {task_id: [doc_id1, doc_id2, ...]} (ordered by score)
    """
    results = {}
    
    for task in rag_data:
        task_id = task.get('task_id', '')
        contexts = task.get('contexts', [])
        
        # Sort by score (descending) and extract document IDs
        sorted_contexts = sorted(
            contexts,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        doc_ids = [ctx.get('document_id', '') for ctx in sorted_contexts if ctx.get('document_id')]
        
        if top_k:
            doc_ids = doc_ids[:top_k]
        
        results[task_id] = doc_ids
    
    return results

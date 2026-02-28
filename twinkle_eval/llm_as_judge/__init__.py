"""
LLM-as-Judge module for evaluating LLM outputs using LLMs as evaluators.

This module provides a flexible framework for implementing various evaluation metrics
that use language models to assess the quality of LLM-generated content.
"""

from .metrics.base_metric import BaseMetric
from .metrics.g_eval import GEval
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.answer_relevancy import AnswerRelevancyMetric
from .metrics.accuracy import AccuracyMetric
from .metrics.retrieval import (
    RetrievalMetrics,
    RetrievalResult,
    AggregatedRetrievalMetrics,
    load_qrels,
    extract_retrieved_docs_from_rag_data,
)
from .models.base_judge import BaseJudgeModel
from .models.openai_judge import OpenAIJudgeModel
from .models.google_judge import GoogleJudgeModel
from .schemas import TestCase, MetricResult, Rubric
from .config_runner import LLMAsJudgeConfigRunner, run_from_config
from .rag_runner import RAGEvaluationRunner, run_rag_evaluation

__all__ = [
    # Metrics
    "BaseMetric",
    "GEval",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "AccuracyMetric",
    # Retrieval metrics
    "RetrievalMetrics",
    "RetrievalResult",
    "AggregatedRetrievalMetrics",
    "load_qrels",
    "extract_retrieved_docs_from_rag_data",
    # Models
    "BaseJudgeModel",
    "OpenAIJudgeModel",
    "GoogleJudgeModel",
    # Schemas
    "TestCase",
    "MetricResult",
    "Rubric",
    # Runners
    "LLMAsJudgeConfigRunner",
    "run_from_config",
    "RAGEvaluationRunner",
    "run_rag_evaluation",
]

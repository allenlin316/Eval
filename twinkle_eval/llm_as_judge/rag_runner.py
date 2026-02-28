"""
RAG Evaluation Runner.

This module provides a comprehensive RAG (Retrieval-Augmented Generation) evaluation
pipeline that combines:
1. Retrieval evaluation (Recall@k, nDCG@k, MRR) - No LLM needed
2. Generation + LLM-as-Judge evaluation (Faithfulness, Answer Relevance, Accuracy)

Usage:
    runner = RAGEvaluationRunner("config_rag.yaml")
    runner.run()
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .config_runner import _resolve_env_vars, LLMAsJudgeConfigRunner
from .metrics.retrieval import (
    RetrievalMetrics,
    AggregatedRetrievalMetrics,
    load_qrels,
    extract_retrieved_docs_from_rag_data,
)
from .metrics.base_metric import BaseMetric
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.answer_relevancy import AnswerRelevancyMetric
from .metrics.accuracy import AccuracyMetric
from .models.base_judge import BaseJudgeModel
from .models.openai_judge import OpenAIJudgeModel
from .models.google_judge import GoogleJudgeModel
from .schemas import TestCase, MetricResult

# Load .env file if exists
load_dotenv()


class RAGEvaluationRunner:
    """
    Run RAG evaluations from configuration file.
    
    This runner handles:
    1. Retrieval metrics (Recall@k, nDCG@k, MRR) - computed without LLM
    2. Answer generation from top-k contexts
    3. LLM-as-Judge evaluation (Faithfulness, Answer Relevance, Accuracy)
    """
    
    def __init__(self, config_path: str = "config_rag.yaml", env_path: str = ".env"):
        """
        Initialize the RAG evaluation runner.
        
        Args:
            config_path: Path to configuration file
            env_path: Path to .env file
        """
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        
        # Models
        self.generator_model: Optional[BaseJudgeModel] = None  # For answer generation
        self.judge_model: Optional[BaseJudgeModel] = None       # For evaluation
        
        # Data
        self.rag_data: List[Dict] = []
        self.qrels: Dict[str, Dict[str, int]] = {}
        
        # Results
        self.retrieval_results: Optional[AggregatedRetrievalMetrics] = None
        self.generation_results: List[Dict] = []
        
        # Load .env
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✓ Loaded environment variables from: {env_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load and parse configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        if "rag_evaluation" not in self.config:
            raise ValueError(
                "Configuration file must contain 'rag_evaluation' section."
            )
        
        # Resolve environment variables
        self.config = _resolve_env_vars(self.config)
        
        return self.config
    
    def load_rag_data(self) -> List[Dict]:
        """Load RAG data from JSONL file."""
        rag_config = self.config["rag_evaluation"]
        data_path = rag_config.get("data", {}).get("rag_file")
        
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"RAG data file not found: {data_path}")
        
        self.rag_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rag_data.append(json.loads(line))
        
        print(f"✓ Loaded {len(self.rag_data)} RAG tasks from: {data_path}")
        return self.rag_data
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments from qrels file."""
        rag_config = self.config["rag_evaluation"]
        qrels_path = rag_config.get("data", {}).get("qrels_file")
        
        if not qrels_path:
            print("⚠ No qrels_file specified, skipping retrieval evaluation")
            return {}
        
        if not os.path.exists(qrels_path):
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
        
        self.qrels = load_qrels(qrels_path)
        print(f"✓ Loaded qrels for {len(self.qrels)} queries from: {qrels_path}")
        return self.qrels
    
    def initialize_models(self):
        """Initialize generator and judge models from configuration."""
        rag_config = self.config["rag_evaluation"]
        
        # Generator model (for answer generation)
        if "generator" in rag_config:
            gen_config = rag_config["generator"]
            self.generator_model = self._create_model(
                gen_config.get("llm_api", {}),
                gen_config.get("model", {}),
            )
            print(f"✓ Initialized generator model: {self.generator_model.model_name}")
        
        # Judge model (for LLM-as-Judge evaluation)
        if "judge" in rag_config:
            judge_config = rag_config["judge"]
            self.judge_model = self._create_model(
                judge_config.get("llm_api", {}),
                judge_config.get("model", {}),
            )
            print(f"✓ Initialized judge model: {self.judge_model.model_name}")
        
        # If only one model is specified, use it for both
        if self.generator_model and not self.judge_model:
            self.judge_model = self.generator_model
            print("  Using generator model as judge model")
        elif self.judge_model and not self.generator_model:
            self.generator_model = self.judge_model
            print("  Using judge model as generator model")
    
    def _create_model(
        self,
        llm_api_config: Dict,
        model_config: Dict,
    ) -> BaseJudgeModel:
        """Create a model instance from configuration."""
        model_type = model_config.get("type", "openai")
        model_name = model_config.get("name", "gpt-4")
        
        if model_type == "openai":
            return OpenAIJudgeModel(
                model_name=model_name,
                api_key=llm_api_config.get("api_key", ""),
                base_url=llm_api_config.get("base_url", "https://api.openai.com/v1"),
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 4096),
                disable_ssl_verify=llm_api_config.get("disable_ssl_verify", False),
                timeout=llm_api_config.get("timeout", 600),
                max_retries=llm_api_config.get("max_retries", 3),
                supports_structured_output=model_config.get("supports_structured_output", False),
                supports_json_mode=model_config.get("supports_json_mode", False),
            )
        elif model_type == "google":
            return GoogleJudgeModel(
                model_name=model_name,
                api_key=llm_api_config.get("api_key", ""),
                temperature=model_config.get("temperature", 0.0),
                max_tokens=model_config.get("max_tokens", 4096),
                timeout=llm_api_config.get("timeout", 600),
                max_retries=llm_api_config.get("max_retries", 3),
                supports_structured_output=model_config.get("supports_structured_output", False),
                supports_json_mode=model_config.get("supports_json_mode", True),
                requests_per_minute=llm_api_config.get("api_rate_limit", 5),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def evaluate_retrieval(self) -> Optional[AggregatedRetrievalMetrics]:
        """
        Evaluate retrieval performance using IR metrics.
        
        This does NOT require an LLM - it computes metrics based on
        comparing retrieved documents to ground truth relevance judgments.
        """
        rag_config = self.config["rag_evaluation"]
        retrieval_config = rag_config.get("retrieval_metrics", {})
        
        if not retrieval_config.get("enabled", True):
            print("⚠ Retrieval evaluation disabled")
            return None
        
        if not self.qrels:
            print("⚠ No qrels loaded, skipping retrieval evaluation")
            return None
        
        # Get k values for evaluation
        k_values = retrieval_config.get("k_values", [1, 3, 5, 10])
        
        # Extract retrieved documents from RAG data
        results = extract_retrieved_docs_from_rag_data(self.rag_data)
        
        # Calculate metrics
        calculator = RetrievalMetrics(k_values=k_values)
        self.retrieval_results = calculator.evaluate(self.qrels, results)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 RETRIEVAL EVALUATION RESULTS (No LLM)")
        print("=" * 60)
        print(f"Number of queries evaluated: {self.retrieval_results.num_queries}")
        print()
        
        print("Recall@k:")
        for k, score in sorted(self.retrieval_results.mean_recall_at_k.items()):
            print(f"  Recall@{k}: {score:.4f}")
        
        print("\nnDCG@k:")
        for k, score in sorted(self.retrieval_results.mean_ndcg_at_k.items()):
            print(f"  nDCG@{k}: {score:.4f}")
        
        print(f"\nMRR: {self.retrieval_results.mean_mrr:.4f}")
        print("=" * 60 + "\n")
        
        return self.retrieval_results
    
    def generate_answers(self) -> List[Dict]:
        """
        Generate answers using top-k contexts and the input question.
        
        This uses the generator LLM to produce answers from retrieved contexts.
        """
        rag_config = self.config["rag_evaluation"]
        gen_config = rag_config.get("generation", {})
        
        if not gen_config.get("enabled", True):
            print("⚠ Answer generation disabled")
            return []
        
        if not self.generator_model:
            raise ValueError("Generator model not initialized")
        
        top_k = gen_config.get("top_k", 5)
        prompt_template = gen_config.get("prompt_template", self._default_generation_prompt())
        
        print(f"\n🤖 Generating answers using top-{top_k} contexts...")
        
        self.generation_results = []
        
        for task in tqdm(self.rag_data, desc="Generating answers"):
            task_id = task.get("task_id", "")
            
            # Extract input (last user message)
            input_messages = task.get("input", [])
            if input_messages:
                last_user_msg = next(
                    (m["text"] for m in reversed(input_messages) if m.get("speaker") == "user"),
                    ""
                )
            else:
                last_user_msg = ""
            
            # Get top-k contexts
            contexts = task.get("contexts", [])
            sorted_contexts = sorted(contexts, key=lambda x: x.get("score", 0), reverse=True)
            top_contexts = sorted_contexts[:top_k]
            
            # Build context string
            context_str = "\n\n".join([
                f"[Document {i+1}]: {ctx.get('text', '')}"
                for i, ctx in enumerate(top_contexts)
            ])
            
            # Generate prompt
            prompt = prompt_template.format(
                context=context_str,
                question=last_user_msg,
            )
            
            # Generate answer
            try:
                generated_answer = self.generator_model.generate(prompt)
            except Exception as e:
                print(f"  Error generating answer for {task_id}: {e}")
                generated_answer = f"Error: {e}"
            
            # Get expected output (target answer)
            targets = task.get("targets", [])
            expected_output = ""
            if targets:
                expected_output = targets[0].get("text", "")
            
            self.generation_results.append({
                "task_id": task_id,
                "input": last_user_msg,
                "contexts": [ctx.get("text", "") for ctx in top_contexts],
                "generated_answer": generated_answer,
                "expected_output": expected_output,
                "metadata": {
                    "conversation_id": task.get("conversation_id", ""),
                    "turn": task.get("turn", ""),
                    "answerability": task.get("Answerability", []),
                }
            })
        
        print(f"✓ Generated {len(self.generation_results)} answers")
        return self.generation_results
    
    def _default_generation_prompt(self) -> str:
        """Default prompt template for answer generation."""
        return """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    def evaluate_with_llm_judge(self) -> List[Dict]:
        """
        Evaluate generated answers using LLM-as-Judge.
        
        This evaluates:
        - Faithfulness: Is the answer grounded in the context?
        - Answer Relevance: Does the answer address the question?
        - Accuracy: Does the answer match the expected output?
        """
        rag_config = self.config["rag_evaluation"]
        judge_config = rag_config.get("llm_judge_metrics", {})
        
        if not judge_config.get("enabled", True):
            print("⚠ LLM-as-Judge evaluation disabled")
            return []
        
        if not self.judge_model:
            raise ValueError("Judge model not initialized")
        
        if not self.generation_results:
            print("⚠ No generated answers to evaluate")
            return []
        
        # Initialize metrics
        metrics: List[BaseMetric] = []
        
        if judge_config.get("faithfulness", {}).get("enabled", True):
            metrics.append(FaithfulnessMetric(
                threshold=judge_config.get("faithfulness", {}).get("threshold", 0.8),
                judge_model=self.judge_model,
            ))
        
        if judge_config.get("answer_relevancy", {}).get("enabled", True):
            metrics.append(AnswerRelevancyMetric(
                threshold=judge_config.get("answer_relevancy", {}).get("threshold", 0.7),
                judge_model=self.judge_model,
            ))
        
        if judge_config.get("accuracy", {}).get("enabled", True):
            metrics.append(AccuracyMetric(
                threshold=judge_config.get("accuracy", {}).get("threshold", 0.7),
                judge_model=self.judge_model,
            ))
        
        print(f"\n⚖️ Evaluating with LLM-as-Judge ({len(metrics)} metrics)...")
        
        evaluation_results = []
        
        for result in tqdm(self.generation_results, desc="Evaluating answers"):
            test_case = TestCase(
                input=result["input"],
                actual_output=result["generated_answer"],
                expected_output=result.get("expected_output"),
                retrieval_context=result.get("contexts"),
            )
            
            metric_results = {}
            for metric in metrics:
                try:
                    metric_result = metric.measure(test_case)
                    metric_results[metric.name] = {
                        "score": metric_result.score,
                        "success": metric_result.success,
                        "reason": metric_result.reason,
                    }
                except Exception as e:
                    print(f"  Error evaluating {metric.name} for {result['task_id']}: {e}")
                    metric_results[metric.name] = {
                        "score": 0.0,
                        "success": False,
                        "reason": f"Error: {e}",
                    }
            
            result["evaluation"] = metric_results
            evaluation_results.append(result)
        
        # Calculate aggregate scores
        self._print_llm_judge_summary(evaluation_results, metrics)
        
        return evaluation_results
    
    def _print_llm_judge_summary(
        self,
        results: List[Dict],
        metrics: List[BaseMetric],
    ):
        """Print summary of LLM-as-Judge evaluation."""
        print("\n" + "=" * 60)
        print("⚖️ LLM-AS-JUDGE EVALUATION RESULTS")
        print("=" * 60)
        print(f"Number of cases evaluated: {len(results)}")
        print()
        
        for metric in metrics:
            scores = [
                r["evaluation"].get(metric.name, {}).get("score", 0)
                for r in results
                if metric.name in r.get("evaluation", {})
            ]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                success_count = sum(
                    1 for r in results
                    if r.get("evaluation", {}).get(metric.name, {}).get("success", False)
                )
                print(f"{metric.name}:")
                print(f"  Average Score: {avg_score:.4f}")
                print(f"  Success Rate: {success_count}/{len(scores)} ({100*success_count/len(scores):.1f}%)")
        
        print("=" * 60 + "\n")
    
    def save_results(self):
        """Save evaluation results to output files."""
        rag_config = self.config["rag_evaluation"]
        output_config = rag_config.get("output", {})
        
        output_dir = output_config.get("directory", "results/rag_evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save retrieval results
        if self.retrieval_results:
            retrieval_output = {
                "timestamp": timestamp,
                "num_queries": self.retrieval_results.num_queries,
                "metrics": {
                    "recall_at_k": self.retrieval_results.mean_recall_at_k,
                    "ndcg_at_k": self.retrieval_results.mean_ndcg_at_k,
                    "mrr": self.retrieval_results.mean_mrr,
                },
            }
            
            if output_config.get("save_per_query_retrieval", False):
                retrieval_output["per_query_results"] = [
                    {
                        "query_id": r.query_id,
                        "recall_at_k": r.recall_at_k,
                        "ndcg_at_k": r.ndcg_at_k,
                        "mrr": r.mrr,
                        "num_relevant": r.num_relevant,
                        "hits_at_k": r.hits_at_k,
                    }
                    for r in self.retrieval_results.per_query_results
                ]
            
            retrieval_path = os.path.join(output_dir, f"retrieval_results_{timestamp}.json")
            with open(retrieval_path, "w", encoding="utf-8") as f:
                json.dump(retrieval_output, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved retrieval results to: {retrieval_path}")
        
        # Save generation + evaluation results
        if self.generation_results:
            gen_path = os.path.join(output_dir, f"generation_results_{timestamp}.json")
            with open(gen_path, "w", encoding="utf-8") as f:
                json.dump(self.generation_results, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved generation + evaluation results to: {gen_path}")
        
        # Save summary
        if output_config.get("save_summary", True):
            summary = self._create_summary()
            summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved summary to: {summary_path}")
    
    def _create_summary(self) -> Dict:
        """Create a summary of all evaluation results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_path": self.config_path,
        }
        
        # Retrieval summary
        if self.retrieval_results:
            summary["retrieval"] = {
                "num_queries": self.retrieval_results.num_queries,
                "recall_at_k": self.retrieval_results.mean_recall_at_k,
                "ndcg_at_k": self.retrieval_results.mean_ndcg_at_k,
                "mrr": self.retrieval_results.mean_mrr,
            }
        
        # Generation + LLM judge summary
        if self.generation_results:
            metrics_summary = {}
            
            # Collect all metric names
            all_metrics = set()
            for r in self.generation_results:
                all_metrics.update(r.get("evaluation", {}).keys())
            
            for metric_name in all_metrics:
                scores = [
                    r["evaluation"][metric_name]["score"]
                    for r in self.generation_results
                    if metric_name in r.get("evaluation", {})
                ]
                success_count = sum(
                    1 for r in self.generation_results
                    if r.get("evaluation", {}).get(metric_name, {}).get("success", False)
                )
                
                if scores:
                    metrics_summary[metric_name] = {
                        "average_score": sum(scores) / len(scores),
                        "success_count": success_count,
                        "total_count": len(scores),
                        "success_rate": success_count / len(scores),
                    }
            
            summary["llm_judge"] = {
                "num_cases": len(self.generation_results),
                "metrics": metrics_summary,
            }
        
        return summary
    
    def run(self) -> Dict:
        """
        Run the complete RAG evaluation pipeline.
        
        Returns:
            Summary of all evaluation results
        """
        print("\n" + "=" * 60)
        print("🚀 RAG EVALUATION PIPELINE")
        print("=" * 60)
        
        # 1. Load configuration
        print("\n📝 Loading configuration...")
        self.load_config()
        print(f"✓ Loaded config from: {self.config_path}")
        
        # 2. Load data
        print("\n📂 Loading data...")
        self.load_rag_data()
        self.load_qrels()
        
        # 3. Initialize models (if needed)
        rag_config = self.config["rag_evaluation"]
        if (rag_config.get("generation", {}).get("enabled", True) or 
            rag_config.get("llm_judge_metrics", {}).get("enabled", True)):
            print("\n🔧 Initializing models...")
            self.initialize_models()
        
        # 4. Evaluate retrieval (no LLM)
        self.evaluate_retrieval()
        
        # 5. Generate answers (LLM)
        if rag_config.get("generation", {}).get("enabled", True):
            self.generate_answers()
        
        # 6. Evaluate with LLM-as-Judge
        if rag_config.get("llm_judge_metrics", {}).get("enabled", True):
            self.evaluate_with_llm_judge()
        
        # 7. Save results
        print("\n💾 Saving results...")
        self.save_results()
        
        print("\n✅ RAG evaluation complete!")
        
        return self._create_summary()


def run_rag_evaluation(config_path: str = "config_rag.yaml") -> Dict:
    """
    Convenience function to run RAG evaluation from config file.
    
    Args:
        config_path: Path to RAG evaluation config
        
    Returns:
        Summary of evaluation results
    """
    runner = RAGEvaluationRunner(config_path)
    return runner.run()

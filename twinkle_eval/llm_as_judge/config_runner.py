"""
Configuration-based runner for LLM-as-Judge evaluations.

This module allows users to run LLM-as-Judge evaluations by defining
everything in a config.yaml file, without writing any code.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file if exists
load_dotenv()

from .models.base_judge import BaseJudgeModel
from .models.openai_judge import OpenAIJudgeModel
from .models.google_judge import GoogleJudgeModel
from .metrics.base_metric import BaseMetric
from .metrics.g_eval import GEval
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.answer_relevancy import AnswerRelevancyMetric
from .schemas import MetricResult, Rubric, TestCase


def _resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variable references in config values.

    Supports formats:
    - ${VAR_NAME} - required, raises error if not set
    - ${VAR_NAME:-default} - optional with default value

    Args:
        value: Config value (string, dict, or list)

    Returns:
        Value with environment variables resolved
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Please set it in .env file or system environment."
                )

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]

    return value


class LLMAsJudgeConfigRunner:
    """
    Run LLM-as-Judge evaluations from configuration file.

    This class reads configuration from YAML and executes evaluations
    without requiring users to write code.
    """

    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """
        Initialize the config runner.

        Args:
            config_path: Path to configuration file
            env_path: Path to .env file (default: .env in current directory)
        """
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        self.judge_model: Optional[BaseJudgeModel] = None
        self.metrics: List[BaseMetric] = []
        self.test_cases: List[TestCase] = []

        # Load .env file if specified path exists
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✓ Loaded environment variables from: {env_path}")

    def load_config(self) -> Dict[str, Any]:
        """
        Load and parse configuration file.

        Environment variables can be referenced in the config using:
        - ${VAR_NAME} - required variable
        - ${VAR_NAME:-default} - optional with default value

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If llm_as_judge section is missing
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        if "llm_as_judge" not in self.config:
            raise ValueError(
                "Configuration file must contain 'llm_as_judge' section. "
                "See config_llm_as_judge.yaml for an example."
            )

        # Resolve environment variables in config
        self.config = _resolve_env_vars(self.config)

        return self.config

    def initialize_judge_model(self) -> BaseJudgeModel:
        """
        Initialize the judge model from configuration.

        Configuration format (same as config.yaml):
           llm_as_judge:
             llm_api:
               base_url: "..."
               api_key: "..."
             model:
               name: "..."
               type: "openai"  # or "google"

        Returns:
            Initialized judge model

        Raises:
            ValueError: If judge model configuration is invalid
        """
        llm_as_judge_config = self.config["llm_as_judge"]

        if "llm_api" not in llm_as_judge_config or "model" not in llm_as_judge_config:
            raise ValueError(
                "Configuration must contain 'llm_api' and 'model' sections under 'llm_as_judge'. "
                "Run 'twinkle-eval --init' to generate example config files."
            )

        llm_api_config = llm_as_judge_config["llm_api"]
        model_config = llm_as_judge_config["model"]

        model_type = model_config.get("type", "openai")
        model_name = model_config.get("name", "gpt-4")

        if model_type == "openai":
            self.judge_model = OpenAIJudgeModel(
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
            self.judge_model = GoogleJudgeModel(
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
            raise ValueError(
                f"Unsupported judge model type: {model_type}. "
                "Supported types: 'openai', 'google'"
            )

        print(f"✓ Initialized judge model: {self.judge_model.model_name}")
        return self.judge_model

    def initialize_metrics(self) -> List[BaseMetric]:
        """
        Initialize evaluation metrics from configuration.

        Returns:
            List of initialized metrics

        Raises:
            ValueError: If metric configuration is invalid
        """
        metrics_config = self.config["llm_as_judge"].get("metrics", [])

        if not metrics_config:
            raise ValueError("No metrics defined in configuration")

        self.metrics = []

        for metric_config in metrics_config:
            metric_type = metric_config.get("type")
            enabled = metric_config.get("enabled", True)

            if not enabled:
                continue

            if metric_type == "answer_relevancy":
                metric = AnswerRelevancyMetric(
                    threshold=metric_config.get("threshold", 0.7),
                    judge_model=self.judge_model,
                )

            elif metric_type == "faithfulness":
                metric = FaithfulnessMetric(
                    threshold=metric_config.get("threshold", 0.8),
                    judge_model=self.judge_model,
                )

            elif metric_type == "g_eval":
                # Parse rubric if provided
                rubric = None
                if "rubric" in metric_config:
                    rubric = [
                        Rubric(
                            score_range=tuple(r["score_range"]),
                            description=r["description"]
                        )
                        for r in metric_config["rubric"]
                    ]

                metric = GEval(
                    name=metric_config.get("name", "G-Eval"),
                    criteria=metric_config.get("criteria", ""),
                    evaluation_params=metric_config.get("evaluation_params", ["input", "actual_output"]),
                    rubric=rubric,
                    strict_mode=metric_config.get("strict_mode", False),
                    threshold=metric_config.get("threshold", 0.5),
                    judge_model=self.judge_model,
                )

            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            self.metrics.append(metric)
            print(f"✓ Initialized metric: {metric.name}")

        return self.metrics

    def load_test_cases(self) -> List[TestCase]:
        """
        Load test cases from configuration.

        Supports multiple formats:
        - dataset_paths: List of directories or files (like general evaluation)
        - file: Single file path
        - cases: Inline test cases

        Returns:
            List of test cases

        Raises:
            ValueError: If test cases are not properly configured
        """
        test_cases_config = self.config["llm_as_judge"].get("test_cases", {})

        # Option 1: Dataset paths (directories or files) - NEW!
        if "dataset_paths" in test_cases_config:
            self.test_cases = self._load_test_cases_from_paths(
                test_cases_config["dataset_paths"]
            )

        # Option 2: Test cases from single file
        elif "file" in test_cases_config:
            self.test_cases = self._load_test_cases_from_file(
                test_cases_config["file"]
            )

        # Option 3: Inline test cases
        elif "cases" in test_cases_config:
            self.test_cases = self._load_inline_test_cases(
                test_cases_config["cases"]
            )

        else:
            raise ValueError(
                "Test cases must be provided as 'dataset_paths', 'file', or 'cases' in configuration"
            )

        print(f"✓ Loaded {len(self.test_cases)} test cases")
        return self.test_cases

    def _load_test_cases_from_paths(self, paths: List[str]) -> List[TestCase]:
        """
        Load test cases from multiple paths (directories or files).

        Similar to general evaluation's dataset_paths functionality.

        Args:
            paths: List of directory paths or file paths

        Returns:
            List of test cases from all paths
        """
        if isinstance(paths, str):
            paths = [paths]

        all_test_cases = []
        all_files = []

        for path in paths:
            if os.path.isdir(path):
                # It's a directory - find all JSON/JSONL files
                files = self._find_test_case_files(path)
                all_files.extend(files)
                print(f"  Found {len(files)} file(s) in directory: {path}")
            elif os.path.isfile(path):
                # It's a file - add directly
                all_files.append(path)
                print(f"  Using file: {path}")
            else:
                raise ValueError(f"Path not found: {path}")

        # Load all files
        for file_path in all_files:
            cases = self._load_test_cases_from_file(file_path)
            all_test_cases.extend(cases)
            print(f"    Loaded {len(cases)} test case(s) from: {os.path.basename(file_path)}")

        return all_test_cases

    def _find_test_case_files(self, directory: str) -> List[str]:
        """
        Find all test case files in a directory.

        Looks for .json and .jsonl files.

        Args:
            directory: Directory path to search

        Returns:
            List of file paths
        """
        import glob

        files = []

        # Search for JSON files
        json_files = glob.glob(os.path.join(directory, "*.json"))
        jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))

        files.extend(json_files)
        files.extend(jsonl_files)

        # Also search in subdirectories
        for root, dirs, _ in os.walk(directory):
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                json_files = glob.glob(os.path.join(subdir_path, "*.json"))
                jsonl_files = glob.glob(os.path.join(subdir_path, "*.jsonl"))
                files.extend(json_files)
                files.extend(jsonl_files)

        return sorted(files)

    def _load_test_cases_from_file(self, file_path: str) -> List[TestCase]:
        """Load test cases from a JSON or JSONL file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test cases file not found: {file_path}")

        test_cases = []

        # Handle JSONL
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data = json.loads(line)
                        test_cases.append(self._create_test_case(data))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON at line {line_num} in {file_path}: {e}")
                        continue

        # Handle JSON
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        test_cases.append(self._create_test_case(item))
                else:
                    test_cases.append(self._create_test_case(data))

        else:
            raise ValueError(f"Unsupported file format: {file_path}. Use .json or .jsonl")

        return test_cases

    def _load_inline_test_cases(self, cases: List[Dict]) -> List[TestCase]:
        """Load test cases defined inline in the config."""
        return [self._create_test_case(case) for case in cases]

    def _create_test_case(self, data: Dict) -> TestCase:
        """Create a TestCase from dictionary data."""
        return TestCase(
            input=data.get("input", ""),
            actual_output=data.get("actual_output", ""),
            expected_output=data.get("expected_output"),
            retrieval_context=data.get("retrieval_context"),
            metadata=data.get("metadata", {}),
        )

    def run_evaluation(self, use_async: bool = True) -> Dict[str, Any]:
        """
        Run the evaluation.

        Args:
            use_async: Whether to use async evaluation (faster)

        Returns:
            Evaluation results dictionary
        """
        print("\n" + "=" * 60)
        print("Starting LLM-as-Judge Evaluation")
        print("=" * 60 + "\n")

        # Load configuration
        self.load_config()

        # Initialize components
        self.initialize_judge_model()
        self.initialize_metrics()
        self.load_test_cases()

        # Run evaluation
        if use_async:
            results = asyncio.run(self._run_async_evaluation())
        else:
            results = self._run_sync_evaluation()

        # Save results
        output_config = self.config["llm_as_judge"].get("output", {})
        output_file = self._save_results(results, output_config)

        print("\n" + "=" * 60)
        print("Evaluation Completed!")
        print(f"Results saved to: {output_file}")
        print("=" * 60)

        return results

    def _run_sync_evaluation(self) -> Dict[str, Any]:
        """Run evaluation synchronously."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config["llm_as_judge"],
            "test_results": [],
            "summary": {},
        }

        print("\nRunning evaluation (sync mode)...\n")

        for test_case in tqdm(self.test_cases, desc="Evaluating test cases"):
            test_result = {
                "test_case": {
                    "input": test_case.input,
                    "actual_output": test_case.actual_output,
                },
                "metric_results": [],
            }

            for metric in self.metrics:
                result = metric.measure(test_case)
                test_result["metric_results"].append({
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "success": result.success,
                    "reason": result.reason,
                })

            results["test_results"].append(test_result)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["test_results"])

        return results

    async def _run_async_evaluation(self) -> Dict[str, Any]:
        """Run evaluation asynchronously."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config["llm_as_judge"],
            "test_results": [],
            "summary": {},
        }

        print("\nRunning evaluation (async mode)...\n")

        # Create all evaluation tasks
        tasks = []
        for test_case in self.test_cases:
            for metric in self.metrics:
                tasks.append(self._evaluate_single_async(test_case, metric))

        # Run all tasks concurrently
        task_results = []
        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                task_results.append(result)
                pbar.update(1)

        # Organize results by test case
        test_results_map = {}
        for test_case, metric_name, metric_result in task_results:
            test_key = test_case.input[:50]  # Use first 50 chars as key

            if test_key not in test_results_map:
                test_results_map[test_key] = {
                    "test_case": {
                        "input": test_case.input,
                        "actual_output": test_case.actual_output,
                    },
                    "metric_results": [],
                }

            test_results_map[test_key]["metric_results"].append({
                "metric_name": metric_name,
                "score": metric_result.score,
                "success": metric_result.success,
                "reason": metric_result.reason,
            })

        results["test_results"] = list(test_results_map.values())
        results["summary"] = self._calculate_summary(results["test_results"])

        return results

    async def _evaluate_single_async(
        self, test_case: TestCase, metric: BaseMetric
    ) -> tuple:
        """Evaluate a single test case with a single metric asynchronously."""
        result = await metric.a_measure(test_case)
        return (test_case, metric.name, result)

    def _calculate_summary(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        summary = {
            "total_test_cases": len(test_results),
            "metrics": {},
        }

        # Aggregate by metric
        metric_scores = {}
        for test_result in test_results:
            for metric_result in test_result["metric_results"]:
                metric_name = metric_result["metric_name"]

                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []

                metric_scores[metric_name].append(metric_result["score"])

        # Calculate statistics for each metric
        for metric_name, scores in metric_scores.items():
            summary["metrics"][metric_name] = {
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "total_evaluations": len(scores),
                "success_rate": sum(1 for r in test_results
                                   for mr in r["metric_results"]
                                   if mr["metric_name"] == metric_name and mr["success"]) / len(scores)
                if scores else 0,
            }

        return summary

    def _save_results(self, results: Dict[str, Any], output_config: Dict) -> str:
        """Save results to file."""
        output_dir = output_config.get("directory", "results")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir,
            f"llm_as_judge_results_{timestamp}.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Also save summary separately if configured
        if output_config.get("save_summary", True):
            summary_file = os.path.join(
                output_dir,
                f"llm_as_judge_summary_{timestamp}.json"
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": results["timestamp"],
                    "summary": results["summary"],
                }, f, indent=2, ensure_ascii=False)

        return output_file


def run_from_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Convenience function to run LLM-as-Judge from config file.

    Args:
        config_path: Path to configuration file

    Returns:
        Evaluation results
    """
    runner = LLMAsJudgeConfigRunner(config_path)
    return runner.run_evaluation()

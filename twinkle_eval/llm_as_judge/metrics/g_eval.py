"""
G-Eval metric implementation.

G-Eval is a flexible framework for evaluating LLM outputs using custom criteria.
Reference: https://arxiv.org/pdf/2303.16634.pdf
"""

from typing import List, Optional

from ..models.base_judge import BaseJudgeModel
from ..schemas import EvaluationSteps, MetricResult, ReasonScore, Rubric, TestCase
from ..templates.g_eval_template import GEvalTemplate
from ..utils import generate_with_schema_and_extract
from .base_metric import BaseMetric


class GEval(BaseMetric):
    """
    G-Eval: Framework for LLM evaluation with custom criteria.

    G-Eval allows you to define custom evaluation criteria and automatically
    generates evaluation steps. It supports flexible scoring with rubrics.
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: List[str],
        rubric: Optional[List[Rubric]] = None,
        strict_mode: bool = False,
        threshold: float = 0.5,
        judge_model: Optional[BaseJudgeModel] = None,
    ):
        """
        Initialize G-Eval metric.

        Args:
            name: Name of the metric
            criteria: Evaluation criteria description
            evaluation_params: List of test case parameters to use
                (e.g., ["input", "actual_output", "expected_output"])
            rubric: Optional scoring rubric for score interpretation
            strict_mode: If True, use binary scoring (0 or 1)
            threshold: Success threshold (default 0.5)
            judge_model: Optional judge model
        """
        if strict_mode:
            threshold = 1  # In strict mode, must score 1 to pass

        super().__init__(name, threshold, judge_model)

        self.criteria = criteria
        self.evaluation_params = evaluation_params
        self.rubric = rubric
        self.strict_mode = strict_mode

        # Will be generated on first use
        self.evaluation_steps: Optional[List[str]] = None

    def _generate_evaluation_steps(self, judge_model: BaseJudgeModel) -> List[str]:
        """
        Generate evaluation steps using the judge model.

        Args:
            judge_model: The judge model to use

        Returns:
            List of evaluation steps
        """
        if self.evaluation_steps is not None:
            return self.evaluation_steps

        prompt = GEvalTemplate.generate_evaluation_steps(
            self.criteria,
            self.evaluation_params,
        )

        response = judge_model.generate(
            prompt,
            system_prompt=GEvalTemplate.SYSTEM_PROMPT,
            schema=EvaluationSteps if judge_model.supports_structured_output() else None,
        )

        # Extract steps
        steps = generate_with_schema_and_extract(
            response,
            EvaluationSteps,
            lambda schema: schema.steps,
            lambda json_data: json_data.get("steps", []),
        )

        if not steps:
            # Fallback to default steps
            steps = [
                "Read the evaluation criteria carefully",
                "Examine the provided data",
                "Assess whether the criteria is met",
                "Assign an appropriate score",
            ]

        self.evaluation_steps = steps
        return steps

    async def _a_generate_evaluation_steps(self, judge_model: BaseJudgeModel) -> List[str]:
        """
        Asynchronously generate evaluation steps.

        Args:
            judge_model: The judge model to use

        Returns:
            List of evaluation steps
        """
        if self.evaluation_steps is not None:
            return self.evaluation_steps

        prompt = GEvalTemplate.generate_evaluation_steps(
            self.criteria,
            self.evaluation_params,
        )

        response = await judge_model.a_generate(
            prompt,
            system_prompt=GEvalTemplate.SYSTEM_PROMPT,
            schema=EvaluationSteps if judge_model.supports_structured_output() else None,
        )

        # Extract steps
        steps = generate_with_schema_and_extract(
            response,
            EvaluationSteps,
            lambda schema: schema.steps,
            lambda json_data: json_data.get("steps", []),
        )

        if not steps:
            steps = [
                "Read the evaluation criteria carefully",
                "Examine the provided data",
                "Assess whether the criteria is met",
                "Assign an appropriate score",
            ]

        self.evaluation_steps = steps
        return steps

    def _get_test_case_params(self, test_case: TestCase) -> dict:
        """
        Extract relevant parameters from test case.

        Args:
            test_case: The test case

        Returns:
            Dictionary of parameter values
        """
        params = {}

        for param_name in self.evaluation_params:
            if param_name == "input":
                params["input"] = test_case.input
            elif param_name == "actual_output":
                params["actual_output"] = test_case.actual_output
            elif param_name == "expected_output":
                params["expected_output"] = test_case.expected_output or "N/A"
            elif param_name == "retrieval_context":
                context = test_case.retrieval_context or []
                params["retrieval_context"] = "\n".join(context) if context else "N/A"
            else:
                # Try to get from metadata
                params[param_name] = test_case.metadata.get(param_name, "N/A")

        return params

    def measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Measure G-Eval score for a test case.

        Args:
            test_case: The test case to evaluate
            judge_model: Optional judge model

        Returns:
            MetricResult with score and reasoning
        """
        model = self._get_judge_model(judge_model)

        # Generate evaluation steps if needed
        steps = self._generate_evaluation_steps(model)

        # Get test case parameters
        test_case_params = self._get_test_case_params(test_case)

        # Generate evaluation prompt
        prompt = GEvalTemplate.generate_evaluation_results(
            self.criteria,
            steps,
            self.evaluation_params,
            test_case_params,
            self.rubric,
            self.strict_mode,
        )

        # Get evaluation with system prompt for better model compatibility
        response = model.generate(
            prompt,
            system_prompt=GEvalTemplate.SYSTEM_PROMPT,
            schema=ReasonScore if model.supports_structured_output() else None,
        )

        # Extract score and reason
        score = generate_with_schema_and_extract(
            response,
            ReasonScore,
            lambda schema: schema.score,
            lambda json_data: json_data.get("score", 0),
        )

        reason = generate_with_schema_and_extract(
            response,
            ReasonScore,
            lambda schema: schema.reason,
            lambda json_data: json_data.get("reason", "No reason provided"),
        )

        # Normalize score to 0-1 range if not in strict mode
        if not self.strict_mode and score is not None:
            score = score / 10.0

        score = score if score is not None else 0.0
        reason = reason or "Failed to extract reasoning"

        return self._create_result(score, reason)

    async def a_measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Asynchronously measure G-Eval score.

        Args:
            test_case: The test case to evaluate
            judge_model: Optional judge model

        Returns:
            MetricResult with score and reasoning
        """
        model = self._get_judge_model(judge_model)

        # Generate evaluation steps if needed
        steps = await self._a_generate_evaluation_steps(model)

        # Get test case parameters
        test_case_params = self._get_test_case_params(test_case)

        # Generate evaluation prompt
        prompt = GEvalTemplate.generate_evaluation_results(
            self.criteria,
            steps,
            self.evaluation_params,
            test_case_params,
            self.rubric,
            self.strict_mode,
        )

        # Get evaluation with system prompt for better model compatibility
        response = await model.a_generate(
            prompt,
            system_prompt=GEvalTemplate.SYSTEM_PROMPT,
            schema=ReasonScore if model.supports_structured_output() else None,
        )

        # Extract score and reason
        score = generate_with_schema_and_extract(
            response,
            ReasonScore,
            lambda schema: schema.score,
            lambda json_data: json_data.get("score", 0),
        )

        reason = generate_with_schema_and_extract(
            response,
            ReasonScore,
            lambda schema: schema.reason,
            lambda json_data: json_data.get("reason", "No reason provided"),
        )

        # Normalize score to 0-1 range if not in strict mode
        if not self.strict_mode and score is not None:
            score = score / 10.0

        score = score if score is not None else 0.0
        reason = reason or "Failed to extract reasoning"

        return self._create_result(score, reason)

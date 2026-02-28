"""
Accuracy metric implementation.

Accuracy measures how factually correct the LLM output is compared to
the expected output (ground truth).
"""

from typing import Optional

from ..models.base_judge import BaseJudgeModel
from ..schemas import MetricResult, ReasonScore, TestCase
from ..templates.accuracy_template import AccuracyTemplate
from ..utils import generate_with_schema_and_extract
from .base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Accuracy: Evaluates factual correctness compared to expected output.

    This metric checks whether the LLM output:
    - Contains the same key facts as the expected answer
    - Has no factual errors or contradictions
    - Captures the essential meaning accurately
    """

    def __init__(
        self,
        threshold: float = 0.7,
        judge_model: Optional[BaseJudgeModel] = None,
    ):
        """
        Initialize Accuracy metric.

        Args:
            threshold: Success threshold (default 0.7)
            judge_model: Optional judge model
        """
        super().__init__("Accuracy", threshold, judge_model)

    def measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Measure accuracy score.

        Args:
            test_case: Test case with input, actual_output, and expected_output
            judge_model: Optional judge model

        Returns:
            MetricResult with accuracy score
        """
        model = self._get_judge_model(judge_model)

        if not test_case.expected_output:
            return MetricResult(
                score=0.0,
                success=False,
                reason="No expected_output provided for accuracy evaluation",
                metric_name=self.name,
            )

        # Build context string if available
        context_str = None
        if test_case.retrieval_context:
            context_str = "\n\n".join(test_case.retrieval_context)

        # Generate evaluation prompt
        prompt = AccuracyTemplate.evaluate_accuracy(
            input_text=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            retrieval_context=context_str,
        )

        # Get evaluation with system prompt
        response = model.generate(
            prompt,
            system_prompt=AccuracyTemplate.SYSTEM_PROMPT,
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

        # Normalize score to 0-1 range
        score = (score / 10.0) if score is not None else 0.0
        reason = reason or "Failed to extract reasoning"

        return self._create_result(score, reason)

    async def a_measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Asynchronously measure accuracy score.

        Args:
            test_case: Test case with input, actual_output, and expected_output
            judge_model: Optional judge model

        Returns:
            MetricResult with accuracy score
        """
        model = self._get_judge_model(judge_model)

        if not test_case.expected_output:
            return MetricResult(
                score=0.0,
                success=False,
                reason="No expected_output provided for accuracy evaluation",
                metric_name=self.name,
            )

        # Build context string if available
        context_str = None
        if test_case.retrieval_context:
            context_str = "\n\n".join(test_case.retrieval_context)

        # Generate evaluation prompt
        prompt = AccuracyTemplate.evaluate_accuracy(
            input_text=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            retrieval_context=context_str,
        )

        # Get evaluation with system prompt
        response = await model.a_generate(
            prompt,
            system_prompt=AccuracyTemplate.SYSTEM_PROMPT,
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

        # Normalize score to 0-1 range
        score = (score / 10.0) if score is not None else 0.0
        reason = reason or "Failed to extract reasoning"

        return self._create_result(score, reason)

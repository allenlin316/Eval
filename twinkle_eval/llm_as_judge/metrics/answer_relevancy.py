"""
Answer Relevancy metric implementation.

Answer Relevancy measures how well the LLM output addresses the input question.
"""

from typing import Optional

from ..models.base_judge import BaseJudgeModel
from ..schemas import MetricResult, ReasonScore, TestCase
from ..templates.answer_relevancy_template import AnswerRelevancyTemplate
from ..utils import generate_with_schema_and_extract
from .base_metric import BaseMetric


class AnswerRelevancyMetric(BaseMetric):
    """
    Answer Relevancy: Evaluates how well output addresses the input.

    This metric checks whether the LLM output:
    - Directly addresses the input question
    - Stays on topic
    - Provides useful, relevant information
    - Avoids unnecessary tangents
    """

    def __init__(
        self,
        threshold: float = 0.7,
        judge_model: Optional[BaseJudgeModel] = None,
    ):
        """
        Initialize Answer Relevancy metric.

        Args:
            threshold: Success threshold (default 0.7)
            judge_model: Optional judge model
        """
        super().__init__("Answer Relevancy", threshold, judge_model)

    def measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Measure answer relevancy score.

        Args:
            test_case: Test case with input and actual_output
            judge_model: Optional judge model

        Returns:
            MetricResult with relevancy score
        """
        model = self._get_judge_model(judge_model)

        # Generate evaluation prompt
        prompt = AnswerRelevancyTemplate.evaluate_relevancy(
            test_case.input,
            test_case.actual_output,
        )

        # Get evaluation with system prompt for better model compatibility
        response = model.generate(
            prompt,
            system_prompt=AnswerRelevancyTemplate.SYSTEM_PROMPT,
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
        Asynchronously measure answer relevancy score.

        Args:
            test_case: Test case with input and actual_output
            judge_model: Optional judge model

        Returns:
            MetricResult with relevancy score
        """
        model = self._get_judge_model(judge_model)

        # Generate evaluation prompt
        prompt = AnswerRelevancyTemplate.evaluate_relevancy(
            test_case.input,
            test_case.actual_output,
        )

        # Get evaluation with system prompt for better model compatibility
        response = await model.a_generate(
            prompt,
            system_prompt=AnswerRelevancyTemplate.SYSTEM_PROMPT,
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

"""
Base class for evaluation metrics.

This module defines the abstract interface for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..models.base_judge import BaseJudgeModel
from ..schemas import MetricResult, TestCase


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.

    Each metric evaluates a specific aspect of LLM output quality using
    an LLM as a judge.
    """

    def __init__(
        self,
        name: str,
        threshold: float = 0.5,
        judge_model: Optional[BaseJudgeModel] = None,
    ):
        """
        Initialize the metric.

        Args:
            name: Name of the metric
            threshold: Threshold for success (metric passes if score >= threshold)
            judge_model: Optional judge model (if None, must be provided during measure)
        """
        self.name = name
        self.threshold = threshold
        self.judge_model = judge_model

        # Results from last measurement
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None

    @abstractmethod
    def measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Measure the metric for a test case.

        Args:
            test_case: The test case to evaluate
            judge_model: Optional judge model to use (overrides instance judge_model)

        Returns:
            MetricResult with score, success, and reason
        """
        pass

    @abstractmethod
    async def a_measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Asynchronously measure the metric for a test case.

        Args:
            test_case: The test case to evaluate
            judge_model: Optional judge model to use (overrides instance judge_model)

        Returns:
            MetricResult with score, success, and reason
        """
        pass

    def _get_judge_model(self, judge_model: Optional[BaseJudgeModel]) -> BaseJudgeModel:
        """
        Get the judge model to use for this measurement.

        Args:
            judge_model: Optional judge model passed to measure()

        Returns:
            The judge model to use

        Raises:
            ValueError: If no judge model is available
        """
        model = judge_model or self.judge_model
        if model is None:
            raise ValueError(
                f"No judge model provided for metric '{self.name}'. "
                "Please provide a judge_model either during initialization or when calling measure()."
            )
        return model

    def _create_result(self, score: float, reason: str) -> MetricResult:
        """
        Create a MetricResult from score and reason.

        Args:
            score: The numeric score
            reason: Explanation for the score

        Returns:
            MetricResult instance
        """
        self.score = score
        self.reason = reason
        self.success = score >= self.threshold

        return MetricResult(
            score=score,
            success=self.success,
            reason=reason,
            metric_name=self.name,
        )

    def is_successful(self) -> bool:
        """
        Check if the last measurement was successful.

        Returns:
            True if score >= threshold

        Raises:
            ValueError: If measure() hasn't been called yet
        """
        if self.success is None:
            raise ValueError(
                f"Metric '{self.name}' has not been measured yet. "
                "Call measure() or a_measure() first."
            )
        return self.success

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', threshold={self.threshold})"

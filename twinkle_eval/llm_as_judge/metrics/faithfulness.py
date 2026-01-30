"""
Faithfulness metric implementation.

Faithfulness measures whether the LLM output is factually consistent
with the provided retrieval context. This is crucial for RAG systems.
"""

from typing import List, Optional

from ..models.base_judge import BaseJudgeModel
from ..schemas import Claims, MetricResult, TestCase, Truths, Verdicts
from ..templates.faithfulness_template import FaithfulnessTemplate
from ..utils import calculate_score, generate_with_schema_and_extract
from .base_metric import BaseMetric


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness: Evaluates factual consistency with retrieval context.

    This metric:
    1. Extracts truths from the retrieval context
    2. Extracts claims from the LLM output
    3. Verifies each claim against the truths
    4. Calculates a faithfulness score
    """

    def __init__(
        self,
        threshold: float = 0.7,
        judge_model: Optional[BaseJudgeModel] = None,
    ):
        """
        Initialize Faithfulness metric.

        Args:
            threshold: Success threshold (default 0.7)
            judge_model: Optional judge model
        """
        super().__init__("Faithfulness", threshold, judge_model)

        # Store intermediate results
        self.truths: Optional[List[str]] = None
        self.claims: Optional[List[str]] = None
        self.verdicts: Optional[List[dict]] = None

    def _extract_truths(self, context: List[str], judge_model: BaseJudgeModel) -> List[str]:
        """
        Extract truths from retrieval context.

        Args:
            context: Retrieval context
            judge_model: Judge model to use

        Returns:
            List of truths
        """
        if not context:
            return []

        prompt = FaithfulnessTemplate.extract_truths(context)

        response = judge_model.generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Truths if judge_model.supports_structured_output() else None,
        )

        truths = generate_with_schema_and_extract(
            response,
            Truths,
            lambda schema: schema.truths,
            lambda json_data: json_data.get("truths", []),
        )

        return truths or []

    async def _a_extract_truths(
        self, context: List[str], judge_model: BaseJudgeModel
    ) -> List[str]:
        """Async version of _extract_truths."""
        if not context:
            return []

        prompt = FaithfulnessTemplate.extract_truths(context)

        response = await judge_model.a_generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Truths if judge_model.supports_structured_output() else None,
        )

        truths = generate_with_schema_and_extract(
            response,
            Truths,
            lambda schema: schema.truths,
            lambda json_data: json_data.get("truths", []),
        )

        return truths or []

    def _extract_claims(self, output: str, judge_model: BaseJudgeModel) -> List[str]:
        """
        Extract claims from LLM output.

        Args:
            output: LLM output
            judge_model: Judge model to use

        Returns:
            List of claims
        """
        prompt = FaithfulnessTemplate.extract_claims(output)

        response = judge_model.generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Claims if judge_model.supports_structured_output() else None,
        )

        claims = generate_with_schema_and_extract(
            response,
            Claims,
            lambda schema: schema.claims,
            lambda json_data: json_data.get("claims", []),
        )

        return claims or []

    async def _a_extract_claims(self, output: str, judge_model: BaseJudgeModel) -> List[str]:
        """Async version of _extract_claims."""
        prompt = FaithfulnessTemplate.extract_claims(output)

        response = await judge_model.a_generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Claims if judge_model.supports_structured_output() else None,
        )

        claims = generate_with_schema_and_extract(
            response,
            Claims,
            lambda schema: schema.claims,
            lambda json_data: json_data.get("claims", []),
        )

        return claims or []

    def _generate_verdicts(
        self, claims: List[str], truths: List[str], judge_model: BaseJudgeModel
    ) -> List[dict]:
        """
        Generate verdicts for claims.

        Args:
            claims: List of claims
            truths: List of truths
            judge_model: Judge model to use

        Returns:
            List of verdict dicts
        """
        if not claims:
            return []

        prompt = FaithfulnessTemplate.generate_verdicts(claims, truths)

        response = judge_model.generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Verdicts if judge_model.supports_structured_output() else None,
        )

        verdicts_obj = generate_with_schema_and_extract(
            response,
            Verdicts,
            lambda schema: schema.verdicts,
            lambda json_data: json_data.get("verdicts", []),
        )

        # Convert to list of dicts
        if verdicts_obj:
            return [
                {"verdict": v.verdict if hasattr(v, "verdict") else v.get("verdict", "idk")}
                for v in verdicts_obj
            ]

        return []

    async def _a_generate_verdicts(
        self, claims: List[str], truths: List[str], judge_model: BaseJudgeModel
    ) -> List[dict]:
        """Async version of _generate_verdicts."""
        if not claims:
            return []

        prompt = FaithfulnessTemplate.generate_verdicts(claims, truths)

        response = await judge_model.a_generate(
            prompt,
            system_prompt=FaithfulnessTemplate.SYSTEM_PROMPT,
            schema=Verdicts if judge_model.supports_structured_output() else None,
        )

        verdicts_obj = generate_with_schema_and_extract(
            response,
            Verdicts,
            lambda schema: schema.verdicts,
            lambda json_data: json_data.get("verdicts", []),
        )

        if verdicts_obj:
            return [
                {"verdict": v.verdict if hasattr(v, "verdict") else v.get("verdict", "idk")}
                for v in verdicts_obj
            ]

        return []

    def measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Measure faithfulness score.

        Args:
            test_case: Test case with retrieval_context
            judge_model: Optional judge model

        Returns:
            MetricResult with faithfulness score
        """
        model = self._get_judge_model(judge_model)

        # Check if retrieval context is provided
        if not test_case.retrieval_context:
            return self._create_result(
                1.0, "No retrieval context provided - cannot evaluate faithfulness"
            )

        # Extract truths from context
        self.truths = self._extract_truths(test_case.retrieval_context, model)

        # Extract claims from output
        self.claims = self._extract_claims(test_case.actual_output, model)

        # Generate verdicts
        self.verdicts = self._generate_verdicts(self.claims, self.truths, model)

        # Calculate score
        score = calculate_score(self.verdicts, len(self.claims))

        # Generate reason
        num_claims = len(self.claims)
        num_faithful = sum(1 for v in self.verdicts if v.get("verdict") == "yes")

        reason = (
            f"Faithfulness score: {score:.2f}. "
            f"{num_faithful}/{num_claims} claims are supported by the retrieval context."
        )

        return self._create_result(score, reason)

    async def a_measure(
        self,
        test_case: TestCase,
        judge_model: Optional[BaseJudgeModel] = None,
    ) -> MetricResult:
        """
        Asynchronously measure faithfulness score.

        Args:
            test_case: Test case with retrieval_context
            judge_model: Optional judge model

        Returns:
            MetricResult with faithfulness score
        """
        model = self._get_judge_model(judge_model)

        if not test_case.retrieval_context:
            return self._create_result(
                1.0, "No retrieval context provided - cannot evaluate faithfulness"
            )

        # Extract truths from context
        self.truths = await self._a_extract_truths(test_case.retrieval_context, model)

        # Extract claims from output
        self.claims = await self._a_extract_claims(test_case.actual_output, model)

        # Generate verdicts
        self.verdicts = await self._a_generate_verdicts(self.claims, self.truths, model)

        # Calculate score
        score = calculate_score(self.verdicts, len(self.claims))

        # Generate reason
        num_claims = len(self.claims)
        num_faithful = sum(1 for v in self.verdicts if v.get("verdict") == "yes")

        reason = (
            f"Faithfulness score: {score:.2f}. "
            f"{num_faithful}/{num_claims} claims are supported by the retrieval context."
        )

        return self._create_result(score, reason)

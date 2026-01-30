"""
Unit tests for LLM-as-Judge module.

These tests verify the basic functionality of the metrics and models.
"""

import pytest
from twinkle_eval.llm_as_judge import (
    BaseJudgeModel,
    OpenAIJudgeModel,
    GEval,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    TestCase,
    Rubric,
)


class MockJudgeModel(BaseJudgeModel):
    """Mock judge model for testing without API calls."""

    def __init__(self, mock_response: str = '{"reason": "Test reason", "score": 8}'):
        super().__init__("mock-model")
        self.mock_response = mock_response
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, schema=None):
        self.call_count += 1
        return self.mock_response

    async def a_generate(self, prompt, system_prompt=None, schema=None):
        self.call_count += 1
        return self.mock_response


class TestTestCase:
    """Test TestCase schema."""

    def test_create_test_case(self):
        tc = TestCase(
            input="Test input",
            actual_output="Test output",
            expected_output="Expected",
            retrieval_context=["Context 1", "Context 2"],
        )

        assert tc.input == "Test input"
        assert tc.actual_output == "Test output"
        assert tc.expected_output == "Expected"
        assert len(tc.retrieval_context) == 2

    def test_test_case_minimal(self):
        tc = TestCase(
            input="Question",
            actual_output="Answer",
        )

        assert tc.input == "Question"
        assert tc.actual_output == "Answer"
        assert tc.expected_output is None
        assert tc.retrieval_context is None


class TestMockJudgeModel:
    """Test mock judge model."""

    def test_mock_judge_basic(self):
        judge = MockJudgeModel()
        response = judge.generate("test prompt")

        assert judge.call_count == 1
        assert "reason" in response
        assert "score" in response

    @pytest.mark.asyncio
    async def test_mock_judge_async(self):
        judge = MockJudgeModel()
        response = await judge.a_generate("test prompt")

        assert judge.call_count == 1
        assert "reason" in response


class TestAnswerRelevancyMetric:
    """Test Answer Relevancy metric."""

    def test_answer_relevancy_basic(self):
        judge = MockJudgeModel('{"reason": "Relevant answer", "score": 9}')
        metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

        test_case = TestCase(
            input="What is Python?",
            actual_output="Python is a programming language.",
        )

        result = metric.measure(test_case)

        assert result.score == 0.9  # Normalized to 0-1
        assert result.success is True
        assert result.metric_name == "Answer Relevancy"
        assert "Relevant" in result.reason

    def test_answer_relevancy_fail(self):
        judge = MockJudgeModel('{"reason": "Not relevant", "score": 3}')
        metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

        test_case = TestCase(
            input="What is Python?",
            actual_output="I like cats.",
        )

        result = metric.measure(test_case)

        assert result.score == 0.3
        assert result.success is False

    @pytest.mark.asyncio
    async def test_answer_relevancy_async(self):
        judge = MockJudgeModel('{"reason": "Relevant", "score": 8}')
        metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

        test_case = TestCase(
            input="Question",
            actual_output="Answer",
        )

        result = await metric.a_measure(test_case)

        assert result.score == 0.8
        assert result.success is True


class TestGEval:
    """Test G-Eval metric."""

    def test_g_eval_basic(self):
        judge = MockJudgeModel('{"reason": "Good quality", "score": 8}')
        metric = GEval(
            name="Quality",
            criteria="Evaluate quality",
            evaluation_params=["input", "actual_output"],
            threshold=0.7,
            judge_model=judge,
        )

        test_case = TestCase(
            input="Question",
            actual_output="Answer",
        )

        result = metric.measure(test_case)

        assert result.score == 0.8  # Normalized from 10-point scale
        assert result.success is True
        assert result.metric_name == "Quality"

    def test_g_eval_strict_mode(self):
        judge = MockJudgeModel('{"reason": "Passes", "score": 1}')
        metric = GEval(
            name="Binary Check",
            criteria="Binary evaluation",
            evaluation_params=["actual_output"],
            strict_mode=True,
            judge_model=judge,
        )

        test_case = TestCase(
            input="Test",
            actual_output="Output",
        )

        result = metric.measure(test_case)

        assert result.score == 1  # Not normalized in strict mode
        assert result.success is True
        assert metric.threshold == 1  # Auto-set in strict mode

    def test_g_eval_with_rubric(self):
        judge = MockJudgeModel('{"reason": "Excellent", "score": 10}')
        metric = GEval(
            name="Quality",
            criteria="Check quality",
            evaluation_params=["actual_output"],
            rubric=[
                Rubric(score_range=(0, 5), description="Poor"),
                Rubric(score_range=(6, 10), description="Good"),
            ],
            threshold=0.7,
            judge_model=judge,
        )

        test_case = TestCase(
            input="Q",
            actual_output="A",
        )

        result = metric.measure(test_case)

        assert result.score == 1.0
        assert result.success is True


class TestFaithfulnessMetric:
    """Test Faithfulness metric."""

    def test_faithfulness_no_context(self):
        judge = MockJudgeModel()
        metric = FaithfulnessMetric(threshold=0.8, judge_model=judge)

        test_case = TestCase(
            input="Question",
            actual_output="Answer",
            # No retrieval_context
        )

        result = metric.measure(test_case)

        assert result.score == 1.0
        assert "No retrieval context" in result.reason

    def test_faithfulness_with_context(self):
        # Mock multiple responses for different steps
        class MultistepMockJudge(BaseJudgeModel):
            def __init__(self):
                super().__init__("mock")
                self.responses = [
                    '{"truths": ["Truth 1", "Truth 2"]}',
                    '{"claims": ["Claim 1", "Claim 2"]}',
                    '{"verdicts": [{"verdict": "yes"}, {"verdict": "yes"}]}',
                ]
                self.call_index = 0

            def generate(self, prompt, system_prompt=None, schema=None):
                response = self.responses[self.call_index % len(self.responses)]
                self.call_index += 1
                return response

            async def a_generate(self, prompt, system_prompt=None, schema=None):
                return self.generate(prompt, system_prompt, schema)

        judge = MultistepMockJudge()
        metric = FaithfulnessMetric(threshold=0.8, judge_model=judge)

        test_case = TestCase(
            input="Question",
            actual_output="Answer with claims",
            retrieval_context=["Context 1", "Context 2"],
        )

        result = metric.measure(test_case)

        assert result.score == 1.0  # 2/2 claims supported
        assert result.success is True
        assert "2/2" in result.reason


class TestBaseMetric:
    """Test base metric functionality."""

    def test_metric_without_judge_model(self):
        metric = AnswerRelevancyMetric(threshold=0.7)

        test_case = TestCase(input="Q", actual_output="A")

        with pytest.raises(ValueError, match="No judge model provided"):
            metric.measure(test_case)

    def test_is_successful_before_measure(self):
        metric = AnswerRelevancyMetric(threshold=0.7)

        with pytest.raises(ValueError, match="has not been measured yet"):
            metric.is_successful()

    def test_is_successful_after_measure(self):
        judge = MockJudgeModel('{"reason": "Good", "score": 9}')
        metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

        test_case = TestCase(input="Q", actual_output="A")
        metric.measure(test_case)

        assert metric.is_successful() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

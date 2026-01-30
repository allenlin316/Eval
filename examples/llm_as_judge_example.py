"""
Example usage of the LLM-as-Judge module.

This example demonstrates how to use various evaluation metrics
with custom judge models.
"""

import asyncio
from twinkle_eval.llm_as_judge import (
    OpenAIJudgeModel,
    GEval,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    TestCase,
    Rubric,
)


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create a judge model
    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",  # Replace with your API key
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    # Create a test case
    test_case = TestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France. It's a beautiful city known for the Eiffel Tower.",
        expected_output="Paris",
    )

    # Evaluate with Answer Relevancy
    metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)
    result = metric.measure(test_case)

    print(f"\nMetric: {result.metric_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")


def example_g_eval_custom_criteria():
    """G-Eval with custom criteria example."""
    print("\n" + "=" * 60)
    print("Example 2: G-Eval with Custom Criteria")
    print("=" * 60)

    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    test_case = TestCase(
        input="Write a Python function to calculate factorial",
        actual_output="""
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
""",
        expected_output="A recursive or iterative function to calculate factorial",
    )

    # Custom criteria for code quality
    metric = GEval(
        name="Code Quality",
        criteria="""
        Evaluate the code quality based on:
        1. Correctness: Does the code correctly implement factorial?
        2. Readability: Is the code clean and easy to understand?
        3. Edge cases: Does it handle edge cases (0, 1)?
        """,
        evaluation_params=["input", "actual_output"],
        threshold=0.7,
        judge_model=judge,
    )

    result = metric.measure(test_case)

    print(f"\nMetric: {result.metric_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")


def example_g_eval_with_rubric():
    """G-Eval with scoring rubric example."""
    print("\n" + "=" * 60)
    print("Example 3: G-Eval with Rubric")
    print("=" * 60)

    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    test_case = TestCase(
        input="Explain quantum computing in simple terms",
        actual_output="Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously due to superposition. This allows quantum computers to process information in parallel.",
        expected_output="A simple explanation of quantum computing",
    )

    metric = GEval(
        name="Explanation Quality",
        criteria="Evaluate how well the explanation introduces quantum computing to a beginner",
        evaluation_params=["input", "actual_output"],
        rubric=[
            Rubric(
                score_range=(0, 3),
                description="Poor - too technical or confusing for beginners"
            ),
            Rubric(
                score_range=(4, 6),
                description="Fair - somewhat understandable but could be simpler"
            ),
            Rubric(
                score_range=(7, 8),
                description="Good - clear explanation with some technical terms"
            ),
            Rubric(
                score_range=(9, 10),
                description="Excellent - very clear, simple, and accessible explanation"
            ),
        ],
        threshold=0.7,
        judge_model=judge,
    )

    result = metric.measure(test_case)

    print(f"\nMetric: {result.metric_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")


def example_faithfulness():
    """Faithfulness metric example for RAG systems."""
    print("\n" + "=" * 60)
    print("Example 4: Faithfulness for RAG")
    print("=" * 60)

    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    test_case = TestCase(
        input="When was the Eiffel Tower built?",
        actual_output="The Eiffel Tower was built in 1889 for the World's Fair in Paris. It was designed by Gustave Eiffel.",
        retrieval_context=[
            "The Eiffel Tower was constructed from 1887 to 1889.",
            "It was built for the 1889 World's Fair (Exposition Universelle).",
            "The tower was designed by engineer Gustave Eiffel's company.",
        ],
    )

    metric = FaithfulnessMetric(threshold=0.8, judge_model=judge)
    result = metric.measure(test_case)

    print(f"\nMetric: {result.metric_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")

    # Show intermediate results
    if hasattr(metric, 'claims') and metric.claims:
        print(f"\nExtracted Claims:")
        for i, claim in enumerate(metric.claims, 1):
            print(f"  {i}. {claim}")

    if hasattr(metric, 'verdicts') and metric.verdicts:
        print(f"\nVerdicts:")
        for i, verdict in enumerate(metric.verdicts, 1):
            print(f"  {i}. {verdict.get('verdict', 'N/A')}")


async def example_async_evaluation():
    """Async evaluation example for multiple test cases."""
    print("\n" + "=" * 60)
    print("Example 5: Async Evaluation")
    print("=" * 60)

    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    # Multiple test cases
    test_cases = [
        TestCase(
            input="What is 2+2?",
            actual_output="2+2 equals 4.",
        ),
        TestCase(
            input="What is the speed of light?",
            actual_output="The speed of light is approximately 299,792,458 meters per second.",
        ),
        TestCase(
            input="Who wrote Romeo and Juliet?",
            actual_output="Romeo and Juliet was written by William Shakespeare.",
        ),
    ]

    metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

    # Evaluate all test cases asynchronously
    tasks = [metric.a_measure(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks)

    print("\nResults:")
    for i, (tc, result) in enumerate(zip(test_cases, results), 1):
        print(f"\n{i}. Question: {tc.input}")
        print(f"   Score: {result.score:.2f}")
        print(f"   Success: {result.success}")


def example_strict_mode():
    """G-Eval strict mode (binary scoring) example."""
    print("\n" + "=" * 60)
    print("Example 6: G-Eval Strict Mode")
    print("=" * 60)

    judge = OpenAIJudgeModel(
        model_name="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.0,
    )

    test_case = TestCase(
        input="Provide a code example",
        actual_output="""
Here's a Python code example:
```python
print("Hello, World!")
```
""",
    )

    metric = GEval(
        name="Contains Code",
        criteria="Check if the output contains executable code",
        evaluation_params=["actual_output"],
        strict_mode=True,  # Binary scoring: 0 or 1
        judge_model=judge,
    )

    result = metric.measure(test_case)

    print(f"\nMetric: {result.metric_name}")
    print(f"Score: {result.score:.0f} (binary: 0 or 1)")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")


def main():
    """Run all examples."""
    print("LLM-as-Judge Examples")
    print("=" * 60)
    print("Note: Replace 'your-api-key' with your actual API key")
    print("=" * 60)

    # Run synchronous examples
    # example_basic_usage()
    # example_g_eval_custom_criteria()
    # example_g_eval_with_rubric()
    # example_faithfulness()
    # example_strict_mode()

    # Run async example
    # asyncio.run(example_async_evaluation())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("Uncomment the example functions you want to run.")
    print("=" * 60)


if __name__ == "__main__":
    main()

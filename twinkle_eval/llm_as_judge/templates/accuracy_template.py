"""
Prompt templates for Accuracy metric.

Accuracy measures how factually correct and accurate the LLM output is
compared to the expected output (ground truth).
"""

import textwrap


class AccuracyTemplate:
    """Template for Accuracy evaluation prompts."""

    # System prompt for evaluation
    SYSTEM_PROMPT = """You are an AI evaluation assistant. Your task is to evaluate text responses based on factual accuracy compared to a reference answer. Always follow the instructions precisely and return only valid JSON. Do not include any additional text, explanations, or markdown formatting outside the JSON object."""

    @staticmethod
    def evaluate_accuracy(
        input_text: str,
        actual_output: str,
        expected_output: str,
        retrieval_context: str = None,
    ) -> str:
        """
        Generate prompt to evaluate factual accuracy.

        Args:
            input_text: The input question/prompt
            actual_output: The LLM's generated output
            expected_output: The expected/reference output (ground truth)
            retrieval_context: Optional context used for generation

        Returns:
            Prompt for evaluating accuracy
        """
        context_section = ""
        if retrieval_context:
            context_section = f"""
            Reference Context:
            {retrieval_context}
            """

        return textwrap.dedent(f"""
            Evaluate the factual accuracy of the answer compared to the expected output.

            Question/Input:
            {input_text}

            Expected Answer (Ground Truth):
            {expected_output}

            Actual Answer (To Evaluate):
            {actual_output}
            {context_section}
            Consider the following:
            1. Does the actual answer contain the same key facts as the expected answer?
            2. Are there any factual errors or contradictions?
            3. Is the information presented accurately?
            4. Does it capture the essential meaning even if worded differently?

            Provide an accuracy score from 0 to 10:
            - 0-2: Completely incorrect or contradicts the expected answer
            - 3-4: Contains significant factual errors
            - 5-6: Partially correct, some key information is wrong or missing
            - 7-8: Mostly correct, captures the main facts with minor differences
            - 9-10: Fully accurate, matches the expected answer in meaning

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (replace the values with your actual evaluation):
            {{"reason": "Your detailed explanation here", "score": 8}}

            Provide your evaluation now as a JSON object:
        """).strip()

"""
Prompt templates for Answer Relevancy metric.

Answer Relevancy measures how well the LLM output addresses the input question.
"""

import textwrap


class AnswerRelevancyTemplate:
    """Template for Answer Relevancy evaluation prompts."""

    # System prompt for evaluation - helps models like Gemma follow instructions
    SYSTEM_PROMPT = """You are an AI evaluation assistant. Your task is to evaluate text responses based on specific criteria and return your evaluation in JSON format. Always follow the instructions precisely and return only valid JSON. Do not include any additional text, explanations, or markdown formatting outside the JSON object."""

    @staticmethod
    def evaluate_relevancy(input_text: str, output_text: str) -> str:
        """
        Generate prompt to evaluate answer relevancy.

        Args:
            input_text: The input question/prompt
            output_text: The LLM's output

        Returns:
            Prompt for evaluating relevancy
        """
        return textwrap.dedent(f"""
            Evaluate how relevant the answer is to the question.

            Question/Input:
            {input_text}

            Answer/Output:
            {output_text}

            Consider the following:
            1. Does the answer directly address the question?
            2. Does the answer stay on topic?
            3. Does the answer provide useful information related to the question?
            4. Is there unnecessary or irrelevant information?

            Provide a relevancy score from 0 to 10:
            - 0-3: Not relevant, does not address the question
            - 4-6: Partially relevant, addresses some aspects
            - 7-8: Mostly relevant, addresses the question well
            - 9-10: Highly relevant, directly and comprehensively addresses the question

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (replace the values with your actual evaluation):
            {{"reason": "Your detailed explanation here", "score": 8}}

            Provide your evaluation now as a JSON object:
        """).strip()

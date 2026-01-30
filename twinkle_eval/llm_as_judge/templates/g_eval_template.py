"""
Prompt templates for G-Eval metric.

G-Eval is a framework for evaluating LLM outputs using LLMs with custom criteria.
Reference: https://arxiv.org/pdf/2303.16634.pdf
"""

import textwrap
from typing import List, Optional

from ..schemas import Rubric


class GEvalTemplate:
    """Template for G-Eval prompts."""

    # System prompt for evaluation - helps models like Gemma follow instructions
    SYSTEM_PROMPT = """You are an AI evaluation assistant. Your task is to evaluate text responses based on specific criteria and return your evaluation in JSON format. Always follow the instructions precisely and return only valid JSON. Do not include any additional text, explanations, or markdown formatting outside the JSON object."""

    @staticmethod
    def generate_evaluation_steps(
        criteria: str,
        evaluation_params: List[str],
    ) -> str:
        """
        Generate a prompt to create evaluation steps.

        Args:
            criteria: The evaluation criteria
            evaluation_params: List of parameters to use in evaluation

        Returns:
            Prompt for generating evaluation steps
        """
        params_str = ", ".join(evaluation_params)

        return textwrap.dedent(f"""
            Given the evaluation criteria:
            {criteria}

            You will evaluate using the following parameters:
            {params_str}

            Create a step-by-step evaluation plan. Each step should describe what to check or verify.
            The steps should be clear, specific, and actionable.

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.

            Required JSON format:
            {{"steps": ["Step 1 description", "Step 2 description", "Step 3 description"]}}

            Generate the evaluation steps now as a JSON object:
        """).strip()

    @staticmethod
    def generate_evaluation_results(
        criteria: str,
        steps: List[str],
        evaluation_params: List[str],
        test_case_params: dict,
        rubric: Optional[List[Rubric]] = None,
        strict_mode: bool = False,
    ) -> str:
        """
        Generate a prompt for evaluating based on steps.

        Args:
            criteria: The evaluation criteria
            steps: Evaluation steps to follow
            evaluation_params: Parameters used in evaluation
            test_case_params: Actual parameter values from test case
            rubric: Optional scoring rubric
            strict_mode: If True, use binary scoring (0 or 1)

        Returns:
            Prompt for evaluation
        """
        # Build steps text
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

        # Build test case parameters text
        params_text = ""
        for param_name in evaluation_params:
            value = test_case_params.get(param_name, "N/A")
            params_text += f"\n{param_name}:\n{value}\n"

        # Build rubric text
        if rubric:
            rubric_text = "\n\nScoring Rubric:\n"
            for r in rubric:
                min_score, max_score = r.score_range
                rubric_text += f"- Score {min_score}-{max_score}: {r.description}\n"
        else:
            rubric_text = ""

        # Build scoring instructions
        if strict_mode:
            score_instruction = textwrap.dedent("""
                Provide a score of either 0 (fail) or 1 (pass) based on whether the criteria is met.
            """).strip()
        else:
            score_instruction = textwrap.dedent("""
                Provide a score from 0 to 10, where:
                - 0-3: Poor quality
                - 4-6: Fair quality
                - 7-8: Good quality
                - 9-10: Excellent quality
            """).strip()

        prompt = textwrap.dedent(f"""
            Evaluation Criteria:
            {criteria}

            Evaluation Steps:
            {steps_text}
            {rubric_text}

            Data to Evaluate:
            {params_text}

            Instructions:
            1. Follow each evaluation step carefully
            2. Analyze the data against the criteria
            3. {score_instruction}
            4. Provide a clear reason explaining your score

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (replace the values with your actual evaluation):
            {{"reason": "Your detailed explanation here", "score": 7}}

            Provide your evaluation now as a JSON object:
        """).strip()

        return prompt

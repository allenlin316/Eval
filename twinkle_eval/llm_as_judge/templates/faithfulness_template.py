"""
Prompt templates for Faithfulness metric.

Faithfulness measures whether the LLM output is factually consistent with
the retrieval context (important for RAG systems).
"""

import textwrap
from typing import List


class FaithfulnessTemplate:
    """Template for Faithfulness evaluation prompts."""

    # System prompt for evaluation - helps models like Gemma follow instructions
    SYSTEM_PROMPT = """You are an AI evaluation assistant. Your task is to analyze text and extract information or make judgments based on specific criteria. Always follow the instructions precisely and return only valid JSON. Do not include any additional text, explanations, or markdown formatting outside the JSON object."""

    @staticmethod
    def extract_truths(context: List[str]) -> str:
        """
        Generate prompt to extract truths from context.

        Args:
            context: List of context strings

        Returns:
            Prompt for extracting truths
        """
        context_text = "\n".join(context)

        return textwrap.dedent(f"""
            Extract all factual statements and truths from the following context.
            Each truth should be a standalone statement that can be verified.

            Context:
            {context_text}

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (replace with actual truths):
            {{"truths": ["Truth 1", "Truth 2", "Truth 3"]}}

            Extract the truths now as a JSON object:
        """).strip()

    @staticmethod
    def extract_claims(output: str) -> str:
        """
        Generate prompt to extract claims from LLM output.

        Args:
            output: The LLM output to analyze

        Returns:
            Prompt for extracting claims
        """
        return textwrap.dedent(f"""
            Extract all factual claims from the following text.
            Each claim should be a standalone statement that can be verified.

            Text:
            {output}

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (replace with actual claims):
            {{"claims": ["Claim 1", "Claim 2", "Claim 3"]}}

            Extract the claims now as a JSON object:
        """).strip()

    @staticmethod
    def generate_verdicts(claims: List[str], truths: List[str]) -> str:
        """
        Generate prompt to verify claims against truths.

        Args:
            claims: List of claims to verify
            truths: List of truths from context

        Returns:
            Prompt for generating verdicts
        """
        claims_text = "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))
        truths_text = "\n".join(f"- {truth}" for truth in truths)

        return textwrap.dedent(f"""
            You will verify whether each claim is supported by the provided truths.

            Truths (from retrieval context):
            {truths_text}

            Claims to verify:
            {claims_text}

            For each claim, provide a verdict:
            - "yes" if the claim is fully supported by the truths
            - "no" if the claim contradicts the truths
            - "idk" if there's insufficient information to verify

            IMPORTANT: You MUST respond with ONLY a valid JSON object.
            Do not include any text, explanation, or markdown before or after the JSON.
            Do not use ```json``` code blocks.

            Required JSON format (one verdict per claim, in order):
            {{"verdicts": [{{"verdict": "yes", "reason": "Supported by truth X"}}, {{"verdict": "no", "reason": "Contradicts truth Y"}}]}}

            Provide your verdicts now as a JSON object:
        """).strip()

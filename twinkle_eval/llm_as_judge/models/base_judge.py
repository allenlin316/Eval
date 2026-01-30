"""
Base class for judge models used in LLM-as-Judge evaluations.

This module defines the abstract interface that all judge models must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class BaseJudgeModel(ABC):
    """
    Abstract base class for all judge models.

    A judge model is an LLM used to evaluate the quality of outputs from other LLMs.
    Subclasses must implement the generate methods to interact with their specific LLM API.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the judge model.

        Args:
            model_name: Name/identifier of the model
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Generate a response from the judge model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def a_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Asynchronously generate a response from the judge model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text response
        """
        pass

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_name

    def supports_structured_output(self) -> bool:
        """
        Check if the model supports structured output with schemas.

        Returns:
            True if structured output is supported
        """
        return False

    def supports_json_mode(self) -> bool:
        """
        Check if the model supports JSON mode.

        Returns:
            True if JSON mode is supported
        """
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

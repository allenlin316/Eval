"""
OpenAI-compatible judge model implementation.

This module provides a judge model that works with OpenAI API and compatible endpoints.
"""

import json
from typing import Any, Dict, List, Optional, Type

import httpx
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .base_judge import BaseJudgeModel


class OpenAIJudgeModel(BaseJudgeModel):
    """
    Judge model using OpenAI API or compatible endpoints.

    This implementation supports:
    - OpenAI's GPT models
    - Azure OpenAI
    - Any OpenAI-compatible API (e.g., vLLM, LM Studio, Ollama)
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        disable_ssl_verify: bool = False,
        timeout: int = 600,
        max_retries: int = 3,
        supports_structured_output: bool = False,
        supports_json_mode: bool = False,
        **kwargs
    ):
        """
        Initialize the OpenAI judge model.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            disable_ssl_verify: Whether to disable SSL verification
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            supports_structured_output: Whether the model supports structured output
            supports_json_mode: Whether the model supports JSON mode
            **kwargs: Additional parameters
        """
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._supports_structured_output = supports_structured_output
        self._supports_json_mode = supports_json_mode

        # Initialize HTTP client
        if disable_ssl_verify:
            httpx_client = httpx.Client(verify=False)
            async_httpx_client = httpx.AsyncClient(verify=False)
        else:
            httpx_client = httpx.Client()
            async_httpx_client = httpx.AsyncClient()

        # Initialize OpenAI clients
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx_client,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=async_httpx_client,
            max_retries=max_retries,
            timeout=timeout,
        )

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported."""
        return self._supports_structured_output

    def supports_json_mode(self) -> bool:
        """Check if JSON mode is supported."""
        return self._supports_json_mode

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages list for the API call."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_request_params(
        self,
        messages: List[Dict[str, str]],
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Build request parameters for the API call."""
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add schema if supported
        if schema is not None:
            if self.supports_structured_output():
                # Use structured output if available
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True,
                    }
                }
            elif self.supports_json_mode():
                # Fall back to JSON mode
                params["response_format"] = {"type": "json_object"}

        # Add any extra parameters
        params.update(self.extra_params)

        return params

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text
        """
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_request_params(messages, schema)

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    async def a_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """
        Asynchronously generate a response from the model.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            schema: Optional Pydantic schema for structured output

        Returns:
            Generated text
        """
        messages = self._build_messages(prompt, system_prompt)
        params = self._build_request_params(messages, schema)

        response = await self.async_client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

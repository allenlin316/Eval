"""
Google Gemini judge model implementation.

This module provides a judge model that works with Google's Gemini API.
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from .base_judge import BaseJudgeModel


class GoogleJudgeModel(BaseJudgeModel):
    """
    Judge model using Google Gemini API.

    This implementation supports:
    - Google's Gemini models (gemini-pro, gemini-1.5-pro, etc.)
    - Structured output via response_mime_type
    - Automatic rate limiting for free tier (5 requests/minute)
    - Smart retry with quota error handling
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 600,
        max_retries: int = 3,
        supports_structured_output: bool = False,
        supports_json_mode: bool = True,
        requests_per_minute: int = 5,  # Free tier limit
        **kwargs
    ):
        """
        Initialize the Google Gemini judge model.

        Args:
            model_name: Name of the model (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: Google API key for authentication
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            supports_structured_output: Whether the model supports structured output
            supports_json_mode: Whether the model supports JSON mode
            requests_per_minute: Rate limit (default 5 for free tier, set higher for paid)
            **kwargs: Additional parameters
        """
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._supports_structured_output = supports_structured_output
        self._supports_json_mode = supports_json_mode
        self.requests_per_minute = requests_per_minute

        # Rate limiting
        self._min_interval = 60.0 / requests_per_minute  # Minimum seconds between requests
        self._last_request_time = 0.0

        # Import google.generativeai here to avoid import errors if not installed
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Google Gemini support. "
                "Install it with: pip install google-generativeai"
            )

        # Configure the API
        self._genai.configure(api_key=api_key)

        # Initialize the model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add JSON mode if supported
        if self._supports_json_mode:
            generation_config["response_mime_type"] = "application/json"

        self.generation_config = generation_config
        self.model = self._genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported."""
        return self._supports_structured_output

    def supports_json_mode(self) -> bool:
        """Check if JSON mode is supported."""
        return self._supports_json_mode

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            print(f"  ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        self._last_request_time = time.time()

    async def _async_wait_for_rate_limit(self):
        """Async version of rate limit wait."""
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            print(f"  ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    def _extract_retry_delay(self, error_message: str) -> Optional[float]:
        """Extract retry delay from quota error message."""
        # Pattern: "Please retry in 58.234225955s"
        match = re.search(r'retry in ([\d.]+)s', str(error_message))
        if match:
            return float(match.group(1))
        return None

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build the full prompt with optional system prompt."""
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

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
        full_prompt = self._build_prompt(prompt, system_prompt)

        # Add schema hint to prompt if provided and structured output is supported
        if schema is not None and self._supports_structured_output:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            full_prompt += f"\n\nRespond with JSON matching this schema:\n{schema_json}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self._wait_for_rate_limit()

                response = self.model.generate_content(full_prompt)
                return response.text or ""

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if it's a quota error
                if "Quota exceeded" in error_str or "quota" in error_str.lower():
                    retry_delay = self._extract_retry_delay(error_str)
                    if retry_delay:
                        print(f"  ⚠️ Quota exceeded. Waiting {retry_delay:.1f}s before retry ({attempt + 1}/{self.max_retries + 1})...")
                        time.sleep(retry_delay + 1)  # Add 1 second buffer
                        continue
                    else:
                        # Default wait for quota errors
                        wait_time = 60
                        print(f"  ⚠️ Quota exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{self.max_retries + 1})...")
                        time.sleep(wait_time)
                        continue

                # For other errors, use exponential backoff
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ Error: {error_str[:100]}... Retrying in {wait_time}s ({attempt + 1}/{self.max_retries + 1})")
                    time.sleep(wait_time)

        # If all retries failed, raise the last error
        if last_error:
            raise last_error
        return ""

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
        full_prompt = self._build_prompt(prompt, system_prompt)

        # Add schema hint to prompt if provided and structured output is supported
        if schema is not None and self._supports_structured_output:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            full_prompt += f"\n\nRespond with JSON matching this schema:\n{schema_json}"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await self._async_wait_for_rate_limit()

                response = await self.model.generate_content_async(full_prompt)
                return response.text or ""

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if it's a quota error
                if "Quota exceeded" in error_str or "quota" in error_str.lower():
                    retry_delay = self._extract_retry_delay(error_str)
                    if retry_delay:
                        print(f"  ⚠️ Quota exceeded. Waiting {retry_delay:.1f}s before retry ({attempt + 1}/{self.max_retries + 1})...")
                        await asyncio.sleep(retry_delay + 1)  # Add 1 second buffer
                        continue
                    else:
                        # Default wait for quota errors
                        wait_time = 60
                        print(f"  ⚠️ Quota exceeded. Waiting {wait_time}s before retry ({attempt + 1}/{self.max_retries + 1})...")
                        await asyncio.sleep(wait_time)
                        continue

                # For other errors, use exponential backoff
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ Error: {error_str[:100]}... Retrying in {wait_time}s ({attempt + 1}/{self.max_retries + 1})")
                    await asyncio.sleep(wait_time)

        # If all retries failed, raise the last error
        if last_error:
            raise last_error
        return ""

"""
Utility functions for LLM-as-Judge evaluations.
"""

import json
import re
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

SchemaType = TypeVar("SchemaType", bound=BaseModel)
ReturnType = TypeVar("ReturnType")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that may contain markdown code blocks or other content.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON dict if found, None otherwise
    """
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON without code blocks
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_with_schema(
    text: str,
    schema_cls: Type[SchemaType],
) -> Optional[SchemaType]:
    """
    Parse text into a Pydantic schema.

    Args:
        text: Text to parse
        schema_cls: Pydantic schema class

    Returns:
        Parsed schema instance or None if parsing fails
    """
    json_data = extract_json_from_text(text)
    if json_data is None:
        return None

    try:
        return schema_cls(**json_data)
    except ValidationError:
        return None


def generate_with_schema_and_extract(
    text: str,
    schema_cls: Type[SchemaType],
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> Optional[ReturnType]:
    """
    Parse text with schema and extract the desired value.

    This function attempts to parse text into a Pydantic schema,
    then extracts the desired value using the provided callback.
    Falls back to JSON extraction if schema parsing fails.

    Args:
        text: LLM output text
        schema_cls: Pydantic schema class
        extract_schema: Function to extract value from schema instance
        extract_json: Function to extract value from raw JSON dict

    Returns:
        Extracted value or None if extraction fails
    """
    # Try parsing with schema
    schema_instance = parse_with_schema(text, schema_cls)
    if schema_instance is not None:
        try:
            return extract_schema(schema_instance)
        except (KeyError, AttributeError):
            pass

    # Fall back to JSON extraction
    json_data = extract_json_from_text(text)
    if json_data is not None:
        try:
            return extract_json(json_data)
        except (KeyError, AttributeError):
            pass

    return None


def trim_and_load_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Trim text and attempt to load as JSON.

    Args:
        text: Text to parse

    Returns:
        Parsed JSON dict or None
    """
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return extract_json_from_text(text)


def calculate_score(verdicts: list, total: int) -> float:
    """
    Calculate a score based on verdicts.

    Args:
        verdicts: List of verdict objects or strings
        total: Total number of items

    Returns:
        Score between 0 and 1
    """
    if total == 0:
        return 1.0

    success_count = 0
    for verdict in verdicts:
        if isinstance(verdict, dict):
            verdict_value = verdict.get("verdict", "").lower()
        elif hasattr(verdict, "verdict"):
            verdict_value = verdict.verdict.lower()
        else:
            verdict_value = str(verdict).lower()

        if verdict_value == "yes":
            success_count += 1

    return success_count / total

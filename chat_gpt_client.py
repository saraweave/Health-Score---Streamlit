"""Simple client for interacting with the ChatGPT API."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"


class ChatGPTClient:
    """Wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Set `OPENAI_API_KEY` or pass `api_key`."
            )
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: Iterable[dict[str, str]],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat conversation and return the assistant response text."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""


def chat_once(prompt: str, system: Optional[str] = None) -> str:
    """Quick helper for single-turn interactions."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = ChatGPTClient()
    return client.chat(messages)


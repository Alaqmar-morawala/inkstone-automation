"""Model inference client supporting OpenAI and Anthropic compatible protocols."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from .auth import AuthManager
from .config import InkStoneConfig


class InferenceClient:
    """Dispatches chat completions to Intern InkStone's hosted frontier models."""

    def __init__(self, config: Optional[InkStoneConfig] = None, auth: Optional[AuthManager] = None):
        self.config = config or InkStoneConfig.load()
        self.auth = auth or AuthManager(self.config)

    def chat(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Send a chat completion request using the OpenAI-compatible endpoint."""
        target_model = model or self.config.default_model

        # Format prompt if single string passed
        formatted_messages: List[Dict[str, Any]] = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        if isinstance(messages, str):
            formatted_messages.append({"role": "user", "content": messages})
        else:
            formatted_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if thinking_budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        url = f"{self.config.api_base_url}/chat/completions"
        headers = self.auth.get_inference_headers(protocol="openai")

        if stream:
            return self._stream_chat(url, headers, payload)
        else:
            return self._sync_chat(url, headers, payload)

    def _sync_chat(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return {"content": "", "raw": data}

        first_choice = choices[0]
        message = first_choice.get("message", {})

        return {
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "model": data.get("model", payload["model"]),
            "finish_reason": first_choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "raw": data,
        }

    def _stream_chat(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Generator[str, None, None]:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue

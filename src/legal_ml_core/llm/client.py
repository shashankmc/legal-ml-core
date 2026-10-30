"""The one LLM adapter: an OpenAI-compatible ``/chat/completions`` client.

That covers OpenRouter, llama.cpp, vLLM and most hosted providers. It sends no
tools and doesn't rely on ``response_format``, which servers support unevenly;
the output is validated by prompts.parse_suggestion instead.
"""

import requests

from .config import int_setting


class LLMError(Exception):
    """A provider failure, with a message that is safe to show the user."""


def estimate_tokens(text):
    return max(1, len(text) // 4)


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_body = extra_body or {}

    def chat(self, messages):
        """Return ``(content, total_tokens)``. Provider bodies are never surfaced."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "max_tokens": int_setting("LLM_MAX_OUTPUT_TOKENS"),
            "temperature": 0,
        }
        try:
            # Redirects stay off so a redirect can't carry the key to another host.
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=int_setting("LLM_TIMEOUT_SECONDS"),
                allow_redirects=False,
            )
        except requests.RequestException:
            raise LLMError("The LLM service could not be reached.")

        if response.status_code in (401, 403):
            raise LLMError("The LLM service rejected the API key.")
        if response.status_code == 429:
            raise LLMError("The LLM service is rate limiting this key. Try again later.")
        if response.status_code != 200:
            raise LLMError("The LLM service returned an error.")

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("The LLM service returned an unreadable response.")
        if not isinstance(content, str):
            raise LLMError("The LLM service returned an unreadable response.")

        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens")
        if not isinstance(tokens, int):
            prompt = "".join(m["content"] for m in messages)
            tokens = estimate_tokens(prompt) + estimate_tokens(content)
        return content, tokens

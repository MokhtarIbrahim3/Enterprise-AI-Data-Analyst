
"""
Thin, provider-agnostic LLM wrapper.

The team has not locked an LLM vendor yet, so the agent depends on this
interface instead of on any SDK. Switching provider = changing one env var.

Supported values of ``LLM_PROVIDER``:

  openai      — any OpenAI-compatible endpoint (OpenAI, Groq, Together,
                NVIDIA NIM, OpenRouter, local vLLM). Set LLM_BASE_URL too.
  gemini      — Google Generative AI.
  rule        — no network: deterministic stub used by unit tests, CI and the
                offline Docker build. The agent degrades gracefully (rule-based
                routing + templated answers) instead of crashing.
"""

from __future__ import annotations

import os
import time


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete(self, prompt: str, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 800) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI  # lazy import

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self.model_name = model
        self._genai = genai

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        model = self._genai.GenerativeModel(
            self.model_name, system_instruction=system
        )
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return (response.text or "").strip()


class RuleBasedLLM(BaseLLM):
    """No-network stub. Returns empty text so callers use their fallbacks."""

    name = "rule"

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=800) -> str:
        return ""


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseLLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "rule")).lower()
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    base_url = base_url or os.getenv("LLM_BASE_URL") or None

    if provider in {"rule", "none", "offline"} or not api_key:
        if provider not in {"rule", "none", "offline"}:
            print("[llm] No LLM_API_KEY found — falling back to rule-based mode.")
        return RuleBasedLLM()

    try:
        if provider == "gemini":
            return GeminiLLM(model, api_key)
        return OpenAICompatibleLLM(model, api_key, base_url)
    except Exception as exc:
        print(f"[llm] Could not initialise provider '{provider}' ({exc}); using rule mode.")
        return RuleBasedLLM()


def timed_complete(llm: BaseLLM, *args, **kwargs) -> tuple[str, float]:
    start = time.perf_counter()
    text = llm.complete(*args, **kwargs)
    return text, round((time.perf_counter() - start) * 1000, 2)

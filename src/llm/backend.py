"""
LLM backend abstraction.

Wraps different model providers behind a single `generate_response`
interface so the agent code doesn't care whether it's talking to a local
GGUF model via llama-cpp, or a hosted API (OpenAI / Anthropic / etc).

Currently implemented:
    - LlamaCppBackend: local GGUF models via llama-cpp-python

Planned extension points (see TODOs):
    - OpenAIBackend
    - AnthropicBackend
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Common interface every backend must implement."""

    @abstractmethod
    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """
        Run a chat completion and return the assistant's text response.

        Args:
            messages: list of {"role": ..., "content": ...} dicts, e.g.
                [{"role": "system", "content": ...},
                 {"role": "user", "content": ...}]
        """
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """Local GGUF model backend, loaded via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        verbose: bool = False,
    ):
        # Imported lazily so the rest of the codebase doesn't require
        # llama-cpp-python to be installed unless this backend is used.
        from llama_cpp import Llama

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop or ["<|eot_id|>", "<|end_of_text|>"]

        self._model = Llama(
            model_path,
            verbose=verbose,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        output: dict[str, Any] = self._model.create_chat_completion(
            messages,
            stop=self.stop,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return output["choices"][0]["message"]["content"]


# TODO(multi-backend): implement additional backends, e.g.
#
# class OpenAIBackend(LLMBackend):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         from openai import OpenAI
#         self._client = OpenAI()
#         self._model = model
#
#     def generate_response(self, messages):
#         resp = self._client.chat.completions.create(
#             model=self._model, messages=messages
#         )
#         return resp.choices[0].message.content
#
# class AnthropicBackend(LLMBackend):
#     def __init__(self, model: str = "claude-sonnet-4-6"):
#         import anthropic
#         self._client = anthropic.Anthropic()
#         self._model = model
#
#     def generate_response(self, messages):
#         system = next((m["content"] for m in messages if m["role"] == "system"), None)
#         user_messages = [m for m in messages if m["role"] != "system"]
#         resp = self._client.messages.create(
#             model=self._model, system=system, messages=user_messages, max_tokens=1000
#         )
#         return resp.content[0].text
#
# Having this interface in place lets you swap backends purely via config,
# which is a nice thing to point to in an interview ("cost/latency vs
# quality tradeoffs between local and hosted models").

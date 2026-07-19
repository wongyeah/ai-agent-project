"""
LLM backend abstraction.

Wraps different model providers behind a single `generate_response`
interface so the agent code doesn't care whether it's talking to a local
GGUF model via llama-cpp, or a hosted API (OpenAI / Anthropic / etc).

Currently implemented:
    - LlamaCppBackend: local GGUF models via llama-cpp-python
    - OpenAIBackend: hosted OpenAI API
    - AnthropicBackend: hosted Anthropic (Claude) API
    - CozeBackend: hosted Coze (扣子, ByteDance) bot API — architecturally
      different from the other two (see its class docstring): it calls a
      pre-built "bot" rather than a raw model completion endpoint.
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


class OpenAIBackend(LLMBackend):
    """
    Hosted LLM backend using the OpenAI API.

    Requires the OPENAI_API_KEY environment variable to be set (the
    OpenAI client reads it automatically — never hardcode API keys in
    code or commit them to git).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        # Imported lazily so the rest of the codebase doesn't require the
        # `openai` package to be installed unless this backend is used.
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """
    Hosted LLM backend using the Anthropic API (Claude).

    Requires the ANTHROPIC_API_KEY environment variable to be set (the
    Anthropic client reads it automatically — never hardcode API keys in
    code or commit them to git). This is a separate credential from any
    claude.ai chat subscription (Free/Pro/Max) — the API is billed
    independently (pay-as-you-go against credits purchased in the
    Anthropic Console), and a chat subscription does not grant API access
    or vice versa.

    Kept deliberately structurally parallel to OpenAIBackend above (same
    constructor shape, same generate_response signature) so swapping
    `llm.backend` in configs/*.yaml is the only change needed anywhere —
    that's the whole point of the LLMBackend abstraction.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        # Imported lazily so the rest of the codebase doesn't require the
        # `anthropic` package to be installed unless this backend is used.
        from anthropic import Anthropic

        self._client = Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        # Unlike OpenAI's Chat Completions API, Anthropic's Messages API
        # takes the system prompt as its own top-level `system` argument
        # rather than as a role inside `messages` — so it has to be pulled
        # out here rather than passed straight through like OpenAIBackend
        # does.
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=user_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        if system is not None:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text


class CozeBackend(LLMBackend):
    """
    Hosted LLM backend using Coze (扣子, ByteDance's AI agent platform).

    This one is architecturally different from OpenAIBackend/
    AnthropicBackend above, not just a different vendor: Coze has no "give
    me a completion for these messages" endpoint. You build and publish a
    *bot* first, in the Coze web UI (that's where the underlying model,
    e.g. Doubao/DeepSeek/etc, and the bot's default persona/system prompt
    get chosen), and the API just sends user turns to that already-
    configured bot and gets replies back. There is no per-call `system`
    parameter the way OpenAI/Anthropic have one — so the distinct system
    prompt this project generates for each draft/debug/improve/reflect
    call (see src/agent/agent.py) can't be passed the "proper" way here;
    it's folded into the front of the single outgoing user message
    instead. That's a real fidelity trade-off versus a true system role,
    not just a cosmetic difference — worth calling out explicitly if you
    write this up for an interview.

    Requires two environment variables (see .env.example):
        - COZE_API_TOKEN: a personal access token from
          https://www.coze.cn/open/oauth/pats (not an OpenAI/Anthropic-
          style secret key — different auth model).
        - COZE_BOT_ID (or pass `bot_id` directly / via `llm.model` in the
          config): the ID of a bot you created and published as an API
          service in the Coze web UI — visible in that bot's URL.

    Cost note: Coze's personal free tier caps out at a small, ONE-TIME
    cumulative number of API calls (check the current number in your own
    Coze console — it is not a per-day/per-month allowance; once used up,
    the account can't call the API at all until you add a paid plan).
    Budget accordingly: each Agent.step() burns several LLM calls, so a
    full multi-dataset eval sweep can exhaust a small free quota within a
    single run. That said, main.py's save_run() writes the journal to
    disk after every single step, so a run that gets cut off mid-way
    still leaves you a partial, genuinely usable result — it isn't wasted
    if you hit the wall.
    """

    def __init__(
        self,
        bot_id: str | None = None,
        max_tokens: int = 4096,  # noqa: ARG002 - accepted for interface parity; Coze bots don't take a per-call token cap
        temperature: float = 0.0,  # noqa: ARG002 - accepted for interface parity; Coze bots don't take a per-call temperature
    ):
        import os

        # Imported lazily so the rest of the codebase doesn't require the
        # `cozepy` package to be installed unless this backend is used.
        from cozepy import COZE_CN_BASE_URL, ChatStatus, Coze, Message, MessageType, TokenAuth

        api_token = os.environ.get("COZE_API_TOKEN")
        if not api_token:
            raise RuntimeError(
                "COZE_API_TOKEN is not set. Get a personal access token "
                "from https://www.coze.cn/open/oauth/pats and put it in "
                "your .env (see .env.example)."
            )

        self._bot_id = bot_id or os.environ.get("COZE_BOT_ID")
        if not self._bot_id:
            raise RuntimeError(
                "No Coze bot_id configured. Create a bot at coze.cn, "
                "publish it as an API service, then either set llm.model "
                "in your config.yaml to that bot's ID, or set COZE_BOT_ID "
                "in your .env."
            )

        # Stashed for use in generate_response without re-importing.
        self._Message = Message
        self._MessageType = MessageType
        self._ChatStatus = ChatStatus
        self._client = Coze(auth=TokenAuth(api_token), base_url=COZE_CN_BASE_URL)

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        # No system role on this API (see class docstring) — fold every
        # system message into the front of one combined user turn.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        other_parts = [m["content"] for m in messages if m["role"] != "system"]
        combined = "\n\n".join(system_parts + other_parts)

        chat_poll = self._client.chat.create_and_poll(
            bot_id=self._bot_id,
            user_id="ai-agent-project",
            additional_messages=[self._Message.build_user_question_text(combined)],
        )

        if chat_poll.chat.status != self._ChatStatus.COMPLETED:
            raise RuntimeError(
                f"Coze chat did not complete successfully "
                f"(status={chat_poll.chat.status}). Common causes: the "
                f"bot_id is wrong, the bot isn't published as an API "
                f"service yet, or your account's free API call quota is "
                f"exhausted."
            )

        # A completed chat's messages can include more than just the
        # final answer (e.g. VERBOSE/FOLLOW_UP entries) if the bot has
        # extra features enabled — filter to ANSWER-type chunks only and
        # join them, since a single answer can occasionally arrive as
        # more than one chunk.
        answer_parts = [
            m.content for m in chat_poll.messages if m.type == self._MessageType.ANSWER
        ]
        return "".join(answer_parts)

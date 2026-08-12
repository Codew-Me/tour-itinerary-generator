"""LangGraph agent graph construction."""

from __future__ import annotations

import urllib.request

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import SYSTEM_PROMPT
from src.config import get_settings
from src.tools import ALL_TOOLS


def _ollama_available(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_llm_config(prefer_ollama: bool = False) -> tuple[str, str]:
    settings = get_settings()

    if prefer_ollama and _ollama_available(settings.ollama_base_url):
        return "ollama", settings.ollama_model

    provider = settings.llm_provider.lower()

    if provider == "openai":
        if settings.openai_api_key.strip():
            return "openai", settings.llm_model
        if _ollama_available(settings.ollama_base_url):
            return "ollama", settings.ollama_model
        raise ValueError("Set OPENAI_API_KEY or start Ollama.")

    if provider == "ollama":
        return "ollama", settings.ollama_model or settings.llm_model

    if provider == "anthropic":
        if not settings.anthropic_api_key.strip():
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        return "anthropic", settings.llm_model

    raise ValueError("Unsupported LLM provider.")


def get_llm(provider: str | None = None, model: str | None = None):
    settings = get_settings()
    if provider is None or model is None:
        provider, model = resolve_llm_config()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0.2)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=0.2)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0.2)


_memory = MemorySaver()
_agents: dict[str, object] = {}


def _build_agent(provider: str, model: str):
    llm = get_llm(provider, model)
    return create_react_agent(
        llm,
        ALL_TOOLS,
        checkpointer=_memory,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


def get_agent(prefer_ollama: bool = False):
    provider, model = resolve_llm_config(prefer_ollama=prefer_ollama)
    key = f"{provider}:{model}"
    if key not in _agents:
        _agents[key] = _build_agent(provider, model)
    return _agents[key], provider, model


def invoke_agent(message: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    agent, provider, model = get_agent()

    try:
        return agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    except Exception as exc:
        err = str(exc).lower()
        if provider == "openai" and _ollama_available(get_settings().ollama_base_url):
            if any(k in err for k in ("429", "quota", "credit", "insufficient", "billing")):
                agent, _, _ = get_agent(prefer_ollama=True)
                return agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        raise


def get_active_llm_info() -> dict:
    provider, model = resolve_llm_config()
    return {"provider": provider, "model": model}

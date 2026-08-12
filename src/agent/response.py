"""Extract clean assistant text from LangGraph agent results."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def _is_tool_call_json(text: str) -> bool:
    t = text.strip()
    if not t.startswith("{"):
        return False
    return '"name"' in t and ('"parameters"' in t or '"arguments"' in t)


def _content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def extract_assistant_response(messages: list[BaseMessage]) -> str | None:
    """Return the last real assistant reply, skipping tool-call JSON leaks."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _content_to_str(msg.content)
            if text and not _is_tool_call_json(text):
                return text
            # Proper tool_calls on AIMessage — keep looking for final synthesis
            if msg.tool_calls and not text:
                continue
        elif isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            if role in ("assistant", "ai"):
                text = _content_to_str(msg.get("content"))
                if text and not _is_tool_call_json(text):
                    return text
    return None


def extract_tool_results(messages: list[BaseMessage]) -> list[dict]:
    """Collect tool outputs for fallback formatting."""
    results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                results.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                results.append({"raw": str(msg.content)})
        elif isinstance(msg, dict) and msg.get("role") == "tool":
            try:
                results.append(json.loads(msg.get("content", "")))
            except (json.JSONDecodeError, TypeError):
                results.append({"raw": str(msg.get("content", ""))})
    return results

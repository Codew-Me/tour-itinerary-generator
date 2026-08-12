"""Orchestrated travel agent entry point (replaces raw chatbot dispatch)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.services.agent_orchestrator import AgentOrchestrator
from src.services.conversation_state import ConversationState


class ChatService:
    """Facade for the itinerary planning agent."""

    def __init__(self, session: Session):
        self.session = session
        self.agent = AgentOrchestrator(session)

    def handle_message(
        self,
        message: str,
        history: list[dict] | None = None,
        state: ConversationState | None = None,
    ) -> dict:
        return self.agent.run_turn(message, history=history, state=state)

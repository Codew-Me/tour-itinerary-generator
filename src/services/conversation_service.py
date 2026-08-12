"""Conversation persistence service."""

import json
from datetime import datetime

from sqlalchemy.orm import Session

from src.database.models import ConversationSession, Message
from src.services.conversation_state import ConversationState


class ConversationService:
    def __init__(self, session: Session):
        self.session = session

    def create_session(self, user_id: int, title: str = "New conversation") -> ConversationSession:
        conv = ConversationSession(user_id=user_id, title=title)
        self.session.add(conv)
        self.session.flush()
        return conv

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(msg)
        conv = self.session.get(ConversationSession, conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()
        self.session.flush()
        return msg

    def list_sessions(self, user_id: int) -> list[dict]:
        sessions = (
            self.session.query(ConversationSession)
            .filter(ConversationSession.user_id == user_id)
            .order_by(ConversationSession.updated_at.desc())
            .all()
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    def get_session(self, user_id: int, conversation_id: int) -> dict | None:
        conv = (
            self.session.query(ConversationSession)
            .filter(
                ConversationSession.id == conversation_id,
                ConversationSession.user_id == user_id,
            )
            .first()
        )
        if not conv:
            return None
        messages = [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in conv.messages
        ]
        return {"id": conv.id, "title": conv.title, "messages": messages}

    def get_conversation_state(self, conversation_id: int) -> ConversationState:
        """Load structured state from the latest assistant message metadata."""
        last = (
            self.session.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
            )
            .order_by(Message.created_at.desc())
            .first()
        )
        if last and last.metadata_json:
            try:
                meta = json.loads(last.metadata_json)
                if isinstance(meta, dict) and "state" in meta:
                    return ConversationState.from_dict(meta["state"])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return ConversationState()

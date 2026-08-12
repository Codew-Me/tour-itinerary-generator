"""FastAPI application."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.planning_flow import start_category_planning
from src.services.preferences import normalize_category
from src.auth.service import authenticate_user, create_access_token, decode_token, register_user
from src.database.models import Attraction, SavedDestination, User
from src.database.postgres import get_db, init_db
from src.database.repositories import AttractionRepository
from src.services.conversation_service import ConversationService
from src.vectorstore.chroma_store import get_chroma_store

app = FastAPI(title="Sri Lanka Travel Recommendation Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    full_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    planning_category: Optional[str] = None  # UI tab selection — confirmed preference
    start_planning: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: int
    thread_id: str
    action: Optional[str] = None
    intent: Optional[str] = None
    agent_phase: Optional[str] = None


class CompareRequest(BaseModel):
    destination1: str
    destination2: str


class SaveDestinationRequest(BaseModel):
    notes: Optional[str] = None


# ---------- Auth dependency ----------


def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    db: Session = Depends(get_db),
) -> User | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        return None
    user = db.get(User, int(payload["sub"]))
    return user


def require_user(user: User | None = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ---------- Startup ----------


@app.on_event("startup")
def startup():
    init_db()


# ---------- Endpoints ----------


@app.get("/health")
def health():
    try:
        chroma_count = get_chroma_store().count()
    except Exception:
        chroma_count = 0
    try:
        from src.agent.graph import get_active_llm_info

        llm_info = get_active_llm_info()
    except Exception as e:
        llm_info = {"error": str(e)}
    return {"status": "ok", "chroma_review_count": chroma_count, "llm": llm_info}


@app.get("/dataset/categories")
def dataset_categories(db: Session = Depends(get_db)):
    """Trip theme tabs — categories and counts from the loaded attractions dataset."""
    from src.services.preferences import VALID_CATEGORIES

    repo = AttractionRepository(db)
    stats, total = repo.list_category_stats()
    descriptions = {
        "Wild": "National parks & wildlife",
        "Heritage": "Temples & ancient sites",
        "Scenic": "Views & landscapes",
        "Pristine": "Beaches & coastlines",
        "Essence": "Culture & local life",
        "Thrills": "Adventure & adrenaline",
    }
    categories = [
        {
            "name": row["name"],
            "count": row["count"],
            "description": descriptions.get(row["name"], ""),
            "in_dataset": row["name"] in VALID_CATEGORIES,
        }
        for row in stats
        if row["name"] in VALID_CATEGORIES
    ]
    return {
        "categories": categories,
        "total_attractions": total,
        "source": "attractions_dataset",
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = register_user(db, body.email, body.password, body.full_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_access_token(user.id, user.email)
    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.id, user.email)
    return AuthResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    conv_service = ConversationService(db)
    if body.conversation_id:
        conv = conv_service.get_session(user.id, body.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = body.conversation_id
    else:
        title = body.message[:60] + ("..." if len(body.message) > 60 else "")
        session = conv_service.create_session(user.id, title=title)
        conversation_id = session.id

    thread_id = f"user-{user.id}-conv-{conversation_id}"
    conv_service.add_message(conversation_id, "user", body.message)

    # Load conversation history (excluding current message just saved)
    conv = conv_service.get_session(user.id, conversation_id)
    history: list[dict] = []
    if conv and len(conv.get("messages", [])) > 1:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in conv["messages"][:-1]
            if m["role"] in ("user", "assistant")
        ]

    conversation_state = conv_service.get_conversation_state(conversation_id)

    # Explicit category from UI tab — treat as confirmed planning preference
    if body.planning_category or body.start_planning:
        cat = normalize_category(body.planning_category) or normalize_category(
            body.message.replace("__start_planning__:", "").strip()
        )
        if cat:
            start_category_planning(conversation_state, cat)

    chat_service = ChatService(db)
    result = chat_service.handle_message(
        body.message,
        history=history,
        state=conversation_state,
    )
    assistant_msg = result["response"]

    conv_service.add_message(
        conversation_id,
        "assistant",
        assistant_msg,
        metadata={"state": result.get("state")},
    )
    return ChatResponse(
        response=assistant_msg,
        conversation_id=conversation_id,
        thread_id=thread_id,
        action=result.get("action"),
        intent=result.get("intent"),
        agent_phase=result.get("agent_phase"),
    )


@app.get("/conversations")
def list_conversations(user: User = Depends(require_user), db: Session = Depends(get_db)):
    return ConversationService(db).list_sessions(user.id)


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    conv = ConversationService(db).get_session(user.id, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/destinations/{destination}")
def get_destination(destination: str, db: Session = Depends(get_db)):
    repo = AttractionRepository(db)
    summary = repo.get_destination_summary(destination)
    if not summary:
        raise HTTPException(status_code=404, detail="Destination not found")

    store = get_chroma_store()
    reviews = store.search(query=destination, k=8)
    summary["traveler_insights"] = reviews[:5]
    return summary


@app.get("/districts/{district}")
def get_district(district: str, db: Session = Depends(get_db)):
    repo = AttractionRepository(db)
    items = repo.list_by_district(district)
    return {"district": district, "attractions": items, "count": len(items)}


@app.post("/compare")
def compare(body: CompareRequest, db: Session = Depends(get_db)):
    from src.tools import compare_destinations

    result = json.loads(compare_destinations.invoke({
        "destination1": body.destination1,
        "destination2": body.destination2,
    }))
    return result


@app.post("/destinations/{attraction_id}/save")
def save_destination(
    attraction_id: int,
    body: SaveDestinationRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    attraction = db.get(Attraction, attraction_id)
    if not attraction:
        raise HTTPException(status_code=404, detail="Attraction not found")
    existing = (
        db.query(SavedDestination)
        .filter(SavedDestination.user_id == user.id, SavedDestination.attraction_id == attraction_id)
        .first()
    )
    if existing:
        return {"message": "Already saved", "id": existing.id}
    saved = SavedDestination(user_id=user.id, attraction_id=attraction_id, notes=body.notes)
    db.add(saved)
    db.flush()
    return {"message": "Saved", "id": saved.id}


@app.delete("/destinations/{attraction_id}/save")
def unsave_destination(
    attraction_id: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    saved = (
        db.query(SavedDestination)
        .filter(SavedDestination.user_id == user.id, SavedDestination.attraction_id == attraction_id)
        .first()
    )
    if not saved:
        raise HTTPException(status_code=404, detail="Saved destination not found")
    db.delete(saved)
    return {"message": "Removed"}


@app.get("/saved-destinations")
def list_saved(user: User = Depends(require_user), db: Session = Depends(get_db)):
    saved = (
        db.query(SavedDestination)
        .filter(SavedDestination.user_id == user.id)
        .all()
    )
    repo = AttractionRepository(db)
    results = []
    for s in saved:
        item = repo.get_by_name(s.attraction.name)
        if item:
            item["saved_at"] = s.saved_at.isoformat()
            item["notes"] = s.notes
            results.append(item)
    return results

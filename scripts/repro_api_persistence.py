"""Simulate API persistence: state loaded from last assistant metadata each turn."""

from src.database.postgres import init_db, get_session_factory
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState
from src.services.conversation_service import ConversationService
from src.services.planning_flow import start_category_planning

init_db()
db = get_session_factory()()
conv_svc = ConversationService(db)
chat = ChatService(db)

# Fake user id 1 - create session
session = conv_svc.create_session(user_id=1, title="Wild trip test")
conv_id = session.id

messages = [
    ("__start_planning__:Wild", {"planning_category": "Wild"}),
    ("8", {}),
    ("seeduwa", {}),
    ("strt from Seeduwa", {}),
]

for msg, extra in messages:
    conv_svc.add_message(conv_id, "user", msg)
    conv = conv_svc.get_session(1, conv_id)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in conv["messages"][:-1]
        if m["role"] in ("user", "assistant")
    ]
    state = conv_svc.get_conversation_state(conv_id)

    if extra.get("planning_category"):
        start_category_planning(state, extra["planning_category"])

    result = chat.handle_message(msg, history=history, state=state)
    conv_svc.add_message(conv_id, "assistant", result["response"], metadata={"state": result.get("state")})

    loaded = conv_svc.get_conversation_state(conv_id)
    print("MSG:", msg)
    print("  response:", result["response"][:90].encode("ascii", "replace").decode())
    print("  saved start:", loaded.starting_location, "step:", loaded.current_planning_step)
    print("  duration:", loaded.duration_days, "travellers:", loaded.travellers)
    print()

db.close()

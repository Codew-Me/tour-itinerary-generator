from src.services.conversation_state import ConversationState
from src.services.chat_service import ChatService
from src.database.postgres import init_db, get_session_factory

init_db()
session = get_session_factory()()
svc = ChatService(session)
history = []
state = None

for msg in ["hi", "plan a 3 day tour", "i dont like it"]:
    r = svc.handle_message(msg, history=history, state=state)
    state = ConversationState.from_dict(r["state"])
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": r["response"]})
    text = r["response"][:120].encode("ascii", "replace").decode()
    print(msg, "->", r["action"], r.get("intent"))
    print(" ", text)
    print("  start:", state.starting_location, "itin:", bool(state.current_itinerary))
    print()

session.close()

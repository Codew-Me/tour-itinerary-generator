"""Simulate the user's test conversation and print results."""

from unittest.mock import patch

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState

init_db()

TURNS = [
    "hi",
    "im happy",
    "im bored",
    "relax",
    "beach",
    "suggest places",
    "no u suggest an generate me an itinerary",
    "3 days",
]

session = get_session_factory()()
history: list = []
state = None

print("=" * 60)
print("CONVERSATION SIMULATION")
print("=" * 60)

with patch("src.services.chat_service.synthesize_response") as mock_synth, patch(
    "src.services.chat_service.generate_itinerary"
) as mock_itin:
    mock_synth.side_effect = lambda msg, hist, rec: (
        "🌴 Here are some relaxing beach options from our dataset:\n\n"
        + "\n".join(
            f"### {i+1}. {c.get('name')} — {c.get('district')}"
            for i, c in enumerate(rec.get("candidates", [])[:4])
        )
        if rec.get("candidates")
        else "No matches found."
    )
    mock_itin.side_effect = lambda st, rec, hist=None: (
        f"## {st.duration_days or 3}-Day Relaxing Beach Itinerary\n\n"
        + "\n".join(
            f"### Day {d+1}\n- Visit {c.get('name')} ({c.get('district')})"
            for d, c in enumerate(rec.get("candidates", [])[: st.duration_days or 3])
        )
    )

    svc = ChatService(session)
    for msg in TURNS:
        print(f"\nUSER: {msg}")
        result = svc.handle_message(msg, history=history, state=state)
        print(f"ASSISTANT [{result['action']}]: {result['response'][:500]}")
        if len(result["response"]) > 500:
            print("...")
        history = history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": result["response"]},
        ]
        state = ConversationState.from_dict(result["state"])
        print(f"STATE: {result['state']}")

session.close()
print("\n" + "=" * 60)

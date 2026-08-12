"""Test exact multi-turn conversation: plan tour → adventure → beach → yh → choose and plan 3 days."""

from unittest.mock import patch

from src.database.postgres import get_session_factory, init_db
from src.services.chat_service import ChatService
from src.services.conversation_state import ConversationState

init_db()

TURNS = [
    "plan a tour for me",
    "adventure",
    "beach",
    "yh",
    "choose some and plan a 3 day trip",
]


def run_conversation():
    session = get_session_factory()()
    history: list = []
    state = None
    results = []

    with patch("src.services.chat_service.synthesize_response") as mock_synth:
        mock_synth.side_effect = lambda msg, hist, rec: (
            "🌴 Here are some places you might like:\n\n"
            + "\n".join(
                f"### {i+1}. {c.get('name')}\n📍 {c.get('district')}"
                for i, c in enumerate(rec.get("candidates", [])[:6])
            )
        )

        svc = ChatService(session)
        for msg in TURNS:
            result = svc.handle_message(msg, history=history, state=state)
            results.append((msg, result))
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": result["response"]},
            ]
            state = ConversationState.from_dict(result["state"])

    session.close()
    return results


if __name__ == "__main__":
    results = run_conversation()
    print("=" * 60)
    for msg, result in results:
        print(f"\nUSER: {msg}")
        print(f"ACTION: {result['action']}")
        print(f"RESPONSE (first 400 chars): {result['response'][:400]}")
        if result["action"] == "generate_itinerary":
            print("STATE last_recommendations count:", len(result["state"].get("last_recommendations", [])))
    print("\n" + "=" * 60)
    final = results[-1][1]
    assert final["action"] == "generate_itinerary", f"Expected itinerary, got {final['action']}"
    assert "tell me what kind of trip" not in final["response"].lower()
    assert "Day 1" in final["response"] or "day 1" in final["response"].lower()
    print("TEST PASSED")

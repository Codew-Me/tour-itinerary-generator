"""Tests for agent intent detection and orchestration."""

import pytest

from src.database.postgres import get_session_factory, init_db
from src.services.agent_intent import AgentIntent, detect_agent_intent
from src.services.agent_orchestrator import AgentOrchestrator
from src.services.conversation_state import ConversationState
from src.services.planning_flow import start_category_planning


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    yield


class TestAgentIntent:
    def test_greeting_when_idle(self):
        intent = detect_agent_intent("hi", ConversationState(), [])
        assert intent == AgentIntent.GREETING

    def test_start_itinerary_from_category_tab(self):
        intent = detect_agent_intent("__start_planning__:Heritage", ConversationState(), [])
        assert intent == AgentIntent.START_ITINERARY

    def test_off_topic_when_idle(self):
        intent = detect_agent_intent("what is 2+2", ConversationState(), [])
        assert intent == AgentIntent.OFF_TOPIC

    def test_plan_collect_during_planning(self):
        state = ConversationState()
        start_category_planning(state, "Heritage")
        intent = detect_agent_intent("3", state, [])
        assert intent in (AgentIntent.PLAN_COLLECT, AgentIntent.PROVIDE_INFORMATION)

    def test_generate_when_planning_complete(self):
        state = ConversationState(
            category_confirmed=True,
            category_tag="Heritage",
            planning_mode=True,
            duration_days=3,
            starting_location="Colombo",
            travellers="solo",
            interests=["adventure"],
            pace="relaxed",
        )
        start_category_planning(state, "Heritage")
        state.duration_days = 3
        state.starting_location = "Colombo"
        state.travellers = "solo"
        state.interests = ["adventure"]
        state.pace = "relaxed"
        state.mark_answered("duration")
        state.mark_answered("start_location")
        state.mark_answered("travellers")
        state.mark_answered("interests")
        state.mark_answered("pace")
        state.mark_answered("category")
        state.last_recommendations = [{"id": 1, "name": "Test", "district": "Colombo"}]
        intent = detect_agent_intent("ok", state, [])
        assert intent == AgentIntent.GENERATE_ITINERARY

    def test_clarify_on_ambiguous_yeah_after_itinerary(self):
        state = ConversationState(category_confirmed=True, category_tag="Heritage")
        state.current_itinerary = {"days": []}
        state.awaiting_itinerary_followup = True
        history = [{
            "role": "assistant",
            "content": (
                "How would you like to adjust it?\n"
                "• Make it more relaxed\n• Add more places"
            ),
        }]
        intent = detect_agent_intent("yeah", state, history)
        assert intent == AgentIntent.CLARIFY

    def test_modify_on_add_more_places(self):
        state = ConversationState()
        state.current_itinerary = {"days": [{"day": 1, "stops": []}]}
        state.session_mode = "itinerary_review"
        intent = detect_agent_intent("add more places", state, [])
        assert intent == AgentIntent.MODIFY_ITINERARY


class TestAgentOrchestrator:
    def test_off_topic_does_not_pretend_to_answer(self):
        session = get_session_factory()()
        try:
            agent = AgentOrchestrator(session)
            result = agent.run_turn("write me a python script", state=ConversationState())
            assert result["intent"] == "off_topic"
            assert result["action"] == "off_topic"
        finally:
            session.close()

    def test_idle_prompts_planning_not_generic_chat(self):
        session = get_session_factory()()
        try:
            agent = AgentOrchestrator(session)
            result = agent.run_turn("hmm", state=ConversationState())
            assert result["action"] == "idle"
            assert "Plan a" in result["response"] or "category" in result["response"]
        finally:
            session.close()

    def test_planning_flow_has_intent_and_phase(self):
        session = get_session_factory()()
        try:
            agent = AgentOrchestrator(session)
            state = None
            for msg in ("Plan a Heritage trip", "3", "Colombo"):
                result = agent.run_turn(msg, state=ConversationState.from_dict(state) if state else None)
                state = result["state"]
            assert result["agent_phase"] in ("planning", "recommending")
            assert result["intent"] in ("start_itinerary", "plan_collect", "provide_information", "recommend")
        finally:
            session.close()

    def test_active_planning_redirects_off_topic(self):
        session = get_session_factory()()
        try:
            agent = AgentOrchestrator(session)
            state = ConversationState()
            start_category_planning(state, "Heritage")
            result = agent.run_turn("what is the capital of france", state=state)
            assert result["intent"] == "off_topic"
            assert "planning your Sri Lanka trip" in result["response"]
        finally:
            session.close()

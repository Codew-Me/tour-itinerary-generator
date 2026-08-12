"""Stateful itinerary planning agent — intent → memory → tools → response."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.services.agent_handlers import (
    handle_change_preference,
    handle_decline,
    handle_explicit_category_change,
    handle_reject_itinerary,
    mark_itinerary_rejected,
)
from src.services.agent_intent import AgentIntent, detect_agent_intent, resolve_agent_phase
from src.services.agent_tools import AgentTools
from src.services.conversation_state import ConversationState
from src.services.planning_flow import (
    build_planning_transition,
    detect_explicit_category_selection,
    has_enough_for_build,
    has_enough_for_suggest,
    has_valid_itinerary,
    next_planning_question,
    planning_ready,
    resolve_planning_category,
    should_suggest_after_planning,
    start_category_planning,
    start_open_planning,
)
from src.services.state_manager import (
    NextAction,
    build_chat_response,
    build_clarify_response,
    decide_next_action,
    extract_days,
    update_state,
)


class AgentOrchestrator:
    """Routes each turn through intent detection, structured memory, and tools."""

    def __init__(self, session: Session):
        self.session = session
        self.tools = AgentTools(session)

    def run_turn(
        self,
        message: str,
        history: list[dict] | None = None,
        state: ConversationState | None = None,
    ) -> dict:
        history = history or []
        state = state or ConversationState()
        answering_key = state.last_question_key
        state = update_state(state, message, history)

        intent = detect_agent_intent(message, state, history)
        state.last_intent = intent.value
        state.agent_phase = resolve_agent_phase(intent, state)
        state.intent_history = (state.intent_history + [intent.value])[-20:]

        if intent == AgentIntent.GREETING:
            return self._result(
                "Hello. I'm your Sri Lanka travel assistant. Tell me about the trip you're planning, "
                "or select a trip theme from the sidebar to get started.",
                "greeting",
                state,
                intent=intent,
            )

        if intent == AgentIntent.ACKNOWLEDGE:
            return self._result(
                build_chat_response(message, state),
                "acknowledge",
                state,
                intent=intent,
            )

        if intent == AgentIntent.OFF_TOPIC:
            return self._result(
                self._off_topic_response(state),
                "off_topic",
                state,
                intent=intent,
            )

        if intent == AgentIntent.DECLINE:
            return self._result(handle_decline(state), "decline", state, intent=intent)

        if intent == AgentIntent.REJECT_ITINERARY:
            response, revise = handle_reject_itinerary(state, message)
            if revise:
                ack = response
                tool_result = self.tools.build_itinerary(message, state, history=history)
                state.session_mode = "itinerary_review"
                state.awaiting_rejection_reason = False
                state.itinerary_modify = False
                return self._result(
                    f"{ack}\n\n{tool_result['response']}",
                    "revise_itinerary",
                    state,
                    searched=True,
                    intent=AgentIntent.REVISE_ITINERARY,
                )
            return self._result(response, "reject_itinerary", state, intent=intent)

        if intent == AgentIntent.CHANGE_PREFERENCE:
            response, revise = handle_change_preference(state, message)
            if revise:
                tool_result = self.tools.build_itinerary(message, state, history=history)
                state.session_mode = "itinerary_review"
                state.itinerary_modify = False
                return self._result(
                    f"{response}\n\n{tool_result['response']}",
                    "revise_itinerary",
                    state,
                    searched=True,
                    intent=AgentIntent.REVISE_ITINERARY,
                )
            return self._result(response, "change_preference", state, intent=intent)

        if intent == AgentIntent.CHANGE_CATEGORY:
            cat = detect_explicit_category_selection(message, state) or resolve_planning_category(message, state=state)
            if cat:
                response = handle_explicit_category_change(state, cat)
                if state.itinerary_modify:
                    tool_result = self.tools.build_itinerary(message, state, history=history)
                    state.session_mode = "itinerary_review"
                    state.itinerary_modify = False
                    return self._result(
                        f"{response}\n\n{tool_result['response']}",
                        "revise_itinerary",
                        state,
                        searched=True,
                        intent=AgentIntent.REVISE_ITINERARY,
                    )
                q = next_planning_question(state)
                if q and not planning_ready(state):
                    return self._result(response, "clarify", state, intent=intent, clarify_key=q)
                if planning_ready(state):
                    tool_result = self.tools.build_itinerary(message, state, history=history)
                    state.session_mode = "itinerary_review"
                    return self._result(
                        f"{response}\n\n{tool_result['response']}",
                        "generate_itinerary",
                        state,
                        searched=True,
                        intent=AgentIntent.GENERATE_ITINERARY,
                    )
                return self._result(response, "change_category", state, intent=intent)

        if intent == AgentIntent.START_ITINERARY:
            cat = resolve_planning_category(message, state=state)
            if cat:
                start_category_planning(state, cat)
            else:
                days = extract_days(message)
                start_open_planning(state, duration_days=days)
            q = next_planning_question(state) or "start_location"
            state.last_question_key = q
            transition = build_planning_transition(state)
            response = transition if transition else build_clarify_response(q, state)
            return self._result(response, "clarify", state, intent=intent, clarify_key=q)

        if intent in (AgentIntent.PLAN_COLLECT, AgentIntent.PROVIDE_INFORMATION):
            q = next_planning_question(state)
            if q:
                state.last_question_key = q
                transition = build_planning_transition(state)
                response = transition if transition else build_clarify_response(q, state)
                return self._result(response, "clarify", state, intent=intent, clarify_key=q)
            if (
                answering_key == "interests"
                and state.category_confirmed
                and has_enough_for_suggest(state)
            ):
                if not state.pace:
                    state.pace = "balanced"
                return self._recommend(message, state, AgentIntent.RECOMMEND)
            if should_suggest_after_planning(state):
                if not state.pace:
                    state.pace = "balanced"
                return self._recommend(message, state, AgentIntent.RECOMMEND)
            if planning_ready(state) and not has_valid_itinerary(state):
                return self._generate_itinerary(message, state, history, intent)

        if intent == AgentIntent.CLARIFY:
            from src.services.itinerary_service import (
                ITINERARY_FOLLOWUP_CLARIFY,
                RECOMMENDATION_FOLLOWUP_CLARIFY,
            )
            from src.services.state_manager import _is_recommendation_followup_prompt, _last_assistant_message

            last_assistant = _last_assistant_message(history)
            if state.last_recommendations and _is_recommendation_followup_prompt(last_assistant):
                return self._result(
                    RECOMMENDATION_FOLLOWUP_CLARIFY,
                    "clarify",
                    state,
                    intent=intent,
                    clarify_key="recommendation_followup",
                )
            state.awaiting_itinerary_followup = True
            return self._result(
                ITINERARY_FOLLOWUP_CLARIFY,
                "clarify",
                state,
                intent=intent,
                clarify_key="itinerary_followup",
            )

        if intent in (AgentIntent.MODIFY_ITINERARY, AgentIntent.REVISE_ITINERARY):
            from src.services.state_manager import _parse_itinerary_modify_mode
            if not state.itinerary_modify_mode:
                state.itinerary_modify_mode = _parse_itinerary_modify_mode(message) or "add"
            state.itinerary_modify = True
            mark_itinerary_rejected(state) if intent == AgentIntent.REVISE_ITINERARY else None
            return self._generate_itinerary(message, state, history, intent, revise=True)

        if intent in (AgentIntent.RECOMMEND, AgentIntent.REQUEST_MORE_OPTIONS):
            return self._recommend(message, state, intent)

        if intent == AgentIntent.GENERATE_ITINERARY:
            if (
                answering_key == "interests"
                and state.category_confirmed
                and has_enough_for_suggest(state)
            ):
                if not state.pace:
                    state.pace = "balanced"
                return self._recommend(message, state, AgentIntent.RECOMMEND)
            if has_enough_for_build(state) or _is_build_from_recommendations(state, message):
                state.use_previous_recommendations = True
            if not planning_ready(state) and not state.use_previous_recommendations:
                q = next_planning_question(state) or "duration"
                response = build_clarify_response(q, state)
                return self._result(response, "clarify", state, intent=AgentIntent.PLAN_COLLECT, clarify_key=q)
            return self._generate_itinerary(message, state, history, intent)

        return self._execute_decided_action(message, state, history, intent)

    def _execute_decided_action(
        self,
        message: str,
        state: ConversationState,
        history: list[dict],
        intent: AgentIntent,
    ) -> dict:
        """Fallback execution via state_manager for legacy and unclassified flows."""
        action, clarify_key = decide_next_action(state, message, history)

        if intent in (AgentIntent.DECLINE, AgentIntent.REJECT_ITINERARY):
            return self._result(handle_decline(state), "decline", state, intent=intent)

        if action == NextAction.GENERATE_ITINERARY:
            if has_enough_for_build(state) or _is_build_from_recommendations(state, message):
                state.use_previous_recommendations = True
            if not planning_ready(state) and not state.use_previous_recommendations:
                q = next_planning_question(state) or "duration"
                return self._result(
                    build_clarify_response(q, state),
                    "clarify",
                    state,
                    intent=AgentIntent.PLAN_COLLECT,
                    clarify_key=q,
                )
            return self._generate_itinerary(message, state, history, intent)

        if action == NextAction.CLARIFY:
            key = clarify_key or "duration"
            state.last_question_key = key
            transition = build_planning_transition(state)
            response = transition if transition else build_clarify_response(key, state)
            return self._result(response, "clarify", state, intent=intent, clarify_key=key)

        if action == NextAction.ASK_DURATION:
            state.pending_action = "awaiting_duration"
            return self._result(
                build_clarify_response("duration", state),
                "ask_duration",
                state,
                intent=intent,
            )

        if action == NextAction.RECOMMEND:
            return self._recommend(message, state, AgentIntent.RECOMMEND)

        if action == NextAction.CHAT:
            if intent == AgentIntent.IDLE:
                return self._result(
                    "I help plan Sri Lanka trips step by step. Try \"Plan a 3 day tour\" or pick a category tab.",
                    "idle",
                    state,
                    intent=intent,
                )
            return self._result(
                build_chat_response(message, state),
                "chat",
                state,
                intent=intent,
            )

        q = next_planning_question(state)
        if q:
            return self._result(
                build_clarify_response(q, state),
                "clarify",
                state,
                intent=AgentIntent.PLAN_COLLECT,
                clarify_key=q,
            )
        return self._result(
            "Tell me about the trip you're planning.",
            "clarify",
            state,
            intent=AgentIntent.PLAN_COLLECT,
        )

    def _generate_itinerary(
        self,
        message: str,
        state: ConversationState,
        history: list[dict],
        intent: AgentIntent,
        *,
        revise: bool = False,
    ) -> dict:
        tool_result = self.tools.build_itinerary(message, state, history=history)
        state.pending_action = None
        state.itinerary_requested = False
        state.itinerary_modify = False
        state.use_previous_recommendations = False
        state.awaiting_rejection_reason = False
        state.session_mode = "itinerary_review"
        state.mark_answered("duration")
        action = "revise_itinerary" if revise else "generate_itinerary"
        return self._result(
            tool_result["response"],
            action,
            state,
            searched=True,
            intent=intent,
        )

    def _recommend(self, message: str, state: ConversationState, intent: AgentIntent) -> dict:
        tool_result = self.tools.search_attractions(
            message,
            state,
            limit=6 if state.planning_mode else 8,
        )
        state.recommendation_requested = False
        state.wants_more_recommendations = False
        state.wants_list_all = False
        AgentTools.persist_recommendations(state, tool_result["candidates"])
        return self._result(
            tool_result["response"],
            "recommend",
            state,
            searched=True,
            intent=intent,
            rec=tool_result["rec"],
        )

    @staticmethod
    def _off_topic_response(state: ConversationState) -> str:
        if state.planning_mode or state.session_mode == "planning":
            q = next_planning_question(state) or state.last_question_key or "duration"
            question = build_clarify_response(q, state)
            return f"I'm focused on planning your Sri Lanka trip right now.\n\n{question}"
        return (
            "I'm a Sri Lanka itinerary planning assistant. "
            "Try \"Plan a 3 day tour\" or pick a category to begin."
        )

    @staticmethod
    def _result(
        response: str,
        action: str,
        state: ConversationState,
        *,
        searched: bool = False,
        intent: AgentIntent,
        clarify_key: str | None = None,
        rec: dict | None = None,
    ) -> dict:
        prefs = state.to_tourism_preferences()
        out: dict = {
            "response": response,
            "action": action,
            "intent": intent.value,
            "agent_phase": state.agent_phase,
            "session_mode": state.session_mode,
            "searched": searched,
            "state": state.to_dict(),
        }
        if clarify_key:
            out["clarify_key"] = clarify_key
        if searched:
            out["search_type"] = "attractions_primary"
            out["preferences"] = {
                "district": prefs.district,
                "starting_location": state.starting_location,
                "category": prefs.category,
                "interests": state.interests,
                "duration_days": state.duration_days,
                "travellers": state.travellers,
                "pace": state.pace,
                "planning_mode": state.planning_mode,
            }
        if rec:
            out["candidate_count"] = len(rec.get("candidates", []))
        return out


def _is_build_from_recommendations(state: ConversationState, message: str) -> bool:
    from src.services.state_manager import _is_build_itinerary_message

    return bool(state.last_recommendations and _is_build_itinerary_message(message, state))

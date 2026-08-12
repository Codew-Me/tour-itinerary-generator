#!/usr/bin/env python3
"""Generate architecture and conversation-flow PDF documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpdf import FPDF


OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "Sri_Lanka_Travel_Agent_Architecture.pdf"


def _ascii_safe(text: str) -> str:
    """FPDF core fonts only support latin-1; strip smart punctuation."""
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2192", "->")
    )


class DocPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(18, 18, 18)

    def _w(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Sri Lanka Travel Agent - Architecture & Conversation Guide", align="L")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 20, 20)
        self.multi_cell(self._w(), 8, _ascii_safe(title))
        self.ln(2)

    def sub_title(self, title: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self._w(), 6, _ascii_safe(title))
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self._w(), 5, _ascii_safe(text))
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self._w(), 5, _ascii_safe(f"  - {text}"))

    def code_block(self, text: str) -> None:
        self.set_font("Courier", "", 8)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(20, 20, 20)
        w = self._w()
        for line in _ascii_safe(text).split("\n"):
            self.multi_cell(w, 4, line, fill=True)
        self.ln(2)

    def file_row(self, path: str, purpose: str) -> None:
        self.set_font("Courier", "B", 8)
        self.set_text_color(0, 80, 160)
        self.multi_cell(self._w(), 4, _ascii_safe(path))
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(self._w(), 4, _ascii_safe(purpose))
        self.ln(1)


def build_pdf() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ---- Title ----
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 15, 15)
    pdf.cell(0, 12, "Sri Lanka Travel Agent", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 8, "Architecture & Conversation Guide", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.cell(0, 6, "Dataset-backed travel planner (350 attractions + 33K reviews)", ln=True, align="C")
    pdf.ln(8)

    pdf.body_text(
        "This document explains how the project is structured, how data flows from your Excel/CSV "
        "datasets into recommendations and itineraries, and how each user message is processed "
        "through the conversation engine - including category tabs and mood_tag-based planning "
        "from attractions.xlsx."
    )

    # ---- 1. Overview ----
    pdf.add_page()
    pdf.section_title("1. System Overview")
    pdf.body_text(
        "The application is a multi-turn Sri Lanka trip planner. Users chat in a Streamlit UI. "
        "The FastAPI backend runs a deterministic AgentOrchestrator that collects trip preferences, "
        "searches the attractions database, and builds day-by-day itineraries. Places and descriptions "
        "always come from the loaded dataset; reviews provide supporting evidence from ChromaDB."
    )
    pdf.sub_title("Technology stack")
    pdf.bullet("Frontend: Streamlit (frontend/app.py) - ChatGPT-style chat UI")
    pdf.bullet("API: FastAPI (api/main.py) - REST endpoints, JWT auth")
    pdf.bullet("Database: SQLite (default) or PostgreSQL - 350 structured attractions")
    pdf.bullet("Vector store: ChromaDB - ~33,542 embedded traveler reviews")
    pdf.bullet("LLM: OpenAI / Ollama / Anthropic - routing & polish (not for inventing places)")
    pdf.bullet("Tests: pytest - 140+ tests for conversation, mood, and planning flows")

    pdf.sub_title("Dataset sources")
    pdf.bullet("attractions.xlsx - 350 rows: Attraction Name, Category, Destination, Details, mood_tag, Image")
    pdf.bullet("Destination Reviews (final).csv - traveler review text")
    pdf.bullet("Categories: Wild (31), Heritage (120), Scenic (97), Pristine (29), Essence (66), Thrills (7)")
    pdf.bullet(
        "mood_tag values: Curious (144), Relaxed (79), Peaceful (76), Adventure (27), "
        "Excited (8), Authentic (7), Spiritual (6), Healing (1), Explore (1), Happy (1)"
    )

    # ---- 2. Architecture ----
    pdf.add_page()
    pdf.section_title("2. Architecture Diagram")
    pdf.code_block(
        """
+------------------+       POST /chat        +-------------------+
|  Streamlit UI    | ----------------------> |  FastAPI (main)   |
|  frontend/app.py | <---------------------- |  api/main.py      |
+------------------+   JSON response        +---------+---------+
                                                      |
                                                      v
                                            +-------------------+
                                            | ConversationService|
                                            | (save/load msgs)   |
                                            +---------+---------+
                                                      |
                                                      v
                                            +-------------------+
                                            |   ChatService     |
                                            | AgentOrchestrator |
                                            +---------+---------+
                                    +-------+---------+-------+
                                    |       |         |       |
                                    v       v         v       v
                              update_state  intent   AgentTools  clarify
                              state_mgr    agent_intent         planning_flow
                                    |       |         |
                                    |       |    +----+----+
                                    |       |    |         |
                                    v       v    v         v
                            ConversationState  Recommend  Itinerary
                                               Service    Planner
                                                    |         |
                                                    v         v
                                            AttractionRepository
                                            (SQLite / PostgreSQL)
                                                    |
                              +---------------------+---------------------+
                              |                                           |
                              v                                           v
                     review_evidence.py                          review_stats.py
                     (ChromaDB linked reviews)                   (cached stats)
"""
    )

    pdf.sub_title("Data ingestion pipeline (offline)")
    pdf.code_block(
        """
  attractions.xlsx + reviews CSV
           |
           v
   scripts/clean_data.py  -->  data/processed/*.csv
           |
           v
   scripts/load_postgres.py  -->  SQLite/PostgreSQL (District, Destination, Attraction)
           |
           v
   scripts/build_vector_db.py  -->  data/chroma/ (review embeddings)
"""
    )

    # ---- 3. Request flow ----
    pdf.add_page()
    pdf.section_title("3. End-to-End Request Flow")
    pdf.body_text("Every user message follows this path:")
    pdf.ln(1)
    steps = [
        "User types in Streamlit chat, clicks a welcome starter chip, or a sidebar category tab.",
        "frontend/app.py POSTs to /chat with message, conversation_id, optional planning_category.",
        "Welcome chips send __starter__:Category|display text (e.g. __starter__:Thrills|Adventure day trips near Gampaha).",
        "api/main.py authenticates JWT, loads conversation history and ConversationState from DB.",
        "If UI sent __start_planning__:Wild or __starter__:Wild|..., category/mood hints are applied.",
        "ChatService.handle_message() calls AgentOrchestrator.run_turn().",
        "update_state() parses duration, location, travellers, interests, mood_tag from the message.",
        "apply_mood_from_message() maps phrases like 'I feel curious' to dataset mood_tag + mood_confirmed.",
        "detect_agent_intent() classifies: greeting, start_itinerary, recommend, clarify, etc.",
        "Orchestrator branches: ask next planning question, search attractions, or build itinerary.",
        "RecommendationService filters strictly by category_confirmed and/or mood_confirmed.",
        "Response text + updated state saved to DB in assistant message metadata.",
        "Streamlit renders markdown and HTML place cards from the response.",
    ]
    for i, step in enumerate(steps, 1):
        pdf.bullet(f"{i}. {step}")

    # ---- 4. Conversation engine ----
    pdf.add_page()
    pdf.section_title("4. How Conversations Work")
    pdf.sub_title("ConversationState (memory between turns)")
    pdf.body_text(
        "Stored as JSON in the last assistant message metadata. Key fields:"
    )
    pdf.bullet("category_tag / category_confirmed - Wild, Heritage, etc. (hard filter like sidebar tab)")
    pdf.bullet("mood_tag / mood_confirmed - Curious, Relaxed, Happy, etc. from mood_tag column (hard filter)")
    pdf.bullet("duration_days, starting_location, travellers, interests, pace")
    pdf.bullet("destination_district - 'near Gampaha' search area (separate from start location)")
    pdf.bullet("planning_mode - true during step-by-step Q&A")
    pdf.bullet("last_recommendations - places shown last turn (for build-itinerary)")
    pdf.bullet("current_itinerary - day-by-day plan if generated")
    pdf.bullet("agent_phase - idle | planning | recommending | itinerary | revision")

    pdf.sub_title("Dual filters: Category and mood_tag")
    pdf.body_text(
        "Both columns come from attractions.xlsx. When confirmed, they behave like hard SQL filters - "
        "only matching rows are suggested or scheduled. Category and mood can combine "
        "(e.g. Heritage tab + 'I feel curious' -> Heritage AND Curious)."
    )
    pdf.code_block(
        """
  category_confirmed=True  -> filter_by_category(..., strict=True)
  mood_confirmed=True        -> filter_by_mood(..., strict=True)

  TourismPreferences.to_tourism_preferences() maps ConversationState -> DB query filters.
  RecommendationService and ItineraryPlanner pass category_strict / mood_strict flags.
"""
    )

    pdf.sub_title("Planning flows")
    pdf.body_text("Flow A - Category tab (sidebar Wild / Heritage / ...):")
    pdf.code_block(
        """
  User clicks "Wild" tab
    -> __start_planning__:Wild
    -> category_confirmed = True, skip interests question
    -> Ask: duration -> start location -> travellers -> recommend Wild places
"""
    )
    pdf.body_text("Flow B - Mood-first planning (user states mood):")
    pdf.code_block(
        """
  User: "I feel curious"  or  "I'm happy"  or  "plan based on mood"
    -> mood_tag = Curious / Happy / ..., mood_confirmed = True
    -> skip interests and category questions
    -> Ask: duration -> start location -> travellers -> recommend Curious places only
"""
    )
    pdf.body_text("Flow C - Welcome starter chips (home screen):")
    pdf.code_block(
        """
  User clicks "Adventure day trips near Gampaha"
    -> __starter__:Thrills|Adventure day trips near Gampaha
    -> category=Thrills, destination_district=Gampaha, skip interests
    -> Ask: duration -> start location -> travellers -> recommend Thrills near Gampaha
"""
    )
    pdf.body_text("Flow D - Open planning (user describes trip):")
    pdf.code_block(
        """
  User: "Plan a 5 day trip"
    -> duration collected
    -> start location -> travellers -> interests (e.g. wildlife)
    -> infer Wild category + Explore mood from interests
    -> recommend matching places
"""
    )
    pdf.body_text("Flow E - Replan by mood after itinerary:")
    pdf.code_block(
        """
  [After itinerary shown with adjustment options]
  User: "I feel curious"  or  "plan another itinerary I feel curious"
    -> mood_tag updated to Curious, mood_confirmed = True
    -> prepare_mood_replan() clears old itinerary/recommendations
    -> rebuild itinerary from Curious-tagged attractions only
"""
    )
    pdf.body_text("Flow F - Bare category after greeting:")
    pdf.code_block(
        """
  User: "Hi"  -> greeting
  User: "WILD" -> start Wild category planning
    -> Ask duration -> location -> travellers -> recommend
"""
    )

    pdf.sub_title("Intent detection (agent_intent.py)")
    pdf.bullet("GREETING - hi, hello (no active session)")
    pdf.bullet("START_ITINERARY - plan trip, category tab, bare category name")
    pdf.bullet("PLAN_COLLECT - answering duration/location/travellers/interests")
    pdf.bullet("RECOMMEND - enough info to suggest places")
    pdf.bullet("GENERATE_ITINERARY - build day-by-day plan")
    pdf.bullet("MODIFY_ITINERARY / REJECT_ITINERARY - change or reject plan")
    pdf.bullet("CHANGE_CATEGORY - switch Wild to Heritage mid-session")
    pdf.bullet("CHANGE_PREFERENCE / mood replan - 'I feel curious', 'plan another itinerary'")
    pdf.bullet("IDLE / CHAT - fallback small talk")

    pdf.sub_title("Mood phrase parsing (planning_input.py + planning_flow.py)")
    pdf.bullet("normalize_mood_feel_phrases() - fixes 'ifeelcurious' -> 'i feel curious'")
    pdf.bullet("apply_mood_from_message() - maps MOOD_ALIASES to VALID_MOODS / mood_tag")
    pdf.bullet("wants_replan_with_mood() - detects mood change during itinerary review")
    pdf.bullet("start_mood_planning() - sets mood_confirmed=True (parallel to start_category_planning)")

    # ---- 5. Recommendations ----
    pdf.add_page()
    pdf.section_title("5. How Recommendations Work")
    pdf.body_text(
        "RecommendationService (recommendation_service.py) is attraction-first:"
    )
    pdf.bullet("Step 1: AttractionRepository.search() queries DB by category, mood_tag, district")
    pdf.bullet("Step 2: filter_by_category() and filter_by_mood() when confirmed (strict mode)")
    pdf.bullet("Step 3: Score each attraction (category match, mood match, geo, interest keywords)")
    pdf.bullet("Step 4: Attach review evidence ONLY for top candidates via review_evidence.py")
    pdf.bullet("Step 5: format_recommendation_response() in attraction_cards.py builds place cards")
    pdf.body_text(
        "Cards show: place name, district, full Details text from dataset, visitor feedback "
        "from linked reviews. The LLM does not choose which places exist."
    )

    pdf.sub_title("Itinerary building")
    pdf.body_text(
        "ItineraryPlanner (itinerary_planner.py) takes ranked attractions and schedules them "
        "across days using district centroids (geography.py) for travel compatibility. "
        "ItineraryService formats the plan into readable day headers and place cards."
    )

    # ---- 6. File guide ----
    pdf.add_page()
    pdf.section_title("6. File-by-File Guide")

    sections = [
        ("API & Frontend", [
            ("api/main.py", "FastAPI app: /chat, /auth, /conversations, /dataset/categories, /health"),
            ("frontend/app.py", "Streamlit UI: login, chat, sidebar themes, logo, place card HTML"),
        ]),
        ("Conversation core", [
            ("src/services/chat_service.py", "Thin facade -> AgentOrchestrator"),
            ("src/services/agent_orchestrator.py", "Main turn router: intent -> action -> response"),
            ("src/services/agent_intent.py", "Detect user intent from message + state + history"),
            ("src/services/agent_handlers.py", "Decline, reject itinerary, change category/preference"),
            ("src/services/agent_tools.py", "search_attractions(), build_itinerary(), revise_itinerary()"),
            ("src/services/conversation_state.py", "Dataclass holding all multi-turn memory"),
            ("src/services/conversation_service.py", "Persist messages; reload state from metadata"),
            ("src/services/state_manager.py", "Parse signals, decide_next_action, clarify responses"),
        ]),
        ("Planning", [
            ("src/services/planning_flow.py", "Planning steps, category/mood start, starter chips, replan"),
            ("src/services/planning_input.py", "Parse duration, location, travellers, interests, mood phrases"),
            ("src/services/preferences.py", "VALID_CATEGORIES, VALID_MOODS, MOOD_ALIASES, TourismPreferences"),
            ("src/services/geography.py", "District coords, haversine distance, start location resolution"),
        ]),
        ("Recommendations & itineraries", [
            ("src/services/recommendation_service.py", "Rank attractions; category_strict + mood_strict filters"),
            ("src/services/itinerary_planner.py", "Geo-aware scheduling; _apply_dataset_gates for category+mood"),
            ("src/services/itinerary_service.py", "Format itinerary text for user"),
            ("src/services/attraction_cards.py", "Markdown/HTML place card formatting"),
            ("src/services/category_filter.py", "Hard category and mood_tag eligibility (filter_by_mood)"),
        ]),
        ("Reviews & evidence", [
            ("src/services/review_evidence.py", "Fetch Chroma reviews linked to one attraction"),
            ("src/services/review_stats.py", "Cached review theme stats from reviews_clean.csv"),
            ("src/vectorstore/chroma_store.py", "ChromaDB persistent review index"),
            ("src/embeddings/embedding_model.py", "Sentence-transformers embeddings"),
        ]),
        ("Database", [
            ("src/database/models.py", "SQLAlchemy: User, Attraction, Destination, District, Conversation"),
            ("src/database/postgres.py", "Engine and session factory (SQLite or PostgreSQL)"),
            ("src/database/repositories.py", "AttractionRepository: search, stats, summaries"),
        ]),
        ("Data pipeline", [
            ("src/data/loader.py", "Read raw CSV and Excel files"),
            ("src/data/cleaner.py", "Normalize and clean attraction/review records"),
            ("src/data/matcher.py", "Link attractions to review destinations"),
            ("src/data/normalizer.py", "District/name normalization helpers"),
            ("scripts/clean_data.py", "Phase 1: raw -> processed CSVs"),
            ("scripts/load_postgres.py", "Phase 2: processed CSVs -> relational DB"),
            ("scripts/build_vector_db.py", "Phase 3: embed reviews into ChromaDB"),
        ]),
        ("Agent (secondary path)", [
            ("src/agent/graph.py", "LangGraph ReAct agent with tools (not used by /chat)"),
            ("src/agent/prompts.py", "System prompt for tool-using agent"),
            ("src/tools/__init__.py", "LangChain tools: search_reviews, search_attractions, etc."),
            ("src/services/conversation_router.py", "LLM router for LangChain tool path"),
        ]),
        ("Auth & config", [
            ("src/auth/service.py", "Register, login, JWT tokens"),
            ("src/config.py", "Paths, DB URL, LLM provider settings"),
        ]),
    ]

    for section_name, files in sections:
        pdf.sub_title(section_name)
        for path, purpose in files:
            pdf.file_row(path, purpose)

    # ---- 7. Example conversations ----
    pdf.add_page()
    pdf.section_title("7. Example Conversation Traces")

    pdf.sub_title("Example 1: Wild trip from sidebar")
    pdf.code_block(
        """
Turn 1  USER: [clicks Wild tab]
          BOT:  How many days are you planning to travel?
          STATE: category=Wild, category_confirmed=true, planning_mode=true

Turn 2  USER: 3 days
          BOT:  Where will you be starting your journey?
          STATE: duration_days=3

Turn 3  USER: Colombo
          BOT:  Who are you travelling with?
          STATE: starting_location=Colombo

Turn 4  USER: solo
          BOT:  [Wild place cards from DB - Yala, Bundala, etc.]
          STATE: last_recommendations=[...], action=recommend
"""
    )

    pdf.sub_title("Example 2: Mood-first planning (Curious)")
    pdf.code_block(
        """
Turn 1  USER: I feel curious
          BOT:  Let's plan around a Curious mood - matched to our attractions dataset.
                How many days are you planning to travel?
          STATE: mood_tag=Curious, mood_confirmed=true, planning_mode=true

Turn 2  USER: 3 days
Turn 3  USER: Colombo
Turn 4  USER: solo
          BOT:  [Curious place cards only - Yapahuwa, Diva Guhawa, Richmond Castle, ...]
          STATE: action=recommend, all candidates mood=Curious
"""
    )

    pdf.sub_title("Example 3: Welcome chip + near district")
    pdf.code_block(
        """
Turn 1  USER: [clicks "Adventure day trips near Gampaha"]
          -> __starter__:Thrills|Adventure day trips near Gampaha
          STATE: category=Thrills, destination_district=Gampaha, interests skipped

Turn 2-4 duration / start / travellers
          BOT:  Thrills places near Gampaha district
"""
    )

    pdf.sub_title("Example 4: Replan itinerary by mood")
    pdf.code_block(
        """
Turn N  BOT:  [3-day itinerary with Hummanaya Blow Hole + adjustment options]
Turn N+1 USER: plan another itinerary I feel curious
          BOT:  Got it - I'll rebuild your trip with a Curious mood from our dataset...
                [new itinerary with Curious-tagged stops only]
          STATE: mood_tag=Curious, mood_confirmed=true, current_itinerary replaced
"""
    )

    pdf.sub_title("Example 5: Open planning with wildlife interest")
    pdf.code_block(
        """
Turn 1  USER: Plan a 5 day trip
Turn 2  USER: Seeduwa
Turn 3  USER: solo
Turn 4  USER: wild
          BOT:  Shortlist of Wild category places (wildlife interest inferred)
          STATE: interests=[wildlife], category=Wild, mood=Explore
"""
    )

    pdf.sub_title("Example 6: Build itinerary from recommendations")
    pdf.code_block(
        """
Turn N  BOT:  [shows 5 recommended places]
Turn N+1 USER: build an itinerary / yes
          BOT:  Day 1: ... Day 2: ... (from ItineraryPlanner)
          STATE: current_itinerary={days:[...]}
"""
    )

    # ---- 8. Operations ----
    pdf.add_page()
    pdf.section_title("8. Running the System")
    pdf.sub_title("Initial setup")
    pdf.code_block(
        """
pip install -r requirements.txt
copy .env.example to .env  (set OPENAI_API_KEY if using OpenAI)

python scripts/clean_data.py
python scripts/load_postgres.py
python scripts/build_vector_db.py   # ~10-15 min

python -m uvicorn api.main:app --port 8000
streamlit run frontend/app.py --server.port 8501
"""
    )
    pdf.sub_title("Health check")
    pdf.body_text("GET http://localhost:8000/health - shows chroma_review_count and LLM config")
    pdf.body_text("GET http://localhost:8000/dataset/categories - category counts from DB")

    pdf.sub_title("Key design rules")
    pdf.bullet("Attractions are NEVER invented - only the 350 loaded rows can appear")
    pdf.bullet("Reviews support evidence - searched only for linked destinations")
    pdf.bullet("Category tabs map 1:1 to dataset Category column (category_confirmed)")
    pdf.bullet("User-stated moods map 1:1 to dataset mood_tag column (mood_confirmed)")
    pdf.bullet("Welcome starter chips pre-fill category, mood, district, and duration when detectable")
    pdf.bullet("Conversation state persists across turns via assistant message metadata")
    pdf.bullet("Main /chat path is rule-driven orchestrator, not free-form LLM chat")

    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build_pdf()

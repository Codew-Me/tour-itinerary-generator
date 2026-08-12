"""ChatGPT-style travel planner UI — dataset-backed categories and recommendations."""

from __future__ import annotations

import base64
import os
import re
import textwrap
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
LOGO_CANDIDATES = (
    PROJECT_ROOT / "logo.jpg",
    PROJECT_ROOT / "logo.jg",
    PROJECT_ROOT / "logo.jpeg",
    FRONTEND_DIR / "logo.jpg",
    FRONTEND_DIR / "logo.jg",
)
LOGO_FILE = next((p for p in LOGO_CANDIDATES if p.is_file()), None)

FALLBACK_CATEGORIES = [
    {"name": c, "count": 0, "description": ""}
    for c in ("Wild", "Heritage", "Scenic", "Pristine", "Essence", "Thrills")
]

ASSISTANT_AVATAR = str(LOGO_FILE) if LOGO_FILE else "assistant"
USER_AVATAR = "user"

APP_CSS = textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-app: #ffffff;
    --bg-main: #ffffff;
    --bg-sidebar: #171717;
    --bg-sidebar-hover: #212121;
    --bg-sidebar-active: #2f2f2f;
    --bg-input: #ffffff;
    --bg-surface: #ffffff;
    --bg-user-row: #f7f7f8;
    --bg-user-bubble: #f4f4f4;
    --border: rgba(0, 0, 0, 0.1);
    --border-subtle: rgba(0, 0, 0, 0.06);
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --text: #0d0d0d;
    --text-secondary: #353740;
    --text-muted: #6e6e80;
    --sidebar-text: #ececec;
    --sidebar-muted: #8e8ea0;
    --accent: #10a37f;
    --accent-soft: rgba(16, 163, 127, 0.12);
    --radius: 12px;
    --radius-pill: 999px;
    --chat-width: 46rem;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text) !important;
}

/* Streamlit still injects dark-theme text in places — force readable contrast */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] a,
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] em,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stTabs"] [data-baseweb="tab"],
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--sidebar-muted) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--sidebar-border) !important;
    margin: 0.75rem 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

.stCaption,
[data-testid="stCaptionContainer"],
.welcome-sub,
.sidebar-brand-sub,
.place-review.quote {
    color: var(--text-muted) !important;
}

.place-desc,
.place-review {
    color: var(--text-secondary) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--sidebar-border) !important;
    min-width: 260px !important;
    max-width: 280px !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--bg-sidebar) !important;
    padding: 0.65rem 0.75rem 1rem !important;
}

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 8rem !important;
    max-width: var(--chat-width) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Sidebar (ChatGPT dark) ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: var(--sidebar-text) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.55rem 0.75rem !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.15s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-sidebar-hover) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    margin-bottom: 0.65rem !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--bg-sidebar-hover) !important;
}

.sidebar-section {
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--sidebar-muted);
    margin: 1.35rem 0 0.45rem 0.35rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.25rem 0.25rem 0.85rem;
    margin-bottom: 0.15rem;
}

.sidebar-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 4px;
    background: #ffffff;
    color: #171717;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}

.brand-logo,
.brand-logo-sidebar {
    width: 40px;
    height: 40px;
    object-fit: contain;
    border-radius: 4px;
    flex-shrink: 0;
    background: #ffffff;
    padding: 3px;
}

.brand-logo-lg {
    width: 72px;
    height: 72px;
    object-fit: contain;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    padding: 6px;
    margin: 0 auto 0.75rem;
    display: block;
}

.brand-logo-xl,
.brand-logo-xxl {
    width: 72px;
    height: 72px;
    object-fit: contain;
    border-radius: 8px;
    background: #ffffff;
    padding: 6px;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}

.sidebar-brand-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: var(--sidebar-text);
    margin: 0;
    line-height: 1.25;
}

.sidebar-brand-sub {
    font-size: 0.6875rem;
    color: var(--sidebar-muted);
    margin: 0.15rem 0 0;
}

.dataset-badge {
    font-size: 0.6875rem;
    color: var(--sidebar-muted);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--sidebar-border);
    border-radius: var(--radius);
    padding: 0.45rem 0.6rem;
    margin: 0.25rem 0.25rem 0.65rem;
    line-height: 1.4;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"],
#MainMenu,
footer {
    visibility: hidden !important;
    height: 0 !important;
    display: none !important;
}

.theme-btn-label {
    display: flex;
    justify-content: space-between;
    width: 100%;
    gap: 0.5rem;
}

.theme-count {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* ── Main chat (ChatGPT layout) ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 1.75rem 0 !important;
    border: none !important;
    max-width: var(--chat-width) !important;
    gap: 1.1rem !important;
    align-items: flex-start !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    background: var(--bg-user-row) !important;
    margin-left: calc(-50vw + 50%) !important;
    margin-right: calc(-50vw + 50%) !important;
    padding-left: max(1rem, calc(50vw - var(--chat-width) / 2)) !important;
    padding-right: max(1rem, calc(50vw - var(--chat-width) / 2)) !important;
    width: 100vw !important;
    max-width: 100vw !important;
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
    background: transparent !important;
    border-radius: 2px !important;
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    overflow: hidden !important;
    box-shadow: none !important;
    margin-top: 0.15rem !important;
}

[data-testid="stChatMessageAvatarAssistant"] img {
    width: 28px !important;
    height: 28px !important;
    object-fit: contain !important;
    border-radius: 2px !important;
    background: #ffffff !important;
    border: none !important;
    padding: 1px !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: var(--accent) !important;
    border-radius: 2px !important;
}

[data-testid="stChatMessageContent"] {
    color: var(--text) !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
    font-weight: 400 !important;
    padding-top: 0 !important;
    max-width: 100% !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    max-width: calc(var(--chat-width) - 3rem) !important;
    margin-left: auto !important;
    font-size: 1rem !important;
}

[data-testid="stChatMessageContent"] h3 {
    font-size: 0.9375rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 1.5rem 0 0.65rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

[data-testid="stChatMessageContent"] p {
    margin-bottom: 0.75rem !important;
}

[data-testid="stChatMessageContent"] ul {
    margin: 0.5rem 0 !important;
    padding-left: 1.25rem !important;
}

/* ── Input bar (ChatGPT pill) ── */
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(to top, var(--bg-main) 75%, transparent) !important;
    border-top: none !important;
    padding: 0.5rem 0 1.75rem !important;
    max-width: var(--chat-width) !important;
    margin: 0 auto !important;
}

[data-testid="stBottomBlockContainer"] .block-container {
    padding-bottom: 0 !important;
}

[data-testid="stChatInput"] > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 1.625rem !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.08) !important;
    max-width: var(--chat-width) !important;
    margin: 0 auto !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
    min-height: 24px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-testid="stChatInput"] button {
    color: var(--text-muted) !important;
}

[data-testid="stChatInput"] button:hover {
    color: var(--text) !important;
}

/* ── Welcome (ChatGPT empty state) ── */
.welcome-wrap {
    min-height: 52vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}

.welcome-title {
    font-size: 1.75rem;
    font-weight: 500;
    color: var(--text);
    margin: 0 0 0.45rem;
    letter-spacing: -0.03em;
}

.welcome-sub {
    font-size: 0.9375rem;
    color: var(--text-muted);
    margin: 0 0 2.25rem;
    max-width: 26rem;
    line-height: 1.55;
}

.starter-card-wrap {
    margin-top: 0.25rem;
}

.starter-card-label {
    display: block;
    text-align: left;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 0.35rem 0.15rem;
}

.starter-accent-wild + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-wild) .stButton > button { border-left-color: #2d6a4f !important; }
.starter-accent-heritage + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-heritage) .stButton > button { border-left-color: #b8860b !important; }
.starter-accent-scenic + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-scenic) .stButton > button { border-left-color: #0077b6 !important; }
.starter-accent-pristine + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-pristine) .stButton > button { border-left-color: #0096c7 !important; }
.starter-accent-essence + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-essence) .stButton > button { border-left-color: #7b2cbf !important; }
.starter-accent-thrills + div .stButton > button,
.main [data-testid="column"]:has(.starter-accent-thrills) .stButton > button { border-left-color: #e85d04 !important; }

/* Suggestion chips — card-style starters (column contains label + button) */
.main [data-testid="column"]:has(.starter-card-label) {
    padding: 0 0.25rem;
}

.main [data-testid="column"]:has(.starter-card-label) .stButton > button {
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--starter-accent, var(--accent)) !important;
    border-radius: 14px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.85rem 1rem !important;
    min-height: 4.25rem !important;
    line-height: 1.35 !important;
    text-align: left !important;
    white-space: normal !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transform: none !important;
    transition: background 0.15s, border-color 0.15s, box-shadow 0.15s, transform 0.15s !important;
}

.main [data-testid="column"]:has(.starter-card-label) .stButton > button:hover {
    background: #ffffff !important;
    border-color: rgba(0, 0, 0, 0.14) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06) !important;
    transform: translateY(-1px) !important;
    color: var(--text) !important;
}

.main [data-testid="column"]:has(.starter-card-label) .stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

/* Generic suggestion chips (fallback) */
.main .stButton > button {
    background: var(--bg-main) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-pill) !important;
    font-size: 0.8125rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 1rem !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background 0.15s, border-color 0.15s !important;
}

.main .stButton > button:hover {
    background: var(--bg-user-row) !important;
    border-color: rgba(0, 0, 0, 0.15) !important;
    color: var(--text) !important;
}

/* ── Place cards ── */
.place-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.75rem 0 1rem;
}

.place-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1rem 1.15rem;
}

.place-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.place-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.4;
    letter-spacing: -0.01em;
}

.place-location {
    flex-shrink: 0;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border-radius: 0;
    padding: 0;
    text-transform: none;
    letter-spacing: 0;
}

.place-desc {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
}

.place-reviews {
    margin-top: 0.85rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--border-subtle);
}

.place-review {
    font-size: 0.8125rem;
    color: var(--text-muted);
    line-height: 1.65;
    margin: 0.25rem 0 0;
}

.place-review.quote {
    font-style: italic;
    padding-left: 0;
    border-left: none;
}

/* ── Login ── */
.login-wrap {
    max-width: 22rem;
    margin: 5rem auto 2rem;
    text-align: center;
}

.login-wrap h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0.75rem 0 0.35rem;
    color: var(--text) !important;
}

.login-wrap p {
    color: var(--text-muted);
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
}

.stButton > button[kind="primary"] {
    background: var(--text) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
}

[data-testid="stTextInput"] label,
[data-testid="stTextInput"] p {
    color: var(--text) !important;
}

[data-testid="stTextInput"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}
</style>
""").strip()

_page_icon = str(LOGO_FILE) if LOGO_FILE else None

st.set_page_config(
    page_title="Travel Agent",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in [
    ("token", None),
    ("user", None),
    ("messages", []),
    ("conversation_id", None),
    ("dataset_meta", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_logo_data_uri() -> str | None:
    for path in LOGO_CANDIDATES:
        if not path.is_file():
            continue
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg", "jg"} else "png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/{mime};base64,{encoded}"
    return None


def _logo_html(*, large: bool = False, xl: bool = False, sidebar: bool = False) -> str:
    uri = _load_logo_data_uri()
    if uri:
        if sidebar:
            css_class, size = "brand-logo-sidebar", 40
        elif xl:
            css_class, size = "brand-logo-xxl", 72
        elif large:
            css_class, size = "brand-logo-lg", 72
        else:
            css_class, size = "brand-logo", 28
        return f'<img class="{css_class}" src="{uri}" alt="Travel Agent logo" width="{size}" height="{size}" />'
    if xl or large:
        return '<div class="sidebar-brand-icon brand-logo-lg">TA</div>'
    return '<div class="sidebar-brand-icon">TA</div>'


def html_block(content: str) -> None:
    st.markdown(content.strip(), unsafe_allow_html=True)


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h


@st.cache_data(ttl=300, show_spinner=False)
def fetch_dataset_meta(api_url: str) -> dict:
    """Load category tabs and counts from the attractions dataset via API."""
    try:
        r = requests.get(f"{api_url}/dataset/categories", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"categories": FALLBACK_CATEGORIES, "total_attractions": 0, "source": "fallback"}


def get_categories() -> list[dict]:
    meta = st.session_state.dataset_meta
    if not meta:
        meta = fetch_dataset_meta(API_URL)
        st.session_state.dataset_meta = meta
    return meta.get("categories") or FALLBACK_CATEGORIES


def get_total_attractions() -> int:
    meta = st.session_state.dataset_meta or {}
    return int(meta.get("total_attractions") or 0)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_card_block(block: str) -> dict | None:
    block = block.strip()
    if block.startswith("### "):
        lines = block.split("\n", 1)
        block = lines[1].strip() if len(lines) > 1 else ""
        if not block:
            return None

    lines = [ln.rstrip() for ln in block.strip().split("\n") if ln.strip()]
    if not lines:
        return None

    new_head = re.match(r"^- \*\*(.+?)\*\* · (.+)$", lines[0])
    if new_head:
        desc_parts: list[str] = []
        review_parts: list[str] = []
        for raw in lines[1:]:
            stripped = raw.strip()
            if stripped.startswith("Description:"):
                desc_parts.append(stripped[len("Description:"):].strip())
            elif stripped.startswith("Visitor feedback:"):
                review_parts.append(stripped[len("Visitor feedback:"):].strip())
            elif stripped.startswith('"') and stripped.endswith('"'):
                review_parts.append(stripped)
            elif stripped.startswith("_") and stripped.endswith("_"):
                review_parts.append(stripped.strip("_"))
            elif not desc_parts:
                desc_parts.append(stripped)
            else:
                review_parts.append(stripped)
        return {
            "name": new_head.group(1).strip(),
            "location": new_head.group(2).strip(),
            "desc": " ".join(desc_parts).strip(),
            "review": review_parts or None,
        }

    legacy_head = re.match(r"^\*\*(.+?)\*\*$", lines[0])
    if legacy_head and len(lines) > 1:
        legacy_body = re.match(r"^District: (.+?) · (.+)$", lines[1])
        if legacy_body:
            review = None
            if len(lines) > 2:
                review_line = lines[2].strip()
                if review_line.startswith("Visitor feedback:"):
                    review = [review_line[len("Visitor feedback:"):].strip()]
                elif not review_line.startswith("Description:"):
                    review = [review_line.lstrip("💬").strip()]
            return {
                "name": legacy_head.group(1).strip(),
                "location": legacy_body.group(1).strip(),
                "desc": legacy_body.group(2).strip(),
                "review": review,
            }
    return None


def _place_card_html(
    name: str,
    location: str,
    desc: str,
    review: str | list[str] | None = None,
) -> str:
    body_parts: list[str] = []
    if desc:
        body_parts.append(f'<p class="place-desc">{_escape_html(desc)}</p>')
    review_lines: list[str] = []
    if isinstance(review, list):
        review_lines = [r for r in review if r]
    elif review:
        review_lines = [review]
    if review_lines:
        review_html = ['<div class="place-reviews">']
        for line in review_lines:
            css = "place-review quote" if line.startswith('"') else "place-review"
            review_html.append(f'<p class="{css}">{_escape_html(line)}</p>')
        review_html.append("</div>")
        body_parts.append("".join(review_html))
    body = "".join(body_parts)
    return (
        f'<div class="place-card">'
        f'<div class="place-header">'
        f'<span class="place-name">{_escape_html(name)}</span>'
        f'<span class="place-location">{_escape_html(location)}</span>'
        f"</div>"
        f"{body}"
        f"</div>"
    )


def render_assistant_message(content: str) -> None:
    blocks = re.split(r"\n\n+", content.strip())
    rendered_cards = False
    pending: list[str] = []

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = "\n\n".join(pending).strip()
        pending = []
        if text:
            st.markdown(text)

    for block in blocks:
        if block.startswith("### "):
            flush_pending()
            if rendered_cards:
                html_block("</div>")
                rendered_cards = False
            st.markdown(block.split("\n", 1)[0])
            remainder = block.split("\n", 1)[1].strip() if "\n" in block else ""
            if remainder:
                for sub in re.split(r"\n\n+", remainder):
                    card = _parse_card_block(sub)
                    if card:
                        if not rendered_cards:
                            html_block('<div class="place-list">')
                            rendered_cards = True
                        html_block(_place_card_html(**card))
            continue

        card = _parse_card_block(block)
        if card:
            if not rendered_cards:
                flush_pending()
                html_block('<div class="place-list">')
                rendered_cards = True
            html_block(_place_card_html(**card))
            continue

        if rendered_cards:
            flush_pending()
            rendered_cards = False
            pending.append(block)
        else:
            pending.append(block)

    if rendered_cards:
        html_block("</div>")
    flush_pending()


def format_conversation_title(raw: str) -> str:
    match = re.match(r"^__start_planning__:(.+)$", raw.strip(), re.IGNORECASE)
    if match:
        cat = match.group(1).strip()
        return f"{cat} trip"
    starter = re.match(r"^__starter__:(.+)$", raw.strip(), re.IGNORECASE)
    if starter:
        payload = starter.group(1).strip()
        _, _, display = payload.partition("|")
        if display:
            return display[:36] + ("…" if len(display) > 36 else "")
        return f"{payload.split('|', 1)[0]} trip"
    return raw[:36] + ("…" if len(raw) > 36 else "")


def send_message(
    text: str,
    *,
    planning_category: str | None = None,
    start_planning: bool = False,
) -> str:
    payload: dict = {"message": text}
    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id
    if planning_category:
        payload["planning_category"] = planning_category
        payload["start_planning"] = True
    elif start_planning:
        payload["start_planning"] = True
    r = requests.post(f"{API_URL}/chat", json=payload, headers=headers(), timeout=180)
    if not r.ok:
        detail = r.text.strip() or f"HTTP {r.status_code}"
        try:
            detail = r.json().get("detail", detail)
        except (ValueError, requests.exceptions.JSONDecodeError):
            pass
        return f"Sorry, something went wrong: {detail}"
    try:
        data = r.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return (
            f"Sorry, the server returned an invalid response (HTTP {r.status_code}). "
            "Is the API running?"
        )
    st.session_state.conversation_id = data["conversation_id"]
    return data["response"]


def start_category_trip(category: str) -> None:
    st.session_state.pending_query = f"__start_planning__:{category}"
    st.session_state.pending_display = f"Plan a {category} trip"
    st.session_state.pending_category = category
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.rerun()


def sidebar() -> None:
    categories = get_categories()
    total = get_total_attractions()

    with st.sidebar:
        html_block(f"""
<div class="sidebar-brand">
  {_logo_html(sidebar=True)}
  <div>
    <p class="sidebar-brand-title">Travel Agent</p>
    <p class="sidebar-brand-sub">Sri Lanka planner</p>
  </div>
</div>
""")

        if total > 0:
            html_block(
                f'<p class="dataset-badge">{total} attractions · '
                f'{len([c for c in categories if c.get("count", 0) > 0])} themes</p>'
            )

        if st.button("New chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        html_block('<p class="sidebar-section">Recent chats</p>')
        try:
            r = requests.get(f"{API_URL}/conversations", headers=headers(), timeout=15)
            if r.ok:
                convs = r.json()[:12]
                if not convs:
                    st.caption("No conversations yet")
                for conv in convs:
                    title = format_conversation_title(conv["title"])
                    if st.button(title, key=f"h_{conv['id']}", use_container_width=True):
                        cr = requests.get(
                            f"{API_URL}/conversations/{conv['id']}",
                            headers=headers(),
                            timeout=15,
                        )
                        if cr.ok:
                            st.session_state.conversation_id = conv["id"]
                            st.session_state.messages = [
                                {"role": m["role"], "content": m["content"]}
                                for m in cr.json()["messages"]
                                if m["role"] in ("user", "assistant")
                            ]
                            st.rerun()
        except Exception:
            st.caption("Could not load history")

        html_block('<p class="sidebar-section">Trip themes · from dataset</p>')
        for cat in categories:
            name = cat["name"]
            count = cat.get("count", 0)
            desc = cat.get("description", "")
            count_label = f"{count} places" if count else "dataset"
            if st.button(
                f"{name}  ·  {count_label}",
                key=f"theme_{name}",
                use_container_width=True,
                help=desc or f"Plan a {name} trip using dataset attractions",
            ):
                start_category_trip(name)

        st.divider()
        user = st.session_state.user or {}
        name = user.get("full_name") or user.get("email", "Account")
        st.caption(name)
        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()


def _build_suggestions(categories: list[dict]) -> list[dict]:
    """Starter prompts aligned to dataset categories — sent as structured hints."""
    by_name = {c["name"]: c for c in categories}
    accents = {
        "Wild": "#2d6a4f",
        "Heritage": "#b8860b",
        "Scenic": "#0077b6",
        "Pristine": "#0096c7",
        "Essence": "#7b2cbf",
        "Thrills": "#e85d04",
    }
    templates: list[dict] = []
    if by_name.get("Wild", {}).get("count", 0):
        templates.append({
            "label": "Plan a 5-day wildlife trip from Colombo",
            "category": "Wild",
            "tag": "Wildlife",
            "accent": accents["Wild"],
            "payload": "__starter__:Wild|Plan a 5-day wildlife trip from Colombo",
        })
    if by_name.get("Heritage", {}).get("count", 0):
        templates.append({
            "label": "Heritage places near Kandy",
            "category": "Heritage",
            "tag": "Heritage",
            "accent": accents["Heritage"],
            "payload": "__starter__:Heritage|Heritage places near Kandy",
        })
    if by_name.get("Pristine", {}).get("count", 0):
        templates.append({
            "label": "Beach trip for 3 days",
            "category": "Pristine",
            "tag": "Beach",
            "accent": accents["Pristine"],
            "payload": "__starter__:Pristine|Beach trip for 3 days",
        })
    if by_name.get("Thrills", {}).get("count", 0):
        templates.append({
            "label": "Adventure day trips near Gampaha",
            "category": "Thrills",
            "tag": "Adventure",
            "accent": accents["Thrills"],
            "payload": "__starter__:Thrills|Adventure day trips near Gampaha",
        })
    if not templates:
        templates = [
            {
                "label": "Plan a 5-day trip from Colombo",
                "category": "Scenic",
                "tag": "Trip",
                "accent": accents["Scenic"],
                "payload": "Plan a 5-day trip from Colombo",
            },
            {
                "label": "Suggest places near Kandy",
                "category": "Heritage",
                "tag": "Explore",
                "accent": accents["Heritage"],
                "payload": "Heritage places near Kandy",
            },
            {
                "label": "Family-friendly nature trip",
                "category": "Essence",
                "tag": "Family",
                "accent": accents["Essence"],
                "payload": "Family-friendly nature trip",
            },
            {
                "label": "Weekend beach getaway",
                "category": "Pristine",
                "tag": "Beach",
                "accent": accents["Pristine"],
                "payload": "Beach trip for 3 days",
            },
        ]
    return templates[:4]


def welcome_screen() -> None:
    categories = get_categories()
    total = get_total_attractions()
    sub = (
        f"Every suggestion comes from our Sri Lanka attractions dataset "
        f"({total} places)." if total else "Every suggestion comes from our Sri Lanka attractions dataset."
    )
    html_block(f"""
<div class="welcome-wrap">
  {_logo_html(xl=True)}
  <p class="welcome-title">Where to today?</p>
  <p class="welcome-sub">{sub}</p>
</div>
""")
    suggestions = _build_suggestions(categories)
    cols = st.columns(2)
    for i, item in enumerate(suggestions):
        tag = item.get("tag", item.get("category", "Trip"))
        category = item.get("category", "Scenic")
        label = item["label"]
        with cols[i % 2]:
            html_block(
                f'<div class="starter-card-wrap">'
                f'<span class="starter-card-label starter-accent-{category.lower()}">{tag}</span></div>'
            )
            if st.button(label, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = item.get("payload", label)
                st.session_state.pending_display = label
                st.rerun()


def login_view() -> None:
    inject_css()
    html_block(f"""
<div class="login-wrap">
  {_logo_html(large=True)}
  <h1>Welcome back</h1>
  <p>Sign in to plan trips from the attractions dataset</p>
</div>
""")
    tab_login, tab_reg = st.tabs(["Log in", "Sign up"])
    with tab_login:
        email = st.text_input("Email address", key="li_email", placeholder="name@example.com")
        pw = st.text_input("Password", type="password", key="li_pw", placeholder="Enter your password")
        if st.button("Continue", use_container_width=True, type="primary"):
            r = requests.post(f"{API_URL}/auth/login", json={"email": email, "password": pw}, timeout=30)
            if r.ok:
                d = r.json()
                st.session_state.token = d["access_token"]
                st.session_state.user = d
                st.session_state.dataset_meta = None
                st.rerun()
            else:
                st.error(r.json().get("detail", "Invalid credentials"))
    with tab_reg:
        email = st.text_input("Email address", key="reg_email", placeholder="name@example.com")
        name = st.text_input("Full name", key="reg_name", placeholder="Optional")
        pw = st.text_input("Password", type="password", key="reg_pw", placeholder="At least 6 characters")
        if st.button("Create account", use_container_width=True, type="primary"):
            r = requests.post(
                f"{API_URL}/auth/register",
                json={"email": email, "password": pw, "full_name": name or None},
                timeout=30,
            )
            if r.ok:
                d = r.json()
                st.session_state.token = d["access_token"]
                st.session_state.user = d
                st.session_state.dataset_meta = None
                st.rerun()
            else:
                st.error(r.json().get("detail", "Registration failed"))


def chat_view() -> None:
    inject_css()
    if st.session_state.dataset_meta is None:
        st.session_state.dataset_meta = fetch_dataset_meta(API_URL)
    sidebar()

    if not st.session_state.messages:
        welcome_screen()

    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                render_assistant_message(msg["content"])
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input("Message Travel Agent")
    display_prompt = prompt
    planning_category = None

    if "pending_query" in st.session_state:
        prompt = st.session_state.pop("pending_query")
        display_prompt = st.session_state.pop("pending_display", prompt)
        planning_category = st.session_state.pop("pending_category", None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": display_prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(display_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner(""):
                response = send_message(
                    prompt,
                    planning_category=planning_category,
                    start_planning=bool(planning_category),
                )
            render_assistant_message(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if st.session_state.token:
    chat_view()
else:
    login_view()

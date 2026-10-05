"""Streamlit interface for the TelecomRAGent FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180


def inject_styles() -> None:
    """Apply the visual system for the operations console."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #12212b;
            --muted: #64747d;
            --line: #dce5e5;
            --paper: #f5f8f7;
            --panel: #ffffff;
            --navy: #102b3a;
            --teal: #0d8f88;
            --teal-soft: #dff4f0;
            --amber: #e49a37;
        }

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: var(--navy); border-right: 0; }
        [data-testid="stSidebar"] * { color: #eaf4f2; }
        [data-testid="stSidebar"] .stCaption { color: #a9c3c3; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff; }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #a9c3c3; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        .block-container { max-width: 1180px; padding: 2.2rem 3rem 5rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
        h1 { font-size: 2.45rem !important; line-height: 1.08 !important; margin-bottom: .35rem !important; }
        .eyebrow { color: var(--teal); font-size: .73rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .subhead { color: var(--muted); font-size: 1rem; margin-bottom: 1.6rem; }
        .hero { background: linear-gradient(115deg, #102b3a 0%, #164b55 70%, #117b76 100%); border-radius: 16px; padding: 1.45rem 1.65rem; color: white; margin: .4rem 0 1.4rem; box-shadow: 0 14px 35px rgba(16,43,58,.14); }
        .hero .eyebrow { color: #86e5d5; }
        .hero h2 { color: white; font-size: 1.35rem; margin: .2rem 0 .35rem; }
        .hero p { color: #d3e7e4; margin: 0; font-size: .92rem; }
        .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; min-height: 94px; }
        .kpi-label { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
        .kpi-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; margin-top: .3rem; }
        .kpi-note { color: var(--muted); font-size: .76rem; margin-top: .15rem; }
        [data-testid="stChatMessage"] { border: 0; padding: .8rem 0; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height: 1.65; }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1.05rem 1.2rem; margin: .55rem 0 1rem; }
        [data-testid="stChatInput"] { border-color: #9dc9c5; box-shadow: 0 7px 20px rgba(13,143,136,.08); }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: rgba(255,255,255,.7); }
        .source-row { border-left: 3px solid var(--teal); background: #f7fbfa; border-radius: 0 8px 8px 0; padding: .75rem .9rem; margin: .55rem 0; }
        .source-meta { color: var(--muted); font-size: .76rem; margin-top: .35rem; }
        .status-pill { display: inline-block; border-radius: 99px; background: var(--teal-soft); color: #086e69; font-size: .72rem; font-weight: 700; padding: .3rem .65rem; }
        .stButton > button, .stDownloadButton > button { border-radius: 8px; border: 1px solid #b8d3d1; color: #086e69; background: white; font-weight: 600; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--teal); color: var(--teal); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_json(path: str) -> dict[str, Any]:
    """Fetch a JSON response from the backend or raise a useful request error."""
    response = requests.get(
        f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def submit_query(question: str, top_k: int) -> dict[str, Any]:
    """Submit a question to the backend query endpoint."""
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question, "top_k": top_k},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved evidence in a compact expandable section."""
    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            location = metadata.get("region", "Unknown region")
            operator = metadata.get("operator", "Unknown operator")
            severity = str(metadata.get("severity", "unknown")).title()
            st.markdown(
                f"<div class='source-row'><strong>{index}. {location} · {operator}</strong>"
                f"<br>{source['document']}"
                f"<div class='source-meta'>{severity} · {metadata.get('network_type', 'Unknown')} · "
                f"{metadata.get('timestamp', 'Unknown date')} · distance {source.get('distance', 0):.3f}</div></div>",
                unsafe_allow_html=True,
            )
            with st.expander("Metadata", expanded=False):
                st.json(metadata)


st.set_page_config(page_title="TelecomRAGent", page_icon="📡", layout="wide", initial_sidebar_state="expanded")
inject_styles()
st.markdown("<div class='eyebrow'>Network intelligence console</div>", unsafe_allow_html=True)
st.title("TelecomRAGent")
st.markdown("<div class='subhead'>Investigate Indian mobile voice quality with evidence you can trace.</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("<div class='eyebrow'>Live telemetry</div>", unsafe_allow_html=True)
    st.header("Network pulse")
    st.caption("Connected to the local retrieval and reasoning stack")
    top_k = st.slider("Evidence records", min_value=1, max_value=10, value=5)
    try:
        health = get_json("/health")
        stats = get_json("/stats")
        st.markdown(f"<span class='status-pill'>● Backend {health['status']}</span>", unsafe_allow_html=True)
        st.metric("Indexed records", f"{stats['indexed_records']:,}")
        st.metric("Regions", stats["regions"])
        st.metric("Operators", len(stats["operators"]))
        st.divider()
        st.caption("Network types")
        st.write(" · ".join(stats["network_types"]))
        st.caption("Severity mix")
        for severity, count in sorted(stats["severity_counts"].items()):
            st.write(f"{severity.title()}  ·  {count:,}")
    except requests.RequestException as error:
        st.error(f"Backend unavailable: {error}")

if not st.session_state.messages:
    st.markdown(
        "<div class='hero'><div class='eyebrow'>Investigation workspace</div>"
        "<h2>Ask a question about the network.</h2>"
        "<p>Trace call drops, compare operators, and surface the next operational action.</p></div>",
        unsafe_allow_html=True,
    )
    kpi_columns = st.columns(3)
    for column, label, value, note in zip(
        kpi_columns,
        ["Evidence layer", "Reasoning", "Output"],
        ["Semantic", "4 tools", "Traceable"],
        ["Chroma retrieval", "Ordered analysis", "Sources + report"],
    ):
        with column:
            st.markdown(
                f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>",
                unsafe_allow_html=True,
            )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            with st.expander("Tool trace", expanded=False):
                st.write(" -> ".join(message.get("tool_trace", [])))

question = st.chat_input("Ask about call quality, operators, regions, or call drops")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = submit_query(question, top_k)
            st.markdown(result["answer"])
            render_sources(result.get("sources", []))
            with st.expander("Tool trace", expanded=False):
                st.write(" -> ".join(result.get("tool_trace", [])))
            st.download_button(
                "Download report",
                data=result["answer"],
                file_name="telecomragent-report.txt",
                mime="text/plain",
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "tool_trace": result.get("tool_trace", []),
                }
            )
        except requests.RequestException as error:
            message = f"The backend request failed: {error}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
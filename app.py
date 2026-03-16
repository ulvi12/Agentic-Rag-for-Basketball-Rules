import streamlit as st
from dotenv import load_dotenv
from src.rag import answer, _format_source_label
import time

load_dotenv()

RATE_LIMIT_MAX = 50
RATE_LIMIT_WINDOW = 30 * 60

st.set_page_config(page_title="Basketball Rules RAG", page_icon="🏀", layout="wide")

st.title("Basketball Rules Chat")
st.markdown("Ask anything about NBA, WNBA, NCAA, or FIBA rules!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_timestamps" not in st.session_state:
    st.session_state.question_timestamps = []

with st.sidebar:
    st.header("Settings")
    show_trace = st.checkbox("Show Reasoning Trace", value=True)
    league_filter = st.selectbox("League Filter (Optional)", ["None", "NBA", "WNBA", "NCAA", "FIBA"])
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Cited Sources"):
                for i, meta in enumerate(msg["sources"], start=1):
                    st.write(f"[{i}] {_format_source_label(meta)}")
        if "trace" in msg and msg["trace"] and show_trace:
            with st.expander("🔎 Agent Reasoning Trace"):
                st.json(msg["trace"])

now = time.time()
st.session_state.question_timestamps = [
    t for t in st.session_state.question_timestamps if now - t < RATE_LIMIT_WINDOW
]
rate_limited = len(st.session_state.question_timestamps) >= RATE_LIMIT_MAX

if rate_limited:
    st.chat_input("How many personal fouls gets you disqualified in NBA vs FIBA?", disabled=True)
    st.warning(f"You've reached the limit of {RATE_LIMIT_MAX} questions per 30 minutes. Please wait a while before asking more.")
elif prompt := st.chat_input("How many personal fouls gets you disqualified in NBA vs FIBA?"):
    st.session_state.question_timestamps.append(now)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Agents are thinking...")

        def on_status(msg):
            status.info(msg)

        with st.spinner():
            league_arg = league_filter if league_filter != "None" else None

            history_for_rag = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]][-10:]

            result = answer(prompt, league=league_arg, history=history_for_rag, trace=True, on_status=on_status)

            status.empty()

            ans_text = result["answer"]
            st.write(ans_text)

            if result.get("sources"):
                with st.expander("📚 Cited Sources"):
                    for i, meta in enumerate(result["sources"], start=1):
                        st.write(f"[{i}] {_format_source_label(meta)}")

            if result.get("trace") and show_trace:
                with st.expander("🔎 Agent Reasoning Trace"):
                    st.json(result["trace"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": ans_text,
        "sources": result.get("sources", []),
        "all_sources": result.get("all_sources", []),
        "trace": result.get("trace", {})
    })

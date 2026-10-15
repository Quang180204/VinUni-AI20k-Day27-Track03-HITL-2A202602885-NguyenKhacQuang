"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import uuid

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from common.schemas import risk_level_for
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Database helper for recent sessions ────────────────────────────────────
def get_recent_sessions(limit: int = 15) -> list[dict]:
    """Retrieve recent review sessions from audit_events table."""
    path = db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_thread_events(thread_id: str) -> list[dict]:
    """Retrieve all audit events for a given thread."""
    path = db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT timestamp, action, confidence, risk_level,
                   reviewer_id, decision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id ASC
            """,
            (thread_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HITL PR Review Agent")
st.caption("Human-in-the-Loop automated code review triaged by LangGraph & LLM confidence scoring")


# ─── Sidebar — recent sessions & legend ─────────────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    sessions = get_recent_sessions()
    if sessions:
        for s in sessions:
            tid = s["thread_id"]
            short_tid = tid[:8]
            short_url = s["pr_url"].replace("https://github.com/", "")
            risk = s.get("worst_risk", "med")
            risk_emoji = "🟢" if risk == "low" else ("🟡" if risk == "med" else "🔴")
            
            label = f"{risk_emoji} `{short_tid}` — {short_url}"
            if st.button(label, key=f"session_{tid}"):
                st.session_state.thread_id = tid
                st.session_state.pr_url = s["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = {"final_action": "loaded_from_history"}
                st.rerun()
    else:
        st.info("No recorded sessions yet in SQLite.")

    st.divider()
    st.markdown("### ⚙️ Confidence Routing")
    st.markdown("""
    - **> 72%** (`auto_approve`): 🟢 Direct commit, no human needed
    - **58–72%** (`human_approval`): 🟡 Approve / Reject / Edit panel
    - **< 58%** (`escalate`): 🔴 Clarification Q&A then re-synthesize
    """)


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    st.markdown("##### Submit a Pull Request for Automated HITL Review")
    col_input, col_preset = st.columns([3, 1])
    with col_input:
        pr_url = st.text_input(
            "PR URL",
            value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
    with col_preset:
        preset = st.selectbox(
            "Quick Demo PRs",
            ["(Custom URL)", "PR #1 (Priority field ~65%)", "PR #2 (Auth & Sync <58%)"],
        )
        if preset == "PR #1 (Priority field ~65%)":
            pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
        elif preset == "PR #2 (Auth & Sync <58%)":
            pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"

    submitted = st.form_submit_button("🚀 Run Review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    risk = risk_level_for(conf)
    
    st.warning(f"⚠️ **Approval Requested** — Confidence: **{conf:.0%}** (Risk: **{risk.upper()}**)")
    st.info(f"**Reasoning:** {payload['confidence_reasoning']}")

    st.markdown("### 📝 Review Summary")
    st.markdown(payload["summary"])

    comments = payload.get("comments", [])
    if comments:
        st.markdown(f"### 💬 Proposed Comments ({len(comments)})")
        for c in comments:
            sev = c.get("severity", "suggestion")
            sev_color = {
                "blocker": "🔴",
                "issue": "🟠",
                "suggestion": "🔵",
                "nit": "⚪",
            }.get(sev, "🔵")
            file_path = c.get("file", "unknown")
            line = c.get("line") or "?"
            st.markdown(f"- {sev_color} **[{sev.upper()}]** `{file_path}:{line}` — {c.get('body', '')}")

    diff_preview = payload.get("diff_preview", "")
    if diff_preview:
        with st.expander("🔍 View Diff Preview"):
            st.code(diff_preview, language="diff")

    st.divider()
    feedback = st.text_input("Reviewer Feedback (optional note to include on GitHub):", key="approval_feedback")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve & Post Comment", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject (Skip Posting)", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit Review", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    risk = risk_level_for(conf)
    
    st.error(f"🚨 **Strong Escalation** — Confidence: **{conf:.0%}** (Risk: **{risk.upper()}**)")
    st.caption(f"**Reasoning:** {payload['confidence_reasoning']}")
    
    if payload.get("risk_factors"):
        st.error("**Identified Risk Factors:**\n- " + "\n- ".join(payload["risk_factors"]))
    
    st.markdown("### 📝 Preliminary Summary")
    st.markdown(payload["summary"])

    st.markdown("### ❓ Clarification Questions for Reviewer")
    st.markdown("The agent detected high-risk patterns. Please answer the clarifying questions below so the model can synthesize an accurate, refined review:")

    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}:** {q}")
            answers[q] = st.text_area(f"Your answer to Q{i+1}:", key=f"esc_q_{i}", height=70)
        
        submit_esc = st.form_submit_button("⚡ Submit Answers & Re-Synthesize", type="primary")
        if submit_esc:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(
                Command(resume=resume_value), cfg
            )
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("⏳ Fetching PR from GitHub & analyzing code changes with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind", "approval_request")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("⏳ Resuming review pipeline & recording decision to audit log..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    
    st.markdown("---")
    if action == "auto_approved" or action.startswith("auto"):
        st.success(f"🎉 **Auto-Approved!** High confidence review comment posted directly to [GitHub PR]({st.session_state.pr_url}) without human intervention.")
    elif action in ("committed", "committed_after_escalation"):
        st.success(f"✅ **Review Successfully Posted!** Status: `{action}`. Comment is live on [GitHub PR]({st.session_state.pr_url}).")
    elif action == "rejected":
        st.warning("🛑 **Review Rejected:** Human reviewer rejected the review. No comment was posted to GitHub.")
    elif action == "loaded_from_history":
        st.info(f"Viewing historical session `{st.session_state.thread_id}`.")
    else:
        st.info(f"Final Action: `{action}`")

    if st.session_state.pr_url:
        st.link_button("🌐 Open Pull Request on GitHub", st.session_state.pr_url)

    # Show structured audit trail for this thread
    if st.session_state.thread_id:
        events = get_thread_events(st.session_state.thread_id)
        if events:
            with st.expander("📊 Audit Trail Timeline (SQLite `audit_events`)", expanded=True):
                st.table([
                    {
                        "Action": e["action"],
                        "Confidence": f"{e['confidence']:.0%}",
                        "Risk": e["risk_level"].upper(),
                        "Decision": e["decision"],
                        "Reviewer": e["reviewer_id"] or "-",
                        "Time (ms)": f"{e['execution_time_ms']} ms",
                        "Reason / Notes": (e["reason"] or "")[:80],
                    }
                    for e in events
                ])

    st.caption(
        f"**thread_id:** `{st.session_state.thread_id}`  ·  "
        f"**Replay CLI:** `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )

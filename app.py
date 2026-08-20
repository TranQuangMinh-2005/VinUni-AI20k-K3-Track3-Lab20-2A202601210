"""Streamlit UI — Multi-Agent Research Lab.

Vibe: Ethereal Glass (OLED #050505, radial glow orbs, hairline white/10 borders).
Layout: Asymmetrical bento — flow diagram (2/3) + metrics (1/3), editorial answer card.

Chạy bằng:  streamlit run app.py
"""

import html
from time import perf_counter

import streamlit as st

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    make_baseline_runner,
    make_multi_agent_runner,
)
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing

st.set_page_config(
    page_title="Multi-Agent Research Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

configure_logging(get_settings().log_level)
setup_tracing()

# ------------------------------------------------------------------------------------
# Design system (injected CSS)
# Fonts: Clash Display (display) + Plus Jakarta Sans (body) — no Inter/Roboto/Arial.
# Motion: single custom cubic-bezier curve, GPU-safe (transform/opacity only).
# ------------------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #050505;
    --card: #0a0a0c;
    --hairline: rgba(255,255,255,0.08);
    --hairline-strong: rgba(255,255,255,0.14);
    --text: #f4f4f5;
    --text-dim: #a1a1aa;
    --violet: #8b5cf6;
    --emerald: #34d399;
    --ease: cubic-bezier(.32,.72,0,1);
}

html, body, .stApp { background: var(--bg) !important; }
.stApp { font-family: 'Plus Jakarta Sans', sans-serif; color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1160px; padding: 2.5rem 2rem 6rem; position: relative; z-index: 1; }

/* --- Ambient glow orbs (fixed, pointer-events none, no filter-blur) --- */
.orb { position: fixed; border-radius: 50%; pointer-events: none; z-index: 0; }
.orb-1 { width: 720px; height: 720px; top: -220px; right: -180px;
         background: radial-gradient(circle at center, rgba(139,92,246,0.15), transparent 62%); }
.orb-2 { width: 900px; height: 900px; bottom: -320px; left: -260px;
         background: radial-gradient(circle at center, rgba(52,211,153,0.09), transparent 62%); }
.orb-3 { width: 420px; height: 420px; top: 38%; left: 55%;
         background: radial-gradient(circle at center, rgba(56,189,248,0.07), transparent 65%); }

/* --- Film grain (fixed overlay, 3%) --- */
.grain { position: fixed; inset: 0; z-index: 2; pointer-events: none; opacity: 0.035;
         background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }

/* --- Sticky glass nav pill (blur only on sticky layer) --- */
.nav { position: sticky; top: 14px; z-index: 30; display: flex; align-items: center;
       justify-content: space-between; padding: 10px 18px 10px 22px; margin-bottom: 3.2rem;
       border-radius: 999px; background: rgba(10,10,12,0.62);
       border: 1px solid var(--hairline); backdrop-filter: blur(18px);
       -webkit-backdrop-filter: blur(18px); }
.nav-brand { display: flex; align-items: center; gap: 12px; font-weight: 600; letter-spacing: -0.01em; }
.nav-dot { width: 10px; height: 10px; border-radius: 50%;
           background: radial-gradient(circle at 30% 30%, #a78bfa, #7c3aed 70%);
           box-shadow: 0 0 14px rgba(139,92,246,0.7); }
.nav-status { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-dim); }
.nav-status b { color: var(--emerald); font-weight: 600; }
.nav a { color: var(--text-dim); text-decoration: none; font-size: 12.5px; font-weight: 500;
         padding: 8px 16px; border-radius: 999px; border: 1px solid var(--hairline);
         transition: all .6s var(--ease); }
.nav a:hover { color: var(--text); border-color: var(--hairline-strong);
               background: rgba(255,255,255,0.05); }

/* --- Typography --- */
.eyebrow { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
           border-radius: 999px; background: rgba(255,255,255,0.04);
           border: 1px solid var(--hairline); font-size: 10px; letter-spacing: 0.24em;
           text-transform: uppercase; color: var(--text-dim); font-weight: 600; }
.eyebrow .pulse { width: 6px; height: 6px; border-radius: 50%; background: var(--emerald);
                  box-shadow: 0 0 10px rgba(52,211,153,0.9); }
h1.display { font-family: 'Clash Display', sans-serif; font-weight: 600;
             font-size: clamp(42px, 6vw, 74px); line-height: 1.02; letter-spacing: -0.025em;
             margin: 1.4rem 0 1.1rem; }
h1.display .grad { background: linear-gradient(100deg, #f4f4f5 30%, #a78bfa 65%, #34d399 95%);
                   -webkit-background-clip: text; background-clip: text; color: transparent; }
.lede { color: var(--text-dim); font-size: 16.5px; font-weight: 300; line-height: 1.65;
        max-width: 640px; margin: 0 0 2.6rem; }

/* --- Double-bezel card (outer shell + inner core) --- */
.bezel { background: rgba(255,255,255,0.035); border: 1px solid var(--hairline);
         border-radius: 2rem; padding: 6px; }
.core { background: var(--card); border-radius: calc(2rem - 6px);
        border: 1px solid rgba(255,255,255,0.055);
        box-shadow: inset 0 1px 1px rgba(255,255,255,0.06); padding: 26px 28px; }
.card-title { font-family: 'Clash Display', sans-serif; font-weight: 500; font-size: 13px;
              letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-dim);
              margin: 0 0 18px; display: flex; align-items: center; gap: 10px; }
.card-title::before { content: ''; width: 7px; height: 7px; border-radius: 50%;
                      background: var(--violet); box-shadow: 0 0 12px rgba(139,92,246,0.8); }

/* --- Flow diagram --- */
.flow { display: flex; flex-wrap: wrap; align-items: stretch; justify-content: center; gap: 2px; }
.node { display: flex; flex-direction: column; align-items: center; justify-content: center;
        padding: 16px 22px; border-radius: 20px; min-width: 148px;
        background: rgba(255,255,255,0.035); border: 1px solid var(--hairline);
        transition: transform .6s var(--ease), border-color .6s var(--ease); }
.node:hover { transform: translateY(-3px); border-color: var(--hairline-strong); }
.node .glyph { font-size: 20px; font-weight: 300; margin-bottom: 6px; }
.node .who { font-family: 'Clash Display', sans-serif; font-weight: 600; font-size: 14.5px;
             letter-spacing: -0.01em; }
.node .why { color: var(--text-dim); font-size: 10.5px; margin-top: 5px; text-align: center;
             max-width: 160px; line-height: 1.45; }
.node.sup { border-color: rgba(255,255,255,0.16); background: rgba(255,255,255,0.055); }
.node.done { border-color: rgba(52,211,153,0.35); background: rgba(52,211,153,0.07); }
.conn { align-self: center; color: rgba(255,255,255,0.28); font-size: 18px; font-weight: 300;
        padding: 0 2px; }

/* --- Stat tiles --- */
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stat { border-radius: 18px; padding: 18px 20px; background: rgba(255,255,255,0.03);
        border: 1px solid var(--hairline); }
.stat .k { font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-dim); }
.stat .v { font-family: 'Clash Display', sans-serif; font-weight: 600; font-size: 26px;
           letter-spacing: -0.02em; margin-top: 6px; }
.stat .v small { font-size: 13px; color: var(--text-dim); font-weight: 500; }

/* --- Answer typography --- */
.answer { font-size: 15.5px; line-height: 1.75; color: #d4d4d8; font-weight: 300; }
.answer h1, .answer h2, .answer h3 { font-family: 'Clash Display', sans-serif;
        color: var(--text); letter-spacing: -0.01em; margin-top: 1.2em; }
.answer strong { color: var(--text); font-weight: 600; }
.answer li { margin-bottom: 6px; }
.notes { font-size: 13.5px; line-height: 1.7; color: #a1a1aa; white-space: pre-wrap;
         font-weight: 300; }
.src { display: block; padding: 12px 16px; margin-bottom: 8px; border-radius: 14px;
       background: rgba(255,255,255,0.025); border: 1px solid var(--hairline);
       text-decoration: none; color: #d4d4d8; transition: all .6s var(--ease); }
.src:hover { border-color: var(--hairline-strong); background: rgba(255,255,255,0.05);
             transform: translateX(4px); }
.src b { font-weight: 600; color: var(--text); }
.src .u { display: block; font-size: 11.5px; color: var(--text-dim); margin-top: 3px; }

/* --- Trace timeline --- */
.tl { border-left: 1px solid rgba(255,255,255,0.1); margin-left: 8px; padding-left: 22px; }
.tl-item { position: relative; margin-bottom: 12px; font-size: 12.5px; color: #a1a1aa; }
.tl-item::before { content: ''; position: absolute; left: -27px; top: 5px; width: 9px; height: 9px;
                   border-radius: 50%; background: var(--violet);
                   box-shadow: 0 0 12px rgba(139,92,246,0.55); }
.tl-item .t { font-weight: 600; color: #e4e4e7; font-size: 13px; }
.tl-item .p { font-size: 11px; color: #71717a; margin-top: 2px; word-break: break-word; }

/* --- Error banner --- */
.err { border-radius: 16px; padding: 16px 20px; margin-top: 14px;
       background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.3);
       color: #fda4af; font-size: 13px; }

/* --- Entry animation (staggered fade-up, GPU-safe) --- */
.fade-up { opacity: 0; animation: fadeUp .9s var(--ease) forwards; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); filter: blur(6px); }
    to   { opacity: 1; transform: none; filter: none; }
}

/* --- Streamlit widget skin --- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--hairline) !important; border-radius: 28px !important;
    background: rgba(255,255,255,0.03) !important; padding: 8px; }
.stTextInput input {
    border-radius: 20px !important; background: var(--card) !important;
    border: 1px solid var(--hairline) !important; color: var(--text) !important;
    padding: 13px 20px !important; font-family: 'Plus Jakarta Sans' !important;
    font-size: 15px !important; transition: all .6s var(--ease) !important; }
.stTextInput input:focus {
    border-color: rgba(139,92,246,0.55) !important;
    box-shadow: 0 0 0 4px rgba(139,92,246,0.12) !important; }
.stButton > button, .stLinkButton > a, a.stLinkButton {
    width: 100%; border-radius: 999px !important; padding: 15px 22px !important;
    background: rgba(255,255,255,0.05) !important; color: var(--text) !important;
    border: 1px solid var(--hairline-strong) !important;
    font-family: 'Plus Jakarta Sans' !important; font-weight: 600 !important;
    font-size: 14px !important; letter-spacing: 0.01em;
    transition: all .7s var(--ease) !important; }
.stButton > button:hover, .stLinkButton > a:hover {
    background: rgba(255,255,255,0.09) !important; transform: translateY(-2px);
    border-color: rgba(255,255,255,0.24) !important; }
.stButton > button:active { transform: scale(0.98) !important; }
.stButton > button p::after, .stLinkButton > a p::after {
    content: '→'; display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; margin-left: 10px; border-radius: 50%;
    background: rgba(255,255,255,0.1); font-weight: 500; vertical-align: middle; }
[data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(#0b0b0e, #0b0b0e) padding-box,
                linear-gradient(135deg, rgba(139,92,246,0.8), rgba(52,211,153,0.55)) border-box
                !important;
    border: 1px solid transparent !important;
    box-shadow: 0 8px 32px rgba(139,92,246,0.18) !important; }
[data-testid="stBaseButton-primary"] > button:hover { box-shadow: 0 12px 44px rgba(139,92,246,0.3) !important; }
details[data-testid="stExpander"] {
    border-radius: 18px; background: rgba(255,255,255,0.025) !important;
    border: 1px solid var(--hairline) !important; overflow: hidden; margin-bottom: 10px; }
details[data-testid="stExpander"] summary { padding: 14px 20px; color: #d4d4d8; }
.stSpinner > div { border-top-color: var(--violet) !important; }
@media (max-width: 768px) {
    .stats { grid-template-columns: 1fr; }
    .block-container { padding: 1.6rem 1rem 4rem; }
}
</style>
<div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>
<div class="grain"></div>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def trace_summary(state: ResearchState) -> dict:
    """Gom latency/cost/token/errors từ state.trace."""
    total_cost = sum(
        float(e["payload"]["cost_usd"])
        for e in state.trace
        if e.get("payload", {}).get("cost_usd") is not None
    )
    total_in = sum(
        e["payload"]["input_tokens"]
        for e in state.trace
        if e.get("payload", {}).get("input_tokens") is not None
    )
    total_out = sum(
        e["payload"]["output_tokens"]
        for e in state.trace
        if e.get("payload", {}).get("output_tokens") is not None
    )
    return {
        "cost": total_cost,
        "in": total_in,
        "out": total_out,
        "errors": len(state.errors),
        "events": len(state.trace),
    }


def render_flow(state: ResearchState) -> str:
    """HTML sơ đồ luồng: Supervisor → worker → ... với lý do route ở mỗi node."""
    route_events = [e for e in state.trace if e["name"] == "route"]
    if not route_events:
        return '<div class="notes">Chưa có dữ liệu luồng.</div>'

    glyphs = {
        "supervisor": "◈",
        "researcher": "◐",
        "analyst": "◒",
        "writer": "◓",
        "done": "✓",
    }
    names = {
        "supervisor": "Supervisor",
        "researcher": "Researcher",
        "analyst": "Analyst",
        "writer": "Writer",
        "done": "Complete",
    }
    parts: list[str] = ['<div class="flow">']
    for i, event in enumerate(route_events):
        route = event["payload"]["next"]
        reason = html.escape(event["payload"]["reason"])
        cls = "node done" if route == "done" else "node"
        if route == "supervisor":
            cls += " sup"
        parts.append(
            f'<div class="{cls}"><div class="glyph">{glyphs.get(route, "•")}</div>'
            f'<div class="who">{names.get(route, route)}</div>'
            f'<div class="why">{reason}</div></div>'
        )
        if i < len(route_events) - 1:
            parts.append('<span class="conn">→</span>')
    parts.append("</div>")
    return "".join(parts)


def card(title: str, body_html: str, delay: float = 0) -> str:
    """Double-bezel card với hiệu ứng fade-up có độ trễ."""
    return (
        f'<div class="bezel fade-up" style="animation-delay: {delay:.2f}s">'
        f'<div class="core"><div class="card-title">{title}</div>{body_html}</div></div>'
    )


def render_result(run_key: str, state: ResearchState, latency: float) -> None:
    """Render toàn bộ kết quả 1 lần chạy (bento grid)."""
    summary = trace_summary(state)

    st.markdown(
        card("Execution flow", render_flow(state), 0.05),
        unsafe_allow_html=True,
    )

    row = st.columns([2, 1], gap="medium")
    with row[0]:
        answer = state.final_answer or "_Chưa có final answer._"
        st.markdown(
            card("Final answer", f'<div class="answer">{answer}</div>', 0.1),
            unsafe_allow_html=True,
        )
    with row[1]:
        stats_html = (
            '<div class="stats">'
            f'<div class="stat"><div class="k">Latency</div><div class="v">{latency:.2f}<small> s</small></div></div>'
            f'<div class="stat"><div class="k">Tokens</div><div class="v">{summary["in"]}<small> in</small> · {summary["out"]}<small> out</small></div></div>'
            f'<div class="stat"><div class="k">Est. cost</div><div class="v">${summary["cost"]:.5f}</div></div>'
            f'<div class="stat"><div class="k">Events / Errors</div><div class="v">{summary["events"]}<small> · </small>{summary["errors"]}</div></div>'
            "</div>"
        )
        st.markdown(card("Run metrics", stats_html, 0.15), unsafe_allow_html=True)

    if state.errors:
        err_html = "<div class='err'>" + "<br>".join(
            html.escape(e) for e in state.errors
        ) + "</div>"
        st.markdown(err_html, unsafe_allow_html=True)

    if state.sources:
        src_html = "".join(
            f'<a class="src" href="{html.escape(doc.url or "#")}" target="_blank">'
            f"<b>[{i}] {html.escape(doc.title)}</b>"
            + ("<span style='color:#34d399'> · mock</span>" if doc.metadata.get("mock") else "")
            + (f"<span style='color:#71717a'> · score {doc.metadata['score']}</span>" if doc.metadata.get("score") is not None else "")
            + f"<span class='u'>{html.escape(doc.url or 'n/a')}</span></a>"
            for i, doc in enumerate(state.sources, start=1)
        )
        st.markdown(card(f"Sources ({len(state.sources)})", src_html, 0.2), unsafe_allow_html=True)

    notes_cols = st.columns(2, gap="medium")
    with notes_cols[0]:
        if state.research_notes:
            st.markdown(
                card("Research notes", f'<div class="notes">{html.escape(state.research_notes)}</div>', 0.25),
                unsafe_allow_html=True,
            )
    with notes_cols[1]:
        if state.analysis_notes:
            st.markdown(
                card("Analysis notes", f'<div class="notes">{html.escape(state.analysis_notes)}</div>', 0.3),
                unsafe_allow_html=True,
            )

    tl_html = '<div class="tl">' + "".join(
        f'<div class="tl-item"><span class="t">{html.escape(e["name"])}</span>'
        f'<div class="p">{html.escape(str(e.get("payload", {})))[:180]}</div></div>'
        for e in state.trace
    ) + "</div>"
    with st.expander(f"Trace timeline — {len(state.trace)} events", expanded=False):
        st.markdown(tl_html, unsafe_allow_html=True)

    st.link_button("Open LangSmith trace ↗", "https://smith.langchain.com/")


# ------------------------------------------------------------------------------------
# Header + hero
# ------------------------------------------------------------------------------------
st.markdown(
    """
<div class="nav">
    <div class="nav-brand"><div class="nav-dot"></div>Multi-Agent Research Lab</div>
    <div class="nav-status">gpt-4o-mini · <b>live</b></div>
    <a href="https://smith.langchain.com/" target="_blank">LangSmith ↗</a>
</div>
<div class="eyebrow"><span class="pulse"></span> Lab 20 · Multi-Agent Systems</div>
<h1 class="display">One question.<br><span class="grad">Four minds, one answer.</span></h1>
<p class="lede">A supervisor orchestrates three specialists — researcher, analyst, writer —
through a shared state. Compare the pipeline against a single-agent baseline, step by step,
in one glass panel.</p>
""",
    unsafe_allow_html=True,
)

query = st.text_input(
    "Research question",
    value="Research GraphRAG state-of-the-art and write a 500-word summary",
    label_visibility="collapsed",
)

btn_l, btn_r = st.columns(2, gap="medium")
with btn_l:
    run_baseline = st.button("Run Single-Agent", use_container_width=True)
with btn_r:
    run_multi = st.button("Run Multi-Agent", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------------
if run_baseline:
    with st.spinner("Single agent is composing an answer…"):
        started = perf_counter()
        try:
            state = make_baseline_runner()(query)
            st.session_state["baseline"] = {"state": state, "latency": perf_counter() - started}
        except Exception as exc:  # hiện lỗi rõ ràng trong UI thay vì crash
            st.session_state["baseline"] = {"error": str(exc)}

if run_multi:
    with st.spinner("Supervisor is orchestrating the pipeline…"):
        started = perf_counter()
        try:
            state = make_multi_agent_runner()(query)
            st.session_state["multi"] = {"state": state, "latency": perf_counter() - started}
        except Exception as exc:
            st.session_state["multi"] = {"error": str(exc)}

base = st.session_state.get("baseline")
multi = st.session_state.get("multi")

if base or multi:
    st.markdown('<div style="height: 2rem"></div>', unsafe_allow_html=True)

if base:
    st.markdown('<div class="eyebrow" style="margin-bottom:1rem">Single-Agent Baseline</div>', unsafe_allow_html=True)
    if "error" in base:
        st.error(base["error"])
    else:
        render_result("baseline", base["state"], base["latency"])

if multi:
    st.markdown(
        '<div class="eyebrow" style="margin:2.2rem 0 1rem">Multi-Agent Pipeline</div>',
        unsafe_allow_html=True,
    )
    if "error" in multi:
        st.error(multi["error"])
    else:
        render_result("multi", multi["state"], multi["latency"])

if base and multi and "state" in base and "state" in multi:
    st.markdown('<div style="height: 2.4rem"></div>', unsafe_allow_html=True)
    m_b = trace_summary(base["state"])
    m_m = trace_summary(multi["state"])
    cmp_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:13.5px;color:#a1a1aa">'
        "<tr style='color:#e4e4e7;font-weight:600'><td style='padding:10px 0'>Metric</td>"
        "<td style='padding:10px 0'>Single-agent</td><td style='padding:10px 0'>Multi-agent</td></tr>"
        + "".join(
            f"<tr style='border-top:1px solid rgba(255,255,255,0.07)'>"
            f"<td style='padding:12px 0'>{k}</td>"
            f"<td style='padding:12px 0'>{a}</td><td style='padding:12px 0'>{b}</td></tr>"
            for k, a, b in [
                ("Latency", f"{base['latency']:.2f}s", f"{multi['latency']:.2f}s"),
                ("Est. cost", f"${m_b['cost']:.5f}", f"${m_m['cost']:.5f}"),
                ("Tokens (in/out)", f"{m_b['in']} / {m_b['out']}", f"{m_m['in']} / {m_m['out']}"),
                ("Trace events", str(m_b["events"]), str(m_m["events"])),
            ]
        )
        + "</table>"
    )
    st.markdown(card("Side-by-side", cmp_html, 0.1), unsafe_allow_html=True)

if not base and not multi:
    st.markdown(
        '<p style="color:#71717a;font-size:13.5px;margin-top:2.2rem">'
        "Run both modes on the same question, then compare latency, cost and flow.</p>",
        unsafe_allow_html=True,
    )

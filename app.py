import streamlit as st
import os
import re
from typing import List, TypedDict
from fpdf import FPDF

# ── graceful imports ──────────────────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_LC_AVAILABLE = True
except ImportError:
    GEMINI_LC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_DIRECT_AVAILABLE = True
except ImportError:
    GEMINI_DIRECT_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from langchain_community.utilities import ArxivAPIWrapper
from langgraph.graph import StateGraph, END


# ── Helper: scrub API keys from any error message ─────────────────────────────
def safe_error(err: Exception, *keys: str) -> str:
    msg = str(err)
    for k in keys:
        if k and len(k) > 6:
            msg = msg.replace(k, k[:4] + "****" + k[-4:])
    # Also mask any Google API key pattern (AIza...)
    msg = re.sub(r'AIza[0-9A-Za-z_\-]{35}', 'AIza****[REDACTED]', msg)
    return msg


# ── 1. State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    topic: str
    papers: List[dict]
    summary: str
    pdf_path: str


# ── 2. ArXiv fetch ────────────────────────────────────────────────────────────
def fetch_arxiv_papers(state: AgentState):
    arxiv = ArxivAPIWrapper(load_max_docs=5, load_all_available_meta=True)
    try:
        results = arxiv.get_summaries_as_docs(state["topic"])
        papers = []
        for doc in results:
            meta = doc.metadata
            raw_authors = meta.get("Authors", meta.get("authors", ""))
            if isinstance(raw_authors, list):
                authors = ", ".join(raw_authors)
            else:
                authors = str(raw_authors) if raw_authors else "Unknown Authors"
            papers.append({
                "title":     meta.get("Title",    meta.get("title",    "No Title")),
                "link":      meta.get("Entry ID", meta.get("entry_id", "No Link")),
                "authors":   authors,
                "published": str(meta.get("Published", meta.get("published", ""))),
                "summary":   doc.page_content,
            })
        return {"papers": papers}
    except Exception as e:
        st.error(f"ArXiv fetch error: {e}")
        return {"papers": []}


# ── 3. Summarise ──────────────────────────────────────────────────────────────
def summarize_papers(state: AgentState, llm_info: dict):
    if not state["papers"]:
        return {"summary": "No papers were found to summarize."}

    context = "\n\n".join([
        f"Title: {p['title']}\nAuthors: {p['authors']}\nAbstract: {p['summary']}"
        for p in state["papers"]
    ])
    prompt = (
        f"Provide a professional, comprehensive overview of the following research papers "
        f"regarding '{state['topic']}':\n\n{context}"
    )

    provider = llm_info["provider"]
    api_key  = llm_info["api_key"]
    model    = llm_info["model"]

    try:
        if provider == "gemini" and GEMINI_LC_AVAILABLE:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.3,
                convert_system_message_to_human=True,
            )
            resp = llm.invoke(prompt)
            return {"summary": resp.content}

        elif provider == "gemini" and GEMINI_DIRECT_AVAILABLE:
            genai.configure(api_key=api_key)
            gm   = genai.GenerativeModel(model)
            resp = gm.generate_content(prompt)
            return {"summary": resp.text}

        elif provider == "groq" and GROQ_AVAILABLE:
            llm  = ChatGroq(model=model, groq_api_key=api_key, temperature=0.3)
            resp = llm.invoke(prompt)
            return {"summary": resp.content}

        else:
            return {"summary": "No LLM provider available. Please check your installation."}

    except Exception as e:
        masked = safe_error(e, api_key)
        return {"summary": f"LLM_ERROR::{masked}"}


# ── 4. PDF export ─────────────────────────────────────────────────────────────
def generate_pdf_report(state: AgentState):
    def clean(text):
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"Research Report: {state['topic']}"), ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Overview", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, clean(state["summary"]))
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Reference Papers", ln=True)
    for p in state["papers"]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, clean(p["title"]))
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, clean(f"Authors: {p['authors']}"), ln=True)
        if p.get("published"):
            pdf.cell(0, 5, clean(f"Published: {p['published']}"), ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(0, 0, 200)
        pdf.write(5, p["link"], p["link"])
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)

    safe = "".join(x for x in state["topic"] if x.isalnum() or x in "._- ")
    path = f"Research_Report_{safe.replace(' ', '_')}.pdf"
    pdf.output(path)
    return {"pdf_path": path}


# ── 5. Validate key format ────────────────────────────────────────────────────
def validate_gemini_key(key: str):
    key = key.strip()
    if not key:
        return False, "API key is empty."
    if not key.startswith("AIza"):
        return False, "Gemini keys start with 'AIza'. Check you copied the full key."
    if len(key) < 35:
        return False, "Key looks too short. Copy the full key from AI Studio."
    return True, ""


# ── 6. Streamlit UI ───────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="ArxivMind Agent", layout="wide", page_icon="🔬")

    st.markdown("""
    <style>
    .paper-card {
        background:linear-gradient(135deg,#1e1e2e 0%,#2a2a3e 100%);
        border:1px solid #3a3a5c; border-radius:12px;
        padding:16px; margin-bottom:14px;
    }
    .paper-title   { color:#a6c8ff; font-size:14px; font-weight:700; margin-bottom:4px; }
    .paper-authors { color:#cdd9f5; font-size:12px; margin-bottom:4px; }
    .paper-date    { color:#888aaa; font-size:11px; margin-bottom:8px; }
    .paper-link a  { color:#7ec8e3; font-size:12px; text-decoration:none; }
    .tip-box { background:#1a2a1a; border:1px solid #3a6a3a;
               border-radius:8px; padding:12px; font-size:13px; color:#cdd9f5; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 ArxivMind: Agentic Research Orchestrator")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🤖 LLM Provider")
        provider_label = st.radio(
            "Choose provider",
            ["Gemini (Free)", "Groq / LLaMA3 (Free)"],
        )

        st.divider()
        st.header("🔑 API Key")

        api_key = ""
        model   = ""

        if provider_label == "Gemini (Free)":
            provider = "gemini"
            api_key = st.text_input(
                "Gemini API Key", type="password", placeholder="AIza...",
                help="Get FREE key → https://aistudio.google.com/app/apikey",
            )
            model = st.selectbox("Model", [
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                "gemini-2.0-flash",
                "gemini-1.5-pro",
            ])
            if api_key:
                ok, msg = validate_gemini_key(api_key)
                st.success("✅ Key format looks good") if ok else st.error(f"⚠️ {msg}")
            else:
                st.markdown("""<div class="tip-box">
                📋 <b>How to get a FREE key:</b><br>
                1. Visit <b>aistudio.google.com</b><br>
                2. Sign in with Google<br>
                3. Click <b>Get API Key → Create API Key</b><br>
                4. Copy the full key (starts with AIza)
                </div>""", unsafe_allow_html=True)

        else:
            provider = "groq"
            api_key = st.text_input(
                "Groq API Key", type="password", placeholder="gsk_...",
                help="Get FREE key → https://console.groq.com",
            )
            model = st.selectbox("Model", [
                "llama3-70b-8192",
                "llama3-8b-8192",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
            ])
            if api_key:
                st.success("✅ Key entered")
            else:
                st.markdown("""<div class="tip-box">
                📋 <b>How to get a FREE Groq key:</b><br>
                1. Visit <b>console.groq.com</b><br>
                2. Sign up (free)<br>
                3. API Keys → Create new key<br>
                4. Copy & paste here
                </div>""", unsafe_allow_html=True)

        st.divider()
        st.caption("🔒 Your API key is never stored, logged, or shown in errors.")

    # ── Main ──────────────────────────────────────────────────────────────────
    topic = st.text_input(
        "What topic are you researching?",
        placeholder="e.g. Large Language Models in Medicine",
    )

    if st.button("🚀 Run Research Agent", type="primary"):
        if not topic:
            st.warning("Please enter a research topic.")
            return
        if not api_key or not api_key.strip():
            st.warning("Please enter your API key in the sidebar.")
            return
        if provider == "gemini":
            ok, msg = validate_gemini_key(api_key)
            if not ok:
                st.error(f"Invalid Gemini key: {msg}")
                return

        llm_info = {"provider": provider, "api_key": api_key.strip(), "model": model}

        try:
            workflow = StateGraph(AgentState)
            workflow.add_node("fetcher",    fetch_arxiv_papers)
            workflow.add_node("summarizer", lambda x: summarize_papers(x, llm_info))
            workflow.add_node("exporter",   generate_pdf_report)
            workflow.set_entry_point("fetcher")
            workflow.add_edge("fetcher",    "summarizer")
            workflow.add_edge("summarizer", "exporter")
            workflow.add_edge("exporter",   END)
            app = workflow.compile()

            with st.status("🔍 Agent running…", expanded=True) as status:
                st.write("📡 Step 1 — Querying ArXiv…")
                results = app.invoke(
                    {"topic": topic, "papers": [], "summary": "", "pdf_path": ""}
                )
                status.update(label="✅ Done!", state="complete")

            summary = results.get("summary", "")

            # ── LLM error path ────────────────────────────────────────────
            if summary.startswith("LLM_ERROR::"):
                err_msg = summary.replace("LLM_ERROR::", "")
                st.error(f"⚠️ LLM Error: {err_msg}")
                st.markdown("""
                **Common fixes:**
                - Make sure you copied the **full** API key (nothing cut off)
                - For Gemini: check quota at [aistudio.google.com](https://aistudio.google.com)
                - Try switching to **Groq / LLaMA3** in the sidebar — it's also free
                - Make sure `langchain-google-genai` is installed: `pip install langchain-google-genai`
                """)
                if results["papers"]:
                    st.subheader("📄 Papers fetched (summary unavailable due to LLM error)")
                    for p in results["papers"]:
                        st.markdown(f"**{p['title']}**  \n👤 {p['authors']}  \n[View on ArXiv]({p['link']})")
                return

            # ── Success path ──────────────────────────────────────────────
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📝 Executive Summary")
                st.markdown(summary)

            with col2:
                st.subheader(f"🔗 Papers ({len(results['papers'])})")
                for paper in results["papers"]:
                    pub = paper.get("published", "")[:10]
                    pub_html = f"<div class='paper-date'>📅 {pub}</div>" if pub else ""
                    st.markdown(f"""
                    <div class="paper-card">
                        <div class="paper-title">{paper['title']}</div>
                        <div class="paper-authors">👤 {paper['authors']}</div>
                        {pub_html}
                        <div class="paper-link">
                            <a href="{paper['link']}" target="_blank">🔗 View on ArXiv</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                if results.get("pdf_path"):
                    with open(results["pdf_path"], "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=results["pdf_path"],
                            mime="application/pdf",
                            use_container_width=True,
                        )

        except Exception as e:
            masked = safe_error(e, api_key)
            st.error(f"Unexpected error: {masked}")
            st.info("💡 Switch to **Groq / LLaMA3 (Free)** in the sidebar if Gemini keeps failing.")


if __name__ == "__main__":
    main()

import streamlit as st
import os
from typing import List, TypedDict
from fpdf import FPDF

# ── graceful imports ──────────────────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from langchain_community.utilities import ArxivAPIWrapper
from langgraph.graph import StateGraph, END


# ── 1. State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    topic: str
    papers: List[dict]
    summary: str
    pdf_path: str


# ── 2. Tool Logic ─────────────────────────────────────────────────────────────
def fetch_arxiv_papers(state: AgentState):
    """Fetches real paper data (including authors) from ArXiv."""
    arxiv = ArxivAPIWrapper(load_max_docs=5, load_all_available_meta=True)
    try:
        results = arxiv.get_summaries_as_docs(state["topic"])
        papers = []
        for doc in results:
            meta = doc.metadata

            # Authors can be a list or a comma-separated string depending on version
            raw_authors = meta.get("Authors", meta.get("authors", ""))
            if isinstance(raw_authors, list):
                authors = ", ".join(raw_authors)
            else:
                authors = str(raw_authors) if raw_authors else "Unknown Authors"

            papers.append(
                {
                    "title":   meta.get("Title",    meta.get("title",    "No Title")),
                    "link":    meta.get("Entry ID", meta.get("entry_id", "No Link")),
                    "authors": authors,
                    "published": str(meta.get("Published", meta.get("published", ""))),
                    "summary": doc.page_content,
                }
            )
        return {"papers": papers}
    except Exception as e:
        st.error(f"Error fetching from ArXiv: {e}")
        return {"papers": []}


def summarize_papers(state: AgentState, llm):
    """Synthesises an overview of all fetched papers."""
    if not state["papers"]:
        return {"summary": "No papers were found to summarize."}

    context = "\n\n".join(
        [
            f"Title: {p['title']}\nAuthors: {p['authors']}\nAbstract: {p['summary']}"
            for p in state["papers"]
        ]
    )
    prompt = (
        f"Provide a professional, comprehensive overview of the following research papers "
        f"regarding '{state['topic']}':\n\n{context}"
    )
    try:
        response = llm.invoke(prompt)
        return {"summary": response.content}
    except Exception as e:
        return {"summary": f"Error during summarization: {str(e)}"}


def generate_pdf_report(state: AgentState):
    """Generates a professional PDF file."""

    def clean(text):
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"Research Report: {state['topic']}"), ln=True, align="C")
    pdf.ln(8)

    # Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Overview", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, clean(state["summary"]))
    pdf.ln(10)

    # Papers with authors
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


# ── 3. LLM factory ────────────────────────────────────────────────────────────
def build_llm(provider: str, gemini_key: str, groq_key: str, model: str):
    """Return an LLM instance, or raise with a helpful message."""
    if provider == "Gemini (Free)" and GEMINI_AVAILABLE:
        if not gemini_key:
            raise ValueError("Please enter your Gemini API key in the sidebar.")
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=gemini_key, temperature=0.3
        )
    elif provider == "Groq / LLaMA3 (Free)" and GROQ_AVAILABLE:
        if not groq_key:
            raise ValueError("Please enter your Groq API key in the sidebar.")
        return ChatGroq(model=model, groq_api_key=groq_key, temperature=0.3)
    else:
        raise ValueError(
            f"Provider '{provider}' is not available. "
            "Install the required package: langchain-google-genai or langchain-groq"
        )


# ── 4. Streamlit UI ───────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ArxivMind Agent", layout="wide", page_icon="🔬"
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .paper-card {
            background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
            border: 1px solid #3a3a5c;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .paper-title { color: #a6c8ff; font-size: 14px; font-weight: 700; margin-bottom: 4px; }
        .paper-authors { color: #cdd9f5; font-size: 12px; margin-bottom: 4px; }
        .paper-date   { color: #888aaa; font-size: 11px; margin-bottom: 8px; }
        .paper-link a { color: #7ec8e3; font-size: 12px; text-decoration: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔬 ArxivMind: Agentic Research Orchestrator")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🤖 LLM Provider")
        provider = st.radio(
            "Choose provider",
            ["Gemini (Free)", "Groq / LLaMA3 (Free)"],
            index=0,
        )

        st.divider()
        st.header("🔑 API Keys")

        gemini_key = ""
        groq_key   = ""
        model      = ""

        if provider == "Gemini (Free)":
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get a FREE key at https://aistudio.google.com/app/apikey",
            )
            model = st.selectbox(
                "Model",
                [
                    "gemini-1.5-flash",       # free, fast
                    "gemini-1.5-flash-8b",    # free, smallest
                    "gemini-1.5-pro",         # free up to quota
                ],
                index=0,
            )
            st.caption("✅ gemini-1.5-flash is FREE on Google AI Studio")
        else:
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Get a FREE key at https://console.groq.com",
            )
            model = st.selectbox(
                "Model",
                [
                    "llama3-70b-8192",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                ],
                index=0,
            )
            st.caption("✅ Groq is FREE (rate-limited)")

        st.divider()
        st.header("⚙️ Settings")
        num_papers = st.slider("Papers to analyse", 1, 10, 5)

    # ── Main area ─────────────────────────────────────────────────────────────
    topic = st.text_input(
        "What topic are you researching?",
        placeholder="e.g. Quantum Computing in Healthcare",
    )

    if st.button("🚀 Run Research Agent", type="primary") and topic:
        try:
            llm = build_llm(provider, gemini_key, groq_key, model)
        except ValueError as e:
            st.warning(str(e))
            return

        try:
            # Build LangGraph workflow
            workflow = StateGraph(AgentState)
            workflow.add_node("fetcher",    fetch_arxiv_papers)
            workflow.add_node("summarizer", lambda x: summarize_papers(x, llm))
            workflow.add_node("exporter",   generate_pdf_report)

            workflow.set_entry_point("fetcher")
            workflow.add_edge("fetcher",    "summarizer")
            workflow.add_edge("summarizer", "exporter")
            workflow.add_edge("exporter",   END)

            app = workflow.compile()

            with st.status(
                "🔍 Agent is fetching & analysing papers…", expanded=True
            ) as status:
                st.write("📡 Querying ArXiv…")
                results = app.invoke(
                    {"topic": topic, "papers": [], "summary": "", "pdf_path": ""}
                )
                status.update(label="✅ Analysis Complete!", state="complete")

            # ── Results layout ─────────────────────────────────────────────
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📝 Executive Summary")
                st.markdown(results["summary"])

            with col2:
                st.subheader(f"🔗 Papers Found ({len(results['papers'])})")

                for paper in results["papers"]:
                    pub = paper.get("published", "")[:10]  # YYYY-MM-DD
                    st.markdown(
                        f"""
                        <div class="paper-card">
                            <div class="paper-title">{paper['title']}</div>
                            <div class="paper-authors">👤 {paper['authors']}</div>
                            {"<div class='paper-date'>📅 " + pub + "</div>" if pub else ""}
                            <div class="paper-link"><a href="{paper['link']}" target="_blank">🔗 View on ArXiv</a></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
            st.error(f"An unexpected error occurred: {e}")
            st.info(
                "💡 Tip: If Gemini fails, switch to **Groq / LLaMA3 (Free)** in the sidebar "
                "and get a free key at https://console.groq.com"
            )


if __name__ == "__main__":
    main()

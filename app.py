import streamlit as st
import os
from typing import Annotated, List, TypedDict
from fpdf import FPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import ArxivAPIWrapper
from langgraph.graph import StateGraph, END

# --- 1. State Definition ---
class AgentState(TypedDict):
    topic: str
    papers: List[dict]
    summary: str
    pdf_path: str

# --- 2. Tool Logic ---
def fetch_arxiv_papers(state: AgentState):
    """Fetches real paper data from ArXiv."""
    # We use the topic from state
    arxiv = ArxivAPIWrapper(load_max_docs=5, load_all_available_meta=True)
    try:
        results = arxiv.get_summaries_as_docs(state['topic'])
        papers = []
        for doc in results:
            papers.append({
                "title": doc.metadata.get("Title", "No Title"),
                "link": doc.metadata.get("Entry ID", "No Link"),
                "summary": doc.page_content
            })
        return {"papers": papers}
    except Exception as e:
        st.error(f"Error fetching from ArXiv: {e}")
        return {"papers": []}

def summarize_papers(state: AgentState, llm):
    """Synthesizes an overview of all fetched papers."""
    if not state['papers']:
        return {"summary": "No papers were found to summarize."}
        
    context = "\n\n".join([f"Title: {p['title']}\nAbstract: {p['summary']}" for p in state['papers']])
    prompt = f"Provide a professional, comprehensive overview of the following research papers regarding '{state['topic']}':\n\n{context}"
    
    try:
        response = llm.invoke(prompt)
        return {"summary": response.content}
    except Exception as e:
        return {"summary": f"Error during summarization: {str(e)}"}

def generate_pdf_report(state: AgentState):
    """Generates a professional PDF file."""
    # Using 'latin-1' or replacing non-ascii to prevent FPDF errors
    def clean_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, clean_text(f"Research Report: {state['topic']}"), ln=True, align='C')
    pdf.ln(10)
    
    # Summary Section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Overview", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, clean_text(state['summary']))
    pdf.ln(10)
    
    # Links Section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Reference Links", ln=True)
    pdf.set_font("Helvetica", size=10)
    for p in state['papers']:
        pdf.write(5, clean_text(f"- {p['title']}: "))
        pdf.set_text_color(0, 0, 255)
        pdf.write(5, p['link'], p['link'])
        pdf.set_text_color(0, 0, 0)
        pdf.ln(7)
        
    safe_topic = "".join(x for x in state['topic'] if x.isalnum() or x in "._- ")
    path = f"Research_Report_{safe_topic.replace(' ', '_')}.pdf"
    pdf.output(path)
    return {"pdf_path": path}

# --- 3. Streamlit UI ---
def main():
    st.set_page_config(page_title="ArxivMind Agent", layout="wide", page_icon="🔬")
    st.title("🔬 ArxivMind: Agentic Research Orchestrator")
    
    with st.sidebar:
        st.header("🔑 Authentication")
        # Direct API key input for the session
        user_api_key = st.text_input("Gemini API Key", type="password", help="Get it from Google AI Studio")
        st.divider()
        st.header("⚙️ Settings")
        model_choice = st.selectbox("Model Version", ["gemini-1.5-flash", "gemini-1.5-pro"])
        num_papers = st.slider("Papers to analyze", 1, 10, 5)

    topic = st.text_input("What topic are you researching?", placeholder="e.g. Quantum Computing in Healthcare")

    if st.button("🚀 Run Research Agent") and topic:
        if not user_api_key:
            st.warning("Please enter your API Key in the sidebar.")
            return

        try:
            # Initialize LLM with explicit API key to avoid Env Var issues
            llm = ChatGoogleGenerativeAI(
                model=model_choice, 
                google_api_key=user_api_key,
                temperature=0.3
            )

            # Build Graph
            workflow = StateGraph(AgentState)
            workflow.add_node("fetcher", fetch_arxiv_papers)
            # Pass LLM explicitly to the node
            workflow.add_node("summarizer", lambda x: summarize_papers(x, llm))
            workflow.add_node("exporter", generate_pdf_report)

            workflow.set_entry_point("fetcher")
            workflow.add_edge("fetcher", "summarizer")
            workflow.add_edge("summarizer", "exporter")
            workflow.add_edge("exporter", END)

            app = workflow.compile()

            with st.status("🔍 Agent is searching and analyzing...", expanded=True) as status:
                results = app.invoke({"topic": topic, "papers": [], "summary": "", "pdf_path": ""})
                status.update(label="✅ Analysis Complete!", state="complete")

            # Layout for Results
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📝 Executive Summary")
                st.markdown(results['summary'])

            with col2:
                st.subheader("🔗 Reference Papers")
                for paper in results['papers']:
                    st.markdown(f"**{paper['title']}**  \n[View Paper]({paper['link']})")
                
                st.divider()
                if results.get('pdf_path'):
                    with open(results['pdf_path'], "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=results['pdf_path'],
                            mime="application/pdf",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

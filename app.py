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
    arxiv = ArxivAPIWrapper(load_max_docs=5, load_all_available_meta=True)
    results = arxiv.get_summaries_as_docs(state['topic'])
    
    papers = []
    for doc in results:
        papers.append({
            "title": doc.metadata.get("Title", "No Title"),
            "link": doc.metadata.get("Entry ID", "No Link"),
            "summary": doc.page_content
        })
    return {"papers": papers}

def summarize_papers(state: AgentState, llm):
    """Synthesizes an overview of all fetched papers."""
    context = "\n\n".join([f"Title: {p['title']}\nAbstract: {p['summary']}" for p in state['papers']])
    prompt = f"Provide a professional, comprehensive overview of the following research papers regarding '{state['topic']}':\n\n{context}"
    
    response = llm.invoke(prompt)
    return {"summary": response.content}

def generate_pdf_report(state: AgentState):
    """Generates a professional PDF file."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"Research Report: {state['topic']}", ln=True, align='C')
    pdf.ln(10)
    
    # Summary Section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Executive Overview", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, state['summary'])
    pdf.ln(10)
    
    # Links Section
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Reference Links", ln=True)
    pdf.set_font("Helvetica", size=10)
    for p in state['papers']:
        pdf.write(5, f"- {p['title']}: ")
        pdf.set_text_color(0, 0, 255)
        pdf.write(5, p['link'], p['link'])
        pdf.set_text_color(0, 0, 0)
        pdf.ln(7)
        
    path = f"Research_Report_{state['topic'].replace(' ', '_')}.pdf"
    pdf.output(path)
    return {"pdf_path": path}

# --- 3. Streamlit UI ---
def main():
    st.set_page_config(page_title="Agentic Research Finder", layout="wide")
    st.title("🔬 LangGraph Research Agent")
    st.markdown("Find papers on **ArXiv** and generate AI-powered summaries using **Gemini**.")

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        num_papers = st.slider("Number of papers", 1, 10, 5)

    topic = st.text_input("Enter Research Topic (e.g., 'Agentic AI Workflows')")

    if st.button("Generate Report") and topic:
        if not api_key:
            st.error("Please provide a Gemini API Key.")
            return

        os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(model=model_choice)

        # Build Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("fetcher", fetch_arxiv_papers)
        workflow.add_node("summarizer", lambda x: summarize_papers(x, llm))
        workflow.add_node("exporter", generate_pdf_report)

        workflow.set_entry_point("fetcher")
        workflow.add_edge("fetcher", "summarizer")
        workflow.add_edge("summarizer", "exporter")
        workflow.add_edge("exporter", END)

        app = workflow.compile()

        with st.status("Agent working...", expanded=True) as status:
            final_state = app.invoke({"topic": topic, "papers": [], "summary": "", "pdf_path": ""})
            status.update(label="Report Generated!", state="complete")

        # Display Results
        st.subheader("Executive Overview")
        st.info(final_state['summary'])

        st.subheader("Sources Found")
        for paper in final_state['papers']:
            st.markdown(f"🔗 [{paper['title']}]({paper['link']})")

        # Download PDF
        with open(final_state['pdf_path'], "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=final_state['pdf_path'],
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

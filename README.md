# 🔬 LangGraph Research Agent

An AI-powered research assistant built with **LangGraph**, **Streamlit**, and **Google Gemini** that autonomously fetches academic papers from ArXiv and generates professional PDF research reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Agent Workflow](#agent-workflow)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

The LangGraph Research Agent is an **agentic AI application** that automates the research discovery process. You simply provide a research topic, and the agent:

1. Searches and fetches relevant academic papers from **ArXiv**
2. Uses **Google Gemini** to synthesize an executive-level overview
3. Exports everything into a downloadable **PDF report** with clickable reference links

It is built using **LangGraph** to orchestrate a stateful, multi-node agent pipeline, and **Streamlit** for an interactive web UI.

---

## ✨ Features

- 🤖 **Agentic Pipeline** — Multi-step LangGraph workflow with clearly defined nodes
- 📄 **Live ArXiv Search** — Fetches real papers with titles, abstracts, and links
- 🧠 **AI Summarization** — Gemini LLM synthesizes a comprehensive executive overview
- 📑 **PDF Report Generation** — Auto-generates a professionally formatted PDF with embedded hyperlinks
- ⚙️ **Configurable** — Choose your Gemini model and number of papers via the sidebar
- 🔐 **Secure API Key Input** — API key entered at runtime, never hardcoded

---

## 🏗️ Architecture

The application follows a linear **LangGraph state machine**:

```
[User Input]
     │
     ▼
┌─────────────┐
│   Fetcher   │  ← Pulls papers from ArXiv API
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Summarizer    │  ← Gemini LLM synthesizes an overview
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    Exporter     │  ← Generates a PDF report
└──────┬──────────┘
       │
       ▼
     [END]
```

State is passed between nodes using a typed `AgentState` dictionary.

---

## ✅ Prerequisites

- Python **3.9+**
- A **Google Gemini API Key** — Get one at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Internet access (for ArXiv API calls)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/langgraph-research-agent.git
cd langgraph-research-agent
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

No `.env` file is required. The **Gemini API key** is entered securely at runtime through the sidebar in the Streamlit UI.

| Setting | Options | Default |
|---|---|---|
| Gemini Model | `gemini-1.5-flash`, `gemini-1.5-pro` | `gemini-1.5-flash` |
| Number of Papers | 1 – 10 | 5 |

> **Note:** `gemini-1.5-flash` is faster and cheaper. `gemini-1.5-pro` produces higher-quality summaries for complex topics.

---

## 🖥️ Usage

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Step-by-Step

1. **Enter your Gemini API Key** in the left sidebar
2. **Select a Gemini model** (flash for speed, pro for quality)
3. **Set the number of papers** to fetch (1–10)
4. **Type a research topic** in the main input field
   - Example: `Agentic AI Workflows`
   - Example: `Transformer attention mechanisms`
   - Example: `Large language model alignment`
5. Click **"Generate Report"**
6. Wait for the agent to complete all three pipeline stages
7. View the **Executive Overview** and **Source Links** on screen
8. Click **"Download PDF Report"** to save your report

---

## 📁 Project Structure

```
langgraph-research-agent/
│
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── Research_Report_*.pdf   # Auto-generated PDF reports (created at runtime)
```

---

## 🔄 Agent Workflow

### Node 1: `fetch_arxiv_papers`

- Uses `ArxivAPIWrapper` from LangChain Community to query the ArXiv API
- Loads up to the configured number of papers with full metadata
- Extracts: `title`, `entry_id` (link), and `abstract` (summary)
- Returns updated state with a list of paper dictionaries

### Node 2: `summarize_papers`

- Concatenates all paper titles and abstracts into a single context prompt
- Invokes the selected Gemini model via `ChatGoogleGenerativeAI`
- Asks for a professional, comprehensive overview of the collected papers
- Returns updated state with the synthesized summary string

### Node 3: `generate_pdf_report`

- Uses `fpdf2` to generate a multi-section PDF
- **Section 1:** Executive Overview — the AI-generated summary
- **Section 2:** Reference Links — all paper titles as clickable hyperlinks
- Saves the PDF to disk with a sanitized filename based on the topic
- Returns updated state with the file path

---

## 📦 Dependencies

```txt
streamlit
langchain-google-genai
langchain-community
langgraph
fpdf2
arxiv
```

Install all at once:

```bash
pip install streamlit langchain-google-genai langchain-community langgraph fpdf2 arxiv
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Full `requirements.txt`

```
streamlit>=1.35.0
langchain-google-genai>=1.0.0
langchain-community>=0.2.0
langgraph>=0.1.0
fpdf2>=2.7.0
arxiv>=2.1.0
```

---

## 🛠️ Troubleshooting

### ❌ `GOOGLE_API_KEY` Error
**Cause:** API key not entered or invalid.
**Fix:** Paste your key in the sidebar before clicking Generate Report. Get a free key at [Google AI Studio](https://aistudio.google.com/).

### ❌ No Papers Found
**Cause:** ArXiv returned no results for the given topic.
**Fix:** Try a broader or differently phrased topic (e.g., `"neural networks"` instead of a very niche phrase).

### ❌ PDF Not Downloading
**Cause:** File permission issue or path conflict.
**Fix:** Ensure the app has write permissions in its working directory. Check that the generated `*.pdf` file exists locally.

### ❌ `ModuleNotFoundError`
**Cause:** Missing dependency.
**Fix:** Run `pip install -r requirements.txt` inside your active virtual environment.

### ❌ Rate Limit / Quota Exceeded (Gemini)
**Cause:** Free tier API quota reached.
**Fix:** Wait a moment and retry, or upgrade your Google AI plan.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling and ArXiv integration
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI language model
- [ArXiv](https://arxiv.org/) — Open-access research paper repository
- [Streamlit](https://streamlit.io/) — Rapid web UI framework
- [fpdf2](https://py-pdf.github.io/fpdf2/) — PDF generation library

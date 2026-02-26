# sql-rag
Hybrid SQL + Agentic RAG System

A cost-efficient, metadata-aware Hybrid Retrieval System that combines structured SQL querying with semantic vector search using FAISS — powered by an agent-based routing architecture.

This project demonstrates how to move beyond naive RAG pipelines and build production-style retrieval systems that are efficient, grounded, and hallucination-resistant.

🔍 Problem Statement

Traditional RAG systems:

Perform vector search over the entire dataset

Increase latency and token usage

Struggle with structured queries (COUNT, FILTER, GROUP BY)

Risk hallucinated aggregations

Pure SQL systems:

Cannot understand unstructured text

Cannot answer semantic questions like
“What risks were identified?”

❓ So how do we combine both efficiently?
🧠 Solution: Hybrid SQL + Agentic RAG Architecture

This system intelligently routes user queries into two optimized paths:

1️⃣ Structured Query Path (SQL Execution)

For aggregation, counting, and filtering queries:

LLM generates SQL query

SQL is sanitized

Query executes directly on SQLite

Real database result is returned

✅ No hallucinated numbers
✅ No fabricated data
✅ Fully grounded answers

2️⃣ Semantic Query Path (Optimized RAG)

For document understanding queries:

Extract structured filters (e.g., department, year)

Retrieve matching document IDs using SQL

Perform FAISS similarity search

Restrict retrieved documents using metadata (doc_id)

Generate answer strictly from retrieved context

⚡ Key Optimization: Metadata-Aware Vector Retrieval

Instead of running FAISS across the entire dataset:

We store doc_id as metadata in FAISS

Use SQL to narrow down relevant IDs first

Filter vector results using those IDs

docs = vectorstore.similarity_search(question, k=20)

docs = [
    doc for doc in docs
    if doc.metadata["doc_id"] in filtered_ids
]
🔥 Benefits

Reduced search space

Lower latency

Reduced token usage

Improved relevance

Hallucination mitigation

This mirrors production-grade retrieval strategies used in scalable AI systems.

🏗 Architecture Overview

User Query
↓
Router Agent (Gemini 2.5 Flash)
↓
🔀 Decision Layer

Structured Query

→ SQL Generation
→ SQLite Execution
→ Direct Answer

Semantic Query

→ SQL Metadata Filtering
→ Retrieve Document IDs
→ FAISS Vector Search (Restricted)
→ Context-Based LLM Answer

🛠 Tech Stack

LangGraph – Agent workflow orchestration

Gemini 2.5 Flash – Routing + Generation

HuggingFace Embeddings (MiniLM) – Text embeddings

FAISS – Vector similarity search

SQLite – Structured storage

Streamlit – UI interface

📂 Database Schema

Example table:

documents(
    id INTEGER PRIMARY KEY,
    department TEXT,
    year INTEGER,
    title TEXT,
    category TEXT,
    author TEXT,
    created_date TEXT,
    content TEXT
)

Supports both structured metadata queries and semantic retrieval.

🚀 Installation & Setup
1️⃣ Clone the repository
git clone <your-repo-url>
cd <project-folder>
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
3️⃣ Install dependencies
pip install -r requirements.txt

(Or manually install: streamlit, langgraph, faiss-cpu, sentence-transformers, etc.)

4️⃣ Add Gemini API Key

Create a .env file:

GOOGLE_API_KEY=your_api_key_here
5️⃣ Run the application
streamlit run app.py
🧪 Example Queries
Structured

How many documents were created in 2023?

Who authored the Legal document?

Count HR reports.

Semantic

What workplace risks were identified?

What compliance gaps were mentioned?

Summarize the HR report.

Hybrid

What risks were identified in HR documents from 2023?

🎯 Key Highlights

✔ Agent-based routing
✔ SQL grounding to prevent hallucination
✔ Metadata-aware vector search
✔ Cost-efficient retrieval strategy
✔ Production-style hybrid architecture

🧠 What This Demonstrates

This project shows:

Understanding of RAG limitations

Cost optimization in retrieval systems

Hybrid structured + semantic search

Hallucination mitigation techniques

Production-oriented system design

📌 Future Improvements

Persistent FAISS indexing

Multi-table support

Query validation & SQL injection protection

Deployment to cloud

Scalable ingestion pipeline

🤝 Connect

If you're building scalable AI retrieval systems or working on intelligent database interfaces, feel free to connect 🚀

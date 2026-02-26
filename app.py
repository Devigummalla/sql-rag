import os
import sqlite3
from typing import TypedDict, List
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph
import streamlit as st

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------
# Initialize LLM (Gemini 2.5 Flash)
# ---------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# ---------------------------------------------------
# Initialize Embeddings
# ---------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# SQLite Connection
# ---------------------------------------------------

conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

# ---------------------------------------------------
# Create Sample Table (Run Once)
# ---------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    department TEXT,
    year INTEGER,
    title TEXT,
    content TEXT
)
""")
conn.commit()

# ---------------------------------------------------
# Build FAISS Index from SQLite
# ---------------------------------------------------

def build_vectorstore():
    cursor.execute("SELECT id, content FROM documents")
    rows = cursor.fetchall()

    docs = []
    for row in rows:
        doc_id = row[0]
        content = row[1]

        docs.append(
            Document(
                page_content=content,
                metadata={"doc_id": doc_id}
            )
        )

    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

vectorstore = build_vectorstore()

# ---------------------------------------------------
# LangGraph State
# ---------------------------------------------------

class AgentState(TypedDict):
    question: str
    route: str
    filtered_doc_ids: List[int]
    retrieved_docs: List[Document]
    final_answer: str

# ---------------------------------------------------
# 1️⃣ Router Node
# ---------------------------------------------------

def router_node(state: AgentState):

    prompt = f"""
    Classify the query as:
    - structured (aggregation, filtering, counting)
    - semantic (needs document understanding)

    Query: {state['question']}
    """

    response = llm.invoke(prompt).content.lower()

    if "structured" in response:
        route = "structured"
    else:
        route = "semantic"

    return {"route": route}

# ---------------------------------------------------
# 2️⃣ SQL Filter Node
# ---------------------------------------------------

def sql_filter_node(state: AgentState):

    question = state["question"]

    prompt = f"""
    Extract SQL filtering conditions from the question.
    Return a WHERE clause only.

    Question: {question}
    """

    where_clause = llm.invoke(prompt).content.strip()

    query = f"SELECT id FROM documents WHERE {where_clause}"
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        doc_ids = [row[0] for row in rows]
    except:
        doc_ids = []

    return {"filtered_doc_ids": doc_ids}

# ---------------------------------------------------
# 3️⃣ Vector Search Node (Restricted Search)
# ---------------------------------------------------

def vector_node(state: AgentState):

    question = state["question"]
    filtered_ids = state.get("filtered_doc_ids", [])

    # Retrieve top 20 globally first
    docs = vectorstore.similarity_search(question, k=20)

    # Filter locally by doc_id
    if filtered_ids:
        docs = [
            doc for doc in docs
            if doc.metadata["doc_id"] in filtered_ids
        ]

    return {"retrieved_docs": docs[:5]}

# ---------------------------------------------------
# 4️⃣ Generator Node
# ---------------------------------------------------

def generator_node(state: AgentState):

    question = state["question"]

    if state["route"] == "structured":

    # Step 1: Generate SQL only
        sql_prompt = f"""
        Convert this question into a valid SQLite SQL query.
        Only return the SQL query without explanation or formatting.
        Table name: documents
       Columns:
       - id (INTEGER)
       - department (TEXT)
       - year (INTEGER)
       - title (TEXT)
       - category (TEXT)
       - author (TEXT)
       - created_date (TEXT)
       - content (TEXT)

        Question: {question}
        """

        sql_query = llm.invoke(sql_prompt).content.strip()

    # 🔥 CLEAN MARKDOWN IF PRESENT
        match = re.search(r"(SELECT .*?;)", sql_query, re.IGNORECASE | re.DOTALL)

        if match:
            sql_query = match.group(1)
        else:
            sql_query = sql_query.strip()

        print("Cleaned SQL:", sql_query)
        try:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            answer = f"Database Result: {result}"
        except Exception as e:
            answer = f"SQL Execution Error: {str(e)}"

    else:
        context = "\n\n".join(
            [doc.page_content for doc in state["retrieved_docs"]]
        )

        prompt = f"""
        Answer using this context:

        {context}

        Question: {question}
        """

        answer = llm.invoke(prompt).content

    return {"final_answer": answer}

# ---------------------------------------------------
# Conditional Routing
# ---------------------------------------------------

def route_decision(state: AgentState):
    if state["route"] == "structured":
        return "generator"
    else:
        return "sql_filter"

# ---------------------------------------------------
# Build LangGraph
# ---------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("sql_filter", sql_filter_node)
builder.add_node("vector", vector_node)
builder.add_node("generator", generator_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "generator": "generator",
        "sql_filter": "sql_filter"
    }
)

builder.add_edge("sql_filter", "vector")
builder.add_edge("vector", "generator")

builder.set_finish_point("generator")

graph = builder.compile()

# ---------------------------------------------------
# Run System
# ---------------------------------------------------

st.set_page_config(page_title="Hybrid SQL RAG", layout="wide")

st.title("🚀 Hybrid SQL + Agentic RAG")

question = st.text_input("Ask your question")

if st.button("Submit"):

    with st.spinner("Thinking..."):
        result = graph.invoke({"question": question})

    st.success("Answer Ready")

    st.markdown("### 📌 Final Answer")
    st.write(result["final_answer"])
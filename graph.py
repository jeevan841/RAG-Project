import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from importlib import import_module
retriever_mod = import_module("retriever")
get_retriever = retriever_mod.get_retriever

load_dotenv()
GROQ_MODEL = "llama-3.1-8b-instant" 
class GraphState(TypedDict):
    query: str
    retrieved_chunks: List[Document]
    llm_response: str
    confidence: str
    escalate: bool
    human_response: Optional[str]
    final_answer: str                 

def retrieve_node(state: GraphState) -> GraphState:
    print("\n Retrieving relevant chunks...")
    retriever = get_retriever()
    chunks = retriever.invoke(state["query"])
    print(f"   Found {len(chunks)} chunk(s)")
    state["retrieved_chunks"] = chunks
    return state



def process_node(state: GraphState) -> GraphState:
    print(" Sending to Groq LLM...")

    if state["retrieved_chunks"]:
        context = "\n\n".join(
            [f"[Chunk {i+1}]: {doc.page_content}"
             for i, doc in enumerate(state["retrieved_chunks"])]
        )
    else:
        context = "No relevant information found in the knowledge base."

    prompt = ChatPromptTemplate.from_template("""
You are a helpful and professional customer support assistant.
Your job is to answer the user's question using ONLY the context provided below.

Rules:
- If the context contains a clear answer, answer confidently and end with: [CONFIDENCE: HIGH]
- If the context is vague, incomplete, or unrelated, say you're not sure and end with: [CONFIDENCE: LOW]
- If the question is completely outside the context, say you'll escalate to a human and end with: [CONFIDENCE: LOW]
- Never make up information not present in the context
- Be concise and professional

Context:
{context}

User Question: {query}

Your Answer:
""")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "query": state["query"]
    })

    response_text = response.content.strip()
    state["llm_response"] = response_text

    if "[CONFIDENCE: LOW]" in response_text:
        state["confidence"] = "low"
        state["escalate"] = True
    else:
        state["confidence"] = "high"
        state["escalate"] = False

    return state

def router(state: GraphState) -> str:
    """
    Returns "answer" or "escalate" based on:
    - LLM confidence tag
    - Whether any chunks were retrieved
    - Whether user asked for a human
    """
    query_lower = state["query"].lower()

    human_keywords = ["human", "agent", "person", "representative", "speak to someone"]
    if any(kw in query_lower for kw in human_keywords):
        print("   ↳ Routing: user requested human agent")
        return "escalate"

    if not state["retrieved_chunks"]:
        print("   ↳ Routing: no chunks found → escalating")
        return "escalate"
    if state["confidence"] == "low" or state["escalate"]:
        print("   ↳ Routing: low confidence → escalating")
        return "escalate"

    print("   ↳ Routing: high confidence → answering directly")
    return "answer"



def output_node(state: GraphState) -> GraphState:
    clean_response = (
        state["llm_response"]
        .replace("[CONFIDENCE: HIGH]", "")
        .replace("[CONFIDENCE: LOW]", "")
        .strip()
    )
    state["final_answer"] = clean_response
    return state



def hitl_node(state: GraphState) -> GraphState:
    print("\n" + "="*50)
    print(" ESCALATING TO HUMAN AGENT")
    print("="*50)
    print(f"User asked: {state['query']}")

    if state["llm_response"]:
        clean = (
            state["llm_response"]
            .replace("[CONFIDENCE: HIGH]", "")
            .replace("[CONFIDENCE: LOW]", "")
            .strip()
        )
        print(f"Bot attempted: {clean}")

    print("-"*50)
    human_input = input(" Human Agent — type your response: ").strip()

    state["human_response"] = human_input
    state["final_answer"] = f"[Agent]: {human_input}"
    return state


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("process", process_node)
    graph.add_node("output", output_node)
    graph.add_node("hitl", hitl_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "process")
    graph.add_conditional_edges(
        "process",
        router,
        {
            "answer": "output",
            "escalate": "hitl"
        }
    )
    graph.add_edge("output", END)
    graph.add_edge("hitl", END)
    return graph.compile()

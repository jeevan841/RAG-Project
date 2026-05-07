import os
from dotenv import load_dotenv
from importlib import import_module
graph_mod = import_module("graph")
build_graph = graph_mod.build_graph

load_dotenv()


def check_setup():
    """Validates environment before starting."""
    errors = []

    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        errors.append(" GROQ_API_KEY not set in .env file please do so now")

    if not os.path.exists("chroma_db"):
        errors.append(" ChromaDB not found — run: python ingest.py first")

    if errors:
        print("\n".join(errors))
        return False

    return True


def main():
    print("\n" + "="*55)
    print("  RAG Customer Support Bot  (powered by Groq)")
    print("="*55)

    if not check_setup():
        return

    graph = build_graph()

    print("\nType your question below.")
    print("Type 'quit' or 'exit' to stop.\n")
    print("-"*55)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! ")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye! ")
            break

        # Build initial graph state
        initial_state = {
            "query": query,
            "retrieved_chunks": [],
            "llm_response": "",
            "confidence": "",
            "escalate": False,
            "human_response": None,
            "final_answer": ""
        }

        # Run the LangGraph workflow
        result = graph.invoke(initial_state)

        # Print final answer
        print(f"\n{'─'*55}")
        print(f"Answer: {result['final_answer']}")
        print(f"{'─'*55}")


if __name__ == "__main__":
    main()

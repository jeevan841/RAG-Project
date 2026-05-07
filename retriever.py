import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
TOP_K = 3


def get_retriever():
    """
    Returns a LangChain retriever backed by ChromaDB.
    Make sure you've run ingest.py before calling this.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}'.\n"
            "Please run: python ingest.py"
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return retriever


if __name__ == "__main__":
    retriever = get_retriever()
    test_query = "What is this document about?"
    results = retriever.invoke(test_query)

    print(f"Query: {test_query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i + 1} (page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:300])
        print()

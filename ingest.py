import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = "data/11.pdf"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest():
    if not os.path.exists(PDF_PATH):
        print(f" PDF not found at '{PDF_PATH}'")
        print("   Please place your PDF inside the data/ folder")
        print("   and rename it to 'your_document.pdf'")
        return

    print(f" Loading PDF from '{PDF_PATH}'...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"   Loaded {len(documents)} page(s)")


    print(f"\n Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

   
    print("\n Creating embeddings (this may take a minute first time)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",      
        model_kwargs={"device": "cpu"}
    )

    print(f" Storing in ChromaDB at '{CHROMA_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    print(f"\n Done! {len(chunks)} chunks stored in ChromaDB.")
    print("   You can now run: python main.py")


if __name__ == "__main__":
    ingest()

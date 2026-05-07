# RAG Customer Support Bot
### Built with LangChain · ChromaDB · LangGraph · Groq (LLaMA 3)

---

## What This Does

A Retrieval-Augmented Generation (RAG) system that:
- Reads your PDF knowledge base
- Answers user questions using only that document
- Uses LangGraph for workflow control
- Escalates to a human agent when it's not confident

---

## Setup (3 Steps)

### Step 1 — Get a free Groq API key
1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Paste it in your `.env` file:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

### Step 2 — Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Step 3 — Add your PDF
- Place your PDF inside the `data/` folder
- Rename it to `your_document.pdf`
  (or edit `PDF_PATH` in `1_ingest.py`)

---

## Running the Bot

### First time only — ingest your PDF:
```bash
python 1_ingest.py
```
This creates the `chroma_db/` folder with your embeddings.

### Every time — start the chatbot:
```bash
python main.py
```

---

## File Structure

```
rag-support-bot/
│
├── data/
│   └── your_document.pdf       ← your knowledge base
│
├── chroma_db/                  ← auto-created after ingestion
│
├── 1_ingest.py                 ← PDF → Chunks → ChromaDB
├── 2_retriever.py              ← ChromaDB → Top-K chunks
├── 3_graph.py                  ← LangGraph workflow + HITL
├── main.py                     ← Chatbot entry point
│
├── .env                        ← Your Groq API key
├── requirements.txt            ← All dependencies
└── README.md
```

---

## How It Works

```
PDF
 └─[1_ingest.py]─→ Chunks ─→ Embeddings ─→ ChromaDB
                                                │
User Query ─→ Embed Query ─→ ChromaDB Search ──┘
                                  │
                            Top 3 Chunks
                                  │
                         [LangGraph Workflow]
                                  │
                    ┌─────────────▼─────────────┐
                    │      Groq LLM (LLaMA 3)   │
                    └─────────────┬─────────────┘
                                  │
                          [Router checks]
                         /                \
                   HIGH conf           LOW conf
                       │                   │
               Answer to User      Human Agent Input
                                           │
                                   Answer to User
```

---

## Groq Model Options

Edit `GROQ_MODEL` in `3_graph.py`:

| Model | Speed | Quality |
|---|---|---|
| `llama3-8b-8192` | ⚡ Fastest | Good |
| `llama3-70b-8192` | Medium | Best |
| `mixtral-8x7b-32768` | Fast | Great |
| `gemma2-9b-it` | Fast | Good |

---

## HITL (Human-in-the-Loop)

When the bot escalates, you'll see:
```
⚠️  ESCALATING TO HUMAN AGENT
User asked: <question>
Bot attempted: <bot's uncertain response>
👤 Human Agent — type your response: 
```
Type your response and press Enter. It gets sent to the user.

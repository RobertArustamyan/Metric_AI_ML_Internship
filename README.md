# Armenian Bank Voice AI Support Agent

A voice AI customer support agent that answers questions about **credits**, **deposits**, and **branch locations** for three Armenian banks: **Ameriabank**, **IDBank**, and **Mellat Bank**.

Built with the open-source [LiveKit](https://livekit.io/) framework. The agent understands and speaks **Armenian**.

## Architecture

```
User speaks Armenian
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  LiveKit Server (self-hosted)                        │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Voice Pipeline (agent/main.py)                 │ │
│  │                                                 │ │
│  │  1. Silero VAD ──── detects when user stops     │ │
│  │  2. OpenAI Whisper ── Armenian speech -> text   │ │
│  │  3. RAG Retriever ── finds relevant bank data   │ │
│  │  4. GPT-4o-mini ──── generates Armenian answer  │ │
│  │  5. OpenAI TTS ────── text -> Armenian speech   │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  ChromaDB Knowledge Base (chroma_db/)           │ │
│  │  1,638 chunks of scraped bank data              │ │
│  │  Embedded with text-embedding-3-large           │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
        │
        V
User hears Armenian response
```

## Model Choices & Justification

| Component | Model | Why                                                                                                                                                                                                     |
|-----------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **VAD** | Silero VAD | Runs locally, no API call. Detects speech boundaries with low latency.                                                                                                                                  |
| **STT** | OpenAI Whisper (`whisper-1`) | Multilingual STT available with strong Armenian support.                                                                                                                                                |
| **LLM** | GPT-4o-mini | Follows system prompt constraints reliably. Low latency for voice conversations.                                                                                                                        |
| **TTS** | OpenAI TTS (`tts-1`, voice: `nova`) | OpenAI TTS handles Armenian text reasonably well. To prevent English number pronunciation(not perfect), the system prompt instructs the LLM to write all numbers as Armenian words.                     |
| **Embeddings** | text-embedding-3-large | OpenAI's best multilingual embedding model. Handles Armenian text and cross-language queries .                                                                                                          |
| **Vector DB** | ChromaDB | Lightweight (no server process), file-based storage.                                                                                                                                                    |

## Project Structure

```
├── agent/                  # LiveKit voice agent
│   ├── main.py             # Voice pipeline (STT → RAG → LLM → TTS)
│   └── prompts.py          # System prompt with guardrails
│
├── retrieval/              # RAG knowledge base
│   ├── ingest.py           # Chunking + embedding + ChromaDB storage
│   └── retriever.py        # Query interface for the agent
│
├── scraper/                # Bank website scrapers
│   ├── base_scraper.py     # Abstract base class
│   ├── ameriabank_scraper.py
│   ├── idbank_scraper.py
│   ├── mellatbank_scraper.py
│   └── run_all.py          # Runs all scrapers in parallel
│
├── data/                   # Scraped bank data (JSON + TXT)
├── chroma_db/              # Pre-built vector knowledge base
├── logs/                   # Agent and ingestion logs
│
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.12+
- OpenAI API key

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Install LiveKit Server

```bash
curl -sSL https://get.livekit.io | bash
```

### 4. Download VAD model

```bash
python -m agent.main download-files
```

### 5. Run

**Terminal 1 — Start LiveKit server:**

```bash
livekit-server --dev
```

**Terminal 2 — Start the agent:**

```bash
# Dev mode (connects to LiveKit server, hot-reload)
python -m agent.main dev

# Or console mode (text I/O, no server needed)
python -m agent.main console
```

**Terminal 3 — Connect via browser playground:**

```bash
# Generate a token
python -c "
from livekit.api import AccessToken, VideoGrants
token = AccessToken('devkey', 'secret') \
    .with_identity('user1') \
    .with_grants(VideoGrants(room_join=True, room='test'))
print(token.to_jwt())
"
```

Open [agents-playground.livekit.io](https://agents-playground.livekit.io), select **Custom Server**, enter `ws://localhost:7880` and paste the token.

## Full Rebuild (Optional)

The repository includes pre-built `data/` and `chroma_db/` directories, so you can skip scraping and ingestion. To rebuild from scratch:

```bash
# 1. Scrape all bank websites
cd scraper
python run_all.py
cd ..

# 2. Build knowledge base
rm -rf chroma_db
python -m retrieval.ingest --data-dir data --db-dir chroma_db

# 3. Run the agent
python -m agent.main dev
```

## Guardrails

The agent has strict constraints enforced via the system prompt:

- **Scope**: Only answers about credits, deposits, and branches
- **Data source**: Only uses scraped bank data (RAG), never its own training knowledge
- **Language**: Always responds in Armenian
- **Accuracy**: Never invents numbers — if data is missing, it says so
- **Off-topic**: Politely declines non-banking questions

## Scalability

The system is designed to add more banks easily:

1. Create a new scraper class extending `BaseBankScraper`
2. Add it to `scraper/run_all.py`
3. Run the scraper, then re-run `retrieval/ingest.py`
4. The agent automatically picks up new bank data



---

*Built by Robert Arustamyan as part of Metric AI/ML Internship*
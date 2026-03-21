"""
Armenian Bank Voice AI Agent.

Pipeline: Silero VAD -> OpenAI Whisper STT -> RAG -> GPT-4o-mini -> OpenAI TTS

Usage:
python -m agent.main dev
python -m agent.main console
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, silero

# Add project root to path for retrieval import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import BankRetriever
from agent.prompts import SYSTEM_PROMPT_TEMPLATE, GREETING

load_dotenv()

logger = logging.getLogger("bank-agent")


# Logging
def setup_logging():
    """Log to both console and logs/agent.log."""
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)


# Global retriever

_retriever = None


def get_retriever():
    """load the BankRetriever singleton."""
    global _retriever
    if _retriever is None:
        db_dir = os.getenv("CHROMA_DB_DIR", "chroma_db")
        logger.info(f"Loading knowledge base from: {db_dir}")
        _retriever = BankRetriever(db_dir=db_dir)
    return _retriever


class BankSupportAgent(Agent):
    """
    Armenian bank support agent with RAG.
    """

    def __init__(self):
        super().__init__(
            instructions=SYSTEM_PROMPT_TEMPLATE.format(
                context="(Context will be loaded when the user asks a question.)"
            ),
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        """
        Called before every LLM invocation.
        Queries ChromaDB for relevant bank data and updates the system prompt.
        """
        # Extract user's latest message
        user_msg = ""
        for item in reversed(chat_ctx.items):
            if hasattr(item, 'role') and item.role == "user":
                if hasattr(item, 'text'):
                    user_msg = item.text
                elif hasattr(item, 'content'):
                    user_msg = str(item.content)
                break

        if user_msg:
            logger.info(f"User: {user_msg}")

            # RAG lookup
            retriever = get_retriever()
            context = retriever.query_as_context(user_msg, top_k=6)
            logger.info(f"RAG: retrieved {len(context)} chars")

            # Inject into system prompt
            updated = SYSTEM_PROMPT_TEMPLATE.format(context=context)
            if (chat_ctx.items
                    and hasattr(chat_ctx.items[0], 'role')
                    and chat_ctx.items[0].role == "system"):
                chat_ctx.items[0].content = updated

        # Pass to default LLM node
        async for chunk in Agent.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext):
    """Called when a user connects to a LiveKit room."""
    logger.info("Agent job started")

    # Pre-load retriever before connecting
    get_retriever()

    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")

    # Build the voice pipeline
    session = AgentSession(
        # VAD: Silero (local)
        vad=silero.VAD.load(),

        # STT: OpenAI Whisper
        stt=openai.STT(model="whisper-1",language="hy"),

        # LLM: GPT-4o-mini
        llm=openai.LLM(model="gpt-4o-mini",temperature=0.3),

        tts=openai.TTS(model="tts-1",voice="nova")
    )

    # Start with RAG-enhanced agent
    await session.start(agent=BankSupportAgent(),room=ctx.room)

    # Greet user in Armenian
    await session.generate_reply(instructions=GREETING,allow_interruptions=True,)

    logger.info("Agent is ready")


if __name__ == "__main__":
    setup_logging()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
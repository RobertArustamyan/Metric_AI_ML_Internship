"""
RAG Retriever.

Query interface over the ChromaDB knowledge base.
Used by the LiveKit agent to find relevant bank data for each user question.

Usage:
from retrieval.retriever import BankRetriever
retriever = BankRetriever(db_dir="chroma_db")
context = retriever.query_as_context("IDBank mortgage rate")
"""

import os
import logging
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("retrieval.retriever")

EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "bank_knowledge"

DEFAULT_TOP_K = 8
MAX_CONTEXT_CHARS = 6000


class BankRetriever:
    """
    Retrieves relevant bank data from the vector knowledge base.

    Supports:
    - Semantic search across all banks and categories
    - Filtering by bank or category (loans/deposits/branches)
    - Formatted context string ready for LLM injection
    """

    def __init__(self, db_dir="chroma_db", model_name=EMBEDDING_MODEL):
        if not os.path.exists(db_dir):
            raise FileNotFoundError(
                f"Knowledge base not found at '{db_dir}'. "
                f"Run `python -m retrieval.ingest` first."
            )

        self._openai = OpenAI()
        self._model_name = model_name
        self._client = chromadb.PersistentClient(path=db_dir)
        self._collection = self._client.get_collection(name=COLLECTION_NAME)

        logger.info(f"BankRetriever initialized: {self._collection.count()} chunks")

    def _embed(self, text):
        """Convert text to embedding vector via OpenAI API."""
        response = self._openai.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    def query(self, question, top_k=DEFAULT_TOP_K,
              bank=None, category=None):
        """
        Search the knowledge base by semantic similarity.

        Args:
            question: User's question
            top_k: Number of results to return
            bank: Filter by bank (e.g. "ameriabank", "idbank")
            category: Filter by category ("loans", "deposits", "branches")

        Returns:
            List of {text, metadata, distance}
        """
        query_embedding = self._embed(question)
        where = self._build_filter(bank, category)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        return output

    def query_as_context(self, question, top_k=DEFAULT_TOP_K,
                         bank=None, category=None,
                         max_chars=MAX_CONTEXT_CHARS):
        """
        Query and return formatted context string for LLM injection.

        This is the main method the agent calls. Returns the most relevant
        bank data chunks joined into a single string, capped at max_chars.
        """
        results = self.query(question, top_k, bank, category)

        if not results:
            return "No relevant data found in knowledge base."

        context_parts = []
        total_chars = 0

        for r in results:
            text = r["text"]
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining] + "..."
            context_parts.append(text)
            total_chars += len(text)

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self):
        """Return knowledge base statistics."""
        count = self._collection.count()
        sample = self._collection.peek(limit=min(count, 500))

        banks = set()
        categories = set()
        if sample["metadatas"]:
            for m in sample["metadatas"]:
                banks.add(m.get("bank", "unknown"))
                categories.add(m.get("category", "unknown"))

        return {
            "total_chunks": count,
            "banks": sorted(banks),
            "categories": sorted(categories),
            "embedding_model": self._model_name,
        }

    @staticmethod
    def _build_filter(bank=None, category=None):
        """Build ChromaDB where-clause filter."""
        conditions = []
        if bank:
            conditions.append({"bank": bank.lower().replace(" ", "_")})
        if category:
            conditions.append({"category": category.lower()})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG retriever")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--bank", "-b", default=None)
    parser.add_argument("--category", "-c", default=None)
    parser.add_argument("--top-k", "-k", type=int, default=5)
    args = parser.parse_args()

    retriever = BankRetriever(db_dir=args.db_dir)

    print(f"\nKB Stats: {retriever.get_stats()}")
    print(f"Query: {args.query}\n")

    results = retriever.query(args.query, args.top_k, args.bank, args.category)

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (distance: {r['distance']:.4f}) ---")
        print(f"  Bank: {r['metadata'].get('bank')}")
        print(f"  Category: {r['metadata'].get('category')}")
        print(f"  Title: {r['metadata'].get('title')}")
        print(f"  Preview: {r['text'][:150]}...\n")
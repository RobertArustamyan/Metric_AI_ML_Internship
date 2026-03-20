"""
RAG Retriever.

Provides a clean query interface over the ChromaDB knowledge base.
Used by the LiveKit agent to retrieve relevant bank data for each user question.

Usage:
    from rag.retriever import BankRetriever

    retriever = BankRetriever(db_dir="chroma_db")
    results = retriever.query("Ի՞նչ տոdelays է Ameriabank-ի delaysdelays")
    context = retriever.query_as_context("Ի՞նչ տodelays է ...")
"""

import os
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


# Must match what was used in ingest.py
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "bank_knowledge"

# Retrieval defaults
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 6000  # Max total context sent to LLM


class BankRetriever:
    """
    Retrieves relevant bank information from the vector knowledge base.

    Supports:
    - Semantic search across all banks and categories
    - Filtering by bank name or category (loans/deposits/branches)
    - Formatted context string ready for LLM injection
    """

    def __init__(self, db_dir: str = "chroma_db",
                 model_name: str = EMBEDDING_MODEL):
        """
        Initialize the retriever.

        Args:
            db_dir: Path to ChromaDB persistent storage
            model_name: Sentence transformer model (must match ingestion model)
        """
        if not os.path.exists(db_dir):
            raise FileNotFoundError(
                f"Knowledge base not found at '{db_dir}'. "
                f"Run `python -m rag.ingest` first."
            )

        self._model = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(path=db_dir)
        self._collection = self._client.get_collection(name=COLLECTION_NAME)

        print(f"BankRetriever initialized: {self._collection.count()} chunks loaded")

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        bank: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the knowledge base.

        Args:
            question: User's question (Armenian or English)
            top_k: Number of results to return
            bank: Filter by bank (e.g., "ameriabank", "idbank", "mellatbank")
            category: Filter by category ("loans", "deposits", "branches")

        Returns:
            List of dicts with keys: text, metadata, distance
        """
        # multilingual-e5 requires "query: " prefix for queries
        query_embedding = self._model.encode(
            f"query: {question}",
            show_progress_bar=False,
        ).tolist()

        # Build where filter
        where = self._build_filter(bank, category)

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return output

    def query_as_context(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        bank: Optional[str] = None,
        category: Optional[str] = None,
        max_chars: int = MAX_CONTEXT_CHARS,
    ) -> str:
        """
        Query and return a formatted context string ready for LLM injection.

        This is the main method the agent should call. It returns a single
        string containing the most relevant bank data, formatted for the
        LLM system prompt.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            bank: Optional bank filter
            category: Optional category filter
            max_chars: Maximum total characters in context

        Returns:
            Formatted context string with source attribution
        """
        results = self.query(question, top_k, bank, category)

        if not results:
            return "Տdelays չdelays գdelays:"  # "No relevant data found"

        context_parts = []
        total_chars = 0

        for r in results:
            text = r["text"]
            # Trim if adding this chunk would exceed limit
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining] + "..."

            context_parts.append(text)
            total_chars += len(text)

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        count = self._collection.count()

        # Sample metadata to count banks and categories
        sample = self._collection.peek(limit=min(count, 100))
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
            "embedding_model": EMBEDDING_MODEL,
            "collection": COLLECTION_NAME,
        }

    @staticmethod
    def _build_filter(bank: Optional[str],
                      category: Optional[str]) -> Optional[dict]:
        """Build ChromaDB where filter."""
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


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG retriever")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--query", "-q", required=True,
                        help="Question to search for")
    parser.add_argument("--bank", "-b", default=None,
                        help="Filter by bank name")
    parser.add_argument("--category", "-c", default=None,
                        help="Filter by category")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    args = parser.parse_args()

    retriever = BankRetriever(db_dir=args.db_dir)

    print(f"\nKB Stats: {retriever.get_stats()}")
    print(f"\nQuery: {args.query}")
    if args.bank:
        print(f"Bank filter: {args.bank}")
    if args.category:
        print(f"Category filter: {args.category}")

    print(f"\n{'='*60}")
    results = retriever.query(args.query, args.top_k, args.bank, args.category)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (distance: {r['distance']:.4f}) ---")
        print(f"Bank: {r['metadata'].get('bank')}")
        print(f"Category: {r['metadata'].get('category')}")
        print(f"Product: {r['metadata'].get('title')}")
        print(f"Section: {r['metadata'].get('section')}")
        print(f"Text preview: {r['text'][:200]}...")

    print(f"\n{'='*60}")
    print(f"\nFormatted context for LLM:")
    print(f"{'='*60}")
    context = retriever.query_as_context(args.query, args.top_k, args.bank, args.category)
    print(context[:2000])
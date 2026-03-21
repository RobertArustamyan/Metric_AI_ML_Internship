"""
RAG Knowledge Base Builder.

Loads scraped bank data (JSON), chunks it, embeds via OpenAI API,
and stores everything in ChromaDB for retrieval.

Usage:
python -m retrieval.ingest --data-dir data --db-dir chroma_db
"""

import argparse
import glob
import json
import os
import re
import time
import uuid
import logging
from collections import Counter

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("retrieval.ingest")

# Configuration

EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "bank_knowledge"
CHUNK_MAX_CHARS = 400
CHUNK_OVERLAP_CHARS = 150
BRANCH_GROUP_SIZE = 1
EMBED_BATCH_SIZE = 100
MAX_EMBED_CHARS = 800  # Safety truncation before embedding



class TextChunker:
    """Splits scraped bank content into retrieval-friendly chunks."""

    def __init__(self, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP_CHARS):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk_product_page(self, content, title, bank, category,
                           product_key, url):
        """Split a product page by [SectionName] markers from scraper output."""
        if not content.strip():
            return []

        header = f"Bank: {bank} | {title}"
        section_pattern = r'\[([^\]]+)\]'
        sections = re.split(section_pattern, content)

        chunks = []
        i = 0

        # Text before first section marker
        if sections and sections[0].strip():
            if not re.match(r'\[.+\]', sections[0]):
                chunks.extend(self._split(sections[0].strip(), header, bank, category,product_key, title, url, "General"))
            i = 1
        elif sections and not sections[0].strip():
            i = 1

        # Process (marker, text) pairs
        while i < len(sections) - 1:
            section_name = sections[i].strip()
            section_text = sections[i + 1].strip() if i + 1 < len(sections) else ""
            i += 2

            if not section_text:
                continue

            section_header = f"{header} | {section_name}"
            chunks.extend(self._split(
                section_text, section_header, bank, category,
                product_key, title, url, section_name
            ))

        # Fallback: no markers found
        if not chunks and content.strip():
            chunks.extend(self._split(
                content.strip(), header, bank, category,
                product_key, title, url, "General"
            ))

        return chunks

    def chunk_branches(self, branches, bank, bank_name, source_url):
        """One chunk per branch for precise retrieval."""
        chunks = []
        for i, b in enumerate(branches):
            lines = [f"Bank: {bank_name} | Branches"]
            lines.append(f"\n{b['name']}")
            lines.append(f"  Address: {b['address']}")
            if b.get('phone'):
                lines.append(f"  Phone: {b['phone']}")
            if b.get('schedule'):
                lines.append(f"  Hours: {b['schedule']}")
            if b.get('description'):
                lines.append(f"  Note: {b['description']}")

            chunks.append({
                "text": "\n".join(lines),
                "metadata": {
                    "bank": bank,
                    "category": "branches",
                    "product_key": "branches",
                    "title": f"{bank_name} branch: {b['name']}",
                    "section": "branches",
                    "url": source_url,
                }
            })
        return chunks

    def _split(self, text, header, bank, category,
               product_key, title, url, section):
        """Split text into chunks. Falls back: paragraph -> sentence -> hard cut."""
        meta = {
            "bank": bank, "category": category, "product_key": product_key,
            "title": title, "section": section, "url": url,
        }

        full_text = f"{header}\n\n{text}"
        if len(full_text) <= self.max_chars:
            return [{"text": full_text, "metadata": dict(meta)}]

        # Try paragraphs first, then sentences
        paragraphs = text.split("\n\n")
        units = paragraphs if len(paragraphs) > 1 else re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = ""
        header_len = len(header) + 2

        for unit in units:
            test_len = header_len + len(current) + 2 + len(unit) if current else header_len + len(unit)

            if test_len > self.max_chars and current:
                chunks.append({"text": f"{header}\n\n{current}", "metadata": dict(meta)})
                # Keep overlap for context continuity
                if len(current) > self.overlap:
                    current = current[-self.overlap:] + "\n\n" + unit
                else:
                    current = unit
            else:
                current = f"{current}\n\n{unit}" if current else unit

        # Remaining text
        if current.strip():
            final = f"{header}\n\n{current}"
            if len(final) > self.max_chars * 2:
                # Hard split for very long single sentences
                body = current
                while body:
                    cut = self.max_chars - header_len - 10
                    piece = body[:cut]
                    body = body[cut:]
                    chunks.append({"text": f"{header}\n\n{piece}", "metadata": dict(meta)})
            else:
                chunks.append({"text": final, "metadata": dict(meta)})

        return chunks


class BankDataLoader:
    """Loads scraped JSON files and converts them to chunks."""

    def __init__(self, data_dir, chunker=None):
        self.data_dir = data_dir
        self.chunker = chunker or TextChunker()

    def load_all(self):
        """Auto-detect and load all bank JSON files from data directory."""
        all_chunks = []
        json_files = sorted(glob.glob(os.path.join(self.data_dir, "*.json")))

        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            return []

        logger.info(f"Found {len(json_files)} JSON files")

        for filepath in json_files:
            filename = os.path.basename(filepath)
            size_kb = os.path.getsize(filepath) / 1024

            try:
                if "_branches" in filename:
                    chunks = self._load_branches(filepath)
                elif "_loans" in filename or "_deposits" in filename:
                    chunks = self._load_products(filepath)
                else:
                    logger.debug(f"Skipping unknown file: {filename}")
                    continue

                logger.info(f"  {filename} ({size_kb:.0f} KB) -> {len(chunks)} chunks")
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"  Failed to load {filename}: {e}")

        return all_chunks

    def _load_products(self, filepath):
        """Load a loans/deposits JSON and chunk each page."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        bank = data.get("bank_name_en", "Unknown").lower().replace(" ", "_")
        category = data.get("category", "unknown")
        chunks = []

        for key, page in data.get("pages", {}).items():
            content = page.get("content", "")
            if not content or len(content) < 30:
                continue

            page_chunks = self.chunker.chunk_product_page(
                content=content,
                title=page.get("title", key),
                bank=bank,
                category=category,
                product_key=key,
                url=page.get("url", ""),
            )
            chunks.extend(page_chunks)

        return chunks

    def _load_branches(self, filepath):
        """Load a branches JSON and chunk each branch."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        bank = data.get("bank_name_en", "Unknown").lower().replace(" ", "_")
        bank_name = data.get("bank_name", data.get("bank_name_en", "Unknown"))
        branches = data.get("branches", [])

        if not branches:
            return []

        return self.chunker.chunk_branches(
            branches=branches,
            bank=bank,
            bank_name=bank_name,
            source_url=data.get("source_url", ""),
        )


class OpenAIEmbedder:
    """Generates embeddings via OpenAI API."""

    def __init__(self, model=EMBEDDING_MODEL, batch_size=EMBED_BATCH_SIZE):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed_chunks(self, chunks):
        """Embed all chunks, returns list of embedding vectors."""
        # Safety truncation for token limit
        texts = []
        for c in chunks:
            text = c["text"]
            if len(text) > MAX_EMBED_CHARS:
                text = text[:MAX_EMBED_CHARS] + "..."
                c["text"] = text
            texts.append(text)

        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"  Embedded {i + len(batch)}/{total} chunks")
            except Exception as e:
                logger.error(f"  Embedding failed at batch {i // self.batch_size + 1}: {e}")
                raise

            if i + self.batch_size < total:
                time.sleep(0.2)

        logger.info(f"  Embedding dim: {len(all_embeddings[0])}")
        return all_embeddings


class KnowledgeBaseBuilder:
    """Orchestrates the full ingestion pipeline: load -> chunk -> embed -> store."""

    def __init__(self, data_dir, db_dir, model=EMBEDDING_MODEL):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.loader = BankDataLoader(data_dir)
        self.embedder = OpenAIEmbedder(model=model)

    def build(self):
        """Run the full pipeline."""
        logger.info("=" * 60)
        logger.info("Building RAG Knowledge Base")
        logger.info(f"  Data dir:    {self.data_dir}")
        logger.info(f"  DB dir:      {self.db_dir}")
        logger.info(f"  Embed model: {self.embedder.model}")
        logger.info("=" * 60)

        # Step 1: Load and chunk
        logger.info("STEP 1: Loading and chunking data...")
        chunks = self.loader.load_all()
        if not chunks:
            logger.error("No data found. Run scrapers first.")
            return
        logger.info(f"Total chunks: {len(chunks)}")

        # Step 2: Embed
        logger.info("STEP 2: Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)

        # Step 3: Store in ChromaDB
        logger.info("STEP 3: Storing in ChromaDB...")
        self._store(chunks, embeddings)

        # Summary
        self._print_summary(chunks)

    def _store(self, chunks, embeddings):
        """Write chunks + embeddings to ChromaDB."""
        db_client = chromadb.PersistentClient(path=self.db_dir)

        # Drop old collection if re-ingesting
        try:
            db_client.delete_collection(COLLECTION_NAME)
            logger.info("  Deleted old collection")
        except Exception:
            pass

        collection = db_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Add in batches
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks[i:end]],
                documents=[c["text"] for c in chunks[i:end]],
                embeddings=embeddings[i:end],
                metadatas=[c["metadata"] for c in chunks[i:end]],
            )
            logger.info(f"  Stored {end}/{len(chunks)} chunks")

        logger.info("=" * 60)
        logger.info(f"DONE! {collection.count()} chunks stored at {self.db_dir}")
        logger.info("=" * 60)

    @staticmethod
    def _print_summary(chunks):
        """Log chunk counts by bank and category."""
        counts = Counter()
        for c in chunks:
            m = c["metadata"]
            counts[(m["bank"], m["category"])] += 1

        logger.info("Breakdown:")
        for (bank, cat), count in sorted(counts.items()):
            logger.info(f"  {bank}/{cat}: {count} chunks")


def setup_logging():
    """Configure logging to both console and file."""
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler("logs/ingest.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


if __name__ == "__main__":
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(description="Build RAG knowledge base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder(args.data_dir, args.db_dir, args.model)
    builder.build()
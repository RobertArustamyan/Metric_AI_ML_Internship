"""
RAG Knowledge Base Builder.

Loads scraped bank data (JSON), chunks it intelligently,
generates embeddings with a multilingual model, and stores
everything in a ChromaDB collection.

Run once after scraping, or re-run when data changes.

Usage:
    python -m rag.ingest --data-dir data --db-dir chroma_db
"""

import argparse
import glob
import json
import os
import re
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "bank_knowledge"
CHUNK_MAX_CHARS = 1500  # Target max per chunk
CHUNK_OVERLAP_CHARS = 150  # Overlap between split chunks
BRANCH_GROUP_SIZE = 4  # Branches per chunk


# ── Chunking Logic ───────────────────────────────────────────────────────────

def chunk_product_page(content: str, title: str, bank: str, category: str,
                       product_key: str, url: str) -> list[dict]:
    """
    Chunk a product page (loan or deposit) into retrieval-friendly pieces.

    Strategy:
    1. Split by [SectionName] markers (from scraper output)
    2. If a section exceeds CHUNK_MAX_CHARS, split further at paragraph boundaries
    3. Prepend context header to each chunk so it's self-contained
    """
    # Context line prepended to every chunk for self-containment
    header = f"Բանկ: {bank} | {title}"

    # Split by section markers like [Summary], [Overview], [FAQ], etc.
    section_pattern = r'\[([^\]]+)\]'
    sections = re.split(section_pattern, content)

    chunks = []

    # sections alternates: [text_before_first_marker, marker1, text1, marker2, text2, ...]
    # If content starts with a marker, sections[0] is empty
    i = 0
    if sections and not sections[0].strip():
        i = 1  # Skip empty leading text

    # Handle any text before the first section marker
    if sections and sections[0].strip() and not re.match(section_pattern, '[' + sections[0] + ']'):
        pre_text = sections[0].strip()
        if pre_text:
            chunks.extend(
                _split_text(pre_text, header, bank, category, product_key,
                            title, url, "General")
            )
        i = 1

    # Process section pairs: (marker, text)
    while i < len(sections) - 1:
        section_name = sections[i].strip()
        section_text = sections[i + 1].strip() if i + 1 < len(sections) else ""
        i += 2

        if not section_text:
            continue

        section_header = f"{header} | {section_name}"
        chunks.extend(
            _split_text(section_text, section_header, bank, category,
                        product_key, title, url, section_name)
        )

    # If no sections were found (no markers), chunk the whole content
    if not chunks and content.strip():
        chunks.extend(
            _split_text(content.strip(), header, bank, category,
                        product_key, title, url, "General")
        )

    return chunks


def _split_text(text: str, header: str, bank: str, category: str,
                product_key: str, title: str, url: str,
                section: str) -> list[dict]:
    """
    Split text into chunks of CHUNK_MAX_CHARS with overlap.
    Falls back from paragraph → sentence → hard char split.
    Returns list of chunk dicts with text and metadata.
    """
    meta = {
        "bank": bank,
        "category": category,
        "product_key": product_key,
        "title": title,
        "section": section,
        "url": url,
    }

    # If short enough, return as single chunk
    full_text = f"{header}\n\n{text}"
    if len(full_text) <= CHUNK_MAX_CHARS:
        return [{"text": full_text, "metadata": dict(meta)}]

    # Try splitting at paragraph boundaries first, then sentences
    paragraphs = text.split("\n\n")
    # If only one big paragraph, split further by sentences
    if len(paragraphs) <= 1:
        # Split on sentence endings (. ? ! followed by space or newline)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        units = sentences
    else:
        units = paragraphs

    chunks = []
    current = ""
    header_len = len(header) + 2  # +2 for "\n\n"

    for unit in units:
        test_len = header_len + len(current) + 2 + len(unit) if current else header_len + len(unit)
        if test_len > CHUNK_MAX_CHARS and current:
            chunks.append({"text": f"{header}\n\n{current}", "metadata": dict(meta)})
            # Keep overlap
            if len(current) > CHUNK_OVERLAP_CHARS:
                current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + unit
            else:
                current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit

    # Last chunk
    if current.strip():
        final = f"{header}\n\n{current}"
        # If still too long (single giant sentence), hard-split by chars
        if len(final) > CHUNK_MAX_CHARS * 2:
            body = current
            while body:
                cut = CHUNK_MAX_CHARS - header_len - 10
                piece = body[:cut]
                body = body[cut:]
                chunks.append({"text": f"{header}\n\n{piece}", "metadata": dict(meta)})
        else:
            chunks.append({"text": final, "metadata": dict(meta)})

    return chunks


def chunk_branches(branches: list, bank: str, bank_name: str,
                   source_url: str) -> list[dict]:
    """
    Chunk branch data into groups of BRANCH_GROUP_SIZE.
    Each chunk contains formatted branch info.
    """
    chunks = []

    for i in range(0, len(branches), BRANCH_GROUP_SIZE):
        group = branches[i:i + BRANCH_GROUP_SIZE]

        lines = [f"Բանկ: {bank_name} | Մասնաճյուdelays"]
        for b in group:
            lines.append(f"\n{b['name']}")
            lines.append(f"  Հdelays: {b['address']}")
            if b.get('phone'):
                lines.append(f"  Հeraxos: {b['phone']}")
            if b.get('schedule'):
                lines.append(f"  Grafik: {b['schedule']}")
            if b.get('description'):
                lines.append(f"  Note: {b['description']}")

        chunk_text = "\n".join(lines)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "bank": bank,
                "category": "branches",
                "product_key": "branches",
                "title": f"{bank_name} branches ({i+1}-{i+len(group)})",
                "section": "branches",
                "url": source_url,
            }
        })

    return chunks


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_product_json(filepath: str) -> list[dict]:
    """Load a product JSON file and return chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bank_en = data.get("bank_name_en", "Unknown")
    bank_name = data.get("bank_name", bank_en)
    category = data.get("category", "unknown")
    pages = data.get("pages", {})

    all_chunks = []
    for key, page_data in pages.items():
        content = page_data.get("content", "")
        if not content or len(content) < 30:
            continue

        title = page_data.get("title", key)
        url = page_data.get("url", "")

        page_chunks = chunk_product_page(
            content=content,
            title=title,
            bank=bank_en.lower().replace(" ", "_"),
            category=category,
            product_key=key,
            url=url,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def load_branches_json(filepath: str) -> list[dict]:
    """Load a branches JSON file and return chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bank_en = data.get("bank_name_en", "Unknown")
    bank_name = data.get("bank_name", bank_en)
    branches = data.get("branches", [])
    source_url = data.get("source_url", "")

    if not branches:
        return []

    return chunk_branches(
        branches=branches,
        bank=bank_en.lower().replace(" ", "_"),
        bank_name=bank_name,
        source_url=source_url,
    )


def load_all_data(data_dir: str) -> list[dict]:
    """
    Load all scraped JSON files from the data directory.
    Auto-detects product files (*_loans.json, *_deposits.json) and
    branch files (*_branches.json).
    """
    all_chunks = []

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return []

    for filepath in sorted(json_files):
        filename = os.path.basename(filepath)
        print(f"Loading: {filename}")

        try:
            if "_branches" in filename:
                chunks = load_branches_json(filepath)
            elif "_loans" in filename or "_deposits" in filename:
                chunks = load_product_json(filepath)
            else:
                print(f"  Skipping unknown file: {filename}")
                continue

            print(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    return all_chunks


# ── ChromaDB Storage ─────────────────────────────────────────────────────────

def build_knowledge_base(data_dir: str, db_dir: str,
                         model_name: str = EMBEDDING_MODEL):
    """
    Main ingestion pipeline:
    1. Load all scraped JSON
    2. Chunk into retrieval-friendly pieces
    3. Embed with multilingual model
    4. Store in ChromaDB
    """
    print(f"\n{'='*60}")
    print(f"Building RAG Knowledge Base")
    print(f"{'='*60}")
    print(f"Data dir:   {data_dir}")
    print(f"DB dir:     {db_dir}")
    print(f"Embed model: {model_name}")
    print(f"{'='*60}\n")

    # 1. Load and chunk all data
    chunks = load_all_data(data_dir)
    if not chunks:
        print("No data to ingest. Run scrapers first.")
        return

    print(f"\nTotal chunks: {len(chunks)}")

    # 2. Load embedding model
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # 3. Generate embeddings
    # multilingual-e5 requires "query: " or "passage: " prefix
    texts = [f"passage: {c['text']}" for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    print(f"Embedding dim: {embeddings.shape[1]}")

    # 4. Store in ChromaDB
    print(f"\nStoring in ChromaDB at {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)

    # Delete existing collection if re-ingesting
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches (ChromaDB limit)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end].tolist()

        ids = [str(uuid.uuid4()) for _ in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [c["metadata"] for c in batch_chunks]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        print(f"  Added batch {i//batch_size + 1}: {len(batch_chunks)} chunks")

    print(f"\n{'='*60}")
    print(f"Knowledge base built successfully!")
    print(f"  Total chunks: {collection.count()}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Stored at: {db_dir}")
    print(f"{'='*60}")

    # Print summary by bank and category
    _print_summary(chunks)


def _print_summary(chunks: list[dict]):
    """Print a summary of ingested data."""
    from collections import Counter

    bank_counts = Counter()
    cat_counts = Counter()
    bank_cat = Counter()

    for c in chunks:
        m = c["metadata"]
        bank_counts[m["bank"]] += 1
        cat_counts[m["category"]] += 1
        bank_cat[(m["bank"], m["category"])] += 1

    print(f"\nBy bank:")
    for bank, count in sorted(bank_counts.items()):
        print(f"  {bank}: {count} chunks")

    print(f"\nBy category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} chunks")

    print(f"\nBy bank × category:")
    for (bank, cat), count in sorted(bank_cat.items()):
        print(f"  {bank}/{cat}: {count} chunks")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAG knowledge base from scraped bank data"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing scraped JSON files (default: data)"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Directory for ChromaDB storage (default: chroma_db)"
    )
    parser.add_argument(
        "--model", default=EMBEDDING_MODEL,
        help=f"Embedding model name (default: {EMBEDDING_MODEL})"
    )
    args = parser.parse_args()

    build_knowledge_base(args.data_dir, args.db_dir, args.model)
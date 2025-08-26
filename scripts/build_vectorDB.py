#!/usr/bin/env python3
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def load_metadata(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_text(entry: dict) -> str:
    """
    Constructs semantic text for vector embedding based on schema entry.
    """
    base = (
        f"Table: {entry.get('table')}, "
        f"Column: {entry.get('column')}, "
        f"Datatype: {entry.get('datatype')}, "
        f"Type: {entry.get('type')}, "
    )
    
    if entry.get("type") == "xml":
        xml_col = entry.get("xml_column", "")
        xpath = entry.get("example_query_syntax", "")
        base += f"XML Column: {xml_col}, XPath: {xpath}, "

    description = entry.get("description", "").strip()
    return base + f"Description: {description}"


def main():
    # ─── CONFIG ─────────────────────────────────────────────────────────────
    repo_root = Path(__file__).resolve().parent.parent
    metadata_path = repo_root / "data" / "metadata" / "schema.json"
    vectordb_folder = repo_root / "data" / "embeddings" / "chroma"
    collection_name = "metadata_collection"
    # ─────────────────────────────────────────────────────────────────────────

    # Sanity check metadata file
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata file: {metadata_path}")

    vectordb_folder.mkdir(parents=True, exist_ok=True)

    print(f"▶ Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"  ↳ {len(metadata)} entries found\n")

    print("▶ Loading embedding model: all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n▶ Initializing Chroma PersistentClient at:\n   {vectordb_folder}")
    client = chromadb.PersistentClient(path=str(vectordb_folder))

    # Delete existing collection if it exists
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)
        print(f"  ↳ Deleted existing collection '{collection_name}'")

    # Create new collection
    coll = client.create_collection(name=collection_name)
    print(f"✅ Created collection '{collection_name}'\n")

    # Prepare data
    docs, metas, ids = [], [], []
    for i, entry in enumerate(metadata):
        txt = build_text(entry)
        docs.append(txt)
        metas.append(entry)
        ids.append(f"meta_{i}")

    print("▶ Generating embeddings...")
    embs = embedder.encode(docs, show_progress_bar=True)

    print("\n▶ Adding documents to Chroma...")
    coll.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embs.tolist() # type: ignore
    )

    print(f"\n✅ Vector DB ready at {vectordb_folder.resolve()}")
    print(f"   Collection '{collection_name}' contains {len(ids)} vectors.")


if __name__ == "__main__":
    main()

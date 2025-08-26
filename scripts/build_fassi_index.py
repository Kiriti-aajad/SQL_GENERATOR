# scripts/build_faiss_index.py

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle


def load_metadata(schema_path: Path):
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ─── Config ────────────────────────────────────────────────
    repo_root = Path(__file__).parent.parent
    schema_path = repo_root / "data" / "metadata" / "schema.json"
    index_folder = repo_root / "data" / "embeddings" / "faiss"
    embed_model_name = "all-MiniLM-L6-v2"
    index_file = index_folder / "index.faiss"
    metadata_file = index_folder / "metadata.pkl"
    # ────────────────────────────────────────────────────────────

    index_folder.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(schema_path)
    print(f"Loaded {len(metadata)} metadata entries")

    print(f"▶ Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    # Prepare embedding texts
    texts = []
    metadatas = []
    for entry in metadata:
        desc = entry.get("description", "").strip().replace("\n", " ")
        text = (
            f"Table: {entry.get('table')}, "
            f"Column: {entry.get('column')}, "
            f"Datatype: {entry.get('datatype')}, "
            f"Type: {entry.get('type')}, "
            f"Description: {desc}"
        )
        texts.append(text)
        metadatas.append(entry)

    print("▶ Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("▶ Building FAISS index...")
    dim = embeddings.shape[1] # type: ignore
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings) # type: ignore

    # Save index and metadata
    faiss.write_index(index, str(index_file))
    with metadata_file.open("wb") as f:
        pickle.dump(metadatas, f)

    print(f"FAISS index saved to: {index_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"FAISS collection contains {len(embeddings)} vectors")


if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path
from typing import List, Tuple

from sqlalchemy.orm import Session
from openai import OpenAI

from scripts.gdpr_highlight_agent import extract_doc_text
from .db import SessionLocal, Embedding


EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
	text = text.strip()
	if not text:
		return []
	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(len(text), start + max_chars)
		chunk = text[start:end]
		chunks.append(chunk)
		if end == len(text):
			break
		start = max(0, end - overlap)
	return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
	resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
	return [d.embedding for d in resp.data]


def ingest_pdfs(src_roots: List[Path]) -> Tuple[int, int]:
	db: Session = SessionLocal()
	openai_client = OpenAI()

	added = 0
	skipped = 0
	try:
		for root in src_roots:
			if not root.exists():
				continue
			for pdf in sorted(root.rglob("*.pdf")):
				try:
					text = extract_doc_text(pdf)
					chunks = chunk_text(text)
					if not chunks:
						skipped += 1
						continue
					embs = embed_texts(openai_client, chunks)
					for i, (chunk, emb) in enumerate(zip(chunks, embs)):
						row = Embedding(
							pdf_path=pdf.as_posix(),
							chunk_index=i,
							text=chunk,
							vector=json.dumps(emb),
						)
						db.add(row)
						added += 1
					db.commit()
				except Exception:
					skipped += 1
	finally:
		db.close()
	return added, skipped



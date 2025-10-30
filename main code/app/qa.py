import os
import json
from typing import List, Tuple
import numpy as np

from openai import OpenAI
from sqlalchemy.orm import Session

from .db import SessionLocal, Embedding


CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
	den = (np.linalg.norm(a) * np.linalg.norm(b))
	if den == 0:
		return 0.0
	return float(np.dot(a, b) / den)


def answer_question(question: str, top_k: int = 6) -> Tuple[str, List[dict]]:
	client = OpenAI()
	q_emb = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
	q_vec = np.array(q_emb, dtype=np.float32)

	db: Session = SessionLocal()
	rows = db.query(Embedding).all()
	# Compute similarity brute-force
	scored: List[Tuple[float, Embedding]] = []
	for r in rows:
		vec = np.array(json.loads(r.vector), dtype=np.float32)
		score = cosine_sim(q_vec, vec)
		scored.append((score, r))
	scored.sort(key=lambda x: x[0], reverse=True)
	top = scored[:top_k]

	context_blocks: List[str] = [r.text for _, r in top]
	sources: List[dict] = [{"pdf_path": r.pdf_path, "chunk_index": r.chunk_index} for _, r in top]

	prompt = (
		"You are a helpful assistant. Answer the question using ONLY the provided context. "
		"If the answer is not present, say you don't know.\n\n" 
		f"Context:\n{\n\n.join(context_blocks)}\n\nQuestion: {question}\nAnswer:"
	)

	chat = client.chat.completions.create(
		model=CHAT_MODEL,
		messages=[{"role": "user", "content": prompt}],
		temperature=0.0,
	)
	answer = chat.choices[0].message.content.strip()
	return answer, sources



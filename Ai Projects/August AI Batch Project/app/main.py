import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import init_db, User
from .auth import get_db, hash_password, verify_password, create_session, require_user
from .ingest import ingest_pdfs
from .qa import answer_question


app = FastAPI(title="PDF GDPR QA")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
	init_db()


@app.post("/auth/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
	if db.query(User).filter(User.username == username).first():
		raise HTTPException(status_code=400, detail="Username already exists")
	user = User(username=username, password_hash=hash_password(password))
	db.add(user)
	db.commit()
	token = create_session(db, user)
	return {"token": token}


@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
	user = db.query(User).filter(User.username == username).first()
	if not user or not verify_password(password, user.password_hash):
		raise HTTPException(status_code=401, detail="Invalid credentials")
	token = create_session(db, user)
	return {"token": token}


@app.post("/ingest")
def ingest(token: str = Form(...), root: Optional[str] = Form(None), db: Session = Depends(get_db)):
	user = require_user(db, token)
	src_roots = []
	if root:
		src_roots = [Path(root)]
	else:
		src_roots = [Path("output") / "processed", Path("output") / "images_as_pdfs"]
	added, skipped = ingest_pdfs(src_roots)
	return {"added": added, "skipped": skipped}


@app.post("/ask")
def ask(token: str = Form(...), question: str = Form(...), db: Session = Depends(get_db)):
	user = require_user(db, token)
	answer, sources = answer_question(question)
	return {"answer": answer, "sources": sources}


@app.get("/")
def index() -> HTMLResponse:
	html = (Path("web") / "index.html").read_text(encoding="utf-8") if (Path("web") / "index.html").exists() else "<h1>Server running</h1>"
	return HTMLResponse(html)



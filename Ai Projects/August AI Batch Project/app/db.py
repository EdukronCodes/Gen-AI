from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime


DB_PATH = Path("data") / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class User(Base):
	__tablename__ = "users"
	id = Column(Integer, primary_key=True)
	username = Column(String(64), unique=True, nullable=False, index=True)
	password_hash = Column(String(200), nullable=False)
	created_at = Column(DateTime, default=datetime.utcnow)
	sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
	__tablename__ = "sessions"
	id = Column(Integer, primary_key=True)
	user_id = Column(Integer, ForeignKey("users.id"))
	token = Column(String(128), unique=True, index=True, nullable=False)
	created_at = Column(DateTime, default=datetime.utcnow)
	user = relationship("User", back_populates="sessions")


class Embedding(Base):
	__tablename__ = "embeddings"
	id = Column(Integer, primary_key=True)
	pdf_path = Column(String(1024), index=True, nullable=False)
	chunk_index = Column(Integer, nullable=False)
	text = Column(Text, nullable=False)
	vector = Column(Text, nullable=False)  # JSON-serialized list[float]
	created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
	Base.metadata.create_all(bind=engine)



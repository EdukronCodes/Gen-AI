import secrets
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from .db import SessionLocal, User, Session as UserSession


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def hash_password(password: str) -> str:
	return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
	return pwd_context.verify(password, password_hash)


def require_user(db: Session, token: str) -> User:
	sess = db.query(UserSession).filter(UserSession.token == token).first()
	if not sess:
		raise HTTPException(status_code=401, detail="Invalid token")
	return sess.user


def create_session(db: Session, user: User) -> str:
	token = secrets.token_hex(32)
	db.add(UserSession(user=user, token=token))
	db.commit()
	return token



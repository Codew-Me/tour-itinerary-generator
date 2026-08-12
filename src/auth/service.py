"""Authentication service."""

from datetime import datetime, timedelta

import bcrypt
import jwt
from sqlalchemy.orm import Session

from src.config import get_settings
from src.database.models import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, email: str) -> str:
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None


def register_user(session: Session, email: str, password: str, full_name: str | None = None) -> User:
    existing = session.query(User).filter(User.email == email.lower()).first()
    if existing:
        raise ValueError("Email already registered")
    user = User(
        email=email.lower().strip(),
        password_hash=hash_password(password),
        full_name=full_name,
    )
    session.add(user)
    session.flush()
    return user


def authenticate_user(session: Session, email: str, password: str) -> User | None:
    user = session.query(User).filter(User.email == email.lower()).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None

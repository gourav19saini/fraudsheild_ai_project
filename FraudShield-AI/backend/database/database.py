"""
FraudShield AI - Database Setup
==================================
SQLite database via SQLAlchemy. Auto-creates all tables on first import.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

os.makedirs("data", exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/fraudshield.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Import models so they register with Base, then create all tables."""
    from backend.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

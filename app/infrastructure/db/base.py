from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine

from app.infrastructure.config import DB_URL 

engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

Base = declarative_base()
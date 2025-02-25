from sqlalchemy.ext.asyncio import create_async_engine
from config import db_host, db_name, db_password, db_username
database_url: str = f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:5432/{db_name}"
engine = create_async_engine(str(database_url))

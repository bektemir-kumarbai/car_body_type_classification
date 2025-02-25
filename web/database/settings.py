from sqlalchemy.ext.asyncio import create_async_engine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import db_host, db_name, db_password, db_username, db_port
database_url: str = f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_async_engine(str(database_url))

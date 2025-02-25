import os
import sys
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from alembic import context

# Добавляем корневую директорию в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Импортируем конфигурацию БД
from web.database.settings import database_url
from web.media_handler.models import Media as media_models

# Настраиваем Alembic
print(f"Using database: {database_url}")  # Для отладки
config = context.config
config.set_main_option("sqlalchemy.url", database_url)

# Настройка логирования, если файл конфигурации указан
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Указываем метаданные для автогенерации
target_metadata = media_models.metadata


def run_migrations_offline() -> None:
    """Запуск миграций в оффлайн-режиме (без подключения к БД)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Запуск миграций в онлайн-режиме с асинхронным подключением."""
    connectable = create_async_engine(database_url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_migrations)


def do_migrations(connection: AsyncConnection):
    """Функция для выполнения миграций в синхронном режиме."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


# Запуск миграций
if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_migrations_online())

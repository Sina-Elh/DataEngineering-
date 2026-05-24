import os
import time
from sqlalchemy import create_engine, text

def get_db_url() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ecommerce_db")
    user = os.getenv("POSTGRES_USER", "de_user")
    password = os.getenv("POSTGRES_PASSWORD", "de_password")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

def get_engine(retries: int = 30, delay: int = 2):
    db_url = get_db_url()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"Database connection established on attempt {attempt}.")
            return engine
        except Exception as exc:
            last_error = exc
            print(f"Database not ready yet, attempt {attempt}/{retries}: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to database after {retries} attempts: {last_error}")

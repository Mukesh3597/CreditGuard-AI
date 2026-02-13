import os
import sqlite3
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(PROJECT_ROOT, "db")
DB_PATH = os.path.join(DB_DIR, "creditguard.db")
SQL_CREATE_PATH = os.path.join(PROJECT_ROOT, "sql", "01_create_tables.sql")


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    with open(SQL_CREATE_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def insert_prediction(row: dict):
    conn = get_connection()
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    values = list(row.values())

    query = f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"
    conn.execute(query, values)
    conn.commit()
    conn.close()


def fetch_history(limit: int = 20) -> pd.DataFrame:
    conn = get_connection()
    q = f"""
    SELECT *
    FROM predictions
    ORDER BY id DESC
    LIMIT {limit}
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

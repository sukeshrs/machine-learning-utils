from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import os

app = FastAPI()
DB_CONN = os.getenv(
    "DB_CONN", "dbname=memory user=postgres password=secret host=localhost")


def get_conn():
    return psycopg2.connect(DB_CONN)


class MemoryItem(BaseModel):
    session_id: str
    summary: str


@app.post("/memory/save")
def save_memory(item: MemoryItem):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            summary TEXT
        )
    """)
    cur.execute("INSERT INTO memory (session_id, summary) VALUES (%s, %s)",
                (item.session_id, item.summary))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/memory/search")
def search_memory(query: str, limit: int = 3):
    # Simple text search — can be swapped for embeddings later
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT session_id, summary 
        FROM memory 
        WHERE summary ILIKE %s 
        ORDER BY id DESC 
        LIMIT %s
    """, (f"%{query}%", limit))
    rows = cur.fetchall()
    conn.close()
    return [{"session_id": r[0], "summary": r[1]} for r in rows]

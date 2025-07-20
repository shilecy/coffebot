import sqlite3
from app.llm_sql_generator import generate_sql_query, extract_sql_codeblock

DB_PATH = "data/outlets.db"

def query_outlets_from_db(question: str):
    sql_raw = generate_sql_query(question)
    sql = extract_sql_codeblock(sql_raw)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        result = [
            {k: v for k, v in zip(columns, row) if k != "id"}
            for row in rows
        ]
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}

    finally:
        conn.close()

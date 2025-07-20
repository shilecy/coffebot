import csv
import sqlite3
import os

DB_PATH = "data/outlets.db"
CSV_PATH = "data/outlets_fixed.csv"

# Ensure DB and table exist
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS outlets (
        id INTEGER PRIMARY KEY,
        name TEXT,
        address TEXT,
        hours TEXT,
        services TEXT
    )
""")

# Import CSV
with open(CSV_PATH, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    rows = [
        (row["id"], row["name"], row["address"], row["hours"], row["services"])
        for row in reader
    ]

cursor.executemany(
    "INSERT OR REPLACE INTO outlets (id, name, address, hours, services) VALUES (?, ?, ?, ?, ?)",
    rows
)
conn.commit()
conn.close()

print("CSV imported to database!")
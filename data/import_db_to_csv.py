import sqlite3
import csv

conn = sqlite3.connect("data/outlets.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM outlets")
rows = cursor.fetchall()

# Get column names
column_names = [description[0] for description in cursor.description]

with open("data/outlets.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(column_names)  # header
    writer.writerows(rows)

conn.close()
print("✅ Exported to data/outlets.csv")

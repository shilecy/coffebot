import requests
from bs4 import BeautifulSoup
import sqlite3
import os

# === Constants ===
URL = "https://zuscoffee.com/category/store/kuala-lumpur-selangor/"
DB_PATH = os.path.join("data", "outlets.db")

# === Get HTML ===
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# === Extract store name and address ===
outlets = []
containers = soup.find_all("div", class_="elementor-widget-container")

for container in containers:
    name_tag = container.find("span", class_="entry-title")
    address_tag = container.find("p")

    if name_tag and address_tag:
        name = name_tag.get_text(strip=True)
        address = address_tag.get_text(strip=True)
        outlets.append((name, address, "N/A"))  # hours not available

# === Save to SQLite ===
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS outlets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        hours TEXT
    )
""")

cursor.executemany("INSERT INTO outlets (name, address, hours) VALUES (?, ?, ?)", outlets)
conn.commit()
conn.close()

print(f"✅ Scraped and saved {len(outlets)} outlets to {DB_PATH}")

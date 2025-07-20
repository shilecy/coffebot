import os
import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

BASE_URL = "https://shop.zuscoffee.com"
COLLECTION_URL = f"{BASE_URL}/collections/drinkware"

HEADERS = {
    "User-Agent": "ZUS-RAG-Scraper/1.0 (Educational project bot)"
}

def get_product_links():
    print("🔍 Sending request to collection page...")
    response = requests.get(COLLECTION_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select('a[href^="/products/"]')
    unique_links = sorted(set(BASE_URL + link['href'] for link in links))
    print(f"✅ Found {len(unique_links)} product links.")
    return unique_links

def parse_main_page_card(card):
    name_tag = card.select_one('.product-card__title a')
    price_tag = card.select_one('sale-price')
    swatch_tags = card.select('fieldset.product-card__variant-list input[type="radio"]')

    name = name_tag.get_text(strip=True) if name_tag else None
    price = price_tag.get_text(strip=True) if price_tag else None
    variations = list({swatch.get('value') for swatch in swatch_tags if swatch.get('value')})

    return name, price, variations

def parse_product_details(soup):
    details = {
        "measurements": {},
        "materials": {},
        "product_info": []
    }

    # Parse measurements and materials
    rich_text = soup.select_one('.accordion__content .metafield-rich_text_field')
    if rich_text:
        paragraphs = rich_text.find_all("p")
        for para in paragraphs:
            text = para.get_text(separator="\n", strip=True)
            if "Measurements" in text:
                section = "measurements"
            elif "Materials" in text:
                section = "materials"
            else:
                section = None

            if section:
                lines = text.split("\n")
                for line in lines:
                    if ":" in line:
                        key, value = map(str.strip, line.split(":", 1))
                        details[section][key] = value

    # Parse product info tags (BPA Free, Stainless Steel, etc.)
    info_tags = soup.select('.product_info_usp .product_info_usp-item > div:nth-of-type(2)')
    for tag in info_tags:
        info_text = tag.get_text(strip=True)
        if info_text:
            details["product_info"].append(info_text)

    return details


def scrape_all_products():
    print("=== 🛠️ ZUS Drinkware Scraper Started ===")
    product_links = get_product_links()
    products = []

    print("📦 Starting product scraping...\n")

    for link in tqdm(product_links, total=len(product_links), desc="🔎 Scraping"):
        try:
            # Fetch individual product page
            response = requests.get(link, headers=HEADERS)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Now parse name, price, variation from this page directly
            name_tag = soup.select_one('.product__title')
            price_tag = soup.select_one('.price__container .price-item')
            variation_tags = soup.select('fieldset input[type="radio"]')

            name = name_tag.get_text(strip=True) if name_tag else None
            price = price_tag.get_text(strip=True) if price_tag else None
            variations = list({tag.get('value') for tag in variation_tags if tag.get('value')})

            # Parse details
            details = parse_product_details(soup)

            products.append({
                "name": name,
                "price": price,
                "variations": variations,
                "product_info": details.get("product_info", []),
                "measurements": details.get("measurements", {}),
                "materials": details.get("materials", {}),
                "url": link
            })

            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error processing {link}: {e}")
            continue

    if products:
        os.makedirs("data", exist_ok=True)

        # Generate filename like: products_2025-07-15_20-44-31.json
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"products_{timestamp}.json"
        output_path = os.path.join("data", filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved to {output_path}")
    else:
        print("⚠️ No products saved. Check scraping output.")

if __name__ == "__main__":
    scrape_all_products()
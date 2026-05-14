import requests
from bs4 import BeautifulSoup
import json
import time
import os

def infer_test_type(name, description):
    text = (name + " " + (description or "")).lower()
    if any(word in text for word in ["personality", "opq", "motivation"]):
        return "P"
    if any(word in text for word in ["verbal", "numerical", "inductive", "ability"]):
        return "A"
    if any(word in text for word in ["situational", "sjt"]):
        return "S"
    if "biodata" in text:
        return "B"
    return "K"

def scrape_shl():
    base_url = "https://www.shl.com"
    catalog_base_url = "https://www.shl.com/products/product-catalog/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    products = []
    start = 0
    page_size = 12
    max_products = 50 # Let's get 50 to be safe
    
    while len(products) < max_products:
        catalog_url = f"{catalog_base_url}?start={start}"
        print(f"Fetching catalog page: {catalog_url}...")
        
        try:
            response = requests.get(catalog_url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching catalog page {start}: {e}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        product_rows = soup.select('div.product-catalogue__list table tr')[1:] # Skip header
        
        if not product_rows:
            print("No more products found.")
            break
            
        print(f"Found {len(product_rows)} potential products on this page.")

        for row in product_rows:
            if len(products) >= max_products:
                break
                
            try:
                link_tag = row.select_one('td.custom__table-heading__title a')
                if not link_tag:
                    continue
                
                name = link_tag.get_text(strip=True)
                url = base_url + link_tag['href']
                
                # Check if product already scraped (to avoid duplicates if any)
                if any(p['url'] == url for p in products):
                    continue

                print(f"Scraping [{len(products)+1}]: {name}")
                
                # Detail page scraping
                time.sleep(0.5) # Polite delay
                detail_res = requests.get(url, headers=headers, timeout=10)
                detail_res.raise_for_status()
                detail_soup = BeautifulSoup(detail_res.content, 'html.parser')
                
                description = ""
                job_levels = ""
                duration = ""
                
                rows = detail_soup.select('div.product-catalogue-training-calendar__row.typ')
                for r in rows:
                    h4 = r.select_one('h4')
                    if not h4:
                        continue
                    header = h4.get_text(strip=True).lower()
                    p_tag = r.select_one('p') or r.select_one('span')
                    content = p_tag.get_text(strip=True) if p_tag else r.get_text(strip=True).replace(h4.get_text(strip=True), "").strip()
                    
                    if "description" in header:
                        description = content
                    elif "job levels" in header:
                        job_levels = content.strip(',')
                    elif "assessment length" in header:
                        duration = content.split('Test Type:')[0].strip()
                
                test_type = infer_test_type(name, description)
                
                products.append({
                    "name": name,
                    "url": url,
                    "description": description,
                    "test_type": test_type,
                    "job_levels": job_levels,
                    "duration": duration
                })
                
            except Exception as e:
                print(f"Error scraping a product: {e}")
                continue
        
        start += page_size

    with open('catalog.json', 'w') as f:
        json.dump(products, f, indent=2)
    
    print(f"\nScraping complete. Total products saved: {len(products)}")

if __name__ == "__main__":
    scrape_shl()

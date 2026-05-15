import requests
from bs4 import BeautifulSoup
import json
import time
import os

# quick and dirty way to guess the test type from the name/desc
def guess_type(name, desc):
    txt = (name + " " + (desc or "")).lower()
    if any(w in txt for w in ["personality", "opq", "motivation"]):
        return "P"
    if any(w in txt for w in ["verbal", "numerical", "inductive", "ability"]):
        return "A"
    if any(w in txt for w in ["situational", "sjt"]):
        return "S"
    if "biodata" in txt:
        return "B"
    return "K" # knowledge/skills is usually the default

def scrape_it():
    site = "https://www.shl.com"
    cat_url = "https://www.shl.com/products/product-catalog/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    all_prods = []
    offset = 0
    size = 12
    # shl catalog is huge, let's grab enough to be useful
    limit = 126 
    
    while len(all_prods) < limit:
        url = f"{cat_url}?start={offset}"
        print(f"checking page: {url}")
        
        try:
            res = requests.get(url, headers=headers, timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"page {offset} blew up: {e}")
            break

        soup = BeautifulSoup(res.content, 'html.parser')
        rows = soup.select('div.product-catalogue__list table tr')[1:] # skip the table header
        
        if not rows:
            print("ran out of products")
            break

        for r in rows:
            if len(all_prods) >= limit: break
                
            try:
                a_tag = r.select_one('td.custom__table-heading__title a')
                if not a_tag: continue
                
                name = a_tag.get_text(strip=True)
                p_url = site + a_tag['href']
                
                if any(p['url'] == p_url for p in all_prods): continue

                print(f"scraping [{len(all_prods)+1}]: {name}")
                
                # polite delay so we don't get blocked
                time.sleep(0.5) 
                d_res = requests.get(p_url, headers=headers, timeout=10)
                d_res.raise_for_status()
                d_soup = BeautifulSoup(d_res.content, 'html.parser')
                
                desc, levels, dur = "", "", ""
                
                # shl nested divs are a mess, loop through the 'typ' rows
                info_rows = d_soup.select('div.product-catalogue-training-calendar__row.typ')
                for info in info_rows:
                    h4 = info.select_one('h4')
                    if not h4: continue
                    
                    lbl = h4.get_text(strip=True).lower()
                    val_tag = info.select_one('p') or info.select_one('span')
                    
                    # fallback if p/span isn't there
                    val = val_tag.get_text(strip=True) if val_tag else info.get_text(strip=True).replace(h4.get_text(strip=True), "").strip()
                    
                    if "description" in lbl:
                        desc = val
                    elif "job levels" in lbl:
                        levels = val.strip(',')
                    elif "assessment length" in lbl:
                        dur = val.split('Test Type:')[0].strip()
                
                all_prods.append({
                    "name": name,
                    "url": p_url,
                    "description": desc,
                    "test_type": guess_type(name, desc),
                    "job_levels": levels,
                    "duration": dur
                })
                
            except Exception as e:
                print(f"skipping one prod because: {e}")
                continue
        
        offset += size

    with open('catalog.json', 'w') as f:
        json.dump(all_prods, f, indent=2)
    
    print(f"\ndone. saved {len(all_prods)} assessments to catalog.json")

if __name__ == "__main__":
    scrape_it()

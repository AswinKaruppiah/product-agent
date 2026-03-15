import os
import requests
from langchain_classic.tools import Tool, StructuredTool
from datetime import datetime
import json


def search_product(product_name: str):
    try:
        print("calling API.....")

        url = "https://serpapi.com/search.json"

        params = {
            "engine": "google_shopping_light",
            "q": product_name,
            "api_key": os.getenv("SERPAPI_KEY"),
            "google_domain": "google.co.in",
            "hl": "en",
            "gl": "in",
            "location": "India",
        }

        response = requests.get(url, params=params)

        print("Status:", response.status_code)

        data = response.json()
        return data.get("shopping_results", [])

    except Exception as e:
        print("❌ Error:", e)
        return []


product_data = Tool(
    name="search_product",
    func=search_product,
    description="Search real-time product pricing data from Google Shopping",
)


def save_to_json(name: str, product_list: list, low_price_of_all: dict=None):
    print("🔥 save_to_json CALLED")

    data = {
        "name": name,
        "product_list": product_list,
        "low_price_of_all": low_price_of_all,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
    }

    with open("product_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✅ Saved to product_output.json")
    return "Saved"


save_tool = StructuredTool.from_function(
    name="save_to_json",
    func=save_to_json,
    description="Saves structured product data to a json file.",
)
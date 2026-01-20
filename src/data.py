"""
src/data.py

Utilities to download recipe datasets (Kaggle), query USDA FoodData Central,
load local samples, and perform basic preprocessing.
"""

import os
import json
import requests
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# --- Kaggle download helper ---
def download_kaggle_dataset(dataset_slug: str, file_names=None, dest_dir: str = "data"):
    """
    Download a dataset from Kaggle using kaggle API.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in env or ~/.kaggle/kaggle.json.
    dataset_slug example: "allen-institute-for-ai/CORD-19-research-challenge" or "food-com-recipes-and-user-interactions"
    file_names: optional list of file names inside dataset to extract.
    """
    try:
        from kaggle import api
    except Exception as e:
        raise RuntimeError("kaggle package not installed or not configured. See README.") from e

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # this will download dataset as a zip to the current working dir, then extract
    print(f"Downloading Kaggle dataset {dataset_slug} to {dest.resolve()}")
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
    print("Download complete. Inspect files in:", dest)
    if file_names:
        return [dest / f for f in file_names]
    # return list of files in dest
    return list(dest.iterdir())


# --- USDA FoodData Central helper ---
USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def query_usda_food(query: str, page_size=5):
    """
    Query USDA FoodData Central for food items by name.
    Returns a list of food items (dicts). Requires USDA_API_KEY in env.
    """
    if not USDA_API_KEY:
        raise RuntimeError("USDA_API_KEY not set in environment.")
    params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": page_size,
    }
    r = requests.get(USDA_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("foods", [])


# --- Loading & basic preprocessing ---
def load_local_recipes(path: str = "data/recipes_sample.csv"):
    """
    Load a local CSV of recipes. Expect columns: recipe_id,title,ingredients,calories,protein_g,carbs_g,fat_g,...
    Ingredients assumed to be a semicolon-separated string.
    """
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Drop naive duplicates
    df = df.drop_duplicates(subset=["title", "ingredients"])
    # Split ingredients into lists
    df["ingredients_list"] = df["ingredients"].fillna(" ").apply(lambda s: [i.strip() for i in s.split(";") if i.strip()])
    return df


def normalize_nutrients(df: pd.DataFrame):
    """
    Ensure nutrient columns exist and are numeric. Fill missing nutrients with NaN or 0 as appropriate.
    """
    nutrient_cols = {
        "calories": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0
    }
    for col, fill in nutrient_cols.items():
        if col not in df.columns:
            df[col] = fill
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill)
    return df


def preprocess_recipes(df: pd.DataFrame):
    """
    Basic cleaning pipeline:
     - normalize nutrients
     - tokenize ingredients
     - lowercase & deduplicate ingredients list
    """
    df = normalize_nutrients(df)
    def norm_list(lst):
        cleaned = []
        for it in lst:
            it2 = it.lower().strip()
            # basic punctuation removal
            it2 = it2.replace(",", "").replace(".", "")
            if it2 and it2 not in cleaned:
                cleaned.append(it2)
        return cleaned
    df["ingredients_list"] = df["ingredients_list"].apply(norm_list)
    # create an ingredients text field for vectorizers
    df["ingredients_text"] = df["ingredients_list"].apply(lambda lst: " ".join(lst))
    return df


def save_processed(df: pd.DataFrame, path: str = "data/recipes_processed.parquet"):
    path = Path(path)
    df.to_parquet(path, index=False)
    print(f"Saved processed data to {path.resolve()}")
    return path

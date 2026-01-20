# Data collection & preprocessing (notebook-style guide)

Open this file in Jupyter and run cells in order. The cells below are runnable Python snippets.

## 1) Setup
```python
# cell
import os
from pathlib import Path
import pandas as pd
from src.data import load_local_recipes, preprocess_recipes, save_processed, download_kaggle_dataset, query_usda_food

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
```

## 2) Quick start with included sample
```python
# cell
df = load_local_recipes("data/recipes_sample.csv")
df = preprocess_recipes(df)
df.head()
```

## 3) (Optional) Download a Kaggle recipes dataset
Follow instructions in README: set KAGGLE_USERNAME and KAGGLE_KEY or put kaggle.json in ~/.kaggle/
```python
# cell
# Example dataset slug (replace with chosen dataset)
dataset_slug = "openrecipes/recipes"  # replace with actual slug you want
# files = download_kaggle_dataset(dataset_slug, dest_dir="data/kaggle")
# Inspect files in data/kaggle/ and then load the CSV with pandas
```

## 4) (Optional) Lookup nutrient info via USDA
Sign up for API key and set USDA_API_KEY in environment.
```python
# cell
# Example search
try:
    foods = query_usda_food("raw chicken breast", page_size=3)
    for f in foods:
        print(f["description"], f.get("foodNutrients", [])[:5])
except Exception as e:
    print("USDA lookup failed (configure USDA_API_KEY?) -", e)
```

## 5) Preprocessing notes and TODOS
- Ingredient unit normalization (grams, cups, tbsp) is hard; for now we treat ingredient names as tokens and use recipe-level nutrient columns when available.
- For precise nutrition matching: map ingredients to USDA items and estimate nutrient contributions using ingredient quantities. This is future work (I'll provide heuristics later).
- Save processed dataset:
```python
# cell
save_processed(df, "data/recipes_processed.parquet")
```

## 6) Next steps
- Build content-based recommender: TF-IDF on `ingredients_text` and cosine similarity.
- Build nutrient-constraint filter to remove recipes violating allergies/diets.
- If you want, I'll implement these next.

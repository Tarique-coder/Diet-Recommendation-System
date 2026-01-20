# Diet Recommendation System

Overview
- Personalized diet/meal recommender that matches recipes/meals to user profiles (age, gender, weight, activity level), dietary preferences/restrictions, and nutritional targets.
- This repo contains data acquisition, preprocessing, baseline recommenders, a meal planner, evaluation, and a simple API demo.

Getting started
1. Create a Python venv and install dependencies:
   - python -m venv venv
   - source venv/bin/activate (Windows: venv\Scripts\activate)
   - pip install -r requirements.txt

2. Dataset options
- Option A (recommended for full features):
  - Kaggle recipes dataset (e.g., "recipes" or "recipe-ingredients-and-nutrition") — requires Kaggle API token.
  - USDA FoodData Central API key for authoritative nutrient lookups.
- Option B (quick start):
  - Use the included sample dataset at `data/recipes_sample.csv`.

3. Environment variables
- Create a `.env` with:
  - KAGGLE_USERNAME and KAGGLE_KEY (if using Kaggle API)
  - USDA_API_KEY (if using USDA FoodData Central)

4. Run the data collection & preprocessing guide
- Open `data_collection.md` in a Jupyter notebook (or copy code cells to a notebook) and run the steps to fetch and preprocess data.

What I implemented so far
- data collection utilities (src/data.py)
- notebook-style guide for dataset download & preprocessing (data_collection.md)
- tiny sample dataset to run pipeline locally (data/recipes_sample.csv)

Next steps (I will implement after your confirmation)
1. Content-based recommender (TF-IDF on ingredients + nutrient matching)
2. Collaborative / hybrid model (LightFM or implicit ALS)
3. Meal planner (linear/integer programming to meet daily macro targets)
4. Evaluation notebooks and metrics
5. FastAPI demo and Dockerfile
6. Final report and slides

Notes
- This project is not medical or dietary advice. Include a disclaimer in any deliverable or demo.
- If you want me to push this to a GitHub repo, tell me the owner/repo name (I will need that to create branches/files). 

Tell me to proceed with modeling (content-based baseline) or if you want different dataset choices.

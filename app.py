import os
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from flask import Flask, request, jsonify
import re
import string
from collections import defaultdict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import process, fuzz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SAVE_DIRECTORY = 'universal_sentence_encoder_model'

print("Loading model and data...")
tf_embedding_model = None
laptop_df = pd.DataFrame()
min_req_df = pd.DataFrame()
rec_req_df = pd.DataFrame()
min_req_kb = {}
rec_req_kb = {}
category_kb = {}
game_series_map = {}
unique_keyword_game_map = {}
tf_game_name_embeddings = None
tf_game_description_embeddings = None
laptop_list_for_nlp = []
laptop_brand_list_for_nlp = []


try:
    model_path = os.path.join(SAVE_DIRECTORY, '') 
    tf_embedding_model = tf.saved_model.load(model_path)
    print("TensorFlow Hub model loaded.")

    laptop_df_path = os.path.join(SAVE_DIRECTORY, 'laptop_data.csv')
    min_req_df_path = os.path.join(SAVE_DIRECTORY, 'min_requirements_data.csv')
    rec_req_df_path = os.path.join(SAVE_DIRECTORY, 'rec_requirements_data.csv')

    laptop_df = pd.read_csv(laptop_df_path)
    min_req_df = pd.read_csv(min_req_df_path)
    rec_req_df = pd.read_csv(rec_req_df_path)
    print("DataFrames loaded.")

    min_req_kb_path = os.path.join(SAVE_DIRECTORY, 'min_req_kb.pkl')
    rec_req_kb_path = os.path.join(SAVE_DIRECTORY, 'rec_req_kb.pkl')
    category_kb_path = os.path.join(SAVE_DIRECTORY, 'category_kb.pkl')
    game_series_map_path = os.path.join(SAVE_DIRECTORY, 'game_series_map.pkl')
    unique_keyword_game_map_path = os.path.join(SAVE_DIRECTORY, 'unique_keyword_game_map.pkl')

    with open(min_req_kb_path, 'rb') as f:
        min_req_kb = pickle.load(f)
    with open(rec_req_kb_path, 'rb') as f:
        rec_req_kb = pickle.load(f)
    with open(category_kb_path, 'rb') as f:
        category_kb = pickle.load(f)
    with open(game_series_map_path, 'rb') as f:
         game_series_map = pickle.load(f)
    with open(unique_keyword_game_map_path, 'rb') as f:
         unique_keyword_game_map = pickle.load(f)

    print("Knowledge Bases loaded.")
    print("Generating TensorFlow embeddings for game names and descriptions...")
    game_names_list = min_req_df['App'].tolist()
    game_descriptions_list = min_req_df['Description'].fillna('').tolist()

    def generate_tf_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        return tf_embedding_model(texts)

    tf_game_name_embeddings = generate_tf_embeddings(game_names_list)
    tf_game_description_embeddings = generate_tf_embeddings(game_descriptions_list)
    print("Embeddings generated.")

    laptop_list_for_nlp = laptop_df['Model'].tolist()
    laptop_brand_list_for_nlp = laptop_df['Brand'].unique().tolist()

except Exception as e:
    print(f"Error loading resources: {e}")
    tf_embedding_model = None
    laptop_df = pd.DataFrame()
    min_req_df = pd.DataFrame()
    rec_req_df = pd.DataFrame()
    min_req_kb = {}
    rec_req_kb = {}
    category_kb = {}
    game_series_map = {}
    unique_keyword_game_map = {}
    tf_game_name_embeddings = None
    tf_game_description_embeddings = None
    laptop_list_for_nlp = []
    laptop_brand_list_for_nlp = []
    print("Failed to load all necessary resources.")

def get_gpu_vendor(gpu_name):
    if pd.isna(gpu_name): return 'intel'
    gpu_name = str(gpu_name).lower()
    if any(x in gpu_name for x in ['rtx', 'gtx', 'mx']): return 'nvidia'
    elif any(x in gpu_name for x in ['radeon', 'rx', 'vega', 'pro']): return 'amd'
    elif any(x in gpu_name for x in ['integrated', 'iris', 'uhd', 'hd graphics']): return 'intel'
    else: return 'intel'

def get_cpu_vendor(cpu_name):
    if pd.isna(cpu_name): return 'intel'
    cpu_name = str(cpu_name).lower()
    if 'intel' in cpu_name or any(x in cpu_name for x in ['core', 'pentium', 'celeron', 'xeon', 'evo']): return 'intel'
    elif 'amd' in cpu_name or any(x in cpu_name for x in ['ryzen', 'fx', 'athlon', 'phenom', 'opteron']): return 'amd'
    else: return 'intel'

def get_cpu_req_score(row, req_row):
    vendor = get_cpu_vendor(row['CPU'])
    if vendor == 'intel': return req_row.get('CPU_Intel_score', 0)
    elif vendor == 'amd': return req_row.get('CPU_AMD_score', 0)
    else: return req_row.get('CPU_Intel_score', 0)

def get_gpu_req_score(row, req_row):
    vendor = get_gpu_vendor(row['GPU'])
    if pd.isna(row['GPU']): return req_row.get('GPU_Intel_score', 0) # Assume integrated if GPU is NaN
    vendor = get_gpu_vendor(row['GPU'])
    if vendor == 'nvidia': return req_row.get('GPU_NVIDIA_score', 0)
    elif vendor == 'amd': return req_row.get('GPU_AMD_score', 0)
    elif vendor == 'intel': return req_row.get('GPU_Intel_score', 0)
    else: return req_row.get('GPU_Intel_score', 0)

def get_storage_type_score(storage_type):
    if pd.isna(storage_type): return 0.0
    storage_type_lower = str(storage_type).lower()
    if 'ssd' in storage_type_lower: return 1.0
    elif 'hdd' in storage_type_lower: return 0.5
    elif 'emmc' in storage_type_lower: return 0.3
    else: return 0.0

def categorize_laptops_adapted(df_laptops_filtered, req_row_min, req_row_rec, ram_weight=1.0, storage_weight=1.0, storage_type_weight=0.5):
    if df_laptops_filtered.empty:
         print("Input DataFrame for categorization is empty.")
         return pd.DataFrame()

    ram_req = req_row_min.get('RAM', 0)
    storage_req = req_row_min.get('File Size', 0)

    def calculate_match(row):
        cpu_req = get_cpu_req_score(row, req_row_rec)
        gpu_req = get_gpu_req_score(row, req_row_rec)
        cpu_score_ratio = row['CPU_score'] / cpu_req if cpu_req > 0 and not pd.isna(row['CPU_score']) else (1.0 if cpu_req == 0 else 0.0)
        gpu_score_ratio = row['GPU_score'] / gpu_req if gpu_req > 0 and not pd.isna(row['GPU_score']) else (1.0 if gpu_req == 0 else 0.0)
        ram_score_ratio = (row['RAM'] / ram_req) * ram_weight if ram_req > 0 and not pd.isna(row['RAM']) else (1.0 if ram_req == 0 else 0.0)
        storage_score_ratio = (row['Storage'] / storage_req) * storage_weight if storage_req > 0 and not pd.isna(row['Storage']) else (1.0 if storage_req == 0 else 0.0)
        storage_type_score = get_storage_type_score(row['Storage type']) * storage_type_weight # Include storage type score
        cpu_score_ratio = min(cpu_score_ratio, 5.0)
        gpu_score_ratio = min(gpu_score_ratio, 5.0)
        ram_score_ratio = min(ram_score_ratio, 5.0)
        storage_score_ratio = min(storage_score_ratio, 5.0)
        cpu_score_ratio = 0.0 if pd.isna(cpu_score_ratio) or cpu_score_ratio == float('inf') else cpu_score_ratio
        gpu_score_ratio = 0.0 if pd.isna(gpu_score_ratio) or gpu_score_ratio == float('inf') else gpu_score_ratio
        ram_score_ratio = 0.0 if pd.isna(ram_score_ratio) or ram_score_ratio == float('inf') else ram_score_ratio
        storage_score_ratio = 0.0 if pd.isna(storage_score_ratio) or storage_score_ratio == float('inf') else storage_score_ratio
        storage_type_score = 0.0 if pd.isna(storage_type_score) or storage_type_score == float('inf') else storage_type_score
        final_score = (cpu_score_ratio + gpu_score_ratio + ram_score_ratio + storage_score_ratio + storage_type_score) / (2 + ram_weight + storage_weight + storage_type_weight) # Adjust denominator
        return final_score

    def categorize(row):
        cpu_min = get_cpu_req_score(row, req_row_min)
        gpu_min = get_gpu_req_score(row, req_row_min)
        cpu_rec = get_cpu_req_score(row, req_row_rec)
        gpu_rec = get_gpu_req_score(row, req_row_rec)
        row_cpu_score = row['CPU_score'] if not pd.isna(row['CPU_score']) else -1
        row_gpu_score = row['GPU_score'] if not pd.isna(row['GPU_score']) else -1
        row_ram = row['RAM'] if not pd.isna(row['RAM']) else -1
        row_storage = row['Storage'] if not pd.isna(row['Storage']) else -1
        cpu_min_req = req_row_min.get('CPU_Intel_score', 0) 
        gpu_min_req = req_row_min.get('GPU_NVIDIA_score', 0) 
        cpu_rec_req = req_row_rec.get('CPU_Intel_score', 0) 
        gpu_rec_req = req_row_rec.get('GPU_NVIDIA_score', 0) 
        ram_min_req = req_row_min.get('RAM', 0)
        storage_min_req = req_row_min.get('File Size', 0)
        cpu_min_check = row_cpu_score >= cpu_min_req if cpu_min_req > 0 else True
        gpu_min_check = row_gpu_score >= gpu_min_req if gpu_min_req > 0 else True
        ram_min_check = row_ram >= ram_min_req if ram_min_req > 0 else True
        storage_min_check = row_storage >= storage_min_req if storage_min_req > 0 else True

        if not (cpu_min_check and gpu_min_check and ram_min_check and storage_min_check):
             return 'Disqualified'

        cpu_rec_flag = row_cpu_score >= cpu_rec_req if cpu_rec_req > 0 else True
        gpu_rec_flag = row_gpu_score >= gpu_rec_req if gpu_rec_req > 0 else True
        if cpu_rec_flag and gpu_rec_flag: return 'Recommended'
        elif cpu_rec_flag or gpu_rec_flag: return 'Mixed'
        else: return 'Minimum'

    df_categorized = df_laptops_filtered.copy()

    if df_categorized.empty:
        print("DataFrame is empty after initial filtering (if any).")
        return pd.DataFrame()

    df_categorized['Match_Score'] = df_categorized.apply(calculate_match, axis=1)
    df_categorized['Category'] = df_categorized.apply(categorize, axis=1)
    df_final = df_categorized[df_categorized['Category'] != 'Disqualified'].copy()
    return df_final[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price', 'Category', 'Match_Score']] # Added Storage type

tokenizer = RegexpTokenizer(r'\w+')
factory = StopWordRemoverFactory()
stopwords_sastrawi = set(factory.get_stop_words())

additional_stopwords = {
    'cocok', 'buat', 'main', 'dengan', 'rekomendasi', 'spesifikasi', 'spek',
    'apa', 'yang', 'laptop', 'harga', 'rp', 'ribu', 'juta', 'budget',
    'model', 'merek', 'brand', 'merk', 'duit', 'uang', 'dana', 'termurah'
}
stopwords_sastrawi.update(additional_stopwords)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords_sastrawi]

def basic_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[’‘]', "'", text)  # Normalisasi apostrof
    text = re.sub(f"[{string.punctuation}]", " ", text)
    tokens = tokenizer.tokenize(text)
    return tokens

def create_keyword_game_map(game_list, stopwords):
    keyword_candidate_map = defaultdict(list)
    for game in game_list:
        tokens = basic_preprocessing(game)
        tokens = remove_stopwords(tokens)
        for token in tokens:
            if len(token) > 2 or token.isdigit():
                keyword_candidate_map[token].append(game)

    unique_keyword_map = {}
    for keyword, games in keyword_candidate_map.items():
        if len(games) == 1:
            unique_keyword_map[keyword] = games[0]

    return unique_keyword_map

def handle_game_series(found_games, query_lower, game_series_map):
    found_games = set(found_games)
    series_to_add = set()

    def extract_number(game_name):
        numbers = re.findall(r'\d+', game_name)
        return int(numbers[-1]) if numbers else 0

    for series_name, games_in_series in game_series_map.items():
        if re.search(r'\b' + re.escape(series_name.lower()) + r'\b', query_lower):
            specific_found = any(game in found_games for game in games_in_series)

            if not specific_found and games_in_series:
                sorted_games = sorted(games_in_series, key=extract_number, reverse=True)
                series_to_add.add(sorted_games[0])

    found_games.update(series_to_add)
    return list(found_games)

def extract_entities_and_budget(user_query, game_list, laptop_list, laptop_brand_list,
                               unique_keyword_game_map, game_series_map,
                               score_cutoff_high=95, score_cutoff_low_game=90,
                               score_cutoff_low_laptop=80):
    found_games = set()
    found_laptops = set()
    extracted_budget = None
    query_lower = user_query.lower()
    query_tokens = basic_preprocessing(user_query)
    query_tokens_cleaned = remove_stopwords(query_tokens)
    budget_pattern = r'(?:harga|rp|rp\.|budget)\s*:?\s*(\d+(?:[,\.]\d+)?)\s*(rb|ribu|jt|juta)?|\b(\d+(?:[,\.]\d+)?)\s*(rb|ribu|jt|juta)\b'
    budget_match = re.search(budget_pattern, query_lower)
    budget_tokens = set()

    if budget_match:
        amount1 = budget_match.group(1)
        unit1 = budget_match.group(2)
        amount2 = budget_match.group(3)
        unit2 = budget_match.group(4)
        amount_str = amount1 if amount1 else amount2
        unit = unit1 if unit1 else unit2

        if amount_str:
            try:
                amount = float(amount_str.replace(',', '.'))
                if unit in ['juta', 'jt']:
                    extracted_budget = int(amount * 1_000_000)
                elif unit in ['ribu', 'rb']:
                    extracted_budget = int(amount * 1_000)
                else:
                    extracted_budget = int(amount)
                budget_tokens.add(amount_str.replace(',', '.'))
                budget_tokens.add(amount_str.replace('.', ','))

            except ValueError:
                extracted_budget = None

    game_matching_tokens = [t for t in query_tokens_cleaned if t not in budget_tokens]
    cleaned_query_string_for_game = " ".join(game_matching_tokens)
    query_numbers = set(t for t in game_matching_tokens if t.isdigit())
    laptop_entities = set(laptop_list + laptop_brand_list)

    for entity in laptop_entities:
        if re.search(r'\b' + re.escape(entity.lower()) + r'\b', query_lower):
            found_laptops.add(entity)

    if not found_laptops:
        for entity in laptop_entities:
            match = process.extractOne(
                entity,
                [query_lower],
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff_high
            )
            if match:
                found_laptops.add(entity)

    laptop_tokens = set()
    for laptop in found_laptops:
        tokens = basic_preprocessing(laptop)
        tokens = remove_stopwords(tokens)
        laptop_tokens.update(tokens)

    game_matching_tokens = [
        t for t in query_tokens_cleaned
        if t not in budget_tokens and t not in laptop_tokens
    ]

    cleaned_query_string_for_game = " ".join(game_matching_tokens)
    query_numbers = set(t for t in game_matching_tokens if t.isdigit())

    matches_high = process.extractBests(
        query_lower,
        game_list,
        scorer=fuzz.token_set_ratio,
        score_cutoff=95,
        limit=5
    )
    for match, score in matches_high:
        found_games.add(match)

    for token in game_matching_tokens:
        if token in unique_keyword_game_map:
            game_from_keyword = unique_keyword_game_map[token]
            found_games.add(game_from_keyword)

    if not found_games:
        matches_low = process.extract(
            cleaned_query_string_for_game,
            game_list,
            limit=10
        )
        for match, score in matches_low:
            if match in found_games:
                continue

            if score < score_cutoff_low_game:
                continue

            match_tokens = set(remove_stopwords(basic_preprocessing(match)))
            query_token_set = set(game_matching_tokens)
            overlap = match_tokens & query_token_set
            match_numbers = set(re.findall(r'\d+', match))
            if query_numbers and match_numbers:
                if not (query_numbers & match_numbers):
                    continue

            elif match_numbers and not query_numbers:
                continue

            min_required = max(1, min(2, len(match_tokens)//2))
            if len(overlap) >= min_required:
                found_games.add(match)

    found_games = handle_game_series(found_games, query_lower, game_series_map)

    if query_numbers:
        filtered_games = []
        for game in found_games:
            game_numbers = set(re.findall(r'\d+', game))
            if game_numbers:
                if not (query_numbers & game_numbers):
                    continue
            filtered_games.append(game)
        found_games = filtered_games

    laptop_entities = set(laptop_list + laptop_brand_list)

    for entity in laptop_entities:
        if re.search(r'\b' + re.escape(entity.lower()) + r'\b', query_lower):
            found_laptops.add(entity)

    if not found_laptops:
        for entity in laptop_entities:
            match = process.extractOne(
                entity,
                [cleaned_query_string_for_game],
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff_high
            )
            if match:
                found_laptops.add(entity)

    return list(found_games), list(found_laptops), extracted_budget

def nlp_pipeline_fuzzy(user_query, game_list, laptop_list, laptop_brand_list,
                       unique_keyword_game_map, game_series_map):
    tokens = basic_preprocessing(user_query)
    tokens = remove_stopwords(tokens)
    found_games, found_laptops, extracted_budget = extract_entities_and_budget(
        user_query, game_list, laptop_list, laptop_brand_list,
        unique_keyword_game_map, game_series_map,
        score_cutoff_low_game=90
    )
    return {
        "tokens": tokens,
        "found_games": found_games,
        "found_laptops": found_laptops,
        "budget": extracted_budget
    }

def recognize_intent_simple(query, found_games, found_laptops, budget):
    query_lower = query.lower()
    query_tokens = set(remove_stopwords(basic_preprocessing(query_lower)))
    intent = "FIND_LAPTOP_FOR_GAME"

    if "terbaik" in query_tokens or "tertinggi" in query_tokens or "high performance" in query_lower:
        intent = "FIND_BEST_LAPTOP_FOR_GAME"
    elif "termurah" in query_tokens or "paling murah" in query_lower or "low budget" in query_lower:
        intent = "FIND_CHEAPEST_LAPTOP_FOR_GAME"
    elif "termahal" in query_tokens or "paling mahal" in query_lower:
        intent = "FIND_MOST_EXPENSIVE_LAPTOP"
    elif "bandingkan" in query_tokens or "vs" in query_tokens:
         intent = "COMPARE_LAPTOPS"
    elif (found_laptops or budget is not None) and not found_games:
         intent = "FILTER_LAPTOPS"

    if found_games and intent in ["FIND_LAPTOP_FOR_GAME", "FILTER_LAPTOPS"]:
        intent = "FIND_LAPTOP_FOR_GAME"
    if not found_games and not found_laptops and budget is None:
        intent = "GENERAL_QUERY"

    return intent

def search_games_tf(query, tf_embedding_model, tf_game_name_embeddings, tf_game_description_embeddings, df, top_n=5, category_kb=None, category_boost=0.5):
    if tf_embedding_model is None or tf_game_name_embeddings is None or tf_game_description_embeddings is None:
        return []

    query_embedding = generate_tf_embeddings(query)

    try:
        query_embedding_np = query_embedding.numpy()
        tf_game_name_embeddings_np = tf_game_name_embeddings.numpy()
        tf_game_description_embeddings_np = tf_game_description_embeddings.numpy()
    except AttributeError:
        return [] 

    name_similarities = cosine_similarity(query_embedding_np, tf_game_name_embeddings_np)[0]
    description_similarities = cosine_similarity(query_embedding_np, tf_game_description_embeddings_np)[0]
    combined_similarities = (name_similarities + description_similarities) / 2.0
    query_tokens_cleaned = basic_preprocessing(query)
    query_tokens_cleaned = remove_stopwords(query_tokens_cleaned)

    boosted_similarities = combined_similarities.copy()
    for i in range(len(df)):
        game_categories_str = df.iloc[i]["Category"]
        game_categories = []
        if isinstance(game_categories_str, str):
             try:
                 game_categories = eval(game_categories_str)
             except (SyntaxError, NameError):
                  pass
        elif isinstance(game_categories_str, list):
             game_categories = game_categories_str

        if game_categories:
            game_categories_lower = set([cat.lower().strip() for cat in game_categories])
            if any(token in game_categories_lower for token in query_tokens_cleaned):
                boosted_similarities[i] += category_boost 

    top_indices = boosted_similarities.argsort(axis=0)[-top_n:][::-1] 

    results = []
    for idx in top_indices:
        idx_int = int(idx)
        results.append({
            "game": df.iloc[idx_int]["App"],
            "category": df.iloc[idx_int]["Category"],
            "similarity": boosted_similarities[idx_int].item() if hasattr(boosted_similarities[idx_int], 'item') else boosted_similarities[idx_int] # Handle numpy or tensor scalar
        })
    return results

def get_laptop_recommendations_with_intent(user_query, laptop_df, min_req_df, rec_req_df,
                               min_req_kb, rec_req_kb,
                               tf_embedding_model, tf_game_name_embeddings, tf_game_description_embeddings,
                               laptop_list, laptop_brand_list, unique_keyword_game_map, game_series_map,
                               category_boost_value=1.0,
                               top_n_semantic_games=10
                               ):

    if 'App' not in min_req_df.columns: 
        return pd.DataFrame({"Status": ["Error"], "Pesan": ["Kolom 'App' tidak ditemukan di min_req_df."]})

    pipeline_result = nlp_pipeline_fuzzy(user_query, min_req_df['App'].tolist(), laptop_df['Model'].tolist(), laptop_df['Brand'].unique().tolist(), unique_keyword_game_map, game_series_map)
    found_games = pipeline_result['found_games']
    found_laptops_entities = pipeline_result['found_laptops']
    extracted_budget = pipeline_result['budget']
    detected_intent = recognize_intent_simple(user_query, found_games, found_laptops_entities, extracted_budget)

    if detected_intent == "COMPARE_LAPTOPS":
        if len(found_laptops_entities) < 2:
             return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan minimal dua nama laptop atau brand untuk dibandingkan."]})

        print("\nIntent: Compare Laptops. Implementasi perbandingan spesifikasi laptop...")
        comparison_laptops = laptop_df[
            laptop_df['Model'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Model'].tolist()]) |
            laptop_df['Brand'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Brand'].unique().tolist()])
        ].copy()
        if comparison_laptops.empty:
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan yang ingin Anda bandingkan."]})

        return comparison_laptops[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Screen', 'Final Price']]

    elif detected_intent == "FILTER_LAPTOPS":
         if not found_laptops_entities and extracted_budget is None:
              return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan brand laptop, model, atau budget yang Anda inginkan."]})

         laptops_to_evaluate = laptop_df.copy()

         # Filter Budget
         if extracted_budget is not None and extracted_budget > 0:
             laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()

         if found_laptops_entities:
             filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
             for entity in found_laptops_entities:
                 if entity in laptop_brand_list:
                     filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                 elif entity in laptop_list:
                      filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                 else:
                     brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                     model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                     if brand_match and brand_match[1] >= 90:
                          filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                     if model_match and model_match[1] >= 90:
                          filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())

             laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()

             if laptops_to_evaluate.empty:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan kriteria filter Anda."]})


         if laptops_to_evaluate.empty:
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang ditemukan berdasarkan kriteria filter Anda."]})

         return laptops_to_evaluate[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price']]

    elif detected_intent == "FIND_MOST_EXPENSIVE_LAPTOP":
        print("\nIntent: Find Most Expensive Laptop. Mencari laptop termahal...")
        laptops_to_evaluate = laptop_df.copy()

        if found_laptops_entities:
            filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
            for entity in found_laptops_entities:
                if entity in laptop_brand_list:
                    filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                elif entity in laptop_list:
                    filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                else:
                    brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                    model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                    if brand_match and brand_match[1] >= 90:
                        filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                    if model_match and model_match[1] >= 90:
                        filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())

            laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()
            if laptops_to_evaluate.empty:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop yang cocok dengan entitas '{', '.join(found_laptops_entities)}'."]})

        if found_games:
            print(f"  - Game terdeteksi: {found_games}. Akan mencari laptop termahal yang memenuhi min. req.")
            target_game_min_req = None
            max_requirement_score = -1
            target_game_name = None

            for game_name in found_games:
                 min_req = min_req_kb.get(game_name)
                 if min_req:
                     current_requirement_score = (min_req.get('CPU_Intel_score', 0) + min_req.get('CPU_AMD_score', 0) +
                                                  min_req.get('GPU_NVIDIA_score', 0) + min_req.get('GPU_AMD_score', 0) + min_req.get('GPU_Intel_score', 0))
                     if current_requirement_score > max_requirement_score:
                         max_requirement_score = current_requirement_score
                         target_game_min_req = min_req
                         target_game_name = game_name

            if target_game_min_req:
                 ram_req = target_game_min_req.get('RAM', 0) 
                 storage_req = target_game_min_req.get('File Size', 0) 

                 laptops_to_evaluate = laptops_to_evaluate[
                     (laptops_to_evaluate['RAM'] >= ram_req) &
                     (laptops_to_evaluate['Storage'] >= storage_req)
                 ].copy()

                 def meets_min_req(row, req_row_min):
                     cpu_min_score_req = req_row_min.get('CPU_Intel_score', 0) 
                     gpu_min_score_req = req_row_min.get('GPU_NVIDIA_score', 0) 
                     laptop_cpu_score = row['CPU_score'] if not pd.isna(row['CPU_score']) else -1
                     laptop_gpu_score = row['GPU_score'] if not pd.isna(row['GPU_score']) else -1
                     cpu_meets = laptop_cpu_score >= get_cpu_req_score(row, req_row_min) if get_cpu_req_score(row, req_row_min) > 0 else True
                     gpu_meets = laptop_gpu_score >= get_gpu_req_score(row, req_row_min) if get_gpu_req_score(row, req_row_min) > 0 else True

                     return cpu_meets and gpu_meets

                 laptops_to_evaluate = laptops_to_evaluate[
                     laptops_to_evaluate.apply(meets_min_req, axis=1, req_row_min=target_game_min_req)
                 ].copy()

                 if laptops_to_evaluate.empty:
                     return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop yang memenuhi persyaratan minimum untuk '{target_game_name}' dalam kriteria Anda."]})

            else:
                print("Tidak dapat menemukan persyaratan minimum untuk game yang terdeteksi.")


        if laptops_to_evaluate.empty:
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang ditemukan berdasarkan kriteria Anda."]})

        most_expensive_laptops = laptops_to_evaluate.sort_values(by='Final Price', ascending=False).reset_index(drop=True)
        return most_expensive_laptops[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price']]


    elif detected_intent in ["FIND_BEST_LAPTOP_FOR_GAME", "FIND_CHEAPEST_LAPTOP_FOR_GAME", "FIND_LAPTOP_FOR_GAME"]:
        game_targets = []
        if not found_games:
            semantic_results = search_games_tf(user_query, tf_embedding_model, tf_game_name_embeddings, tf_game_description_embeddings, min_req_df, top_n=top_n_semantic_games, category_boost=category_boost_value)
            game_targets = [res['game'] for res in semantic_results if res['similarity'] > 0.5]
            if game_targets:
                 print(f"  - Menemukan game relevan dari Semantic Search: {game_targets}")
            else:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan game relevan dari query Anda."]})

        else:
             game_targets = found_games

        target_game_min_req = None
        target_game_rec_req = None
        target_game_name = None
        max_requirement_score = -1

        if game_targets:
            valid_game_targets = []
            for game_name in game_targets:
                min_req = min_req_kb.get(game_name)
                rec_req = rec_req_kb.get(game_name)

                if min_req and rec_req:
                    valid_game_targets.append(game_name)
                    current_requirement_score = (rec_req.get('CPU_Intel_score', 0) + rec_req.get('CPU_AMD_score', 0) +
                                                 rec_req.get('GPU_NVIDIA_score', 0) + rec_req.get('GPU_AMD_score', 0) + rec_req.get('GPU_Intel_score', 0))
                    if current_requirement_score == 0:
                         current_requirement_score = (min_req.get('CPU_Intel_score', 0) + min_req.get('CPU_AMD_score', 0) +
                                                      min_req.get('GPU_NVIDIA_score', 0) + min_req.get('GPU_AMD_score', 0) + min_req.get('GPU_Intel_score', 0))

                    if current_requirement_score > max_requirement_score:
                        max_requirement_score = current_requirement_score
                        target_game_name = game_name
                        target_game_min_req = min_req
                        target_game_rec_req = rec_req

            if not target_game_name:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan persyaratan yang valid untuk game target dari knowledge base."]})
            game_targets = valid_game_targets

        else:
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada game relevan yang ditemukan dari query Anda."]})

        laptops_to_evaluate = laptop_df.copy()

        if extracted_budget is not None and extracted_budget > 0:
            laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()
            if laptops_to_evaluate.empty:
                return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop dalam budget {extracted_budget:,} yang tersedia."]})

        if found_laptops_entities:
            filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
            for entity in found_laptops_entities:
                if entity in laptop_brand_list:
                    filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                elif entity in laptop_list:
                     filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                else:
                    brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                    model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                    if brand_match and brand_match[1] >= 90:
                         filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                    if model_match and model_match[1] >= 90:
                         filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())

            laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()

            if laptops_to_evaluate.empty:
                return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop yang cocok dengan '{', '.join(found_laptops_entities)}' dalam kriteria Anda."]})

        final_recommendations = categorize_laptops_adapted(
            laptops_to_evaluate,
            target_game_min_req,
            target_game_rec_req
        )

        if final_recommendations.empty:
            return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada rekomendasi laptop yang memenuhi persyaratan minimum game yang relevan dalam kriteria Anda."]})

        else:
            if detected_intent == "FIND_BEST_LAPTOP_FOR_GAME":
                final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                )
                final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')
            elif detected_intent == "FIND_CHEAPEST_LAPTOP_FOR_GAME":
                 final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                 )
                 final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Final Price', 'Match_Score'], ascending=[True, True, False]).drop(columns='Category_Order') # Sort by price then match score
            else: 
                 final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                 )
                 final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')

            return final_recommendations

    elif detected_intent == "GENERAL_QUERY":
        semantic_results = search_games_tf(user_query, tf_embedding_model, tf_game_name_embeddings, tf_game_description_embeddings, min_req_df, top_n=top_n_semantic_games, category_boost=category_boost_value)
        relevant_apps = [res['game'] for res in semantic_results if res['similarity'] > 0.5] 

        if relevant_apps:
            target_app_name = relevant_apps[0]
            min_req = min_req_kb.get(target_app_name)
            rec_req = rec_req_kb.get(target_app_name)

            if min_req and rec_req:
                 requirements_message = f"Berikut persyaratan untuk '{target_app_name}':\n\n  - Minimum: CPU={min_req.get('CPU_Intel', 'N/A')}/{min_req.get('CPU_AMD', 'N/A')}, GPU={min_req.get('GPU_NVIDIA', 'N/A')}/{min_req.get('GPU_AMD', 'N/A')}/{min_req.get('GPU_Intel', 'N/A')}, RAM: {min_req.get('RAM', 'N/A')} GB, Storage: {min_req.get('File Size', 'N/A')} GB\n  - Rekomendasi: CPU={rec_req.get('CPU_Intel', 'N/A')}/{rec_req.get('CPU_AMD', 'N/A')}, GPU={rec_req.get('GPU_NVIDIA', 'N/A')}/{rec_req.get('GPU_AMD', 'N/A')}/{rec_req.get('GPU_Intel', 'N/A')}, RAM: {rec_req.get('RAM', 'N/A')} GB, Storage: {rec_req.get('File Size', 'N/A')} GB\n"
                 recommended_laptops = categorize_laptops_adapted(
                     laptop_df.copy(),
                     min_req,
                     rec_req
                 )

                 if not recommended_laptops.empty:
                      top_recommended = recommended_laptops[recommended_laptops['Category'] == 'Recommended'].head(5).copy() 
                      if not top_recommended.empty:
                           return top_recommended[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price', 'Category', 'Match_Score']] 
                      else:
                           return pd.DataFrame({"Status": ["Informasi"], "Pesan": [requirements_message + "\nTidak ada laptop dalam dataset yang memenuhi persyaratan rekomendasi."]})
                 else:
                     return pd.DataFrame({"Status": ["Informasi"], "Pesan": [requirements_message + "\nTidak ada laptop dalam dataset yang memenuhi persyaratan rekomendasi."]})

            else:
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak dapat menemukan persyaratan untuk aplikasi '{target_app_name}'."]})

        else:
            return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan game/aplikasi relevan dari query Anda."]})

    else:
        return pd.DataFrame({"Status": ["Intent Tidak Dikenali"], "Pesan": ["Mohon maaf, niat Anda belum dapat saya proses saat ini."]})

app = Flask(__name__)

@app.route('/')
def home():
    return "Laptop Recommendation API is running!"

@app.route('/recommend', methods=['GET', 'POST']) 
def recommend():
    if tf_embedding_model is None or laptop_df.empty or min_req_df.empty or rec_req_df.empty:
        return jsonify({"status": "Error", "message": "Resources not loaded. Cannot provide recommendations. Check server logs for details."}), 500

    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query')
    else: 
        query = request.args.get('query') 

    if not query:
        return jsonify({"status": "Error", "message": "No query provided."}), 400

    print(f"Received query: {query}")

    try:
        result_df = get_laptop_recommendations_with_intent( 
            query,
            laptop_df,
            min_req_df,
            rec_req_df,
            min_req_kb,
            rec_req_kb,
            tf_embedding_model, 
            tf_game_name_embeddings, 
            tf_game_description_embeddings, 
            laptop_list_for_nlp, 
            laptop_brand_list_for_nlp, 
            unique_keyword_game_map, 
            game_series_map 
        )

        if 'Status' in result_df.columns:
             response_data = result_df.to_dict(orient='records')
             status_code = 404 if response_data and response_data[0].get('Status') in ['Tidak Ditemukan', 'Intent Tidak Dikenali', 'Informasi Kurang'] else 200
             return jsonify(response_data), status_code
        else:
            return jsonify(result_df.to_dict(orient='records')), 200


    except Exception as e:
        print(f"Error during recommendation process: {e}")
        return jsonify({"status": "Error", "message": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, use_reloader=False, port=5000)
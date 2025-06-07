from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from fuzzywuzzy import process, fuzz
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Load datasets
print("Loading datasets...")
laptop_df = pd.read_csv("cleaned_dataset_v4.csv")
min_req_df = pd.read_csv("minimum_clean.csv")
rec_req_df = pd.read_csv("recommended_clean.csv")

# Precompute embeddings
print("Precomputing embeddings...")
model_sentence = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
descriptions = min_req_df['Description'].fillna('').tolist()
description_embeddings = model_sentence.encode(descriptions, convert_to_tensor=True)

# Build knowledge base
min_req_kb = min_req_df.set_index('App').to_dict('index')
rec_req_kb = rec_req_df.set_index('App').to_dict('index')

# Prepare lists
app_list = min_req_df['App'].tolist()
laptop_list = laptop_df['Model'].tolist()
laptop_brand_list = laptop_df['Brand'].unique().tolist()

# Initialize NLP components
print("Initializing NLP pipeline components...")
tokenizer_nlp = RegexpTokenizer(r'\w+')
factory = StopWordRemoverFactory()
stopwords_sastrawi = set(factory.get_stop_words())

additional_stopwords = {
    'cocok', 'buat', 'kerja', 'dengan', 'rekomendasi', 'spesifikasi', 'spek',
    'apa', 'yang', 'laptop', 'harga', 'rp', 'ribu', 'juta', 'budget',
    'model', 'merek', 'brand', 'merk', 'duit', 'uang', 'dana'
}
stopwords_sastrawi.update(additional_stopwords)

SYNONYM_MAPPING = {
    # ... [isi dengan mapping synonym dari notebook] ...
}

# NLP Functions
def basic_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[’‘]', "'", text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = tokenizer_nlp.tokenize(text)
    return tokens

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords_sastrawi]

def map_synonyms(tokens):
    return [SYNONYM_MAPPING.get(t, t) for t in tokens]

# === App Series Handling ===
def handle_app_series(found_apps, query_lower, app_series_map):
    found_apps = set(found_apps)
    series_to_add = set()

    # Fungsi untuk ekstrak angka dari judul app (ambil angka terakhir)
    def extract_number(app_name):
        numbers = re.findall(r'\d+', app_name)
        return int(numbers[-1]) if numbers else 0

    for series_name, apps_in_series in app_series_map.items():
        # Cek apakah nama seri disebut di query (dengan regex word boundary)
        if re.search(r'\b' + re.escape(series_name.lower()) + r'\b', query_lower):
            # Cek apakah sudah ada app spesifik dari seri ini
            specific_found = any(app in found_apps for app in apps_in_series)

            # Jika belum ada spesifik, tambahkan app terbaru dari seri
            if not specific_found and apps_in_series:
                # Urutkan app berdasarkan versi (descending)
                sorted_apps = sorted(apps_in_series, key=extract_number, reverse=True)
                series_to_add.add(sorted_apps[0])

    found_apps.update(series_to_add)
    return list(found_apps)

def create_keyword_app_map(app_list, stopwords):
    keyword_candidate_map = defaultdict(list)
    for app in app_list:
        tokens = basic_preprocessing(app)
        tokens = remove_stopwords(tokens)
        for token in tokens:
            if len(token) > 2 or token.isdigit():
                keyword_candidate_map[token].append(app)

    unique_keyword_map = {}
    for keyword, apps in keyword_candidate_map.items():
        if len(apps) == 1:
            unique_keyword_map[keyword] = apps[0]

    return unique_keyword_map

unique_keyword_app_map = create_keyword_app_map(app_list, stopwords_sastrawi)

def create_app_series_map(app_list):
    series_map = defaultdict(set)
    patterns_to_remove = [
        r'\s*\d+$',
        r'\s*\(?\d{4}\)?$',
        r'\s*(?:hd|remastered|remake|definitive edition|enhanced edition|anniversary edition)\s*$',
        r'\s*:\s*.*$',
        r'\s*-+\s*.*$',
    ]

    for app in app_list:
        series_name = app
        for pattern in patterns_to_remove:
            series_name = re.sub(pattern, '', series_name, flags=re.IGNORECASE).strip()

        if series_name and series_name != app:
            series_map[series_name.lower()].add(app)

    filtered_series_map = {
        series: list(apps) for series, apps in series_map.items() if len(apps) > 1
    }
    return filtered_series_map

app_series_map = create_app_series_map(app_list)

def extract_entities_and_budget(user_query, app_list, laptop_list, laptop_brand_list,
                               unique_keyword_app_map, app_series_map,
                               score_cutoff_high=95, score_cutoff_low_app=90,
                               score_cutoff_low_laptop=80):
    # ... [implementasi lengkap dari notebook] ...
    found_apps = set()
    found_laptops = set()
    extracted_budget = None

    query_lower = user_query.lower()
    query_tokens = basic_preprocessing(user_query)
    query_tokens_cleaned = remove_stopwords(query_tokens)

    # Ekstrak budget
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
                # Replace comma with dot for float conversion
                amount = float(amount_str.replace(',', '.'))
                if unit in ['juta', 'jt']:
                    extracted_budget = int(amount * 1_000_000)
                elif unit in ['ribu', 'rb']:
                    extracted_budget = int(amount * 1_000)
                else:
                    extracted_budget = int(amount)
                # Add both original and normalized amount string to budget_tokens
                budget_tokens.add(amount_str.replace(',', '.'))
                budget_tokens.add(amount_str.replace('.', ','))

            except ValueError:
                extracted_budget = None

    # Hapus token budget dari pemrosesan app
    app_matching_tokens = [t for t in query_tokens_cleaned if t not in budget_tokens]
    cleaned_query_string_for_app = " ".join(app_matching_tokens)

    # Angka yang ada di query untuk app (pastikan tidak termasuk angka budget)
    query_numbers = set(t for t in app_matching_tokens if t.isdigit())

    # 1. High threshold fuzzy matching dengan token_set_ratio
    matches_high = process.extractBests(
        query_lower,
        app_list,
        scorer=fuzz.token_set_ratio,
        score_cutoff=95,
        limit=5
    )
    for match, score in matches_high:
        found_apps.add(match)

    # 2. Unique Keyword Matching for Apps
    for token in app_matching_tokens:
        if token in unique_keyword_app_map:
            app_from_keyword = unique_keyword_app_map[token]
            found_apps.add(app_from_keyword)

    # 3. Tightened Fuzzy Matching for Apps
    if not found_apps:
        matches_low = process.extract(
            cleaned_query_string_for_app,
            app_list,
            limit=10
        )
        for match, score in matches_low:
            if match in found_apps:
                continue

            # Filter berdasarkan score
            if score < score_cutoff_low_app:
                continue

            # Hitung overlap token
            match_tokens = set(remove_stopwords(basic_preprocessing(match)))
            query_token_set = set(app_matching_tokens)
            overlap = match_tokens & query_token_set

            # Angka di app
            match_numbers = set(re.findall(r'\d+', match))

            # Jika query menyebut angka, app harus mengandung angka yang sama
            if query_numbers and match_numbers:
                if not (query_numbers & match_numbers):
                    continue

            # Jika app mengandung angka, tapi query tidak menyebut angka spesifik
            elif match_numbers and not query_numbers:
                continue

            # Minimal 50% token match atau minimal 2 token cocok
            min_required = max(1, min(2, len(match_tokens)//2))
            if len(overlap) >= min_required:
                found_apps.add(match)

    # 4. Handle App Series
    found_apps = handle_app_series(found_apps, query_lower, app_series_map)

    # 5. Filter app berdasarkan angka query
    if query_numbers:
        filtered_apps = []
        for app in found_apps:
            app_numbers = set(re.findall(r'\d+', app))
            if app_numbers:
                if not (query_numbers & app_numbers):
                    continue
            filtered_apps.append(app)
        found_apps = filtered_apps

    # Laptop Matching
    laptop_entities = set(laptop_list + laptop_brand_list)

    # Prioritas 1: Exact matching
    for entity in laptop_entities:
        if re.search(r'\b' + re.escape(entity.lower()) + r'\b', query_lower):
            found_laptops.add(entity)

    # Prioritas 2: Fuzzy matching (only if no exact matches found for laptops)
    if not found_laptops:
        for entity in laptop_entities:
            # Use the cleaned query string for app matching, excluding budget tokens
            match = process.extractOne(
                entity,
                [cleaned_query_string_for_app],
                scorer=fuzz.token_set_ratio,
                score_cutoff=score_cutoff_high
            )
            if match:
                found_laptops.add(entity)

    return list(found_apps), list(found_laptops), extracted_budget

def nlp_pipeline_fuzzy(user_query, app_list, laptop_list, laptop_brand_list,
                       unique_keyword_app_map, app_series_map):
    # ... [implementasi lengkap dari notebook] ...
    tokens = basic_preprocessing(user_query)
    tokens = remove_stopwords(tokens)
    tokens = map_synonyms(tokens)
    found_apps, found_laptops, extracted_budget = extract_entities_and_budget(
        user_query, app_list, laptop_list, laptop_brand_list,
        unique_keyword_app_map, app_series_map,
        score_cutoff_low_app=90
    )
    return {
        "tokens": tokens,
        "found_apps": found_apps,
        "found_laptops": found_laptops,
        "budget": extracted_budget
    }

def recognize_intent_simple(user_query, found_apps, found_laptops, budget):
    # ... [implementasi lengkap dari notebook] ...
    query_lower = user_query.lower()
    query_tokens = set(basic_preprocessing(query_lower))

    intent = "FIND_LAPTOP_FOR_APP"

    # Check for specific keywords indicating intent
    if "terbaik" in query_tokens or "tertinggi" in query_tokens or "high performance" in query_lower:
        # Prioritize recommended specs or highest match score
        intent = "FIND_BEST_LAPTOP_FOR_APP"
    elif "termurah" in query_tokens or "paling murah" in query_lower or "low budget" in query_lower:
        # Prioritize minimum specs or lowest price within category
        intent = "FIND_CHEAPEST_LAPTOP_FOR_APP"
    elif "bandingkan" in query_tokens or "compare" in query_tokens:
        # Needs at least two laptop entities to be meaningful
        if len(found_laptops) >= 2:
            intent = "COMPARE_LAPTOPS"
        else:
            # If compare keyword is present but not enough laptops, fallback or ask for clarification
            intent = "UNKNOWN"
    elif (found_laptops or budget is not None) and not found_apps:
         intent = "FILTER_LAPTOPS"

    # If app is detected and no other strong intent, it's likely about finding a laptop for that app
    if found_apps and intent in ["FIND_LAPTOP_FOR_APP", "FILTER_LAPTOPS"]:
        intent = "FIND_LAPTOP_FOR_APP"

    # If nothing clear is found and no app/laptop detected
    if not found_apps and not found_laptops and budget is None:
        intent = "GENERAL_QUERY"

    return intent

def search_app(query, model, description_embeddings, df, top_n=5, category_kb=None, category_boost=0.5):
    # ... [implementasi lengkap dari notebook] ...
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Hitung cosine similarity awal
    similarities = torch.nn.functional.cosine_similarity(query_embedding, description_embeddings)

    # Preprocess query untuk category matching
    query_tokens_cleaned = basic_preprocessing(query)
    query_tokens_cleaned = remove_stopwords(query_tokens_cleaned)
    query_tokens_cleaned = map_synonyms(query_tokens_cleaned)

    # Beri bobot berdasarkan kategori yang cocok
    boosted_similarities = similarities.clone()
    for i in range(len(df)):
        app_categories_str = df.iloc[i]["Category"]
        app_categories = []
        if isinstance(app_categories_str, str):
             try:
                app_categories = eval(app_categories_str)
             except (SyntaxError, NameError):
                  pass
        elif isinstance(app_categories_str, list):
             app_categories = app_categories_str


        if app_categories:
            # Check if any cleaned query token matches any category of the app
            app_categories_lower = set([cat.lower().strip() for cat in app_categories])
            # Check for intersection between cleaned query tokens and app categories
            if any(token in app_categories_lower for token in query_tokens_cleaned):
                boosted_similarities[i] += category_boost

    # Urutkan berdasarkan boosted similarities
    top_indices = boosted_similarities.argsort(descending=True)[:top_n]

    results = []
    for idx in top_indices:
        idx = idx.item()
        results.append({
            "app": df.iloc[idx]["App"],
            "category": df.iloc[idx]["Category"],
            "similarity": boosted_similarities[idx].item()
        })
    return results

def categorize_laptops_adapted(df_laptops_filtered, req_row_min, req_row_rec, ram_weight=1.0, storage_weight=1.0, storage_type_weight=0.5):
    # ... [implementasi lengkap dari notebook] ...
    if df_laptops_filtered.empty:
         return pd.DataFrame()

    ram_req = req_row_min['RAM']
    # storage_req = req_row_min['File Size']

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
        if vendor == 'nvidia': return req_row.get('GPU_NVIDIA_score', 0)
        elif vendor == 'amd': return req_row.get('GPU_AMD_score', 0)
        elif vendor == 'intel': return req_row.get('GPU_Intel_score', 0)
        else: return req_row.get('GPU_Intel_score', 0)

    # Assign a score based on storage type
    def get_storage_type_score(storage_type):
        if pd.isna(storage_type): return 0
        storage_type_lower = storage_type.lower()
        if 'ssd' in storage_type_lower: return 1.0
        elif 'hdd' in storage_type_lower: return 0.5
        elif 'emmc' in storage_type_lower: return 0.3
        else: return 0.0

    def calculate_match(row):
        cpu_req = get_cpu_req_score(row, req_row_rec)
        gpu_req = get_gpu_req_score(row, req_row_rec)

        # Avoid division by zero and handle potential NaN/inf
        cpu_score_ratio = row['CPU_score'] / cpu_req if cpu_req > 0 and not pd.isna(row['CPU_score']) else (1.0 if cpu_req == 0 else 0.0)
        gpu_score_ratio = row['GPU_score'] / gpu_req if gpu_req > 0 and not pd.isna(row['GPU_score']) else (1.0 if gpu_req == 0 else 0.0)
        ram_score_ratio = (row['RAM'] / ram_req) * ram_weight if ram_req > 0 and not pd.isna(row['RAM']) else (1.0 if ram_req == 0 else 0.0)
        # storage_score_ratio = (row['Storage'] / storage_req) * storage_weight if storage_req > 0 and not pd.isna(row['Storage']) else (1.0 if storage_req == 0 else 0.0)
        storage_type_score = get_storage_type_score(row['Storage type']) * storage_type_weight # Include storage type score

        # Cap the ratios to prevent extreme values from dominating
        cpu_score_ratio = min(cpu_score_ratio, 5.0)
        gpu_score_ratio = min(gpu_score_ratio, 5.0)
        ram_score_ratio = min(ram_score_ratio, 5.0)
        # storage_score_ratio = min(storage_score_ratio, 5.0)

        # Ensure ratios are not NaN or Inf
        cpu_score_ratio = 0.0 if pd.isna(cpu_score_ratio) or cpu_score_ratio == float('inf') else cpu_score_ratio
        gpu_score_ratio = 0.0 if pd.isna(gpu_score_ratio) or gpu_score_ratio == float('inf') else gpu_score_ratio
        ram_score_ratio = 0.0 if pd.isna(ram_score_ratio) or ram_score_ratio == float('inf') else ram_score_ratio
        # storage_score_ratio = 0.0 if pd.isna(storage_score_ratio) or storage_score_ratio == float('inf') else storage_score_ratio
        storage_type_score = 0.0 if pd.isna(storage_type_score) or storage_type_score == float('inf') else storage_type_score


        final_score = (cpu_score_ratio + gpu_score_ratio + ram_score_ratio + storage_type_score) / (2 + ram_weight + storage_weight + storage_type_weight) # Adjust denominator
        return final_score

    def categorize(row):
        cpu_min = get_cpu_req_score(row, req_row_min)
        gpu_min = get_gpu_req_score(row, req_row_min)
        cpu_rec = get_cpu_req_score(row, req_row_rec)
        gpu_rec = get_gpu_req_score(row, req_row_rec)

        # Ensure scores are not NaN before comparison
        row_cpu_score = row['CPU_score'] if not pd.isna(row['CPU_score']) else -1
        row_gpu_score = row['GPU_score'] if not pd.isna(row['GPU_score']) else -1
        row_ram = row['RAM'] if not pd.isna(row['RAM']) else -1
        row_storage = row['Storage'] if not pd.isna(row['Storage']) else -1

        # Handle cases where req_row_min or req_row_rec might be missing keys
        cpu_min = req_row_min.get('CPU_Intel_score', 0)
        gpu_min = req_row_min.get('GPU_NVIDIA_score', 0)
        cpu_rec = req_row_rec.get('CPU_Intel_score', 0)
        gpu_rec = req_row_rec.get('GPU_NVIDIA_score', 0)


        if row_cpu_score < cpu_min or row_gpu_score < gpu_min or row_ram < ram_req:
             return 'Disqualified'

        cpu_rec_flag = row_cpu_score >= cpu_rec
        gpu_rec_flag = row_gpu_score >= gpu_rec

        if cpu_rec_flag and gpu_rec_flag: return 'Recommended'
        elif cpu_rec_flag or gpu_rec_flag: return 'Mixed'
        else: return 'Minimum'

    df_filtered_req = df_laptops_filtered[
        (df_laptops_filtered['RAM'] >= ram_req)
    ].copy()
    print(f"Jumlah laptop setelah filter RAM/Storage minimum: {len(df_filtered_req)}")

    if df_filtered_req.empty:
         print("Tidak ada laptop yang memenuhi persyaratan RAM dan Storage minimum.")
         return pd.DataFrame()


    df_filtered_req['Match_Score'] = df_filtered_req.apply(calculate_match, axis=1)
    df_filtered_req['Category'] = df_filtered_req.apply(categorize, axis=1)


    df_final = df_filtered_req[df_filtered_req['Category'] != 'Disqualified'].copy()
    df_final = df_final.sort_values(by='Match_Score', ascending=False).reset_index(drop=True)
    return df_final[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price', 'Category', 'Match_Score']] # Added Storage type

def get_laptop_recommendations_with_intent(user_query):
    # ... [implementasi lengkap dari notebook] ...
    print(f"Query Pengguna: {user_query}")

    # 1. Jalankan NLP Pipeline & Kenali Intent
    pipeline_result = nlp_pipeline_fuzzy(user_query, min_req_df['App'].tolist(), laptop_df['Model'].tolist(), laptop_df['Brand'].unique().tolist(), unique_keyword_app_map, app_series_map)
    found_apps = pipeline_result['found_apps']
    found_laptops_entities = pipeline_result['found_laptops']
    extracted_budget = pipeline_result['budget']

    detected_intent = recognize_intent_simple(user_query, found_apps, found_laptops_entities, extracted_budget)

    print(f"Hasil NLP Pipeline: App={found_apps}, Laptop/Entity={found_laptops_entities}, Budget={extracted_budget}")
    print(f"Intent Terdeteksi: {detected_intent}")

    # --- Logika berdasarkan Intent ---
    if detected_intent == "COMPARE_LAPTOPS":
        if len(found_laptops_entities) < 2:
             return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan minimal dua nama laptop atau brand untuk dibandingkan."]})

        # Implement comparison logic here (e.g., show specs side-by-side)
        print("\nIntent: Compare Laptops. Implementasi perbandingan spesifikasi laptop...")
        # Filter laptop_df to show specs for the detected laptop entities
        comparison_laptops = laptop_df[
            laptop_df['Model'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Model'].tolist()]) |
            laptop_df['Brand'].str.lower().isin([ent.lower() for ent in found_laptops_entities if ent in laptop_df['Brand'].unique().tolist()])
        ].copy()
        if comparison_laptops.empty:
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan yang ingin Anda bandingkan."]})

        # display(comparison_laptops[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Screen', 'Final Price']])
        return comparison_laptops[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Screen', 'Final Price']]

    elif detected_intent == "FILTER_LAPTOPS":
         if not found_laptops_entities and extracted_budget is None:
              return pd.DataFrame({"Status": ["Informasi Kurang"], "Pesan": ["Mohon sebutkan brand laptop, model, atau budget yang Anda inginkan."]})

         print("\nIntent: Filter Laptops. Menerapkan filter berdasarkan kriteria yang terdeteksi...")
         laptops_to_evaluate = laptop_df.copy()

         # Filter Budget
         if extracted_budget is not None and extracted_budget > 0:
             print(f"  - Memfilter laptop dengan budget <= {extracted_budget}")
             laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()
             print(f"    Jumlah laptop setelah filter budget: {len(laptops_to_evaluate)}")


         # Filter Brand atau Model Spesifik
         if found_laptops_entities:
             print(f"  - Memfilter laptop berdasarkan entitas: {found_laptops_entities}")
             filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
             for entity in found_laptops_entities:
                 if entity in laptop_brand_list:
                     filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                 elif entity in laptop_list:
                      filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                 else:
                     # Use fuzzy matching for entities not in the predefined lists
                     brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                     model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                     if brand_match and brand_match[1] >= 90:
                          filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                     if model_match and model_match[1] >= 90:
                          filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())


             laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()
             print(f"    Jumlah laptop setelah filter entitas: {len(laptops_to_evaluate)}")

         if laptops_to_evaluate.empty:
             print("Tidak ada laptop yang cocok dengan kriteria filter.")
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada laptop yang cocok dengan kriteria filter Anda."]})

        #  print("\nHasil Laptop yang Difilter:")
        #  display(laptops_to_evaluate[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price']])
         return laptops_to_evaluate[['Brand', 'Model', 'CPU', 'GPU', 'RAM', 'Storage', 'Storage type', 'Final Price']]


    elif detected_intent in ["FIND_BEST_LAPTOP_FOR_APP", "FIND_CHEAPEST_LAPTOP_FOR_APP", "FIND_LAPTOP_FOR_APP"]:
        # --- Logika yang sudah ada untuk rekomendasi laptop berdasarkan app ---
        app_targets = []
        if not found_apps:
            print("\nTidak ada app spesifik terdeteksi dari query. Melakukan Semantic Search...")
            semantic_results = search_app(user_query, model, description_embeddings, min_req_df, top_n=top_n_semantic_apps, category_boost=category_boost_value)
            app_targets = [res['app'] for res in semantic_results if res['similarity'] > 0.6]
            if app_targets:
                 print(f"  - Menemukan app relevan dari Semantic Search: {app_targets}")
            else:
                 print("  - Semantic Search tidak menemukan app relevan.")
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan app relevan dari query Anda."]})
        else:
             app_targets = found_apps
             print(f"\nMenggunakan app yang terdeteksi spesifik oleh NLP Pipeline: {app_targets}")

        target_app_min_req = None
        target_app_rec_req = None
        target_app_name = None
        max_requirement_score = -1

        if app_targets:
            print(f"Mengevaluasi persyaratan untuk app target: {app_targets}")
            valid_app_targets = []
            for app_name in app_targets:
                min_req = min_req_kb.get(app_name)
                rec_req = rec_req_kb.get(app_name)

                if min_req and rec_req:
                    valid_app_targets.append(app_name)
                    current_requirement_score = (min_req.get('CPU_Intel_score', 0) + min_req.get('CPU_AMD_score', 0) +
                                                 min_req.get('GPU_NVIDIA_score', 0) + min_req.get('GPU_AMD_score', 0) + min_req.get('GPU_Intel_score', 0))

                    print(f"  - {app_name}: Requirement Score = {current_requirement_score}")

                    if current_requirement_score > max_requirement_score:
                        max_requirement_score = current_requirement_score
                        target_app_name = app_name
                        target_app_min_req = min_req
                        target_app_rec_req = rec_req

            if not target_app_name:
                 print("Tidak dapat menemukan persyaratan yang valid untuk app target dari knowledge base.")
                 return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan persyaratan yang valid untuk app target dari knowledge base."]})

            app_targets = valid_app_targets

            print(f"\nMemilih app dengan persyaratan tertinggi: {target_app_name}")
            print(f"Persyaratan Min: CPU={target_app_min_req['CPU_Intel']}/{target_app_min_req['CPU_AMD']}, GPU={target_app_min_req['GPU_NVIDIA']}/{target_app_min_req['GPU_AMD']}/{target_app_min_req['GPU_Intel']}, RAM: {target_app_min_req['RAM']} GB")
            print(f"Persyaratan Rec: CPU={target_app_rec_req['CPU_Intel']}/{target_app_rec_req['CPU_AMD']}, GPU={target_app_rec_req['GPU_NVIDIA']}/{target_app_rec_req['GPU_AMD']}/{target_app_rec_req['GPU_Intel']}, RAM: {target_app_rec_req['RAM']} GB")


        else:
             print("Tidak ada app relevan yang ditemukan dari query setelah semua upaya.")
             return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak dapat menemukan app relevan dari query Anda."]})


        # Siapkan Daftar Laptop
        laptops_to_evaluate = laptop_df.copy()

        # Filter Budget
        if extracted_budget is not None and extracted_budget > 0:
            print(f"\nMemfilter laptop dengan budget <= {extracted_budget:,}")
            laptops_to_evaluate = laptops_to_evaluate[laptops_to_evaluate['Final Price'] <= extracted_budget].copy()
            print(f"Jumlah laptop setelah filter budget: {len(laptops_to_evaluate)}")
            if laptops_to_evaluate.empty:
                print("Tidak ada laptop dalam budget yang terdeteksi.")
                return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop dalam budget {extracted_budget:,} yang tersedia."]})


        # Filter Brand atau Model Spesifik (if any mentioned alongside app)
        if found_laptops_entities:
            print(f"Memfilter laptop berdasarkan entitas: {found_laptops_entities}")
            filter_mask = pd.Series([False] * len(laptops_to_evaluate), index=laptops_to_evaluate.index)
            for entity in found_laptops_entities:
                if entity in laptop_brand_list:
                    filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == entity.lower())
                elif entity in laptop_list:
                     filter_mask |= (laptops_to_evaluate['Model'].str.lower() == entity.lower())
                else:
                    # Use fuzzy matching for entities not in the predefined lists
                    brand_match = process.extractOne(entity, laptops_to_evaluate['Brand'].unique().tolist(), score_cutoff=90)
                    model_match = process.extractOne(entity, laptops_to_evaluate['Model'].tolist(), score_cutoff=90)
                    if brand_match and brand_match[1] >= 90:
                         filter_mask |= (laptops_to_evaluate['Brand'].str.lower() == brand_match[0].lower())
                    if model_match and model_match[1] >= 90:
                         filter_mask |= (laptops_to_evaluate['Model'].str.lower() == model_match[0].lower())

            laptops_to_evaluate = laptops_to_evaluate[filter_mask].copy()
            print(f"Jumlah laptop setelah filter entitas: {len(laptops_to_evaluate)}")

            if laptops_to_evaluate.empty:
                print(f"Tidak ada laptop yang cocok dengan entitas '{', '.join(found_laptops_entities)}' dalam budget yang terdeteksi.")
                return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": [f"Tidak ada laptop yang cocok dengan '{', '.join(found_laptops_entities)}' dalam kriteria Anda."]})


        print("\nMengkategorikan laptop berdasarkan persyaratan app target...")
        final_recommendations = categorize_laptops_adapted( # Use the adapted function
            laptops_to_evaluate,
            target_app_min_req,
            target_app_rec_req
        )

        if final_recommendations.empty:
            print("\nTidak ada rekomendasi laptop yang ditemukan berdasarkan kriteria Anda.")
            print("Tidak ada laptop yang memenuhi persyaratan minimum app yang relevan di antara laptop yang sudah difilter.")
            return pd.DataFrame({"Status": ["Tidak Ditemukan"], "Pesan": ["Tidak ada rekomendasi laptop yang memenuhi persyaratan minimum app yang relevan dalam kriteria Anda."]})
        else:
            print("\nHasil Rekomendasi Laptop:")
            # Add sorting based on intent (e.g., best -> sort by Match_Score DESC, cheapest -> sort by Final Price ASC within categories)
            if detected_intent == "FIND_BEST_LAPTOP_FOR_APP":
                final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                )
                final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')
                print("\nDiurutkan berdasarkan Match Score (Terbaik ke Terburuk) dalam setiap kategori.")
            elif detected_intent == "FIND_CHEAPEST_LAPTOP_FOR_APP":
                 final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                 )
                 final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Final Price', 'Match_Score'], ascending=[True, True, False]).drop(columns='Category_Order') # Sort by price then match score
                 print("\nDiurutkan berdasarkan Harga (Termurah ke Termahal) dalam setiap kategori.")
            else: # Default FIND_LAPTOP_FOR_APP
                 final_recommendations['Category_Order'] = final_recommendations['Category'].apply(
                    lambda x: 0 if x == 'Recommended' else (1 if x == 'Mixed' else 2)
                 )
                 final_recommendations = final_recommendations.sort_values(by=['Category_Order', 'Match_Score'], ascending=[True, False]).drop(columns='Category_Order')
                 print("\nDiurutkan berdasarkan Match Score (Default).")

            # display(final_recommendations)
            return final_recommendations

    else:
        print("\nIntent tidak dikenali atau tidak didukung saat ini.")
        return pd.DataFrame({"Status": ["Intent Tidak Dikenali"], "Pesan": ["Mohon maaf, niat Anda belum dapat saya proses saat ini."]})

# Flask endpoint
@app.route("/rekomendasi", methods=["GET"])
def api_rekomendasi():
    teks = request.args.get("teks")
    if not teks:
        return jsonify({"error": "Parameter 'teks' wajib diisi"}), 400

    try:
        results = get_laptop_recommendations_with_intent(teks)
        
        if isinstance(results, pd.DataFrame):
            return jsonify({
                "input_user": teks,
                "rekomendasi": results.to_dict(orient="records")
            })
        else:
            return jsonify({
                "input_user": teks,
                "message": results
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
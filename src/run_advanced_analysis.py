import os
import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from src.metrics import calculate_cosine_similarity, calculate_diversity_metrics, calculate_structure_score, calculate_readability, calculate_concept_coverage
from src.reference_loader import load_reference_stories
import re

def get_latest_output_dir():
    # Target the NEW batch
    return "outputs/user_stories_20260112_130331"

def parse_filename_detailed(filename):
    # Format: {p_type}_{subtype}_{model}_{source}_{i}.txt
    parts = filename.replace(".txt", "").split("_")
    
    # 1. Prompt Type
    p_type = parts[0]
    
    # 2. Subtype (Persona)
    subtype = parts[1]
    
    # 3. Model
    models = ["GPT-4o", "GPT-5.2", "Claude-3.5-Sonnet", "DeepSeek-V3", "Gemini-3-Pro", "Gemini-1.5-Pro", "GPT-4-Turbo"]
    found_model = "Unknown"
    for m in models:
        sanitized = m.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
        if sanitized in filename:
            found_model = m
            break
            
    return found_model, p_type, subtype

def calculate_similarity_between_groups(group_a_texts, group_b_texts):
    if not group_a_texts or not group_b_texts:
        return 0.0
    
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = group_a_texts + group_b_texts
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        return 0.0
    
    matrix_a = tfidf_matrix[:len(group_a_texts)]
    matrix_b = tfidf_matrix[len(group_a_texts):]
    
    sim_matrix = cosine_similarity(matrix_a, matrix_b)
    return np.mean(sim_matrix)

def get_top_keywords(texts, top_n=10):
    if not texts:
        return []
    
    # Custom stopwords
    custom_stops = list(CountVectorizer(stop_words='english').get_stop_words())
    custom_stops.extend([
        "want", "user", "story", "stories", "ability", "need", "needs", "data", "relevant", "mainly", "main",
        "able", "like", "use", "using", "ensure", "provide", "platform", "system"
    ])
    
    vectorizer = CountVectorizer(stop_words=custom_stops, max_features=top_n)
    try:
        dtm = vectorizer.fit_transform(texts)
        counts = dtm.sum(axis=0)
        vocab = vectorizer.get_feature_names_out()
        
        freqs = [(vocab[i], counts[0, i]) for i in range(len(vocab))]
        return sorted(freqs, key=lambda x: x[1], reverse=True)
    except ValueError: 
        return []

def main():
    output_dir = get_latest_output_dir()
    print(f"📊 Analyzing artifacts in: {output_dir}\n")
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist.")
        return

    files = glob.glob(os.path.join(output_dir, "*.txt"))
    
    # Load human validation stories (Standard list)
    reference_stories = [] 
    reference_texts = []   
    try:
        with open("data/validation_stories.json", "r") as f:
            reference_stories = json.load(f)
        for item in reference_stories:
            if isinstance(item, dict):
                text = item.get("story") or item.get("text") or item.get("Requirement") or str(item)
                reference_texts.append(text)
            elif isinstance(item, str):
                reference_texts.append(item)
        print(f"Loaded {len(reference_stories)} human validation stories.")
    except Exception as e:
        print(f"Warning: Could not load validation stories: {e}")

    data = []
    
    print(f"Processing {len(files)} files...")
    
    for f in files:
        with open(f, "r") as file:
            content = file.read()
            if not content.strip(): continue
            
        fname = os.path.basename(f)
        model, p_type, subtype = parse_filename_detailed(fname)
        
        # 1. Load Specific Reference for this Subtype
        refs = load_reference_stories(subtype)
        
        # 2. Similarity (vs Human Expert)
        human_sim = 0.0
        coverage = 0.0
        if refs:
            scores = calculate_cosine_similarity(content, refs)
            human_sim = scores['max_similarity'] 
            coverage = calculate_concept_coverage(content, refs)
            
        # 3. Diversity & Quality (Internal)
        # Split content into rough "stories" (lines)
        # Using simple heuristic (>20 chars) for splitting, similar to analyze_results
        rough_stories = [line.strip() for line in content.split('\n') if len(line.strip()) > 20]
        
        div_metrics = calculate_diversity_metrics(rough_stories)
        structure_score = calculate_structure_score(rough_stories)
        readability_grade = calculate_readability(rough_stories)
        
        data.append({
            "model": model,
            "prompt": p_type,
            "subtype": subtype,
            "text": content,
            "filename": fname,
            "human_sim": human_sim,
            "semantic_diversity": div_metrics['semantic_diversity'],
            "lexical_diversity": div_metrics['lexical_diversity'],
            "Structure_Score": structure_score,
            "Readability_Grade": readability_grade,
            "Concept_Coverage": coverage,
            "Story_Count": len(rough_stories)
        })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found.")
        return

    # Create a unique config key: "prompt (subtype)"
    df['config'] = df.apply(lambda x: f"{x['prompt']} ({x['subtype']})" if x['subtype'] != 'standard' else x['prompt'], axis=1)

    # Collectors
    consistency_data = [] # Internal
    agreement_data = []   # Cross-Model
    keyword_data = []     # Keywords

    # 1. Similarity: Within Model (Prompt/Subtype Comparison)
    print("\n### 1. Model Internal Consistency (Granular)")
    models = df['model'].unique()
    configs = df['config'].unique()
    
    for m in models:
        model_df = df[df['model'] == m]
        baseline = "zero-shot" 
        if baseline not in model_df['config'].values:
            zero_shots = [c for c in model_df['config'].unique() if 'zero-shot' in c]
            if zero_shots:
                baseline = zero_shots[0]
            else:
                continue
            
        base_df = model_df[model_df['config'] == baseline]
        base_texts = base_df['text'].tolist()
        base_human_sim = base_df['human_sim'].mean()
        
        for cfg in configs:
            if cfg == baseline: continue
            
            target_df = model_df[model_df['config'] == cfg]
            target_texts = target_df['text'].tolist()
            if not target_texts: continue
            
            target_human_sim = target_df['human_sim'].mean()
            
            sim = calculate_similarity_between_groups(base_texts, target_texts)
            
            p_type = target_df.iloc[0]['prompt']
            subtype = target_df.iloc[0]['subtype']
            
            consistency_data.append({
                "Model": m,
                "Baseline_Config": baseline,
                "Comparison_Config": cfg,
                "Comparison_Prompt_Type": p_type,
                "Comparison_Subtype": subtype,
                "Internal_Similarity": sim,
                "Baseline_Human_Sim": base_human_sim,
                "Comparison_Human_Sim": target_human_sim,
                "Human_Sim_Delta": target_human_sim - base_human_sim
            })

    # 2. Similarity: Cross-Model (Same Prompt/Subtype Config)
    print("### 2. Cross-Model Agreement (Granular)")
    baseline_model = "GPT-4o"
    if baseline_model not in models:
        baseline_model = models[0] if len(models) > 0 else "Unknown"
        
    for cfg in configs:
        config_df = df[df['config'] == cfg]
        
        base_df = config_df[config_df['model'] == baseline_model]
        if base_df.empty: continue
        
        base_texts = base_df['text'].tolist()
        base_human_sim = base_df['human_sim'].mean()
        
        for m in models:
            if m == baseline_model: continue
            target_df = config_df[config_df['model'] == m]
            target_texts = target_df['text'].tolist()
            if not target_texts: continue
            
            target_human_sim = target_df['human_sim'].mean()
            
            sim = calculate_similarity_between_groups(base_texts, target_texts)
            
            p_type = target_df.iloc[0]['prompt']
            subtype = target_df.iloc[0]['subtype']
        
            agreement_data.append({
                "Config": cfg,
                "Prompt_Type": p_type,
                "Subtype": subtype,
                "Baseline_Model": baseline_model,
                "Comparison_Model": m,
                "Agreement_Similarity": sim,
                "Baseline_Human_Sim": base_human_sim,
                "Comparison_Human_Sim": target_human_sim
            })

    # 3. Keyword Analysis
    print("### 3. Top Keywords Analysis")
    
    # A. Human Baseline
    if reference_texts:
        keywords = get_top_keywords(reference_texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({
            "Category": "Reference",
            "Name": "HUMAN_STORIES",
            "Keywords": k_str
        })
    
    # B. Per Model (Aggregated)
    for m in models:
        texts = df[df['model'] == m]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({
            "Category": "By Model",
            "Name": m,
            "Keywords": k_str
        })

    # C. Per Prompt Type (Aggregated)
    prompts = df['prompt'].unique()
    for p in prompts:
        texts = df[df['prompt'] == p]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({
            "Category": "By Prompt",
            "Name": p,
            "Keywords": k_str
        })
        
    # D. Per Persona Subtype (Aggregated)
    subtypes = df['subtype'].unique()
    for s in subtypes:
        if s == 'standard': continue # Skip standard
        texts = df[df['subtype'] == s]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({
            "Category": "By Persona",
            "Name": s,
            "Keywords": k_str
        })

    # E. Per Model x Prompt (Granular)
    for m in models:
        for p in prompts:
            subset = df[(df['model'] == m) & (df['prompt'] == p)]
            if subset.empty: continue
            texts = subset['text'].tolist()
            keywords = get_top_keywords(texts, top_n=10)
            k_str = ", ".join([f"{k}({c})" for k,c in keywords])
            keyword_data.append({
                "Category": "By Model x Prompt",
                "Name": f"{m} | {p}",
                "Keywords": k_str
            })

    # F. Per Model x Persona (Granular)
    for m in models:
        for s in subtypes:
            if s == 'standard': continue
            subset = df[(df['model'] == m) & (df['subtype'] == s)]
            if subset.empty: continue
            texts = subset['text'].tolist()
            keywords = get_top_keywords(texts, top_n=10)
            k_str = ", ".join([f"{k}({c})" for k,c in keywords])
            keyword_data.append({
                "Category": "By Model x Persona",
                "Name": f"{m} | {s}",
                "Keywords": k_str
            })

    # Export
    excel_path = os.path.join(output_dir, "advanced_analysis_report.xlsx")
    print(f"\nSaving Excel report to: {excel_path}")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            cols_export = ['model', 'prompt', 'subtype', 'filename', 'human_sim', 
                           'Structure_Score', 'Readability_Grade', 'Concept_Coverage', 
                           'semantic_diversity', 'lexical_diversity', 'Story_Count']
                           
            df[cols_export].sort_values(
                by=['model', 'prompt', 'subtype']
            ).to_excel(writer, sheet_name='Granular_Metrics', index=False)
                
            pd.DataFrame(consistency_data).to_excel(writer, sheet_name='Internal_Consistency', index=False)
            pd.DataFrame(agreement_data).to_excel(writer, sheet_name='Cross_Model_Agreement', index=False)
            pd.DataFrame(keyword_data).to_excel(writer, sheet_name='Keyword_Analysis', index=False)
            
        print("✅ Excel file saved successfully!")
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")

if __name__ == "__main__":
    main()

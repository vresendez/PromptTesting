import pandas as pd
import os
import glob
import json
import re
from src.metrics import calculate_cosine_similarity, calculate_diversity_metrics, calculate_structure_score, calculate_readability, calculate_concept_coverage
from src.reference_loader import load_reference_stories
from sklearn.feature_extraction.text import CountVectorizer

def get_top_keywords(texts, top_n=10):
    if not texts:
        return []
    
    # Custom stopwords for User Stories
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

def analyze():
    output_dir = get_latest_output_dir()
    print(f"Analyzing directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist.")
        return

    files = glob.glob(os.path.join(output_dir, "*.txt"))
    if not files:
        print("No files found.")
        return

    data = []
    print(f"Processing {len(files)} files...")
    
    for f in files:
        fname = os.path.basename(f)
        model, p_type, subtype = parse_filename_detailed(fname)
        
        with open(f, "r") as file:
            content = file.read()
            if not content.strip(): continue
            
        # 1. Load Specific Reference for this Subtype
        refs = load_reference_stories(subtype)
        
        # 2. Similarity (vs Human Expert)
        max_sim = 0.0
        avg_sim = 0.0
        coverage = 0.0
        if refs:
            scores = calculate_cosine_similarity(content, refs)
            max_sim = scores['max_similarity']
            avg_sim = scores['avg_similarity']
            coverage = calculate_concept_coverage(content, refs)
            
        # 3. Diversity & Quality (Internal)
        # Split content into rough "stories" (lines)
        rough_stories = [line.strip() for line in content.split('\n') if len(line.strip()) > 20]
        div_metrics = calculate_diversity_metrics(rough_stories)
        structure_score = calculate_structure_score(rough_stories)
        readability_grade = calculate_readability(rough_stories)
            
        data.append({
            "model": model,
            "prompt_type": p_type,
            "subtype": subtype,
            "text": content, # Needed for keyword analysis
            "filename": fname,
            "max_similarity": max_sim,
            "avg_similarity": avg_sim,
            "semantic_diversity": div_metrics['semantic_diversity'],
            "lexical_diversity": div_metrics['lexical_diversity'],
            "Structure_Score": structure_score,
            "Readability_Grade": readability_grade,
            "Concept_Coverage": coverage,
            "Story_Count": len(rough_stories)
        })
        
    # 4. Calculate Human Expert Baselines
    human_baselines = []
    # Unique subtypes from data
    subtypes = set(d['subtype'] for d in data)
    for s in subtypes:
        refs = load_reference_stories(s)
        if not refs: continue
        
        # Calculate metrics for the HUMAN stories themselves
        div = calculate_diversity_metrics(refs)
        structure = calculate_structure_score(refs)
        read = calculate_readability(refs)
        human_cnt = len(refs)
        
        human_baselines.append({
            "model": "HUMAN_BASELINE",
            "prompt_type": "N/A",
            "subtype": s,
            "text": " ".join(refs), # Needed for keyword analysis
            "filename": f"Reference_{s}",
            "max_similarity": 1.0, #Human is 100% similar to Human
            "avg_similarity": 1.0,
            "semantic_diversity": div['semantic_diversity'],
            "lexical_diversity": div['lexical_diversity'],
            "Structure_Score": structure,
            "Readability_Grade": read,
            "Concept_Coverage": 1.0, # Covers itself
            "Story_Count": human_cnt
        })
    
    # Add human baselines to main data for comparison
    data.extend(human_baselines)
        
    df = pd.DataFrame(data)
    
    # Rename for clarity
    df.rename(columns={"max_similarity": "Human_Similarity_Score"}, inplace=True)
    
    if df.empty:
        print("No data extracted.")
        return

    print("\n--- RESULTS ANALYSIS ---\n")
    
    # 1. Aggregate Performance by Model
    cols_eval = ['Human_Similarity_Score', 'Structure_Score', 'Readability_Grade', 'Concept_Coverage', 'semantic_diversity', 'Story_Count']
    model_perf = df.groupby("model")[cols_eval].mean().sort_values("Human_Similarity_Score", ascending=False)
    print("Aggregate Performance by Model (Including Human Baseline):")
    print(model_perf)
    
    # 2. Performance by Prompt Strategy
    # Filter out human for this one as prompt_type N/A messes it up or keep it separate
    ai_df = df[df['model'] != "HUMAN_BASELINE"]
    prompt_perf = ai_df.groupby(["prompt_type", "model"])[cols_eval].mean()
    print("\nPerformance by Prompt Strategy:")
    print(prompt_perf)
    
    # 3. Performance by Human Persona (Subtype)
    human_only = df[df['model'] == "HUMAN_BASELINE"]
    if not human_only.empty:
        print("\nHuman Expert Baseline Counts:")
        print(human_only[['subtype', 'Story_Count']])

    # 3. Keyword Analysis
    print("\n--- Generating Keyword Analysis ---")
    keyword_data = []
    
    # A. By Model
    models = df['model'].unique()
    for m in models:
        texts = df[df['model'] == m]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({"Category": "By Model", "Name": m, "Keywords": k_str})
        
    # B. By Prompt
    prompts = [p for p in df['prompt_type'].unique() if p != "N/A"]
    for p in prompts:
        texts = df[df['prompt_type'] == p]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({"Category": "By Prompt", "Name": p, "Keywords": k_str})
        
    # C. By Persona (Subtype)
    subtypes = df['subtype'].unique()
    for s in subtypes:
        texts = df[df['subtype'] == s]['text'].tolist()
        keywords = get_top_keywords(texts)
        k_str = ", ".join([f"{k}({c})" for k,c in keywords])
        keyword_data.append({"Category": "By Persona", "Name": s, "Keywords": k_str})

    # 4. Statistical Significance
    print("\n--- Calculating Statistical Significance ---")
    from scipy.stats import f_oneway, ttest_ind
    
    significance_data = []
    
    # Filter out Baseline for stats to compare AI vs AI first (fair comparison)
    stats_df = df[df['model'] != "HUMAN_BASELINE"]
    
    metrics_to_test = ['Human_Similarity_Score', 'Structure_Score', 'Readability_Grade', 'Concept_Coverage', 'semantic_diversity', 'Story_Count']
    
    for metric in metrics_to_test:
        # 1. ANOVA across ALL models
        model_groups = []
        model_names = []
        for m in stats_df['model'].unique():
            values = stats_df[stats_df['model'] == m][metric].dropna().tolist()
            if len(values) > 1:
                model_groups.append(values)
                model_names.append(m)
        
        if len(model_groups) > 1:
            f_val, p_val = f_oneway(*model_groups)
            is_sig = p_val < 0.05
            significance_data.append({
                "Metric": metric,
                "Test_Type": "ANOVA (All Models)",
                "Comparison": "Across All AI Models",
                "F_Statistic": f_val,
                "P_Value": p_val,
                "Significant": "YES" if is_sig else "NO"
            })
            
            # Post-hoc: If significant, compare Top 1 vs Top 2
            if is_sig:
                # Find top 2 means
                means = stats_df.groupby("model")[metric].mean().sort_values(ascending=False)
                top_1 = means.index[0]
                top_2 = means.index[1]
                
                group1 = stats_df[stats_df['model'] == top_1][metric].dropna()
                group2 = stats_df[stats_df['model'] == top_2][metric].dropna()
                
                t_val, p_pair = ttest_ind(group1, group2, equal_var=False)
                significance_data.append({
                    "Metric": metric,
                    "Test_Type": "T-Test (Welch)",
                    "Comparison": f"{top_1} vs {top_2} (Top 2)",
                    "F_Statistic": t_val, # T-stat in F column
                    "P_Value": p_pair,
                    "Significant": "YES" if p_pair < 0.05 else "NO"
                })

    # Export to Excel
    excel_path = os.path.join(output_dir, "analysis_results.xlsx")
    print(f"\nSaving Analysis Results to: {excel_path}")
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            model_perf.to_excel(writer, sheet_name='Model_Performance')
            prompt_perf.to_excel(writer, sheet_name='Prompt_Strategy')
            # Separate sheet for granular including baselines
            cols_export = ['model', 'prompt_type', 'subtype', 'filename'] + metrics_to_test + ['lexical_diversity']
            df[cols_export].sort_values("Human_Similarity_Score", ascending=False).to_excel(writer, sheet_name='All_Generations', index=False)
            
            pd.DataFrame(keyword_data).to_excel(writer, sheet_name='Keyword_Analysis', index=False)
            pd.DataFrame(significance_data).to_excel(writer, sheet_name='Statistical_Significance', index=False)
            
        print("✅ Excel file saved successfully!")
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")

if __name__ == "__main__":
    analyze()

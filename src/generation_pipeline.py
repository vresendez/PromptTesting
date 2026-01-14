import pandas as pd
import json
import os
from src.llm_clients import LLMClient
from src.utils import save_output
from src.metrics import calculate_cosine_similarity
from src import llm_clients as llm

# 1. Load Data
try:
    with open("data/requirements.txt", "r") as f:
        full_requirements_text = f.read()
    
    # Parse requirements into chunks by source
    requirements_chunks = {}
    current_source = "Overall_Context"
    current_text = []
    
    lines = full_requirements_text.split('\n')
    for line in lines:
        if line.startswith("=== SOURCE:"):
            # Save previous chunk
            if current_text:
                requirements_chunks[current_source] = "\n".join(current_text)
            # Start new chunk
            current_source = line.replace("=== SOURCE:", "").replace("===", "").strip()
            current_text = []
        elif line.startswith("=== END SOURCE:"):
            continue # Skip end marker
        else:
            current_text.append(line)
    
    # Add the last chunk
    if current_text:
        requirements_chunks[current_source] = "\n".join(current_text)
        
    print(f"Loaded requirements from {len(requirements_chunks)} sources.")

except FileNotFoundError:
    requirements_chunks = {"default": "Requirements file not found."}

try:
    with open("data/few_shot_examples.json", "r") as f:
        few_shot_data = json.load(f)
        examples_str = json.dumps(few_shot_data, indent=2)
except FileNotFoundError:
    examples_str = ""

# Load Reference Stories for Metrics (Priority: Validation Stories > Few-Shot Examples)
reference_stories = []
if os.path.exists("data/validation_stories.json"):
    try:
        with open("data/validation_stories.json", "r") as f:
            reference_stories = json.load(f)
        print(f"Loaded {len(reference_stories)} validation stories for metrics.")
    except Exception as e:
        print(f"Error loading validation stories: {e}")

if not reference_stories and examples_str:
     reference_stories = [item['output'] for item in few_shot_data]

with open("data/prompts.json") as f:
    prompts_config = json.load(f)

# 2. Initialize LLM Clients
llm = LLMClient()
# Models to compare
models = {
    "Claude-3.5-Sonnet": lambda p: llm.poe_call(p, bot_name="Claude-3.5-Sonnet"),
    "GPT-5.2": lambda p: llm.poe_call(p, bot_name="GPT-5.2"), # User requested
    "DeepSeek-V3": lambda p: llm.poe_call(p, bot_name="DeepSeek-V3"), 
    "Gemini-3-Pro": lambda p: llm.poe_call(p, bot_name="Gemini-3-Pro")
}

import datetime

# Output directory with Timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# timestamp = "20260111_163705" # Resume previous batch
output_dir = f"outputs/user_stories_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving outputs to: {output_dir}")

n_generations = 1 # Usually 1 is enough for qualitative comparison, or more for variability

results = []

print(f"Starting generation with {len(prompts_config)} prompt strategies on {len(models)} models...")

# 3. Experiment Loop
for prompt_cfg in prompts_config:
    p_type = prompt_cfg['type']
    base_template = prompt_cfg['template']
    
    # Determine iterations for this prompt type
    sub_variations = []
    if p_type == 'persona':
        for persona in prompt_cfg['personas']:
            sub_variations.append({
                "subtype": persona["persona_name"], # Use specific name
                "base_prompt": base_template.replace("{persona_name}", persona["persona_name"])
                                            .replace("{persona_description}", persona.get("persona_description", ""))
                                            .replace("{persona_tasks}", persona.get("persona_tasks", ""))
                                            .replace("{persona_constraints}", persona.get("persona_constraints", ""))
            })
    elif p_type == 'few-shot':
        sub_variations.append({
            "subtype": "standard",
            "base_prompt": base_template.replace("{examples}", examples_str)
        })
    else: # zero-shot, reasoning
        sub_variations.append({
            "subtype": "standard",
            "base_prompt": base_template
        })

    for variation in sub_variations:
        subtype = variation['subtype']
        base_prompt_template = variation['base_prompt']
        
        print(f"  Running {p_type} - {subtype}...")

        # Optimize: Iterate over Requirement Sources (Chunks)
        # FILTER: Only process specific sources to save credits during testing
        TARGET_SOURCES = ["Overall_Context", "data/RequirementsProtectSum.xlsx", "data/PCSummary.docx"]
        
        for source_name, source_text in requirements_chunks.items():
            # Skip empty sources
            if not source_text.strip():
                continue
            
            # Skip sources that are not in our target list (unless list is empty)
            if TARGET_SOURCES and source_name not in TARGET_SOURCES:
                continue

            # Truncate if still too massive (safety net, e.g. 50k chars)
            safe_text = source_text[:100000] 
            
            final_prompt = base_prompt_template.replace("{requirements}", safe_text)
            
            # Shorten source name for filename
            # sanitized_source = source_name.replace("data/", "").replace(".pdf", "").replace(".docx", "").replace(".xlsx", "")[:20]

            for model_name, model_func in models.items():
                print(f"    - Model: {model_name} | Source: {source_name}")
                try:
                    for i in range(n_generations):
                        # Construct filename
                        sanitized_model = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
                        sanitized_source = source_name.replace('data/', '').replace('.txt', '').replace('.xlsx', '').replace('.docx', '')
                        filename = f"{p_type}_{subtype}_{sanitized_model}_{sanitized_source}_{i}.txt"
                        filepath = os.path.join(output_dir, filename)
                        
                        # SKIP IF EXISTS
                        if os.path.exists(filepath):
                            print(f"      ⏩ Skipping existing file: {filename}")
                            continue

                        output = model_func(final_prompt)
                        
                        with open(filepath, "w") as f:
                            f.write(output)

                        # Calculate Metrics
                        similarity_scores = calculate_cosine_similarity(output, reference_stories)

                        results.append({
                            "prompt_type": p_type,
                            "subtype": subtype,
                            "model": model_name,
                            "source": source_name,
                            "iteration": i,
                            "output_file": filepath,
                            "max_similarity": similarity_scores['max_similarity'],
                            "avg_similarity": similarity_scores['avg_similarity']
                        })
                except Exception as e:
                    print(f"      ❌ Error {model_name} on {source_name}: {e}")

# 4. Save Log
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("outputs/generation_log.csv", index=False)
print("Done! Results saved to outputs/generation_log.csv")

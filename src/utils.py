import json
import os
from datetime import datetime

def save_output(context_id, context_text, llm, prompt_type, generated_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "context_id": context_id,
        "context_text": context_text,
        "llm": llm,
        "prompt_type": prompt_type,
        "generated_text": generated_text,
        "timestamp": datetime.now().isoformat()
    }
    filename = f"{context_id}_{llm}_{prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

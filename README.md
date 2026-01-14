# LLM User Story Generation Framework 🧪🤖

This project implements a scientific framework for evaluating Large Language Models (LLMs) on their ability to generate user stories from requirements.

It employs a rigorous **Generation -> Analysis -> Validation** pipeline, comparing AI outputs against **Human Expert Baselines**.

## 🚀 Quick Start

### 1. Generation Phase
Run the main pipeline to generate user stories using different models and prompt strategies.
```bash
python -m src.generation_pipeline
```
*   **Input**: `data/requirements.txt`, `data/prompts.json`
*   **Output**: `outputs/user_stories_YYYYMMDD_HHMMSS/`

### 2. Analysis Phase (Performance & Stats)
Run the primary analysis script to calculate similarity, diversity, quality scores, and statistical significance.
```bash
python analyze_results.py
```
*   **Output**: `analysis_results.xlsx` (in the latest output folder)
*   **Sheets**: Model Performance, Prompt Strategy, Keyword Analysis, Statistical Significance.

### 3. Deep-Dive Phase (consistency)
Run the advanced analysis for internal consistency verification (Consistency Matrices).
```bash
python -m src.run_advanced_analysis
```
*   **Output**: `advanced_analysis_report.xlsx`

---

## 🧠 Methodology

### A. Models Evaluated
We evaluate a diverse set of LLMs models via the Poe API:
*   **Gemini 3 Pro**
*   **GPT-5.2**
*   **DeepSeek V3**
*   **Claude 3.5 Sonnet**
*   (Benchmarks against **GPT-4o**)

### B. Prompting Strategies
We test 4 distinct prompting techniques to see which yields the best engineering quality:
1.  **Zero-Shot**: "Generate user stories for X."
2.  **Few-Shot**: "Here are 3 examples of good stories. Generate stories for X."
3.  **Chain-of-Thought (Reasoning)**: "Think step-by-step about the users before generating."
4.  **Persona-Based**: "You are an Expert Geneticist. Write user stories for..."

### C. Human Baselines
We evaluate AI not just against itself, but against **Ground Truth** data:
*   `UserStoriesGenetists.docx`
*   `UserStoriesPediatricians.docx`
*   `UserStoriesStatistician.docx`
*   `UserStoriesParsing.docx` (Standard)

---

## 📊 Metrics Definitions

The framework uses 7 key metrics to evaluate quality:

### Quantitative Metrics
1.  **Human Similarity (Cosine)**: How semantically similar is the AI output to the Human Expert stories? (0.0 - 1.0)
2.  **Semantic Diversity**: How distinct are the AI's stories from each other? (1.0 = highly unique, 0.0 = repetitive)
3.  **Lexical Diversity (TTR)**: Type-Token Ratio. Measures vocabulary richness.

### Qualitative Metrics
4.  **Structure Score**: % of stories that strictly follow the *"As a... I want... so that..."* Agile format.
5.  **Readability (Flesch-Kincaid)**: The US Grade Level required to understand the story (e.g., Grade 12 for Geneticists vs Grade 8 for General).
6.  **Concept Coverage**: What percentage of unique High-Value Concepts (words >4 chars) from the Human Expert baseline appear in the AI output?
7.  **Story Count**: Volume of output (AI tends to be concise/lazy compared to Humans).

### Statistical Analysis
*   **ANOVA**: Tests if there is a statistically significant difference across all models.
*   **T-Test (Welch)**: If ANOVA is significant, compares the Top 1 vs Top 2 models to declare a "Winner".

---

## 📂 Project Structure

*   `src/generation_pipeline.py`: Main orchestration script.
*   `analyze_results.py`: Primary evaluation script (Stats, Metrics).
*   `src/metrics.py`: Library of metric functions.
*   `src/reference_loader.py`: Logic for loading specific human baselines per persona.
*   `data/`: Contains prompt templates and human reference DOCX files.

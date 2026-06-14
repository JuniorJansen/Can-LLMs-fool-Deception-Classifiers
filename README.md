# Can LLMs fool deception classifiers?

This is the code for my Bachelor End Project, *"Can Large Language Models Fool Deception Classifiers? Adversarial Paraphrase Attacks Using Open-Source LLMs."*

The idea is straightforward. I take a DistilBERT classifier that was trained to tell truthful personal stories from made-up ones, and I check whether an open-source LLM can reword a story so the meaning stays the same but the classifier flips its prediction. I do this for six attacker models (Llama 3.2 3B, Mistral 7B, Gemma 3 4B, Qwen 2.5 7B, Llama 3.3 70B and GPT-OSS 120B) and three prompting strategies (zero-, one- and few-shot), on a fixed sample of 200 HippoCorpus narratives, and then aggregate everything into the tables, figures and statistical tests reported in the thesis.

## Layout

```
.
├── requirements.txt
├── src/
│   ├── config.py
│   ├── main.py
│   ├── analyze_stats.py
│   ├── make_tables.py
│   ├── make_figures.py
│   ├── generate_examples.py
│   ├── attack/
│   │   ├── attack_loop.py
│   │   ├── paraphraser.py
│   │   └── similarity.py
│   ├── data/
│   │   └── load_data.py
│   └── model/
│       └── classifier.py
├── data/
├── models/DistilBERT/
├── results/
└── figures/
```

The scripts you actually run sit in `src/` and are meant to be called from the project root:

- `src/main.py` runs the attack for one model or for all of them.
- `src/analyze_stats.py` runs the four hypothesis tests (H1 to H4).
- `src/make_tables.py` prints the numbers behind the thesis tables.
- `src/make_figures.py` writes the figure PDFs.
- `src/generate_examples.py` is the little helper I used to pick the few-shot example paraphrases.

Everything else under `src/` is library code: the iterative attack loop, the paraphraser (Ollama for the small models, hosted APIs for the two big ones), the Sentence-BERT similarity check, the data loader, and the DistilBERT wrapper.

## Setup

You need Python 3.11+ and the pinned dependencies:

```
pip install -r requirements.txt
```

The four smaller attackers run locally through [Ollama](https://ollama.com):

```
ollama pull llama3.2
ollama pull mistral
ollama pull gemma3:4b
ollama pull qwen2.5:7b
```

The two larger ones go through hosted endpoints, so you'll need API keys for those:

```
set HYPERBOLIC_API_KEY=...   # Llama 3.3 70B
set TOGETHER_API_KEY=...     # GPT-OSS 120B
```

## Getting the data and the weights

I don't ship the HippoCorpus narratives or the fine-tuned classifier weights with this repo. The dataset license and the ethics statement in the thesis both rule out redistributing the narratives or the model outputs, so they stay out. You don't actually need either one to check my results, because the per-sample outcomes are already in `results/` and the analysis scripts read straight from there. The data and weights only come into play if you want to re-run the attacks from scratch.

For the data, grab HippoCorpus from the official Microsoft Research Open Data release (Sap et al., 2020; it's listed under "HippoCorpus" on [msropendata.com](https://msropendata.com)). The loader expects a CSV at `data/hippocorpus_test_truncated.csv` with a `condition` column (`truthful` for recalled stories, `deceptive` for imagined ones) and a `text_truncated` column holding each story truncated to fit DistilBERT's 512-token limit. The 200-narrative sample is drawn from that file with `SEED = 42`, so you end up with exactly the same rows I used.

For the classifier, `models/DistilBERT/` already contains the config and tokenizer. Drop the fine-tuned weights (`*.safetensors`) in alongside them, or point `MODEL_PATH` in `src/config.py` at your own DistilBERT trained on the HippoCorpus train split.

## Running it

Run the attack for every model, or pick one with `--model`:

```
python -m src.main
python -m src.main --model llama3.2
python -m src.main --model openai/gpt-oss-120b
```

Each run writes three per-strategy JSON files plus a summary file into `results/`. It checkpoints after every narrative, so an interrupted run resumes on its own, and it skips any strategy whose file already exists.

Once the results are in place, reproduce the statistics and the figures:

```
python -m src.analyze_stats     # H1 to H4 tests
python -m src.make_tables       # the table numbers
python -m src.make_figures      # asymmetry.pdf, robust_accuracy.pdf, asr_heatmap.pdf
```

Passing `--correct` to `analyze_stats.py` re-runs the tests on only the narratives the classifier got right to begin with, which is the robustness check in the thesis.

## Settings

These all live in `src/config.py` and match the values in the thesis:

| Setting | Value |
|---|---|
| Balanced sample size | `N = 200` (100 truthful + 100 deceptive) |
| Candidates per iteration | `NUM_CANDIDATES = 5` |
| Max attack iterations | `MAX_ITER = 10` |
| Sentence-BERT cosine threshold | `SIMILARITY_THRESHOLD = 0.85` |
| Length tolerance | `LENGTH_TOLERANCE = 0.10` (±10%) |
| Random seed | `SEED = 42` (seeds `random`, `numpy`, `torch`) |

## The statistics, in short

`analyze_stats.py` runs four tests, each Bonferroni-corrected within its own family:

- **H1 (model size):** unpaired chi-squared for every (small model, large model, strategy) pair.
- **H2 (architecture):** paired McNemar of Llama 3.3 against GPT-OSS, per strategy.
- **H3 (prompting):** pairwise McNemar across the three strategies, within each model.
- **H4 (direction):** chi-squared (Fisher's exact when a cell drops below 5) comparing the deceptive-to-truthful and truthful-to-deceptive success rates, per condition.

So McNemar handles the paired contrasts and chi-squared the unpaired ones.

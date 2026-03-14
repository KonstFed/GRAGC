This project aims to train GraphSAGE for Retrieval Augmented Code Generation with context of whole repository.

You can read more about general idea in [report](./assets/RACG%20IDEA.pdf).

Authors:
- Konstantin Fedorov (k.fedorov@innopolis.university)
- Boris Zarubin (b.zarubin@innopolis.university)

## Installation

```bash
uv sync
```

## Evaluation on EvoCodeBench

### 1. Prepare dataset

Clone [EvoCodeBenchPlus](https://github.com/Poseidondon/EvoCodeBenchPlus) and set up repos according to its instructions.

Parse repos into a graph dataset:

```bash
uv run python -m ragc.datasets.create_dataset \
    --evocodebench /path/to/EvoCodeBench/dataset/repos \
    configs/evocodebench/create_ds.yml
```

This creates cached PyG graphs under `data/torch_cache/evocodebench/`.

### 2. Configure

Edit a config under `configs/evocodebench/gnn/<model>/greedy.yml`:

- `retrieval.model_path` — path to your trained `BEST_CHECKPOINT.pt`
- `fusion.generator` — model path or API endpoint for the LLM
- `task_path` — path to EvoCodeBench `oracle.jsonl`
- `repos_path` — path to EvoCodeBench cloned repos

### 3. Run inference

**Code completion:**

```bash
uv run python -m ragc.test.inference \
    -t completion \
    -o output/evocodebench/completions.jsonl \
    -c configs/evocodebench/gnn/<model>/greedy.yml
```

**Retrieval metrics only (recall / precision):**

```bash
uv run python -m ragc.test.inference \
    -t retrieval \
    -o output/evocodebench/retrieval_metrics.json \
    -c configs/evocodebench/gnn/<model>/greedy.yml
```

Output is a JSONL file with `namespace` and `completion` fields, compatible with EvoCodeBench evaluation scripts.

## Evaluation on SWE-QA-Bench

### 1. Clone benchmark repos

```bash
git clone https://github.com/peng-weihan/SWE-QA-Bench.git
cd SWE-QA-Bench
bash clone_repos.sh
cd ..
```

### 2. Prepare dataset

Parse the cloned repos into a graph dataset:

```bash
uv run python -m ragc.datasets.create_swe_qa_dataset \
    configs/swe_qa_bench/create_ds.yml \
    --repos-dir SWE-QA-Bench/SWE-QA-Bench/datasets/repos
```

This creates cached PyG graphs under `data/torch_cache/swe_qa_bench/`.

### 3. Configure

Edit `configs/swe_qa_bench/gnn/greedy.yml`:

- `retrieval.model_path` — path to your trained `BEST_CHECKPOINT.pt`
- `inference.fusion.generator.model` — LLM model name (e.g. `gpt-oss-120b`)
- `inference.fusion.generator.base_url` — API endpoint
- `questions_dir` — path to `SWE-QA-Bench/SWE-QA-Bench/datasets/questions`
- `repos` — (optional) list of repo names to evaluate; `null` for all available

Set the API key:

```bash
export API_KEY=your-api-key
```

### 4. Run inference

```bash
uv run python -m ragc.test.swe_qa_inference \
    -c configs/swe_qa_bench/gnn/greedy.yml \
    -o output/swe_qa_bench/gnn
```

This produces per-repo JSONL files (e.g. `flask.jsonl`, `django.jsonl`) in the output directory. Each line contains `question`, `final_answer`, and `retrieved_context`.

### 5. Score

Use the SWE-QA-Bench scorer to evaluate answers against reference:

```bash
cd SWE-QA-Bench
# set OPENAI_API_KEY, OPENAI_BASE_URL, MODEL, METHOD in .env
python -m SWE-QA-Bench.score.main
```

Scores are written to `SWE-QA-Bench/datasets/scores/<model>/<method>/`.

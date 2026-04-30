# GRAGC: Graph-Based Retrieval-Augmented Code Generation

**GRAGC** trains a heterogeneous Graph Neural Network (HeteroGraphSAGE) to retrieve relevant code context from an entire repository for LLM-based code generation. It constructs a whole-repository call graph and learns to rank functions and classes by relevance to a query at inference time.

## Data Availability

The training dataset consists of 1,672 public GitHub repositories listed in [`meta.csv`](./meta.csv). Each entry contains `owner/repo` and the exact commit hash at which it was accessed, allowing the training graphs to be reproduced deterministically. No proprietary data was used.

---

## Installation

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Reproducing Results

### EvoCodeBench

**1. Clone the benchmark**

```bash
git clone https://github.com/Poseidondon/EvoCodeBenchPlus.git
```

Follow its setup instructions to download the benchmark repositories.

**2. Build graph dataset**

```bash
uv run python -m ragc.datasets.create_dataset \
    --evocodebench /path/to/EvoCodeBenchPlus/dataset/repos \
    configs/evocodebench/create_ds.yml
```

Caches PyG graphs under `data/torch_cache/evocodebench/`.

**3. Train the GNN** — see [Training](#training-the-gnn) below.

**4. Configure** — edit `configs/evocodebench/<method>/greedy_*.yml`:

| Field | Description |
|---|---|
| `retrieval.model_path` | Path to trained checkpoint |
| `fusion.generator` | LLM model name or API endpoint |
| `task_path` | Path to EvoCodeBench `oracle.jsonl` |
| `repos_path` | Path to cloned EvoCodeBench repos |

**5. Run inference**

```bash
# Code completion
uv run python -m ragc.test.inference \
    -t completion \
    -o output/evocodebench/completions.jsonl \
    -c configs/evocodebench/gnn/greedy_deepseekv3.yml

# Retrieval metrics only (recall / precision)
uv run python -m ragc.test.inference \
    -t retrieval \
    -o output/evocodebench/retrieval.json \
    -c configs/evocodebench/gnn/greedy_deepseekv3.yml
```

---

## Retrieval Methods

| Method | Config dir | Description |
|---|---|---|
| **GNN** (proposed) | `gnn/` | HeteroGraphSAGE trained on call-graph triplets |
| GNN + local context | `gnn_local_context/` | GNN retrieval combined with surrounding file context |
| Local context | `local_context/` | Embedding retrieval over functions in the same file |
| Golden context | `golden_context/` | Oracle: ground-truth context provided to the LLM |
| Local golden | `local_golden/` | Oracle local context (ground-truth functions in same file) |
| No context | `without_context/` | LLM only, no retrieval |

---

## Training the GNN

First build the graph dataset (Step 2 above). Training is driven by the `Trainer` class in `ragc/train/gnn/training.py` — instantiate it with a `TorchGraphDataset`, a `HeteroGraphSAGE` model, and a `TripletLoss`, then call `trainer.train()`. The checkpoint is saved as a standard PyTorch state dict.

---

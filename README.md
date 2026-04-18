# Multi-Agent Object Detection Training System

A closed-loop, LLM-driven multi-agent framework for automated YOLO object detection training optimization using [Ultralytics](https://github.com/ultralytics/ultralytics).

## Architecture

```
Training Loop
     │
     ▼
Monitor Agent ──── (epoch callbacks) ──▶ alerts / early-stop signals
     │
     ▼ (training end)
Analyzer Agent ─── per-class AP, confusion matrix, FP/FN, size, anchors
     │
     ▼
Memory Agent ────── recall similar past (weakness → fix) patterns
     │
     ▼
Planner Agent ───── LLM Chain-of-Thought reasoning → config delta proposals
     │
     ▼
Auto-Trainer Agent ─ dedup check, VRAM estimate, spawn new training run
     │
     └──────────────────────────────────────────────── (loop)
```

### Agents

| Agent | Role |
|-------|------|
| **Monitor** | Hooks into Ultralytics callbacks (`on_fit_epoch_end`). Tracks mAP, losses, grad norm, LR. Detects overfitting, plateau, gradient explosion, LR collapse. |
| **Analyzer** | Post-training analysis: per-class AP, FP/FN breakdown (background vs confusion), confusion matrix, object-size mAP (small/medium/large), anchor assignment ratio. |
| **Memory** ★ | Cross-session vector store. Saves `(weakness_signature → config_delta, actual_improvement)`. Enables retrieval-augmented planning — Planner queries Memory before reasoning. |
| **Planner** | LLM (Claude) reasoning via Chain-of-Thought prompt. Outputs minimal config **deltas** with confidence scores and expected improvement. Falls back to rule-based logic without API key. |
| **Auto-Trainer** | Validates proposals (dedup by config hash, VRAM estimation). Registers runs in SQLite registry. Spawns Ultralytics training jobs. |

## Installation

```bash
# Clone / unzip
cd ultralytics_multiagent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key (for LLM Planner)
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Quick Start

```bash
# Full multi-agent training loop
python main.py train \
  --data path/to/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --iterations 5 \
  --imgsz 640 \
  --batch 16

# Without LLM (rule-based Planner)
python main.py train --data data.yaml --no-llm

# Demo mode (no real training, mock results for testing)
python main.py train --data data.yaml --demo

# Analyze an existing run only
python main.py analyze \
  --data data.yaml \
  --results-dir runs/detect/exp

# Generate plan for a specific weakness
python main.py plan \
  --weakness-desc "small objects performing poorly, mAP50 < 0.2" \
  --current-map 0.45

# Check experiment registry
python main.py status

# Memory Agent stats
python main.py memory-stats
```

## Configuration

All settings in `config/default_config.yaml`. Key sections:

```yaml
system:
  max_iterations: 10       # max agent loop iterations
  gpu_memory_gb: 16.0      # for VRAM check

monitor:
  plateau_patience: 8      # epochs without improvement → early stop
  overfitting_patience: 5  # consecutive val_loss increase → alert
  grad_norm_max: 100.0     # gradient explosion threshold

analyzer:
  weak_class_ap_threshold: 0.3   # AP below this → weakness
  small_obj_size: 32             # px boundary: small objects
  medium_obj_size: 96            # px boundary: medium objects

planner:
  model: claude-sonnet-4-20250514
  max_suggestions: 3
  confidence_threshold: 0.6      # proposals below this are dropped

memory:
  max_entries: 500
  similarity_threshold: 0.75     # min cosine sim for recall
  top_k_recall: 5
```

## Output Structure

```
experiments/
├── registry.db                    # SQLite experiment registry
├── memory_store.pkl               # Memory Agent vector store
├── iteration_history.json         # Per-iteration metrics
├── logs/
│   ├── orchestrator.log
│   ├── analyzeragent.log
│   └── planneragent.log
└── run_20240101_120000_abc123/
    ├── train/                     # Ultralytics training output
    │   ├── weights/
    │   └── results.csv
    ├── analysis/
    │   └── weakness_report.json
    ├── planner/
    │   ├── proposals.json
    │   └── llm_cot_response.txt   # Full LLM chain-of-thought
    └── plots/
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── per_class_ap.png
        └── size_breakdown.png
```

## Research Contributions

| Component | Contribution |
|-----------|-------------|
| **Planner Agent** | LLM-driven hyperparameter reasoning for object detection with interpretable CoT |
| **Memory Agent** | Cross-session retrieval-augmented planning — system improves over time |
| **Closed loop** | Zero human intervention between iterations |
| **Failure-targeted** | Diagnose-first then search (vs blind AutoML) |
| **Config delta** | Only changed params per proposal → natural ablation study |

## Benchmark Suggestions

Compare against:
- Random search (same compute budget)
- Optuna / Bayesian optimization
- Human expert (same time budget)
- Ablation: Planner without Memory Agent

## Requirements

- Python 3.9+
- CUDA GPU (recommended, 8GB+ VRAM)
- `ANTHROPIC_API_KEY` in `.env` (optional — falls back to rule-based planner)

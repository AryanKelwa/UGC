# Enterprise Real Estate Lead Qualification Fine-Tuning Repository

Welcome to the **Enterprise LLM Fine-Tuning** repository for Anywhere Real Estate. This project consists of a high-performance, concurrent data pipeline designed to generate synthetic conversation datasets, mask personal identifiers (PII), structure inputs into standard Chat formats, and run Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) pipelines.

---

## 📂 Repository Directory Layout

The workspace has been organized into a professional, enterprise-grade ML directory layout:

```text
llm-finetuning-project/
│
├── configs/                   # Central configurations (YAML format)
│   ├── model/                 # Model hyper-parameters (Llama3, Mistral, Qwen)
│   ├── training/              # Training parameters (SFT, DPO, LoRA, QLoRA)
│   ├── dataset/               # Dataset parsing settings (real_estate, support)
│   └── inference/             # Inference configs (generation settings)
│
├── data/                      # Dataset folders
│   ├── raw/                   # Raw datasets (conversations/ scapers/ csv/)
│   ├── interim/               # Semi-processed data (PII anonymized / cleaned)
│   ├── processed/             # Ready-to-train datasets (train.jsonl, etc.)
│   └── cache/                 # Temporary preprocessing directories
│
├── notebooks/                 # Exploratory research notebooks
│
├── src/                       # Main Package Code
│   ├── data/                  # Ingestion, Anonymization, and Splitting
│   ├── training/              # Training wrappers, Callbacks, and Metrics
│   ├── models/                # HuggingFace loader, PEFT, and Quantization
│   ├── evaluation/            # Validation benchmarks
│   ├── inference/             # Local text generation
│   ├── deployment/            # merge, GGUF exports, Triton deployment
│   ├── utils/                 # Structured loggers, configs, CUDA monitors
│   └── pipelines/             # Orchestrated sequential workflows (SFT Pipeline)
│
├── scripts/                   # Shell scripts for compute jobs
├── docker/                    # Infrastructure container configurations
├── requirements/              # Dependency requirement splits
│
├── main.py                    # High-performance Unified CLI Entrypoint
├── Makefile                   # Developer convenience short-keys
├── pyproject.toml             # Standard package metadata
├── requirements.txt           # Python dependency file
└── README.md                  # Project overview manual
```

---

## ⚡ Quick Start

### 1. Setup Environment
Install core package dependencies and download the PII anonymization NLP model:
```bash
make setup
```

Configure your `.env` file in the root directory:
```env
GOOGLE_GEMINI_API_KEY=AIzaSy...
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
```

### 2. Generate Synthetic Training Data
Fetch raw synthetic lead qualification datasets from the Google Gemini API (concurrently rotates keys and handles rate-limits with backoff):
```bash
make generate
```

### 3. Mask Personal Identifiers (PII)
Concurrent redaction of emails, names, and phone numbers in raw data:
```bash
make clean
```

### 4. Create Splits (Train / Val / Test)
Merges batches, shuffles with a deterministic seed, structures records into standard OpenAI-style chat schemas, and outputs final datasets into `data/processed/`:
```bash
make prepare
```

### 5. Trigger SFT Training
Launch parameter-efficient fine-tuning (PEFT/LoRA) using configurations:
```bash
make train-sft
```

---

## 🧬 Pipeline Integration CLI (`main.py`)

All operations are controlled through a clean, unified terminal interface:

```bash
# Run the entire sequential pipeline end-to-end (generate -> clean -> prepare -> train)
python main.py run-all --model llama3_8b --training sft
```

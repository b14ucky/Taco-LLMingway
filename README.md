# Taco LLMingway - Core Library & ML Training

This is the core Python library and heavy-duty training pipeline for the Taco LLMingway project. It features a custom **GPT (Generative Pre-trained Transformer)** implementation built from scratch using PyTorch, optimized for generating long-form lyrics in the style of Taco Hemingway.

## Features

- **Scalable GPT Architecture**: Causal Transformer decoder capable of running both "mini" local tests and deep Kaggle training sessions.
- **Sinusoidal Position Encodings**: Fixed embeddings for precise sequence positioning.
- **Production-Ready Trainer**: Supports `DataParallel` for multi-GPU training, automatic checkpointing, and training resumption.
- **Tokenization**: Custom character-level/word-level tokenizer with JSON export/import.
- **Kaggle Optimized**: Seamless integration with Kaggle environments for GPU-accelerated training.

## Kaggle Training & Data

The heavy lifting (training the final model) was performed on Kaggle. You can explore the training process and the dataset used via the links below:

- **[Training Notebook on Kaggle](https://www.kaggle.com/code/b14ucky/taco-llmingway-training)** – Full training logs, hyper-parameter tuning, and loss visualization.
- **[Taco Hemingway Lyrics Dataset](https://www.kaggle.com/datasets/b14ucky/taco-hemingway-lyrics)** – The raw text dataset used to train the model.

## Tech Stack

- **Core**: [PyTorch](https://pytorch.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/)
- **Progress Bars**: [tqdm](https://github.com/tqdm/tqdm)
- **Packaging**: [Setuptools](https://setuptools.pypa.io/)

## Project Structure

```bash
├── main.py              # Sample training entry point
├── pyproject.toml       # Package metadata and dependencies
├── src/
│   └── taco_llmingway/  # Main package logic
│       ├── model.py     # GPT architecture
│       ├── train.py     # Advanced Trainer class
│       └── ...          # Utilities, Tokenizer, Logger
├── data/raw/            # Source lyrics dataset
└── checkpoints/         # Intermediate model states (.pth)
```

## Model Configuration (Production Grade)

The model used for the final inference was trained with these parameters:

| Parameter           | Value              |
| ------------------- | ------------------ |
| **Context Length**  | 128                |
| **Embed Dim**       | 256                |
| **Attention Heads** | 8                  |
| **Decoder Blocks**  | 6                  |
| **FFN Dim**         | 1024               |
| **Batch Size**      | 128                |
| **Learning Rate**   | $3 \times 10^{-4}$ |

## Installation

To install the `taco_llmingway` package in editable mode:

```bash
pip install -e .
```

## Project Ecosystem

This repository is the heart of the **Taco LLMingway** project. To see how this model is served and presented, check out the other components:

- **[taco-llmingway-frontend](https://github.com/b14ucky/taco-llmingway-frontend)** – A minimalist Next.js web interface with real-time token streaming.
- **[taco-llmingway-backend](https://github.com/b14ucky/taco-llmingway-backend)** – FastAPI inference server that bridges the model with the web UI.
- **[taco-llmingway](https://github.com/b14ucky/taco-llmingway)** – (This Repository) The core library and training logic.

## License

This project is licensed under the **MIT License**.

# Comprehensive Documentation: MindCore LLM Project

This document provides a detailed overview of the MindCore project, which focuses on building a small-scale Large Language Model (LLM) from scratch using PyTorch. The goal is to provide an educational and functional implementation of the core Transformer architecture.

## 1. Project Structure

The project is organized into logical directories to separate model implementation, data handling, training, and serving components.

| Directory/File | Description |
| :--- | :--- |
| `src/model/` | Core implementation of the Transformer architecture (Attention, Encoder, Decoder, MultiHeadAttention, etc.). |
| `src/data/` | Data handling, including dataset loading and tokenization. |
| `src/training/` | Training loop, optimization, and utility functions. |
| `src/inference/` | Logic for generating text using the trained model. |
| `src/api/` | FastAPI application for serving the model via HTTP endpoints. |
| `src/config/model_config.py` | Central configuration file for model hyperparameters and file paths. |
| `scripts/` | Shell scripts for common operations (setup, training, API launch). |
| `data/` | Contains raw data, processed data, and sample corpus files. |
| `models/` | Storage for model checkpoints and fine-tuned versions. |
| `notebooks/` | Jupyter notebooks for data exploration and experimentation. |
| `requirements.txt` | List of Python dependencies. |

## 2. Setup and Installation

### Prerequisites

*   Python 3.8+
*   `git`

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MindCore
    ```

2.  **Set up the environment:**
    The project uses a virtual environment (`.venv`) and installs dependencies from `requirements.txt`. By default, it uses the CPU-only version of PyTorch to minimize installation size.

    ```bash
    ./scripts/setup_env.sh
    ```

3.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

## 3. Model Architecture

The LLM is based on the standard Transformer architecture, consisting of an Encoder and a Decoder stack, although it is primarily designed for sequence-to-sequence or decoder-only tasks depending on the specific implementation of `src/model/transformer_model.py`.

Key components include:

*   **`src/model/attention.py`**: Defines the basic Scaled Dot-Product Attention mechanism.
*   **`src/model/multihead_attention.py`**: Implements the Multi-Head Attention layer.
*   **`src/model/encoder_block.py` / `src/model/decoder_block.py`**: Defines the individual layers of the Transformer stack, including attention, residual connections, layer normalization, and feed-forward networks.
*   **`src/model/transformer.py`**: Assembles the Encoder and Decoder stacks.
*   **`src/model/transformer_model.py`**: The main model class, handling token embeddings, positional encoding, and the final linear layer for vocabulary prediction.

## 4. Configuration

All critical hyperparameters and file paths are managed in [`src/config/model_config.py`](src/config/model_config.py) within the `Config` class.

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 23641 | Size of the vocabulary used by the tokenizer. |
| `max_seq_len` | 128 | Maximum sequence length the model can handle. |
| `embed_dim` | 512 | Dimensionality of the token and positional embeddings. |
| `n_heads` | 8 | Number of attention heads in the Multi-Head Attention layers. |
| `n_layers` | 6 | Number of Encoder/Decoder blocks in the Transformer stack. |
| `ffn_dim` | 2048 | Dimensionality of the inner layer of the Feed-Forward Network. |
| `dropout` | 0.1 | Dropout rate applied throughout the model. |
| `batch_size` | 32 | Batch size used during training. |
| `num_epochs` | 10 | Total number of training epochs. |
| `learning_rate` | 3e-4 | Learning rate for the optimizer. |
| `DATA_PATH` | "data/sample/corpus.txt" | Path to the training data file. |
| `MODEL_PATH` | "checkpoints/model.pt" | Path where the trained model checkpoint is saved. |
| `VOCAB_PATH` | "data/vocab.txt" | Path to the generated vocabulary file. |
| `device` | "cuda" or "cpu" | Device used for computation (automatically detects CUDA if available). |

## 5. Usage

### 5.1. Data Preparation

Ensure your training data is placed in the location specified by `DATA_PATH` in the configuration. The tokenizer will process this data to create a vocabulary file (`VOCAB_PATH`).

### 5.2. Training the Model

The training process is managed by `src/training/train_loop.py` and executed via a shell script.

```bash
./scripts/run_train.sh
```
This script handles data loading, model initialization, and saving the final checkpoint to `models/checkpoints/model.pt`.

### 5.3. Running Inference

The `src/inference/generate.py` module contains logic for generating text sequences using a trained model.

### 5.4. Running the API Server

The project includes a simple API built with FastAPI to serve the trained model for real-time inference.

1.  Ensure the model is trained and the checkpoint is available.
2.  Run the API server:
    ```bash
    ./scripts/run_api.sh
    ```
3.  The API will be accessible at `http://127.0.0.1:8000`. Check the API documentation (e.g., `/docs` endpoint) for available routes, typically including a `/generate` endpoint for text generation.

## 6. Dependencies

The required Python packages are listed in [`requirements.txt`](requirements.txt):

```
torch==2.9.0+cpu
numpy
tqdm
fastapi
uvicorn[standard]
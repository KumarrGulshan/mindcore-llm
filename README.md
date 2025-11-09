# MindCore: LLM From Scratch Project

This project aims to build a small-scale Large Language Model (LLM) from scratch using PyTorch, primarily for educational and learning purposes. It implements a minimal Transformer architecture and includes scripts for training and serving the model via a simple API.

## Project Structure

The codebase is organized as follows:

-   `src/model/`: Core implementation of the Transformer architecture (`attention.py`, `transformer.py`).
-   `src/data/`: Data handling, including dataset loading and tokenization (`dataset_loader.py`, `tokenizer.py`).
-   `src/training/`: Training loop and utilities (`train_loop.py`, `utils.py`).
-   `src/inference/`: Model generation/inference logic (`generate.py`).
-   `src/api/`: FastAPI application for serving the model (`app.py`).
-   `scripts/`: Shell scripts for common operations (`run_train.sh`, `run_api.sh`, `setup_env.sh`).
-   `data/`: Directory for raw, processed, and sample data.
-   `models/`: Directory for storing model checkpoints and fine-tuned versions.
-   `notebooks/`: Jupyter notebooks for exploration and experimentation.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd MindCore
    ```

2.  **Set up the environment:**
    The project uses a virtual environment and requires PyTorch (CPU-only version is used by default to manage disk space).

    ```bash
    ./scripts/setup_env.sh
    ```
    *Note: This script creates a `.venv` and installs dependencies from `requirements.txt`.*

3.  **Activate the environment:**
    ```bash
    source .venv/bin/activate
    ```

## Usage

### 1. Training the Model

Use the provided script to start the training process. Ensure your data is prepared and configured correctly in `src/config/model_config.py`.

```bash
./scripts/run_train.sh
```

### 2. Running the API Server

Once the model is trained and saved, you can run the inference API using FastAPI/Uvicorn.

```bash
./scripts/run_api.sh
```
The API will typically be available at `http://127.0.0.1:8000`.

## Dependencies

Dependencies are managed via `requirements.txt`:
-   `torch` (PyTorch)
-   `numpy`
-   `tqdm`
-   (If running the API, you may need to manually install `fastapi` and `uvicorn` if they are not in `requirements.txt`.)
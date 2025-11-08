# LLM From Scratch Project

This project aims to build a small-scale Large Language Model (LLM) from scratch using PyTorch, primarily for educational and learning purposes.

## Project Scope and Architecture
*   **Framework:** PyTorch (CPU-only version installed due to disk space constraints).
*   **Model Target:** A small-scale Transformer-based model (e.g., 1-2 layers, small vocabulary) for foundational understanding.

## Current Status: Environment Setup
The development environment has been successfully set up:

1.  **Dependencies Defined:** A `requirements.txt` file was created listing `torch`, `numpy`, and `tqdm`.
2.  **Virtual Environment Created:** A Python virtual environment (`.venv`) was created to manage dependencies locally.
3.  **Dependencies Installed:** All required packages were installed into the virtual environment using the CPU-only index for PyTorch to bypass disk quota limitations.

## Next Steps
The next phase involves data preparation and model implementation:

1.  Prepare the training data (collection, cleaning, tokenization).
2.  Implement the model architecture (e.g., Transformer layers, attention mechanisms).
3.  Train the model.
4.  Evaluate and fine-tune the model.
5.  Deploy the model.
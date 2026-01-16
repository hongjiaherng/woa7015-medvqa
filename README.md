# woa7015-medvqa

From CNN-LSTM to VLM: A Comparative Analysis on Med-VQA with SLAKE

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Install the dev dependencies:

```bash
uv sync --dev --extra cpu
# If you have a CUDA-enabled GPU, you can install the GPU version:
# uv sync --dev --extra cu130
```

## Development

Use jupyter lab/notebook for development:

```bash
# Do this
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
jupyter lab # jupyter notebook

# Or this
uv run jupyter lab  # uv run jupyter notebook
```

Or use any IDE/text editor :)

## Reproducing Results

To reproduce the results, run the notebooks in the following order:

0. Check out the `data/README.md` for downloading instructions of the SLAKE dataset.
1. `01_prelim.ipynb`: Preliminaries (Not good, first try on the subject, months ago)
2. `02_eda.ipynb`: Exploratory Data Analysis (To answer some questions about the dataset)
3. `03_model_train.ipynb`: Model Training (We run the training on A100 80GB GPU on Colab). Time taken for models:
   - CNN-LSTM (Frozen): 11 mins 23 secs (overfitted)
   - CNN-LSTM (Unfrozen): 12 mins 16 secs (overfitted)
   - BLIP (LoRA): 51 mins 43 secs (not overfitted yet, with enough compute, we can train longer)
4. `04_model_eval.ipynb`: Model Evaluation (After training, we evaluate models on my local machine with RTX5060 8GB Laptop GPU. Mainly loading results from previous training runs, so should be fast).

## Project Structure

- `data/`: Dataset directory (not included in repo, see `data/README.md` for downloading instructions)
- `notebooks/`: Jupyter notebooks for development and experimentation
- `checkpoints/`: Saved model checkpoints (not included in repo, due to size constraints, history and evaluation results are saved instead)
- `docs/`: Submission report and related documentation
- Source code is organized as follows:

  ```bash
  src
  └── woa7015_medvqa
      ├── __init__.py
      ├── v1                               # Preliminary version
      │   ├── dataset
      │   │   ├── collate_fn.py
      │   │   ├── slake_dataset.py
      │   │   ├── tokenizer.py
      │   │   └── vocab.py
      │   ├── __init__.py
      │   └── models
      │       └── resnet_lstm.py
      └── v2                               # Final version
          ├── data
          │   ├── collate.py
          │   ├── __init__.py
          │   ├── slake.py
          │   ├── tokenizers.py
          │   └── transforms.py
          ├── eval
          │   ├── evaluate_blip.py
          │   ├── evaluate_cnn_lstm.py
          │   ├── __init__.py
          │   └── metrics.py
          ├── __init__.py
          ├── models
          │   ├── blip_lora.py
          │   ├── cnn_lstm.py
          │   └── __init__.py
          ├── train
          │   ├── __init__.py
          │   ├── train_blip.py
          │   └── train_cnn_lstm.py
          └── utils.py
  ```

## References

- [SLAKE Dataset](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [BLIP Paper](https://proceedings.mlr.press/v162/li22n/li22n.pdf)
- [BLIP from HF Model](https://huggingface.co/Salesforce/blip-vqa-base)
- [PEFT](https://huggingface.co/docs/peft/en/index)

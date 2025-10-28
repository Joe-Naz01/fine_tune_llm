# Fine-Tuning a Language Model — Custom Dataset Adaptation

**Problem.** Large language models provide general capabilities, but domain-specific accuracy and tone require targeted fine-tuning. This project demonstrates the workflow for customizing a pretrained model on a curated dataset.

**Approach.**
- Selected a lightweight pretrained transformer (TinyLlama / GPT-2 variant).
- Preprocessed domain text: cleaning, tokenization, train-test split.
- Used Hugging Face `Trainer` with `transformers` + `datasets` libraries.
- Configured LoRA / PEFT fine-tuning for efficiency on limited GPU.
- Saved intermediate checkpoints and final model weights.
- Logged loss curves and sample generations for qualitative evaluation.

**Results (qualitative).**
- Model adapted to the dataset’s style and vocabulary after several epochs.
- Fine-tuning improved contextual relevance and reduced generic phrasing.
- Demonstrated parameter-efficient fine-tuning with manageable compute cost.

**What I Learned.**
- Practical steps for preparing text datasets for language-model training.
- Differences between full fine-tuning and PEFT (LoRA, adapters).
- How hyperparameters (learning rate, batch size, max seq len) affect convergence.
- Managing checkpoints and early stopping in constrained environments.

## Quick Start
```bash
git clone https://github.com/Joe-Naz01/fine_tune_llm.git
cd fine_tune_llm

conda env create -f environment.yml
conda activate fine_tune_llm
jupyter notebook

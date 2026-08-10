# Smart MCQ Solver — End-to-End Deep Learning Solution

An ensemble pipeline that predicts the top-3 most likely correct answers (out of options A–E) for knowledge-based multiple-choice questions, scored by **MAP@3** (Mean Average Precision @ 3).

## 🧠 Overview

This project combines **three structurally different models** — one built entirely from scratch, one fine-tuned from a pretrained transformer, and one tree-based model on engineered features — then blends their predictions in a tuned weighted ensemble.

| # | Model | Category | Idea |
|---|-------|----------|------|
| 1 | TF-IDF + PyTorch MLP | **Built from scratch** | No pretrained weights anywhere — learns purely from the training data using hand-engineered lexical features (TF-IDF cosine similarity, length ratio, word overlap) |
| 2 | DeBERTa-v3 (`AutoModelForMultipleChoice`) | **Pretrained, fine-tuned** | Transfer learning — HuggingFace's multiple-choice head fine-tuned on this task |
| 3 | XGBoost on engineered similarity features | **Additional model of choice** | Tree-based, feature-driven — combines TF-IDF similarity, zero-shot Sentence-Transformer similarity, lexical overlap, and relative-rank features |
| — | Weighted ensemble of the three | **Final submission** | Probability outputs blended with weights tuned on a held-out validation split |

## 📂 Notebook Structure

1. **Setup & data loading** — environment verification, W&B experiment tracking init, deterministic seeding
2. **Light EDA** — missing values, answer-class balance, prompt/option length distributions
3. **MAP@3 metric implementation** — the competition's scoring function, built from scratch
4. **Train/validation split** — 15% held out, stratified by answer label
5. **Model 1 — TF-IDF + MLP (from scratch)** — TF-IDF vectorization → hand-engineered similarity features → 3-layer MLP with dropout, trained with Adam + cross-entropy
6. **Model 2 — DeBERTa-v3 fine-tuning** — each question turned into 5 (prompt, option) pairs, tokenized, fine-tuned via HuggingFace `Trainer`
7. **Model 3 — XGBoost** — TF-IDF similarity, Sentence-Transformer (`all-MiniLM-L6-v2`) similarity, word overlap, and relative-rank features feeding a multiclass XGBoost classifier
8. **Ensembling** — grid search over blend weights, evaluated on validation MAP@3
9. **Final inference** — predictions on `test.csv`, submission file generation
10. *(Optional, commented out)* Zero-shot LLM prompting extension

## 🛠️ Tech Stack

- **Core:** Python, NumPy, Pandas, scikit-learn
- **Deep Learning:** PyTorch, HuggingFace `transformers`, `datasets`, `accelerate`
- **Additional models:** XGBoost, Sentence-Transformers
- **Experiment tracking:** Weights & Biases (W&B)
- **Environment:** Kaggle Notebooks (GPU-accelerated)

## 📊 Experiment Tracking

All training runs (from-scratch MLP, DeBERTa fine-tuning, XGBoost, and the final ensemble) are logged to **Weights & Biases**, tracking training loss, validation MAP@3, accuracy, and macro-F1 per run — see the `smart-mcq-solver` W&B project, group `mcq-ensemble-v1`.

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers==4.46.0 datasets==3.1.0 accelerate==1.1.0 \
    sentence-transformers==3.0.1 xgboost==2.1.1 wandb
```
> NumPy, Pandas, scikit-learn, and PyTorch are assumed to already be present (e.g. via the Kaggle base image) — do not reinstall/pin these, to avoid dependency conflicts.

### Data
Place `train.csv`, `test.csv`, and `sample_submission.csv` in the expected data directory (auto-detected for Kaggle competition input paths; adjust `get_data_path()` for local/Colab use).

### Running
Open `notebook.ipynb` and run cells top to bottom. A valid `WANDB_API_KEY` (via Kaggle Secrets or environment variable) is required for experiment logging.

### Output
Produces `submission.csv` with columns `ID` and `Prediction` (space-separated top-3 predicted option letters per question).

## 📈 Evaluation Metric

**MAP@3** — for each question, a top-3 ranked list of predicted answers is submitted. Score per question is `1/rank` of the correct answer if it appears in the top 3 (1.0 for 1st place, 0.5 for 2nd, 0.333 for 3rd), else 0. Final score is the mean across all questions.

## 📁 Repository Structure

```
├── notebook.ipynb          # Full pipeline: EDA → 3 models → ensembling → inference
├── submission.csv          # Final predictions (generated)
└── README.md
```

## 🔮 Future Improvements

- Refit all three models on the full training set (train + held-out validation) before final inference, to use all available labeled data
- Explore a larger DeBERTa checkpoint or additional pretrained backbones
- Add more engineered features to the XGBoost model
- Try stacking (a learned meta-model) instead of a simple weighted average for ensembling

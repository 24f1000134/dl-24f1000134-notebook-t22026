# Smart MCQ Solver

A deep learning project for ranking multiple-choice answers by how well each option fits a given question — the model reads the question and its five candidate options, then scores and ranks them by relevance/correctness.

The challenge focuses on evaluating a model's ability to understand context, reason across options, and rank answers effectively.

## How it works

Given a question and five options (A–E), the model produces a relevance score for each option and ranks them, with the top-ranked option treated as the predicted answer. Several architectures were explored during experimentation, with a Siamese BiLSTM ultimately used for the deployed app:

- Encodes the question and each option independently through a shared embedding + BiLSTM encoder
- Combines question/option representations (concat, absolute difference, elementwise product) into a joint feature vector
- Scores each option through a small feed-forward head
- Ranks options by score, using an ensemble average across two trained folds

## Project structure

```
.
├── notebooks/
│   ├── EDA.ipynb                          # Exploratory data analysis
│   ├── Feature_Engineering.ipynb          # Feature engineering
│   ├── DL-24f1000134-notebook-t22026.ipynb# Main project notebook
│   ├── milestones/                        # Milestone 
│   └── model_training/                    # Model experiments
│       ├── Model #1_Bi-LSTM.ipynb
│       ├── Model #2_Deberta.ipynb
│       ├── Model #3_ELECTRA.ipynb
│       └── Model #4 Deberta with Rag.ipynb
├── model_deployment/
│   ├── app.py                             # Streamlit inference app
│   └── models/                            # Trained checkpoints + vocab
├── requirements.txt
└── README.md
```

## Models explored

The `notebooks/model_training/` directory tracks the progression of approaches tried for this task:

1. **Bi-LSTM** — Siamese bidirectional LSTM encoder
2. **DeBERTa** — transformer-based fine-tuning
3. **ELECTRA** — transformer-based fine-tuning
4. **DeBERTa + RAG** — retrieval-augmented approach combining DeBERTa with retrieved context

## Tech stack

- **PyTorch** — model implementation and inference
- **Transformers / Sentence-Transformers / FAISS** — transformer baselines and RAG experiments
- **scikit-learn, NumPy, pandas, SciPy** — data handling and evaluation
- **Streamlit** — deployment/demo app
- **Weights & Biases** — experiment tracking
- **Matplotlib / Seaborn / WordCloud** — EDA and visualization

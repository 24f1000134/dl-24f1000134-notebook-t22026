import json
import os
import re

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

DEVICE = torch.device("cpu")

PAD = 0
UNK = 1

EMBED_DIM = 128
HIDDEN_SIZE = 384
DROPOUT = 0.3

MAX_Q_LEN = 80
MAX_OPT_LEN = 25

OPTIONS = ["A", "B", "C", "D", "E"]

st.set_page_config(page_title="Smart MCQ Solver", layout="centered")

_BOILERPLATE_PAT = re.compile(
    r"pick the best possible answer:?|\bcarefully\b\.?|among the listed options\.?|"
    r"choose the (correct|best) (option|answer|response)\.?|"
    r"select the (best|correct|most appropriate) (option|answer|response)\.?|"
    r"determine the correct option:?|identify the correct statement:?|"
    r"based on the given context\.?|which of the following( statements)?( is true)?",
    flags=re.IGNORECASE,
)

_WS = re.compile(r"\s+")


def normalize_text(text):
    text = _BOILERPLATE_PAT.sub(" ", str(text))
    return _WS.sub(" ", text).strip()


def encode_text(text, vocab, max_len):
    tokens = re.findall(r"[a-zA-Z']+|\d+", text.lower())[:max_len]
    ids = [vocab.get(token, UNK) for token in tokens]
    ids += [PAD] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


class SiameseBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * 2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, ids):
        emb = self.embed(ids)
        out, _ = self.lstm(emb)

        mask = (ids != PAD).unsqueeze(-1).float()
        out = out * mask + (1 - mask) * -1e9

        pooled, _ = out.max(dim=1)
        return self.dropout(pooled)

    def forward(self, q_ids, opt_ids):
        B = q_ids.size(0)

        h_q = self.encode(q_ids)
        h_o = self.encode(opt_ids.view(B * 5, -1)).view(B, 5, -1)
        h_q = h_q.unsqueeze(1).expand(-1, 5, -1)

        joined = torch.cat([h_q, h_o, torch.abs(h_q - h_o), h_q * h_o], dim=-1)
        return self.head(joined).squeeze(-1)


@st.cache_resource
def load_vocab():
    vocab_path = os.path.join(MODEL_DIR, "vocab.json")

    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Missing vocab.json: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_models():
    vocab = load_vocab()
    models = []

    for fold in range(2):
        checkpoint = os.path.join(MODEL_DIR, f"best_lstm_fold{fold}.pth")

        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        model = SiameseBiLSTM(vocab_size=len(vocab), embed_dim=128, hidden=384, dropout=0.3)

        state_dict = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        models.append(model)

    return vocab, models


@torch.inference_mode()
def predict(question, options, vocab, models):
    question = normalize_text(question)
    options = [normalize_text(option) for option in options]

    q_ids = encode_text(question, vocab, MAX_Q_LEN)
    option_ids = np.stack([encode_text(option, vocab, MAX_OPT_LEN) for option in options])

    q_tensor = torch.from_numpy(q_ids).long().unsqueeze(0)
    option_tensor = torch.from_numpy(option_ids).long().unsqueeze(0)

    fold_scores = [model(q_tensor, option_tensor).squeeze(0) for model in models]
    scores = torch.stack(fold_scores).mean(dim=0)

    ranking = torch.argsort(scores, descending=True)

    return [
        {
            "rank": rank,
            "option": OPTIONS[index.item()],
            "score": float(scores[index.item()].item()),
        }
        for rank, index in enumerate(ranking, start=1)
    ]


def main():
    st.title("Smart MCQ Solver")
    st.write("Enter a multiple-choice question and its five options.")

    with st.spinner("Loading model..."):
        try:
            vocab, models = load_models()
        except Exception as error:
            st.error("The model could not be loaded.")
            st.exception(error)
            st.stop()

    st.subheader("Question")
    question = st.text_area(
        "Question",
        height=160,
        placeholder="Enter your question here...",
        label_visibility="collapsed",
    )

    st.subheader("Options")
    options = [
        st.text_input("Option A", placeholder="Enter option A"),
        st.text_input("Option B", placeholder="Enter option B"),
        st.text_input("Option C", placeholder="Enter option C"),
        st.text_input("Option D", placeholder="Enter option D"),
        st.text_input("Option E", placeholder="Enter option E"),
    ]

    if not st.button("Click to get the answer.", use_container_width=True):
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    if any(not option.strip() for option in options):
        st.warning("Please fill in all five options.")
        return

    with st.spinner("Ranking options..."):
        results = predict(question, options, vocab, models)

    best = results[0]

    st.divider()
    st.subheader("Predicted Answer")
    st.success(f"Option {best['option']}")

    st.subheader("Top 3 Ranking")
    for result in results[:3]:
        st.markdown(f"**Option {result['option']}**")
        st.caption(f"Model score: {result['score']:.4f}")

    with st.expander("View all option scores"):
        for result in results:
            st.write(f"**Option {result['option']}** — {result['score']:.4f}")

#running app
if __name__ == "__main__":
    main()

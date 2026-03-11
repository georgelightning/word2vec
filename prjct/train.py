import numpy as np
from tqdm import tqdm
from model import Word2Vec
from preprocessing import Tokenizer

# 1. Load and Preprocess
print("--- Loading Text ---")
try:
    with open("karamazov.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
except FileNotFoundError:
    print("Error: 'karamazov.txt' not found. Please check the filename.")
    exit()

t = Tokenizer(min_count=5)
words = t.clean_text(raw_text)
data = t.vocabulary(words)  # This creates word2id, id2word, and unigram_table

model = Word2Vec(vocab_size=t.vocab_size, embedding_dim=30)


window_size = 4
initial_lr = 0.001
epochs = 10
total_tokens = len(data)


def get_similarity(word, tokenizer, model, top_n=5):
    """Finds the most similar words based on Cosine Similarity."""
    if word not in tokenizer.word2id:
        return

    target_idx = tokenizer.word2id[word]
    v_target = model.W_in[target_idx]

    scores = {}
    for w, idx in tokenizer.word2id.items():
        if idx == target_idx: continue
        v_w = model.W_in[idx]
        # Cosine Similarity Formula
        sim = np.dot(v_target, v_w) / (np.linalg.norm(v_target) * np.linalg.norm(v_w) + 1e-9)
        scores[w] = sim

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# Training Loop
print(f"--- Starting Training ({t.vocab_size} unique words) ---")
for epoch in range(epochs):
    epoch_loss = 0
    pbar = tqdm(range(total_tokens), desc=f"Epoch {epoch + 1}/{epochs}")

    for i in pbar:
        center_id = data[i]

        # Learning Rate Decay (Linear)
        progress = (epoch * total_tokens + i) / (epochs * total_tokens)
        learning_rate = max(initial_lr * (1.0 - progress), initial_lr * 0.0001)

        # Sliding Window
        start = max(0, i - window_size)
        end = min(total_tokens, i + window_size + 1)

        for j in range(start, end):
            if i == j: continue

            context_id = data[j]

            neg_indices = np.random.randint(0, len(t.unigram_table), 20)
            negative_ids = t.unigram_table[neg_indices]
            v_c, pos_prob, neg_probs = model.forward(center_id, context_id, negative_ids)

            # Backward Pass & Loss Calculation
            # Loss = -log(pos) - sum(log(1-negs))
            epoch_loss += model.backward(center_id, context_id, negative_ids, v_c, pos_prob, neg_probs, learning_rate)
        if i % 100 == 0:
            pbar.set_postfix({"loss": f"{epoch_loss / ((i + 1) * window_size * 2):.4f}"})



# Final Results
print("\n--- Training Complete! ---")
# Examples
test_words = ["man", "sick", "gentleman", "underground", "zossima", "alyosha", "faith", "ivan", "god"]
for w in test_words:
    print(f"Closest to '{w}': {get_similarity(w, t, model, 10)}")


import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        self.v_size = vocab_size
        self.dim = embedding_dim

        limit = np.sqrt(6 / (self.v_size + self.dim))

        self.W_in = np.random.uniform(-limit, limit, (self.v_size, self.dim))
        self.W_out = np.random.uniform(-limit, limit, (self.v_size, self.dim))

    def sigmoid(self, x):
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))

    def forward(self, center_id, context_id, negative_ids):
        v_c = self.W_in[center_id]  # Center word (1, dim)
        v_pos = self.W_out[context_id]  # Positive neighbor (1, dim)
        v_negs = self.W_out[negative_ids]  # Negative samples (k, dim)

        # Positive Score: v_pos . v_c
        pos_score = np.dot(v_c, v_pos)
        pos_prob = self.sigmoid(pos_score)

        # Negative Scores: v_negs . v_c
        neg_scores = np.dot(v_negs, v_c)
        neg_probs = self.sigmoid(neg_scores)

        return v_c, pos_prob, neg_probs

    def backward(self, center_id, context_id, negative_ids, v_c, pos_prob, neg_probs, learning_rate):
        # ERRORS: (Prediction - Label)
        # Label for positive is 1; Label for negative is 0
        err_pos = pos_prob - 1
        err_negs = neg_probs

        v_pos_old = self.W_out[context_id].copy()
        self.W_out[context_id] -= learning_rate * err_pos * v_c

        v_negs_old = self.W_out[negative_ids].copy()
        updt_matrix = -learning_rate * np.outer(err_negs, v_c)
        np.add.at(self.W_out, negative_ids, updt_matrix)

        grad_pos = err_pos * v_pos_old
        grad_negs_sum = np.dot(err_negs, v_negs_old)

        self.W_in[center_id] -= learning_rate * (grad_pos + grad_negs_sum)
        loss = -np.log(pos_prob + 1e-9) - np.sum(np.log(1 - neg_probs + 1e-9))
        return loss
import re
import numpy as np
import collections


class Tokenizer:
    def __init__(self, min_count=2):
        self.min_count = min_count
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.unigram_table = []

    def clean_text(self, text):
        text = text.lower()

        text = re.sub(r'(?i)(part|chapter|section)\s+([ivx0-9]+)', ' ', text)

        words = re.findall(r"\b\w+(?:'\w+)?\b", text)

        return words

    def vocabulary(self, words):

        counts = collections.Counter(words)
        self.id2word = [word for word, count in counts.items() if count >= self.min_count]
        self.word2id = {i: word for word, i in enumerate(self.id2word)}
        self.vocab_size = len(self.id2word)

        threshold = 1e-3  
        total_count = len(words)

        #the numerical version of the text
        data = []

        def get_discard_prob(word):
            freq = counts[word] / total_count
            return max(0, 1 - np.sqrt(threshold / freq))

        for word in words:
            if word in self.word2id:
                # if random > discard_prob, we keep it
                if np.random.random() > get_discard_prob(word):
                    data.append(self.word2id[word])

        word_counts = np.array([counts[word] for word in self.id2word])
        weights = word_counts ** 0.75
        probabilities = weights / weights.sum()


        table_size = 10 ** 6
        self.unigram_table = np.random.choice(range(self.vocab_size), size=table_size, p=probabilities)

        return data

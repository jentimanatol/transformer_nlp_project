
import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, max_vocab_size=25000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>"}
    
    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        
        most_common = counter.most_common(self.max_vocab_size - len(self.word2idx))
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text, max_len):
        tokens = ["<CLS>"] + self.tokenize(text)
        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        ids = ids[:max_len]
        padding_length = max_len - len(ids)
        ids += [self.word2idx["<PAD>"]] * padding_length
        return ids

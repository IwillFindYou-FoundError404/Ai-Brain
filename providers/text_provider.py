import numpy as np
import json
import os
from providers.base_provider import BaseProvider

class TextProvider(BaseProvider):
    def __init__(self, vocab_size=10, hidden_dim=8, output_dim=2):
        self.memory_file = "brains.json"
        self.vocab = {"pad": 0, "chao": 1, "hello": 2, "hi": 3, "ban": 4, "ghet": 5, "xau": 6, "buc": 7, "toi": 8, "vui": 9}
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.t = 0
        
        # Khởi tạo ma trận Adam
        self.m_w1, self.v_w1 = np.zeros((vocab_size, hidden_dim)), np.zeros((vocab_size, hidden_dim))
        self.m_b1, self.v_b1 = np.zeros((1, hidden_dim)), np.zeros((1, hidden_dim))
        self.m_w2, self.v_w2 = np.zeros((hidden_dim, output_dim)), np.zeros((hidden_dim, output_dim))
        self.m_b2, self.v_b2 = np.zeros((1, output_dim)), np.zeros((1, output_dim))
        
        if os.path.exists(self.memory_file) and os.path.getsize(self.memory_file) > 2:
            self.load_memory()
        else:
            self.w1 = np.random.randn(vocab_size, hidden_dim) * np.sqrt(2.0 / vocab_size)
            self.b1 = np.zeros((1, hidden_dim))
            self.w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
            self.b2 = np.zeros((1, output_dim))
            self.save_memory()

    def text_to_vector(self, text):
        vector = np.zeros(self.vocab_size)
        for word in text.lower().split():
            if word in self.vocab: vector[self.vocab[word]] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def forward(self, X):
        self.X = np.atleast_2d(X)
        self.z1 = np.dot(self.X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=-1, keepdims=True))
        self.a2 = exp_z2 / np.sum(exp_z2, axis=-1, keepdims=True) # Softmax
        return self.a2

    def execute(self, data):
        text = data.get('text', '')
        vector = self.text_to_vector(text)
        prediction = self.forward(vector)[0]
        categories = ["Thân thiện / Vui vẻ", "Tiêu cực / Bực bội"]
        chosen_idx = int(np.argmax(prediction))
        return {"decision": categories[chosen_idx], "vector": vector.tolist(), "probabilities": [float(p) for p in prediction]}

    def train_step(self, text, label, lr=0.05):
        X = np.atleast_2d(self.text_to_vector(text))
        y_true = np.zeros((1, self.output_dim))
        y_true[0, label] = 1.0
        
        y_pred = self.forward(X)
        error_out = y_pred - y_true
        grad_w2 = np.dot(self.a1.T, error_out)
        grad_b2 = np.sum(error_out, axis=0, keepdims=True)
        
        error_hid = np.dot(error_out, self.w2.T) * (self.z1 > 0).astype(float)
        grad_w1 = np.dot(X.T, error_hid)
        grad_b1 = np.sum(error_hid, axis=0, keepdims=True)
        
        self.t += 1
        # Cập nhật hệ số Adam nâng cao (Tránh tư duy lối mòn)
        for g, m, v, w in [(grad_w1, self.m_w1, self.v_w1, self.w1), (grad_b1, self.m_b1, self.v_b1, self.b1),
                           (grad_w2, self.m_w2, self.v_w2, self.w2), (grad_b2, self.m_b2, self.v_b2, self.b2)]:
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * (g ** 2)
            m_hat = m / (1 - 0.9 ** self.t)
            v_hat = v / (1 - 0.999 ** self.t)
            w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.save_memory()

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump({"w1": self.w1.tolist(), "b1": self.b1.tolist(), "w2": self.w2.tolist(), "b2": self.b2.tolist(), "t": self.t}, f)

    def load_memory(self):
        with open(self.memory_file, "r") as f: data = json.load(f)
        self.w1, self.b1 = np.array(data["w1"]), np.array(data["b1"])
        self.w2, self.b2 = np.array(data["w2"]), np.array(data["b2"])
        self.t = data.get("t", 0)

text_provider = TextProvider()

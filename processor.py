import numpy as np

class TextProcessor:
    def __init__(self):
        # Bảng từ vựng giới hạn tự định nghĩa (Vocabulary)
        self.vocab = {
            "pad": 0, "chao": 1, "hello": 2, "hi": 3, "ban": 4, 
            "ghet": 5, "xau": 6, "bực": 7, "tồi": 8, "vui": 9
        }
        self.vocab_size = len(self.vocab)

    def text_to_vector(self, text):
        """Biến câu chữ thành một Vector tần suất từ (Bag of Words)"""
        vector = np.zeros(self.vocab_size)
        words = text.lower().split()
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1
        # Chuẩn hóa Vector để tránh nổ số (Normalization)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

processor = TextProcessor()

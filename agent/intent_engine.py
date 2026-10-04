class IntentEngine:
    """Phân rã ngữ cảnh câu nói để định hướng luồng xử lý"""
    def detect(self, text):
        raw = text.lower()
        if "video" in raw or "luồng" in raw: return "spatial_video"
        if "ảnh" in raw or "hình" in raw: return "spatial_image"
        return "text_classification"

intent_engine = IntentEngine()

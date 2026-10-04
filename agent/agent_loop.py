from agent.intent_engine import intent_engine
from agent.world_model import world_model
from providers.text_provider import text_provider
from providers.spatial_provider import spatial_provider

class AgentLoop:
    def __init__(self):
        self.text_engine = text_provider
        self.spatial_engine = spatial_provider

    def run_cycle(self, input_text):
        # 1. Cập nhật trạng thái tự thân (Self State)
        world_model.update("status", "planning")
        
        # 2. Nhận diện mục đích (Intent Detection)
        intent = intent_engine.detect(input_text)
        world_model.update("current_task", f"executing_{intent}")
        
        # 3. Định tuyến và xử lý qua lớp Provider thích hợp
        if intent == "text_classification":
            res = self.text_engine.execute({"text": input_text})
            world_model.update("status", "idle")
            return {"type": "text", "data": res}
        elif intent == "spatial_image":
            res = self.spatial_engine.execute({"frame_idx": 12}) # Lấy frame cố định đại diện cho ảnh
            world_model.update("status", "idle")
            return {"type": "image", "data": res}
        elif intent == "spatial_video":
            world_model.update("status", "observing")
            return {"type": "video_stream", "data": None}

agent_loop = AgentLoop()
